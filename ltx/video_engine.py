"""
Video Generation Engine for LTX-2.3 22B Distilled.
Manages GPU acceleration (Dual T4 x2 / Single GPU), model loading, memory offloading (mmgp),
runtime patches (Gemma 3 SDPA/RoPE, FP32 VAE), audio processing, and video generation.
"""

import os
import sys
import gc
import json
import time
import glob
import random
import logging
import tempfile
import traceback
import subprocess
from typing import Optional, Tuple, Callable, Any

logger = logging.getLogger("video_engine")

# 1. Environment & Performance Settings
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.6"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
os.environ["MALLOC_TRIM_THRESHOLD_"] = "0"

# Preserve original directory in sys.path before chdir
ORIG_DIR = os.path.dirname(os.path.abspath(__file__))
if ORIG_DIR not in sys.path:
    sys.path.insert(0, ORIG_DIR)

# Bootstrap Wan2GP
WAN2GP_DIR = os.path.abspath("Wan2GP")
if not os.path.exists(WAN2GP_DIR) and os.path.exists(os.path.join(ORIG_DIR, "Wan2GP")):
    WAN2GP_DIR = os.path.join(ORIG_DIR, "Wan2GP")

from queue_service import OUTPUT_DIR

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.environ.setdefault("GRADIO_TEMP_DIR", os.path.join(OUTPUT_DIR, ".gradio_cache"))

if WAN2GP_DIR not in sys.path:
    sys.path.insert(0, WAN2GP_DIR)
if os.path.exists(WAN2GP_DIR):
    os.chdir(WAN2GP_DIR)

# Safe imports for dependencies
try:
    import numpy as np
except ImportError:
    np = None

try:
    import psutil
except ImportError:
    psutil = None

try:
    import soundfile as sf
except ImportError:
    sf = None

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False


def _inference_mode_decorator(fn):
    if HAS_TORCH and hasattr(torch, "inference_mode"):
        return torch.inference_mode()(fn)
    return fn


# Detect Hardware
NUM_GPUS = torch.cuda.device_count() if (HAS_TORCH and torch.cuda.is_available()) else 0
_GPU_SM = torch.cuda.get_device_capability(0) if (HAS_TORCH and NUM_GPUS > 0) else (0, 0)
_IS_SM60 = (_GPU_SM[0] == 6)

print("=" * 65)
print("🎤 LTX-2.3 22B Distilled - Audio-to-Video Engine")
print("📺 Created by: AIQUEST Academy")
print(f"🎮 Accelerator: {'Dual GPU (T4 x2)' if NUM_GPUS > 1 else 'Single GPU'} | Compute: sm_{_GPU_SM[0]}{_GPU_SM[1]}")
print("=" * 65)

# CUDA Attention & Convolution Optimizations for Tesla T4 (Turing sm_75)
if HAS_TORCH and torch.cuda.is_available():
    torch.backends.cuda.enable_flash_sdp(False)          # sm_75 does not support FlashAttention v2
    torch.backends.cuda.enable_mem_efficient_sdp(True)   # Fast memory-efficient attention (Cutlass kernel)
    torch.backends.cuda.enable_math_sdp(True)            # Safe fallback
    torch.backends.cudnn.benchmark = False               # Autotuning FP32 conv3d on a T4 stalls the VAE


# ==== GEMMA 3 SDPA & ROPE DTYPE PATCH ====
def _patch_gemma3_attention():
    """Fixes query/key/value dtype mismatch in Gemma 3."""
    try:
        import transformers.models.gemma3.modeling_gemma3 as g3_mod
        _orig_rope = g3_mod.apply_rotary_pos_emb

        def _safe_apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
            if cos.dtype != q.dtype:
                cos = cos.to(dtype=q.dtype)
            if sin.dtype != q.dtype:
                sin = sin.to(dtype=q.dtype)
            return _orig_rope(q, k, cos, sin, unsqueeze_dim=unsqueeze_dim)

        g3_mod.apply_rotary_pos_emb = _safe_apply_rotary_pos_emb
        print("  [Gemma3 Patch] ✅ apply_rotary_pos_emb patched (RoPE dtype aligned with query)")
    except Exception as e:
        print(f"  [Gemma3 Patch] ⚠️ Could not patch apply_rotary_pos_emb: {e}")

    try:
        import transformers.integrations.sdpa_attention as sdpa_mod
        _orig_sdpa = sdpa_mod.sdpa_attention_forward

        def _safe_sdpa_attention_forward(module, query, key, value, *args, **kwargs):
            if key.dtype != query.dtype:
                key = key.to(dtype=query.dtype)
            if value.dtype != query.dtype:
                value = value.to(dtype=query.dtype)
            return _orig_sdpa(module, query, key, value, *args, **kwargs)

        sdpa_mod.sdpa_attention_forward = _safe_sdpa_attention_forward
        print("  [SDPA Patch] ✅ sdpa_attention_forward patched (query/key/value dtype auto-aligned)")
    except Exception as e:
        print(f"  [SDPA Patch] ⚠️ Could not patch sdpa_attention_forward: {e}")


# ==== GGUF EXTENSION HANDLER & CONFIG PATCH ====
def _register_gguf_handler():
    """Register the GGUF handler with mmgp's quant_router."""
    try:
        import shared.qtypes.gguf  # noqa: F401
        print("  [GGUF] ✅ Extension handler registered with mmgp (Wan2GP native)")
    except Exception as e:
        print(f"  [GGUF] ⚠️ Could not register gguf handler: {e}")


def _patch_ltx2_config_loading():
    """Patch _load_config_from_checkpoint to handle GGUF metadata gracefully."""
    try:
        import models.ltx2.ltx2 as ltx2_mod
        _original = ltx2_mod._load_config_from_checkpoint

        def _patched(path, fallback_config_path=None):
            from mmgp import quant_router
            if isinstance(path, (list, tuple)):
                path = path[0] if path else ""
            if not path:
                return {}
            try:
                _, metadata = quant_router.load_metadata_state_dict(path)
                if metadata:
                    config_raw = metadata.get("config")
                    if config_raw:
                        config = ltx2_mod._normalize_config(config_raw)
                        if config:
                            return config
            except Exception:
                pass
            if fallback_config_path and os.path.isfile(fallback_config_path):
                try:
                    with open(fallback_config_path, "r", encoding="utf-8") as f:
                        config = ltx2_mod._normalize_config(json.load(f))
                        if config:
                            print(f"  [GGUF Patch] ✅ Config loaded from {os.path.basename(fallback_config_path)}")
                            return config
                except Exception:
                    pass
            return {}

        ltx2_mod._load_config_from_checkpoint = _patched
        print("  [GGUF Patch] ✅ Config loading patched for GGUF")
    except Exception as e:
        print(f"  [GGUF Patch] ⚠️ Could not patch config loading: {e}")


_patch_gemma3_attention()
_register_gguf_handler()
_patch_ltx2_config_loading()


# ==== RESOLUTION & FRAME HELPERS ====
AUTO_ASPECT = "Auto (match reference image)"


def get_resolution(base_res_str: str, aspect_ratio_str: str, ref_image: Optional[Any] = None) -> Tuple[int, int]:
    """Calculate width and height snapped to 32px increments."""
    base_resolutions = {"1080p": 1088, "720p": 704, "540p": 544, "480p": 480, "240p": 240}
    ratios = {
        "16:9 Landscape": 16 / 9,
        "4:3 Standard": 4 / 3,
        "1:1 Square": 1.0,
        "3:4 Portrait": 3 / 4,
        "9:16 Portrait": 9 / 16,
    }
    base = base_resolutions.get(base_res_str, 704)
    if aspect_ratio_str == AUTO_ASPECT and ref_image is not None and hasattr(ref_image, "width"):
        ratio = ref_image.width / ref_image.height
    else:
        ratio = ratios.get(aspect_ratio_str, 16 / 9)
    if ratio >= 1.0:
        height = base
        width = int(base * ratio)
    else:
        width = base
        height = int(base / ratio)
    return (width // 32) * 32, (height // 32) * 32


def snap_to_ltx_frames(duration_sec: float, fps: float = 24.0, max_frames: int = 481) -> int:
    """Convert audio duration to valid LTX frame count (8k + 1)."""
    raw = duration_sec * fps
    k = max(0, round((raw - 1) / 8))
    frames = 8 * k + 1
    return int(max(49, min(frames, max_frames)))


# Latent & VAE Helpers
def _latent_stats(tag, t):
    try:
        t = t[0] if isinstance(t, (list, tuple)) else t
        f = t.detach().float()
        nonfinite = int((~torch.isfinite(f)).sum().item())
        f = torch.nan_to_num(f)
        print(f"  🔬 {tag}: shape={tuple(t.shape)} mean={f.mean().item():+.3f} std={f.std().item():.3f} nonfinite={nonfinite}")
        sys.stdout.flush()
        return nonfinite
    except Exception as _e:
        print(f"  🔬 {tag}: stats unavailable ({_e})")
        return 0


def _fast_vae_tiling(height, width):
    from models.ltx2.ltx_core.model.video_vae.tiling import TilingConfig, SpatialTilingConfig, TemporalTilingConfig
    budget = (torch.cuda.get_device_properties(0).total_memory - 1.6e9 - 3.0e9) if (HAS_TORCH and torch.cuda.is_available()) else 10e9
    bytes_per_pixel_frame = 230
    spatial = None
    tile_area = height * width
    if budget / (bytes_per_pixel_frame * tile_area) < 96:
        spatial = SpatialTilingConfig(tile_size_in_pixels=512, tile_overlap_in_pixels=64)
        tile_area = min(height, 512) * min(width, 512)
    frames = max(32, min(160, int(budget / (bytes_per_pixel_frame * tile_area)) // 8 * 8))
    return TilingConfig(spatial_config=spatial,
                        temporal_config=TemporalTilingConfig(tile_size_in_frames=frames, tile_overlap_in_frames=24))


class LTXVideoEngine:
    """
    LTX-2.3 22B Distilled Video Generation Engine.
    Manages model initialization, GPU placement, and execution.
    """

    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = output_dir or OUTPUT_DIR
        os.makedirs(self.output_dir, exist_ok=True)

        self.model = None
        self.pipe = None
        self.is_loaded = False
        self.aux_on_gpu1 = False
        self.ups_on_gpu1 = False

    def load_model(self):
        """Load and configure the LTX-2.3 22B Distilled model components."""
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is not available. Please ensure torch is installed.")

        if self.is_loaded:
            return

        print("\nLoading LTX-2.3 22B Distilled (GGUF Q4_K_M)...")
        sys.stdout.flush()

        from mmgp import offload
        from models.ltx2.ltx2_handler import family_handler
        from shared.utils import files_locator as fl

        search_paths = [
            "models",
            "ckpts",
            "/kaggle/tmp/models",
            os.path.join(ORIG_DIR, "tmp_models"),
            os.path.join(WAN2GP_DIR, "models"),
            ".",
        ]
        fl.set_checkpoints_paths(search_paths)

        base_model_type = "ltx2_22B"
        model_def = {"ltx2_pipeline": "distilled"}
        extra = family_handler.query_model_def(base_model_type, model_def)
        model_def.update(extra)

        # Locate Gemma text encoder
        gemma_candidates = [
            "/kaggle/tmp/models/gemma-3-12b-it-qat-q4_0-unquantized",
            os.path.join(WAN2GP_DIR, "models/gemma-3-12b-it-qat-q4_0-unquantized"),
            os.path.join(ORIG_DIR, "tmp_models/gemma-3-12b-it-qat-q4_0-unquantized"),
        ]
        gemma_folder = None
        for cand in gemma_candidates:
            if os.path.exists(cand):
                gemma_folder = cand
                break
        if not gemma_folder:
            gemma_folder = "/kaggle/tmp/models/gemma-3-12b-it-qat-q4_0-unquantized"

        gemma_files = sorted(glob.glob(os.path.join(gemma_folder, "*.safetensors"))) if os.path.exists(gemma_folder) else []
        quanto_files = [f for f in gemma_files if "quanto" in f]
        text_encoder_file = quanto_files[0] if quanto_files else (gemma_files[0] if gemma_files else None)
        if not text_encoder_file:
            print(f"  ⚠️ Warning: No .safetensors in {gemma_folder}. Searching search_paths...")
            for sp in search_paths:
                found = glob.glob(os.path.join(sp, "**/*gemma*/*.safetensors"), recursive=True)
                if found:
                    text_encoder_file = found[0]
                    break
            if not text_encoder_file:
                raise FileNotFoundError(f"Gemma safetensors missing. Check model download paths: {gemma_candidates}")

        print(f"  Text encoder: {os.path.basename(text_encoder_file)}")

        # Locate Transformer GGUF
        transformer_candidates = [
            "/kaggle/tmp/models/ltx-2.3-22b-distilled-Q4_K_M_light.gguf",
            os.path.join(WAN2GP_DIR, "models/ltx-2.3-22b-distilled-Q4_K_M_light.gguf"),
            os.path.join(ORIG_DIR, "tmp_models/ltx-2.3-22b-distilled-Q4_K_M_light.gguf"),
        ]
        transformer_path = None
        for cand in transformer_candidates:
            if os.path.isfile(cand):
                transformer_path = cand
                break
        if not transformer_path:
            transformer_path = "/kaggle/tmp/models/ltx-2.3-22b-distilled-Q4_K_M_light.gguf"
            if not os.path.isfile(transformer_path):
                raise FileNotFoundError(f"{transformer_path} missing. Check model download paths.")

        print(f"  Transformer : {os.path.basename(transformer_path)}")
        sys.stdout.flush()

        MODEL_DTYPE = torch.float16
        VAE_DTYPE = torch.float32

        ltx2_model, pipe = family_handler.load_model(
            model_filename=transformer_path,
            model_type="ltx2_22B_distilled",
            base_model_type=base_model_type,
            model_def=model_def,
            dtype=MODEL_DTYPE,
            VAE_dtype=VAE_DTYPE,
            text_encoder_filename=text_encoder_file,
        )

        # Align Gemma embed_tokens to bfloat16
        try:
            _g_embed = ltx2_model.text_encoder.model.model.embed_tokens
            _g_embed.to(torch.bfloat16)
        except Exception:
            try:
                ltx2_model.text_encoder.model.embed_tokens.to(torch.bfloat16)
            except Exception:
                pass

        # FP32 VIDEO VAE & DTYPE-SAFE FORWARDS
        for _key in ("video_decoder", "video_encoder", "spatial_upsampler"):
            _mod = getattr(ltx2_model, _key, None)
            if _mod is None:
                continue
            _mod.to(torch.float32)
            _mod._model_dtype = torch.float32
            for _m in _mod.modules():
                _m._lock_dtype = torch.float32
        print("  [VAE] ✅ Video VAE + spatial upsampler converted to FP32")

        import models.ltx2.ltx_core.model.video_vae.video_vae as _video_vae_mod
        _orig_video_decoder_forward = _video_vae_mod.VideoDecoder.forward
        _orig_video_encoder_forward = _video_vae_mod.VideoEncoder.forward

        def _fp32_video_decoder_forward(self, sample, *args, **kwargs):
            param = next(self.parameters(), None)
            if param is not None and sample.dtype != param.dtype:
                sample = sample.to(param.dtype)
            return _orig_video_decoder_forward(self, sample, *args, **kwargs)

        def _fp32_video_encoder_forward(self, sample, *args, **kwargs):
            in_dtype, in_device = sample.dtype, sample.device
            param = next(self.parameters(), None)
            if param is not None and param.device.type == "cuda" and sample.device != param.device:
                sample = sample.to(param.device)
            if param is not None and sample.dtype != param.dtype:
                sample = sample.to(param.dtype)
            with torch.cuda.device(sample.device if sample.is_cuda else torch.cuda.current_device()):
                out = _orig_video_encoder_forward(self, sample, *args, **kwargs)
            if torch.is_tensor(out) and out.is_floating_point():
                out = out.to(device=in_device, dtype=in_dtype if in_dtype.is_floating_point else out.dtype)
            return out

        _video_vae_mod.VideoDecoder.forward = _fp32_video_decoder_forward
        _video_vae_mod.VideoEncoder.forward = _fp32_video_encoder_forward

        # Distilled pipeline probed upsampler & decoder
        import models.ltx2.ltx_pipelines.distilled as _distilled_mod
        _orig_pipeline_upsample = _distilled_mod.upsample_video
        _orig_pipeline_decode = _distilled_mod.vae_decode_video_to_tensor

        def _probed_upsample_video(latent, video_encoder, upsampler):
            if _latent_stats("Stage 1 latent", latent):
                raise RuntimeError("Stage 1 produced NaN latents (FP16 overflow).")
            in_dtype, in_device = latent.dtype, latent.device
            if self.ups_on_gpu1:
                with torch.cuda.device(1):
                    out = _orig_pipeline_upsample(latent=latent.to("cuda:1"), video_encoder=video_encoder, upsampler=upsampler)
            else:
                out = _orig_pipeline_upsample(latent=latent, video_encoder=video_encoder, upsampler=upsampler)
            out = out.to(device=in_device, dtype=in_dtype)
            _latent_stats("Upscaled latent", out)
            return out

        def _probed_decode(latent, video_decoder, tiling_config=None, *args, **kwargs):
            _latent_stats("Final latent -> VAE", latent)
            latent_tensor = latent[0] if isinstance(latent, list) else latent
            fast_config = _fast_vae_tiling(kwargs.get("expected_height") or 480, kwargs.get("expected_width") or 832)
            print(f"  🎞️ VAE tiling: {'whole frame' if fast_config.spatial_config is None else '512 px tiles'}, "
                  f"{fast_config.temporal_config.tile_size_in_frames}-frame chunks (24 overlap)")
            t_dec = time.time()
            if HAS_TORCH and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(0)
            try:
                out = _orig_pipeline_decode([latent_tensor] if isinstance(latent, list) else latent_tensor,
                                            video_decoder, fast_config, *args, **kwargs)
            except Exception as oom:
                if HAS_TORCH and isinstance(oom, getattr(torch, "OutOfMemoryError", Exception)):
                    print("  ⚠️ VAE out of memory with large chunks, retrying with default tiling...")
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    out = _orig_pipeline_decode([latent_tensor] if isinstance(latent, list) else latent_tensor,
                                                video_decoder, tiling_config, *args, **kwargs)
                else:
                    raise oom
            peak_vram = (torch.cuda.max_memory_allocated() / 1024**3) if (HAS_TORCH and torch.cuda.is_available()) else 0
            print(f"  🎞️ VAE decode: {time.time() - t_dec:.1f}s (peak VRAM {peak_vram:.1f} GB)")
            return out

        _distilled_mod.upsample_video = _probed_upsample_video
        _distilled_mod.vae_decode_video_to_tensor = _probed_decode

        # DUAL T4 ACCELERATION
        self.aux_on_gpu1 = False
        self.ups_on_gpu1 = False

        if NUM_GPUS > 1:
            print("\n🚀 Dual GPU Active (GPU T4 x2) - Offloading Gemma 3 Text Encoder to GPU 1...")
            try:
                ltx2_model.text_encoder.to("cuda:1")
                _gemma = ltx2_model.text_encoder.model
                _gemma_lm = _gemma.model if hasattr(_gemma, "model") else _gemma
                if getattr(_gemma, "lm_head", None) is not None:
                    _gemma.lm_head = torch.nn.Identity()

                _embed = _gemma_lm.embed_tokens
                _embed.to("cpu")
                _orig_embed_fwd = _embed.forward

                def _cpu_embed_fwd(input_ids, *args, **kwargs):
                    return _orig_embed_fwd(input_ids.to("cpu"), *args, **kwargs).to("cuda:1")

                _embed.forward = _cpu_embed_fwd
                _gemma.__class__ = type(_gemma.__class__.__name__, (_gemma.__class__,),
                                        {"device": property(lambda self: torch.device("cuda", 1))})
                pipe.pop("text_encoder", None)

                _orig_encode_raw = ltx2_model.text_encoder.encode_raw

                def _dual_gpu_encode_raw(text, padding_side="left"):
                    raw_emb = _orig_encode_raw(text, padding_side=padding_side)
                    hs_cuda0 = tuple(h.to(device="cuda:0") for h in raw_emb.hidden_states)
                    _absmax = max(float(h.abs().max()) for h in hs_cuda0)
                    print(f"  🔬 Gemma hidden states: {len(hs_cuda0)} layers, dtype={hs_cuda0[0].dtype}, absmax={_absmax:.0f}")
                    mask_cuda0 = raw_emb.attention_mask.to("cuda:0")
                    return raw_emb.__class__(hs_cuda0, mask_cuda0, raw_emb.padding_side)

                ltx2_model.text_encoder.encode_raw = _dual_gpu_encode_raw

                try:
                    import models.ltx2.ltx_core.text_encoders.gemma.encoders.base_encoder as _base_enc_mod
                    _orig_postprocess = _base_enc_mod.postprocess_text_embeddings

                    def _safe_postprocess(embeddings, feat_ext, emb_conn, aud_conn, return_attention_masks=False):
                        target_dev = next(feat_ext.parameters()).device if any(True for _ in feat_ext.parameters()) else torch.device("cuda:0")
                        safe_embs = []
                        for item in embeddings:
                            hs = tuple(h.to(device=target_dev) for h in item.hidden_states)
                            mask = item.attention_mask.to(target_dev)
                            safe_embs.append(item.__class__(hs, mask, item.padding_side))
                        return _orig_postprocess(safe_embs, feat_ext, emb_conn, aud_conn, return_attention_masks=return_attention_masks)

                    _base_enc_mod.postprocess_text_embeddings = _safe_postprocess
                    print("  [GPU T4 x2] ✅ postprocess_text_embeddings device & dtype alignment active.")
                except Exception as _pe:
                    print(f"  [GPU T4 x2] ⚠️ Postprocess patch note: {_pe}")

                self.aux_on_gpu1 = True
                _free1 = torch.cuda.mem_get_info(1)[0] / 1024**3
                print(f"  [GPU T4 x2] ✅ Gemma 3 text encoder resident on GPU 1 ({_free1:.1f} GB free on GPU 1).")
            except Exception as e:
                print(f"  [GPU T4 x2] ⚠️ Dual GPU setup notice: {e}, falling back to single GPU mode")
                self.aux_on_gpu1 = False

        if self.aux_on_gpu1 and pipe.get("spatial_upsampler") is not None:
            with torch.cuda.device(1):
                torch.cuda.empty_cache()
            try:
                for _key in ("spatial_upsampler", "video_encoder"):
                    pipe[_key].to("cuda:1")
                for _key in ("spatial_upsampler", "video_encoder"):
                    pipe.pop(_key)
                self.ups_on_gpu1 = True
                print(f"  [GPU T4 x2] ✅ Spatial upsampler + video encoder resident on GPU 1 ({torch.cuda.mem_get_info(1)[0] / 1024**3:.1f} GB free).")
            except Exception as e:
                print(f"  [GPU T4 x2] ⚠️ Upsampler / encoder stay on GPU 0 ({e})")
                for _key in ("spatial_upsampler", "video_encoder"):
                    getattr(ltx2_model, _key).to("cpu")
                with torch.cuda.device(1):
                    torch.cuda.empty_cache()

        # APPLY MMGP OFFLOADING ON GPU 0
        transformer_budget = 11000 if self.aux_on_gpu1 else 6000
        print(f"\nApplying mmgp Profile 4 (Transformer VRAM budget: {transformer_budget} MB)...")
        sys.stdout.flush()

        budgets = {
            "transformer":       transformer_budget,
            "vae":               3500,
            "spatial_upsampler": 2000,
            "video_encoder":     2000,
            "audio_encoder":     1000,
            "audio_decoder":     1000,
            "vocoder":           500,
            "*":                 1000,
        }
        if not self.aux_on_gpu1:
            budgets["text_encoder"] = 1500
        if self.ups_on_gpu1:
            budgets.pop("spatial_upsampler", None)
            budgets.pop("video_encoder", None)

        offload.profile(
            pipe,
            profile_no=4,
            quantizeTransformer=False,
            convertWeightsFloatTo=torch.float16,
            budgets=budgets,
        )
        offload.shared_state["_attention"] = "sdpa"

        self.model = ltx2_model
        self.pipe = pipe
        self.is_loaded = True
        print("\n✅ Setup complete! Distilled Audio-to-Video pipeline active.")
        sys.stdout.flush()

    @_inference_mode_decorator
    def generate_video(
        self,
        prompt: str,
        input_image_start: Optional[str] = None,
        input_audio: Optional[str] = None,
        seed: int = -1,
        duration_dropdown: str = "5 Seconds (121 frames)",
        resolution_dropdown: str = "720p",
        aspect_ratio_dropdown: str = AUTO_ASPECT,
        match_audio_dur: bool = False,
        job_id: Optional[str] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        check_cancelled: Optional[Callable[[], bool]] = None,
    ) -> Tuple[Optional[str], str]:
        """
        Execute full video generation pipeline.
        Returns: (output_video_path, status_summary_string)
        """
        try:
            self.load_model()

            gc.collect()
            if HAS_TORCH and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            if progress_callback:
                progress_callback(0.01, "Starting generation...")

            duration_map = {
                "2 Seconds (49 frames)":  49,
                "3 Seconds (73 frames)":  73,
                "5 Seconds (121 frames)": 121,
                "8 Seconds (193 frames)": 193,
                "10 Seconds (241 frames)": 241,
                "15 Seconds (361 frames)": 361,
                "20 Seconds (481 frames)": 481,
            }
            frame_rate = 24.0

            # Load & prepare audio
            input_waveform, input_waveform_sample_rate = None, None
            audio_duration_sec = None
            if input_audio and os.path.exists(input_audio) and sf is not None:
                wav, sr = sf.read(input_audio)
                if wav.ndim > 1:
                    wav = wav.mean(axis=1)
                input_waveform = wav.astype(np.float32) if np is not None else wav
                input_waveform_sample_rate = int(sr)
                audio_duration_sec = len(wav) / sr
                print(f"  [Audio] Loaded: sr={sr}, duration={audio_duration_sec:.1f}s, shape={wav.shape}")

            if match_audio_dur and audio_duration_sec is not None:
                num_frames = snap_to_ltx_frames(audio_duration_sec, frame_rate)
                print(f"  [Duration] Match-Audio mode: {audio_duration_sec:.1f}s → {num_frames} frames ({num_frames/frame_rate:.1f}s @ {frame_rate}fps)")
            else:
                num_frames = duration_map.get(duration_dropdown, 121)

            if seed is None or seed < 0:
                seed = random.randint(0, 2**32 - 1)
            seed = int(seed)

            # Load reference image
            image_start = None
            if input_image_start and os.path.exists(input_image_start) and Image is not None:
                image_start = Image.open(input_image_start).convert("RGB")

            width, height = get_resolution(resolution_dropdown, aspect_ratio_dropdown, ref_image=image_start)
            guide_phases = 2

            free_vram0 = (torch.cuda.mem_get_info(0)[0] / 1024**3) if (HAS_TORCH and torch.cuda.is_available()) else 0
            free_vram1 = (torch.cuda.mem_get_info(1)[0] / 1024**3) if (HAS_TORCH and NUM_GPUS > 1) else 0
            ram_avail = (psutil.virtual_memory().available / 1024**3) if psutil else 0

            print(f"\n{'='*65}")
            print(f"🎬 Generating Video: {width}x{height}, {num_frames} frames, Seed={seed}")
            print(f"  VRAM Free: GPU0={free_vram0:.2f} GB" + (f", GPU1={free_vram1:.2f} GB" if NUM_GPUS > 1 else ""))
            print(f"  RAM Free: {ram_avail:.1f} GB | Audio: {'YES' if input_waveform is not None else 'NO'}")
            print(f"  Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
            print(f"{'='*65}")
            sys.stdout.flush()

            total_steps = [8]
            current_step = [0]
            current_pass = [1]

            def cb(step, latent, is_start, override_num_inference_steps=None, pass_no=None, **kwargs):
                if check_cancelled and check_cancelled():
                    raise InterruptedError("Job cancelled by user.")

                if kwargs.get("progress_unit") is not None:
                    return
                if is_start:
                    if override_num_inference_steps is not None:
                        total_steps[0] = override_num_inference_steps
                    if pass_no is not None:
                        current_pass[0] = pass_no
                    current_step[0] = 0
                    return
                current_step[0] += 1
                stage_name = "Stage 1 (half-res)" if current_pass[0] == 1 else ("Stage 2 (refine)" if current_pass[0] == 2 else "Diffusion")
                free_v = (torch.cuda.mem_get_info(0)[0] / 1024**3) if (HAS_TORCH and torch.cuda.is_available()) else 0
                print(f"  [{stage_name}] step {current_step[0]}/{total_steps[0]} | GPU0 VRAM free: {free_v:.2f} GB")
                sys.stdout.flush()

                frac = current_step[0] / max(total_steps[0], 1)
                frac = 0.73 + 0.22 * frac if current_pass[0] == 2 else frac * 0.73
                desc = f"{stage_name}: {current_step[0]}/{total_steps[0]}"
                if progress_callback:
                    progress_callback(min(frac, 0.95), desc)

            vae_tile_size = 512

            gen_kwargs = dict(
                input_prompt=prompt,
                image_start=image_start,
                height=height,
                width=width,
                frame_num=num_frames,
                fps=frame_rate,
                seed=seed,
                callback=cb,
                VAE_tile_size=vae_tile_size,
                input_video_strength=1.0,
                denoising_strength=1.0,
                guide_scale=1.0,
                audio_cfg_scale=1.0,
                alt_guide_scale=1.0,
                sampling_steps=8,
                guide_phases=guide_phases,
                n_prompt="",
                video_prompt_type="",
                audio_prompt_type="",
            )

            if input_waveform is not None:
                gen_kwargs["input_waveform"] = input_waveform
                gen_kwargs["input_waveform_sample_rate"] = input_waveform_sample_rate
                gen_kwargs["audio_scale"] = 1.0
                print("  Audio conditioning: ENABLED")

            _stage_labels = {
                "VAE Encoding":  ("🎞️ VAE Encoding input frames...",    0.05),
                "VAE Decoding":  ("🎬 VAE Decoding latents to video...", 0.88),
                "Upsampling":    ("🔭 Spatial upsampling latents...",    0.80),
            }
            _t = [time.time()]

            def set_progress_status(status: str):
                if check_cancelled and check_cancelled():
                    raise InterruptedError("Job cancelled by user.")
                dt = time.time() - _t[0]
                _t[0] = time.time()
                label, frac = _stage_labels.get(status, (f"⏳ {status}...", 0.85))
                print(f"  [{status}] {label} (+{dt:.1f}s)")
                sys.stdout.flush()
                if progress_callback:
                    progress_callback(frac, label)

            gen_kwargs["set_progress_status"] = set_progress_status

            t_start = time.time()
            result = self.model.generate(**gen_kwargs)
            total_gen_time = time.time() - t_start
            print(f"  [Pipeline Completed in {total_gen_time:.1f}s]")

            if progress_callback:
                progress_callback(0.96, "✅ Finalizing video output...")

            if result is None:
                return None, "Generation failed or was interrupted."

            # Parse result
            audio_data = None
            audio_sr = None
            if isinstance(result, dict):
                video_tensor = result.get("x")
                audio_data = result.get("audio")
                audio_sr = result.get("audio_sampling_rate", 24000)
            elif isinstance(result, tuple):
                video_tensor = result[0]
                if len(result) > 1: audio_data = result[1]
                if len(result) > 2: audio_sr = result[2]
            else:
                video_tensor = result

            if video_tensor is None or not (HAS_TORCH and torch.is_tensor(video_tensor)):
                return None, f"❌ No video tensor produced. Got: {type(video_tensor)}"

            video_tensor = video_tensor.cpu()
            gc.collect()
            if HAS_TORCH and torch.cuda.is_available():
                torch.cuda.empty_cache()

            from shared.utils.audio_video import save_video

            out_path = os.path.join(self.output_dir, f"ltx23_a2v_{time.strftime('%Y%m%d_%H%M%S')}_seed{seed}.mp4")
            if video_tensor.dtype == torch.uint8:
                save_video(tensor=video_tensor.unsqueeze(0), save_file=out_path, fps=frame_rate, nrow=1)
            else:
                save_video(tensor=video_tensor.unsqueeze(0).float() / 127.5 - 1.0, save_file=out_path, fps=frame_rate,
                           nrow=1, normalize=True, value_range=(-1, 1))

            # Mux audio into video
            if input_audio and audio_data is None:
                audio_data = input_waveform
                audio_sr = input_waveform_sample_rate

            if audio_data is not None:
                try:
                    audio_tmp = tempfile.mktemp(suffix=".wav")
                    if np is not None and isinstance(audio_data, np.ndarray) and sf is not None:
                        audio_np = audio_data
                        if audio_np.ndim == 2 and audio_np.shape[0] <= 2:
                            audio_np = audio_np.T
                        sf.write(audio_tmp, audio_np, int(audio_sr or 24000))
                    elif HAS_TORCH and torch.is_tensor(audio_data):
                        import torchaudio
                        cpu_audio = audio_data.cpu().float()
                        if cpu_audio.dim() == 1: cpu_audio = cpu_audio.unsqueeze(0)
                        if cpu_audio.dim() == 3: cpu_audio = cpu_audio.squeeze(0)
                        torchaudio.save(audio_tmp, cpu_audio, int(audio_sr or 24000))

                    final_path = out_path.replace(".mp4", "_with_audio.mp4")
                    subprocess.run([
                        "ffmpeg", "-y", "-i", out_path, "-i", audio_tmp,
                        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                        "-shortest", final_path
                    ], check=True, capture_output=True)
                    if os.path.exists(final_path) and os.path.getsize(final_path) > 0:
                        os.remove(out_path)
                        out_path = final_path
                        print(f"  ✅ Audio muxed: {out_path}")
                    if os.path.exists(audio_tmp):
                        os.remove(audio_tmp)
                except Exception as e:
                    print(f"  ⚠️ Audio mux note: {e}")

            del video_tensor
            gc.collect()
            if HAS_TORCH and torch.cuda.is_available():
                torch.cuda.empty_cache()

            if progress_callback:
                progress_callback(1.0, "Done!")

            summary = f"✅ Done in {total_gen_time:.1f}s! Seed: {seed} | {width}x{height} | {num_frames} frames"
            return out_path, summary

        except InterruptedError as ie:
            print(f"  🛑 Generation stopped: {ie}")
            gc.collect()
            if HAS_TORCH and torch.cuda.is_available():
                torch.cuda.empty_cache()
            return None, "🛑 Generation stopped by user."

        except Exception as e:
            traceback.print_exc()
            gc.collect()
            if HAS_TORCH and torch.cuda.is_available():
                torch.cuda.empty_cache()
            return None, f"❌ Error: {str(e)}"


# Global singleton instance
video_engine = LTXVideoEngine()
