#version 1
import gc
import os
import sys
import json
import random
import tempfile
import glob
import time
import traceback
import numpy as np
import subprocess
import psutil
import soundfile as sf
from PIL import Image

import zipfile
import shutil

# 1. Environment & Performance Settings
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.6"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
os.environ["MALLOC_TRIM_THRESHOLD_"] = "0"

# ---- bootstrap Wan2GP ----
WAN2GP_DIR = os.path.abspath("Wan2GP")

# Videos are saved in the working folder (/kaggle/working/outputs), visible in Kaggle's Output panel
OUTPUT_DIR = os.path.join(os.path.dirname(WAN2GP_DIR), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.environ.setdefault("GRADIO_TEMP_DIR", os.path.join(OUTPUT_DIR, ".gradio_cache"))
if WAN2GP_DIR not in sys.path:
    sys.path.insert(0, WAN2GP_DIR)
if os.path.exists(WAN2GP_DIR):
    os.chdir(WAN2GP_DIR)

import torch
import gradio as gr
from shared.utils.audio_video import save_video

# Detect Hardware
NUM_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 0
_GPU_SM = torch.cuda.get_device_capability(0) if NUM_GPUS > 0 else (0, 0)
_IS_SM60 = (_GPU_SM[0] == 6)

print("=" * 65)
print("🎤 LTX-2.3 22B Distilled - Audio-to-Video Engine")
print("📺 Created by: AIQUEST Academy")
print(f"🎮 Accelerator: {'Dual GPU (T4 x2)' if NUM_GPUS > 1 else 'Single GPU'} | Compute: sm_{_GPU_SM[0]}{_GPU_SM[1]}")
print("=" * 65)

# CUDA Attention & Convolution Optimizations for Tesla T4 (Turing sm_75)
torch.backends.cuda.enable_flash_sdp(False)          # sm_75 does not support FlashAttention v2
torch.backends.cuda.enable_mem_efficient_sdp(True)   # Fast memory-efficient attention (Cutlass kernel)
torch.backends.cuda.enable_math_sdp(True)            # Safe fallback
torch.backends.cudnn.benchmark = False               # Autotuning FP32 conv3d on a T4 stalls the VAE for 10+ minutes

# ==== 1. GEMMA 3 SDPA & ROPE DTYPE PATCH ====
def _patch_gemma3_attention():
    """
    Fixes RuntimeError: Expected query, key, and value to have the same dtype,
    got query.dtype: float key.dtype: float and value.dtype: c10::BFloat16.
    """
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

_patch_gemma3_attention()

# ==== 2. GGUF EXTENSION HANDLER & CONFIG PATCH ====
def _register_gguf_handler():
    """Register the GGUF handler with mmgp's quant_router."""
    try:
        import shared.qtypes.gguf
        print("  [GGUF] ✅ Extension handler registered with mmgp (Wan2GP native)")
    except Exception as e:
        print(f"  [GGUF] ⚠️ Could not register gguf handler: {e}")

def _patch_ltx2_config_loading():
    """Patch _load_config_from_checkpoint to handle GGUF metadata gracefully."""
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

_register_gguf_handler()
_patch_ltx2_config_loading()

# ==== 3. GPU SPECIFICATIONS & MEMORY STATS ====
if torch.cuda.is_available():
    print(f"Primary GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    if NUM_GPUS > 1:
        print(f"Secondary GPU: {torch.cuda.get_device_name(1)} (VRAM: {torch.cuda.get_device_properties(1).total_memory / 1024**3:.1f} GB)")
ram = psutil.virtual_memory()
print(f"System RAM: {ram.total / 1024**3:.1f} GB total, {ram.available / 1024**3:.1f} GB available")
sys.stdout.flush()

# ==== 4. LOAD LTX-2.3 MODEL COMPONENTS ====
print("\nLoading LTX-2.3 22B Distilled (GGUF Q4_K_M)...")
sys.stdout.flush()

from mmgp import offload
from shared.utils import files_locator as fl
from models.ltx2.ltx2_handler import family_handler

fl.set_checkpoints_paths(["models", "ckpts", "."])

base_model_type = "ltx2_22B"
model_def = {"ltx2_pipeline": "distilled"}
extra = family_handler.query_model_def(base_model_type, model_def)
model_def.update(extra)

gemma_folder = "models/gemma-3-12b-it-qat-q4_0-unquantized"
gemma_files = sorted(glob.glob(os.path.join(gemma_folder, "*.safetensors")))
quanto_files = [f for f in gemma_files if "quanto" in f]
text_encoder_file = quanto_files[0] if quanto_files else (gemma_files[0] if gemma_files else None)
if not text_encoder_file:
    raise FileNotFoundError(f"No .safetensors in {gemma_folder}. Check download cell.")
print(f"  Text encoder: {os.path.basename(text_encoder_file)}")

transformer_path = "/kaggle/working/Wan2GP/models/ltx-2.3-22b-distilled-1.1-Q4_K_M.gguf"
if not os.path.isfile(transformer_path):
    raise FileNotFoundError(f"{transformer_path} missing. Check download cell.")
print(f"  Transformer : {os.path.basename(transformer_path)}")
sys.stdout.flush()

# FP16 transformer: the T4's tensor cores make it ~3x faster than BF16 (the original May setup).
# Gemma's hidden states are kept in BF16 (see the dual-GPU section); a NaN guard after Stage 1 stops
# early if anything overflows. The video VAE + upsampler run in FP32 (converted below).
MODEL_DTYPE = torch.float16
VAE_DTYPE   = torch.float32

ltx2_model, pipe = family_handler.load_model(
    model_filename=transformer_path,
    model_type="ltx2_22B_distilled",
    base_model_type=base_model_type,
    model_def=model_def,
    dtype=MODEL_DTYPE,
    VAE_dtype=VAE_DTYPE,
    text_encoder_filename=text_encoder_file,
)

# Align Gemma embed_tokens to bfloat16 to avoid float32 inputs_embeds
try:
    _g_embed = ltx2_model.text_encoder.model.model.embed_tokens
    _g_embed.to(torch.bfloat16)
except Exception:
    try:
        ltx2_model.text_encoder.model.embed_tokens.to(torch.bfloat16)
    except Exception:
        pass

# ==== 4b. FP32 VIDEO VAE, DTYPE-SAFE FORWARDS & STAGE PROBES ====
# The LTX video VAE + latent upsampler must run in FP32 on a T4: in FP16/BF16 the decoded video
# collapses to flat grey with neon edges. The pipeline feeds them BF16 latents (the cause of
# "Input type (BFloat16) and bias type (Half)"), so their forwards cast the input to the weight
# dtype/device and the encoder hands its result back in the caller's dtype and device.
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

# distilled.py binds these helpers by name at import, so wrap them inside that module
import models.ltx2.ltx_pipelines.distilled as _distilled_mod
from models.ltx2.ltx_core.model.video_vae.tiling import TilingConfig, SpatialTilingConfig, TemporalTilingConfig
_orig_pipeline_upsample = _distilled_mod.upsample_video
_orig_pipeline_decode = _distilled_mod.vae_decode_video_to_tensor
UPS_ON_GPU1 = False  # set below once the upsampler + encoder are moved to GPU 1

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

def _probed_upsample_video(latent, video_encoder, upsampler):
    if _latent_stats("Stage 1 latent", latent):
        # Fail fast: NaN here always ends as a black video, no point spending minutes on Stage 2 + VAE
        raise RuntimeError("Stage 1 produced NaN latents (FP16 overflow). Check the '🔬 Gemma hidden states' line: "
                           "they must stay BF16. As a last resort set MODEL_DTYPE = torch.bfloat16 (slower).")
    in_dtype, in_device = latent.dtype, latent.device
    if UPS_ON_GPU1:
        with torch.cuda.device(1):
            out = _orig_pipeline_upsample(latent=latent.to("cuda:1"), video_encoder=video_encoder, upsampler=upsampler)
    else:
        out = _orig_pipeline_upsample(latent=latent, video_encoder=video_encoder, upsampler=upsampler)
    out = out.to(device=in_device, dtype=in_dtype)
    _latent_stats("Upscaled latent", out)
    return out

def _fast_vae_tiling(height, width):
    # Whole frames when they fit, official LTX overlaps (24 frames / 64 px), longest chunks the
    # T4 allows while the transformer is offloaded. ~217 B per pixel-frame measured for this decoder.
    budget = torch.cuda.get_device_properties(0).total_memory - 1.6e9 - 3.0e9
    bytes_per_pixel_frame = 230
    spatial = None
    tile_area = height * width
    if budget / (bytes_per_pixel_frame * tile_area) < 96:
        spatial = SpatialTilingConfig(tile_size_in_pixels=512, tile_overlap_in_pixels=64)
        tile_area = min(height, 512) * min(width, 512)
    frames = max(32, min(160, int(budget / (bytes_per_pixel_frame * tile_area)) // 8 * 8))
    return TilingConfig(spatial_config=spatial,
                        temporal_config=TemporalTilingConfig(tile_size_in_frames=frames, tile_overlap_in_frames=24))

def _probed_decode(latent, video_decoder, tiling_config=None, *args, **kwargs):
    _latent_stats("Final latent -> VAE", latent)
    latent_tensor = latent[0] if isinstance(latent, list) else latent  # decode clears the list; keep it for a retry
    fast_config = _fast_vae_tiling(kwargs.get("expected_height") or 480, kwargs.get("expected_width") or 832)
    print(f"  🎞️ VAE tiling: {'whole frame' if fast_config.spatial_config is None else '512 px tiles'}, "
          f"{fast_config.temporal_config.tile_size_in_frames}-frame chunks (24 overlap)")
    t_dec = time.time()
    torch.cuda.reset_peak_memory_stats(0)
    try:
        out = _orig_pipeline_decode([latent_tensor] if isinstance(latent, list) else latent_tensor,
                                    video_decoder, fast_config, *args, **kwargs)
    except torch.OutOfMemoryError:
        print("  ⚠️ VAE out of memory with large chunks, retrying with Wan2GP's default tiling...")
        gc.collect()
        torch.cuda.empty_cache()
        out = _orig_pipeline_decode([latent_tensor] if isinstance(latent, list) else latent_tensor,
                                    video_decoder, tiling_config, *args, **kwargs)
    print(f"  🎞️ VAE decode: {time.time() - t_dec:.1f}s (peak VRAM {torch.cuda.max_memory_allocated() / 1024**3:.1f} GB)")
    return out

_distilled_mod.upsample_video = _probed_upsample_video
_distilled_mod.vae_decode_video_to_tensor = _probed_decode

# ==== 5. DUAL T4 ACCELERATION (GPU T4 x2) ====
AUX_ON_GPU1 = False

if NUM_GPUS > 1:
    print("\n🚀 Dual GPU Active (GPU T4 x2) - Offloading Gemma 3 Text Encoder to GPU 1...")
    try:
        # Move Gemma 3 text encoder to GPU 1
        ltx2_model.text_encoder.to("cuda:1")
        _gemma = ltx2_model.text_encoder.model
        _gemma_lm = _gemma.model if hasattr(_gemma, "model") else _gemma
        if getattr(_gemma, "lm_head", None) is not None:
            _gemma.lm_head = torch.nn.Identity()

        # Token-embedding lookup on CPU (few ms) gives GPU 1 maximum free VRAM
        _embed = _gemma_lm.embed_tokens
        _embed.to("cpu")
        _orig_embed_fwd = _embed.forward

        def _cpu_embed_fwd(input_ids, *args, **kwargs):
            return _orig_embed_fwd(input_ids.to("cpu"), *args, **kwargs).to("cuda:1")

        _embed.forward = _cpu_embed_fwd
        _gemma.__class__ = type(_gemma.__class__.__name__, (_gemma.__class__,),
                                {"device": property(lambda self: torch.device("cuda", 1))})
        pipe.pop("text_encoder", None)

        # Wrap encode_raw to send output hidden_states & attention_mask back to cuda:0 for downstream pipeline
        _orig_encode_raw = ltx2_model.text_encoder.encode_raw

        def _dual_gpu_encode_raw(text, padding_side="left"):
            raw_emb = _orig_encode_raw(text, padding_side=padding_side)
            # Device move only: Gemma's hidden states must stay BF16 (values far above FP16's range
            # once squared by the feature extractor's RMS norm; casting them to FP16 gave all-NaN latents)
            hs_cuda0 = tuple(h.to(device="cuda:0") for h in raw_emb.hidden_states)
            _absmax = max(float(h.abs().max()) for h in hs_cuda0)
            print(f"  🔬 Gemma hidden states: {len(hs_cuda0)} layers, dtype={hs_cuda0[0].dtype}, absmax={_absmax:.0f}")
            mask_cuda0 = raw_emb.attention_mask.to("cuda:0")
            return raw_emb.__class__(hs_cuda0, mask_cuda0, raw_emb.padding_side)

        ltx2_model.text_encoder.encode_raw = _dual_gpu_encode_raw

        # Fail-safe device and dtype alignment in postprocess_text_embeddings
        try:
            import models.ltx2.ltx_core.text_encoders.gemma.encoders.base_encoder as _base_enc_mod
            _orig_postprocess = _base_enc_mod.postprocess_text_embeddings

            def _safe_postprocess(embeddings, feat_ext, emb_conn, aud_conn, return_attention_masks=False):
                target_dev = next(feat_ext.parameters()).device if any(True for _ in feat_ext.parameters()) else torch.device("cuda:0")
                safe_embs = []
                for item in embeddings:
                    hs = tuple(h.to(device=target_dev) for h in item.hidden_states)  # keep BF16 (see above)
                    mask = item.attention_mask.to(target_dev)
                    safe_embs.append(item.__class__(hs, mask, item.padding_side))
                return _orig_postprocess(safe_embs, feat_ext, emb_conn, aud_conn, return_attention_masks=return_attention_masks)

            _base_enc_mod.postprocess_text_embeddings = _safe_postprocess
            print("  [GPU T4 x2] ✅ postprocess_text_embeddings device & dtype alignment active.")
        except Exception as _pe:
            print(f"  [GPU T4 x2] ⚠️ Postprocess patch note: {_pe}")

        AUX_ON_GPU1 = True
        _free1 = torch.cuda.mem_get_info(1)[0] / 1024**3
        print(f"  [GPU T4 x2] ✅ Gemma 3 text encoder resident on GPU 1 ({_free1:.1f} GB free on GPU 1).")
        print("  [GPU T4 x2] ✅ GPU 0 is dedicated to the Transformer and VAE decoder.")
    except Exception as e:
        print(f"  [GPU T4 x2] ⚠️ Dual GPU setup notice: {e}, falling back to single GPU mode")
        AUX_ON_GPU1 = False

# The upsampler (between the stages) and the video encoder (reference image, once per stage) would
# each evict the transformer from GPU 0 under mmgp. Both fit next to Gemma on GPU 1 in FP32.
if AUX_ON_GPU1 and pipe.get("spatial_upsampler") is not None:
    with torch.cuda.device(1):
        torch.cuda.empty_cache()
    try:
        for _key in ("spatial_upsampler", "video_encoder"):
            pipe[_key].to("cuda:1")
        for _key in ("spatial_upsampler", "video_encoder"):
            pipe.pop(_key)
        UPS_ON_GPU1 = True
        print(f"  [GPU T4 x2] ✅ Spatial upsampler + video encoder resident on GPU 1 ({torch.cuda.mem_get_info(1)[0] / 1024**3:.1f} GB free).")
    except Exception as e:
        print(f"  [GPU T4 x2] ⚠️ Upsampler / encoder stay on GPU 0 ({e})")
        for _key in ("spatial_upsampler", "video_encoder"):
            getattr(ltx2_model, _key).to("cpu")
        with torch.cuda.device(1):
            torch.cuda.empty_cache()

sys.stdout.flush()

# ==== 6. APPLY MMGP OFFLOADING ON GPU 0 ====
# When Gemma 3 is on GPU 1, transformer budget on GPU 0 can be increased to 11000 MB!
# This keeps the ~11.8 GB Q4_K_M model resident in VRAM during diffusion (no PCIe streaming).
transformer_budget = 11000 if AUX_ON_GPU1 else 6000
print(f"\nApplying mmgp Profile 4 (Transformer VRAM budget: {transformer_budget} MB)...")
sys.stdout.flush()

budgets = {
    "transformer":       transformer_budget,
    "vae":               3500,   # FP32 video decoder (1.5 GB) + working memory
    "spatial_upsampler": 2000,
    "video_encoder":     2000,
    "audio_encoder":     1000,
    "audio_decoder":     1000,
    "vocoder":           500,
    "*":                 1000,
}
if not AUX_ON_GPU1:
    budgets["text_encoder"] = 1500
if UPS_ON_GPU1:
    budgets.pop("spatial_upsampler")
    budgets.pop("video_encoder")

offload.profile(
    pipe,
    profile_no=4,
    quantizeTransformer=False,
    convertWeightsFloatTo=torch.float16,
    budgets=budgets,
)
offload.shared_state["_attention"] = "sdpa"

print("\n✅ Setup complete! Distilled Audio-to-Video pipeline active.")
sys.stdout.flush()

# ==== 7. HELPER FUNCTIONS ====
AUTO_ASPECT = "Auto (match reference image)"

def get_resolution(base_res_str, aspect_ratio_str, ref_image=None):
    base_resolutions = {"1080p": 1088, "720p": 704, "540p": 544, "480p": 480, "320p": 320, "180p": 180}
    ratios = {
        "16:9 Landscape": 16/9, "4:3 Standard": 4/3,
        "1:1 Square": 1.0, "3:4 Portrait": 3/4, "9:16 Portrait": 9/16,
    }
    base = base_resolutions.get(base_res_str, 704)
    if aspect_ratio_str == AUTO_ASPECT and ref_image is not None:
        ratio = ref_image.width / ref_image.height  # no center-crop of the portrait / start frame
    else:
        ratio = ratios.get(aspect_ratio_str, 16/9)
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

# ==== 8. VIDEO GENERATION INFERENCE FUNCTION ====
@torch.inference_mode()
def Video_Generation(prompt, input_image_start, input_audio, seed, duration_dropdown,
                     resolution_dropdown, aspect_ratio_dropdown,
                     match_audio_dur=False, progress=gr.Progress()):
    try:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        progress(0, desc="Starting...")

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
        if input_audio is not None:
            wav, sr = sf.read(input_audio)
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            input_waveform = wav.astype(np.float32)
            input_waveform_sample_rate = int(sr)
            audio_duration_sec = len(wav) / sr
            print(f"  [Audio] Loaded: sr={sr}, duration={audio_duration_sec:.1f}s, shape={wav.shape}")

        if match_audio_dur and audio_duration_sec is not None:
            num_frames = snap_to_ltx_frames(audio_duration_sec, frame_rate)
            print(f"  [Duration] Match-Audio mode: {audio_duration_sec:.1f}s → {num_frames} frames "
                  f"({num_frames/frame_rate:.1f}s @ {frame_rate}fps)")
        else:
            num_frames = duration_map.get(duration_dropdown, 121)

        if seed is None or seed < 0:
            seed = random.randint(0, 2**32 - 1)
        seed = int(seed)

        image_start = None
        if input_image_start is not None:
            image_start = Image.open(input_image_start).convert("RGB")

        width, height = get_resolution(resolution_dropdown, aspect_ratio_dropdown, ref_image=image_start)

        guide_phases = 2  # two-stage: half-res pass, 2x latent upscale, refine pass

        free_vram0 = torch.cuda.mem_get_info(0)[0] / 1024**3
        free_vram1 = (torch.cuda.mem_get_info(1)[0] / 1024**3) if NUM_GPUS > 1 else 0
        ram_avail = psutil.virtual_memory().available / 1024**3

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
            if kwargs.get("progress_unit") is not None:
                return  # Gemma layer / VAE tile progress, not a diffusion step
            if is_start:
                if override_num_inference_steps is not None:
                    total_steps[0] = override_num_inference_steps
                if pass_no is not None:
                    current_pass[0] = pass_no
                current_step[0] = 0
                return
            current_step[0] += 1
            stage_name = "Stage 1 (half-res)" if current_pass[0] == 1 else ("Stage 2 (refine)" if current_pass[0] == 2 else "Diffusion")
            free_v = torch.cuda.mem_get_info(0)[0] / 1024**3
            # print(f"  [{stage_name}] step {current_step[0]}/{total_steps[0]} | GPU0 VRAM free: {free_v:.2f} GB")
            sys.stdout.flush()

            frac = current_step[0] / max(total_steps[0], 1)
            frac = 0.73 + 0.22 * frac if current_pass[0] == 2 else frac * 0.73
            progress(min(frac, 0.95), desc=f"{stage_name}: {current_step[0]}/{total_steps[0]}")

        vae_tile_size = 512  # replaced by the larger whole-frame plan in _probed_decode

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
            guide_scale=1.0,       # distilled model: guidance is baked in (Wan2GP uses 1.0)
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
            print("  Audio conditioning: ENABLED (soundtrack from the uploaded audio)")

        _stage_labels = {
            "VAE Encoding":  ("🎞️ VAE Encoding input frames...",    0.05),
            "VAE Decoding":  ("🎬 VAE Decoding latents to video...", 0.88),
            "Upsampling":    ("🔭 Spatial upsampling latents...",    0.80),
        }
        _t = [time.time()]

        def set_progress_status(status: str):
            dt = time.time() - _t[0]
            _t[0] = time.time()
            label, frac = _stage_labels.get(status, (f"⏳ {status}...", 0.85))
            print(f"  [{status}] {label} (+{dt:.1f}s)")
            sys.stdout.flush()
            progress(frac, desc=label)

        gen_kwargs["set_progress_status"] = set_progress_status

        t_start = time.time()
        result = ltx2_model.generate(**gen_kwargs)
        total_gen_time = time.time() - t_start
        print(f"  [Pipeline Completed in {total_gen_time:.1f}s]")

        progress(0.96, desc="✅ Finalizing video output...")
        sys.stdout.flush()

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

        if video_tensor is None or not torch.is_tensor(video_tensor):
            return None, f"❌ No video tensor produced. Got: {type(video_tensor)}"

        video_tensor = video_tensor.cpu()
        gc.collect()
        torch.cuda.empty_cache()

        # Save output video
        out_path = os.path.join(OUTPUT_DIR, f"ltx23_a2v_{time.strftime('%Y%m%d_%H%M%S')}_seed{seed}.mp4")
        if video_tensor.dtype == torch.uint8:
            save_video(tensor=video_tensor.unsqueeze(0), save_file=out_path, fps=frame_rate, nrow=1)
        else:
            save_video(tensor=video_tensor.unsqueeze(0).float() / 127.5 - 1.0, save_file=out_path, fps=frame_rate,
                       nrow=1, normalize=True, value_range=(-1, 1))

        # Mux audio into video
        if input_audio is not None and audio_data is None:
            audio_data = input_waveform
            audio_sr = input_waveform_sample_rate

        if audio_data is not None:
            try:
                audio_tmp = tempfile.mktemp(suffix=".wav")
                if isinstance(audio_data, np.ndarray):
                    audio_np = audio_data
                    if audio_np.ndim == 2 and audio_np.shape[0] <= 2:
                        audio_np = audio_np.T
                    sf.write(audio_tmp, audio_np, int(audio_sr or 24000))
                elif torch.is_tensor(audio_data):
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
                    os.remove(out_path)  # keep only the version with audio in the outputs folder
                    out_path = final_path
                    print(f"  ✅ Audio muxed: {out_path}")
                if os.path.exists(audio_tmp):
                    os.remove(audio_tmp)
            except Exception as e:
                print(f"  ⚠️ Audio mux note: {e}")

        del video_tensor
        gc.collect()
        torch.cuda.empty_cache()
        progress(1.0, desc="Done!")
        return out_path, f"✅ Done in {total_gen_time:.1f}s! Seed: {seed} | {width}x{height} | {num_frames} frames"

    except Exception as e:
        traceback.print_exc()
        gc.collect()
        torch.cuda.empty_cache()
        return None, f"❌ Error: {str(e)}"

UPLOAD_DIR = "/kaggle/working/uploads"

# ==== PROJECT PROCESSING ====

def process_video(image_file, audio_file, prompt, piece_number=1):
    """
    Generate one video from an image + audio + prompt.
    """

    print("\n" + "=" * 65)
    print(f"🎬 Processing piece {piece_number}")
    print(f"🖼️ Image : {image_file}")
    print(f"🎤 Audio : {audio_file}")
    print(f"📝 Prompt: {prompt}")
    print("=" * 65)
    sys.stdout.flush()

    if not image_file or not os.path.isfile(image_file):
        print(f"❌ Image not found: {image_file}")
        return None, f"Image not found: {image_file}"

    if not audio_file or not os.path.isfile(audio_file):
        print(f"❌ Audio not found: {audio_file}")
        return None, f"Audio not found: {audio_file}"

    try:
        result = Video_Generation(
            prompt=prompt,
            input_image_start=image_file,
            input_audio=audio_file,
            seed=-1,
            duration_dropdown="5 Seconds (121 frames)",
            resolution_dropdown="720p",
            aspect_ratio_dropdown=AUTO_ASPECT,
            match_audio_dur=True,
        )

        video_path, status = result

        if video_path:
            print(f"✅ Piece {piece_number} complete:")
            print(f"   {video_path}")
        else:
            print(f"❌ Piece {piece_number} failed:")
            print(f"   {status}")

        return video_path, status

    except Exception as e:
        traceback.print_exc()
        return None, f"❌ Piece {piece_number} error: {e}"


def process_project(folder_path):
    """
    Read manifest.json and generate all project pieces.
    """

    if not folder_path:
        return "❌ No project folder selected."

    manifest_path = os.path.join(folder_path, "manifest.json")

    if not os.path.exists(manifest_path):
        return (
            f"❌ manifest.json not found.\n\n"
            f"Expected:\n{manifest_path}"
        )

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception as e:
        return f"❌ Could not read manifest.json:\n{e}"

    pieces = manifest.get("pieces", [])

    if not pieces:
        return "❌ No pieces found in manifest.json"

    print("\n" + "=" * 65)
    print(f"🚀 PROJECT START")
    print(f"📁 Folder : {folder_path}")
    print(f"🎬 Pieces : {len(pieces)}")
    print("=" * 65)
    sys.stdout.flush()

    results = []
    successful = 0
    failed = 0

    for index, piece in enumerate(pieces, start=1):

        image_name = piece.get("imageFileName")
        audio_name = piece.get("audioFileName")
        prompt = piece.get("prompt", "")

        if not image_name:
            failed += 1
            results.append(f"❌ Piece {index}: missing imageFileName")
            continue

        if not audio_name:
            failed += 1
            results.append(f"❌ Piece {index}: missing audioFileName")
            continue

        # Support files inside subdirectories of the ZIP.
        image_file = os.path.join(folder_path, image_name)
        audio_file = os.path.join(folder_path, audio_name)

        # Also support manifest paths that use forward slashes.
        image_file = os.path.normpath(image_file)
        audio_file = os.path.normpath(audio_file)

        video_path, status = process_video(
            image_file=image_file,
            audio_file=audio_file,
            prompt=prompt,
            piece_number=index,
        )

        if video_path:
            successful += 1
            results.append(
                f"✅ Piece {index}: {os.path.basename(video_path)}"
            )
        else:
            failed += 1
            results.append(
                f"❌ Piece {index}: {status}"
            )

        gc.collect()
        torch.cuda.empty_cache()

    summary = (
        "\n" +
        "=" * 65 +
        "\n"
        "🏁 PROJECT COMPLETE\n"
        f"Total pieces : {len(pieces)}\n"
        f"Successful   : {successful}\n"
        f"Failed       : {failed}\n"
        f"Output       : {OUTPUT_DIR}\n"
        + "=" * 65 +
        "\n\n"
        + "\n".join(results)
    )

    print(summary)
    sys.stdout.flush()

    return summary

# ============================================================
# ZIP PROJECT UPLOAD / EXTRACTION
# ============================================================

PROJECT_ROOT = "/kaggle/working/projects"
os.makedirs(PROJECT_ROOT, exist_ok=True)


def upload_and_extract(zip_file):
    """
    Extract an uploaded project ZIP into /kaggle/working/projects.

    Expected ZIP structure:

        project.zip
        ├── manifest.json
        ├── image_001.png
        ├── audio_001.wav
        └── ...

    Returns:
        status message
        extracted project folder
    """

    if not zip_file:
        return "❌ Please upload a project ZIP file.", ""

    try:
        # Gradio may return a filepath string or an object depending on version
        if isinstance(zip_file, str):
            zip_path = zip_file
        elif hasattr(zip_file, "name"):
            zip_path = zip_file.name
        else:
            zip_path = str(zip_file)

        if not os.path.isfile(zip_path):
            return f"❌ ZIP file not found:\n{zip_path}", ""

        if not zipfile.is_zipfile(zip_path):
            return "❌ The uploaded file is not a valid ZIP archive.", ""

        print("\n" + "=" * 65)
        print("📦 PROJECT ZIP UPLOAD")
        print(f"ZIP: {zip_path}")
        print("=" * 65)
        sys.stdout.flush()

        # Create unique project folder
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        zip_name = os.path.splitext(os.path.basename(zip_path))[0]

        # Remove unsafe characters
        safe_name = "".join(
            c if c.isalnum() or c in "-_" else "_"
            for c in zip_name
        )

        project_folder = os.path.join(
            PROJECT_ROOT,
            f"{safe_name}_{timestamp}"
        )

        os.makedirs(project_folder, exist_ok=True)

        # ----------------------------------------------------
        # Secure ZIP extraction
        # ----------------------------------------------------
        with zipfile.ZipFile(zip_path, "r") as z:
            members = z.infolist()

            for member in members:

                # Prevent Zip Slip/path traversal
                target_path = os.path.abspath(
                    os.path.join(project_folder, member.filename)
                )

                if not target_path.startswith(
                    os.path.abspath(project_folder) + os.sep
                ):
                    raise RuntimeError(
                        f"Unsafe ZIP path detected: {member.filename}"
                    )

            z.extractall(project_folder)

        print(f"✅ Extracted to: {project_folder}")

        # ----------------------------------------------------
        # Find manifest.json
        # ----------------------------------------------------
        manifest_path = os.path.join(
            project_folder,
            "manifest.json"
        )

        # Sometimes ZIP contains a top-level directory:
        #
        # project.zip
        # └── MyProject/
        #     ├── manifest.json
        #
        if not os.path.isfile(manifest_path):

            found_manifests = []

            for root, dirs, files in os.walk(project_folder):
                if "manifest.json" in files:
                    found_manifests.append(
                        os.path.join(root, "manifest.json")
                    )

            if len(found_manifests) == 1:

                manifest_path = found_manifests[0]

                # Use the folder containing manifest.json
                project_folder = os.path.dirname(manifest_path)

                print(
                    f"📄 Found nested manifest: {manifest_path}"
                )

            elif len(found_manifests) == 0:

                return (
                    "❌ ZIP extracted successfully, but "
                    "manifest.json was not found.\n\n"
                    f"Extracted to:\n{project_folder}",
                    project_folder
                )

            else:

                return (
                    "❌ Multiple manifest.json files found.\n\n"
                    + "\n".join(found_manifests),
                    project_folder
                )

        # ----------------------------------------------------
        # Validate manifest
        # ----------------------------------------------------
        try:

            with open(
                manifest_path,
                "r",
                encoding="utf-8"
            ) as f:
                manifest = json.load(f)

        except Exception as e:

            return (
                f"❌ manifest.json was found but could not be read:\n\n{e}",
                project_folder
            )

        pieces = manifest.get("pieces", [])

        if not pieces:

            return (
                "⚠️ ZIP extracted and manifest.json was found, "
                "but it contains no pieces.\n\n"
                f"Manifest:\n{manifest_path}",
                project_folder
            )

        # ----------------------------------------------------
        # Validate referenced files
        # ----------------------------------------------------
        missing = []

        for index, piece in enumerate(pieces, start=1):

            image_name = piece.get("imageFileName")
            audio_name = piece.get("audioFileName")

            if not image_name:
                missing.append(
                    f"Piece {index}: missing imageFileName"
                )

            if not audio_name:
                missing.append(
                    f"Piece {index}: missing audioFileName"
                )

            if image_name:
                image_path = os.path.normpath(
                    os.path.join(project_folder, image_name)
                )

                if not os.path.isfile(image_path):
                    missing.append(
                        f"Piece {index}: image not found: {image_name}"
                    )

            if audio_name:
                audio_path = os.path.normpath(
                    os.path.join(project_folder, audio_name)
                )

                if not os.path.isfile(audio_path):
                    missing.append(
                        f"Piece {index}: audio not found: {audio_name}"
                    )

        # ----------------------------------------------------
        # Build status
        # ----------------------------------------------------
        status_lines = [
            "✅ PROJECT ZIP EXTRACTED",
            "",
            f"📦 ZIP: {os.path.basename(zip_path)}",
            f"📂 Folder: {project_folder}",
            f"📄 Manifest: {manifest_path}",
            f"🎬 Pieces: {len(pieces)}",
            "",
        ]

        if missing:

            status_lines.append(
                f"⚠️ {len(missing)} validation problem(s):"
            )

            status_lines.extend(
                f"  ❌ {item}"
                for item in missing[:30]
            )

            if len(missing) > 30:
                status_lines.append(
                    f"  ... and {len(missing) - 30} more"
                )

        else:

            status_lines.append(
                "✅ All image and audio files referenced "
                "by manifest.json were found."
            )

            status_lines.append("")
            status_lines.append(
                "▶️ Click 'Run Project' to start generation."
            )

        status = "\n".join(status_lines)

        print(status)
        sys.stdout.flush()

        return status, project_folder

    except Exception as e:

        traceback.print_exc()

        return (
            f"❌ ZIP extraction failed:\n\n{str(e)}",
            ""
        )

# ==== 9. AIQUEST BRANDED GRADIO INTERFACE ====
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
* { font-family: 'Inter', sans-serif !important; }
.gradio-container { max-width: 1000px !important; margin: auto !important; }
.brand-header { text-align: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 28px; border-radius: 15px; margin-bottom: 20px; box-shadow: 0 10px 25px rgba(102,126,234,0.3); }
.brand-title { color: white; font-size: 2em; font-weight: 700; margin: 0 0 6px 0; }
.brand-subtitle { color: rgba(255,255,255,0.88); font-size: 1em; margin-bottom: 16px; }
.btn-row { display: flex; justify-content: center; gap: 10px; flex-wrap: wrap; }
.social-btn { display: inline-flex; align-items: center; justify-content: center; min-width: 150px; padding: 10px 18px; border-radius: 10px; font-weight: 700; font-size: 13px; text-decoration: none; color: white; white-space: nowrap; }
.yt-btn  { background: #FF0000; box-shadow: 0 4px 12px rgba(255,0,0,0.3); }
.x-btn   { background: #000000; box-shadow: 0 4px 12px rgba(0,0,0,0.25); }
.sup-btn { background: linear-gradient(135deg,#f6d365,#fda085); box-shadow: 0 4px 12px rgba(253,160,133,0.35); }
button.primary { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important; color: white !important; font-weight: 600 !important; border-radius: 12px !important; }
#stop-btn { background: linear-gradient(135deg, #ef4444 0%, #b91c1c 100%) !important; color: white !important; font-weight: 600 !important; border-radius: 12px !important; }
#clear-btn { background: linear-gradient(135deg, #6b7280 0%, #374151 100%) !important; color: white !important; font-weight: 600 !important; border-radius: 12px !important; }
.status-line { text-align: center; color: #6b7280; font-size: 13px; margin: -8px 0 12px 0; }
.brand-footer { text-align: center; color: #6b7280; font-size: 12px; margin-top: 24px; }
"""

BRAND_HTML = """
<div class="brand-header">
  <div class="brand-title">🎤 LTX-2.3 22B Audio-to-Video / Lip Sync</div>
  <div class="brand-subtitle">Created by <strong>AIQuest Academy</strong> &nbsp;|&nbsp; Kaggle GPU T4 x2 Edition</div>
  <div class="btn-row">
    <a href="https://www.youtube.com/@aiquestacademy?sub_confirmation=1" target="_blank" class="social-btn yt-btn">▶ Subscribe</a>
    <a href="https://x.com/aiquestacademy" target="_blank" class="social-btn x-btn">𝕏 Follow on X</a>
    <a href="https://aiquest.site" target="_blank" class="social-btn sup-btn">❤️ Support My Work</a>
  </div>
</div>
"""

_gpu_layout = "Gemma + upsampler on GPU 1" if UPS_ON_GPU1 else ("Gemma on GPU 1" if AUX_ON_GPU1 else "single GPU")
STATUS_HTML = f'<div class="status-line">⚡ GGUF Q4_K_M transformer (FP16) &nbsp;|&nbsp; {_gpu_layout} &nbsp;|&nbsp; FP32 VAE</div>'

FOOTER_HTML = """
<div class="brand-footer">⚡ Made with ❤️ by <strong>AIQUEST Academy</strong> · aiquest.site · © All rights reserved</div>
"""

with gr.Blocks(theme=gr.themes.Soft(), css=CSS, title="LTX-2.3 Audio-to-Video | AIQUEST Academy") as demo:
    gr.HTML(BRAND_HTML)
    gr.HTML(STATUS_HTML)

    gr.Markdown(
        "💡 **Tip:** Tag your prompt with these exciting things `[VISUAL] <scene> [SPEECH] <dialogue> [SOUND] <voice description>` for highest accuracy."
    )

        # ============================================================
    # PROJECT ZIP PROCESSING
    # ============================================================

    with gr.Accordion("📦 Batch Project Processing", open=True):

        gr.Markdown(
            """
            ### 📦 Process a complete AI video project

            Upload a ZIP containing:

            `manifest.json`

            plus the referenced image and audio files.

            Example:

            ```
            project.zip
            ├── manifest.json
            ├── image_001.png
            ├── audio_001.wav
            ├── image_002.png
            ├── audio_002.wav
            └── ...
            ```

            The manifest should contain:

            ```json
            {
              "pieces": [
                {
                  "imageFileName": "image_001.png",
                  "audioFileName": "audio_001.wav",
                  "prompt": "[VISUAL] ... [SPEECH] ... [SOUND] ..."
                }
              ]
            }
            ```
            """
        )

        with gr.Row():

            project_zip = gr.File(
                label="📦 Upload Project ZIP",
                file_types=[".zip"],
                type="filepath"
            )

            project_folder = gr.Textbox(
                label="📂 Extracted Project Folder",
                value="",
                interactive=False
            )

        with gr.Row():

            extract_btn = gr.Button(
                "📂 Extract ZIP",
                variant="secondary",
                size="lg"
            )

            run_project_btn = gr.Button(
                "▶️ Run Project",
                variant="primary",
                size="lg"
            )

        project_status = gr.Textbox(
            label="📋 Project Status",
            lines=12,
            interactive=False
        )

        extract_btn.click(
            fn=upload_and_extract,
            inputs=[project_zip],
            outputs=[project_status, project_folder],
        )

        run_project_btn.click(
            fn=process_project,
            inputs=[project_folder],
            outputs=[project_status],
        )

    with gr.Column():
        prompt = gr.Textbox(
            label="📝 Prompt / Prompt Queue",
            lines=8,
            placeholder=(
                "Enter one prompt per line. Use Generate Video for a single prompt, "
                "or Queue Prompts to generate each non-empty line sequentially.\n\n"
                "[VISUAL] A professional speaker talking directly to camera in a modern studio. "
                "[SPEECH] Welcome everyone to our new AI tutorial! [SOUND] Clear expressive voice.\n"
                "[VISUAL] A cinematic drone shot over a medieval castle at sunrise. [SOUND] Natural ambience."
            )
        )

        with gr.Row():
            input_image_start = gr.Image(type="filepath", label="🖼️ Reference Image (Portrait / Start Frame)")
            input_audio = gr.Audio(type="filepath", label="🎤 Speech / Vocals Audio Track")

        with gr.Row():
            seed = gr.Number(label="🎲 Seed (-1 for Random)", value=-1, precision=0)
            duration_dropdown = gr.Dropdown(
                label="⏱️ Duration",
                choices=[
                    "2 Seconds (49 frames)",
                    "3 Seconds (73 frames)",
                    "5 Seconds (121 frames)",
                    "8 Seconds (193 frames)",
                    "10 Seconds (241 frames)",
                    "15 Seconds (361 frames)",
                    "20 Seconds (481 frames)",
                ],
                value="5 Seconds (121 frames)",
            )
            match_audio_dur = gr.Checkbox(
                label="🎵 Auto-match video duration to uploaded audio file",
                value=False,
                info="When checked, snaps video frame count automatically to the exact duration of your audio file.",
            )

        match_audio_dur.change(
            fn=lambda chk: gr.update(interactive=not chk),
            inputs=[match_audio_dur],
            outputs=[duration_dropdown],
        )

        with gr.Row():
            resolution_dropdown = gr.Dropdown(
                label="📐 Base Resolution",
                choices=["720p", "540p", "480p", "360p", "180p"],
                value="720p",
            )
            aspect_ratio_dropdown = gr.Dropdown(
                label="📏 Aspect Ratio",
                choices=[AUTO_ASPECT, "16:9 Landscape", "4:3 Standard", "1:1 Square", "3:4 Portrait", "9:16 Portrait"],
                value=AUTO_ASPECT,
                info="Auto follows the reference image (16:9 without one)",
            )

        with gr.Row():
            gen_btn  = gr.Button("🎬 Generate Video", variant="primary", size="lg", elem_id="gen-btn")
            queue_btn = gr.Button("📋 Queue Prompts", variant="primary", size="lg", elem_id="queue-btn")
            stop_btn = gr.Button("🛑 Stop", variant="secondary", size="lg", elem_id="stop-btn")
            clear_btn = gr.Button("🗑️ Clear", variant="secondary", size="lg", elem_id="clear-btn")

        video_out = gr.Video(label="🎥 Generated Video")
        latest_btn = gr.Button("📂 Load Latest Video (use if the UI showed an error)", variant="secondary")
        status_out = gr.Textbox(label="ℹ️ Status", interactive=False)

        def load_latest_video():
            videos = sorted(glob.glob(os.path.join(OUTPUT_DIR, "*.mp4")), key=os.path.getmtime)
            if not videos:
                return None, f"No videos in {OUTPUT_DIR} yet."
            return videos[-1], f"📁 Loaded latest video: {videos[-1]}"

        latest_btn.click(fn=load_latest_video, outputs=[video_out, status_out])

        generation_inputs = [
            prompt, input_image_start, input_audio, seed, duration_dropdown,
            resolution_dropdown, aspect_ratio_dropdown, match_audio_dur
        ]

        gen_event = gen_btn.click(
            fn=Video_Generation,
            inputs=generation_inputs,
            outputs=[video_out, status_out],
        )

        def Generate_Prompt_Queue(prompts_text, input_image_start, input_audio, seed,
                                  duration_dropdown, resolution_dropdown,
                                  aspect_ratio_dropdown, match_audio_dur,
                                  progress=gr.Progress()):
            """Generate each non-empty prompt line sequentially with the loaded model."""
            prompts = [line.strip() for line in (prompts_text or "").splitlines() if line.strip()]
            if not prompts:
                yield None, "❌ Queue is empty. Enter one prompt per line."
                return

            total = len(prompts)
            completed = 0
            failed = 0
            latest_video = None
            log = [f"📋 Queue created: {total} prompt(s). Processing sequentially."]
            yield None, "\n".join(log)

            for index, prompt_text in enumerate(prompts, start=1):
                progress((index - 1) / total, desc=f"Starting prompt {index}/{total}")
                log.append(f"\n⏳ [{index}/{total}] {prompt_text}")
                yield latest_video, "\n".join(log)

                # Pass the same selected image/audio/settings to each prompt.
                # Video_Generation handles random seeds when seed is -1.
                video_path, status = Video_Generation(
                    prompt_text,
                    input_image_start,
                    input_audio,
                    seed,
                    duration_dropdown,
                    resolution_dropdown,
                    aspect_ratio_dropdown,
                    match_audio_dur,
                    progress=progress
                )

                if video_path:
                    latest_video = video_path
                    completed += 1
                    log.append(f"✅ [{index}/{total}] Completed: {video_path}\n{status}")
                else:
                    failed += 1
                    log.append(f"❌ [{index}/{total}] Failed: {status}")

                progress(index / total, desc=f"Finished prompt {index}/{total}")
                yield latest_video, "\n".join(log)

            log.append(f"\n🏁 Queue finished: {completed} completed, {failed} failed, {total} total.")
            yield latest_video, "\n".join(log)

        queue_event = queue_btn.click(
            fn=Generate_Prompt_Queue,
            inputs=[
                prompt, input_image_start, input_audio, seed, duration_dropdown,
                resolution_dropdown, aspect_ratio_dropdown, match_audio_dur
            ],
            outputs=[video_out, status_out],
        )

        stop_btn.click(fn=None, cancels=[gen_event, queue_event])
        clear_btn.click(
            fn=lambda: (None, "", "", -1),
            outputs=[video_out, status_out, prompt, seed],
        )

    gr.HTML(FOOTER_HTML)

# ==== 10. CLOUDFLARE QUICK TUNNEL (second public link in case the Gradio share link fails) ====
SERVER_PORT = 7860

def start_cloudflare_tunnel(port):
    import re
    import stat
    import threading
    import urllib.request
    binary = "/tmp/cloudflared"
    try:
        if not os.path.exists(binary):
            urllib.request.urlretrieve(
                "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64", binary)
            os.chmod(binary, os.stat(binary).st_mode | stat.S_IEXEC)
        tunnel = subprocess.Popen([binary, "tunnel", "--url", f"http://127.0.0.1:{port}", "--no-autoupdate"],
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    except Exception as e:
        print(f"⚠️ Cloudflare tunnel unavailable: {e}")
        return

    def _watch():
        for line in tunnel.stdout:
            match = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", line)
            if match:
                print(f"* Cloudflare tunnel URL: {match.group(0)}  (use this if the Gradio link does not load)")
                sys.stdout.flush()
                break
        for _ in tunnel.stdout:  # keep draining so cloudflared never blocks on a full pipe
            pass

    threading.Thread(target=_watch, daemon=True).start()

print("\nLaunching Gradio Interface...")
sys.stdout.flush()
start_cloudflare_tunnel(SERVER_PORT)
demo.queue()
demo.launch(server_name="127.0.0.1", server_port=SERVER_PORT, share=True, inline=False, debug=True, show_error=True, max_threads=1, ssr_mode=False, allowed_paths=[OUTPUT_DIR])