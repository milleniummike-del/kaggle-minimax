import gc
import os
import sys
import json
import random
import tempfile
import glob
import traceback
import numpy as np
import subprocess
import psutil
import soundfile as sf
from PIL import Image

# ---- bootstrap Wan2GP ----
WAN2GP_DIR = os.path.abspath("Wan2GP")
sys.path.insert(0, WAN2GP_DIR)
os.chdir(WAN2GP_DIR)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.5"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

import torch

_GPU_SM = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
_IS_SM60 = (_GPU_SM[0] == 6)
_IS_SM70_PLUS = (_GPU_SM[0] >= 7)

if _IS_SM60:
    os.environ["WGP_DTYPE"] = "fp16"
    print(f"  [GPU] sm_60 detected (P100) — FP16 mode + CPU audio patches enabled")
else:
    print(f"  [GPU] sm_{_GPU_SM[0]}{_GPU_SM[1]} detected — native CUDA mode, no patches needed")
import gradio as gr
from shared.utils.audio_video import save_video

def _register_gguf_handler():
    import shared.qtypes.gguf
    print("  [GGUF] ✅ Extension handler registered with mmgp (Wan2GP native)")

def _patch_ltx2_config_loading():
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
        except Exception as e:
            print(f"  [GGUF Patch] Metadata read: {type(e).__name__}, using fallback config")
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

print(f"GPU: {torch.cuda.get_device_name()}")
print(f"Compute Capability: {torch.cuda.get_device_capability()}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
ram = psutil.virtual_memory()
print(f"RAM: {ram.total / 1024**3:.1f} GB total, {ram.available / 1024**3:.1f} GB available")
sys.stdout.flush()

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)

print("\nLoading LTX-2.3 22B Distilled (GGUF Q4_K_M)...")
sys.stdout.flush()

from mmgp import offload
from shared.utils import files_locator as fl

fl.set_checkpoints_paths(["models", "ckpts", "."])

from models.ltx2.ltx2_handler import family_handler

_register_gguf_handler()
_patch_ltx2_config_loading()

class _AudioEncoderP100Wrapper:
    def __init__(self, encoder):
        object.__setattr__(self, '_enc', encoder)

    def __call__(self, mel):
        if torch.cuda.is_available():
            mel = mel.to(
                device=torch.device("cuda", torch.cuda.current_device()),
                dtype=torch.float16,
            )
        return object.__getattribute__(self, '_enc')(mel)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, '_enc'), name)

    def __setattr__(self, name, value):
        if name == '_enc':
            object.__setattr__(self, name, value)
        else:
            setattr(object.__getattribute__(self, '_enc'), name, value)

    def __repr__(self):
        return f"_AudioEncoderP100Wrapper({object.__getattribute__(self, '_enc')!r})"

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

transformer_path = os.path.join("models", "ltx-2.3-22b-distilled-Q4_K_M_light.gguf")
if not os.path.isfile(transformer_path):
    raise FileNotFoundError(f"{transformer_path} missing. Check download cell.")
print(f"  Transformer : {os.path.basename(transformer_path)}")
sys.stdout.flush()

MODEL_DTYPE = torch.float16
VAE_DTYPE   = torch.float16

ltx2_model, pipe = family_handler.load_model(
    model_filename=transformer_path,
    model_type="ltx2_22B_distilled",
    base_model_type=base_model_type,
    model_def=model_def,
    dtype=MODEL_DTYPE,
    VAE_dtype=VAE_DTYPE,
    text_encoder_filename=text_encoder_file,
)

print("\n--- Pipeline Components ---")
for name, component in pipe.items():
    if component is not None:
        ctype = type(component).__name__
        if hasattr(component, 'parameters'):
            try:
                p = next(component.parameters())
                print(f"  {name}: {ctype} (dtype={p.dtype})")
            except StopIteration:
                print(f"  {name}: {ctype} (no params)")
        else:
            print(f"  {name}: {ctype}")
    else:
        print(f"  {name}: None")
sys.stdout.flush()

if _IS_SM60:
    try:
        import torch.nn.functional as _F
        from models.ltx2.ltx_core.model.audio_vae.causal_conv_2d import CausalConv2d as _CC2d

        def _cc2d_cpu_pad(self, x: torch.Tensor) -> torch.Tensor:
            if x.is_cuda:
                dev, dt = x.device, x.dtype
                x_cpu = x.detach().cpu().float()
                x_cpu = _F.pad(x_cpu, self.padding)
                w = self.conv.weight.detach().cpu().float()
                b = self.conv.bias.detach().cpu().float() if self.conv.bias is not None else None
                out = _F.conv2d(x_cpu, w, b,
                                self.conv.stride, self.conv.padding,
                                self.conv.dilation, self.conv.groups)
                return out.to(device=dev, dtype=dt)
            else:
                x = _F.pad(x, self.padding)
                return self.conv(x)

        _CC2d.forward = _cc2d_cpu_pad
        print("  [sm_60 Fix] ✅ CausalConv2d patched: pad+conv run on CPU")
    except Exception as _e:
        print(f"  [sm_60 Fix] ❌ Could not patch CausalConv2d: {_e}")
else:
    print("  [sm_60 Fix] ⏭️  Skipped (not sm_60)")

print("\nApplying mmgp Profile 4 with per-model budgets...")
sys.stdout.flush()

offload.profile(
    pipe,
    profile_no=4,
    quantizeTransformer=False,
    convertWeightsFloatTo=torch.float16,
    budgets={
        "transformer":       6000,
        "text_encoder":      1500,
        "video_encoder":     2000,
        "video_decoder":     3000,
        "audio_encoder":     1000,
        "audio_decoder":     1000,
        "vocoder":           500,
        "spatial_upsampler": 1500,
        "vae":               1000,
        "*":                 1000,
    },
)
offload.shared_state["_attention"] = "sdpa"

print("\n✅ Setup complete! Distilled Audio-to-Video pipeline active.")
sys.stdout.flush()

def get_resolution(base_res_str, aspect_ratio_str):
    base_resolutions = {"1080p": 1088, "720p": 704, "540p": 544, "480p": 480}
    ratios = {
        "16:9 Landscape": 16/9, "4:3 Standard": 4/3,
        "1:1 Square": 1.0, "3:4 Portrait": 3/4, "9:16 Portrait": 9/16,
    }
    base = base_resolutions.get(base_res_str, 704)
    ratio = ratios.get(aspect_ratio_str, 16/9)
    if ratio >= 1.0:
        height = base
        width = int(base * ratio)
    else:
        width = base
        height = int(base / ratio)
    return (width // 32) * 32, (height // 32) * 32

def get_vae_tile_size(height, width):
    vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024**2)
    effective_vram = vram_mb / 1.5
    if effective_vram >= 24000: vae_config = 1
    elif effective_vram >= 8000: vae_config = 2
    else: vae_config = 3
    if max(height, width) > 480: vae_config += 1
    if vae_config <= 1: return 0
    elif vae_config == 2: return 512
    elif vae_config == 3: return 256
    return 128

def snap_to_ltx_frames(duration_sec: float, fps: float = 24.0, max_frames: int = 721) -> int:
    raw = duration_sec * fps
    k = max(0, round((raw - 1) / 8))
    frames = 8 * k + 1
    frames = max(49, min(frames, max_frames))
    return int(frames)

@torch.inference_mode()
def Video_Generation(prompt, input_image_start, input_audio, seed, duration_dropdown,
                     resolution_dropdown, aspect_ratio_dropdown,
                     guide_scale=4.0, match_audio_dur=False, progress=gr.Progress()):
    try:
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
        progress(0, desc="Starting...")

        duration_map = {
            "2 Seconds (49 frames)":  49,
            "3 Seconds (73 frames)":  73,
            "5 Seconds (121 frames)": 121,
            "8 Seconds (193 frames)": 193,
            "10 Seconds (241 frames)": 241,
            "15 Seconds (361 frames)": 361,
            "20 Seconds (481 frames)": 481,
            "25 Seconds (601 frames)": 601,
            "30 Seconds (721 frames)": 721,
        }
        frame_rate = 24.0

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
            print(f"  [Duration] Match-Audio mode: {audio_duration_sec:.1f}s → {num_frames} frames")
        else:
            num_frames = duration_map.get(duration_dropdown, 121)

        width, height = get_resolution(resolution_dropdown, aspect_ratio_dropdown)

        if seed is None or seed < 0:
            seed = random.randint(0, 2**32 - 1)
        seed = int(seed)

        image_start = None
        if input_image_start is not None:
            image_start = Image.open(input_image_start).convert("RGB")

        free_vram = torch.cuda.mem_get_info()[0] / 1024**3
        ram = psutil.virtual_memory()
        print(f"\n{'='*60}")
        print(f"Generating: {width}x{height}, {num_frames} frames, seed={seed}")
        print(f"  VRAM free: {free_vram:.2f} GB | RAM free: {ram.available / 1024**3:.1f} GB")
        print(f"  Audio: {'YES' if input_waveform is not None else 'NO'}")
        print(f"  Prompt: {prompt[:120]}")
        print(f"{'='*60}")
        sys.stdout.flush()

        vae_tile_size = 256

        total_steps = [8]
        current_step = [0]
        current_pass = [1]

        def cb(step, latent, is_start, override_num_inference_steps=None, pass_no=None, **kwargs):
            if is_start:
                if override_num_inference_steps is not None:
                    total_steps[0] = override_num_inference_steps
                if pass_no is not None:
                    current_pass[0] = pass_no
                current_step[0] = 0
                return
            current_step[0] += 1
            if current_pass[0] == 1:
                stage_name = "Stage 1 (half-res)"
            elif current_pass[0] == 2:
                stage_name = "Stage 2 (full-res refine)"
            else:
                stage_name = "Diffusion"
            free_v = torch.cuda.mem_get_info()[0] / 1024**3
            print(f"  [{stage_name}] step {current_step[0]}/{total_steps[0]} | VRAM free: {free_v:.2f} GB")
            sys.stdout.flush()
            frac = current_step[0] / max(total_steps[0], 1)
            if current_pass[0] == 2:
                frac = 0.73 + 0.22 * frac
            else:
                frac = frac * 0.73
            progress(min(frac, 0.95), desc=f"{stage_name}: {current_step[0]}/{total_steps[0]}")

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
            guide_scale=float(guide_scale),
            sampling_steps=8,
            guide_phases=2,
            n_prompt="",
            video_prompt_type="",
            audio_prompt_type="",
        )

        if input_waveform is not None:
            gen_kwargs["input_waveform"] = input_waveform
            gen_kwargs["input_waveform_sample_rate"] = input_waveform_sample_rate
            gen_kwargs["audio_prompt_type"] = "2"
            gen_kwargs["audio_scale"] = 1.0
            print("  Audio conditioning: ENABLED (type=2, soundtrack mode)")

        _stage_labels = {
            "VAE Encoding":  ("🎞️  VAE Encoding input frames...",       0.05),
            "VAE Decoding":  ("🎬 VAE Decoding latents → frames...",     0.88),
            "Upsampling":    ("🔭 Spatial upsampling latents...",         0.83),
        }

        import time as _time
        _t = [_time.time()]

        def _elapsed():
            now = _time.time()
            dt = now - _t[0]
            _t[0] = now
            return dt

        def set_progress_status(status: str):
            dt = _elapsed()
            label, frac = _stage_labels.get(status, (f"⏳ {status}...", 0.85))
            print(f"  [{status}] {label}  (+{dt:.1f}s since last event)")
            sys.stdout.flush()
            progress(frac, desc=label)

        gen_kwargs["set_progress_status"] = set_progress_status

        print("  Diffusion starting...")
        sys.stdout.flush()
        _t[0] = _time.time()

        result = ltx2_model.generate(**gen_kwargs)

        progress(0.97, desc="✅ Generation done — saving video...")
        print("  Pipeline complete.")
        sys.stdout.flush()

        if result is None:
            return None, "Generation failed or was interrupted."

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
            return None, f"❌ No video tensor. Got: {type(video_tensor)}"

        print(f"  Video tensor: {video_tensor.shape}, dtype={video_tensor.dtype}")
        video_tensor = video_tensor.cpu()
        gc.collect(); torch.cuda.empty_cache()

        out_path = tempfile.mktemp(suffix=".mp4")
        video_for_save = video_tensor.unsqueeze(0).float() / 127.5 - 1.0
        save_video(tensor=video_for_save, save_file=out_path, fps=frame_rate, normalize=True, value_range=(-1, 1))
        print(f"  ✅ Video saved: {out_path}")

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
                    out_path = final_path
                    print(f"  ✅ Audio muxed: {out_path}")
            except Exception as e:
                print(f"  ⚠️ Audio mux failed: {e}")

        del video_tensor, video_for_save
        gc.collect(); torch.cuda.empty_cache()
        progress(1.0, desc="Done!")
        return out_path, f"✅ Done! Seed: {seed} | {width}x{height} | {num_frames} frames"

    except Exception as e:
        traceback.print_exc()
        gc.collect(); torch.cuda.empty_cache()
        return None, f"❌ Error: {str(e)}"

CSS = """@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
* { font-family: 'Inter', sans-serif !important; }
.gradio-container { max-width: 1000px !important; margin: auto !important; }
.brand-header { text-align: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 28px; border-radius: 15px; margin-bottom: 20px; box-shadow: 0 10px 25px rgba(102,126,234,0.3); }
.brand-title { color: white; font-size: 2em; font-weight: 700; margin: 0 0 6px 0; }
.brand-subtitle { color: rgba(255,255,255,0.88); font-size: 1em; margin-bottom: 16px; }
.social-buttons { display: flex; justify-content: center; gap: 12px; flex-wrap: wrap; }
.social-btn { padding: 10px 24px; border-radius: 8px; font-weight: 700; font-size: 15px; text-decoration: none; display: inline-block; color: white; transition: all 0.3s; box-shadow: 0 4px 12px rgba(0,0,0,0.2); }
.social-btn:hover { transform: translateY(-2px); box-shadow: 0 6px 16px rgba(0,0,0,0.3); }
.youtube-btn { background: linear-gradient(135deg, #FF0000 0%, #CC0000 100%); }
.x-btn { background: linear-gradient(135deg, #000000 0%, #333333 100%); }
button.primary { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important; color: white !important; font-weight: 600 !important; border-radius: 12px !important; }
#stop-btn { background: linear-gradient(135deg, #ef4444 0%, #b91c1c 100%) !important; color: white !important; font-weight: 600 !important; border-radius: 12px !important; }
#clear-btn { background: linear-gradient(135deg, #6b7280 0%, #374151 100%) !important; color: white !important; font-weight: 600 !important; border-radius: 12px !important; }
.footer { text-align: center; padding: 20px; margin-top: 30px; border-top: 2px solid #e5e7eb; color: #6b7280; }
"""

with gr.Blocks(css=CSS, theme=gr.themes.Soft(), title="LTX-2.3 Audio-to-Video | AIQUEST") as demo:
    gr.HTML('<div class="brand-header"><div class="brand-title">🎤 LTX-2.3 22B — Audio-to-Video / Lip Sync</div><div class="brand-subtitle">Created by <strong>AIQuest Academy</strong> | Kaggle T4 GPU</div><div class="social-buttons"><a href="https://youtube.com/@aiquestacademy" target="_blank" class="social-btn youtube-btn">▶️ Subscribe on YouTube</a><a href="https://x.com/aiquestacademy" target="_blank" class="social-btn x-btn">𝕏 Follow on X</a></div></div>')

    gr.Markdown(
        "**Audio-driven lip sync pipeline** using distilled model (8 steps) + spatial 2x upscaler\n\n"
        "**Hint:** Use `[VISUAL] <scene> [SPEECH] <dialogue> [SOUND] <voice description>` prompt tags for best results"
    )

    with gr.Column():
        prompt = gr.Textbox(label="📝 Prompt", lines=3,
                   placeholder="[VISUAL] A beautiful woman talking to camera in a studio. [SPEECH] Hello everyone, welcome! [SOUND] Clear female voice, studio quality.")

        with gr.Row():
            input_image_start = gr.Image(type="filepath", label="🖼️ Reference Image (Start Frame)")
            input_audio = gr.Audio(type="filepath", label="🎤 Upload Speech/Song Audio")

        with gr.Row():
            seed = gr.Number(label="🎲 Seed (-1 for Random)", value=-1, precision=0)
            duration_dropdown = gr.Dropdown(
                label="⏱️ Duration (ignored when Match Audio is on)",
                choices=[
                    "2 Seconds (49 frames)",
                    "3 Seconds (73 frames)",
                    "5 Seconds (121 frames)",
                    "8 Seconds (193 frames)",
                    "10 Seconds (241 frames)",
                    "15 Seconds (361 frames)",
                    "20 Seconds (481 frames)",
                    "25 Seconds (601 frames)",
                    "30 Seconds (721 frames)",
                ],
                value="5 Seconds (121 frames)",
            )

        match_audio_dur = gr.Checkbox(
            label="🎵 Match video duration to audio length (auto-snap to nearest valid frame count)",
            value=False,
            info="When checked, the Duration dropdown above is ignored and the video will be as long as your audio (capped at 30s).",
        )

        def _toggle_dropdown(checked):
            return gr.update(interactive=not checked)

        match_audio_dur.change(
            fn=_toggle_dropdown,
            inputs=[match_audio_dur],
            outputs=[duration_dropdown],
        )

        with gr.Row():
            resolution_dropdown = gr.Dropdown(
                label="📐 Base Resolution",
                choices=["1080p", "720p", "540p", "480p"],
                value="720p",
            )
            aspect_ratio_dropdown = gr.Dropdown(
                label="📏 Aspect Ratio",
                choices=["16:9 Landscape", "4:3 Standard", "1:1 Square", "3:4 Portrait", "9:16 Portrait"],
                value="16:9 Landscape",
            )

        guide_scale = gr.Slider(
            label="🎯 Prompt Influence (guide_scale) — higher = text prompt stronger",
            minimum=1.0, maximum=8.0, step=0.5, value=4.0,
            info="4 = balanced | 6–8 = strong prompt | 1–2 = audio-only feel",
        )

        with gr.Row():
            gen_btn = gr.Button("🎬 Generate Video", variant="primary", size="lg", elem_id="gen-btn")
            stop_btn = gr.Button("🛑 Stop", variant="secondary", size="lg", elem_id="stop-btn")
            clear_btn = gr.Button("🗑️ Clear", variant="secondary", size="lg", elem_id="clear-btn")
        video_out = gr.Video(label="🎥 Output")
        status_out = gr.Textbox(label="ℹ️ Status", interactive=False)

        gen_event = gen_btn.click(
            fn=Video_Generation,
            inputs=[prompt, input_image_start, input_audio, seed, duration_dropdown,
                    resolution_dropdown, aspect_ratio_dropdown, guide_scale, match_audio_dur],
            outputs=[video_out, status_out],
        )
        stop_btn.click(fn=None, cancels=[gen_event])
        clear_btn.click(
            fn=lambda: (None, None, "", -1),
            outputs=[video_out, status_out, prompt, seed],
        )

gr.HTML('<div class="footer"><p style="font-size: 16px; margin: 5px 0;">🎤 Created by <strong>AIQuest Academy</strong></p><p style="font-size: 14px; margin: 5px 0; color: #9ca3af;">Free & Open Source | LTX-2.3 22B Distilled | Kaggle T4 GPU</p><p style="font-size: 13px; margin: 10px 0;"><a href="https://youtube.com/@aiquestacademy" target="_blank" style="color: #667eea; text-decoration: none; margin: 0 10px;">YouTube</a> | <a href="https://x.com/aiquestacademy" target="_blank" style="color: #667eea; text-decoration: none; margin: 0 10px;">X (Twitter)</a> | <a href="https://aiquest.site" target="_blank" style="color: #667eea; text-decoration: none; margin: 0 10px;">aiquest.site</a></p></div>')

print("\nLaunching Gradio...")
sys.stdout.flush()
demo.queue()
demo.launch(share=True, inline=False, debug=True, show_error=True, max_threads=1, ssr_mode=False)