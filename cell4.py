import os
print("Running cell4.py")
import gc, os, sys, json, random, tempfile, glob, traceback, time
import subprocess
from pathlib import Path
from datetime import datetime

import numpy as np
import psutil
import soundfile as sf
from PIL import Image

# ── Bootstrap Wan2GP ──────────────────────────────────
WAN2GP_DIR = os.path.abspath("Wan2GP")
sys.path.insert(0, WAN2GP_DIR)
os.chdir(WAN2GP_DIR)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.5"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

import torch
import gradio as gr
from shared.utils.audio_video import save_video

# ── GPU Info ──────────────────────────────────────────
_GPU_SM = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
print(f"  GPU: {torch.cuda.get_device_name()}")
print(f"  Compute: sm_{_GPU_SM[0]}{_GPU_SM[1]}")
print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
ram = psutil.virtual_memory()
print(f"  RAM: {ram.total / 1024**3:.1f} GB total, {ram.available / 1024**3:.1f} GB available")

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)

# ── GGUF Handler Registration ─────────────────────────
def _register_gguf_handler():
    import shared.qtypes.gguf
    print("  [GGUF] Handler registered")

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
        except Exception:
            pass
        if fallback_config_path and os.path.isfile(fallback_config_path):
            try:
                with open(fallback_config_path, "r", encoding="utf-8") as f:
                    config = ltx2_mod._normalize_config(json.load(f))
                    if config:
                        return config
            except Exception:
                pass
        return {}

    ltx2_mod._load_config_from_checkpoint = _patched

# ── Load Model ────────────────────────────────────────
print("\n  Loading LTX-2.3 22B Distilled 1.1 (GGUF Q4_K_M)...")

from mmgp import offload
from shared.utils import files_locator as fl
fl.set_checkpoints_paths(["models", "ckpts", "."])
from models.ltx2.ltx2_handler import family_handler

_register_gguf_handler()
_patch_ltx2_config_loading()

base_model_type = "ltx2_22B"
model_def = {"ltx2_pipeline": "distilled"}
extra = family_handler.query_model_def(base_model_type, model_def)
model_def.update(extra)

gemma_folder = "models/gemma-3-12b-it-qat-q4_0-unquantized"
gemma_files = sorted(glob.glob(os.path.join(gemma_folder, "*.safetensors")))
quanto_files = [f for f in gemma_files if "quanto" in f]
text_encoder_file = quanto_files[0] if quanto_files else (gemma_files[0] if gemma_files else None)
if not text_encoder_file:
    raise FileNotFoundError(f"No .safetensors in {gemma_folder}")

transformer_path = os.path.join("models", "ltx-2.3-22b-distilled-1.1-Q4_K_M.gguf")
if not os.path.isfile(transformer_path):
    raise FileNotFoundError(f"{transformer_path} missing")

MODEL_DTYPE = torch.float16
VAE_DTYPE = torch.float16

ltx2_model, pipe = family_handler.load_model(
    model_filename=transformer_path,
    model_type="ltx2_22B_distilled",
    base_model_type=base_model_type,
    model_def=model_def,
    dtype=MODEL_DTYPE,
    VAE_dtype=VAE_DTYPE,
    text_encoder_filename=text_encoder_file,
)

# ── mmgp Profile 4 ────────────────────────────────────
print("\n  Applying mmgp Profile 4...")
offload.profile(
    pipe,
    profile_no=4,
    quantizeTransformer=False,
    convertWeightsFloatTo=torch.float16,
    budgets={
        "transformer": 6000, "text_encoder": 1500,
        "video_encoder": 2000, "video_decoder": 3000,
        "audio_encoder": 1000, "audio_decoder": 1000,
        "vocoder": 500, "spatial_upsampler": 1500,
        "vae": 1000, "*": 1000,
    },
)
offload.shared_state["_attention"] = "sdpa"
print("  ✓ Model ready")

# ── Gallery ───────────────────────────────────────────
OUTPUT_DIR = Path('/kaggle/working/generated_videos')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
GALLERY_FILE = OUTPUT_DIR / 'gallery.json'

def load_gallery():
    if GALLERY_FILE.exists():
        return json.loads(GALLERY_FILE.read_text())
    return []

def save_gallery_entry(video_path, prompt, mode, params):
    gallery = load_gallery()
    gallery.append({
        'video': str(video_path), 'prompt': prompt,
        'mode': mode, 'params': params,
        'timestamp': datetime.now().isoformat()
    })
    if len(gallery) > 20:
        for old in gallery[:-20]:
            p = Path(old['video'])
            if p.exists(): p.unlink()
        gallery = gallery[-20:]
    GALLERY_FILE.write_text(json.dumps(gallery, indent=2))

# ── Resolution Helper ─────────────────────────────────
def get_resolution(base_res, aspect_ratio):
    bases = {"1080p": 1088, "720p": 704, "540p": 544, "480p": 480}
    ratios = {"16:9": 16/9, "4:3": 4/3, "1:1": 1.0, "3:4": 3/4, "9:16": 9/16}
    base = bases.get(base_res, 704)
    ratio = ratios.get(aspect_ratio, 16/9)
    if ratio >= 1.0:
        h, w = base, int(base * ratio)
    else:
        w, h = base, int(base / ratio)
    return (w // 32) * 32, (h // 32) * 32

# ── Generation ────────────────────────────────────────
@torch.inference_mode()
def generate_video(prompt, input_image_start, input_image_end, seed,
                   duration, resolution, aspect_ratio,
                   guide_scale, num_steps, progress=gr.Progress()):
    try:
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
        progress(0, desc="Starting...")

        duration_map = {
            "2s (49f)": 49, "3s (73f)": 73, "5s (121f)": 121,
            "8s (193f)": 193, "10s (241f)": 241, "15s (361f)": 361,
        }
        num_frames = duration_map.get(duration, 121)
        width, height = get_resolution(resolution, aspect_ratio)

        if seed is None or seed < 0:
            seed = random.randint(0, 2**32 - 1)
        seed = int(seed)

        image_start = None; image_end = None
        if input_image_start is not None:
            image_start = Image.open(input_image_start).convert("RGB")
        if input_image_end is not None:
            image_end = Image.open(input_image_end).convert("RGB")

        mode = "T2V" if image_start is None else ("I2V" if image_end is None else "I2V first+last")
        print(f"\n{'='*50}")
        print(f"  [{mode}] {width}x{height}, {num_frames}f, seed={seed}")
        print(f"  Prompt: {prompt[:100]}...")
        print(f"{'='*50}")

        total_steps = [8]; current_step = [0]; current_pass = [1]

        def cb(step, latent, is_start, override_num_inference_steps=None, pass_no=None, **kwargs):
            if is_start:
                if override_num_inference_steps is not None:
                    total_steps[0] = override_num_inference_steps
                if pass_no is not None:
                    current_pass[0] = pass_no
                current_step[0] = 0
                return
            current_step[0] += 1
            stage = "Stage 1" if current_pass[0] == 1 else "Stage 2"
            frac = current_step[0] / max(total_steps[0], 1)
            if current_pass[0] == 2:
                frac = 0.73 + 0.22 * frac
            else:
                frac = frac * 0.73
            progress(min(frac, 0.95), desc=f"{stage}: {current_step[0]}/{total_steps[0]}")

        def set_progress_status(status: str):
            labels = {
                "VAE Encoding": (0.05, "Encoding..."),
                "VAE Decoding": (0.88, "Decoding..."),
                "Upsampling": (0.83, "Upsampling..."),
            }
            frac, desc = labels.get(status, (0.85, status))
            progress(frac, desc=desc)

        gen_kwargs = dict(
            input_prompt=prompt, image_start=image_start,
            height=height, width=width, frame_num=num_frames,
            fps=24.0, seed=seed, callback=cb,
            VAE_tile_size=256, input_video_strength=1.0,
            denoising_strength=1.0, guide_scale=float(guide_scale),
            sampling_steps=int(num_steps), guide_phases=2,
            n_prompt="", enhance_prompt=False,
            video_prompt_type="", audio_prompt_type="",
            set_progress_status=set_progress_status,
        )
        if image_end is not None:
            gen_kwargs["image_end"] = image_end

        result = ltx2_model.generate(**gen_kwargs)
        progress(0.97, desc="Saving video...")

        if isinstance(result, dict):
            video_tensor = result.get("x")
        elif isinstance(result, tuple):
            video_tensor = result[0]
        else:
            video_tensor = result

        if video_tensor is None:
            return None, "Error: No video generated"

        video_tensor = video_tensor.cpu()
        gc.collect(); torch.cuda.empty_cache()

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_path = str(OUTPUT_DIR / f'video_{timestamp}.mp4')

        video_for_save = video_tensor.unsqueeze(0).float() / 127.5 - 1.0
        save_video(tensor=video_for_save, save_file=out_path, fps=24.0,
                   normalize=True, value_range=(-1, 1))

        params = {'steps': num_steps, 'guidance': guide_scale,
                  'resolution': f'{width}x{height}', 'frames': num_frames}
        save_gallery_entry(out_path, prompt, mode, params)

        del video_tensor, video_for_save
        gc.collect(); torch.cuda.empty_cache()
        progress(1.0, desc="Done!")
        return out_path, f"✓ Done! Seed: {seed} | {width}x{height} | {num_frames}f"

    except Exception as e:
        traceback.print_exc()
        gc.collect(); torch.cuda.empty_cache()
        return None, f"Error: {str(e)}"

# ── Audio Mixing ──────────────────────────────────────
def mix_audio_video(video_path, audio_path, volume=0.5):
    if not audio_path or not video_path:
        return video_path
    out_path = str(Path(video_path).with_suffix('.mixed.mp4'))
    cmd = ['ffmpeg', '-y', '-i', video_path, '-i', audio_path,
           '-c:v', 'copy', '-c:a', 'aac', '-af', f'volume={volume}',
           '-shortest', out_path]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return video_path
    return out_path

# ── Gradio UI ─────────────────────────────────────────
CSS = """
.gradio-container { max-width: 1100px !important; margin: auto; }
.gen-btn { background: linear-gradient(135deg, #6a11cb, #2575fc) !important; color: white !important; font-weight: 600 !important; }
footer { display: none !important; }
.brand-header { text-align: center; padding: 20px 15px; margin-bottom: 15px; background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%); border-radius: 15px; }
.brand-title { font-size: 1.6em; font-weight: 700; color: #ffffff; margin-bottom: 5px; }
.brand-subtitle { font-size: 0.95em; color: #c0c0ff; margin-bottom: 12px; }
.social-buttons { display: flex; gap: 10px; justify-content: center; flex-wrap: wrap; }
.social-btn { padding: 8px 18px; border-radius: 25px; text-decoration: none; color: white; font-weight: 600; font-size: 0.85em; transition: transform 0.2s; display: inline-block; }
.social-btn:hover { transform: scale(1.05); }
.youtube-btn { background: linear-gradient(135deg, #FF0000 0%, #CC0000 100%); }
.instagram-btn { background: linear-gradient(135deg, #833AB4 0%, #FD1D1D 50%, #F77737 100%); }
.tiktok-btn { background: linear-gradient(135deg, #000000 0%, #333333 100%); }
.footer { text-align: center; padding: 20px; margin-top: 30px; border-top: 2px solid #e5e7eb; color: #6b7280; }
.footer a { color: #667eea; text-decoration: none; margin: 0 8px; }
"""

with gr.Blocks(css=CSS, theme=gr.themes.Soft(), title="LTX-Video Generator") as demo:
    gr.HTML('''<div class="brand-header"><div class="brand-title">🎬 LTX-2.3 22B Distilled 1.1 Q4 — Video Generator</div><div class="brand-subtitle">Created by <strong>TheBuildAI</strong> | T4 GPU</div><div class="social-buttons"><a href="https://youtube.com/@thebuildai" target="_blank" class="social-btn youtube-btn">▶️ Subscribe on YouTube</a><a href="https://instagram.com/thebuildai" target="_blank" class="social-btn instagram-btn">📷 Follow on Instagram</a><a href="https://tiktok.com/@the.build.ai" target="_blank" class="social-btn tiktok-btn">🎵 Follow on TikTok</a></div></div>''')

    with gr.Tabs():
        with gr.Tab('🎬 Generate'):
            prompt = gr.Textbox(
                label='Prompt', lines=3,
                placeholder='A cinematic shot of a cat walking through a neon-lit cyberpunk city at night...'
            )

            with gr.Accordion('🖼️ Image-to-Video (Optional)', open=False):
                with gr.Row():
                    input_image_start = gr.Image(type='filepath', label='Start Frame')
                    input_image_end = gr.Image(type='filepath', label='End Frame (optional)')

            with gr.Row():
                seed = gr.Number(label='Seed (-1 = random)', value=-1, precision=0)
                duration = gr.Dropdown(
                    label='Duration',
                    choices=['2s (49f)', '3s (73f)', '5s (121f)', '8s (193f)', '10s (241f)', '15s (361f)'],
                    value='5s (121f)'
                )

            with gr.Row():
                resolution = gr.Dropdown(
                    label='Resolution', choices=['1080p', '720p', '540p', '480p'], value='720p'
                )
                aspect_ratio = gr.Dropdown(
                    label='Aspect Ratio', choices=['16:9', '4:3', '1:1', '3:4', '9:16'], value='16:9'
                )

            with gr.Row():
                guide_scale = gr.Slider(1.0, 8.0, value=3.0, step=0.5, label='Guidance Scale')
                num_steps = gr.Slider(2, 8, value=8, step=1, label='Diffusion Steps')

            audio_file = gr.Audio(label='Background Audio (optional)', type='filepath')
            audio_volume = gr.Slider(0.1, 2.0, value=0.5, step=0.1, label='Audio Volume')

            with gr.Row():
                gen_btn = gr.Button('🎬 Generate Video', variant='primary', elem_classes='gen-btn')
                stop_btn = gr.Button('🛑 Stop')

            video_out = gr.Video(label='Output')
            status_out = gr.Textbox(label='Status', interactive=False)

            gen_event = gen_btn.click(
                fn=generate_video,
                inputs=[prompt, input_image_start, input_image_end, seed,
                        duration, resolution, aspect_ratio, guide_scale, num_steps],
                outputs=[video_out, status_out]
            ).then(
                fn=mix_audio_video,
                inputs=[video_out, audio_file, audio_volume],
                outputs=video_out
            )
            stop_btn.click(fn=None, cancels=[gen_event])

        with gr.Tab('🖼️ Gallery'):
            gr.Markdown('### Generation History (last 20)')

            def refresh_gallery():
                gallery = load_gallery()
                rows = []
                for g in reversed(gallery):
                    rows.append([
                        g['timestamp'][:19], g['mode'],
                        g['prompt'][:80] + ('...' if len(g['prompt']) > 80 else ''),
                        json.dumps(g['params']), g['video']
                    ])
                return rows

            gallery_list = gr.Dataframe(
                headers=['Time', 'Mode', 'Prompt', 'Params', 'Video'],
                label='History', interactive=False
            )
            gr.Button('🔄 Refresh').click(fn=refresh_gallery, outputs=gallery_list)

    gr.HTML('''<div class="footer"><p style="font-size: 16px; margin: 5px 0;">🎬 Created by <strong>TheBuildAI</strong></p><p style="font-size: 14px; margin: 5px 0; color: #9ca3af;">Free &amp; Open Source | LTX-2.3 22B Distilled 1.1 Q4_K_M | T4 GPU</p><p style="font-size: 13px; margin: 10px 0;"><a href="https://youtube.com/@thebuildai" target="_blank">YouTube</a> | <a href="https://instagram.com/thebuildai" target="_blank">Instagram</a> | <a href="https://tiktok.com/@the.build.ai" target="_blank">TikTok</a> | <a href="https://thebuildai.tech" target="_blank">thebuildai.tech</a></p></div>''')

print("\n  Launching Gradio...")
demo.queue(max_size=5)
demo.launch(share=True, debug=False, allowed_paths=['/kaggle/working/generated_videos'])