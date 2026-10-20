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

print("Running run_scenea.py")

# ---- bootstrap Wan2GP ----
WAN2GP_DIR = os.path.abspath("Wan2GP")
# Audio is saved in the working folder (/kaggle/working/outputs), visible in Kaggle's Output panel
OUTPUT_DIR = os.path.join(os.path.dirname(WAN2GP_DIR), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.environ.setdefault("GRADIO_TEMP_DIR", os.path.join(OUTPUT_DIR, ".gradio_cache"))
sys.path.insert(0, WAN2GP_DIR)
os.chdir(WAN2GP_DIR)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.5"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

import torch
import torchaudio

# ---- Patch torchaudio to bypass torchcodec loading issues ----
def patched_torchaudio_load(filepath, frame_offset=0, num_frames=-1, normalize=True, channels_first=True, format=None, buffer_size=4096):
    data, samplerate = sf.read(filepath, dtype='float32')
    tensor = torch.from_numpy(data)
    if channels_first:
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        else:
            tensor = tensor.T
    return tensor, samplerate

def patched_torchaudio_save(filepath, src, sample_rate, channels_first=True, format=None, encoding=None, bits_per_sample=None, buffer_size=4096):
    if torch.is_tensor(src):
        src_np = src.detach().cpu().numpy()
    else:
        src_np = src
    if channels_first and src_np.ndim == 2:
        src_np = src_np.T
    sf.write(filepath, src_np, int(sample_rate))

torchaudio.load = patched_torchaudio_load
torchaudio.save = patched_torchaudio_save
print("✅ Patched torchaudio load/save to use soundfile backend (bypassing torchcodec)")
sys.stdout.flush()

import gradio as gr

# ==== GPU INFO ====
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB" if torch.cuda.is_available() else "VRAM: N/A")
ram = psutil.virtual_memory()
print(f"RAM: {ram.total / 1024**3:.1f} GB total, {ram.available / 1024**3:.1f} GB available")
sys.stdout.flush()

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)

# ==== LOAD MODEL ====
print("\nLoading Scenema Audio Model...")
sys.stdout.flush()

from mmgp import offload
from shared.utils import files_locator as fl

# ==== INT8 Tensor Core kernels (Wan2GP's "INT8 Kernels: Auto") ====
# The Scenema transformer is quanto int8: without this every int8 layer is dequantized to BF16 and
# run as a BF16 matmul, which a T4 can only do on its slow CUDA cores. Set to False to compare.
USE_INT8_KERNELS = True
NUM_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 0
int8_backend = None
if USE_INT8_KERNELS:
    try:
        from shared.kernels import int8_backend
        int8_backend.configure("auto", 1)
    except Exception as e:
        int8_backend = None
        print(f"⚠️ INT8 kernels unavailable, using PyTorch matmul: {e}")

from models.ltx2.ltx_audio_tts_handler import family_handler as audio_family_handler

fl.set_checkpoints_paths(["/kaggle/tmp/models", "ckpts", "."])

gemma_folder = "models/gemma-3-12b-it-qat-q4_0-unquantized"
gemma_files = sorted(glob.glob(os.path.join(gemma_folder, "*.safetensors")))
quanto_files = [f for f in gemma_files if "quanto" in f]
text_encoder_file = quanto_files[0] if quanto_files else (gemma_files[0] if gemma_files else None)
if not text_encoder_file:
    raise FileNotFoundError(f"No .safetensors text encoder weight found in {gemma_folder}.")

transformer_path = os.path.join("models", "scenema-audio-transformer_quanto_bf16_int8.safetensors")
if not os.path.isfile(transformer_path):
    raise FileNotFoundError(f"Transformer model path {transformer_path} missing.")

base_model_type = "scenema_audio"
model_def = audio_family_handler.query_model_def(base_model_type, {})

MODEL_DTYPE = torch.bfloat16
VAE_DTYPE   = torch.float16

pipeline, pipe = audio_family_handler.load_model(
    model_filename=transformer_path,
    model_type="scenema_audio",
    base_model_type=base_model_type,
    model_def=model_def,
    dtype=MODEL_DTYPE,
    VAE_dtype=VAE_DTYPE,
    text_encoder_filename=text_encoder_file,
)

# Extract pipe components and coTenantsMap
pipe_dict = pipe["pipe"]
co_tenants = dict(pipe["coTenantsMap"])

# ==== Gemma 3 text encoder resident on GPU 1 (Kaggle GPU T4 x2) ====
# Gemma is 12B: under mmgp it streams ~13 GB over PCIe for every prompt chunk and evicts every other
# model from GPU 0. On the second T4 it stays loaded. The 1.9 GB token-embedding table runs on CPU
# (a lookup of a few hundred tokens) and lm_head is dropped (Scenema only reads hidden states).
TEXT_ENCODER_ON_GPU1 = False
if NUM_GPUS > 1:
    _te = pipeline.text_encoder
    _gemma = _te.model
    _gemma_lm = _gemma.model if hasattr(_gemma, "model") else _gemma
    try:
        print("\n📦 Moving Gemma 3 text encoder to GPU 1 (second T4)...")
        sys.stdout.flush()
        _te.to("cuda:1")
        if getattr(_gemma, "lm_head", None) is not None:
            _gemma.lm_head = torch.nn.Identity()
        _embed = _gemma_lm.embed_tokens
        _embed.to("cpu")
        _orig_embed_forward = _embed.forward

        def _cpu_embed_forward(input_ids, *args, **kwargs):
            return _orig_embed_forward(input_ids.to("cpu"), *args, **kwargs).to("cuda:1")

        _embed.forward = _cpu_embed_forward
        # HF's model.device reports the first parameter (now the CPU embedding): pin it to GPU 1
        _gemma.__class__ = type(_gemma.__class__.__name__, (_gemma.__class__,),
                                {"device": property(lambda self: torch.device("cuda", 1))})
        with torch.cuda.device(1):
            torch.cuda.empty_cache()
        pipe_dict.pop("text_encoder", None)
        TEXT_ENCODER_ON_GPU1 = True
        print(f"✅ Gemma 3 resident on GPU 1 ({torch.cuda.mem_get_info(1)[0] / 1024**3:.1f} GB free for encoding).")
    except Exception as e:
        print(f"⚠️ Gemma stays on GPU 0 with offloading ({e})")
        if type(_gemma).__dict__.get("device") is not None:
            _gemma.__class__ = type(_gemma).__bases__[0]
        _gemma_lm.embed_tokens.__dict__.pop("forward", None)
        _te.to("cpu")
        with torch.cuda.device(1):
            torch.cuda.empty_cache()
    sys.stdout.flush()

# ==== Exact text encoding ====
# Gemma has extreme activation outlier channels: the INT8 kernels quantize each activation row to
# int8, which would flatten its normal channels. Text encoding (cached per prompt) uses the exact
# int8-weight path; the Scenema transformer keeps the fast kernels. Hidden states go back to GPU 0
# with their dtype unchanged (never FP16: the RMS norm that follows squares them).
import models.ltx2.ltx_audio_tts as _tts_mod
import models.ltx2.scenema_audio as _scenema_mod
try:
    from optimum.quanto.tensor.weights import qbytes as _qbytes
except Exception:
    _qbytes = None
_orig_encode_text = _tts_mod.encode_text
_main_device = torch.device("cuda", 0)

def _exact_encode_text(text_encoder, prompts):
    saved_forward = None
    if _qbytes is not None and int8_backend is not None and getattr(int8_backend, "_original_forward", None) is not None:
        saved_forward = _qbytes.WeightQBytesLinearFunction.__dict__["forward"]
        _qbytes.WeightQBytesLinearFunction.forward = staticmethod(int8_backend._original_forward)
    try:
        with torch.cuda.device(1 if TEXT_ENCODER_ON_GPU1 else 0):
            out = _orig_encode_text(text_encoder, prompts)
    finally:
        if saved_forward is not None:
            _qbytes.WeightQBytesLinearFunction.forward = saved_forward
    return [r._replace(hidden_states=tuple(h.to(_main_device) for h in r.hidden_states),
                       attention_mask=r.attention_mask.to(_main_device)) for r in out]

_tts_mod.encode_text = _exact_encode_text
_scenema_mod.encode_text = _exact_encode_text

# ==== Keep every other model resident together on GPU 0 ====
# mmgp unloads all models whenever it loads one that is not a declared co-tenant, so each prompt
# chunk used to shuffle Gemma, the transformer, Whisper, Kokoro and SeedVC in and out of VRAM.
# Without Gemma the rest fits together (~10 GB), so they are all co-tenants of each other.
_resident = [k for k in pipe_dict if k != "text_encoder"]
for _k in _resident:
    co_tenants[_k] = sorted(set(co_tenants.get(_k, [])) | (set(_resident) - {_k}))

_budgets = {
    "transformer": 4000,                 # 3.35 GB int8: fully resident
    "text_embedding_projection": 2500,   # fully resident
    "text_embeddings_connector": 1000,
    "audio_encoder": 1000,
    "audio_decoder": 1000,
    "vocoder": 500,
    "alignment_whisper": 2000,
    "kokoro": 1000,
    "seedvc_whisper_small": 1000,
    "seedvc_campplus": 500,
    "seedvc_bigvgan": 500,
    "*": 1000,
}
if not TEXT_ENCODER_ON_GPU1:
    _budgets["text_encoder"] = 7000

# ==== Apply mmgp Profile 4 ====
print("\nApplying mmgp Profile 4 memory offloading...")
sys.stdout.flush()

offload.profile(
    pipe_dict,
    profile_no=4,
    quantizeTransformer=False,
    convertWeightsFloatTo=torch.bfloat16,
    coTenantsMap=co_tenants,
    pinnedMemory=True,
    budgets=_budgets,
)
offload.shared_state["_attention"] = "sdpa"

print("\n✅ Setup complete! Scenema Audio pipeline active.")
sys.stdout.flush()

# ==== GENERATION FUNCTION ====
@torch.inference_mode()
def Audio_Generation(prompt, voice_instruction, ref_mode, ref_audio1, ref_audio2,
                     seed, duration_seconds, pace, vc_steps, vc_cfg_rate,
                     progress=gr.Progress()):
    try:
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
        progress(0, desc="Starting...")

        if seed is None or seed < 0:
            seed = random.randint(0, 2**32 - 1)
        seed = int(seed)

        # Map Reference Mode to audio_prompt_type
        if ref_mode == "Speaker 1 reference using SeedVC":
            audio_prompt_type = "A2"
        elif ref_mode == "Two Speakers references using SeedVC":
            audio_prompt_type = "AB2"
        else:
            audio_prompt_type = ""

        # Print debug parameters
        print(f"\n{'='*60}")
        print(f"Generating [Scenema Audio]: seed={seed}, duration={duration_seconds}s, pace={pace}")
        print(f"  Prompt: {prompt[:120]}{'...' if len(prompt) > 120 else ''}")
        print(f"  Voice Instruction: {voice_instruction}")
        print(f"  Reference Mode: {ref_mode} ({audio_prompt_type})")
        print(f"  SeedVC: steps={vc_steps}, cfg={vc_cfg_rate}")
        print(f"{'='*60}")
        sys.stdout.flush()

        # Define callback and progress updater
        def cb(step, latent, is_start, override_num_inference_steps=None, pass_no=None, **kwargs):
            nonlocal current_step, total_steps
            if is_start:
                if override_num_inference_steps is not None:
                     total_steps = override_num_inference_steps
                current_step = 0
                return
            current_step += 1
            frac = current_step / max(total_steps, 1)
            progress(min(frac * 0.85, 0.95), desc=f"Diffusion step {current_step}/{total_steps}")

        current_step = 0
        total_steps = 8

        def set_progress_status(status: str):
            print(f"  [Status] {status}")
            sys.stdout.flush()
            progress(0.9, desc=f"{status}...")

        # Prepare generator kwargs
        gen_kwargs = dict(
            input_prompt=prompt,
            alt_prompt=voice_instruction,
            audio_prompt_type=audio_prompt_type,
            audio_guide=ref_audio1,
            audio_guide2=ref_audio2,
            seed=seed,
            duration_seconds=float(duration_seconds),
            custom_settings={
                "pace": float(pace),
                "vc_steps": int(vc_steps),
                "vc_cfg_rate": float(vc_cfg_rate),
            },
            callback=cb,
            set_progress_status=set_progress_status,
        )

        progress(0.05, desc="Generating audio chunks...")
        t_start = time.time()
        result = pipeline.generate(**gen_kwargs)
        gen_time = time.time() - t_start
        print(f"  ⏱️ Generation time: {gen_time:.1f}s")

        if result is None:
            return None, "❌ Generation failed. Model returned None."

        audio_data = result.get("x")
        audio_sr = result.get("audio_sampling_rate", 24000)

        if audio_data is None:
            return None, "❌ No audio data generated."

        # Save audio to the outputs folder
        out_path = os.path.join(OUTPUT_DIR, f"scenema_{time.strftime('%Y%m%d_%H%M%S')}_seed{seed}.wav")
        if torch.is_tensor(audio_data):
            audio_np = audio_data.cpu().numpy()
        else:
            audio_np = audio_data
        
        # Audio is usually shape (channels, samples)
        if audio_np.ndim == 2:
            if audio_np.shape[0] <= 2:
                # soundfile expects (samples, channels)
                audio_np = audio_np.T
        sf.write(out_path, audio_np, int(audio_sr))

        print(f"  ✅ Audio saved: {out_path}")
        sys.stdout.flush()

        progress(1.0, desc="Done!")
        return out_path, f"✅ Done in {gen_time:.1f}s! Seed: {seed} | Max duration: {duration_seconds}s | Pace: {pace}"

    except Exception as e:
        traceback.print_exc()
        gc.collect(); torch.cuda.empty_cache()
        return None, f"❌ Error: {str(e)}"

# ==== GRADIO UI (AIQUEST BRANDED) ====
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
  <div class="brand-title">🎵 Scenema Audio Expressive Speech Generator</div>
  <div class="brand-subtitle">Created by <strong>AIQuest Academy</strong> &nbsp;|&nbsp; Kaggle GPU T4 x2 Edition</div>
  <div class="btn-row">
    <a href="https://www.youtube.com/@aiquestacademy?sub_confirmation=1" target="_blank" class="social-btn yt-btn">▶ Subscribe</a>
    <a href="https://x.com/aiquestacademy" target="_blank" class="social-btn x-btn">𝕏 Follow on X</a>
    <a href="https://aiquest.site" target="_blank" class="social-btn sup-btn">❤️ Support My Work</a>
  </div>
</div>
"""

_int8_label = {"kitchen": "Comfy Kitchen INT8 kernels", "triton": "Triton INT8 kernels"}.get(
    getattr(int8_backend, "_backend", "pytorch") if int8_backend is not None else "pytorch", "PyTorch INT8")
STATUS_HTML = (f'<div class="status-line">⚡ {_int8_label} &nbsp;|&nbsp; '
               f'{"Gemma on GPU 1" if TEXT_ENCODER_ON_GPU1 else "single GPU"} &nbsp;|&nbsp; models resident on GPU 0</div>')

FOOTER_HTML = """
<div class="brand-footer">⚡ Made with ❤️ by <strong>AIQUEST Academy</strong> · aiquest.site · © All rights reserved</div>
"""

with gr.Blocks(theme=gr.themes.Soft(), css=CSS, title="Scenema Audio Expressive Speech Generator | AIQUEST Academy") as demo:
    gr.HTML(BRAND_HTML)
    gr.HTML(STATUS_HTML)

    gr.Markdown(
        "💡 **Tip:** Use square brackets like `[Sigh]` or `[Soft, close to the microphone]` to inject performance and emotion cues. "
        "You can also define multi-speaker scripts using `Speaker 1:` and `Speaker 2:` blocks."
    )

    with gr.Column():
        prompt = gr.Textbox(
            label="💬 Speech / Dialogue Prompt (or Scenema XML)", lines=6,
            value='Speaker 1{voice="Confident adult man, skeptical and emphatic", gender="male", scene="a spirited debate beside a map"}: [Pointing at the map with absolute confidence] Look at this map, it\'s a perfect circle with the North Pole right in the middle.\nSpeaker 2{voice="Patient adult woman, calm and logical", gender="female", scene="a spirited debate beside a map"}: [Calmly correcting, measured and patient] That\'s just a projection, not how the world actually looks from space.',
            placeholder="[Excited, speaking fast] Oh my god, you won't believe what just happened!..."
        )
        voice_instruction = gr.Textbox(
            label="🎙️ Voice Instruction / Override (Optional)", lines=2,
            placeholder="Confident adult man, skeptical and emphatic..."
        )

        with gr.Accordion("👥 Voice Cloning & References (SeedVC)", open=False):
            ref_mode = gr.Radio(
                label="Voice Cloning Mode",
                choices=["None", "Speaker 1 reference using SeedVC", "Two Speakers references using SeedVC"],
                value="None"
            )
            with gr.Row():
                ref_audio1 = gr.Audio(type="filepath", label="Speaker 1 Reference Audio")
                ref_audio2 = gr.Audio(type="filepath", label="Speaker 2 Reference Audio (Optional)")

        with gr.Row():
            seed = gr.Number(label="🎲 Seed (-1 for Random)", value=-1, precision=0)
            duration_seconds = gr.Slider(label="⏱️ Max Duration (seconds)", minimum=5.0, maximum=300.0, step=5.0, value=120.0)
            pace = gr.Slider(label="🏃 Pace (Speed Multiplier)", minimum=0.2, maximum=3.0, step=0.1, value=1.5)

        with gr.Row():
            vc_steps = gr.Slider(label="⚡ SeedVC Inference Steps", minimum=5, maximum=50, step=1, value=25)
            vc_cfg_rate = gr.Slider(label="🎯 SeedVC CFG Rate", minimum=0.1, maximum=1.0, step=0.05, value=0.5)

        with gr.Row():
            gen_btn   = gr.Button("🎵 Generate Audio", variant="primary", size="lg", elem_id="gen-btn")
            stop_btn  = gr.Button("🛑 Stop",            variant="secondary", size="lg", elem_id="stop-btn")
            clear_btn = gr.Button("🗑️ Clear",           variant="secondary", size="lg", elem_id="clear-btn")
        
        audio_out  = gr.Audio(label="🔊 Generated Speech Output")
        latest_btn = gr.Button("📂 Load Latest Audio (use if the UI showed an error)", variant="secondary")
        status_out = gr.Textbox(label="ℹ️ Status", interactive=False)

        def load_latest_audio():
            files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "*.wav")), key=os.path.getmtime)
            if not files:
                return None, f"No audio in {OUTPUT_DIR} yet."
            return files[-1], f"📁 Loaded latest audio: {files[-1]}"

        latest_btn.click(fn=load_latest_audio, outputs=[audio_out, status_out])

        gen_event = gen_btn.click(
            fn=Audio_Generation,
            inputs=[prompt, voice_instruction, ref_mode, ref_audio1, ref_audio2,
                    seed, duration_seconds, pace, vc_steps, vc_cfg_rate],
            outputs=[audio_out, status_out],
        )
        stop_btn.click(fn=None, cancels=[gen_event])
        clear_btn.click(
            fn=lambda: ("", "", "None", None, None, -1, 120.0, 1.5, 25, 0.5, None, ""),
            outputs=[prompt, voice_instruction, ref_mode, ref_audio1, ref_audio2,
                     seed, duration_seconds, pace, vc_steps, vc_cfg_rate, audio_out, status_out],
        )

    gr.HTML(FOOTER_HTML)

# ==== CLOUDFLARE QUICK TUNNEL (second public link in case the Gradio share link fails) ====
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

print("\nLaunching Gradio...")
sys.stdout.flush()
start_cloudflare_tunnel(SERVER_PORT)
demo.queue()
demo.launch(server_name="127.0.0.1", server_port=SERVER_PORT, share=True, inline=False, debug=True,
            show_error=True, max_threads=1, ssr_mode=False, allowed_paths=[OUTPUT_DIR])