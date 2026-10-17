"""LTX-2.5 22B Distilled video + audio engine for Kaggle GPU T4 x2 - AIQUEST Academy.

Step 2 of the notebook writes this file as run_ltx25.py and runs it. It drives Wan2GP's LTX-2.5 code directly:

* Wan2GP is pinned (Step 1) to the commit this engine was built on. Later Wan2GP versions read settings from the
  full Wan2GP UI module (wgp), which a standalone engine does not load.
* Transformer: the official 22B distilled checkpoint in INT8 ConvRot. Every INT8 linear runs on Comfy Kitchen's INT8
  tensor-core GEMM (W8A8 with the ConvRot Hadamard rotation) instead of being dequantized to FP16 at every step.
* GPU 0: the transformer, streamed by MMGP from pinned RAM (about half of it stays in VRAM), then the VAEs.
* GPU 1: the Gemma-4 12B text encoder, resident for the whole session, so a new prompt costs seconds instead of
  re-reading 13 GB. It runs on the exact INT8 path (LLM activations have outliers) and on the prompt's real tokens
  only: LTX pads every prompt to 1024 tokens, and the padding never reaches the video model.
* One 8-step distilled pass at the requested resolution in FP16 (Turing has FP16 tensor cores, not BF16 ones).
"""
import faulthandler
import gc
import glob
import os
import random
import re
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
import warnings
from contextlib import contextmanager

warnings.filterwarnings("ignore")
faulthandler.enable(file=sys.__stderr__, all_threads=True)  # a hard crash (segfault, abort) prints where it happened
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

WAN2GP_DIR = os.path.abspath("Wan2GP")
WORK_DIR = os.path.dirname(WAN2GP_DIR)
OUTPUT_DIR = os.path.join(WORK_DIR, "outputs")
MODELS_DIR = os.environ.get("LTX25_MODELS_DIR", "/kaggle/tmp/ltx25_models")
GEMMA_DIR = os.path.join(MODELS_DIR, "gemma4-12b-ltx-v1")
TRANSFORMER_FILE = os.path.join(MODELS_DIR, "ltx-2.5-22b-distilled_diffusion_model_int8_convrot.safetensors")
GEMMA_FILE = os.path.join(GEMMA_DIR, "gemma4-12b-ltx-v1_int8_convrot.safetensors")
SERVER_PORT = 7860
FPS = 24
if not os.path.isdir(WAN2GP_DIR):
    raise SystemExit("❌ Wan2GP is missing: run Step 1 first.")
_missing = [p for p in (TRANSFORMER_FILE, GEMMA_FILE) if not os.path.exists(p)]
if _missing:
    raise SystemExit(f"❌ Model files missing ({', '.join(map(os.path.basename, _missing))}): run Step 1 first.")
os.makedirs(OUTPUT_DIR, exist_ok=True)
if WAN2GP_DIR not in sys.path:
    sys.path.insert(0, WAN2GP_DIR)
os.chdir(WAN2GP_DIR)

import numpy as np
import psutil
import torch
from PIL import Image

import gradio as gr


def _gb(n):
    return f"{n / 1024**3:.1f} GB"


def _ram_in_use():
    """total - available: unlike psutil's 'used', this counts pinned (page-locked) memory, which Linux files as shared."""
    vm = psutil.virtual_memory()
    return f"{_gb(vm.total - vm.available)} / {_gb(vm.total)}"


def _fmt_secs(seconds):
    seconds = int(round(seconds))
    return f"{seconds // 60}m{seconds % 60:02d}s" if seconds >= 60 else f"{seconds}s"


class ConsoleBar:
    """One live progress line per phase, redrawn every second with a carriage return (Kaggle's cell output overwrites it in place).
    A finished phase is replaced by a one-line summary. Step-based phases (denoising) interpolate inside the current
    step from the previous step's duration, so the bar keeps moving and the ETA counts down between callbacks."""

    WIDTH, LINE = 24, 104
    WATCHED = {"decode": 300, "audio": 120, "save": 120}  # phase -> seconds before a stack dump

    def __init__(self, stream):
        self.out = stream
        self.lock = threading.RLock()
        self.phase = None
        self.visible = False
        self.text_pending = False  # another writer is in the middle of a line: don't draw over it
        threading.Thread(target=self._loop, daemon=True).start()

    def start(self, key, label, total=None, unit=""):
        with self.lock:
            if self.phase is not None and self.phase["key"] == key:
                return
            self.finish()
            now = time.time()
            self.phase = dict(key=key, label=label, total=total, unit=unit, done=0, t0=now, t_done=now, dt=None)
            if key in self.WATCHED:  # a phase that should take seconds: dump every thread's stack if it hangs
                faulthandler.dump_traceback_later(self.WATCHED[key], file=sys.__stderr__)
            self._draw()

    def update(self, done, total=None):
        with self.lock:
            p = self.phase
            if p is None:
                return
            now = time.time()
            if total:
                p["total"] = total
            if done > p["done"]:
                p["dt"] = (now - p["t_done"]) / (done - p["done"])
                p["t_done"], p["done"] = now, done
            self._draw()

    def finish(self):
        with self.lock:
            p, self.phase = self.phase, None
            if p is None:
                return
            faulthandler.cancel_dump_traceback_later()
            elapsed = time.time() - p["t0"]
            line = f"  ✅ {p['label']}"
            if p["total"]:
                line += f" {p['done']}/{p['total']} {p['unit']}"
            line += f" in {_fmt_secs(elapsed)}"
            if p["unit"] == "steps" and p["done"]:
                line += f" ({elapsed / p['done']:.1f}s/step)"
            self._clear()
            self.out.write(line + "\n")
            self.out.flush()

    def abort(self):
        """End the live line of a phase that failed (no ✅ summary)."""
        with self.lock:
            p, self.phase = self.phase, None
            if p is None:
                return
            faulthandler.cancel_dump_traceback_later()
            self._clear()
            self.out.write(f"  ❌ {p['label']} stopped after {_fmt_secs(time.time() - p['t0'])}\n")
            self.out.flush()

    def _render(self):
        p, now = self.phase, time.time()
        elapsed = now - p["t0"]
        if not p["total"]:
            spinner = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"[int(now * 4) % 10]
            return f"  {spinner} {p['label']} | {_fmt_secs(elapsed)}"
        progress = p["done"]
        if p["dt"] and p["done"] < p["total"]:
            progress += min(0.97, (now - p["t_done"]) / p["dt"])
        frac = min(1.0, progress / p["total"])
        fill = int(round(frac * self.WIDTH))
        line = (f"  ⏳ {p['label']:<16} [{'█' * fill}{'░' * (self.WIDTH - fill)}] {frac * 100:3.0f}% | "
                f"{p['done']}/{p['total']} {p['unit']}")
        if p["dt"]:
            eta = max(0.0, (p["total"] - p["done"]) * p["dt"] - (now - p["t_done"]))
            line += f" | {p['dt']:.1f}s each | ETA {_fmt_secs(eta)}"
        return line + f" | {_fmt_secs(elapsed)}"

    def _draw(self):
        if self.phase is None or self.text_pending:
            return
        self.out.write("\r" + self._render().ljust(self.LINE))
        self.out.flush()
        self.visible = True

    def _clear(self):
        if self.visible:
            self.out.write("\r" + " " * self.LINE + "\r")
            self.visible = False

    def _loop(self):
        while True:
            time.sleep(1.0)
            with self.lock:
                self._draw()


class _BarAwareStream:
    """stdout / stderr wrapper: ordinary log lines clear the live bar first and print above it."""

    def __init__(self, stream, bar):
        self._stream, self._bar = stream, bar

    def write(self, data):
        with self._bar.lock:
            if data:
                self._bar._clear()
                self._bar.text_pending = not data.endswith("\n")
            return self._stream.write(data)

    def flush(self):
        self._stream.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


BAR = ConsoleBar(sys.stdout)
sys.stdout = _BarAwareStream(sys.stdout, BAR)
sys.stderr = _BarAwareStream(sys.stderr, BAR)

if not torch.cuda.is_available():
    raise SystemExit("❌ No GPU found. Settings → Accelerator → GPU T4 x2.")
TWO_GPUS = torch.cuda.device_count() >= 2
MAIN_DEV = torch.device("cuda", 0)                        # transformer, VAEs, sampler state
TE_DEV = torch.device("cuda", 1 if TWO_GPUS else 0)      # Gemma-4 text encoder
torch.cuda.set_device(MAIN_DEV)
print("=" * 70)
print("🎬 LTX-2.5 22B Distilled - AIQUEST Academy")
print("=" * 70)
for _i in range(torch.cuda.device_count()):
    _p = torch.cuda.get_device_properties(_i)
    print(f"  GPU {_i}: {_p.name} ({_p.total_memory / 1024**3:.1f} GB, sm_{_p.major}{_p.minor})")
print(f"  RAM: {_ram_in_use()} in use")
if not TWO_GPUS:
    print("  ⚠️ Only one GPU: the text encoder shares it with the video model (slower prompts). Use GPU T4 x2 for full speed.")
sys.stdout.flush()

# T4 (sm_75) has no FlashAttention kernel: memory-efficient SDPA is the fast path. cuDNN autotuning stays off: on a T4
# it benchmarks every new conv3d shape of the video VAE, which costs far more than it saves on a single decode.
torch.backends.cudnn.benchmark = False
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)


# ==== Wan2GP source fixes (Step 1 resets the repo, so they are applied on every start) ====
def _patch(rel_path, old, new, marker, regex=False):
    path = os.path.join(WAN2GP_DIR, rel_path)
    with open(path, encoding="utf-8") as f:
        text = f.read()
    if marker in text:
        return True
    if regex:
        text, count = re.subn(old, new, text, flags=re.M)
    else:
        count = text.count(old)
        text = text.replace(old, new)
    if count == 0:
        print(f"  ⚠️ Wan2GP fix not applied, the code it targets changed: {rel_path}")
        return False
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return True


_LTX = "models/ltx2/"
_FIXES = [
    # The distilled pipeline hard-codes BF16, which the T4 runs without tensor cores: compute in FP16 instead.
    (_LTX + "ltx_pipelines/distilled.py", r"^(\s+)(self\.)?dtype = torch\.bfloat16$", r"\1\2dtype = torch.float16  # AIQ fp16",
     "# AIQ fp16", True),
    # PixelNorm squares its input: FP16 overflows past 256 (flat grey frames). Compute the RMS in FP32.
    (_LTX + "ltx_core/model/common/normalization.py",
     "        mean_sq = torch.mean(x**2, dim=self.dim, keepdim=True)\n        # Normalize by the root-mean-square (RMS).\n"
     "        rms = torch.sqrt(mean_sq + self.eps)\n        return x / rms",
     "        orig_dtype = x.dtype  # AIQ fp32 rms\n        x_float = x.float()\n"
     "        mean_sq = torch.mean(x_float**2, dim=self.dim, keepdim=True)\n        rms = torch.sqrt(mean_sq + self.eps)\n"
     "        return (x_float / rms).to(orig_dtype)", "# AIQ fp32 rms", False),
    # Same overflow in the per-token RMS of the Gemma hidden states (all-NaN text features in FP16).
    (_LTX + "ltx_core/text_encoders/gemma/feature_extractor.py",
     "    b, t, d, l = encoded_text.shape\n    variance = torch.mean(encoded_text**2, dim=2, keepdim=True)\n"
     "    normed = encoded_text * torch.rsqrt(variance + 1e-6)\n    normed = normed.reshape(b, t, d * l)\n",
     "    orig_dtype = encoded_text.dtype  # AIQ fp32 token rms\n    enc_f = encoded_text.float()\n    b, t, d, l = enc_f.shape\n"
     "    variance = torch.mean(enc_f**2, dim=2, keepdim=True)\n    normed = enc_f * torch.rsqrt(variance + 1e-6)\n"
     "    normed = normed.reshape(b, t, d * l).to(orig_dtype)\n", "# AIQ fp32 token rms", False),
    # Latents reach the VAE convolutions in the pipeline dtype; the FP32 video encoder (I2V) needs them cast.
    (_LTX + "ltx_core/model/video_vae/convolution.py", "        x = self.conv(x)",
     "        if x.dtype != self.conv.weight.dtype:  # AIQ conv cast\n            x = x.to(dtype=self.conv.weight.dtype)\n"
     "        x = self.conv(x)", "# AIQ conv cast", False),
    (_LTX + "ltx_core/model/audio_vae/audio_vae.py", "    decoded_audio = audio_decoder(latent)",
     "    if hasattr(audio_decoder, 'conv_in') and hasattr(audio_decoder.conv_in, 'weight'):  # AIQ audio cast\n"
     "        latent = latent.to(dtype=audio_decoder.conv_in.weight.dtype)\n    decoded_audio = audio_decoder(latent)\n"
     "    if hasattr(vocoder, 'vocoder') and hasattr(vocoder.vocoder, 'conv_pre'):\n"
     "        decoded_audio = decoded_audio.to(dtype=vocoder.vocoder.conv_pre.weight.dtype)", "# AIQ audio cast", False),
    (_LTX + "ltx_core/model/audio_vae/vocoder.py",
     "return F.conv1d(x, self.filter.expand(n_channels, -1, -1), stride=self.stride, groups=n_channels)",
     "filt = self.filter.to(dtype=x.dtype, device=x.device).expand(n_channels, -1, -1)  # AIQ filter cast\n"
     "        return F.conv1d(x, filt, stride=self.stride, groups=n_channels)", "# AIQ filter cast", False),
    (_LTX + "ltx_core/model/audio_vae/vocoder.py",
     "spec = F.conv1d(y, self.forward_basis, stride=self.hop_length, padding=0)",
     "basis = self.forward_basis.to(dtype=y.dtype, device=y.device)  # AIQ basis cast\n"
     "        spec = F.conv1d(y, basis, stride=self.hop_length, padding=0)", "# AIQ basis cast", False),
]
_fixed = sum(_patch(path, old, new, marker, regex) for path, old, new, marker, regex in _FIXES)
print(f"🔧 Wan2GP fixes: {_fixed}/{len(_FIXES)} applied")

# ==== Wan2GP + MMGP ====
from mmgp import offload, quant_router

quant_router.unregister_handler(".fp8_quanto_bridge")
quant_router.register_handler("shared.qtypes.int8_convrot")  # the INT8 ConvRot checkpoints (wgp.py does this at import)

# Comfy Kitchen INT8 tensor-core GEMM ("INT8 Kernels: Auto" in the Wan2GP UI, which a standalone engine must turn on).
# Kitchen quantizes activations per row in chunks of this scratch size: 16 MB means ~160-row GEMMs on LTX's FFN,
# 128 MB lets each call cover ~1300 rows, a much better shape for the tensor cores.
from shared.kernels import int8_backend

int8_backend._SCRATCH_BYTES = 128 * 1024 * 1024
try:
    int8_backend.configure("auto", 1)
except Exception as e:
    print(f"⚠️ INT8 kernels unavailable ({e}): INT8 layers are dequantized to FP16 instead (slower)")
INT8_LABEL = {"kitchen": "INT8 tensor-core kernels (Comfy Kitchen)", "triton": "INT8 Triton kernels"}.get(
    int8_backend._backend, "PyTorch INT8 fallback (slow)")

from optimum.quanto.tensor.weights import qbytes as _qbytes
from shared.utils import files_locator as fl

fl.set_checkpoints_paths([MODELS_DIR, "ckpts", "."])
from models.ltx2.ltx2_handler import family_handler
from models.ltx2.ltx_pipelines import distilled as ltx_distilled
import models.ltx2.ltx_pipelines.utils.helpers as ltx_helpers
from models.ltx2.ltx_core.text_encoders.gemma.encoders.base_encoder import RawTextEmbeddings
from models.ltx2.ltx_core.model.video_vae.tiling import SpatialTilingConfig, TilingConfig


@contextmanager
def exact_int8():
    """Run INT8 layers on the exact path (dequantized weights, unquantized activations) inside this block.
    For the LLM text encoder: its activations have extreme outlier channels, and the per-row INT8 activation
    quantization of the fast kernels flattens everything else (worse prompt understanding)."""
    saved_backend = int8_backend._backend
    saved_forward = _qbytes.WeightQBytesLinearFunction.__dict__.get("forward")
    if int8_backend._original_forward is not None:
        _qbytes.WeightQBytesLinearFunction.forward = staticmethod(int8_backend._original_forward)
    int8_backend._backend = "pytorch"
    try:
        yield
    finally:
        int8_backend._backend = saved_backend
        if saved_forward is not None:
            _qbytes.WeightQBytesLinearFunction.forward = saved_forward


# ==== Text encoding: Gemma-4 on its own GPU, real tokens only ====
TIMES = {}  # phase timings of the current generation


def _encode_one(text_encoder, prompt):
    """Same hidden states as text_encoder.encode_raw(prompt) for the real tokens, without running Gemma on padding.
    The tokenizer left-pads every prompt to 1024 tokens. Gemma never lets a real token attend to padding, and the
    feature extractor and connectors ignore padded positions, so Gemma runs on the n real tokens alone, at the
    positions they have in the padded sequence (1024-n .. 1023, so RoPE sees the same angles), and the hidden
    states are padded back with zeros."""
    pairs = text_encoder.tokenizer.tokenize_with_weights(prompt)["gemma"]
    ids = torch.tensor([[t[0] for t in pairs]], dtype=torch.long)
    mask = torch.tensor([[t[1] for t in pairs]], dtype=torch.long)
    total, n = ids.shape[1], int(mask.sum())
    if n == 0 or not bool(mask[0, total - n:].all()):  # not plain left padding: run the original code
        raw = text_encoder.encode_raw(prompt)
        return RawTextEmbeddings(tuple(h.to(MAIN_DEV) for h in raw.hidden_states), raw.attention_mask.to(MAIN_DEV),
                                 raw.padding_side), 0
    gemma = text_encoder.model
    language_model = gemma.model if hasattr(gemma, "model") else gemma
    device = gemma.device  # what encode_raw uses: GPU 1 here, or MMGP's device when it manages Gemma
    real_ids = ids[:, total - n:].to(device)
    positions = torch.arange(total - n, total, device=device).unsqueeze(0)
    out = language_model(input_ids=real_ids, attention_mask=torch.ones_like(real_ids), position_ids=positions,
                         output_hidden_states=True, use_cache=False)
    hidden = []
    for h in out.hidden_states:
        full = torch.zeros((h.shape[0], total, h.shape[-1]), dtype=h.dtype, device=MAIN_DEV)
        full[:, total - n:] = h.to(MAIN_DEV)
        hidden.append(full)
    return RawTextEmbeddings(tuple(hidden), mask.to(MAIN_DEV), "left"), n


def encode_text_fast(text_encoder, prompts):
    t0 = time.time()
    BAR.start("text", "Prompt encoding")
    results, tokens = [], []
    try:
        with torch.no_grad(), exact_int8(), torch.cuda.device(TE_DEV):
            for prompt in prompts:
                raw, n = _encode_one(text_encoder, prompt)
                results.append(raw)
                tokens.append(n)
    except BaseException:
        BAR.abort()
        raise
    BAR.finish()
    TIMES["text"] = TIMES.get("text", 0.0) + time.time() - t0
    bad = sum(int((~torch.isfinite(r.hidden_states[-1][:, -k:])).sum()) for r, k in zip(results, tokens) if k)
    print(f"  📝 Gemma-4 on GPU {TE_DEV.index}: {', '.join(map(str, tokens))} token(s) in {time.time() - t0:.1f}s"
          + (f" | ⚠️ {bad} non-finite values in the text features" if bad else ""))
    return results


_orig_postprocess = ltx_distilled.postprocess_text_embeddings


def postprocess_exact(*args, **kwargs):
    # The feature projection and the two embedding connectors (INT8 ConvRot) also run once per new prompt: exact path.
    with exact_int8():
        return _orig_postprocess(*args, **kwargs)


ltx_distilled.encode_text = encode_text_fast  # distilled.py imported these by name
ltx_distilled.postprocess_text_embeddings = postprocess_exact


# ==== Image-to-Video conditioning ====
# The video VAE encoder overflows in FP16 and returns an all-NaN conditioning latent, which spreads through the joint
# audio/video attention (black clip, noise audio, no error). The encoder runs in FP32 (set up after loading) and the
# latent is handed back in the pipeline dtype. A latent that is still non-finite is re-encoded on the CPU in FP32.
def _cond_attr(item):
    return "latent" if hasattr(item, "latent") else ("keyframes" if hasattr(item, "keyframes") else None)


def _wrap_conditioner(fn, label):
    def inner(*args, **kwargs):
        pipe_dtype = kwargs.get("dtype")
        encoder = kwargs.get("video_encoder")
        try:
            enc_dtype = next(encoder.parameters()).dtype
        except Exception:
            enc_dtype = None
        if enc_dtype is not None and enc_dtype != pipe_dtype:
            kwargs = dict(kwargs, dtype=enc_dtype)
        if kwargs.get("images"):
            BAR.start("encode", "Image encoding")
        items = fn(*args, **kwargs)

        def nonfinite(entries):
            for it in entries:
                a = _cond_attr(it)
                v = getattr(it, a) if a else None
                if torch.is_tensor(v) and not bool(torch.isfinite(v).all()):
                    return True
            return False

        if encoder is not None and nonfinite(items):
            print(f"  ⚠️ {label}: non-finite conditioning latent, re-encoding on the CPU in FP32")
            origin = next(encoder.parameters()).device
            try:
                encoder.to(device="cpu", dtype=torch.float32)
                cpu_items = fn(*args, **dict(kwargs, dtype=torch.float32, device=torch.device("cpu")))
                for it in cpu_items:
                    a = _cond_attr(it)
                    if a is not None and kwargs.get("device") is not None:
                        setattr(it, a, getattr(it, a).to(device=kwargs["device"]))
                items = cpu_items
            except Exception as retry_error:
                print(f"  ⚠️ CPU re-encode failed ({retry_error}); non-finite values are zeroed instead")
            finally:
                encoder.to(device=origin)
        for item in items:
            a = _cond_attr(item)
            lat = getattr(item, a) if a else None
            if not torch.is_tensor(lat):
                continue
            fixed = lat
            if not bool(torch.isfinite(fixed).all()):
                fixed = torch.nan_to_num(fixed.float(), nan=0.0, posinf=0.0, neginf=0.0)
            if pipe_dtype is not None and fixed.dtype != pipe_dtype:
                fixed = fixed.to(pipe_dtype)
            if fixed is not lat:
                setattr(item, a, fixed)
        if kwargs.get("images"):
            BAR.finish()
        return items
    return inner


for _name, _label in (("image_conditionings_by_replacing_latent", "start frame"),
                      ("image_conditionings_by_adding_guiding_latent", "end frame")):
    _orig = getattr(ltx_helpers, _name, None)
    if _orig is not None:
        _wrapped = _wrap_conditioner(_orig, _label)
        setattr(ltx_helpers, _name, _wrapped)
        if hasattr(ltx_distilled, _name):  # imported by name in distilled.py
            setattr(ltx_distilled, _name, _wrapped)


# ==== VAE decode: timed, with a tiled retry if a large clip does not fit ====
_orig_decode_video = ltx_distilled.vae_decode_video_to_tensor
_orig_decode_audio = ltx_distilled.vae_decode_audio


def _tiling_with_spatial(tiling_config, tile):
    overlap = max(0, (tile // 4) // 32 * 32)
    temporal = tiling_config.temporal_config if tiling_config is not None else None
    return TilingConfig(spatial_config=SpatialTilingConfig(tile_size_in_pixels=tile, tile_overlap_in_pixels=overlap),
                        temporal_config=temporal)


def decode_video(latent, video_decoder, tiling_config=None, *args, **kwargs):
    latent_tensor = latent[0] if isinstance(latent, list) else latent  # the decoder empties the list: keep it for a retry
    if isinstance(latent, list):
        latent.clear()
    t0 = time.time()
    BAR.start("decode", "VAE decoding")
    try:
        configs = [tiling_config] + [_tiling_with_spatial(tiling_config, t) for t in (512, 256)]
        for attempt, config in enumerate(configs):
            try:
                out = _orig_decode_video([latent_tensor], video_decoder, config, *args, **kwargs)
                break
            except torch.cuda.OutOfMemoryError:
                if attempt == len(configs) - 1:
                    raise
                torch.cuda.synchronize()  # never release memory while kernels queued before the failure still run
                gc.collect()
                torch.cuda.empty_cache()
                tile = configs[attempt + 1].spatial_config.tile_size_in_pixels
                print(f"  ⚠️ VAE decode out of memory, retrying with {tile} px tiles")
    except BaseException:
        BAR.abort()
        raise
    BAR.finish()
    TIMES["decode"] = TIMES.get("decode", 0.0) + time.time() - t0
    return out


def decode_audio(*args, **kwargs):
    t0 = time.time()
    BAR.start("audio", "Audio decoding")
    try:
        out = _orig_decode_audio(*args, **kwargs)
    except BaseException:
        BAR.abort()
        raise
    BAR.finish()
    TIMES["decode"] = TIMES.get("decode", 0.0) + time.time() - t0
    return out


ltx_distilled.vae_decode_video_to_tensor = decode_video
ltx_distilled.vae_decode_audio = decode_audio


# ==== Denoising steps ====
# The distilled pipeline ignores sampling_steps: it walks the module-level 9-sigma schedule (8 steps). Shorter
# schedules drop sigmas from the 1.0 -> 0.975 cluster, which barely moves the latent, and keep the ones that do the work.
FULL_SIGMAS = list(ltx_distilled.DISTILLED_SIGMA_VALUES)
STEP_SCHEDULES = {
    8: FULL_SIGMAS,
    7: [1.0, 0.99375, 0.9875, 0.975, 0.909375, 0.725, 0.421875, 0.0],
    6: [1.0, 0.9875, 0.975, 0.909375, 0.725, 0.421875, 0.0],
    5: [1.0, 0.975, 0.909375, 0.725, 0.421875, 0.0],
    4: [1.0, 0.909375, 0.725, 0.421875, 0.0],
}


def set_denoising_steps(num_steps):
    schedule = STEP_SCHEDULES.get(int(num_steps), FULL_SIGMAS)
    ltx_distilled.DISTILLED_SIGMA_VALUES = schedule
    return len(schedule) - 1


# ==== Sizes ====
RESOLUTIONS = {"384p (fastest)": 384, "480p (recommended)": 480, "704p (high quality, slow)": 704,
               "1088p (very slow)": 1088}
ASPECTS = {"16:9 Landscape": 16 / 9, "9:16 Portrait": 9 / 16, "1:1 Square": 1.0, "4:3 Standard": 4 / 3,
           "3:4 Portrait": 3 / 4}
DURATIONS = {"2 s (49 frames)": 49, "3 s (73 frames)": 73, "5 s (121 frames)": 121, "8 s (193 frames)": 193,
             "10 s (241 frames)": 241}


def snap64(value):
    # LTX rounds width / height up to a multiple of 64 before building the latent grid: snap here as well, so the start
    # frame is prepared at exactly the size the video is generated at.
    return max(64, int(round(float(value) / 64.0)) * 64)


def size_for(resolution, aspect):
    base = RESOLUTIONS.get(resolution, 480)
    ratio = ASPECTS.get(aspect, 16 / 9)
    if ratio >= 1.0:
        return snap64(base * ratio), snap64(base)
    return snap64(base), snap64(base / ratio)


def fit_to_image(image, width, height):
    # Image-to-Video: follow the frame's own aspect ratio at the preset's pixel budget, so nothing is cropped.
    img_w, img_h = image.size
    scale = ((width * height) / float(img_w * img_h)) ** 0.5
    return snap64(img_w * scale), snap64(img_h * scale)


def prepare_conditioning_image(image, width, height):
    # Centre-crop to the target ratio, then resample to exactly W x H.
    img_w, img_h = image.size
    target, src = width / float(height), img_w / float(img_h)
    if src > target:
        new_w = int(round(img_h * target))
        left = (img_w - new_w) // 2
        image = image.crop((left, 0, left + new_w, img_h))
    elif src < target:
        new_h = int(round(img_w / target))
        top = (img_h - new_h) // 2
        image = image.crop((0, top, img_w, top + new_h))
    return image.resize((width, height), Image.Resampling.LANCZOS)


def vae_tile_size(width, height):
    # Whole-frame decode up to 1024 px (tiles leave seams); above that a full-frame decode does not fit a T4.
    longest = max(int(width), int(height))
    return 0 if longest <= 1024 else (512 if longest <= 1536 else 256)


@torch.no_grad()
def save_mp4(video, audio, sample_rate, path, fps=FPS):
    """Streams the decoded uint8 [3, T, H, W] clip and its audio to ffmpeg in one pass."""
    import soundfile as sf

    _, t, h, w = video.shape
    wav = path[:-4] + ".wav"
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps),
           "-i", "-"]
    if audio is not None:
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim == 2 and audio.shape[0] in (1, 2) and audio.shape[1] > audio.shape[0]:
            audio = audio.T  # soundfile wants [samples, channels]
        sf.write(wav, audio, int(sample_rate))
        cmd += ["-i", wav, "-c:a", "aac", "-b:a", "192k"]
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "17", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            "-shortest", path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)  # no preexec_fn: a real fork() would copy the pinned RAM
    try:
        frames = video if video.dtype == torch.uint8 else video.clamp(0, 255).to(torch.uint8)
        for start in range(0, t, 16):
            proc.stdin.write(frames[:, start:start + 16].permute(1, 2, 3, 0).contiguous().numpy().tobytes())
        proc.stdin.close()
    except BrokenPipeError:
        pass
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed with exit code {proc.returncode}")
    if os.path.exists(wav):
        os.remove(wav)


# ==== Load the model (once; it stays in memory) ====
print("\n📦 Loading LTX-2.5 22B Distilled (INT8 ConvRot) + Gemma-4 12B (INT8 ConvRot)...")
sys.stdout.flush()
t_load = time.time()
BASE_MODEL_TYPE = "ltx2_25_22B"
model_def = {"ltx2_pipeline": "distilled", "name": "LTX-2 2.5 Distilled 22B", "architecture": BASE_MODEL_TYPE}
model_def.update(family_handler.query_model_def(BASE_MODEL_TYPE, model_def))

# Loaded under no_grad, not inference_mode: inference tensors fail when touched from the generation thread.
with torch.no_grad():
    wan_model, pipe = family_handler.load_model(
        model_filename=TRANSFORMER_FILE,
        model_type="ltx2_25_22B_distilled",
        base_model_type=BASE_MODEL_TYPE,
        model_def=model_def,
        dtype=torch.float16,
        VAE_dtype=torch.float16,
        text_encoder_filename=GEMMA_FILE,
    )
mmgp_kwargs = {}
if isinstance(pipe, dict) and "pipe" in pipe:
    mmgp_kwargs = pipe
    pipe = mmgp_kwargs.pop("pipe")
loras = mmgp_kwargs.pop("loras", [])
if "transformer" in pipe and "transformer" not in loras:
    loras.append("transformer")

# Single-stage generation never uses the x2 spatial upsampler (it is only loaded because the checkpoint set lists it):
# drop it, which frees its 1 GB FP16 copy in RAM.
pipe.pop("spatial_upsampler", None)
wan_model.spatial_upsampler = None
if getattr(wan_model.pipeline, "models", None) is not None:
    wan_model.pipeline.models.spatial_upsampler = None

# FP32 video encoder (Image-to-Video only, one frame per generation). _lock_dtype keeps MMGP from folding it back to FP16.
_video_encoder = pipe.get("video_encoder")
if _video_encoder is not None:
    _video_encoder.float()
    for _m in _video_encoder.modules():
        _m._lock_dtype = torch.float32

# Gemma-4 on GPU 1 for the whole session. LTX only reads its hidden states, so the LM head is not needed.
text_encoder = wan_model.text_encoder
if TWO_GPUS:
    try:
        if hasattr(text_encoder.model, "lm_head"):
            text_encoder.model.lm_head = torch.nn.Identity()
        text_encoder.to(TE_DEV)
        pipe.pop("text_encoder", None)
        with torch.cuda.device(TE_DEV):
            torch.cuda.empty_cache()
        print(f"  ✅ Gemma-4 text encoder resident on GPU 1 ({_gb(torch.cuda.memory_allocated(TE_DEV))}, "
              f"{_gb(torch.cuda.mem_get_info(TE_DEV)[0])} free)")
    except Exception as e:
        print(f"  ⚠️ Gemma-4 stays on GPU 0 with offloading ({e})")
        text_encoder.to("cpu")
        TE_DEV = MAIN_DEV
        with torch.cuda.device(1):
            torch.cuda.empty_cache()
gc.collect()

# The INT8 transformer (19.5 GB) does not fit a 15 GB T4: MMGP keeps as many blocks in VRAM as the budget allows and
# streams the others from pinned RAM at every step, overlapped with the compute. The budget leaves ~4.5 GB for
# activations (enough for 1088p).
_free0_mb = int(torch.cuda.mem_get_info(MAIN_DEV)[0] / 1024**2)
TRANSFORMER_BUDGET_MB = int(max(6000, min(10500, _free0_mb - 4500)))
budgets = {"transformer": TRANSFORMER_BUDGET_MB, "video_encoder": 3000, "vae": 3000, "*": 1000}
if "text_encoder" in pipe:
    budgets["text_encoder"] = int(max(1500, min(8000, _free0_mb - 6000)))
print(f"  ⚡ MMGP: transformer budget {TRANSFORMER_BUDGET_MB} MB in VRAM, the rest streamed from pinned RAM")
sys.stdout.flush()
offload.profile(
    pipe,
    profile_no=4,
    pinnedMemory="transformer",
    partialPinning=True,
    perc_reserved_mem_max=0.40 if "text_encoder" not in pipe else 0.35,  # RAM is shared with the page cache of the other models
    asyncTransfers=True,
    quantizeTransformer=False,
    convertWeightsFloatTo=torch.float16,
    loras=loras,
    budgets=budgets,
    **mmgp_kwargs,
)
offload.shared_state["_attention"] = "sdpa"
offload.shared_state["_radial"] = False
torch.cuda.empty_cache()
print(f"✅ Model ready in {time.time() - t_load:.0f}s | {INT8_LABEL} | RAM {_ram_in_use()}")
sys.stdout.flush()


# ==== Generation ====
PROGRESS = {"active": False, "fraction": 0.0, "text": "", "started": 0.0, "finished": 0.0, "video": None, "message": ""}


def report(fraction, text):
    PROGRESS.update(active=True, fraction=float(fraction), text=text)


def run_generation(prompt, image_start, image_end, width, height, frames, steps, seed):
    TIMES.clear()
    t0 = time.time()
    state = {"t": None, "total": steps, "last": None}

    def callback(step_idx=-1, latent=None, force_refresh=True, read_state=False, override_num_inference_steps=-1,
                 pass_no=-1, progress_unit=None, progress_title=None, **kwargs):
        if read_state:  # interrupt polling, once per transformer block
            return
        if progress_unit is not None:  # layer / tile progress of other phases
            return
        now = time.time()
        if step_idx < 0:  # denoising starts
            state["total"] = override_num_inference_steps if override_num_inference_steps and override_num_inference_steps > 0 else steps
            state["t"] = state["last"] = now
            BAR.start("denoise", "Denoising", state["total"], "steps")
            report(0.12, f"🎬 Denoising step 0/{state['total']}")
            return
        done = step_idx + 1
        dt = now - (state["last"] or now)
        state["last"] = now
        TIMES["denoise"] = now - (state["t"] or now)
        BAR.update(done, state["total"])
        report(0.12 + 0.76 * done / state["total"], f"🎬 Denoising step {done}/{state['total']} ({dt:.1f}s/step)")
        if done >= state["total"]:
            BAR.finish()
            report(0.9, "🎞️ Decoding video + audio...")

    def set_progress_status(status):
        if status and "decod" not in status.lower():
            report(PROGRESS["fraction"], f"⏳ {status}...")

    kwargs = dict(input_prompt=prompt, image_start=image_start, height=height, width=width, frame_num=frames, fps=float(FPS),
                  seed=seed, callback=callback, set_progress_status=set_progress_status,
                  VAE_tile_size=vae_tile_size(width, height), input_video_strength=1.0, denoising_strength=1.0,
                  guide_scale=1.0, audio_cfg_scale=1.0, sampling_steps=int(steps), guide_phases=1,
                  sample_solver="distilled_8_steps", self_refiner_setting=0, n_prompt="")
    if image_end is not None:
        kwargs["image_end"] = image_end
    wan_model._interrupt = False
    report(0.02, "📝 Encoding the prompt...")
    try:
        with torch.no_grad():  # grad mode is per thread and ON in a new thread
            result = wan_model.generate(**kwargs)
    except BaseException:
        BAR.abort()
        raise
    BAR.abort() if wan_model._interrupt else BAR.finish()
    TIMES["total"] = time.time() - t0
    return result


def generate_video(prompt, start_img, end_img, duration, resolution, aspect, steps, seed):
    PROGRESS.update(active=True, fraction=0.0, text="Starting...", started=time.time(), finished=0.0, video=None, message="")
    try:
        prompt = (prompt or "").strip()
        if not prompt:
            return None, "❌ Please write a prompt."
        frames = DURATIONS.get(duration, 121)
        width, height = size_for(resolution, aspect)
        image_start = start_img.convert("RGB") if start_img is not None else None
        image_end = end_img.convert("RGB") if end_img is not None else None
        ref = image_start if image_start is not None else image_end
        if ref is not None:  # Image-to-Video: the output follows the frame's aspect ratio
            width, height = fit_to_image(ref, width, height)
            image_start = prepare_conditioning_image(image_start, width, height) if image_start is not None else None
            image_end = prepare_conditioning_image(image_end, width, height) if image_end is not None else None
        steps = set_denoising_steps(steps)
        if seed is None or int(seed) < 0:
            seed = random.randint(0, 2**31 - 1)
        seed = int(seed)
        mode = ("First + Last Frame → Video" if image_start is not None and image_end is not None
                else "Image → Video" if image_start is not None else "Last Frame → Video" if image_end is not None
                else "Text → Video")
        print(f"\n{'=' * 70}\n🎬 {mode} | {width}x{height} | {frames} frames ({frames / FPS:.1f}s) | {steps} steps | seed {seed}")
        print(f"📝 {prompt[:300]}{'...' if len(prompt) > 300 else ''}\n{'=' * 70}")
        sys.stdout.flush()
        torch.cuda.reset_peak_memory_stats(MAIN_DEV)

        result = run_generation(prompt, image_start, image_end, width, height, frames, steps, seed)
        if wan_model._interrupt:
            return None, "⏹️ Generation stopped."
        if not isinstance(result, dict) or not torch.is_tensor(result.get("x")):
            return None, "❌ The pipeline returned no video."
        report(0.97, "💾 Saving MP4...")
        t_save = time.time()
        out_path = os.path.join(OUTPUT_DIR, f"ltx25_{time.strftime('%Y%m%d_%H%M%S')}_seed{seed}.mp4")
        BAR.start("save", "Saving MP4")
        save_mp4(result["x"], result.get("audio"), result.get("audio_sampling_rate", 24000), out_path)
        BAR.finish()
        t_text, t_denoise, t_decode = TIMES.get("text", 0.0), TIMES.get("denoise", 0.0), TIMES.get("decode", 0.0)
        per_step = t_denoise / max(1, steps)
        total = TIMES.get("total", 0.0) + time.time() - t_save
        status = "\n".join([
            f"✅ Done in {total:.0f}s ({mode})",
            f"⚡ Prompt {t_text:.0f}s | Denoising {t_denoise:.0f}s ({per_step:.1f}s/step) | VAE + audio {t_decode:.0f}s",
            f"📐 {width}x{height} | {frames} frames @ {FPS} fps ({frames / FPS:.1f}s) | {steps} steps | seed {seed}",
            f"📁 {out_path}",
        ])
        print(f"✅ Saved {out_path} | total {total:.0f}s = prompt {t_text:.0f}s + denoise {t_denoise:.0f}s "
              f"({per_step:.1f}s/step) + VAE/audio {t_decode:.0f}s + MP4 {time.time() - t_save:.0f}s | "
              f"peak VRAM GPU0 {_gb(torch.cuda.max_memory_allocated(MAIN_DEV))} | RAM {_ram_in_use()}\n")
        sys.stdout.flush()
        return out_path, status
    except torch.cuda.OutOfMemoryError:
        traceback.print_exc()
        return None, "❌ Out of GPU memory. Pick a lower resolution or a shorter duration and try again."
    except Exception as e:
        traceback.print_exc()
        if "CUDA error" in str(e):  # e.g. an illegal memory access: the GPU context is unusable from here on
            print("❌ The GPU is in an error state. Re-run Step 2 to restart the engine.")
            return None, f"❌ GPU error: {e}\nRe-run Step 2 to restart the engine."
        return None, f"❌ Error: {e}"
    finally:
        BAR.abort()
        PROGRESS["active"] = False
        gc.collect()
        torch.cuda.empty_cache()


# ==== Generation as a background job ====
# The Generate button only starts a job and returns: the page asks for its state every 2 s with short, independent
# requests (queue=False: plain HTTP, no event stream), so a dropped tunnel or a reloaded page never loses the video.
JOB = {"id": 0, "done": 0, "running": False, "video": None, "status": "Ready."}


def _job_worker(job_id, args):
    try:
        torch.cuda.set_device(MAIN_DEV)  # the current CUDA device is per thread
        video, status = generate_video(*args)
    except BaseException as e:  # generate_video reports its own errors; this is a last resort
        video, status = None, f"❌ Error: {e}"
    JOB.update(running=False, done=job_id, video=video, status=status)
    PROGRESS.update(active=False, finished=time.time(), video=os.path.basename(video) if video else None,
                    message=status.splitlines()[0] if status else "")


def start_generation(*args):
    if JOB["running"]:
        return gr.skip(), "⏳ A video is already being generated. It appears here as soon as it is ready."
    JOB.update(id=JOB["id"] + 1, running=True)
    PROGRESS.update(active=True, fraction=0.0, text="Starting...", started=time.time(), finished=0.0, video=None, message="")
    threading.Thread(target=_job_worker, args=(JOB["id"], args), daemon=True).start()
    return None, ("🚀 Generation started. Progress is shown at the top of the page and the video appears here when it "
                  "is ready. Reloading the page or a dropped connection does not stop the job.")


def poll_job(seen):
    """Every 2 s: live progress while a job runs, the result once (per page) when it is done."""
    if JOB["running"]:
        elapsed = time.time() - PROGRESS["started"] if PROGRESS["started"] else 0.0
        return gr.skip(), f"⏳ {PROGRESS['text'] or 'Working...'} · {_fmt_secs(elapsed)}", seen
    if JOB["done"] and JOB["done"] != seen:
        return JOB["video"], JOB["status"], JOB["done"]
    return gr.skip(), gr.skip(), seen


def stop_generation():
    if not JOB["running"]:
        return "Nothing is running."
    wan_model._interrupt = True
    return "⏹️ Stopping..."


def load_latest_video():
    videos = sorted(glob.glob(os.path.join(OUTPUT_DIR, "*.mp4")), key=os.path.getmtime)
    if not videos:
        return None, f"No videos in {OUTPUT_DIR} yet."
    return videos[-1], f"📁 Loaded latest video: {videos[-1]}"


# ==== Gradio UI (AIQUEST Academy branding) ====
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
  <div class="brand-title">🎬 LTX-2.5 22B Distilled - Video + Audio</div>
  <div class="brand-subtitle">Created by <strong>AIQuest Academy</strong> &nbsp;|&nbsp; Kaggle GPU T4 x2 Edition</div>
  <div class="btn-row">
    <a href="https://www.youtube.com/@aiquestacademy?sub_confirmation=1" target="_blank" class="social-btn yt-btn">▶ Subscribe</a>
    <a href="https://x.com/aiquestacademy" target="_blank" class="social-btn x-btn">𝕏 Follow on X</a>
    <a href="https://aiquest.site" target="_blank" class="social-btn sup-btn">❤️ Support My Work</a>
  </div>
</div>
"""
STATUS_HTML = (f'<div class="status-line">⚡ {INT8_LABEL} &nbsp;|&nbsp; '
               + ("Transformer on GPU 0, Gemma-4 text encoder on GPU 1" if TE_DEV != MAIN_DEV else "Single GPU")
               + ' &nbsp;|&nbsp; 24 fps video + synced audio</div>')
FOOTER_HTML = '<div class="brand-footer">⚡ Made with ❤️ by <strong>AIQUEST Academy</strong> · aiquest.site · © All rights reserved</div>'

PROGRESS_HEAD = """
<script>
(function () {
  // Progress + result overlay fed by plain HTTP polling: it keeps working when a tunnel drops Gradio's event stream.
  var box = null, fill = null, label = null, hideAt = 0;
  function build() {
    box = document.createElement("div");
    box.style.cssText = "position:fixed;top:12px;left:50%;transform:translateX(-50%);z-index:99999;width:min(560px,92vw);" +
      "background:#1f2937;border-radius:12px;box-shadow:0 8px 24px rgba(0,0,0,.25);padding:10px 14px;display:none;" +
      "font-family:Inter,sans-serif;color:#fff;font-size:13px";
    label = document.createElement("div");
    label.style.cssText = "margin-bottom:6px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis";
    var track = document.createElement("div");
    track.style.cssText = "height:8px;background:rgba(255,255,255,.18);border-radius:6px;overflow:hidden";
    fill = document.createElement("div");
    fill.style.cssText = "height:100%;width:0;background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);transition:width .6s ease";
    track.appendChild(fill); box.appendChild(label); box.appendChild(track); document.body.appendChild(box);
  }
  async function tick() {
    try {
      var r = await fetch("aiq_progress?t=" + Date.now(), {cache: "no-store"});
      var p = await r.json();
      if (!box && document.body) build();
      if (box) {
        if (p.active) {
          box.style.display = "block"; hideAt = 0;
          fill.style.width = Math.round(p.fraction * 100) + "%";
          label.textContent = "⏳ " + p.text + " · " + Math.round(p.elapsed) + "s";
        } else if (box.style.display === "block") {
          if (!hideAt) {
            fill.style.width = "100%";
            label.textContent = (p.video ? "✅ Done in " : "⚠️ Finished after ") + Math.round(p.elapsed) + "s ";
            if (p.video) {
              var link = document.createElement("a");
              link.href = "aiq_video?t=" + Date.now(); link.target = "_blank";
              link.style.cssText = "color:#c7d2fe;font-weight:600";
              link.textContent = "· ▶ open / download the video";
              label.appendChild(link);
            } else if (p.message) { label.textContent += "· " + p.message; }
            hideAt = Date.now() + 60000;
          } else if (Date.now() > hideAt) { box.style.display = "none"; }
        }
      }
    } catch (e) {}
    setTimeout(tick, 1000);
  }
  tick();
})();
</script>
"""

EXAMPLES = [
    ["A cinematic close-up of a snow leopard resting on a rocky ledge in the Himalayas during light snowfall. It slowly "
     "turns its head toward the camera and blinks, breath fogging in the cold air. Soft wind howls across the peaks and "
     "snowflakes tick against the rocks."],
    ["A street musician plays a violin on a rainy neon-lit Tokyo street at night. The camera slowly circles him as puddles "
     "reflect pink and blue signs. The violin melody echoes between the buildings over the hiss of rain and distant traffic."],
    ["A young woman in a cozy cafe looks up from her laptop, smiles warmly at the camera and says: \"You made it! I saved "
     "you the best seat.\" Quiet chatter, an espresso machine hissing and cups clinking in the background."],
]

print("🖥️ Building the UI...")
sys.stdout.flush()
with gr.Blocks(theme=gr.themes.Soft(), css=CSS, head=PROGRESS_HEAD, title="LTX-2.5 22B Distilled | AIQUEST Academy") as demo:
    gr.HTML(BRAND_HTML)
    gr.HTML(STATUS_HTML)
    gr.Markdown("💡 **Tips:** describe the subject, the action, the camera, the light, and the sounds or speech you want "
                "(the model makes the audio too). 480p + 5 s is the sweet spot on T4; 384p is the fastest preview. "
                "Time per step grows with resolution x duration.")
    with gr.Row():
        with gr.Column(scale=3):
            prompt = gr.Textbox(label="🎬 Prompt", lines=5,
                                placeholder="A cinematic shot of ... (subject, action, camera, light, sound)")
            with gr.Accordion("🖼️ Image → Video (start / end frame, optional)", open=False):
                with gr.Row():
                    start_img = gr.Image(type="pil", label="Start frame", height=200)
                    end_img = gr.Image(type="pil", label="End frame", height=200)
                gr.Markdown("*Start only = animate the image. Start + end = first/last-frame interpolation. With a frame, "
                            "the video follows the frame's aspect ratio (the Aspect ratio box is ignored).*")
        with gr.Column(scale=2):
            resolution = gr.Dropdown(list(RESOLUTIONS), value="480p (recommended)", label="📐 Resolution")
            aspect = gr.Dropdown(list(ASPECTS), value="16:9 Landscape", label="📏 Aspect ratio")
            duration = gr.Dropdown(list(DURATIONS), value="5 s (121 frames)", label="⏱️ Duration")
            steps = gr.Slider(4, 8, value=8, step=1, label="⚡ Denoising steps",
                              info="8 = full quality; fewer steps are faster but softer")
            seed = gr.Number(value=-1, precision=0, label="🎲 Seed (-1 = random)")

    with gr.Row():
        gen_btn = gr.Button("🎬 Generate Video", variant="primary", size="lg")
        stop_btn = gr.Button("🛑 Stop", size="lg", elem_id="stop-btn")
        clear_btn = gr.Button("🗑️ Clear", size="lg", elem_id="clear-btn")
    video_out = gr.Video(label="🎥 Generated video (with audio)")
    latest_btn = gr.Button("📂 Load latest video (use if the UI lost its connection)")
    status_out = gr.Textbox(label="ℹ️ Status", lines=5, interactive=False)
    seen_job = gr.State(0)
    poll_timer = gr.Timer(2.0)

    gr.Examples(EXAMPLES, inputs=[prompt], label="✨ Example prompts")
    gr.HTML(FOOTER_HTML)

    gen_btn.click(start_generation, inputs=[prompt, start_img, end_img, duration, resolution, aspect, steps, seed],
                  outputs=[video_out, status_out], queue=False)
    poll_timer.tick(poll_job, inputs=seen_job, outputs=[video_out, status_out, seen_job], queue=False,
                    show_progress="hidden")
    stop_btn.click(stop_generation, outputs=status_out, queue=False)
    latest_btn.click(load_latest_video, outputs=[video_out, status_out], queue=False)
    clear_btn.click(lambda: (None, "Ready.", "", None, None, -1),
                    outputs=[video_out, status_out, prompt, start_img, end_img, seed], queue=False)


# ==== Cloudflare quick tunnel: a second public link in case the Gradio share link fails ====
# Started without preexec_fn: that would force a real fork() of this process, which has to copy every pinned page.
# The link only works once cloudflared has registered with Cloudflare's edge and the random *.trycloudflare.com name
# is in DNS; a browser that asks too early caches the "not found" answer for minutes. cloudflared tries QUIC (UDP)
# first, which Kaggle often blocks: hence --protocol http2.
def _dns_published(host):
    """Cloudflare's DNS-over-HTTPS: True / False, or None when the lookup itself is unavailable."""
    try:
        import json
        req = urllib.request.Request(f"https://cloudflare-dns.com/dns-query?name={host}&type=A",
                                     headers={"accept": "application/dns-json"})
        return bool(json.loads(urllib.request.urlopen(req, timeout=5).read()).get("Answer"))
    except Exception:
        return None


def start_cloudflare_tunnel(port):
    import stat
    found = {"url": None, "registered": False, "ready": False}
    binary = "/tmp/cloudflared"
    try:
        if not os.path.exists(binary):
            urllib.request.urlretrieve(
                "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64", binary)
            os.chmod(binary, os.stat(binary).st_mode | stat.S_IEXEC)
        tunnel = subprocess.Popen([binary, "tunnel", "--url", f"http://127.0.0.1:{port}", "--no-autoupdate",
                                   "--protocol", "http2"],
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        import atexit
        atexit.register(tunnel.kill)
    except Exception as e:
        print(f"⚠️ Cloudflare tunnel unavailable: {e}")
        return found

    def _watch():
        for line in tunnel.stdout:  # keep draining so cloudflared never blocks on a full pipe
            match = re.search(r"https://(?!api\.)[a-z0-9-]+\.trycloudflare\.com", line)  # skip cloudflared's API host
            if match and not found["url"]:
                found["url"] = match.group(0)
            if "Registered tunnel connection" in line:
                found["registered"] = True

    def _wait_ready():
        deadline, registered_at = time.time() + 300, None
        while time.time() < deadline:
            if found["registered"] and registered_at is None:
                registered_at = time.time()
            published = _dns_published(found["url"].split("//", 1)[1]) if found["url"] and registered_at else False
            if published or (published is None and time.time() - registered_at > 15):
                found["ready"] = True
                print(f"\n✅ Cloudflare link is ready: {found['url']}\n")
                sys.stdout.flush()
                return
            time.sleep(3)
        print("⚠️ The Cloudflare link did not come up within 5 minutes; use the Gradio share link.")

    threading.Thread(target=_watch, daemon=True).start()
    threading.Thread(target=_wait_ready, daemon=True).start()
    return found


print("🌐 Starting the public links (Cloudflare + Gradio share)...")
sys.stdout.flush()
cf = start_cloudflare_tunnel(SERVER_PORT)
demo.queue(max_size=20)
demo.launch(server_name="127.0.0.1", server_port=SERVER_PORT, share=True, inline=False, show_error=True,
            prevent_thread_lock=True, quiet=True, ssr_mode=False, max_threads=8, allowed_paths=[OUTPUT_DIR])
from fastapi.responses import FileResponse, JSONResponse
from fastapi.routing import APIRoute


def _progress_route():
    end = PROGRESS["finished"] or time.time()
    elapsed = end - PROGRESS["started"] if PROGRESS["started"] else 0.0
    return JSONResponse({**PROGRESS, "elapsed": elapsed}, headers={"Cache-Control": "no-store"})


def _video_route():
    """The newest video as a plain file (plays in the browser, with a download option): no Gradio involved."""
    videos = sorted(glob.glob(os.path.join(OUTPUT_DIR, "*.mp4")), key=os.path.getmtime)
    if not videos:
        return JSONResponse({"error": "no video yet"}, status_code=404)
    name = os.path.basename(videos[-1])
    return FileResponse(videos[-1], media_type="video/mp4", headers={"Content-Disposition": f'inline; filename="{name}"',
                                                                      "Cache-Control": "no-store"})


for _path, _route in (("/aiq_progress", _progress_route), ("/aiq_video", _video_route)):
    demo.app.router.routes.insert(0, APIRoute(_path, _route, methods=["GET"]))  # ahead of Gradio's routes
for _ in range(60):  # give the tunnel up to ~30 s to come up before printing the links
    if cf["ready"]:
        break
    time.sleep(0.5)

print("\n" + "=" * 70)
print("🎬 LTX-2.5 22B Distilled is LIVE!")
print("=" * 70)
if cf["ready"]:
    print(f"🌐 Cloudflare tunnel: {cf['url']}")
else:
    print(f"🌐 Cloudflare tunnel: ⏳ warming up{' (' + cf['url'] + ')' if cf['url'] else ''} - wait for the "
          f"'✅ Cloudflare link is ready' line before opening it; opening it too early can keep it unreachable for minutes")
print(f"🔗 Gradio share:      {getattr(demo, 'share_url', None) or '(unavailable, use the Cloudflare link)'}")
print(f"💾 Videos are saved to {OUTPUT_DIR} (Kaggle Output panel)")
print(f"🎞️ Latest video, without the UI: <link>/aiq_video (e.g. {cf['url'] or 'https://...trycloudflare.com'}/aiq_video)")
print("=" * 70 + "\n")
sys.stdout.flush()
while True:  # keep the server process alive; generation logs stream to the notebook cell
    time.sleep(3600)