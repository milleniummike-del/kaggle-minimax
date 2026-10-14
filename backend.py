"""MiniMax H3 Turbo (INT8 ConvRot) video + audio backend engine for Kaggle GPU T4 x2 - AIQUEST Academy.

Backend engine driving Wan2GP's MiniMax H3 code directly:

* DiT: a pruned 20B H3 checkpoint in INT8 ConvRot. The default ("pdmd") is the base H3 transformer (FL2VA / T2VA
  partition) with the PDMD 4-step distillation LoRA applied live and exact ("dmad": the DMAD 4-step LoRA
  instead): text / image / first+last frame to video in 4 steps. "fused" is MATLOWAI's Ref2VA turbo file, the
  variant that also does reference-to-video.
* Every INT8 linear runs on Comfy Kitchen's INT8 tensor-core GEMM (W8A8 with the ConvRot Hadamard rotation).
* Activations stay in BF16 like upstream H3. Only attention runs in FP16 (Turing has no BF16 attention kernel);
  q and k are RMS-normalized there, so FP16 is safe. The residual stream never sees FP16, so it cannot overflow.
* Tensor parallel DiT: every block is split in half across the two T4s (attention heads and FFN columns), so both GPUs
  compute every block at the same time and only exchange the block's partial outputs. Nothing is streamed per step.
* Qwen3-VL-32B (GGUF Q4_K_M) and the VAEs are pinned in RAM and run by MMGP on GPU 1, unloaded before denoising.
"""
import copy
import faulthandler
import functools
import gc
import glob
import json
import math
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

warnings.filterwarnings("ignore")
faulthandler.enable(file=sys.__stderr__, all_threads=True)  # a hard crash (segfault, abort) prints where it happened
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

WAN2GP_DIR = os.path.abspath("../Wan2GP")
WORK_DIR = os.path.dirname(WAN2GP_DIR)
OUTPUT_DIR = os.path.join(WORK_DIR, "outputs")
MODELS_DIR = os.environ.get("H3_MODELS_DIR", "/kaggle/tmp/h3_models")
LORA_DIR = os.environ.get("H3_LORA_DIR", "/kaggle/tmp/h3_loras")
ZIP_PROJECT_DIR = os.path.join(WORK_DIR, "manifest_projects")
VARIANT = os.environ.get("H3_VARIANT", "pdmd")
DIT_FILE = os.environ.get("H3_DIT_FILE", "MiniMax-H3-FL2VA-pruned_rank8_int8_convrot.safetensors")
TE_FILE = os.environ.get("H3_TE_FILE", "qwen3vl-32B-MiniMax-H3-Q4_K_M.gguf")
TURBO_LORA_FILE = os.environ.get("H3_TURBO_LORA", "lora_model_0.safetensors" if VARIANT == "pdmd" else "")  # distillation LoRA
PIN_TEXT_ENCODER = os.environ.get("H3_PIN_TE", "1") == "1"
SERVER_PORT = 7860
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LORA_DIR, exist_ok=True)
os.makedirs(ZIP_PROJECT_DIR, exist_ok=True)
os.environ.setdefault("GRADIO_TEMP_DIR", os.path.join(OUTPUT_DIR, ".gradio_cache"))
if WAN2GP_DIR not in sys.path:
    sys.path.insert(0, WAN2GP_DIR)
os.chdir(WAN2GP_DIR)

import numpy as np
import psutil
import torch
from PIL import Image


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
    WATCHED = {"decode": 120, "stitch": 120, "audio": 120, "save": 120}  # phase -> seconds before a stack dump

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
            self.out.write(f"  ❌ {p['label']} failed after {_fmt_secs(time.time() - p['t0'])}\n")
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


if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
    raise SystemExit("❌ This notebook needs Kaggle's 'GPU T4 x2' accelerator (Settings → Accelerator → GPU T4 x2).")

DIT_DEV = torch.device("cuda", 0)   # first half of the DiT blocks
MAIN_DEV = torch.device("cuda", 1)  # second half + text encoder + VAEs + sampler state
torch.cuda.set_device(MAIN_DEV)     # Wan2GP / MMGP use the current device for "cuda"

torch.backends.cuda.enable_flash_sdp(False)       # Turing (SM 7.5) has no flash kernel
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)
torch.backends.cudnn.benchmark = False

_props = torch.cuda.get_device_properties(0)
print("=" * 70)
print("🎬 MiniMax H3 Turbo INT8 - Video + Audio Generator | AIQUEST Academy")
print(f"⚡ GPUs: 2 x {_props.name} ({_gb(_props.total_memory)} each, SM {_props.major}.{_props.minor}) | "
      f"RAM: {_gb(psutil.virtual_memory().total)} ({_gb(psutil.virtual_memory().available)} free)")
print("=" * 70)
sys.stdout.flush()

# Turing has no BF16 convolution kernels. Convolutions that reach BF16 (the Qwen3-VL patch embedding) run in FP32.
if _props.major < 8:
    def _conv_fp32_on_turing(original):
        def forward(self, input, weight, bias):
            if input.is_cuda and input.dtype == torch.bfloat16:
                bias32 = None if bias is None else bias.float()
                return original(self, input.float(), weight.float(), bias32).to(torch.bfloat16)
            return original(self, input, weight, bias)
        return forward
    for _conv in (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d):
        _conv._conv_forward = _conv_fp32_on_turing(_conv._conv_forward)

# ==== Wan2GP runtime: quantization handlers + INT8 kernels (what wgp.py does at import) ====
from mmgp import offload, quant_router

quant_router.unregister_handler(".fp8_quanto_bridge")
for _handler in ("shared.qtypes.int8_convrot", "shared.qtypes.gguf"):  # DiT + VAE decoder, Qwen3-VL GGUF
    quant_router.register_handler(_handler)
from shared.qtypes import gguf as _gguf_handler

quant_router.register_file_extension("gguf", _gguf_handler)

from shared.kernels import int8_backend

try:
    int8_backend.configure("auto", 1)
except Exception as e:
    print(f"⚠️ INT8 kernels unavailable, using the PyTorch path: {e}")
# Kitchen splits each INT8 GEMM into row tiles that fit a 16 MB scratch buffer, which on a T4 means 64-row GEMMs
# for the 28672-wide FFN. 128 MB gives ~700-row tiles: far better tensor-core use for ~100 MB of extra scratch.
int8_backend._SCRATCH_BYTES = 128 * 1024 * 1024
INT8_LABEL = {"kitchen": "Comfy Kitchen INT8 tensor cores", "triton": "Triton INT8 kernels"}.get(
    int8_backend._backend, "PyTorch INT8 fallback (slow)")
offload.shared_state["_attention"] = "sdpa"

from shared.utils import files_locator as fl

fl.set_checkpoints_paths([MODELS_DIR, LORA_DIR, "ckpts", "."])

import models.minimax_h3.minimax_h3_main as h3_main
from models.minimax_h3 import pipeline as h3_pipeline
from models.minimax_h3.minimax_h3_handler import family_handler
from models.minimax_h3.minimax_h3_main import VIDEO_VAE_INT8_FILE

# Two-phase (low-res + upscale) generation is not used: skip loading the 0.7 GB latent upscaler.
h3_main._load_latent_upscaler = lambda *args, **kwargs: None

# ComfyUI-native pruned checkpoints also carry the AdaLN basis / mean they were compressed with. Wan2GP's model has
# no such tensors (it derives LoRA conversions from its own canonical maps), so drop them before loading.
_orig_strip_wrappers = h3_main._strip_wrappers


def _strip_wrappers(state_dict, *args, **kwargs):
    for key in [k for k in state_dict if k.endswith(("adaln_basis", "adaln_mean"))]:
        state_dict.pop(key)
    return _orig_strip_wrappers(state_dict, *args, **kwargs)


h3_main._strip_wrappers = _strip_wrappers

# ==== Text encoder compute dtype ====
# Without llama.cpp kernels (no wheel for Kaggle's Python), each GGUF linear is dequantized to a dense weight and run
# as a normal matmul. Turing has no BF16 tensor cores, so a BF16 GEMM can be slower than plain FP32. Measure both on
# the real layer shape and keep the faster one (FP32 is also the more precise of the two).
def _gemm_seconds(dtype, m=512, k=5120, n=25600, reps=3):
    a = torch.randn(m, k, device=MAIN_DEV, dtype=dtype)
    b = torch.randn(n, k, device=MAIN_DEV, dtype=dtype)
    torch.nn.functional.linear(a, b)
    torch.cuda.synchronize(MAIN_DEV)
    t = time.time()
    for _ in range(reps):
        torch.nn.functional.linear(a, b)
    torch.cuda.synchronize(MAIN_DEV)
    return (time.time() - t) / reps


try:
    _bf16_s, _fp32_s = _gemm_seconds(torch.bfloat16), _gemm_seconds(torch.float32)
    TE_DTYPE = torch.float32 if _fp32_s < _bf16_s else torch.bfloat16
    print(f"📏 Text encoder GEMM on {_props.name}: BF16 {_bf16_s * 1000:.0f} ms, FP32 {_fp32_s * 1000:.0f} ms "
          f"-> using {'FP32' if TE_DTYPE == torch.float32 else 'BF16'}")
except Exception as e:
    TE_DTYPE = torch.bfloat16
    print(f"⚠️ GEMM benchmark failed ({e}), text encoder stays in BF16")
torch.cuda.empty_cache()
_orig_load_text_encoder = h3_main._load_text_encoder
h3_main._load_text_encoder = lambda filename, dtype: _orig_load_text_encoder(filename, TE_DTYPE)

# ==== Load the model ====
VARIANTS = {  # label, Wan2GP model type, Reference → Video support, default sampler, default steps
    "pdmd": ("PDMD 4-step (base H3 + PDMD LoRA)", "minimax_h3_fl2va_pruned", False, "euler", 4),
    "dmad": ("DMAD 4-step (base H3 + DMAD LoRA)", "minimax_h3_fl2va_pruned", False, "euler", 4),
    "fused": ("MATLOWAI fused turbo (Ref2VA)", "minimax_h3_ref2va_pruned", True, "res_multistep", 4),  # Ref2VA AdaLN table (verified)
    "official": ("Ref2VA + lightx2v 8-step LoRA", "minimax_h3_ref2va_pruned", True, "res_multistep", 8),
}
VARIANT_LABEL, MODEL_TYPE, REFERENCES_SUPPORTED, DEFAULT_SAMPLER, DEFAULT_STEPS = VARIANTS[VARIANT]
print(f"\n📦 Loading MiniMax H3 {VARIANT_LABEL} INT8 ConvRot + Qwen3-VL-32B GGUF...")
sys.stdout.flush()
_t_load = time.time()
BAR.start("load", "Reading checkpoints")
model_def = family_handler.query_model_def(MODEL_TYPE, {})
model_def["qkv_splitting"] = False             # keep the fused QKV GEMM: bigger tiles, and LoRAs map 1:1
model_def["video_vae_file"] = VIDEO_VAE_INT8_FILE  # INT8 ConvRot VAE decoder: half the VRAM, Kitchen GEMMs
pipeline, _ = family_handler.load_model(
    model_filename=[os.path.join(MODELS_DIR, DIT_FILE)],
    model_type=MODEL_TYPE,
    base_model_type=MODEL_TYPE,
    model_def=model_def,
    dtype=torch.bfloat16,
    VAE_dtype=torch.float32,
    text_encoder_filename=os.path.join(MODELS_DIR, "Qwen3-VL-32B-Instruct", TE_FILE),
)
transformer = pipeline.transformer
transformer.cache = None

# Modules H3 pins to FP32 (patch projections, output heads, AdaLN curve projections) are converted here,
# since this transformer is not handed to MMGP's profile (which would otherwise apply the locks).
for _module in transformer.modules():
    if hasattr(_module, "_lock_type") and _module._lock_type == torch.float32:
        _module.to(torch.float32)

# ==== Tensor-parallel DiT: each block is split in half across the two T4s ====
# Within a block, GPU 0 computes attention heads 0-27 and FFN columns 0-7167, GPU 1 the other half. Both start from
# the same residual stream (kept as a replica on each GPU), and after attention and after the FFN each GPU sends its
# partial output projection to the other, so both GPUs hold the full sum. The weight slices follow the ConvRot
# layout exactly: QKV / FFN-up are split by output rows (Q, K, V are grouped), output / FFN-down by input columns on
# 256-wide rotation-group boundaries, so every INT8 GEMM computes precisely its share of the full product.
from models.minimax_h3.transformer import MLP as H3MLP, Attention as H3Attention, _gated_residual, _norm_modulate, _take
from shared.qtypes.int8_convrot import _rotate_activation

DEVICES = (DIT_DEV, MAIN_DEV)
_tp_cache = {"rope": None, "rope_copy": None, "temb": None, "temb_copy": None, "src": None, "replica": None}


class Int8Shard(torch.nn.Module):
    """One GPU's slice of an INT8 ConvRot linear (a set of output rows, or a range of input columns)."""

    def __init__(self, linear, device, rows=None, cols=None):
        super().__init__()
        data, scale = linear.weight._data, linear.weight._scale.reshape(-1)
        if rows is not None:
            data, scale = data.index_select(0, rows), scale.index_select(0, rows)
        if cols is not None:
            data = data[:, cols]
        # copy=True: own storage even on the same device (never a view into the memory-mapped checkpoint)
        self.register_buffer("qdata", data.contiguous().to(device, copy=True), persistent=False)
        self.register_buffer("qscale", scale.float().contiguous().to(device, copy=True), persistent=False)
        self.group = int(getattr(linear, "_convrot_group_size", 0) or 0)
        self.out_features, self.in_features = self.qdata.shape

    def forward(self, x):
        x2 = x.reshape(-1, self.in_features)
        if int8_backend.kitchen_enabled() and self.group == 256 and x2.is_cuda:
            out = int8_backend._linear_impl(x2, self.qdata, self.qscale, None, True)
        else:  # reference path (no Kitchen): rotate, dequantize, matmul
            weight = self.qdata.to(x2.dtype) * self.qscale.to(x2.dtype)[:, None]
            out = torch.nn.functional.linear(_rotate_activation(x2, self.group), weight)
        return out.reshape(*x.shape[:-1], self.out_features)


def _copy_to(value, dev, slot):
    """Per-forward copies of temb / rope for GPU 0 (they are the same tensor object for all 50 blocks)."""
    if value is None:
        return None
    first = value[0] if isinstance(value, tuple) else value
    if first.device == dev:
        return value
    if _tp_cache[slot] is not first:
        _tp_cache[slot] = first
        _tp_cache[slot + "_copy"] = tuple(v.to(dev) for v in value) if isinstance(value, tuple) else value.to(dev)
    return _tp_cache[slot + "_copy"]


def _all_reduce_residual(hiddens, partials, gates, segments):
    """hidden += gate * (partial_0 + partial_1) on both GPUs."""
    incoming = [partials[1].to(DEVICES[0], non_blocking=True, copy=True),
                partials[0].to(DEVICES[1], non_blocking=True, copy=True)]
    out = []
    for s, dev in enumerate(DEVICES):
        with torch.cuda.device(dev):
            branch = partials[s].add_(incoming[s])
            out.append(_gated_residual([hiddens[s]], [gates[s]], [branch], segments))
    return out


# ==== Long-video mode: the same block math on 4096-row slices ====
# A normal block builds several full-length intermediates per GPU (modulated input, fused QKV, BF16 and FP16 copies of
# Q/K/V, the output-projection input, the exchanged partial sums): roughly 75-110 KB per token. For long or 720p clips
# that does not fit next to half of the model, so in this mode only what attention needs at full length is kept (the
# residual stream and FP16 Q/K/V, ~40 KB per token). Norms, projections, RoPE, the FFN and the GPU exchange run on
# 8192-row slices (~0.4 GB of temporaries). Same arithmetic, a little more Python per block; attention dominates long clips either way.
import models.minimax_h3.transformer as _h3_transformer

LONG_VIDEO = {"on": False, "rows": 8192}


def _row_chunks(n):
    step = LONG_VIDEO["rows"]
    return [(a, min(n, a + step)) for a in range(0, n, step)]


def _slice_segments(segments, a, b):
    """(start, stop, modality row) segments of the packed sequence, clipped to rows [a, b) and shifted to 0."""
    return [(max(s0, a) - a, min(s1, b) - a, row) for s0, s1, row in segments if min(s1, b) > max(s0, a)]


def _apply_rope(query, key, rope):
    """H3's rotary embedding on [1, n, heads, dim] tensors, in place (Attention.forward's unfused path)."""
    pairs = rope.shape[-2]
    cosine, sine = rope[..., 0], rope[..., 1]
    scratch = torch.empty_like(query[..., :pairs])
    for tensor in (query, key):
        first, second = tensor[..., :pairs], tensor[..., pairs:2 * pairs]
        scratch.copy_(first)
        first.mul_(cosine).addcmul_(second, sine, value=-1)
        second.mul_(cosine).addcmul_(scratch, sine)


def _sliced_attention(shard, hidden, mod, segments, rope):
    attn = shard.attn
    rows, heads, dim = hidden.shape[0], attn.heads, attn.head_dim
    inner = heads * dim
    if isinstance(rope, tuple):
        rope = rope[0]
    q, k, v = (torch.empty(1, rows, heads, dim, device=hidden.device, dtype=torch.float16) for _ in range(3))
    for a, b in _row_chunks(rows):
        h = _norm_modulate(shard.norm1, hidden[a:b], [mod[0], mod[1]], _slice_segments(segments, a, b))
        qkv = attn.qkv_proj(h)
        del h
        query, key, value = (t.view(1, b - a, heads, dim) for t in qkv.split(inner, dim=-1))
        query, key = attn.q_norm(query), attn.k_norm(key)
        if rope is not None:
            _apply_rope(query, key, rope[:, a:b])
        q[:, a:b].copy_(query)
        k[:, a:b].copy_(key)
        v[:, a:b].copy_(value)
        del qkv, query, key, value
    qkv_list = [q, k, v]
    del q, k, v
    out = _h3_transformer.pay_attention(qkv_list, recycle_q=True)[0]  # [rows, heads, dim] FP16
    partial = torch.empty_like(hidden)
    for a, b in _row_chunks(rows):
        partial[a:b] = attn.out_proj(out[a:b].reshape(b - a, inner).to(hidden.dtype))
    return partial


def _sliced_mlp(shard, hidden, mod, segments):
    partial = torch.empty_like(hidden)
    for a, b in _row_chunks(hidden.shape[0]):
        h = _norm_modulate(shard.norm2, hidden[a:b], [mod[3], mod[4]], _slice_segments(segments, a, b))
        partial[a:b] = shard.mlp([h])
    return partial


def _all_reduce_sliced(hiddens, partials, gates, segments):
    """_all_reduce_residual without full-length copies: the partial sums cross between GPUs slice by slice."""
    for a, b in _row_chunks(hiddens[0].shape[0]):
        incoming = [partials[1][a:b].to(DEVICES[0], non_blocking=True, copy=True),
                    partials[0][a:b].to(DEVICES[1], non_blocking=True, copy=True)]
        for s, dev in enumerate(DEVICES):
            with torch.cuda.device(dev):
                partials[s][a:b].add_(incoming[s])
        del incoming
    out = []
    for s, dev in enumerate(DEVICES):
        with torch.cuda.device(dev):
            out.append(_gated_residual([hiddens[s]], [gates[s]], [partials[s]], segments))
    return out


class TPBlock(torch.nn.Module):
    def __init__(self, block):
        super().__init__()
        attn, mlp = block.attn, block.mlp
        heads, head_dim = attn.heads, attn.head_dim
        inner, ffn, hidden = heads * head_dim, mlp.ffn, mlp.hidden
        half_inner, half_ffn = inner // 2, ffn // 2
        self.splits = []
        shards = []
        for s, dev in enumerate(DEVICES):
            h = torch.arange(s * half_inner, (s + 1) * half_inner)
            f = torch.arange(s * half_ffn, (s + 1) * half_ffn)
            qkv_rows = torch.cat((h, h + inner, h + 2 * inner))
            fc1_rows = torch.cat((f, f + ffn))
            out_cols = slice(s * half_inner, (s + 1) * half_inner)
            fc2_cols = slice(s * half_ffn, (s + 1) * half_ffn)
            self.splits.append({"qkv": qkv_rows, "fc1": fc1_rows, "out": out_cols, "fc2": fc2_cols})
            with torch.device("meta"):
                sattn = H3Attention(hidden, heads // 2, head_dim, attn.q_norm.eps, dtype=torch.bfloat16)
                smlp = H3MLP(hidden, half_ffn, mlp.chunk_size, dtype=torch.bfloat16)
            sattn.qkv_proj = Int8Shard(attn.qkv_proj, dev, rows=qkv_rows)
            sattn.out_proj = Int8Shard(attn.out_proj, dev, cols=out_cols)
            sattn.q_norm = copy.deepcopy(attn.q_norm).to(dev)
            sattn.k_norm = copy.deepcopy(attn.k_norm).to(dev)
            smlp.fc1 = Int8Shard(mlp.fc1, dev, rows=fc1_rows)
            smlp.fc2 = Int8Shard(mlp.fc2, dev, cols=fc2_cols)
            shard = torch.nn.Module()
            shard.attn, shard.mlp = sattn, smlp
            shard.norm1 = copy.deepcopy(block.norm1).to(dev)
            shard.norm2 = copy.deepcopy(block.norm2).to(dev)
            shard.adaln_proj = copy.deepcopy(block.adaln_proj).to(dev)
            shards.append(shard)
        self.shards = torch.nn.ModuleList(shards)
        self.dims = {"qkv": (hidden, 3 * inner), "out": (inner, hidden), "fc1": (hidden, 2 * ffn), "fc2": (ffn, hidden)}

    @property
    def attn(self):  # H3's forward checks block.attn for q_proj / vdn
        return self.shards[1].attn

    def forward(self, x_list, temb, segments, rope, residual_signature_elements=0):
        if residual_signature_elements:
            raise RuntimeError("First Block Cache is not supported by the tensor-parallel DiT")
        hidden = _take(x_list)
        hidden = hidden if hidden.device == MAIN_DEV else hidden.to(MAIN_DEV)
        replica = _tp_cache["replica"] if _tp_cache["src"] is hidden else hidden.to(DIT_DEV, copy=True)
        hiddens = [replica, hidden]
        tembs = [_copy_to(temb, DIT_DEV, "temb"), temb]
        ropes = [_copy_to(rope, DIT_DEV, "rope"), rope]
        if LONG_VIDEO["on"]:
            return self._forward_sliced(hiddens, tembs, ropes, segments)
        mods, partials = [], []
        for s, dev in enumerate(DEVICES):  # kernels are queued on both GPUs before either is waited for
            shard = self.shards[s]
            with torch.cuda.device(dev):
                mod = shard.adaln_proj(tembs[s])
                mods.append(mod)
                h = _norm_modulate(shard.norm1, hiddens[s], [mod[0], mod[1]], segments)
                partials.append(shard.attn([h], rope=ropes[s]))
        hiddens = _all_reduce_residual(hiddens, partials, [m[2] for m in mods], segments)
        partials = []
        for s, dev in enumerate(DEVICES):
            shard = self.shards[s]
            with torch.cuda.device(dev):
                h = _norm_modulate(shard.norm2, hiddens[s], [mods[s][3], mods[s][4]], segments)
                partials.append(shard.mlp([h]))
        hiddens = _all_reduce_residual(hiddens, partials, [m[5] for m in mods], segments)
        _tp_cache["src"], _tp_cache["replica"] = hiddens[1], hiddens[0]
        return hiddens[1]

    def _forward_sliced(self, hiddens, tembs, ropes, segments):
        mods, partials = [], []
        for s, dev in enumerate(DEVICES):  # kernels are queued on both GPUs before either is waited for
            with torch.cuda.device(dev):
                mods.append(self.shards[s].adaln_proj(tembs[s]))
                partials.append(_sliced_attention(self.shards[s], hiddens[s], mods[s], segments, ropes[s]))
        hiddens = _all_reduce_sliced(hiddens, partials, [m[2] for m in mods], segments)
        partials = []
        for s, dev in enumerate(DEVICES):
            with torch.cuda.device(dev):
                partials.append(_sliced_mlp(self.shards[s], hiddens[s], mods[s], segments))
        hiddens = _all_reduce_sliced(hiddens, partials, [m[5] for m in mods], segments)
        _tp_cache["src"], _tp_cache["replica"] = hiddens[1], hiddens[0]
        return hiddens[1]


NUM_BLOCKS = len(transformer.blocks)
BAR.start("split", "Splitting DiT", NUM_BLOCKS, "blocks")
for _i in range(NUM_BLOCKS):
    transformer.blocks[_i] = TPBlock(transformer.blocks[_i])
    BAR.update(_i + 1)
BAR.finish()
for _name, _child in transformer.named_children():
    if _name not in ("blocks", "token_refiner"):
        _child.to(MAIN_DEV)
for _name, _buffer in list(transformer.named_buffers(recurse=False)):
    transformer._buffers[_name] = _buffer.to(MAIN_DEV)
gc.collect()

# ==== Exact-size pinned RAM ====
# PyTorch's pinned-memory allocator rounds every block up to the next power of two: MMGP's ~210 MB text-encoder
# chunks become 256 MB each (+2.8 GB) and the token refiner's 17 tensors would take 2.25 GB for 1.44 GB. Page-locking
# ordinary RAM with cudaHostRegister pins exactly the bytes asked for (same fast async copies to the GPU).
_HOST_REGISTERED = []  # registered RAM stays alive (and pinned) for the whole session
PINNED_BYTES = {}


def _dont_fork(ptr, nbytes):
    """madvise(MADV_DONTFORK): a fork() must not duplicate pinned RAM. Linux cannot share pinned pages copy-on-write,
    so it copies them into the child right away: forking this process (subprocess with preexec_fn does) would need
    another ~13 GB at once and gets the engine killed. CUDA marks its own cudaHostAlloc blocks this way."""
    try:
        import ctypes
        page = os.sysconf("SC_PAGE_SIZE")
        start = -(-ptr // page) * page
        end = (ptr + nbytes) // page * page
        if end > start:
            ctypes.CDLL("libc.so.6").madvise(ctypes.c_void_p(start), ctypes.c_size_t(end - start), 10)  # MADV_DONTFORK
    except Exception:
        pass


def pinned_empty_exact(nbytes):
    t = torch.empty(nbytes, dtype=torch.uint8)
    torch.cuda.check_error(torch.cuda.cudart().cudaHostRegister(t.data_ptr(), nbytes, 0))
    _dont_fork(t.data_ptr(), nbytes)
    _HOST_REGISTERED.append(t)
    return t


_torch_empty = torch.empty


def _empty_exact_pinned(*size, pin_memory=False, **kwargs):
    """torch.empty for MMGP's pinning: its non-power-of-two uint8 chunks are registered at their exact size."""
    n = size[0] if len(size) == 1 and isinstance(size[0], int) else 0
    if pin_memory and n > 0 and n & (n - 1) and kwargs.get("dtype") == torch.uint8:
        try:
            return pinned_empty_exact(n)
        except Exception:
            pass
    return _torch_empty(*size, pin_memory=pin_memory, **kwargs)


# The 1.4 GiB token refiner runs once per prompt: keep it in pinned RAM and swap it onto GPU 1 only for that.
_refiner = transformer.token_refiner
_ref_list = list(_refiner.parameters())
_ref_offsets, _ref_total = [], 0
for _p in _ref_list:
    _ref_offsets.append(_ref_total)
    _ref_total += -(-_p.numel() * _p.element_size() // 64) * 64
try:
    _ref_buf = pinned_empty_exact(_ref_total)
    _refiner_params = []
    for _p, _off in zip(_ref_list, _ref_offsets):
        _cpu = _ref_buf[_off:_off + _p.numel() * _p.element_size()].view(_p.dtype).view(_p.shape)
        _cpu.copy_(_p.data)
        _refiner_params.append((_p, _cpu))
except Exception as e:  # no cudaHostRegister: PyTorch's (rounded) pinned blocks
    print(f"⚠️ Exact pinning unavailable ({e}); using PyTorch pinned memory")
    _refiner_params = [(p, p.data.to("cpu").pin_memory()) for p in _ref_list]
PINNED_BYTES["token refiner"] = _ref_total
for _p, _cpu in _refiner_params:
    _p.data = _cpu
_orig_preprocess_text = transformer.preprocess_text_embeds


def _preprocess_text_embeds(text_states):
    if text_states.shape[-1] == transformer.hidden_size:
        return text_states
    for p, cpu in _refiner_params:
        p.data = cpu.to(MAIN_DEV, non_blocking=True)
    try:
        return _orig_preprocess_text(text_states.to(MAIN_DEV))
    finally:
        for p, cpu in _refiner_params:
            p.data = cpu


transformer.preprocess_text_embeds = _preprocess_text_embeds

# ==== Text encoder + VAEs: MMGP on GPU 1, unloaded before denoising ====
pipeline.audio_vae._convertWeightsFloatTo = None  # keep the audio VAE in FP32 (Turing has no BF16 convs)
aux_pipe = {
    "text_encoder": pipeline.text_encoder.language_model,
    "vision_encoder": pipeline.text_encoder.visual,
    "vae": pipeline.video_decoder,
    "video_encoder": pipeline.video_encoder,
    "audio_vae": pipeline.audio_vae,
}
pipeline.video_decoder._budget = 0  # decode with the whole VAE resident (it is called once per clip and tile)
if TE_DTYPE == torch.float32:
    pipeline.text_encoder.language_model._convertWeightsFloatTo = None
    pipeline.text_encoder.visual._convertWeightsFloatTo = None
# The text encoder (13.6 GB) and the video VAE decoder (2.3 GB) are pinned in RAM. Kaggle's disk reads at ~180 MB/s,
# so a memory-mapped copy that has dropped out of the page cache costs ~15 s per video before decoding can even
# start; from pinned RAM the same weights reach the GPU in well under a second. The small audio VAE and image
# encoder stay as they are (a few seconds at most).
_pin = (["text_encoder"] if PIN_TEXT_ENCODER else []) + ["vae"]
BAR.start("pin", "Pinning text encoder + video decoder in RAM (one-time)")
try:
    torch.empty = _empty_exact_pinned
    try:
        aux_offload = offload.profile(aux_pipe, profile_no=4, quantizeTransformer=False, convertWeightsFloatTo=torch.bfloat16,
                                      pinnedMemory=_pin, asyncTransfers=True, verboseLevel=-1,
                                      budgets={"text_encoder": 1500, "vision_encoder": 0, "*": 3000})
    finally:
        torch.empty = _torch_empty
except Exception as e:
    if not _pin:
        raise
    print(f"⚠️ Pinning failed ({e}); text encoder streams from regular RAM")
    aux_offload = offload.profile(aux_pipe, profile_no=4, quantizeTransformer=False, convertWeightsFloatTo=torch.bfloat16,
                                  pinnedMemory=False, asyncTransfers=True, verboseLevel=-1,
                                  budgets={"text_encoder": 1500, "vision_encoder": 0, "*": 3000})
pipeline.set_offload_handoff(aux_offload, None)  # pipeline unloads these before handing the GPU to the DiT
BAR.finish()
for _key, _label in (("text_encoder", "text encoder"), ("vae", "video decoder")):
    PINNED_BYTES[_label] = getattr(aux_pipe[_key], "_pinned_bytes", 0)
if hasattr(torch._C, "_host_emptyCache"):
    torch._C._host_emptyCache()  # MMGP's temporary probe block

gc.collect()
for _dev in (DIT_DEV, MAIN_DEV):
    with torch.cuda.device(_dev):
        torch.cuda.empty_cache()
_free0, _free1 = torch.cuda.mem_get_info(0)[0], torch.cuda.mem_get_info(1)[0]
print(f"✅ Loaded in {time.time() - _t_load:.0f}s | {INT8_LABEL}")
print(f"   GPU 0: half of all {NUM_BLOCKS} DiT blocks ({_gb(torch.cuda.memory_allocated(0))}), {_gb(_free0)} free")
print(f"   GPU 1: other half + heads ({_gb(torch.cuda.memory_allocated(1))}), {_gb(_free1)} free")
print(f"   RAM in use: {_ram_in_use()} | pinned " + " + ".join(f"{k} {_gb(v)}" for k, v in PINNED_BYTES.items() if v))
try:
    _a = torch.empty(64 * 2**20, dtype=torch.uint8, device=DIT_DEV)
    _b = torch.empty(64 * 2**20, dtype=torch.uint8, device=MAIN_DEV)
    _a.to(MAIN_DEV); _b.to(DIT_DEV); torch.cuda.synchronize(0); torch.cuda.synchronize(1)
    _t = time.time()
    for _ in range(4):
        _a.to(MAIN_DEV); _b.to(DIT_DEV)
    torch.cuda.synchronize(0); torch.cuda.synchronize(1)
    print(f"   GPU<->GPU exchange: {8 * 64 / 1024 / (time.time() - _t):.1f} GB/s "
          f"(peer access: {torch.cuda.can_device_access_peer(0, 1)})")
    del _a, _b
except Exception as e:
    print(f"   GPU<->GPU exchange benchmark skipped: {e}")
sys.stdout.flush()

# ==== LoRA runtime ====
# Applied as forward hooks on the resident (two-GPU) transformer: out += strength * (x @ A^T) @ B^T, written in place
# with addmm_ so no output-sized temporary is created. Keys go through Wan2GP's own H3 converter, which maps
# diffusers / kohya names and re-projects AdaLN adapters onto the pruned 8-wide AdaLN curves.
LORA_CATALOG = {
    "🧑 Realism People (fal)": {
        "repo": "fal/MiniMax-H3-Realism-People-LoRA", "file": "h3-realism-people-t2v-i2v-r2v.safetensors",
        "scale": 1.0, "trigger": "r34l1sm", "info": "Natural skin, faces, expressions and film lighting for people shots."},
    "🎥 Camera Motion": {
        "repo": "Jojocodex/minimax-h3-Camera-Motion-lora", "file": "camera_motion_h3_lora_v1_3000_pruned.safetensors",
        "scale": 0.9, "trigger": "camera motion", "info": "Push-in, pull-back, orbit, aerial, handheld... describe the move after the trigger."},
    "🔄 360° Orbit (same image as first + last frame)": {
        "repo": "pablodawson/MiniMax-H3-360-Orbit-LoRA", "file": "minimax_h3_flf2v_lora_v1.safetensors",
        "scale": 1.0, "trigger": "One frozen instant. Only the camera moves. In a continuous 360 orbit.",
        "info": "Put the SAME image as start and end frame: the camera orbits the frozen scene and lands back on it."},
    "⚔️ Weapon Combat (BUNNY)": {
        "repo": "JOKER141/MiniMax-H3-Weapon-Combat-LoRA", "file": "Bunny_weapon_combatV1.safetensors",
        "scale": 1.0, "trigger": "BUNNY", "info": "Weapon swings, blocks, parries and dynamic fight choreography."},
    "🧱 Spatial & Physics": {
        "repo": "Jojocodex/minimax-h3-spatial-physics-lora", "file": "wushu_spatial_physics_clean_3000_pruned.safetensors",
        "scale": 0.4, "trigger": "", "info": "More physical collisions, bounces and gravity. Keep it at 0.3-0.5."},
    "🗣️ Natural Face & Speech": {
        "repo": "vpakarinen/natural-face-speech-h3-lora", "file": "natural_face_speech_h3_lora_v2_500.safetensors",
        "scale": 0.7, "trigger": "", "info": "Talking heads: facial muscle motion and clear speech. Put the line in 'Spoken line'."},
    "🎞️ Studio 1939 Animation": {
        "repo": "lovis93/studio-1939-old-animation-lora-minimax-h3", "file": "studio1939-light.safetensors",
        "scale": 1.0, "trigger": "gulliv3r,", "info": "Hand-painted golden-age animation look."},
    "🌸 Anime Motion": {
        "repo": "prithivMLmods/MiniMax-H3-I2V-Anime-Motion-LoRA", "file": "MiniMax-H3-I2V-Anime-Motion-LoRA-1400.safetensors",
        "scale": 1.0, "trigger": "Anime-Motion", "info": "Subtle looping anime motion, best with a start image."},
    "📼 VHS Tape (1980s)": {
        "repo": "KennethFal/vh5tape-vhs-lora-minimax-h3", "file": "vh5tape.safetensors",
        "scale": 1.0, "trigger": "vh5tape, worn VHS tape recording with visible tape damage.",
        "info": "Worn VHS look and sound. Damage tiers: lightly worn / worn / badly damaged."},
}
NO_LORA = "None"
LORA_CHOICES = [NO_LORA] + list(LORA_CATALOG)
_LORA_PREFIXES = ("model.diffusion_model.", "diffusion_model.", "base_model.model.", "transformer.", "model.")
_LORA_SUFFIXES = {".default.weight": ".weight", ".lora.A.weight": ".lora_A.weight", ".lora.B.weight": ".lora_B.weight",
                  ".lora.down.weight": ".lora_down.weight", ".lora.up.weight": ".lora_up.weight"}

LORA_ROWS = 8192


class LoraRuntime:
    """LoRAs as forward hooks. Each original layer name maps to one or more target modules (the two halves of a
    tensor-parallel layer), and every target gets the matching slice of the LoRA: B rows for row-split layers, A columns
    for column-split layers (whose partial outputs are summed across GPUs, so a bias delta goes to one half only)."""

    def __init__(self, model, model_type):
        self.model = model
        self.model_type = model_type
        self.targets = {}  # name -> (in_features, out_features, [(module, device, dtype, kind, index)])
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) and not name.startswith("blocks."):
                dev = MAIN_DEV if name.startswith("token_refiner") or not module.weight.is_cuda else module.weight.device
                dtype = torch.float32 if module.weight.dtype == torch.float32 else torch.bfloat16
                self.targets[name] = (module.in_features, module.out_features, [(module, dev, dtype, None, None)])
        for i, block in enumerate(model.blocks):
            for short, attr, kind in (("qkv", "attn.qkv_proj", "rows"), ("out", "attn.out_proj", "cols"),
                                      ("fc1", "mlp.fc1", "rows"), ("fc2", "mlp.fc2", "cols")):
                parts = [(functools.reduce(getattr, attr.split("."), block.shards[s]), DEVICES[s], torch.bfloat16, kind,
                          block.splits[s][short]) for s in range(2)]
                self.targets[f"blocks.{i}.{attr}"] = (*block.dims[short], parts)
            adaln = [(block.shards[s].adaln_proj.linear, DEVICES[s], torch.float32, None, None) for s in range(2)]
            first = adaln[0][0]
            self.targets[f"blocks.{i}.adaln_proj.linear"] = (first.in_features, first.out_features, adaln)
        self.loaded = {}    # key -> [(module, [(A, B, out_start, out_end)], alpha_scale, diff_b)]
        self.strength = {}  # key -> current strength
        self.active = {}    # id(module) -> [(key, segments, alpha_scale, diff_b)]
        self.handles = {}   # id(module) -> hook handle

    def _normalize(self, state_dict):
        out = {}
        for key, value in state_dict.items():
            for prefix in _LORA_PREFIXES:
                if key.startswith(prefix):
                    key = key[len(prefix):]
                    break
            for src, dst in _LORA_SUFFIXES.items():
                if key.endswith(src):
                    key = key[:-len(src)] + dst
                    break
            out[key] = value
        return self.model.preprocess_loras(self.model_type, out)

    def _iter_groups(self, path):
        """(layer name, {"A", "B", "alpha", "diff_b"}) for every layer of a LoRA file, read one block at a time: a
        rank-128 LoRA is 1.4 GB, and Wan2GP's converter makes several copies of what it is given, which a fully loaded
        file does not leave room for next to the pinned text encoder."""
        from safetensors import safe_open

        with safe_open(path, framework="pt", device="cpu") as f:
            batches = {}
            for k in f.keys():
                m = re.search(r"blocks[._](\d+)[._]", k)
                batches.setdefault(("refiner" if "refiner" in k else "block", int(m.group(1))) if m else ("other", 0), []).append(k)
            for keys in batches.values():
                groups = {}
                for k, v in self._normalize({k: f.get_tensor(k) for k in keys}).items():
                    for down, up in (("lora_A.weight", "lora_B.weight"), ("lora_down.weight", "lora_up.weight")):
                        if k.endswith("." + down):
                            groups.setdefault(k[:-len(down) - 1], {})["A"] = v
                        elif k.endswith("." + up):
                            groups.setdefault(k[:-len(up) - 1], {})["B"] = v
                    if k.endswith(".alpha"):
                        groups.setdefault(k[:-len(".alpha")], {})["alpha"] = float(v.item() if torch.is_tensor(v) else v)
                    elif k.endswith(".diff_b"):
                        groups.setdefault(k[:-len(".diff_b")], {})["diff_b"] = v
                yield from groups.items()

    def _read_groups(self, path):
        return dict(self._iter_groups(path))

    @staticmethod
    def _segments(A, B):
        """[(A_i, B_i, out_start, out_end)]. Separate q / k / v adapters arrive fused as cat(A) and block_diag(B),
        which is 2/3 zeros: they are split back into their three blocks (a third of the memory and the work)."""
        r3, o3 = A.shape[0], B.shape[0]
        if r3 % 3 == 0 and o3 % 3 == 0 and r3 >= 3:
            r, o = r3 // 3, o3 // 3
            off = [B[i * o:(i + 1) * o, j * r:(j + 1) * r] for i in range(3) for j in range(3) if i != j]
            if all(not bool(t.any()) for t in off):
                return [(A[i * r:(i + 1) * r], B[i * o:(i + 1) * o, i * r:(i + 1) * r], i * o, (i + 1) * o) for i in range(3)]
        return [(A, B, 0, o3)]

    def load(self, key, path):
        adapters, layers, skipped = [], 0, 0
        for name, g in self._iter_groups(path):
            target = self.targets.get(name)
            A, B, diff_b = g.get("A"), g.get("B"), g.get("diff_b")
            if target is None or (A is None) != (B is None) or (A is None and diff_b is None):
                skipped += 1
                continue
            in_f, out_f, parts = target
            if A is not None and (A.shape[1] != in_f or B.shape[0] != out_f or A.shape[0] != B.shape[1]):
                skipped += 1
                continue
            alpha_scale = g["alpha"] / A.shape[0] if A is not None and "alpha" in g else 1.0
            segments = self._segments(A, B) if A is not None else []
            for n, (module, dev, dtype, kind, index) in enumerate(parts):
                segs, d = [], diff_b
                if dtype == torch.bfloat16:  # BF16 layers get their LoRA math in FP16 on the tensor cores (see _hook)
                    dtype = torch.float16
                for a, b, o0, o1 in segments:
                    if kind == "rows":  # this shard computes the output rows `index` (in order)
                        pos = ((index >= o0) & (index < o1)).nonzero().flatten()
                        if not len(pos):
                            continue
                        assert int(pos[-1] - pos[0]) + 1 == len(pos), "non-contiguous LoRA rows in a shard"
                        b, o0, o1 = b.index_select(0, index[pos] - o0), int(pos[0]), int(pos[-1]) + 1
                    elif kind == "cols":
                        a = a[:, index]
                    segs.append((a.to(dev, dtype).contiguous(), b.to(dev, dtype).contiguous(), o0, o1))
                if kind == "rows" and diff_b is not None:
                    d = diff_b.index_select(0, index)
                elif kind == "cols":
                    d = diff_b if n == 0 else None
                adapters.append((module, segs, alpha_scale, None if d is None else d.to(dev, dtype)))
            layers += 1
        if not adapters:
            raise ValueError(f"No layer of this LoRA matches MiniMax H3 ({skipped} groups checked)")
        self.loaded[key] = adapters
        mb = sum(sum(t.numel() * t.element_size() for seg in segs for t in seg[:2]) for _, segs, _, _ in adapters) / 2**20
        print(f"  🎨 LoRA '{key}': {layers} layers ({mb:.0f} MB VRAM){f', {skipped} unmatched groups skipped' if skipped else ''}")

    def _hook(self, module_id):
        def hook(module, args, output):
            adapters = self.active.get(module_id)
            if not adapters:
                return output
            x = args[0]
            x2 = x.reshape(-1, x.shape[-1])
            out2 = output.view(-1, output.shape[-1])
            for key, segments, alpha_scale, diff_b in adapters:
                s = self.strength.get(key, 0.0) * alpha_scale
                if s == 0.0:
                    continue
                if segments and segments[0][0].dtype == torch.float16:
                    # FP16 tensor cores: a rank-128 LoRA on every layer is ~3% of the FLOPs, but in FP32 (no tensor
                    # cores on Turing, and BF16 is slower still) it cost ~30% of each step. The input is scaled by a
                    # power of two before the FP16 cast and the result scaled back, so large activations (FFN-down
                    # input) cannot overflow; FP16's 11-bit mantissa is plenty for a correction term.
                    pre = 1.0 / 512 if x2.shape[-1] > 8192 else 1.0 / 64
                    for start in range(0, x2.shape[0], LORA_ROWS):
                        end = min(x2.shape[0], start + LORA_ROWS)
                        xs = (x2[start:end] * pre).to(torch.float16)
                        for a16, b16, o0, o1 in segments:
                            delta = (xs @ a16.t()) @ b16.t()
                            out2[start:end, o0:o1].add_(delta.to(out2.dtype), alpha=s / pre)
                            del delta
                        del xs
                elif segments:  # FP32 layers (AdaLN curves)
                    for start in range(0, x2.shape[0], LORA_ROWS):
                        end = min(x2.shape[0], start + LORA_ROWS)
                        xf = x2[start:end].float()
                        for a32, b32, o0, o1 in segments:
                            out2[start:end, o0:o1].add_(((xf @ a32.float().t()) @ b32.float().t()).mul_(s).to(out2.dtype))
                        del xf
                if diff_b is not None:
                    out2.add_(diff_b, alpha=s)
            return output
        return hook

    def apply(self, selections):
        """selections: {key: (path, strength)} - loads new LoRAs, frees unused ones, sets strengths."""
        for key in [k for k in self.loaded if k not in selections]:
            del self.loaded[key]
        for key, (path, strength) in selections.items():
            if key not in self.loaded:
                self.load(key, path)
            self.strength[key] = float(strength)
        self.active, modules = {}, {}
        for key, adapters in self.loaded.items():
            for module, segments, alpha_scale, diff_b in adapters:
                self.active.setdefault(id(module), []).append((key, segments, alpha_scale, diff_b))
                modules[id(module)] = module
        for module_id in list(self.handles):
            if module_id not in self.active:
                self.handles.pop(module_id).remove()
        for module_id, module in modules.items():
            if module_id not in self.handles:
                self.handles[module_id] = module.register_forward_hook(self._hook(module_id))
        gc.collect()
        for dev in DEVICES:
            with torch.cuda.device(dev):
                torch.cuda.empty_cache()


lora_runtime = LoraRuntime(transformer, MODEL_TYPE)

# ==== DiT attention backend: PyTorch SDPA or tensor-core GEMM attention, whichever is faster on this GPU ====
# Attention is most of the DiT's work (it grows with the square of the video length: a 10 s clip costs ~3.5x a 5 s
# one). On Turing, PyTorch's memory-efficient SDPA kernel runs well below what cuBLAS FP16 GEMMs reach, so an exact
# chunked attention (Q·Kᵀ GEMM -> softmax -> P·V GEMM, a few hundred query rows at a time) can be faster. Both are
# timed on the real head shape at startup; GEMM is used only if it is faster and matches SDPA. The VAE keeps
# Wan2GP's own attention and 256 px tiles untouched.
import shared.attention as _w2g_attention

_orig_pay_attention = _w2g_attention.pay_attention
ATTENTION = {"dit": "sdpa"}
ATTENTION_CHUNK_BYTES = 640 * 2**20  # logits + probabilities of one chunk


def gemm_attention(q, k, v):
    """Exact softmax attention for [B, L, H, D] FP16 tensors using batched GEMMs over query-row chunks."""
    batch, length, heads, dim = q.shape
    keys = k.shape[1]
    rows = max(64, min(length, ATTENTION_CHUNK_BYTES // (4 * heads * keys)))
    out = torch.empty_like(q)
    q = q * (dim ** -0.5)
    for b in range(batch):
        q_b, k_t, v_b = q[b].transpose(0, 1), k[b].permute(1, 2, 0), v[b].transpose(0, 1)  # strided views, no copies
        for start in range(0, length, rows):
            probs = torch.softmax(torch.matmul(q_b[:, start:start + rows], k_t), dim=-1)
            out[b, start:start + rows] = torch.matmul(probs, v_b).transpose(0, 1)
            del probs
    return out


def _fast_pay_attention(kind):
    def pay_attention(qkv_list, *args, **kwargs):
        if (ATTENTION[kind] != "gemm" or args or kwargs.get("attention_mask") is not None or kwargs.get("causal")
                or kwargs.get("q_lens") is not None or kwargs.get("k_lens") is not None):
            return _orig_pay_attention(qkv_list, *args, **kwargs)
        q, k, v = qkv_list
        qkv_list.clear()
        out_dtype = q.dtype
        q, k, v = [t if t.dtype == torch.float16 else t.to(torch.float16) for t in (q, k, v)]
        return gemm_attention(q, k, v).to(out_dtype)
    return pay_attention


_h3_transformer.pay_attention = _fast_pay_attention("dit")


def _time_it(fn, reps=2):
    fn()
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t) / reps


def benchmark_attention(kind, batch, length, heads, dim):
    try:
        with torch.cuda.device(DIT_DEV):
            q, k, v = (torch.randn(batch, length, heads, dim, device=DIT_DEV, dtype=torch.float16) for _ in range(3))
            sdpa = lambda: _orig_pay_attention([q, k, v], force_attention="sdpa")
            gemm = lambda: gemm_attention(q, k, v)
            ref, got = sdpa().float(), gemm().float()
            err = ((got - ref).norm() / ref.norm()).item()
            t_sdpa, t_gemm = _time_it(sdpa), _time_it(gemm)
            del q, k, v, ref, got
            torch.cuda.empty_cache()
        ATTENTION[kind] = "gemm" if t_gemm < 0.95 * t_sdpa and err < 1e-2 else "sdpa"
        print(f"📏 DiT attention ({length} tokens x {heads} heads): SDPA {t_sdpa * 1000:.0f} ms, "
              f"GEMM {t_gemm * 1000:.0f} ms (diff {err:.1e}) -> using {ATTENTION[kind].upper()}")
    except Exception as e:
        ATTENTION[kind] = "sdpa"
        print(f"⚠️ {kind} attention benchmark failed ({e}); using SDPA")


benchmark_attention("dit", 1, 16384, 28, 128)  # 480p / 5 s, half the heads per GPU

# ==== VAE decode on both GPUs ====
# The video VAE decodes the latent in independent temporal clips (only the cross-fade between neighbours is
# sequential), so the clips are split between the two GPUs: GPU 1 uses the MMGP-managed decoder, GPU 0 a second
# decoder whose weights are copied, only for the duration of the decode, from the first one (no second copy in RAM;
# between videos it holds no weights at all). Tiling, attention and cross-fades are exactly Wan2GP's. On the first video GPU 0 also
# decodes clip 0 and must match GPU 1's result; otherwise decoding falls back to GPU 1 alone.
_vae_b = h3_main._load_video_vae(VIDEO_VAE_INT8_FILE, False)  # same architecture; its own weights are dropped below
_vae_b.encoder = _vae_b.quant_conv = None  # decode-only copy
_vae_b_parts = torch.nn.ModuleDict({"post_quant_conv": _vae_b.post_quant_conv, "decoder": _vae_b.decoder})
_vae_b_slots = []  # (module, "_parameters" | "_buffers", name, source module)
_source_modules = dict(pipeline.video_decoder.named_modules())
for _name, _module in _vae_b_parts.named_modules():
    for _store in ("_parameters", "_buffers"):
        for _key, _value in getattr(_module, _store).items():
            if _value is not None:
                _vae_b_slots.append((_module, _store, _key, _source_modules[_name]))


def _vae_b_attach(device):
    """Copy the first decoder's weights (pinned RAM, or GPU 1 if MMGP has it loaded) to `device`."""
    for module, store, key, source in _vae_b_slots:
        value = getattr(source, store)[key].to(device, non_blocking=True)
        if store == "_parameters":
            value = torch.nn.Parameter(value, requires_grad=False)
        getattr(module, store)[key] = value


def _vae_b_release():
    """Drop every weight of the second decoder. It must not keep references to the first decoder's tensors either:
    MMGP swaps in new GPU tensors when it loads a model, so a shared reference taken while the decoder is on GPU 1
    would keep those 2.3 GB of VRAM alive after MMGP unloads it."""
    for module, store, key, _ in _vae_b_slots:
        getattr(module, store)[key] = None


_vae_b_release()
gc.collect()

# Wan2GP decodes each clip as 256 px tiles stacked into ONE decoder batch: ~12 tiles at 480p (decode already peaks at
# ~13.3 GB of the T4's 14.6), ~28 at 720p, which does not fit next to half of the DiT + the LoRA. The decoder is a ViT
# that treats every tile on its own, so running the tiles in smaller groups gives exactly the same frames. The group
# size is planned per GPU from its free VRAM before each decode (~80 MB per tile measured, 100 MB budgeted), so the
# decoder never has to run out of memory; halving on an out-of-memory error stays only as a last resort.
VAE_TILE_GROUP = {}
VAE_TILE_BYTES = 100 * 2**20


def _tile_grouped(decoder):
    full_forward = decoder.forward

    def forward(hidden_states, *args, **kwargs):
        dev, tiles = hidden_states.device.index, hidden_states.shape[0]
        while True:
            n = min(VAE_TILE_GROUP.get(dev, tiles), tiles)
            try:
                if n >= tiles:
                    return full_forward(hidden_states, *args, **kwargs)
                return torch.cat([full_forward(hidden_states[i:i + n], *args, **kwargs) for i in range(0, tiles, n)])
            except torch.OutOfMemoryError:
                if n <= 1:
                    raise
                VAE_TILE_GROUP[dev] = max(1, n // 2)
                print(f"  ⚙️ VAE: GPU {dev} decodes {VAE_TILE_GROUP[dev]} tiles at a time from now on (out of memory with {n})")
                with torch.cuda.device(hidden_states.device):
                    torch.cuda.synchronize()  # kernels queued before the failed allocation may still use freed blocks
                    gc.collect()
                    torch.cuda.empty_cache()

    decoder.forward = forward


_tile_grouped(pipeline.vae.decoder)
_tile_grouped(_vae_b.decoder)


def _plan_tile_groups():
    """Tiles per decoder call that fit each GPU's free VRAM right now (GPU 1 loads its 2.3 GB decoder on first use)."""
    resident = any(p.is_cuda for p in pipeline.video_decoder.parameters())
    for dev in DEVICES:
        with torch.cuda.device(dev):
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        free = torch.cuda.mem_get_info(dev)[0]
        if dev == MAIN_DEV and not resident:
            free -= PINNED_BYTES.get("video decoder", 0)
        VAE_TILE_GROUP[dev.index] = max(1, int((free - 512 * 2**20) // VAE_TILE_BYTES))


# ==== Drop the checkpoint files from the page cache ====
# Loading reads ~40 GB of checkpoints, and Linux keeps whatever fits of them cached in RAM. That cache is free to
# reclaim (it never causes an out-of-memory restart), but Kaggle's RAM meter counts it, so the meter shows several GB
# more than the engine really uses. Everything is already pinned or on the GPUs, so the cached copies are useless.
# Pages a model still has memory-mapped are left alone by the kernel.
def drop_file_cache(*roots):
    for root in roots:
        for dirpath, _, files in os.walk(root, followlinks=True):
            for name in files:
                try:
                    fd = os.open(os.path.join(dirpath, name), os.O_RDONLY)
                    try:
                        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                    finally:
                        os.close(fd)
                except (OSError, AttributeError):
                    pass


def _ram_report():
    vm = psutil.virtual_memory()
    cache = getattr(vm, "cached", 0) + getattr(vm, "buffers", 0)
    return f"{_ram_in_use()} used by the engine | {_gb(cache)} file cache (reclaimable, Kaggle's meter counts it)"


drop_file_cache(MODELS_DIR)
print(f"   RAM: {_ram_report()}")
_orig_vae_decode = pipeline.vae._decode
PARALLEL_VAE = {"enabled": True, "verified": False}
PEAKS = {}


def parallel_vae_decode(z):
    vae = pipeline.vae
    chunk_tokens, token_drop = vae.tokens_chunk_size, vae.config.token_drop
    ratio = vae.temporal_compression_ratio
    chunk_frames = chunk_tokens * ratio
    if z.shape[2] == 1:
        return _orig_vae_decode(z)
    num_tokens = z.shape[2] + token_drop
    pad_tokens = (-num_tokens) % chunk_tokens
    num_chunks = (num_tokens + pad_tokens) // chunk_tokens - int(token_drop > 0)
    if num_chunks < 2 or not PARALLEL_VAE["enabled"]:
        return _orig_vae_decode(z)
    if pad_tokens > 0:
        z = torch.cat([z, z[:, :, -1:].repeat(1, 1, pad_tokens, 1, 1)], dim=2)
    intra_tail = vae.config.clip_length % ratio
    tokens_before_pad = z.shape[2] - pad_tokens
    pad_frames = sum(intra_tail if intra_tail and (tokens_before_pad + i) % chunk_tokens == 0 else ratio
                     for i in range(pad_tokens))
    output_frames = num_chunks * (chunk_frames - vae.frame_pre_padding) + vae.frame_overlap - pad_frames

    _vae_b.use_tiling = vae.use_tiling
    for attr in ("tile_sample_min_height", "tile_sample_min_width", "tile_sample_min_overlap_height", "tile_sample_min_overlap_width"):
        setattr(_vae_b, attr, getattr(vae, attr))
    PEAKS["denoise"] = [torch.cuda.max_memory_allocated(d) for d in DEVICES]  # the DiT phase ends here
    for d in DEVICES:
        torch.cuda.reset_peak_memory_stats(d)
    clips, check, errors, done, lock = [None] * num_chunks, {}, [], [0], threading.Lock()
    BAR.start("decode", "VAE decoding", num_chunks, "clips")

    def decode_one(dev, model, i):
        start = i * chunk_tokens
        clip = model._decode_clip(z[:, :, start:start + chunk_tokens + vae.token_overlap].to(dev))
        return clip.to("cpu")

    def work(dev, model, indices):
        try:
            with torch.cuda.device(dev), torch.inference_mode():
                for i in indices:
                    if pipeline._interrupt:
                        return
                    if i < 0:  # verification copy of clip 0 on GPU 0
                        check["clip"] = decode_one(dev, model, 0)
                        continue
                    clips[i] = decode_one(dev, model, i)
                    with lock:
                        done[0] += 1
                        BAR.update(done[0])
                        report(0.86 + 0.11 * done[0] / num_chunks, f"🎞️ VAE decoding {done[0]}/{num_chunks} clips")
        except BaseException as e:
            errors.append(e)

    with torch.cuda.device(DIT_DEV):
        torch.cuda.empty_cache()
        _vae_b_attach(DIT_DEV)
    _plan_tile_groups()
    gpu0_clips = list(range(1, num_chunks, 2))
    if not PARALLEL_VAE["verified"]:
        gpu0_clips = [-1] + gpu0_clips
    try:
        helper = threading.Thread(target=work, args=(DIT_DEV, _vae_b, gpu0_clips))
        helper.start()
        work(MAIN_DEV, vae, list(range(0, num_chunks, 2)))
        helper.join()
    finally:
        try:  # let every queued kernel finish before GPU 0's decoder weights are dropped and its memory released
            for dev in DEVICES:
                torch.cuda.synchronize(dev)
        except Exception as e:
            errors.append(e)
        _vae_b_release()  # drop GPU 0's copy
        if not errors:
            with torch.cuda.device(DIT_DEV):
                torch.cuda.empty_cache()
    if errors:
        raise errors[0]
    if pipeline._interrupt:
        from models.minimax_h3.interrupt import GenerationInterrupted
        raise GenerationInterrupted
    if not PARALLEL_VAE["verified"] and "clip" in check:
        a, b = check["clip"].float(), clips[0].float()
        mismatch = ((a - b).norm() / b.norm().clamp_min(1e-6)).item()
        PARALLEL_VAE["verified"] = True
        if mismatch > 1e-2:
            PARALLEL_VAE["enabled"] = False
            print(f"⚠️ GPU 0's VAE copy disagrees with GPU 1 (rel diff {mismatch:.3f}); decoding on GPU 1 only from now on")
            with torch.cuda.device(MAIN_DEV), torch.inference_mode():
                for i in range(1, num_chunks, 2):
                    clips[i] = decode_one(MAIN_DEV, vae, i)
        else:
            print(f"  ✅ Two-GPU VAE decode verified (GPU 0 vs GPU 1 rel diff {mismatch:.1e})")

    BAR.start("stitch", "Stitching clips")
    decoded, write_position, overlap = None, 0, None
    for i in range(num_chunks):
        clip = clips[i].float()
        clips[i] = None
        for j in range(int(token_drop > 0) + 1):
            chunk = clip[:, :, j * chunk_frames:(j + 1) * chunk_frames][:, :, vae.frame_pre_padding:]
            if j == 0:
                if overlap is not None:
                    chunk = vae._blend(overlap, chunk, vae.frame_overlap, dim=-3)
                copy_frames = min(chunk.shape[2], output_frames - write_position)
                if copy_frames > 0:
                    chunk = vae._prepare_decoded_chunk(chunk[:, :, :copy_frames])
                    if decoded is None:
                        # FP16 halves the RAM of a long 720p video (3.9 -> 2 GB); [-1, 1] keeps 0.06-level precision of 8 bits
                        decoded = torch.empty(*chunk.shape[:2], output_frames, *chunk.shape[3:], dtype=torch.float16)
                    decoded[:, :, write_position:write_position + copy_frames].copy_(chunk)
                    write_position += copy_frames
            else:
                overlap = chunk.contiguous()
    if overlap is not None and write_position < output_frames:
        copy_frames = min(overlap.shape[2], output_frames - write_position)
        chunk = vae._prepare_decoded_chunk(overlap[:, :, :copy_frames])
        decoded[:, :, write_position:write_position + copy_frames].copy_(chunk)
        write_position += copy_frames
    if write_position != output_frames:
        raise RuntimeError(f"MiniMax H3 VAE decoded {write_position} frames, expected {output_frames}")
    return decoded


pipeline.vae._decode = parallel_vae_decode
_orig_audio_decode = pipeline.audio_vae.decode


def _audio_decode(*args, **kwargs):
    BAR.start("audio", "Audio decoding")
    return _orig_audio_decode(*args, **kwargs)


pipeline.audio_vae.decode = _audio_decode


# The distillation LoRA is applied live to every generation, exactly as trained. Merging it into the INT8 weights
# does not work: its changes are ~0.1-0.2% of the weights, smaller than half an INT8 step for >99% of them, so
# re-quantizing rounds almost all of it away (measured on the real FL2VA weights: 0.1-25% of the effect survived),
# which leaves the undistilled base model sampled in 4 steps (soft, painterly video). Cost: ~0.85 GB of VRAM per GPU
# and a few % of each step (FP16 tensor cores; it was ~30% in FP32).
PERMANENT_LORAS = {}
if TURBO_LORA_FILE:
    _distill_label = {"pdmd": "⚡ PDMD 4-step", "dmad": "⚡ DMAD 4-step"}.get(VARIANT, "⚡ Turbo (lightx2v)")
    PERMANENT_LORAS[_distill_label] = (os.path.join(LORA_DIR, TURBO_LORA_FILE), 1.0)
    lora_runtime.apply(PERMANENT_LORAS)
    drop_file_cache(LORA_DIR)
    print(f"   RAM after the LoRA: {_ram_report()} | GPU free: "
          f"{_gb(torch.cuda.mem_get_info(0)[0])} / {_gb(torch.cuda.mem_get_info(1)[0])}")
    sys.stdout.flush()


def fetch_lora(entry_or_url):
    """Download a catalog LoRA or a direct .safetensors URL (Hugging Face 'resolve' links work) into LORA_DIR."""
    if isinstance(entry_or_url, dict):
        from huggingface_hub import hf_hub_download
        return hf_hub_download(repo_id=entry_or_url["repo"], filename=entry_or_url["file"], local_dir=LORA_DIR)
    url = entry_or_url.strip()
    m = re.match(r"https?://huggingface\.co/([^/]+/[^/]+)/(?:resolve|blob)/[^/]+/(.+?)(?:\?.*)?$", url)
    if m:
        from huggingface_hub import hf_hub_download
        return hf_hub_download(repo_id=m.group(1), filename=urllib.request.unquote(m.group(2)), local_dir=LORA_DIR)
    name = re.sub(r"[^A-Za-z0-9._-]", "_", os.path.basename(url.split("?")[0])) or "custom_lora.safetensors"
    if not name.endswith(".safetensors"):
        name += ".safetensors"
    path = os.path.join(LORA_DIR, name)
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path + ".part")
        os.replace(path + ".part", path)
    return path


# ==== Prompt builder (MiniMax H3 prompt contract) ====
LANGUAGES = ["English", "Chinese", "Spanish", "French", "German", "Japanese", "Korean", "Hindi", "Urdu", "Arabic",
             "Portuguese", "Russian", "Italian", "Turkish", "Indonesian"]
_STRUCTURED = re.compile(r"(integrated_multimodal_description:|detailed_description:|subject_definitions:)")


def build_prompt(scene, dialogue, language, sound, music, duration_s, has_start, has_end, n_refs, ref_names, has_voice,
                 triggers):
    scene = (scene or "").strip()
    triggers = [t.strip() for t in triggers if t and t.strip().lower() not in scene.lower()]
    if _STRUCTURED.search(scene):  # the user wrote a full H3 prompt: keep it, only put trigger words in front
        return (" ".join(triggers) + " " + scene).strip() if triggers else scene
    # LoRA trigger words open the shot description, the way the LoRAs were captioned
    trigger = "".join(t + (" " if t[-1] in ".,;:" else ", ") for t in triggers)
    secs = f"{duration_s:.1f}".rstrip("0").rstrip(".")
    speaker = "<Subject 1>" if n_refs else "The main character"
    line = ""
    if dialogue and dialogue.strip():
        voice = " in the voice from <Audio 1>" if has_voice else ""
        line = (f" {speaker} speaks clearly{voice} with natural, synchronized lip movements and says "
                f"(S1) <d>[{language}] {dialogue.strip()}</d>")
    sound = (sound or "").strip() or "Natural ambient sound that matches the scene and the visible action."
    music = (music or "").strip() or "N/A"
    pictures = 0
    anchors = []
    if has_start:
        pictures += 1
        anchors.append(f"For the target video, at 0.00 seconds into the target video, <Picture {pictures}> (from [Shot 1]) is fully referenced.")
    if has_end:
        pictures += 1
        anchors.append(f"For the target video, at {duration_s:.2f} seconds into the target video, <Picture {pictures}> (from [Shot 1]) is fully referenced.")
    if not n_refs:
        lead = "Continue directly from <Picture 1>, preserving its subject, composition and lighting. " if has_start else ""
        body = (f"integrated_multimodal_description: [Shot 1] {trigger}A {secs}-second cinematic single take. {lead}{scene}{line}\n"
                f"overall_soundscape: {sound}\nnon_diegetic_music: {music}")
        return "\n".join(anchors + [body])
    names = [n.strip() for n in (ref_names or "").split(",") if n.strip()]
    subjects, retention = [], []
    for i in range(n_refs):
        label = names[i] if i < len(names) else ("the main subject" if i == 0 else "the subject")
        picture = pictures + i + 1
        subjects.append(f"<Subject {i + 1}> is {label} in <Picture {picture}>, preserving their exact identity, "
                        f"appearance, proportions, clothing and distinctive details.")
        retention.append(f"<Subject {i + 1}> (appears in [Shot 1]): fully_preserved - identity and appearance are retained; "
                         f"the setting, action, camera and lighting are new.")
    if has_voice:
        retention.append("<Audio 1>: reference - the voice timbre and delivery of <Subject 1> follow <Audio 1>.")
    task = "[reference generation + audio reference]" if has_voice else "[reference generation]"
    return "\n".join(anchors + [
        "subject_definitions:", *subjects,
        "summary:", f"{task} {scene}",
        "retention_analysis:", *retention,
        "detailed_description:",
        f"{trigger}The target video is a {secs}-second cinematic single take. [Shot 1] {scene}{line} Identities, motion and "
        f"lighting remain stable and physically coherent.",
        f"overall_soundscape: {sound}",
        f"non_diegetic_music: {music}",
    ])


# ==== Sizes, durations and a VRAM guard ====
RESOLUTIONS = {"360p (fastest)": (640, 352), "480p (recommended)": (832, 480), "540p": (960, 544), "720p (slow)": (1280, 704)}
ASPECTS = {"Auto (match image)": None, "16:9 Landscape": 16 / 9, "9:16 Portrait": 9 / 16, "1:1 Square": 1.0,
           "4:3 Classic": 4 / 3, "3:4 Portrait": 3 / 4, "21:9 Cinema": 21 / 9}
STEP_PRESETS = {"⚡ Fast — 4 steps": 4, "⚖️ Balanced — 6 steps": 6, "💎 Quality — 8 steps": 8}
DURATIONS = {"3 s": 73, "4.5 s": 107, "5 s": 124, "6.5 s": 158, "8 s": 192, "10 s": 243, "12 s": 294, "15 s": 362}


def size_for(resolution, aspect, image):
    w0, h0 = RESOLUTIONS[resolution]
    ratio = ASPECTS.get(aspect)
    if ratio is None:
        ratio = image.width / image.height if image is not None else 16 / 9
    area = w0 * h0
    w = max(256, round(math.sqrt(area * ratio) / 32) * 32)
    h = max(256, round(area / w / 32) * 32)
    return w, h


def estimate_rows(w, h, frames, images_in_prompt):
    per_frame = (h // 32) * (w // 32)
    video = h3_pipeline.video_latent_frames(frames) * per_frame
    audio = round(frames / 24 * 40) * 2
    text = 1200 + images_in_prompt * per_frame  # prompt tokens + Qwen vision tokens of each picture
    return video + audio + images_in_prompt * per_frame + text


def free_for_denoise():
    aux_offload.unload_all()
    gc.collect()
    frees = []
    for dev in DEVICES:
        with torch.cuda.device(dev):
            torch.cuda.empty_cache()
        frees.append(torch.cuda.mem_get_info(dev)[0])
    return frees


def needed_bytes(rows, long_video=False):
    """Working VRAM per GPU: activations per packed row (half the heads per GPU) + scratch. Estimates."""
    return rows * (48 if long_video else 110) * 1024 + 700 * 1024**2


# ==== Live progress (also polled by the page on the Cloudflare link, which holds back Gradio's SSE stream) ====
PROGRESS = {"active": False, "fraction": 0.0, "text": "", "started": 0.0, "finished": 0.0, "video": None, "message": ""}


def report(fraction, text):
    PROGRESS.update(active=True, fraction=float(fraction), text=text)


def _to_video_tensor(image):
    return h3_pipeline._pil_to_video(image.convert("RGB"))


def _audio_to_wav(path):
    out = os.path.join(OUTPUT_DIR, ".gradio_cache", f"voice_{int(time.time() * 1000)}.wav")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", path, "-ac", "2", "-ar", "32000", out], check=True)
    import soundfile as sf
    return out, sf.info(out).duration


@torch.inference_mode()  # the decoded video is an inference tensor
def save_mp4(video, audio, sample_rate, path, fps=24):
    """Streams [3, T, H, W] video in [-1, 1] to ffmpeg a few frames at a time: no full-size copies in RAM."""
    import soundfile as sf

    _, t, h, w = video.shape
    wav = path[:-4] + ".wav"
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-"]
    if audio is not None:
        sf.write(wav, np.asarray(audio, dtype=np.float32), int(sample_rate))
        cmd += ["-i", wav, "-c:a", "aac", "-b:a", "192k"]
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "17", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            "-shortest", path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        for start in range(0, t, 16):
            frames = video[:, start:start + 16].to(torch.float32, copy=True).clamp_(-1, 1).add_(1).mul_(127.5).round_()
            proc.stdin.write(frames.to(torch.uint8).permute(1, 2, 3, 0).contiguous().numpy().tobytes())
        proc.stdin.close()
    except BrokenPipeError:
        pass
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed with exit code {proc.returncode}")
    if os.path.exists(wav):
        os.remove(wav)


@torch.inference_mode()
def run_generation(prompt_text, frames, w, h, steps, sampler, shift, seed, start_img=None, end_img=None, ref_imgs=(),
                   voice_path=None, progress=None):
    pipeline._interrupt = False
    t0 = time.time()
    timings = {"text": 0.0, "denoise": 0.0, "decode": 0.0}
    phase = {"name": "prep", "t": time.time(), "step_t": time.time()}

    def ui(fraction, text):
        report(fraction, text)
        if progress is not None:
            try:
                progress(fraction, desc=text)
            except Exception:
                pass

    def callback(step, latent=None, is_start=False, override_num_inference_steps=None, progress_unit=None,
                 progress_title=None, **kwargs):
        now = time.time()
        if progress_unit == "layers":  # Qwen3-VL text / image encoding
            if phase["name"] != "text":
                phase.update(name="text", t=now)
            total = override_num_inference_steps or 50
            BAR.start("text", "Prompt encoding", total, "layers")
            BAR.update(max(0, step + 1), total)
            ui(0.02 + 0.13 * min(1.0, (step + 1) / total), f"📝 {progress_title or 'Encoding prompt'} {max(0, step + 1)}/{total}")
            return
        if progress_unit == "tiles":  # VAE encode / decode
            if "Decod" in (progress_title or "") and phase["name"] != "decode":
                if phase["name"] == "denoise":
                    timings["denoise"] = now - phase["t"]
                phase.update(name="decode", t=now)
            total = override_num_inference_steps or 1
            decoding = "Decod" in (progress_title or "")
            BAR.start("decode" if decoding else "encode", "VAE decoding" if decoding else "Image encoding", total, "tiles")
            BAR.update(max(0, step + 1), total)
            ui(0.86 + 0.11 * min(1.0, (step + 1) / total), f"🎞️ {progress_title or 'VAE'} {max(0, step + 1)}/{total}")
            return
        if is_start:
            if phase["name"] == "text":
                timings["text"] = now - phase["t"]
            phase.update(name="denoise", t=now, step_t=now, total=override_num_inference_steps or steps)
            BAR.start("denoise", "Denoising", phase["total"], "steps")
            ui(0.15, f"🎬 Denoising step 0/{phase['total']}")
            return
        total = phase.get("total", steps)
        dt = now - phase["step_t"]
        phase["step_t"] = now
        done = step + 1
        BAR.update(done, total)
        ui(0.15 + 0.70 * done / total, f"🎬 Denoising step {done}/{total} ({dt:.1f}s/it)")

    kwargs = dict(input_prompt=prompt_text, frame_num=frames, height=h, width=w, sampling_steps=int(steps), seed=int(seed),
                  shift=float(shift), sample_solver=sampler, fps=24, guide_phases=1, callback=callback,
                  image_refs_relative_size=100, video_prompt_type="I" if ref_imgs else "", audio_prompt_type="")
    if start_img is not None:
        kwargs["image_start"] = _to_video_tensor(start_img)
    if end_img is not None:
        kwargs["image_end"] = _to_video_tensor(end_img)
    if ref_imgs:
        kwargs["input_ref_images"] = [img.convert("RGB") for img in ref_imgs]
    if voice_path:
        kwargs.update(audio_guide=voice_path, audio_prompt_type="A")
    ui(0.01, "⚡ Preparing...")
    BAR.start("prep", "Preparing inputs")
    try:
        result = pipeline.generate(**kwargs)
    finally:
        BAR.finish()
    if result is None:
        return None, None, None, timings
    if phase["name"] == "decode":
        timings["decode"] = time.time() - phase["t"]
    elif phase["name"] == "denoise":  # decoding reports through the parallel decoder, not the step callback
        timings["denoise"] = phase["step_t"] - phase["t"]
        timings["decode"] = time.time() - phase["step_t"]
    timings["total"] = time.time() - t0
    return result["x"], result.get("audio"), result.get("audio_sampling_rate", 32000), timings


def on_main_gpu(fn):
    """CUDA's current device is per thread: Gradio workers start on GPU 0, which is full of DiT blocks.
    Run every handler with GPU 1 current, the device the sampler, text encoder and VAEs use for "cuda"."""
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        with torch.cuda.device(MAIN_DEV):
            return fn(*args, **kwargs)
    return wrapped


@on_main_gpu
def generate_video(prompt, dialogue, language, sound, music, start_img, end_img, ref1, ref2, ref3, ref_names, voice,
                   resolution, aspect, duration, steps, sampler, shift, seed, lora1, s1, lora2, s2, custom_url,
                   custom_file, s3, auto_trigger, custom_steps=0, audio_input=None, progress=None):
    PROGRESS.update(active=True, fraction=0.0, text="Starting...", started=time.time(), finished=0.0, video=None, message="")
    try:
        refs = [r for r in (ref1, ref2, ref3) if r is not None]
        if refs and not REFERENCES_SUPPORTED:
            return None, ("❌ Reference → Video needs the Ref2VA model: set H3_VARIANT = \"fused\" in Step 1. "
                          "This variant does text, start-frame and first + last frame to video.")
        if not (prompt or "").strip() and not refs and start_img is None:
            return None, "❌ Please write a prompt."
        frames = DURATIONS[duration]
        steps = int(custom_steps) if custom_steps and int(custom_steps) > 0 else STEP_PRESETS.get(steps, 4)
        w, h = size_for(resolution, aspect, start_img if start_img is not None else (end_img if end_img is not None else (refs[0] if refs else None)))
        if seed is None or int(seed) < 0:
            seed = random.randint(0, 2**31 - 1)
        seed = int(seed)

        # LoRAs: catalog picks + optional custom URL / file
        report(0.01, "🎨 Preparing LoRAs...")
        selections, triggers = dict(PERMANENT_LORAS), []
        for choice, strength in ((lora1, s1), (lora2, s2)):
            if choice and choice != NO_LORA and float(strength) != 0.0 and choice not in selections:
                entry = LORA_CATALOG[choice]
                selections[choice] = (fetch_lora(entry), strength)
                if auto_trigger and entry["trigger"]:
                    triggers.append(entry["trigger"])
        custom = None
        if custom_file is not None:
            custom = custom_file if isinstance(custom_file, str) else custom_file.name
        elif custom_url and custom_url.strip():
            custom = fetch_lora(custom_url)
        if custom and float(s3) != 0.0:
            selections[f"custom:{os.path.basename(custom)}"] = (custom, s3)
        lora_runtime.apply(selections)

        # A project WAV is an audio guide/conditioning input for the H3 pipeline.
        # Keep the separate voice-reference feature for reference-image voice cloning.
        audio_guide_path = None
        if audio_input:
            audio_guide_path, audio_len = _audio_to_wav(audio_input)
            if audio_len < 1.0:
                return None, f"❌ The audio input must be at least 1 second long (got {audio_len:.1f}s)."
        voice_path = None
        if voice:
            if not refs:
                return None, "❌ A voice reference needs at least one reference image (H3 binds each voice to a reference)."
            voice_path, voice_len = _audio_to_wav(voice)
            if voice_len < 2.0:
                return None, f"❌ The voice reference must be at least 2 seconds long (got {voice_len:.1f}s)."
        guide_path = audio_guide_path or voice_path

        images_in_prompt = (start_img is not None) + (end_img is not None) + len(refs)
        rows = estimate_rows(w, h, frames, images_in_prompt)
        frees = free_for_denoise()
        LONG_VIDEO["on"] = needed_bytes(rows) > min(frees)
        if needed_bytes(rows, LONG_VIDEO["on"]) > min(frees):
            return None, (f"❌ {w}x{h} for {duration} needs about {_gb(needed_bytes(rows, True))} of working VRAM per GPU "
                          f"even in long-video mode, but only {_gb(frees[0])} (GPU 0) / {_gb(frees[1])} (GPU 1) is free. "
                          f"Pick a shorter duration or a lower resolution.")

        final_prompt = build_prompt(prompt, dialogue, language, sound, music, frames / 24, start_img is not None,
                                    end_img is not None, len(refs), ref_names, guide_path is not None, triggers)
        final_prompt = prompt
        mode = ("Reference → Video" if refs else "First + Last Frame → Video" if start_img is not None and end_img is not None
                else "Image → Video" if start_img is not None else "Last Frame → Video" if end_img is not None else "Text → Video")
        print(f"\n{'=' * 70}\n🎬 {mode} | {w}x{h} | {frames} frames ({frames / 24:.1f}s) | {int(steps)} steps {sampler} | seed {seed}")
        if len(selections):
            print("🎨 LoRAs: " + ", ".join(f"{k} @ {float(v[1]):.2f}" for k, v in selections.items()))
        if LONG_VIDEO["on"]:
            print(f"🐢 Long-video mode: {rows} tokens, blocks run in {LONG_VIDEO['rows']}-row slices to fit the VRAM")
        print(f"📝 {final_prompt[:1300]}{'...' if len(final_prompt) > 1300 else ''}\n{'=' * 70}")
        sys.stdout.flush()

        for dev in DEVICES:
            torch.cuda.reset_peak_memory_stats(dev)
        PEAKS.clear()
        video, audio, sr, timings = run_generation(final_prompt, frames, w, h, steps, sampler, shift, seed, start_img, end_img,
                                                   refs, guide_path, progress=progress)
        if video is None:
            return None, "⏹️ Generation stopped."
        report(0.97, "💾 Saving MP4...")
        t_save = time.time()
        out_path = os.path.join(OUTPUT_DIR, f"h3_{time.strftime('%Y%m%d_%H%M%S')}_seed{seed}.mp4")
        BAR.start("save", "Saving MP4")
        save_mp4(video, audio, sr, out_path)
        BAR.finish()
        total = timings.get("total", 0.0)
        per_step = timings["denoise"] / max(1, int(steps))
        status = "\n".join([
            f"✅ Done in {total:.0f}s  ({mode})",
            f"⚡ Prompt encoding {timings['text']:.0f}s | Denoising {timings['denoise']:.0f}s ({per_step:.1f}s/step) | VAE + audio {timings['decode']:.0f}s",
            f"📐 {w}x{h} | {frames} frames @ 24 fps ({frames / 24:.1f}s) | 32 kHz stereo audio",
            f"🎲 Seed {seed} | {int(steps)} steps {sampler} | shift {float(shift):g}",
            f"🎨 LoRAs: {', '.join(f'{k} @ {float(v[1]):.2f}' for k, v in selections.items()) or 'none'}",
            f"📁 {out_path}",
        ])
        print(f"✅ Saved {out_path} | total {total:.0f}s = prompt {timings['text']:.0f}s + denoise {timings['denoise']:.0f}s "
              f"({per_step:.1f}s/step) + VAE/audio {timings['decode']:.0f}s + MP4 {time.time() - t_save:.0f}s | "
              f"RAM {_ram_in_use()} | peak VRAM GPU0/GPU1: "
              + (f"denoise {_gb(PEAKS['denoise'][0])}/{_gb(PEAKS['denoise'][1])}, " if "denoise" in PEAKS else "")
              + f"decode {_gb(torch.cuda.max_memory_allocated(0))}/{_gb(torch.cuda.max_memory_allocated(1))}\n")
        drop_file_cache(LORA_DIR, OUTPUT_DIR)
        sys.stdout.flush()
        return out_path, status
    except torch.OutOfMemoryError:
        traceback.print_exc()
        return None, "❌ Out of GPU memory. Pick a lower resolution or a shorter duration and try again."
    except Exception as e:
        traceback.print_exc()
        if "CUDA error" in str(e):  # e.g. an illegal memory access: the GPU context is unusable from here on
            print("❌ The GPUs are in an error state. Re-run Step 2 to restart the engine (the models reload in ~3 min).")
            return None, f"❌ GPU error: {e}\nRe-run Step 2 to restart the engine."
        return None, f"❌ Error: {e}"
    finally:
        BAR.abort()
        PROGRESS["active"] = False
        for key in _tp_cache:  # last hidden state / rope / temb copies held on both GPUs
            _tp_cache[key] = None
        gc.collect()
        for dev in DEVICES:
            with torch.cuda.device(dev):
                torch.cuda.empty_cache()


# ==== Generation as a background job ====
# A long clip takes 15-60 minutes, longer than the Gradio share / Cloudflare tunnels reliably keep one event stream
# open. So the Generate button only starts a job and returns; the page asks for the job's state every 2 s with short,
# independent requests (queue=False: plain HTTP, no event stream), and a reloaded page picks the result up again.
JOB = {"id": 0, "done": 0, "running": False, "video": None, "status": "Ready.", "label": "", "queue_len": 0}
BATCH_JOB = {
    "running": False,
    "done": False,
    "video": None,
    "status": "Ready."
}
JOB_QUEUE = []
JOB_HISTORY = []
_QUEUE_LOCK = threading.Lock()
_JOB_COUNTER = 0


def _job_worker(job_id, args, label=""):
    try:
        video, status = generate_video(*args)
    except BaseException as e:  # generate_video reports its own errors; this is a last resort
        video, status = None, f"❌ Error: {e}"
    with _QUEUE_LOCK:
        JOB.update(running=False, done=job_id, video=video, status=status, label=label)
        PROGRESS.update(active=False, finished=time.time(), video=os.path.basename(video) if video else None,
                        message=status.splitlines()[0] if status else "")
        JOB_HISTORY.append({
            "id": job_id,
            "label": label or f"Job #{job_id}",
            "video": video,
            "status": status,
            "finished_at": time.time(),
        })
        if JOB_QUEUE:
            next_job = JOB_QUEUE.pop(0)
            JOB.update(id=next_job["id"], running=True, label=next_job["label"], queue_len=len(JOB_QUEUE))
            PROGRESS.update(active=True, fraction=0.0, text=f"Starting {next_job['label']}...",
                            started=time.time(), finished=0.0, video=None, message="")
            threading.Thread(target=_job_worker, args=(next_job["id"], next_job["args"], next_job["label"]), daemon=True).start()
        else:
            JOB.update(queue_len=0)


def queue_job(args, label=""):
    """Queue a generation job. If no job is currently running, starts immediately.
    Returns (job_id, queue_position). position == 0 means started immediately."""
    global _JOB_COUNTER
    with _QUEUE_LOCK:
        _JOB_COUNTER += 1
        job_id = _JOB_COUNTER
        clean_label = label or f"Job #{job_id}"
        job_item = {
            "id": job_id,
            "args": args,
            "label": clean_label,
            "queued_at": time.time(),
        }
        if not JOB["running"]:
            JOB.update(id=job_id, running=True, label=clean_label, queue_len=len(JOB_QUEUE))
            PROGRESS.update(active=True, fraction=0.0, text=f"Starting {clean_label}...",
                            started=time.time(), finished=0.0, video=None, message="")
            threading.Thread(target=_job_worker, args=(job_id, args, clean_label), daemon=True).start()
            return job_id, 0
        else:
            JOB_QUEUE.append(job_item)
            pos = len(JOB_QUEUE)
            JOB.update(queue_len=pos)
            return job_id, pos


def start_job(*args, label=""):
    """Backward-compatible entry point: queues or starts the generation."""
    return queue_job(args, label=label)


def clear_queue():
    """Clear all pending jobs from the queue without interrupting the running job."""
    with _QUEUE_LOCK:
        cleared = len(JOB_QUEUE)
        JOB_QUEUE.clear()
        JOB.update(queue_len=0)
        return cleared


def get_queue_status():
    with _QUEUE_LOCK:
        return {
            "running": JOB["running"],
            "current_id": JOB["id"],
            "current_label": JOB.get("label", f"Job #{JOB['id']}"),
            "queue_len": len(JOB_QUEUE),
            "queue": [{"id": j["id"], "label": j["label"]} for j in JOB_QUEUE],
            "history": list(JOB_HISTORY),
        }


def stop_generation():
    with _QUEUE_LOCK:
        cleared = len(JOB_QUEUE)
        JOB_QUEUE.clear()
        JOB.update(queue_len=0)
    if not JOB["running"]:
        if cleared > 0:
            return f"🗑️ Cancelled {cleared} queued job(s)."
        return "Nothing is running."
    pipeline._interrupt = True
    if cleared > 0:
        return f"⏹️ Stopping after current step (cancelled {cleared} queued job(s))..."
    return "⏹️ Stopping after the current step..."


def load_latest_video():
    videos = sorted(glob.glob(os.path.join(OUTPUT_DIR, "*.mp4")), key=os.path.getmtime)
    if not videos:
        return None, f"No videos in {OUTPUT_DIR} yet."
    return videos[-1], f"📁 Loaded latest video: {videos[-1]}"


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🎬 MiniMax H3 Turbo Backend is ready!")
    print("=" * 70)
    print(f"⚡ GPUs: 2 x {_props.name} | Model: {VARIANT_LABEL}")
    print(f"📁 Output directory: {OUTPUT_DIR}")
    print("To launch the web interface, run: python frontend.py")
    print("=" * 70 + "\n")
