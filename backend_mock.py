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

PROGRESS = {"active": False, "fraction": 0.0, "text": "", "started": 0.0, "finished": 0.0, "video": None, "message": ""}
LANGUAGES = ["English", "Chinese", "Spanish", "French", "German", "Japanese", "Korean", "Hindi", "Urdu", "Arabic",
             "Portuguese", "Russian", "Italian", "Turkish", "Indonesian"]
_STRUCTURED = re.compile(r"(integrated_multimodal_description:|detailed_description:|subject_definitions:)")

RESOLUTIONS = {"360p (fastest)": (640, 352), "480p (recommended)": (832, 480), "540p": (960, 544), "720p (slow)": (1280, 704)}
ASPECTS = {"Auto (match image)": None, "16:9 Landscape": 16 / 9, "9:16 Portrait": 9 / 16, "1:1 Square": 1.0,
           "4:3 Classic": 4 / 3, "3:4 Portrait": 3 / 4, "21:9 Cinema": 21 / 9}
STEP_PRESETS = {"⚡ Fast — 4 steps": 4, "⚖️ Balanced — 6 steps": 6, "💎 Quality — 8 steps": 8}
DURATIONS = {"3 s": 73, "4.5 s": 107, "5 s": 124, "6.5 s": 158, "8 s": 192, "10 s": 243, "12 s": 294, "15 s": 362}

VARIANTS = {  # label, Wan2GP model type, Reference → Video support, default sampler, default steps
    "pdmd": ("PDMD 4-step (base H3 + PDMD LoRA)", "minimax_h3_fl2va_pruned", False, "euler", 4),
    "dmad": ("DMAD 4-step (base H3 + DMAD LoRA)", "minimax_h3_fl2va_pruned", False, "euler", 4),
    "fused": ("MATLOWAI fused turbo (Ref2VA)", "minimax_h3_ref2va_pruned", True, "res_multistep", 4),  # Ref2VA AdaLN table (verified)
    "official": ("Ref2VA + lightx2v 8-step LoRA", "minimax_h3_ref2va_pruned", True, "res_multistep", 8),
}
VARIANT_LABEL, MODEL_TYPE, REFERENCES_SUPPORTED, DEFAULT_SAMPLER, DEFAULT_STEPS = VARIANTS[VARIANT]
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
INT8_LABEL = {"kitchen": "Comfy Kitchen INT8 tensor cores"}
import numpy as np
import psutil
import torch
from PIL import Image

def _fmt_secs(seconds):
    seconds = int(round(seconds))
    return f"{seconds // 60}m{seconds % 60:02d}s" if seconds >= 60 else f"{seconds}s"

def _gb(n):
    return f"{n / 1024**3:.1f} GB"


def _ram_in_use():
    """total - available: unlike psutil's 'used', this counts pinned (page-locked) memory, which Linux files as shared."""
    vm = psutil.virtual_memory()
    return f"{_gb(vm.total - vm.available)} / {_gb(vm.total)}"

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

def _empty_exact_pinned(*size, pin_memory=False, **kwargs):
    """torch.empty for MMGP's pinning: its non-power-of-two uint8 chunks are registered at their exact size."""
    n = size[0] if len(size) == 1 and isinstance(size[0], int) else 0
    if pin_memory and n > 0 and n & (n - 1) and kwargs.get("dtype") == torch.uint8:
        try:
            return pinned_empty_exact(n)
        except Exception:
            pass
    return _torch_empty(*size, pin_memory=pin_memory, **kwargs)


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


def _audio_decode(*args, **kwargs):
    BAR.start("audio", "Audio decoding")
    return _orig_audio_decode(*args, **kwargs)



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
JOB = {"id": 0, "done": 0, "running": False, "video": None, "status": "Ready."}
BATCH_JOB = {
    "running": False,
    "done": False,
    "video": None,
    "status": "Ready."
}


def _job_worker(job_id, args):
    try:
        video, status = generate_video(*args)
    except BaseException as e:  # generate_video reports its own errors; this is a last resort
        video, status = None, f"❌ Error: {e}"
    JOB.update(running=False, done=job_id, video=video, status=status)
    PROGRESS.update(active=False, finished=time.time(), video=os.path.basename(video) if video else None,
                    message=status.splitlines()[0] if status else "")


def start_job(*args):
    """Start generation worker in a background thread."""
    JOB.update(id=JOB["id"] + 1, running=True)
    PROGRESS.update(active=True, fraction=0.0, text="Starting...", started=time.time(), finished=0.0, video=None, message="")
    threading.Thread(target=_job_worker, args=(JOB["id"], args), daemon=True).start()


def stop_generation():
    if not JOB["running"]:
        return "Nothing is running."
    pipeline._interrupt = True
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