import os
print("Running h3 cell1.py")
# Step 1: Environment, pinned Wan2GP, dependencies and model downloads

# ╔══════════════════════════════════════════════════════════════════════════════════════╗
# ║  👉 CHOOSE YOUR MODEL: "pdmd" (default) · "dmad" · "fused" · "official"              ║
# ║     pdmd / dmad: newest 4-step distillations; text, image, first + last frame        ║
# ║     fused / official: also Reference → Video + voice clone (see the table above)     ║
# ╚══════════════════════════════════════════════════════════════════════════════════════╝
H3_VARIANT = "fused"

import os
import sys
import time
import shutil
import subprocess
import threading
import psutil
from concurrent.futures import ThreadPoolExecutor

def stop_old_engine():
    # A previous run's engine (or one left behind by a kernel restart) still holds both GPUs and ~23 GB of RAM.
    stale = []
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            args, name = p.info["cmdline"] or [], (p.info["name"] or "").lower()
            engine = "python" in name and any(os.path.basename(a) == "run_h3.py" for a in args[1:3])
            if p.pid != os.getpid() and (engine or name == "cloudflared" or name.startswith("frpc")):
                p.kill()
                stale.append(p)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    psutil.wait_procs(stale, timeout=20)
    if stale:
        print(f"🧹 Stopped {len(stale)} leftover engine process(es) from an earlier run: GPU memory and RAM are free again")


stop_old_engine()

# ---- Settings ----
WAN2GP_COMMIT = "b8b18f8114e432eea8f3d7e853a51dd91fa99571"  # Wan2GP, 1 Oct 2026 (the version this notebook was built on)
MODELS_DIR = "/kaggle/tmp/h3_models" if os.path.isdir("/kaggle") else os.path.abspath("h3_models")
LORA_DIR = "/kaggle/tmp/h3_loras" if os.path.isdir("/kaggle") else os.path.abspath("h3_loras")

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
os.environ["HF_XET_CHUNK_CACHE_SIZE_BYTES"] = "0"  # no second copy of 40 GB in the xet cache
os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("=" * 70)
print("🎬 MiniMax H3 Turbo INT8 - Video + Audio Generator")
print("📺 Created by: AIQUEST Academy | YouTube & X: @AIQuestAcademy")
print("=" * 70)
MODEL_MENU = {
    "pdmd": "Base H3 + PDMD 4-step LoRA       | text, image, first+last frame | 4 steps",
    "dmad": "Base H3 + DMAD 4-step LoRA       | text, image, first+last frame | 4 steps",
    "fused": "MATLOWAI fused turbo (Ref2VA)    | + Reference → Video, voice    | 4 steps",
    "official": "Ref2VA + lightx2v 8-step LoRA    | + Reference → Video, voice    | 8 steps",
}
if H3_VARIANT not in MODEL_MENU:
    raise SystemExit(f'❌ Unknown H3_VARIANT "{H3_VARIANT}". Set it to one of: ' + ", ".join(f'"{k}"' for k in MODEL_MENU))
print("🧠 Model (H3_VARIANT, first line of this cell):")
for key, text in MODEL_MENU.items():
    print(f"   {'👉' if key == H3_VARIANT else '  '} {key:<9} {text}")
print("   To switch: edit H3_VARIANT at the top of this cell and re-run Step 1 + Step 2.")
print("=" * 70)
ram = psutil.virtual_memory()
print(f"RAM: {ram.total / 1024**3:.1f} GB total, {ram.available / 1024**3:.1f} GB available")
try:
    smi = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], text=True).strip().splitlines()
    for i, line in enumerate(smi):
        print(f"✅ GPU {i}: {line}")
    if len(smi) < 2:
        print("❌ This notebook needs 2 GPUs: Settings → Accelerator → GPU T4 x2")
except Exception:
    print("❌ No GPU found: Settings → Accelerator → GPU T4 x2")

# ---- 1. Wan2GP, pinned to a tested commit ----
t0 = time.time()
if not os.path.isdir("Wan2GP/.git"):
    print("\n📥 Fetching Wan2GP...")
    subprocess.run(["git", "init", "-q", "Wan2GP"], check=True)
    subprocess.run(["git", "-C", "Wan2GP", "remote", "add", "origin", "https://github.com/DeepBeepMeep/Wan2GP.git"], check=True)
subprocess.run(["git", "-C", "Wan2GP", "fetch", "-q", "--depth", "1", "origin", WAN2GP_COMMIT], check=True)
subprocess.run(["git", "-C", "Wan2GP", "reset", "-q", "--hard", "FETCH_HEAD"], check=True)  # also undoes stale edits on re-runs
print(f"✅ Wan2GP @ {WAN2GP_COMMIT[:10]} ({time.time() - t0:.0f}s)")

# ---- 2. Dependencies: only what the H3 path imports ----
# (Wan2GP's full requirements.txt pins an onnxruntime nightly that is no longer on PyPI.)
print("📦 Installing dependencies...")
t0 = time.time()
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-warn-conflicts", "--timeout", "120", "--retries", "5",
                "mmgp==3.8.2", "comfy-kitchen==0.2.35", "transformers==4.54.0", "diffusers==0.36.0", "gguf==0.17.1", "optimum-quanto", "accelerate>=1.1.1", "gradio==5.29.0", "huggingface_hub[hf_xet]", "hf_xet>=1.5.2", "sentencepiece", "soundfile"], check=True)
print(f"✅ Dependencies installed ({time.time() - t0:.0f}s)")

# ---- 3. Model weights (parallel; files from an attached Kaggle Dataset are reused) ----
H3 = "DeepBeepMeep/MiniMax-H3"
TE = "Qwen3-VL-32B-Instruct"
BASE_DIT = (H3, "MiniMax-H3-FL2VA-pruned_rank8_int8_convrot.safetensors", 21.06)  # the base T2VA / FL2VA transformer
if H3_VARIANT == "pdmd":  # PDMD (UCSD / ByteDance): https://github.com/ZeamoxWang/pdmd
    DIT, TURBO_LORA = BASE_DIT, ("pdmd2026/pdmd_4NFE_lora", "lora_model_0.safetensors", 1.38)
elif H3_VARIANT == "dmad":  # DMAD (TAMU / ByteDance): https://github.com/Yzmblog/DMAD
    DIT, TURBO_LORA = BASE_DIT, ("ZhengmingYu/DMAD", "minimax_h3/dmad_minimax_h3_4step_full_critic.safetensors", 1.38)
elif H3_VARIANT == "fused":
    DIT = ("MATLOWAI/minimax-h3-fused-turbo-int8-convrot",
           "diffusion_models/minimax_h3_fused_refdelta_r1024_turbo8_mystic07_int8_convrot.safetensors", 20.98)
    TURBO_LORA = None
elif H3_VARIANT == "official":
    DIT = (H3, "MiniMax-H3-Ref2VA-pruned_rank8_int8_convrot.safetensors", 21.06)
    TURBO_LORA = (H3, "loras/minimax_h3_lightx2v_ref2v_turbo_8step_alpha8_v1.0_768p_bf16.safetensors", 1.38)

BIG = [DIT,
       (H3, f"{TE}/qwen3vl-32B-MiniMax-H3-Q4_K_M.gguf", 14.58),
       (H3, "minimax_h3/minimax_h3_video_vae_int8_convrot.safetensors", 2.81),
       (H3, "MiniMax-H3-audio_vae_fp32.safetensors", 0.61)]
SMALL = [(H3, f"{TE}/{name}", 0.0) for name in
         ("config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "preprocessor_config.json", "chat_template.json")]
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LORA_DIR, exist_ok=True)

attached = {}
if os.path.isdir("/kaggle/input"):
    for root, _, files in os.walk("/kaggle/input"):
        for f in files:
            attached.setdefault(f, os.path.join(root, f))


def fetch(item, local_dir=MODELS_DIR):
    from huggingface_hub import hf_hub_download
    repo, filename, size_gb = item
    dest = os.path.join(local_dir, filename)
    if os.path.exists(dest):
        return f"✓ {filename}"
    source = attached.get(os.path.basename(filename))
    if source and os.path.getsize(source) > 0:
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        os.symlink(source, dest)
        return f"✓ {filename} (from attached dataset)"
    for attempt in range(1, 4):
        try:
            hf_hub_download(repo_id=repo, filename=filename, local_dir=local_dir)
            return f"✅ {filename}" + (f" ({size_gb:.1f} GB)" if size_gb else "")
        except Exception as e:
            if attempt == 3:
                raise
            print(f"  ⚠️ {filename}: {e} - retrying ({attempt}/3)")
            time.sleep(5 * attempt)


def folder_bytes(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            p = os.path.join(root, f)
            if not os.path.islink(p):
                try:
                    total += os.path.getsize(p)
                except OSError:
                    pass
    return total


todo = [item for item in BIG + ([TURBO_LORA] if TURBO_LORA else [])
        if not os.path.exists(os.path.join(MODELS_DIR if item is not TURBO_LORA else LORA_DIR, item[1]))
        and os.path.basename(item[1]) not in attached]
need_gb = sum(size for _, _, size in todo)
try:  # exact sizes for an exact progress bar (the sizes above are rounded)
    from huggingface_hub import HfApi
    exact = 0
    for repo in {item[0] for item in todo}:
        files = [item[1] for item in todo if item[0] == repo]
        exact += sum(info.size or 0 for info in HfApi().get_paths_info(repo, files))
    need_gb = exact / 1e9 if exact else need_gb
except Exception:
    pass
free_gb = shutil.disk_usage(MODELS_DIR).free / 1024**3
print(f"\n🚀 Downloading {need_gb:.1f} GB of weights to {MODELS_DIR} ({free_gb:.0f} GB free)...")
if need_gb > free_gb - 2:
    raise SystemExit(f"❌ Not enough disk space: need {need_gb:.0f} GB, {free_gb:.0f} GB free.")
done = threading.Event()
print_lock = threading.Lock()
bar_shown = [False]


def fmt_time(seconds):
    seconds = int(seconds)
    return f"{seconds // 60}m{seconds % 60:02d}s" if seconds >= 60 else f"{seconds}s"


def say(message):  # a normal line printed above the live bar
    with print_lock:
        if bar_shown[0]:
            sys.stdout.write("\r" + " " * 100 + "\r")
            bar_shown[0] = False
        print(message, flush=True)


def monitor():
    # One live line: GB done / total, %, smoothed speed, ETA. hf_xet writes into partial files, so bytes on disk
    # (minus what was already there) is the real progress.
    base = folder_bytes(MODELS_DIR) + folder_bytes(LORA_DIR)
    total = max(need_gb * 1e9, 1.0)
    start, samples = time.time(), []
    while not done.wait(1.0):
        got = folder_bytes(MODELS_DIR) + folder_bytes(LORA_DIR) - base
        now = time.time()
        samples = [(t, b) for t, b in samples if now - t <= 10] + [(now, got)]
        speed = (samples[-1][1] - samples[0][1]) / max(samples[-1][0] - samples[0][0], 1e-6) if len(samples) > 1 else 0
        got = min(got, total)
        frac = got / total
        fill = int(frac * 28)
        eta = fmt_time(max(0.0, total - got) / speed) if speed > 1e6 else "--"
        line = (f"  ⬇️ [{'█' * fill}{'░' * (28 - fill)}] {got / 1e9:5.1f} / {total / 1e9:.1f} GB {frac * 100:3.0f}% | "
                f"{speed / 1e6:4.0f} MB/s | ETA {eta} | {fmt_time(now - start)}")
        with print_lock:
            sys.stdout.write("\r" + line.ljust(100))
            sys.stdout.flush()
            bar_shown[0] = True


threading.Thread(target=monitor, daemon=True).start()
t0 = time.time()
try:
    with ThreadPoolExecutor(max_workers=6) as pool:
        jobs = [pool.submit(fetch, item) for item in BIG + SMALL]
        if TURBO_LORA:
            jobs.append(pool.submit(fetch, TURBO_LORA, LORA_DIR))
        for job in jobs:
            say("  " + job.result())
finally:
    done.set()
    time.sleep(1.1)
say(f"  ✅ {need_gb:.1f} GB downloaded in {fmt_time(time.time() - t0)}"
    + (f" (avg {need_gb * 1e3 / max(time.time() - t0, 1):.0f} MB/s)" if need_gb else ""))
for cache in (os.path.join(MODELS_DIR, ".cache"), os.path.join(LORA_DIR, ".cache")):
    shutil.rmtree(cache, ignore_errors=True)

# Settings for Step 2's engine (also saved to a file, so Step 2 can be re-run alone after a kernel restart)
import json
H3_SETTINGS = {"H3_VARIANT": H3_VARIANT, "H3_MODELS_DIR": MODELS_DIR, "H3_LORA_DIR": LORA_DIR, "H3_DIT_FILE": DIT[1],
               "H3_TURBO_LORA": TURBO_LORA[1] if TURBO_LORA else ""}
os.environ.update(H3_SETTINGS)
with open("h3_settings.json", "w") as f:
    json.dump(H3_SETTINGS, f)
print("\n" + "=" * 70)
print(f"✅ All weights ready in {time.time() - t0:.0f}s | disk free: {shutil.disk_usage(MODELS_DIR).free / 1024**3:.0f} GB")
print("=" * 70)