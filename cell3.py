import os
print("Running cell3.py")
import shutil
from huggingface_hub import hf_hub_download

REPO = 'Abiray/LTX-2.3-22B-DISTILLED-1.1-GGUF'
COMPANION_REPO = 'DeepBeepMeep/LTX-2'
MODEL_DIR = 'Wan2GP/models'
TMP_DIR = '/kaggle/tmp/models'
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)

# ── GGUF Transformer (~17.8 GB) ──
TRANSFORMER_FILE    = 'LTX-2.3-22B-distilled-1.1-Q4_K_M.gguf'
TRANSFORMER_RENAMED = 'ltx-2.3-22b-distilled-1.1-Q4_K_M.gguf'

dest = os.path.join(MODEL_DIR, TRANSFORMER_RENAMED)
if os.path.exists(dest):
    print(f'  ✓ Already exists: {TRANSFORMER_RENAMED}')
else:
    print(f'  Downloading {TRANSFORMER_FILE} → /kaggle/tmp ...')
    hf_hub_download(repo_id=REPO, filename=TRANSFORMER_FILE, local_dir=TMP_DIR)
    actual = os.path.join(TMP_DIR, TRANSFORMER_FILE)
    os.symlink(actual, dest)
    print(f'  ✓ {TRANSFORMER_RENAMED} (symlinked)')

# ── Companion large files ──
LARGE_FILES = [
    'ltx-2.3-22b-distilled-lora-384.safetensors',
    'ltx-2.3-22b_embeddings_connector.safetensors',
    'ltx-2.3-22b_text_embedding_projection.safetensors',
    'ltx-2.3-22b_vae.safetensors',
]

for f in LARGE_FILES:
    dest = os.path.join(MODEL_DIR, f)
    if os.path.exists(dest):
        print(f'  ✓ Already exists: {f}')
        continue
    print(f'  Downloading {f} → /kaggle/tmp ...')
    hf_hub_download(repo_id=COMPANION_REPO, filename=f, local_dir=TMP_DIR)
    actual = os.path.join(TMP_DIR, f)
    os.symlink(actual, dest)
    print(f'  ✓ {f} (symlinked)')

# ── Small companion files ──
SMALL_FILES = [
    'ltx-2.3-22b_audio_vae.safetensors',
    'ltx-2.3-22b_vocoder.safetensors',
    'ltx-2.3-spatial-upscaler-x2-1.1.safetensors',
    'ltx-2.3-temporal-upscaler-x2-1.0.safetensors',
]

for f in SMALL_FILES:
    dest = os.path.join(MODEL_DIR, f)
    if os.path.exists(dest):
        print(f'  ✓ Already exists: {f}')
        continue
    print(f'  Downloading {f} ...')
    hf_hub_download(repo_id=COMPANION_REPO, filename=f, local_dir=MODEL_DIR)
    print(f'  ✓ {f}')

# ── Gemma Text Encoder ──
GEMMA_FOLDER = 'gemma-3-12b-it-qat-q4_0-unquantized'
GEMMA_FILES = [
    'gemma-3-12b-it-qat-q4_0-unquantized_quanto_bf16_int8.safetensors',
    'added_tokens.json', 'chat_template.json', 'config_light.json',
    'generation_config.json', 'preprocessor_config.json', 'processor_config.json',
    'special_tokens_map.json', 'tokenizer.json', 'tokenizer.model', 'tokenizer_config.json',
]

gemma_dest = os.path.join(MODEL_DIR, GEMMA_FOLDER)
gemma_tmp = os.path.join(TMP_DIR, GEMMA_FOLDER)
if os.path.exists(gemma_dest):
    print(f'  ✓ Already exists: {GEMMA_FOLDER}/')
else:
    os.makedirs(gemma_tmp, exist_ok=True)
    for gf in GEMMA_FILES:
        tmp_file = os.path.join(gemma_tmp, gf)
        if os.path.exists(tmp_file):
            continue
        print(f'  Downloading gemma/{gf} → /kaggle/tmp ...')
        hf_hub_download(repo_id=COMPANION_REPO, filename=f'{GEMMA_FOLDER}/{gf}', local_dir=TMP_DIR)
    os.symlink(gemma_tmp, gemma_dest)
    print(f'  ✓ {GEMMA_FOLDER}/ (symlinked)')

# ── Cleanup HF cache ──
for d in [os.path.join(MODEL_DIR, '.cache'), os.path.join(TMP_DIR, '.cache')]:
    if os.path.exists(d):
        shutil.rmtree(d, ignore_errors=True)

print('  ✓ All models ready')