print("Running cell2.py")
import os
import shutil
from huggingface_hub import hf_hub_download

REPO = 'DeepBeepMeep/LTX-2'
MODEL_DIR = 'Wan2GP/models'
TMP_DIR = '/kaggle/tmp/models' if os.path.exists('/kaggle') else os.path.abspath('tmp_models')
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)

# Large files → /kaggle/tmp, symlinked back
LARGE_FILES = [
    'ltx-2.3-22b-distilled-Q4_K_M_light.gguf',
    'ltx-2.3-22b_embeddings_connector.safetensors',
    'ltx-2.3-22b_text_embedding_projection.safetensors',
    'ltx-2.3-22b_vae.safetensors',
]

for f in LARGE_FILES:
    dest = os.path.join(MODEL_DIR, f)
    if os.path.exists(dest):
        print(f'  ✓ Already exists: {f}')
        continue
    print(f'Downloading {f} → /kaggle/tmp ...')
    hf_hub_download(repo_id=REPO, filename=f, local_dir=TMP_DIR)
    actual = os.path.join(TMP_DIR, f)
    if not os.path.exists(dest):
        os.symlink(actual, dest)
    print(f'  ✓ {f} (symlinked)')

# Small files download directly
SMALL_FILES = [
    'ltx-2.3-22b_audio_vae.safetensors',
    'ltx-2.3-22b_vocoder.safetensors',
    'ltx-2.3-spatial-upscaler-x2-1.1.safetensors',
]

for f in SMALL_FILES:
    dest = os.path.join(MODEL_DIR, f)
    if os.path.exists(dest):
        print(f'  ✓ Already exists: {f}')
        continue
    print(f'Downloading {f}...')
    hf_hub_download(repo_id=REPO, filename=f, local_dir=TMP_DIR)
    print(f'  ✓ {f}')

# Gemma text encoder → /kaggle/tmp
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
        if os.path.exists(tmp_file): continue
        print(f'Downloading gemma/{gf} → /kaggle/tmp ...')
        hf_hub_download(repo_id=REPO, filename=f'{GEMMA_FOLDER}/{gf}', local_dir=TMP_DIR)
    if not os.path.exists(gemma_dest):
        os.symlink(gemma_tmp, gemma_dest)
    print(f'  ✓ {GEMMA_FOLDER}/ (symlinked)')

# Cleanup HF cache
for d in [os.path.join(MODEL_DIR, '.cache'), os.path.join(TMP_DIR, '.cache')]:
    if os.path.exists(d): shutil.rmtree(d)

if os.path.exists('/kaggle'):
    os.system('df -h /kaggle/working /kaggle/tmp')
print('\n✅ All downloads complete!')