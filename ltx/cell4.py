import os
print("Running cell4.py")

import shutil
from huggingface_hub import hf_hub_download

REPO = 'DeepBeepMeep/LTX-2'
MODEL_DIR = 'Wan2GP/models'
TMP_DIR = '/kaggle/input/datasets/mikegriffin2027/ltx-23-models'
os.makedirs(MODEL_DIR, exist_ok=True)

# Large files → /kaggle/tmp, symlinked back
LARGE_FILES = [
    'ltx-2.3-22b-distilled-Q4_K_M_light.gguf',
    'ltx-2.3-22b_embeddings_connector.safetensors',
    'ltx-2.3-22b_text_embedding_projection.safetensors',
    'ltx-2.3-22b_vae.safetensors',
    'ltx-2.3-22b_audio_vae.safetensors',
    'ltx-2.3-22b_vocoder.safetensors',
    'ltx-2.3-spatial-upscaler-x2-1.1.safetensors'
]

for f in LARGE_FILES:
    dest = os.path.join(MODEL_DIR, f)
    print(dest)
    if os.path.exists(dest):
        print(f'  ✓ Already exists: {f}')
        continue
    actual = os.path.join(TMP_DIR, f)
    if not os.path.exists(dest):
        os.symlink(actual, dest)
    print(f'  ✓ {f} (symlinked)')

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
    if not os.path.exists(gemma_dest):
        os.symlink(gemma_tmp, gemma_dest)
    print(f'  ✓ {GEMMA_FOLDER}/ (symlinked)')
