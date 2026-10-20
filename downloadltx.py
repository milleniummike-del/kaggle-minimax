import os
import subprocess
os.environ['HF_HOME'] = '/kaggle/tmp/hf_home'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

from huggingface_hub import hf_hub_download

print("Running downloadltx.py")

COMPANION_REPO = 'DeepBeepMeep/LTX-2'
WAN_REPO = 'DeepBeepMeep/Wan2.1'
MODEL_DIR = 'Wan2GP/models'
TMP_DIR = '/kaggle/tmp/models'

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)

def download_and_link(repo, filename, subfolder=None):
    if subfolder:
        huggingface_filename = f"{subfolder}/{filename}"
        dest_dir = os.path.join(MODEL_DIR, subfolder)
        tmp_dest_dir = os.path.join(TMP_DIR, subfolder)
    else:
        huggingface_filename = filename
        dest_dir = MODEL_DIR
        tmp_dest_dir = TMP_DIR
    
    os.makedirs(dest_dir, exist_ok=True)
    os.makedirs(tmp_dest_dir, exist_ok=True)
    
    dest_path = os.path.join(dest_dir, filename)
    if os.path.exists(dest_path):
        print(f"  ✓ Already exists: {os.path.join(subfolder or '', filename)}")
        return
        
    print(f"Downloading {huggingface_filename} → /kaggle/tmp...")
    hf_hub_download(repo_id=repo, filename=huggingface_filename, local_dir=TMP_DIR, local_dir_use_symlinks=False)
    
    actual_path = os.path.join(TMP_DIR, huggingface_filename)
    os.symlink(actual_path, dest_path)
    print(f"  ✓ {os.path.join(subfolder or '', filename)} (symlinked)")

# 1. Main model weights
MAIN_FILES = [
    'scenema-audio-transformer_quanto_bf16_int8.safetensors',
    'ltx-2.3-22b_audio_vae.safetensors',
    'ltx-2.3-22b_vocoder.safetensors',
    'ltx-2.3-22b_text_embedding_projection.safetensors',
    'ltx-2.3-22b_embeddings_connector.safetensors',
]
for f in MAIN_FILES:
    download_and_link(COMPANION_REPO, f)

# 2. Gemma-3 Text Encoder
GEMMA_FILES = [
    'gemma-3-12b-it-qat-q4_0-unquantized_quanto_bf16_int8.safetensors',
    'added_tokens.json', 'chat_template.json', 'config_light.json',
    'generation_config.json', 'preprocessor_config.json', 'processor_config.json',
    'special_tokens_map.json', 'tokenizer.json', 'tokenizer.model', 'tokenizer_config.json',
]
for f in GEMMA_FILES:
    download_and_link(COMPANION_REPO, f, 'gemma-3-12b-it-qat-q4_0-unquantized')

# 3. Whisper Medium
WHISPER_MEDIUM_FILES = ['config.json', 'model.safetensors']
for f in WHISPER_MEDIUM_FILES:
    download_and_link(WAN_REPO, f, 'whisper_medium')

# 4. Kokoro TTS
KOKORO_FILES = ['config.json', 'kokoro-v1_0.pth']
for f in KOKORO_FILES:
    download_and_link(COMPANION_REPO, f, 'kokoro')

KOKORO_VOICES = ['af_heart.pt']
for f in KOKORO_VOICES:
    download_and_link(COMPANION_REPO, f, 'kokoro/voices')

# 5. SeedVC Voice Cloning
SEEDVC_FILES = [
    'DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth',
    'config_dit_mel_seed_uvit_whisper_small_wavenet.yml',
    'campplus_cn_common.bin',
]
for f in SEEDVC_FILES:
    download_and_link(COMPANION_REPO, f, 'seed-vc')

BIGVGAN_FILES = ['config.json', 'bigvgan_generator.pt']
for f in BIGVGAN_FILES:
    download_and_link(COMPANION_REPO, f, 'bigvgan_v2_22khz_80band_256x')

WHISPER_SMALL_FILES = [
    'added_tokens.json', 'config.json', 'generation_config.json', 'merges.txt',
    'model.safetensors', 'normalizer.json', 'preprocessor_config.json',
    'special_tokens_map.json', 'tokenizer.json', 'tokenizer_config.json', 'vocab.json',
]
for f in WHISPER_SMALL_FILES:
    download_and_link(COMPANION_REPO, f, 'whisper-small')

# Cleanup HF cache to save disk space
import shutil
for d in [os.path.join(MODEL_DIR, '.cache'), os.path.join(TMP_DIR, '.cache'), '/kaggle/tmp/hf_home']:
    if os.path.exists(d):
        try:
            shutil.rmtree(d)
        except Exception:
            pass

os.system('df -h /kaggle/working /kaggle/tmp')
print('\n✅ All downloads and links complete!')