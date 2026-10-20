import os

DATASET_DIR = '/kaggle/input/datasets/trailtalknick/ltx-23-22b-q4-gguf'
MODEL_DIR = '/kaggle/working/Wan2GP/models'
os.makedirs(MODEL_DIR, exist_ok=True)

# Symlink all safetensors from dataset root
for f in os.listdir(DATASET_DIR):
    if not f.endswith('.safetensors'):
        continue
    src = os.path.join(DATASET_DIR, f)
    dst = os.path.join(MODEL_DIR, f)
    if not os.path.exists(dst):
        os.symlink(src, dst)
        print(f'✅ Linked: {f}')
    else:
        print(f'✓ Already linked: {f}')

# Symlink GGUF from ltx_upload
gguf_src = os.path.join(DATASET_DIR, 'ltx_upload', 'ltx-2.3-22b-distilled-Q4_K_M_light.gguf')
gguf_dst = os.path.join(MODEL_DIR, 'ltx-2.3-22b-distilled-Q4_K_M_light.gguf')
if not os.path.exists(gguf_dst):
    os.symlink(gguf_src, gguf_dst)
    print('✅ Linked: ltx-2.3-22b-distilled-Q4_K_M_light.gguf')
else:
    print('✓ Already linked: gguf')

# Symlink Gemma with correct name from ltx_upload
gemma_src = os.path.join(DATASET_DIR, 'ltx_upload', 'gemma-3-12b-it-qat-q4_0-unquantized')
gemma_dst = os.path.join(MODEL_DIR, 'gemma-3-12b-it-qat-q4_0-unquantized')
if not os.path.exists(gemma_dst):
    os.symlink(gemma_src, gemma_dst)
    print('✅ Linked: gemma-3-12b-it-qat-q4_0-unquantized')
else:
    print('✓ Already linked: gemma')

os.system('df -h /kaggle/working')
print('\n✅ All models linked!')