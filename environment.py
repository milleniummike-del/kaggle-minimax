import os, gc, psutil

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,garbage_collection_threshold:0.6'
os.environ['MALLOC_TRIM_THRESHOLD_'] = '0'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

print(f'RAM: {psutil.virtual_memory().total / 1024**3:.1f} GB total, {psutil.virtual_memory().available / 1024**3:.1f} GB available')
os.system('df -h /kaggle/working')
print('✅ Environment ready!')

import shutil
import subprocess
import sys

print("Python:", sys.version)

nvidia_smi = shutil.which("nvidia-smi")

if nvidia_smi:
    subprocess.run([nvidia_smi], check=False)
else:
    print("⚠️ nvidia-smi executable not found.")

try:
    import torch

    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("GPU count:", torch.cuda.device_count())

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("⚠️ PyTorch cannot access a CUDA GPU.")

except ImportError:
    print("PyTorch is not installed in this Python environment.")
