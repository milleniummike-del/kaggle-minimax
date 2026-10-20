import subprocess
try:
    subprocess.run(['nvidia-smi'], check=True)
    print('GPU Active!')
except Exception:
    print('WARNING: No GPU. Go to Settings → Accelerator → GPU T4 x1')

!git clone https://github.com/DeepBeepMeep/Wan2GP.git /kaggle/working/Wan2GP

!pip install -q torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 \
    --index-url https://download.pytorch.org/whl/cu121

!pip install --timeout 120 --retries 5 -q -r /kaggle/working/Wan2GP/requirements.txt
!pip install --timeout 120 --retries 5 -q mmgp gradio gguf soundfile ffmpeg-python
print('✅ Wan2GP ready!')