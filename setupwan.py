
import os
import sys
import subprocess
import shutil

REPO_URL = "https://github.com/DeepBeepMeep/Wan2GP.git"
REPO_DIR = "/kaggle/working/Wan2GP"

def run_command(command, description):
    print(f"\n{'=' * 60}")
    print(f"🚀 {description}")
    print(f"{'=' * 60}")
    subprocess.run(command, check=True)

# 2. Clone or update Wan2GP
if os.path.isdir(os.path.join(REPO_DIR, ".git")):
    print("📦 Wan2GP repository already exists.")
    run_command(
        ["git", "-C", REPO_DIR, "pull", "--ff-only"],
        "Updating Wan2GP"
    )
else:
    if os.path.exists(REPO_DIR):
        shutil.rmtree(REPO_DIR)

    run_command(
        ["git", "clone", "--depth", "1", REPO_URL, REPO_DIR],
        "Cloning Wan2GP"
    )

# 3. Install PyTorch CUDA 12.1 build
# NOTE: This version may conflict with newer Kaggle packages.
# Only use it if your Wan2GP setup requires these exact versions.
run_command(
    [
        sys.executable, "-m", "pip", "install",
        "torch==2.3.1",
        "torchvision==0.18.1",
        "torchaudio==2.3.1",
        "--index-url", "https://download.pytorch.org/whl/cu121"
    ],
    "Installing PyTorch"
)

# 4. Install repository requirements
requirements = os.path.join(REPO_DIR, "requirements.txt")

if os.path.isfile(requirements):
    run_command(
        [
            sys.executable, "-m", "pip", "install",
            "--timeout", "120",
            "--retries", "5",
            "-r", requirements
        ],
        "Installing Wan2GP requirements"
    )
else:
    raise FileNotFoundError(f"Missing requirements file: {requirements}")

# 5. Install additional dependencies
run_command(
    [
        sys.executable, "-m", "pip", "install",
        "--timeout", "120",
        "--retries", "5",
        "mmgp", "gradio", "gguf",
        "soundfile", "ffmpeg-python"
    ],
    "Installing additional dependencies"
)

# 6. Verify PyTorch and CUDA
run_command(
    [
        sys.executable, "-c",
        "import torch; "
        "print('PyTorch:', torch.__version__); "
        "print('CUDA available:', torch.cuda.is_available()); "
        "print('GPU count:', torch.cuda.device_count()); "
        "print('GPU:', torch.cuda.get_device_name(0) "
        "if torch.cuda.is_available() else 'None')"
    ],
    "Verifying installation"
)

print("\n✅ Wan2GP setup complete!")
print(f"📂 Repository: {REPO_DIR}")
