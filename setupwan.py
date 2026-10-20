import os
import sys

REPO_DIR = 'Wan2GP'
if not os.path.exists(REPO_DIR):
    print('\n📥 Cloning Wan2GP repository...')
    subprocess.run(['git', 'clone', '--depth', '1', 'https://github.com/DeepBeepMeep/Wan2GP.git', REPO_DIR], check=True)
else:
    # Drop patches from earlier runs; the current ones are re-applied below
    subprocess.run(['git', '-C', REPO_DIR, 'reset', '-q', '--hard'], check=True)
    print('\n📂 Wan2GP repository already present.')

# Kaggle's PyTorch is kept (no downgrade). Wan2GP's requirements pin an onnxruntime-gpu dev build
# that is not on PyPI (it makes the whole install fail part-way), so that line uses the released package.
print('📦 Installing Wan2GP requirements...')
with open(f'{REPO_DIR}/requirements.txt') as f:
    reqs = [('onnxruntime-gpu>=1.22.0' if line.startswith('onnxruntime-gpu==') and 'dev' in line else line.rstrip('\n'))
            for line in f]
with open('wan2gp_requirements.txt', 'w') as f:
    f.write('\n'.join(reqs) + '\n')
subprocess.run([sys.executable, '-m', 'pip', 'install', '--timeout', '120', '--retries', '5',
                '-q', '--no-warn-conflicts', '-r', 'wan2gp_requirements.txt'], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '--timeout', '120', '--retries', '5',
                '-q', '--no-warn-conflicts', 'gradio>=5.0.0', 'gguf', 'soundfile'], check=True)
print('✅ Dependencies installed.')

# --- Programmatic Patch for older ffmpeg compatibility ---
import os
import shutil

video_decode_path = 'Wan2GP/shared/utils/video_decode.py'
if os.path.exists(video_decode_path):
    print(f"Patching {video_decode_path} for older ffmpeg compatibility...")
    with open(video_decode_path, 'r') as f:
        content = f.read()
    
    target_str = 'cmd += ["-fps_mode", "passthrough", "-frames:v", str(requested_frames), "-f", "rawvideo", "-pix_fmt", out_pix_fmt, "pipe:1"]'
    replacement_str = """    # Dynamic check for -fps_mode vs -vsync support
    fps_mode_opt = "-fps_mode"
    try:
        import subprocess
        res = subprocess.run([ffmpeg_path, "-h"], capture_output=True, text=True, errors="ignore")
        if "fps_mode" not in res.stdout and "fps_mode" not in res.stderr:
            fps_mode_opt = "-vsync"
    except Exception:
        pass
    out_pix_fmt = "gbrpf32le" if hdr_linear else "rgb24"
    cmd += [fps_mode_opt, "passthrough", "-frames:v", str(requested_frames), "-f", "rawvideo", "-pix_fmt", out_pix_fmt, "pipe:1"]"""
    
    if target_str in content:
        content = content.replace(target_str, replacement_str)
        with open(video_decode_path, 'w') as f:
            f.write(content)
        print("Successfully patched video_decode.py!")
    else:
        print("Warning: target string not found in video_decode.py!")

plugin_path = 'Wan2GP/plugins/wan2gp-motion-designer/plugin.py'
if os.path.exists(plugin_path):
    print(f"Patching {plugin_path} for older ffmpeg compatibility...")
    with open(plugin_path, 'r') as f:
        content = f.read()
    
    target_plugin = '                    **{\n                        "vsync": "cfr",\n                        "fps_mode": "cfr",\n                        "fflags": "+genpts",\n                        "copyts": None,\n                    },'
    replacement_plugin = '                    **{\n                        "vsync": "cfr",\n                        "fflags": "+genpts",\n                        "copyts": None,\n                    },'
    
    if target_plugin in content:
        content = content.replace(target_plugin, replacement_plugin)
        with open(plugin_path, 'w') as f:
            f.write(content)
        print("Successfully patched plugin.py!")
    elif '"fps_mode": "cfr",' in content:
        content = content.replace('"fps_mode": "cfr",', '')
        with open(plugin_path, 'w') as f:
            f.write(content)
        print("Successfully removed fps_mode from plugin.py (fallback)!")
    else:
        print("Warning: target plugin string not found in plugin.py!")