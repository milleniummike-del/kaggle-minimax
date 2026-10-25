print("Running cell1.py")
%cd /kaggle/working
import os
import sys
import gc
import psutil
import subprocess
import warnings

# 1. High-Performance Environment & Protobuf Settings
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,garbage_collection_threshold:0.6"
os.environ["MALLOC_TRIM_THRESHOLD_"] = "0"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

print('=== Kaggle GPU T4 x2 Environment Setup ===')
ram = psutil.virtual_memory()
print(f'RAM: {ram.total / 1024**3:.1f} GB total, {ram.available / 1024**3:.1f} GB available')

# Drop OS cache to maximize RAM
os.system('echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null 2>&1')
os.system('echo 1 | sudo tee /proc/sys/vm/overcommit_memory > /dev/null 2>&1')
gc.collect()

# 2. Verify GPU Accelerator
try:
    smi = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'], text=True).strip()
    print(f'✅ GPU Detected:\n{smi}')
except Exception:
    print('WARNING: No GPU. Go to Settings → Accelerator → GPU T4 x2')

# 3. Wan2GP pinned to the version this notebook was built and tuned on (13 May 2026).
# Newer Wan2GP releases changed the LTX-2 code path: with this GGUF they overflow in FP16 (black
# video) and need slow BF16 instead. The pinned commit gives the original speed and lip-sync quality.
REPO_DIR = 'Wan2GP'
WAN2GP_URL = 'https://github.com/DeepBeepMeep/Wan2GP.git'
WAN2GP_COMMIT = 'b8880bd5d292edfbe2384f5c951d95757ad59ea0'

def _git(*args):
    return subprocess.run(['git', '-C', REPO_DIR, *args], check=True, capture_output=True, text=True).stdout.strip()

if not os.path.exists(REPO_DIR):
    os.makedirs(REPO_DIR)
    _git('init', '-q')
    _git('remote', 'add', 'origin', WAN2GP_URL)
head = subprocess.run(['git', '-C', REPO_DIR, 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
if head != WAN2GP_COMMIT:
    print(f'\n📥 Fetching Wan2GP @ {WAN2GP_COMMIT[:10]} (May 2026)...')
    _git('fetch', '-q', '--depth', '1', 'origin', WAN2GP_COMMIT)
    _git('checkout', '-q', '-f', 'FETCH_HEAD')  # also resets earlier runtime patches, re-applied below
    print('✅ Wan2GP ready.')
else:
    print(f'\n📂 Wan2GP @ {WAN2GP_COMMIT[:10]} already present.')
    _git('reset', '-q', '--hard')  # drop runtime patches from earlier runs; the current ones are re-applied below

# 4. Install Dependencies
# Wan2GP's own requirements first (ffmpeg-python, imageio, rembg, opencv, mmgp 3.7.6, ... are
# imported by the pipeline), then the extras this notebook uses on top. The May list pins an
# onnxruntime-gpu dev build that is no longer on PyPI, so that one line uses the released package.
print('📦 Installing Wan2GP requirements...')
with open(f'{REPO_DIR}/requirements.txt') as f:
    reqs = [('onnxruntime-gpu>=1.22.0' if line.startswith('onnxruntime-gpu==') and 'dev' in line else line.rstrip('\n'))
            for line in f]
with open('wan2gp_requirements.txt', 'w') as f:
    f.write('\n'.join(reqs) + '\n')
subprocess.run([
    sys.executable, '-m', 'pip', 'install', '--timeout', '120', '--retries', '5',
    '-q', '--no-warn-conflicts', '-r', 'wan2gp_requirements.txt'
], check=True)
REQUIRED_DEPS = [  # mmgp stays at the version Wan2GP pins (3.7.6)
    'gradio>=5.0.0',
    'gguf',
    'soundfile',
    'einops',
    'sentencepiece',
    'hf_transfer',
    'accelerate',
]
subprocess.run([
    sys.executable, '-m', 'pip', 'install', '--timeout', '120', '--retries', '5',
    '-q', '--no-warn-conflicts', *REQUIRED_DEPS
], check=True)
print('✅ Dependencies installed successfully.')

# 5. Apply Wan2GP Core Runtime Patches
print('🔧 Applying Wan2GP runtime patches...')

# Patch 1: ltx2.py runtime fixes
ltx2_py = os.path.join(REPO_DIR, 'models/ltx2/ltx2.py')
if os.path.exists(ltx2_py):
    with open(ltx2_py, 'r') as f:
        content = f.read()
    old_line1 = 'input_video_strength = max(0.0, min(1.0, input_video_strength))'
    new_line1 = 'input_video_strength = max(0.0, min(1.0, input_video_strength)) if input_video_strength is not None else 1.0'
    if old_line1 in content:
        content = content.replace(old_line1, new_line1)
    old_line2 = 'source = source.to(device=self.device, dtype=torch.bfloat16)'
    new_line2 = 'source = source.to(device=self.device, dtype=self.dtype)'
    if old_line2 in content:
        content = content.replace(old_line2, new_line2)
    with open(ltx2_py, 'w') as f:
        f.write(content)
    print('  ✓ Patched Wan2GP ltx2.py routines')

# Patch 2: Spatial Upsampler Dtype Auto-Alignment
upsampler_file = os.path.join(REPO_DIR, 'models/ltx2/ltx_core/model/upsampler/model.py')
if os.path.exists(upsampler_file):
    with open(upsampler_file, 'r') as f:
        ucontent = f.read()
    old_up = 'latent = upsampler(latent)'
    new_up = ('upsampler_dtype = next(upsampler.parameters()).dtype if any(True for _ in upsampler.parameters()) else torch.float16\n'
              '    if latent.dtype != upsampler_dtype:\n'
              '        latent = latent.to(upsampler_dtype)\n'
              '    latent = upsampler(latent)')
    if old_up in ucontent:
        ucontent = ucontent.replace(old_up, new_up)
    with open(upsampler_file, 'w') as f:
        f.write(ucontent)
    print('  ✓ Patched Wan2GP spatial upsampler auto-alignment')

# Patch 3: Video VAE Dtype Auto-Alignment
vae_file = os.path.join(REPO_DIR, 'models/ltx2/ltx_core/model/video_vae/video_vae.py')
if os.path.exists(vae_file):
    with open(vae_file, 'r') as f:
        vcontent = f.read()
    old_v = 'sample = self.conv_in(sample, causal=self.causal)'
    new_v = ('conv_param = next(self.conv_in.parameters(), None)\n'
             '        if conv_param is not None and sample.dtype != conv_param.dtype:\n'
             '            sample = sample.to(conv_param.dtype)\n'
             '        sample = self.conv_in(sample, causal=self.causal)')
    if old_v in vcontent:
        vcontent = vcontent.replace(old_v, new_v)
    old_enc = 'sample = self.conv_in(sample)'
    new_enc = ('enc_param = next(self.conv_in.parameters(), None)\n'
               '        if enc_param is not None and sample.dtype != enc_param.dtype:\n'
               '            sample = sample.to(enc_param.dtype)\n'
               '        sample = self.conv_in(sample)')
    if old_enc in vcontent:
        vcontent = vcontent.replace(old_enc, new_enc)
    with open(vae_file, 'w') as f:
        f.write(vcontent)
    print('  ✓ Patched Wan2GP video VAE auto-alignment')

# Patch 4: Audio VAE Dtype Auto-Alignment
audio_vae_file = os.path.join(REPO_DIR, 'models/ltx2/ltx_core/model/audio_vae/audio_vae.py')
if os.path.exists(audio_vae_file):
    with open(audio_vae_file, 'r') as f:
        acontent = f.read()
    old_a = 'decoded_audio = audio_decoder(latent)'
    new_a = ('decoder_param = next(audio_decoder.parameters(), None)\n'
             '    if decoder_param is not None and latent.dtype != decoder_param.dtype:\n'
             '        latent = latent.to(decoder_param.dtype)\n'
             '    decoded_audio = audio_decoder(latent)\n'
             '    vocoder_param = next(vocoder.parameters(), None)\n'
             '    if vocoder_param is not None and decoded_audio.dtype != vocoder_param.dtype:\n'
             '        decoded_audio = decoded_audio.to(vocoder_param.dtype)')
    if old_a in acontent:
        acontent = acontent.replace(old_a, new_a)
    with open(audio_vae_file, 'w') as f:
        f.write(acontent)
    print('  ✓ Patched Wan2GP audio VAE auto-alignment')

# Patch 5: Vocoder STFT & Filter Dtype Auto-Alignment
vocoder_file = os.path.join(REPO_DIR, 'models/ltx2/ltx_core/model/audio_vae/vocoder.py')
if os.path.exists(vocoder_file):
    with open(vocoder_file, 'r') as f:
        voc_content = f.read()
    old_stft = 'spec = F.conv1d(y, self.forward_basis, stride=self.hop_length, padding=0)'
    new_stft = ('forward_basis = self.forward_basis.to(dtype=y.dtype, device=y.device) if self.forward_basis.dtype != y.dtype else self.forward_basis\n'
                '        spec = F.conv1d(y, forward_basis, stride=self.hop_length, padding=0)')
    if old_stft in voc_content:
        voc_content = voc_content.replace(old_stft, new_stft)
    old_low = 'return F.conv1d(x, self.filter.expand(n_channels, -1, -1), stride=self.stride, groups=n_channels)'
    new_low = ('filt = self.filter.to(dtype=x.dtype, device=x.device) if self.filter.dtype != x.dtype else self.filter\n'
               '        return F.conv1d(x, filt.expand(n_channels, -1, -1), stride=self.stride, groups=n_channels)')
    if old_low in voc_content:
        voc_content = voc_content.replace(old_low, new_low)
    with open(vocoder_file, 'w') as f:
        f.write(voc_content)
    print('  ✓ Patched Wan2GP vocoder auto-alignment')

# Patch 6: Wan2GP Base Text Encoder Device Alignment (Fixes Multi-GPU device mismatch)
# Gemma's hidden states stay in BF16: the feature extractor squares them for RMS normalization, and in
# FP16 any value above 256 overflows (inf -> NaN text features -> black video).
base_enc_file = os.path.join(REPO_DIR, 'models/ltx2/ltx_core/text_encoders/gemma/encoders/base_encoder.py')
if os.path.exists(base_enc_file):
    with open(base_enc_file, 'r') as f:
        bcontent = f.read()
    old_b = 'attention_mask = item.attention_mask.to("cuda")\n        encoded_video_input, encoded_audio_input = _apply_feature_extractor(\n            item.hidden_states,'
    new_b = ('target_dev = next(feature_extractor_linear.parameters()).device if any(True for _ in feature_extractor_linear.parameters()) else torch.device("cuda:0")\n'
             '        attention_mask = item.attention_mask.to(target_dev)\n'
             '        hidden_states = tuple(h.to(device=target_dev) for h in item.hidden_states)\n'
             '        encoded_video_input, encoded_audio_input = _apply_feature_extractor(\n'
             '            hidden_states,')
    if old_b in bcontent:
        bcontent = bcontent.replace(old_b, new_b)
        with open(base_enc_file, 'w') as f:
            f.write(bcontent)
        print('  ✓ Patched Wan2GP base_encoder device & dtype auto-alignment')

# Patch 7: Wan2GP Gemma Feature Extractor & Connector Dtype Auto-alignment
fe_file = os.path.join(REPO_DIR, 'models/ltx2/ltx_core/text_encoders/gemma/feature_extractor.py')
if os.path.exists(fe_file):
    with open(fe_file, 'r') as f:
        fe_content = f.read()
    old_fe = 'normed = _norm_and_concat_per_token_rms(encoded, attention_mask).to(encoded.dtype)\n            v_dim = self.video_aggregate_embed.out_features'
    new_fe = ('v_target_dtype = next(self.video_aggregate_embed.parameters()).dtype if any(True for _ in self.video_aggregate_embed.parameters()) else encoded.dtype\n'
              '            v_target_dev = next(self.video_aggregate_embed.parameters()).device if any(True for _ in self.video_aggregate_embed.parameters()) else encoded.device\n'
              '            normed = _norm_and_concat_per_token_rms(encoded.to(device=v_target_dev), attention_mask.to(v_target_dev)).to(dtype=v_target_dtype)\n'
              '            v_dim = self.video_aggregate_embed.out_features')
    if old_fe in fe_content:
        fe_content = fe_content.replace(old_fe, new_fe)
    old_fe_aud = 'audio = self.audio_aggregate_embed(_rescale_norm(normed, a_dim, self.embedding_dim))'
    new_fe_aud = ('a_target_dtype = next(self.audio_aggregate_embed.parameters()).dtype if any(True for _ in self.audio_aggregate_embed.parameters()) else v_target_dtype\n'
                  '                normed_a = normed.to(dtype=a_target_dtype) if a_target_dtype != v_target_dtype else normed\n'
                  '                audio = self.audio_aggregate_embed(_rescale_norm(normed_a, a_dim, self.embedding_dim))')
    if old_fe_aud in fe_content:
        fe_content = fe_content.replace(old_fe_aud, new_fe_aud)
    with open(fe_file, 'w') as f:
        f.write(fe_content)
    print('  ✓ Patched Wan2GP feature_extractor dtype auto-alignment')

conn_file = os.path.join(REPO_DIR, 'models/ltx2/ltx_core/text_encoders/gemma/embeddings_connector.py')
if os.path.exists(conn_file):
    with open(conn_file, 'r') as f:
        conn_content = f.read()
    old_reg = 'learnable_registers = torch.tile(self.learnable_registers, (num_registers_duplications, 1))'
    new_reg = 'learnable_registers = torch.tile(self.learnable_registers.to(device=hidden_states.device, dtype=hidden_states.dtype), (num_registers_duplications, 1))'
    if old_reg in conn_content:
        conn_content = conn_content.replace(old_reg, new_reg)
        with open(conn_file, 'w') as f:
            f.write(conn_content)
        print('  ✓ Patched Wan2GP embeddings_connector register auto-alignment')

print('✅ Environment & Wan2GP setup complete!')