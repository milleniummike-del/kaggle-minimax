import os, gc, psutil

print('=== Kaggle T4 Environment Setup ===')
print(f'RAM: {psutil.virtual_memory().total / 1024**3:.1f} GB total, {psutil.virtual_memory().available / 1024**3:.1f} GB available')

os.system('echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null 2>&1')
os.system('echo 1 | sudo tee /proc/sys/vm/overcommit_memory > /dev/null 2>&1')
gc.collect()

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,garbage_collection_threshold:0.6'
os.environ['MALLOC_TRIM_THRESHOLD_'] = '0'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

print('✅ Environment optimized!')