import os, gc, psutil

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,garbage_collection_threshold:0.6'
os.environ['MALLOC_TRIM_THRESHOLD_'] = '0'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

print(f'RAM: {psutil.virtual_memory().total / 1024**3:.1f} GB total, {psutil.virtual_memory().available / 1024**3:.1f} GB available')
os.system('df -h /kaggle/working')
print('✅ Environment ready!')
