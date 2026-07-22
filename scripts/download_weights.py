"""下载 SegFormer 预训练权重到 pretrained/（Windows 友好，用 local_dir 避免符号链接权限问题）。
Download SegFormer pretrained weights into pretrained/ (Windows-friendly,
uses local_dir to avoid symlink-permission issues).

用法 / Usage:
  f:\\anacondaenvs\\pytorch\\python.exe scripts/download_weights.py
"""
import os
from huggingface_hub import snapshot_download

os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')

REPOS = {
    'segformer-b0-ade': 'nvidia/segformer-b0-finetuned-ade-512-512',
    'segformer-b1-ade': 'nvidia/segformer-b1-finetuned-ade-512-512',
    'segformer-b2-ade': 'nvidia/segformer-b2-finetuned-ade-512-512',
    'segformer-b3-ade': 'nvidia/segformer-b3-finetuned-ade-512-512',
    'sam-vit-base': 'facebook/sam-vit-base',  # Tier1: SAM 边界精修用 / for SAM boundary refinement
}

if __name__ == '__main__':
    root = os.path.join(os.path.dirname(__file__), '..', 'pretrained')
    for sub, repo in REPOS.items():
        dst = os.path.normpath(os.path.join(root, sub))
        print(f'downloading {repo} -> {dst}')
        snapshot_download(repo_id=repo, local_dir=dst,
                          allow_patterns=['*.json', '*.safetensors', '*.bin', '*.txt'])
        print('OK', dst)
