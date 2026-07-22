# AutoDL 4090 配置指南 / AutoDL 4090 Setup Guide

> 中文 + English bilingual. 预计总耗时约 15-20 分钟（含环境 + 下载 + 解压数据）。
> Total estimated time: ~15-20 min (setup + downloads + data extraction).

---

## 1. 租机器 / Rent a Machine

- **GPU**: RTX 4090 x 1 (24GB VRAM), 数据盘 50GB+
- **镜像 / Image**: PyTorch 2.x + CUDA 12.x（AutoDL 社区镜像，预装 torch/transformers/timm）
  - 推荐 / Recommended: `PyTorch 2.3.0 + CUDA 12.1 + Python 3.10`
- **计费 / Pricing**: ~2-3 RMB/hour (4090)
- **预计总时长 / Estimated total runtime**: 
  - PEFT-SAM 训练 ~2h / PEFT-SAM training ~2h
  - SegFormer-B3 训练 ~1h / B3 training ~1h
  - ConvNeXt 训练 ~1h / ConvNeXt training ~1h
  - **合计约 4-6 小时，花费约 8-18 RMB / Total ~4-6 hours, ~8-18 RMB**

---

## 2. 配环境 / Environment Setup (~3 min)

```bash
# 克隆项目 / Clone project
git clone https://github.com/yanming22133/ic-dermoscopy-project
cd ic-dermoscopy-project

# 安装缺失依赖（AutoDL 镜像已有 torch, transformers, timm, scipy, opencv-python）
# Install missing dependencies (AutoDL image already has torch, transformers, timm, scipy, opencv-python)
pip install albumentations faiss-cpu -q

# 如果缺什么再补装 / If anything is missing, install:
# pip install transformers timm scipy tqdm pandas matplotlib opencv-python -q

# 可选：PEFT-SAM 的 LoRA 支持（pip install peft）
# Optional: LoRA support for PEFT-SAM
pip install peft -q
```

---

## 3. 下权重 / Download Pretrained Weights (~2 min)

```bash
# 国内用户设置 HF 镜像加速 / Set HF mirror for China access
export HF_ENDPOINT=https://hf-mirror.com

# 下载全部预训练权重：SegFormer B0/B1/B2/B3 + SAM ViT-B + DINOv2
# Download all pretrained weights
python scripts/download_weights.py

# 验证 / Verify:
ls pretrained/sam-vit-base/model.safetensors   # SAM (~360MB)
ls pretrained/segformer-b3-ade/model.safetensors  # SegFormer B3 (~180MB)
```

> **注意 / Note**: AutoDL 服务器网速很快（~50-100 MB/s），下载 ~2GB 权重约 30 秒-1 分钟。
> AutoDL server has fast internet (~50-100 MB/s), downloading ~2GB takes ~30s-1min.

---

## 4. 传数据 / Upload Data

```bash
# 方式 1: AutoDL 网页上传（推荐，稳定）
# 在 AutoDL 控制台 - 文件管理 - 上传 summer_school_project_train.zip（~10GB）

# 方式 2: scp 上传
# scp summer_school_project_train.zip root@<your-instance-ip>:/root/autodl-tmp/

# 解压 / Extract
unzip summer_school_project_train.zip -d /root/autodl-tmp/

# 确认数据结构 / Verify data structure:
ls summer_school_project_train/train/images/*.jpg | wc -l   # 应有 ~2700 张图
```

---

## 5. 配置项目路径 / Configure Project Paths

编辑 `src/config.py`，修改数据目录为 AutoDL 实际路径：
Edit `src/config.py`, update data directory to AutoDL actual path:

```python
# AutoDL 常用路径 / Common AutoDL paths:
PROJ_DIR = r'/root/autodl-tmp/ic-dermoscopy-project'
DATA_DIR = os.path.join(PROJ_DIR, 'summer_school_project_train', 'train')
```

> **技巧 / Tip**: 数据放 `/root/autodl-tmp/` 下读写更快（系统盘是 SSD）。
> Put data under `/root/autodl-tmp/` for faster I/O (system disk is SSD).

---

## 6. 开跑 / Run Training

### 6.1 冒烟测试（先跑 20 张确认环境 OK）/ Smoke Test (run 20 images first)

```bash
# PEFT-SAM 冒烟 / PEFT-SAM smoke test
python -u -m src.train_task1_sam --epochs 1 --batch 2 --size 1024 --max_train 20 --out outputs/smoke_sam

# SegFormer 冒烟 / SegFormer smoke test
python -u -m src.train_task1 --model segformer --variant b0 --epochs 1 --batch 2 --max_train 20 --out outputs/smoke_b0
```

### 6.2 正式训练 / Full Training

```bash
# ====== PEFT-SAM (2026 SOTA 路线) ======
# 冻结 SAM encoder，训练 mask_decoder + LoRA。1024x1024，batch=8，约 2 小时。
# Freeze SAM encoder, train mask_decoder + LoRA. 1024x1024, batch=8, ~2 hours.
python -u -m src.train_task1_sam \
    --epochs 50 --batch 8 --size 1024 --num_workers 8 \
    --lora --boundary_loss --cosine_lr --freq_loss --ch_attn \
    --out outputs/task1_sam

# ====== SegFormer-B3 + 全开（对比基线） ======
# B3 backbone，512x512，batch=16，约 1 小时。
# B3 backbone, 512x512, batch=16, ~1 hour.
python -u -m src.train_task1 \
    --model segformer --variant b3 \
    --epochs 50 --batch 16 --size 512 --num_workers 8 \
    --boundary_loss --cosine_lr --freq_loss --ch_attn \
    --out outputs/task1_b3_full

# ====== ConvNeXt-Base + FPN（CNN 视角对比） ======
# CNN backbone 天然高频感知强，512x512，batch=12，约 1 小时。
# CNN backbone naturally strong at HF, 512x512, batch=12, ~1 hour.
python -u -m src.train_task1 \
    --model convnext --variant base \
    --epochs 50 --batch 12 --size 512 --num_workers 8 \
    --cosine_lr \
    --out outputs/task1_convnext

# ====== DeepLabV3+ ResNet101 ======
python -u -m src.train_task1 \
    --model deeplab \
    --epochs 50 --batch 16 --size 512 --num_workers 8 \
    --cosine_lr \
    --out outputs/task1_deeplab
```

### 6.3 训练命令速查表 / Training Commands Quick Reference

| 模型 / Model | 分辨率 / Size | Batch | VRAM | 预估时间 / Est. Time |
|---|---|---|---|---|
| **PEFT-SAM** | 1024 | 8 | ~20GB | ~2h |
| **SegFormer-B3** | 512 | 16 | ~16GB | ~1h |
| **ConvNeXt-Base** | 512 | 12 | ~18GB | ~1h |
| **DeepLabV3+ R101** | 512 | 16 | ~14GB | ~1h |
| **SegFormer-B2** | 512 | 16 | ~10GB | ~45m |

完整命令见 6.2 节 / Full commands in section 6.2.

> `-u` 标志禁用 Python 输出缓冲，确保 AutoDL 日志实时可见。
> `-u` flag disables Python output buffering, ensures real-time log visibility.

---

## 7. 结果文件 / Output Files

每个实验的输出目录 `outputs/{name}/` 下：

| 文件 / File | 用途 / Purpose |
|---|---|
| `best.pth` | 最佳 checkpoint（按 val Dice）/ Best checkpoint (by val Dice) |
| `last.pth` | 最近 checkpoint（断点续训用）/ Latest checkpoint (for crash resume) |
| `best_metrics.json` | 最佳轮次指标 / Best epoch metrics |
| `final_val_metrics.json` | 最终 val Dice/IoU/HD95 |
| `final_testlocal_metrics.json` | 最终 test-local Dice/IoU/HD95 |

---

## 8. 断点续训 / Crash Recovery

训练意外中断后，加 `--resume` 即可从 `last.pth` 继续：
After unexpected interruption, add `--resume` to continue from `last.pth`:

```bash
# PEFT-SAM 续训 / PEFT-SAM resume
python -u -m src.train_task1_sam --resume --out outputs/task1_sam

# SegFormer 续训 / SegFormer resume
python -u -m src.train_task1 --resume --model segformer --variant b3 --out outputs/task1_b3
```

---

## 9. 常见问题 / FAQ

**Q: HF 下载很慢？/ HF download is slow?**

```bash
export HF_ENDPOINT=https://hf-mirror.com  # 国内镜像 / China mirror
```

**Q: CUDA OOM？**

- 减小 `--batch` 或增大 `--accum_steps`（等效 batch = batch x accum_steps）
- PEFT-SAM 在 4090 上 batch=8 @ 1024 刚好，降到 4 如果 OOM

**Q: PEFT-SAM 训练很慢（>4h）？**

- 确认 `--size 1024` 且没被改小
- 关闭 `--boundary_loss`（phi 距离图计算是 CPU 瓶颈）

**Q: 没有 peft 包？/ No peft package?**

```bash
pip install peft -q
# 不装也能跑，只是没有 LoRA（只训练 mask_decoder，约 4M 参数）
# Works without peft, just no LoRA (trains only mask_decoder, ~4M params)
```

---

## 10. 快速启动脚本 / Quick Start Script

保存为 `run_all.sh`，一键跑所有实验：
Save as `run_all.sh`, run all experiments with one command:

```bash
#!/bin/bash
set -e
export HF_ENDPOINT=https://hf-mirror.com

# 冒烟测试 / Smoke test
echo "=== Smoke Test ==="
python -u -m src.train_task1_sam --epochs 1 --batch 2 --size 1024 --max_train 20 --out outputs/smoke

# 正式训练 / Full training (按需取消注释)
echo "=== PEFT-SAM (SOTA) ==="
python -u -m src.train_task1_sam --epochs 50 --batch 8 --size 1024 --num_workers 8 --lora --boundary_loss --cosine_lr --freq_loss --ch_attn --out outputs/task1_sam

echo "=== SegFormer-B3 ==="
python -u -m src.train_task1 --model segformer --variant b3 --epochs 50 --batch 16 --size 512 --num_workers 8 --boundary_loss --cosine_lr --freq_loss --ch_attn --out outputs/task1_b3_full

echo "=== ConvNeXt-Base ==="
python -u -m src.train_task1 --model convnext --variant base --epochs 50 --batch 12 --size 512 --num_workers 8 --cosine_lr --out outputs/task1_convnext

echo "All done!"
```
