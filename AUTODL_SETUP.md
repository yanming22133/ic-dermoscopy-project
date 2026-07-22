# AutoDL 4090 配置指南 / AutoDL 4090 Setup Guide

## 租机器 / Rent GPU

1. AutoDL (autodl.com) → 选 **RTX 4090 × 1**
2. 镜像选 **PyTorch 2.x + CUDA 12.x**（预装 torch/torchvision/transformers）
3. 数据盘 50GB（够用）
4. 计费：约 **2-3 元/小时**

## 配环境（3 分钟）/ Setup (3 min)

```bash
# 1. Clone 代码（git 已预装）
git clone https://github.com/yanming22133/ic-dermoscopy-project
cd ic-dermoscopy-project

# 2. 装补充依赖（镜像已有 torch/transformers/timm/scipy/opencv）
pip install albumentations faiss-cpu diffusers -q

# 如果需要 LoRA：
pip install peft -q

# 3. 设 HF 镜像（国内加速）
export HF_ENDPOINT=https://hf-mirror.com

# 4. 下载预训练权重 (~2 分钟)
python scripts/download_weights.py
# 这会下载: SegFormer B0/B1/B2/B3 + SAM ViT-B + DINOv2
```

## 传数据 / Upload Data

在 AutoDL 网页控制台上传 `summer_school_project_train.zip`（约 10GB），然后：
```bash
unzip summer_school_project_train.zip
# 解压后：summer_school_project_train/train/images/  (2700 jpg)
#         summer_school_project_train/train/task1_gt/
#         summer_school_project_train/train/task2_gt/
```

## 训练命令 / Training Commands

约 4-6 小时跑完全部。推荐按顺序跑：

### 第 1 批：主力模型（并行对比）

```bash
# PEFT-SAM (2026 SOTA, ~2h)
python -u -m src.train_task1 --model peft_sam --epochs 50 --batch 8 \
  --size 1024 --num_workers 8 --boundary_loss --cosine_lr \
  --freq_loss --ch_attn --patience 10 --out outputs/task1_sam &
PID1=$!

# SegFormer-B3 全开 (对比, ~1h)
python -u -m src.train_task1 --model segformer --variant b3 \
  --epochs 50 --batch 16 --size 512 --num_workers 8 \
  --boundary_loss --cosine_lr --freq_loss --ch_attn \
  --patience 10 --out outputs/task1_b3_full &
PID2=$!

# ConvNeXt (CNN backbone 对比, ~1h)
python -u -m src.train_task1 --model convnext --variant base \
  --epochs 50 --batch 12 --size 512 --num_workers 8 \
  --cosine_lr --patience 10 --out outputs/task1_convnext &
PID3=$!

wait $PID1 $PID2 $PID3
```

### 第 2 批：Task2 属性检测

```bash
python -u -m src.train_task2 --model segformer --variant b2 \
  --epochs 50 --batch 16 --size 512 --num_workers 8 \
  --cosine_lr --balanced --patience 10 --out outputs/task2_b2
```

### 第 3 批：推理 + 集成

```bash
# Task1 集成推理 (TTA + 多尺度 + 后处理 全开)
python -u -m src.infer_task1 \
  --ensemble outputs/task1_b3_full/best.pth,outputs/task1_b2/best.pth,outputs/task1_convnext/best.pth \
  --split val --tta 1 --ms_tta 1 --postproc 1 \
  --save_dir submit/task1

# Task2 推理
python -u -m src.infer_task2 --ckpt outputs/task2_b2/best.pth \
  --task1_mask_dir submit/task1 --split val \
  --tta 1 --ms_tta 1 --save_dir submit/task2

# Task3 报告
python -u -m src.report_task3 \
  --task1_mask_dir submit/task1 \
  --presence submit/task2/presence.json \
  --save_dir submit/task3 --split test

# Bonus
python -u -m src.bonus_clip build --encoder clip
python -u -m src.bonus_clip retrieve --encoder clip \
  --image_dir <test_dir> --task1_mask_dir submit/task1 \
  --save_dir submit/bonus --k 3
```

### 可选：扩散增强（需额外装 diffusers + 下载 SD 2.1 权重 ~5GB）

```bash
# 推理时加扩散精修
python -u -m src.infer_task1 --ckpt outputs/task1_b3_full/best.pth \
  --split val --postproc 1 --diffusion_refine 1

# 训练时加潜空间流形损失（仅 4090，8GB 会 OOM）
python -u -m src.train_task1 --model segformer --variant b3 \
  --epochs 50 --batch 8 --size 512 --num_workers 4 \
  --boundary_loss --cosine_lr --freq_loss --ch_attn \
  --diffusion_loss --patience 10 --out outputs/task1_b3_sd
```

## 预计耗时 / Expected Time

| 任务 | 模型 | 时间 |
|---|---|---|
| Task1 | PEFT-SAM | ~2h |
| Task1 | SegFormer-B3 全开 | ~1h |
| Task1 | ConvNeXt | ~1h |
| Task2 | SegFormer-B2 | ~1h |
| 推理+集成 | — | ~10min |
| Bonus | CLIP 索引+检索 | ~5min |
| **合计** | | **约 5-6h** |

## 费用 / Cost

4090 ≈ 2-3 RMB/h × 6h ≈ **15-20 元**

## 从笔记本迁移的 checkpoint

如果已经有了本地训练的 B2 checkpoint，可以上传继续用：
```bash
# 上传 outputs/task1_b2/best.pth 到 AutoDL
# 然后参与集成推理：
python -u -m src.infer_task1 \
  --ensemble outputs/task1_b2/best.pth,outputs/task1_b3_full/best.pth \
  --split val --postproc 1
```

## 注意事项 / Notes

1. HF 镜像：`export HF_ENDPOINT=https://hf-mirror.com`（国内必须设）
2. 扩散模型首次下载 SD 2.1 约 5GB，需等几分钟
3. PEFT-SAM 的 SAM 权重已在 download_weights.py 中（360MB），已下好
4. 训练完成后 **记得把生成物下载到本地**（AutoDL 关机后数据盘会清除）
5. 40

[TRUNCATED]