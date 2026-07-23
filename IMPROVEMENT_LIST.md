# 改进清单（精简版 / Simplified）
## PEFT-SAM epoch 31: val Dice 0.9441（4090 训练中）

---

## 已落地（训练报告消融表用 / For report ablation）

| # | 项 | 效果 |
|---|---|---|
| 1 | PEFT-SAM ViT-B 替代 SegFormer-B2 | Dice 0.904 → **0.9441** |
| 2 | Boundary Loss + Tversky + freq_loss 三损失栈 | HD95 降 + 频域补盲 |
| 3 | 余弦退火 LR + 早停 + 断点续训 | 收敛更稳 |
| 4 | 预处理缓存（DullRazor 一次，推理走缓存） | 推理不卡 CPU |
| 5 | Task2: Focal + 平衡采样 + 边缘损失 (edge_loss) | 稀疏类 recall↑ |
| 6 | 文献综述 7 个 Future Directions → 4 个已代码化 | 答辩加分 |

---

## 跑完 PEFT-SAM 后立即做（不重训，推理侧免费提分 / Inference-only, free gains）

| # | 项 | 命令 | 预期 |
|---|---|---|---|
| R1 | **TTA + 多尺度 TTA + 后处理** | `--tta 1 --ms_tta 1 --postproc 1` | Dice +0.3~0.8 |
| R2 | **HD95 最终评估** | `infer_task1 --split val`（代码已自动报） | 拿边界数字 |
| R3 | **Task2 推理** | `infer_task2` → atmosphere.json → report_task3 | 出 presence + 报告 |

---

## 做完 Task1/2/3 后（有时间才做 / Optional, time permitting）

| # | 项 | 成本 |
|---|---|---|
| O1 | LoRA PEFT（`model_sam.py --lora` 已代码化，接一下线） | 中 |
| O2 | EWT 跨频融合（`ewt_fusion.py` 已代码化，接 SAM decoder 出口） | 中 |
| O3 | Task2 属性关系图后处理（`--attr_rules 1`） | 低 |

---

## 放最终报告 Future Work 讨论（不需要代码 / Report future work only）

| 方向 | 文献依据 |
|---|---|
| Mamba-Transformer 混合架构 | LEDNet-SwinUMamba, MambaLiteUNet |
| 统一 seg+attr+report 端到端 | 文献综述 Future Direction #5 |
| 跨数据集外部验证 (PH2/HAM10000) | ISIC 2018 Challenge 分析 |
