# 改进清单（Improvement Roadmap）
## SOTA 论文驱动的重构版（2026-07-23）

> 基于 papers_sota/ 5 篇 2025-2026 顶会/顶刊论文：
> WA-NET (Sci Reports 2025, Dice 0.9458)、LEDNet-SwinUMamba (Nature Sci Reports 2026, Dice 0.9753)、
> SAM-Adapter (Diagnostics 2026, Dice 0.9427)、MambaLiteUNet (CVPR 2026, HD95 10.55)、
> LSF-Mamba (Algorithms 2026, 0.049M 参数)
>
> **PEFT-SAM epoch 5: val Dice 0.9362（训练中）**

---

## Tier 1 — 已在代码或训练中

| # | 改进项 | 来源 | 状态 |
|---|---|---|---|
| 1 | PEFT-SAM ViT-B（冻结 encoder，训 mask_decoder，GT bbox prompt） | SAM-Adapter 同类 PEFT 路线 | 🔄 4090 训练中，epoch 5 Dice 0.936 |
| 2 | freq_loss（Haar DWT：LL→Dice，HH→MSE） | WA-NET EWT 思路，但未做跨频融合 | ✅ 已启用 |
| 3 | Boundary Loss（水平集 signed distance） | — | ✅ 已启用 |
| 4 | Tversky（β 偏 recall，少漏病灶） | — | ✅ 已启用 |
| 5 | 余弦退火 LR + 早停 + 断点续训 | — | ✅ 已启用 |
| 6 | TTA（翻转）+ 多尺度 TTA（0.8/1/1.2×） | — | ✅ 代码就绪 |
| 7 | 后处理（形态学 + 最大连通域） | SAM-Adapter 形态学精修思路 | ✅ 代码就绪 |
| 8 | Task2 Focal + 稀疏类平衡采样 | — | ✅ 代码就绪 |

---

## Tier 2 — 高收益、可立即实施（等 PEFT-SAM 跑完后加）

| # | 改进项 | 论文依据 | 怎么做 | 预期 | 成本 |
|---|---|---|---|---|---|
| A1 | **边缘监督损失** | WA-NET composite loss（BCE+Dice+edge） | 在现有 loss 上加一条 Sobel 边缘 MSE：对 pred 和 GT 分别做 Sobel → MSE。几行代码 | HD95 ↓ 3-5 | 极低 |
| A2 | **跨频小波融合（EWT 简化版）** | WA-NET EWT：DWT→通道注意力→跨频拼接 | 对 SAM 256² 解码特征做一次 Haar DWT，把 HH/LH/HL 用 1×1 Conv 融合后加回 LL，再接 1×1 Conv | Dice +0.3-0.5 | 中 |
| A3 | **交叉门控注意力（CGA）** | MambaLiteUNet CVPR 2026 | 在 SAM decoder 的 skip 连接上加一个轻量门控——对浅层特征做 sigmoid gate，乘到深层特征上，只 100 参数 | HD95 ↓ 2-3 | 低 |
| A4 | **形态学边界精修（推理侧）** | SAM-Adapter morphological refinement | 推理时对 mask 边界做高斯平滑→重新阈值化。不改训练。 | HD95 ↓ 3-5 | 极低 |
| A5 | **SAM decoder 出口加 DWT 残差** | LSF-Mamba FRE（小波分解+门控残差注入） | 在 SAM mask_decoder 的输出 logits 前，对 decoder 256² 特征做一层 Haar DWT→1×1 Conv→sigmoid gate→乘回原特征 | 边界锐利↑ | 中 |

---

## Tier 3 — 架构级改进（重训一次，冲 0.96+）

| # | 改进项 | 论文依据 | 怎么做 | 预期 | 成本 |
|---|---|---|---|---|---|
| B1 | **独立边界检测支路（BRM）** | WA-NET BRM：Sobel 边缘提取+注意力融合 | 从 SAM 中间特征拉一条支路，做 Sobel 边缘检测→和主路融合 | HD95 ↓ 5-10 | 中高 |
| B2 | **LEDNet 式边缘引导** | LEDNet：Siamese 边缘检测→引导主分割 | 用一个微型 CNN（3 层 Conv）专门学病灶边缘，输出作为加权 mask | Dice +1-2，HD95 大幅降 | 高 |
| B3 | **Mamba decoder 替换 SAM 的 Transformer decoder** | LEDNet-SwinUMamba / LSF-Mamba | SAM mask_decoder（2 层 Transformer）换成 2 层 VSS 状态空间块 | 参数量↓，边界↑ | 高（需 mamba_ssm, Linux OK） |
| B4 | **Swin backbone 替换 ViT-B** | LEDNet-SwinUMamba MambaLiteUNet | 用 Swin-Tiny 作为第二编码器，和 SAM ViT-B 特征做交叉注意力融合 | Dice +1-2 | 高 |

---

## Tier 4 — 冲顶（0.97+），需大改架构

| # | 改进项 | 论文依据 |
|---|---|---|
| C1 | **LEDNet + SAM 混合** | LEDNet-SwinUMamba 的 Siamese 结构移植到 SAM：SAM 做全局，LEDNet 做边缘，跨注意力融合 |
| C2 | **Mamba + ViT 双编码器** | LEDNet-SwinUMamba 双 backbone 范式：Mamba→局部连续性，ViT→全局依赖 |
| C3 | **全频带 DWT 融合管线** | 结合 WA-NET EWT + LSF-Mamba FRE + 你的 DDA I4 中频偏置——SOTA 三合一频域方案 |

---

## 执行纪律

1. **等 PEFT-SAM 跑完** → 拿最终 Dice/IoU/HD95 做基线
2. **A1-A4 一起加**（4 个都是低成本、不改模型结构），重训一次验证收益
3. **A5 + B1** 如果 HD95 还高于 15，重点攻
4. Tier 3/4 只在 7/28 前富余时考虑

> 对标：当前 PEFT-SAM 趋势 0.94+，加 Tier 2（A1-A4）→ 0.945-0.95（追 WA-NET），加 Tier 3（B1-B2）→ 0.95-0.96，Tier 4 → 接近 0.97（顶尖）。
