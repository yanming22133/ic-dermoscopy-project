# 改进清单（Improvement Roadmap）
## 基线（MVP）成功后的升级项

> **本轮策略（2026-07-22）**：Tier 1 已加进代码并随训练启用；PEFT-SAM 已移入 Tier 1（代码就绪，待在 4090 上训练）；**Tier 2/3 只记录、不训练**（按用户决定）。
> This round: Tier 1 is coded and enabled in training; PEFT-SAM moved to Tier 1 (code ready, pending 4090 training); Tier 2/3 are documented only, not trained (per user decision).
>
> Task1 baseline 已跑完：**val Dice 0.904 / IoU 0.840 / HD95 24.75**（test-local 0.901/0.836/24.90，泛化好）。
> Task1 baseline done: val Dice 0.904 / IoU 0.840 / HD95 24.75.

每项标注：**预期增益**、**成本**、**风险**、**状态**。改进单独跑 val，记 with/without，进报告消融表。

---

## Tier 1 — 已加进代码 / Coded

| # | 改进项 | 模块 | 预期增益 | 成本 | 状态 |
|---|---|---|---|---|---|
| 1 | **Tversky 边界损失**（BCE+Dice+Tversky） | Task1 | HD95 降 | 低 | ✅ Task1 已训 |
| 2 | **TTA**（翻转 + 多尺度 0.8/1/1.2×）| Task1/2 | IoU +0.5~1.5 | 低 | ✅ 代码就绪 |
| 3 | **Task2 Focal + 稀疏类平衡采样** | Task2 | 稀疏类 recall +5~10% | 低 | ✅ 代码就绪 |
| 4 | **更强增强**（RandomGamma+CLAHE） | Task1/2 | Dice +0.3~0.5 | 低 | ✅ Task1 已用 |
| 5 | **预处理缓存**（DullRazor 跑一次，推理走缓存） | 全局 | 训练 3-5x，推理快 | 低 | ✅ 已用 |
| 6 | **早停 + 断点续训 + 梯度累积**（last.pth 每轮存） | Task1/2 | 防崩丢进度 | 低 | ✅ 已用 |
| 7 | **余弦退火 LR**（CosineAnnealingLR） | Task1/2 | Dice +0.3~0.5 | 低 | ✅ 代码就绪（--cosine_lr） |
| 8 | **后处理**（形态学开闭 + 最大连通域） | Task1 | Dice +0.1~0.5，HD95 降 | 低 | ✅ 代码就绪（--postproc），**不重训** |
| 9 | **Boundary Loss**（水平集距离图，Kervadec） | Task1 | HD95 显著降 | 中 | ✅ 代码就绪（--boundary_loss） |
| 10 | **PEFT-SAM 当主力**（冻结 SAM encoder，训 mask_decoder+LoRA，box prompt 来自 GT mask） | Task1 | IoU +1~2，HD95 显著降 | 中 | ✅ 代码就绪，待在 4090 上训练（24GB 可跑 batch=8 @ 1024） |
| 11 | **集成推理**（多 ckpt 概率平均） | Task1 | IoU +0.5~1.5 | 中 | ✅ 代码就绪（--ensemble） |
| 12 | **SegFormer B1/B2/B3 选项** | Task1/2 | B3 vs B2 +0.3~0.8 | 中 | ✅ 代码就绪（--variant b1/b2/b3） |
| 13 | **DINOv2 检索 + 病灶裁剪**（双特征对比） | Bonus | 相关性 +10% | 低 | ✅ 代码就绪（bonus_clip --encoder dinov2） |
| 14 | **🆕 频域解耦损失**（Haar DWT：LL Dice + HH MSE，分频单独优化形状/边界） | Task1 | HD95 显著降（直接填 ViT 高频短板） | 中 | 低 | DDA 频域方法论逆向应用：把高频边界梯度独立注入 ViT |
| 15 | **🆕 通道注意力**（SE/ECA，解码器每层加通道重标定，~百参数） | Task1/2 | IoU +0.2~0.5 | 极低 | 低 | 特征降维精炼，抑制噪声通道 |
| 16 | **🆕 DWT 数据增强**（训练时随机 DWT 分解→调频带权重→逆变换，强制模型不依赖单一频带） | Task1/2 | Dice +0.3~0.5，泛化↑ | 低 | 低 | DDA I1 多尺度 DWT 思路反向 |
| 17 | **🆕 ConvNeXt/Swin backbone**（现代 CNN，对 2700 张数据更友好，天然高频感知强于 ViT） | Task1/2 | HD95 ↓ + IoU ↑ | 中 | 低 | 有预训练权重 |

## Tier 2 — 仅记录，本轮不训练 / Documented only, not trained

| # | 改进项 | 模块 | 预期增益 | 成本 | 风险 | 依据 |
|---|---|---|---|---|---|---|
| 18 | **Mamba 消融**（UltraLBM-UNet，需 mamba_ssm） | Task1 | 可能 +0.5 | 低 | 高（Win） | CVPR 2026 |
| 19 | **多任务联合训练**（Task1+Task2 共享 encoder） | Task1/2 | 各 +0.5~1 | 中 | 中 | 表示共享互益 |
| 20 | **多尺度特征金字塔**（SegFormer 四层特征再融合，保留低频细节） | Task1 | HD95↓ IoU↑ | 中 | 中 | FPN 类结构补偿 1/4 分辨率损失 |
| 21 | **DermINO backbone** | Bonus/Task2 | +1~2 | 中 | 中 | 皮肤科 FM 2025 |
| 22 | **RAG 措辞增强** | Task3/Bonus | 可读性↑ | 中 | 中 | MMed-RAG ICLR2025 |
| 23 | **MedSAM ViT-H PEFT**（ViT-H 版 SAM，从 Tier 3 升级，4090 可跑） | Task1 | +0.5~1 | 中高 | 中 | PEFT-MedSAM 原配置，4090 24GB 刚好 |
| 24 | **扩散模型推理精修**（冻结扩散 U-Net，交叉注意力边界引导，从 Tier 3 升级，4090 推理可行） | Task1 | 边界锐利↑ | 中 | 中 | DDA 论文潜空间牵引逆向用 |

## Tier 3 — 理论方向，高成本 / Theoretical, high cost

| # | 改进项 | 模块 | 预期增益 | 成本 | 风险 | 依据 |
|---|---|---|---|---|---|---|
| 25 | **扩散模型分割**（MLFFM-SegDiff，需 4090+ 大显存，训练慢） | Task1 | +1~2 | 极高 | 极高 | MLFFM-SegDiff 2026.06 |
| 26 | **VAE 潜空间流形约束**（冻结 VAE 编码器 → latent 特征对齐，4090 或可尝试） | Task1 | 边界一致性↑ | 高 | 中（4090 可行） | DDIM 降维思路 |
| 27 | **半监督**（测试集无标签图 Mean-Teacher/MIRA-U） | Task1/2 | 分布适应 +1~3 | 高 | 中 | MIRA-U 2025 |
| 28 | **知识图增强属性检测**（CKTG 思路） | Task2 | AUC +1~3 | 高 | 高 | CKTG TNNLS 2025 AUC 88.6% |

---

## 执行纪律

1. **Tier 1 已随训练启用**；Tier 2/3 本轮只记录不训练（按决定）。
2. 每项改进单独跑 val，记 with/without，进报告消融表（评委看深度）。
3. 改进不达预期就回滚；负面结果也写进报告（"tried X, no gain because Y"）= 加分项。
4. 7/28 后冻结模型，不再上新项。

> 一句话：本轮 Tier 1 全做（已启用），Tier 2/3 留作后续冲奖或 future work。
