# 改进清单（Improvement Roadmap）
## 基线（MVP）成功后的升级项

> **本轮策略（2026-07-22）**：Tier 1 已加进代码并随训练启用；**Tier 2/3 只记录、不训练**（按用户决定）。
> This round: Tier 1 is coded and enabled in training; Tier 2/3 are documented only, not trained (per user decision).
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
| 10 | **SAM 边界精修**（box prompt，冻结 SAM） | Task1 | IoU +0.5~1，HD95 降 | 中 | ⚠️ 代码就绪，未跑（重，需先下 SAM 权重） |
| 11 | **集成推理**（多 ckpt 概率平均） | Task1 | IoU +0.5~1.5 | 中 | ✅ 代码就绪（--ensemble） |
| 12 | **SegFormer B1/B2/B3 选项** | Task1/2 | B3 vs B2 +0.3~0.8 | 中 | ✅ 代码就绪（--variant b1/b2/b3） |
| 13 | **DINOv2 检索 + 病灶裁剪**（双特征对比） | Bonus | 相关性 +10% | 低 | ✅ 代码就绪（bonus_clip --encoder dinov2） |

## Tier 2 — 仅记录，本轮不训练 / Documented only, not trained

| # | 改进项 | 模块 | 预期增益 | 成本 | 风险 | 依据 |
|---|---|---|---|---|---|---|
| 14 | **PEFT-SAM 当主力**（冻结 encoder，训 decoder+LoRA） | Task1 | IoU +1~2 | 中高 | 中 | PEFT-MedSAM 2026.06 IoU 0.8918 |
| 15 | **Mamba 消融**（UltraLBM-UNet，0.034M 参数，需 mamba_ssm） | Task1 | 可能 +0.5 | 低 | 高（Win 缺 mamba_ssm） | CVPR 2026 SOTA 轻量 |
| 16 | **多任务联合训练**（Task1+Task2 共享 encoder） | Task1/2 | 各 +0.5~1 | 中 | 中 | 表示共享互益 |
| 17 | **DermINO backbone** 做检索/属性特征 | Bonus/Task2 | +1~2 | 中 | 中（需下权重） | DermINO 2025.08 皮肤科 FM |
| 18 | **RAG 措辞增强 + 小 LLM 润色**（一致性后置校验） | Task3/Bonus | 可读性↑ | 中 | 中（需 LLM） | MMed-RAG ICLR2025 |
| 19 | **测试时自适应 TTA**（CM-TTA 思路，SAM 版） | Task1 | 分布漂移 +1~2 | 中 | 中 | 复杂 |

## Tier 3 — 仅记录，本轮不训练 / Documented only, not trained

| # | 改进项 | 模块 | 预期增益 | 成本 | 风险 | 依据 |
|---|---|---|---|---|---|---|
| 17 | **扩散模型分割**（MLFFM-SegDiff 思路） | Task1 | 可能 +1~2 | 极高 | 极高（8GB 训练慢） | MLFFM-SegDiff 2026.06 Dice 0.9207 |
| 18 | **MedSAM ViT-H 全量 PEFT** | Task1 | +0.5~1 | 极高 | 高（OOM） | PEFT-MedSAM 原配置 |
| 19 | **半监督**（测试集无标签图 Mean-Teacher/MIRA-U） | Task1/2 | 分布适应 +1~3 | 高 | 中（规则允许性需确认） | MIRA-U 2025 |
| 20 | **知识图增强属性检测**（CKTG 思路） | Task2 | AUC +1~3 | 高 | 高 | CKTG TNNLS 2025 AUC 88.6% |

---

## 执行纪律

1. **Tier 1 已随训练启用**；Tier 2/3 本轮只记录不训练（按决定）。
2. 每项改进单独跑 val，记 with/without，进报告消融表（评委看深度）。
3. 改进不达预期就回滚；负面结果也写进报告（"tried X, no gain because Y"）= 加分项。
4. 7/28 后冻结模型，不再上新项。

> 一句话：本轮 Tier 1 全做（已启用），Tier 2/3 留作后续冲奖或 future work。
