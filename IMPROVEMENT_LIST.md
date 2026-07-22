# 改进清单（Improvement Roadmap）
## 基线（MVP）成功后的升级项

> **本轮策略（2026-07-22）**：Tier 1 已加进代码并随训练启用；**Tier 2/3 只记录、不训练**（按用户决定）。
> This round: Tier 1 is coded and enabled in training; Tier 2/3 are documented only, not trained (per user decision).
>
> Task1 baseline 已跑完：**val Dice 0.904 / IoU 0.840 / HD95 24.75**（test-local 0.901/0.836/24.90，泛化好）。
> Task1 baseline done: val Dice 0.904 / IoU 0.840 / HD95 24.75.

每项标注：**预期增益**、**成本**、**风险**、**状态**。改进单独跑 val，记 with/without，进报告消融表。

---

## Tier 1 — 已加进代码并启用 / Coded & enabled

| # | 改进项 | 模块 | 预期增益 | 成本 | 风险 | 状态 |
|---|---|---|---|---|---|---|
| 1 | **Tversky 边界损失**（BCE+Dice+Tversky） | Task1 | Hausdorff 降 | 低 | 低 | ✅ 已用，Task1 训练中 |
| 2 | **TTA**（4 翻转平均） | Task1/2 | IoU +0.5~1 | 极低 | 低 | ✅ 已加代码，Task1 已跑 TTA 推理 |
| 3 | **Task2 Focal Loss + 稀疏类平衡采样** | Task2 | 稀疏类 recall +5~10% | 低 | 低 | ✅ 已加代码，随 Task2 训练启用 |
| 4 | **更强增强**（RandomGamma+CLAHE） | Task1/2 | Dice +0.3~0.5 | 低 | 低 | ✅ 已用 |
| 5 | **预处理缓存**（DullRazor 只跑一次） | 全局 | 训练快 3-5x | 低 | 低 | ✅ 已用（2430 图缓存） |
| 6 | **早停 + 断点续训**（last.pth） | Task1/2 | 防崩丢进度 | 低 | 低 | ✅ 已用（Task1 崩过自动续） |
| 7 | **SAM 边界精修**（box prompt，冻结 SAM） | Task1 | IoU +0.5~1，HD95 降 | 中 | 中（显存/崩） | ⚠️ 代码就绪，未跑（重、易崩，留作可选） |
| 8 | **DINOv2 检索 + Task1 mask 裁剪**（双特征对比） | Bonus | 检索相关性 +10% | 低 | 低 | ✅ 已加代码（bonus_clip 支持 dinov2+裁剪） |

## Tier 2 — 仅记录，本轮不训练 / Documented only, not trained

| # | 改进项 | 模块 | 预期增益 | 成本 | 风险 | 依据 |
|---|---|---|---|---|---|---|
| 9 | **PEFT-SAM 当主力**（冻结 encoder，训 decoder+LoRA） | Task1 | IoU +1~2 | 中高 | 中（8GB 需 ViT-B） | PEFT-MedSAM 2026.06 IoU 0.8918 |
| 10 | **模型集成**（SegFormer + PEFT-SAM + Mamba 掩码平均） | Task1 | IoU +0.5~1.5 | 中 | 中 | 集成稳提分 |
| 11 | **Mamba 消融**（UltraLBM-UNet，0.034M 参数） | Task1 | 对比/可能 +0.5 | 低 | 低 | CVPR 2026 SOTA 轻量 |
| 12 | **多任务联合训练**（Task1+Task2 共享 encoder） | Task1/2 | 各 +0.5~1 | 中 | 中 | 表示共享互益 |
| 13 | **更大 backbone**（B2→B3 或 Swin） | Task1/2 | Dice +0.3~0.8 | 中 | 中（显存） | — |
| 14 | **DermINO backbone** 做检索/属性特征 | Bonus/Task2 | +1~2 | 中 | 中 | DermINO 2025.08 皮肤科 FM |
| 15 | **RAG 措辞增强 + 小 LLM 润色**（一致性后置校验） | Task3/Bonus | 可读性↑（不影响一致性分） | 中 | 中 | MMed-RAG ICLR2025 |
| 16 | **测试时自适应 TTA**（CM-TTA 思路，用 SAM 非 SAM3） | Task1 | 分布漂移 +1~2 | 中 | 中 | SAM3 被否决 |

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
