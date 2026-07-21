# 改进清单（Improvement Roadmap）
## 基线（MVP）成功后再尝试的升级项 = Stretch

> **与 PROJECT_PLAN.md §3.6 的关系**：本清单即 **Stretch 层**。**MVP（SegFormer T1 + 6类SegFormer T2 + 规则 T3 + CLIP Bonus）必须在 7/27 前全跑通**，否则不碰本清单任何一项。
>
> 原则：MVP 达标后按下表逐项尝试。每项标注**预期增益**、**成本**、**风险**，按"高性价比优先"排序。时间不够就砍底部。每项改进单独跑 val，记 with/without，进报告消融表。7/28 后冻结，不再上新项。

---

## Tier 1 — 高性价比，成功后优先做

| # | 改进项 | 模块 | 预期增益 | 成本 | 风险 | 依据 |
|---|---|---|---|---|---|---|
| 1 | **SAM encoder 加 LoRA**（不只训 decoder） | Task1 | IoU +0.5~1.5 | 低（LoRA 参数极少） | 低 | PEFT-MedSAM 路线延伸 |
| 2 | **TTA 扩展**：加多尺度+多旋转平均 | Task1 | IoU +0.5~1 | 极低（仅推理） | 低 | 通用技巧 |
| 3 | **边界损失调权**：Tversky/Boundary Loss 权重网格搜索 | Task1 | Hausdorff 显著降 | 低 | 低 | 冲 Best Seg |
| 4 | **Task2 稀疏类过采样 + Focal γ 调参** | Task2 | 稀疏类 recall +5~10% | 低 | 低 | 稀疏属性是硬骨头 |
| 5 | **DINOv2 检索 + Task1 mask 裁剪**（双特征对比） | Bonus | 检索相关性 +10% | 低 | 低 | DINOv2 医学检索常胜 |
| 6 | **模型集成**：PEFT-SAM + SegFormer + Mamba mask 平均 | Task1 | IoU +0.5~1.5 | 中（推理×3） | 中 | 集成稳提分 |

## Tier 2 — 中等性价比，时间允许再做

| # | 改进项 | 模块 | 预期增益 | 成本 | 风险 | 依据 |
|---|---|---|---|---|---|---|
| 7 | **MedSAM ViT-H 全量 PEFT**（替代 ViT-B） | Task1 | IoU +0.5~1 | 高（显存紧，需 batch 1+梯度累积） | 高 OOM | PEFT-MedSAM 原配置 |
| 8 | **多任务联合训练**：Task1+Task2 共享 encoder 双解码器 | Task1/2 | 两 task 各 +0.5~1 | 中（重写训练循环） | 中 | 表示共享有益 |
| 9 | **DermINO backbone** 做检索/属性特征 | Bonus/Task2 | 检索/属性 +1~2 | 中（需下载权重） | 中（权重可用性） | DermINO 2025.08 皮肤科 FM |
| 10 | **RAG 措辞增强 + 小 LLM 润色**（带一致性后置校验） | Task3/Bonus | 报告可读性↑（不影响一致性分） | 中 | 中（一致性风险，需 checklist 兜底） | MMed-RAG ICLR2025 |
| 11 | **测试时自适应 TTA**（CM-TTA 思路，但用 SAM 而非 SAM3） | Task1 | 分布漂移下 +1~2 | 中 | 中 | SAM3 被否决，用 SAM 版本 |

## Tier 3 — 高风险/高成本，仅冲奖且时间富余

| # | 改进项 | 模块 | 预期增益 | 成本 | 风险 | 依据 |
|---|---|---|---|---|---|---|
| 12 | **扩散模型分割**（MLFFM-SegDiff 思路） | Task1 | 可能 +1~2 | 极高（训练慢、8GB 风险） | 极高 | MLFFM-SegDiff 2026.06 Dice 0.9207 |
| 13 | **半监督**：用测试集无标签图做 Mean-Teacher/MIRA-U | Task1/2 | 分布适应 +1~3 | 高 | 中（规则允许性需确认） | MIRA-U 2025 |
| 14 | **知识图增强属性检测**（CKTG 思路） | Task2 | AUC +1~3 | 高（需建属性关系图） | 高 | CKTG TNNLS 2025 AUC 88.6% |
| 15 | **CLIP 零样本属性探测**做 Task2 无监督 ablation | Task2 | 亮点（非提分） | 低 | 低 | 创新性 ablation |

---

## 执行纪律

1. **基线达标前不动 Tier 1**——先保证 PROJECT_PLAN.md 全跑通。
2. **每项改进单独跑一次 val**，记录 with/without 指标，进报告消融表（评委看深度）。
3. **7/28 后冻结模型**，不再上 Tier 2/3 新项，只调 Tier 1。
4. **改进不达预期就回滚**，别让一项实验拖垮时间线。
5. 所有改进的负面结果也写进报告（"tried X, no gain because Y"）= 评委加分项。

> 一句话：基线稳 → Tier 1 全做 → 时间够碰 Tier 2 → 冲奖且显存/时间富余才考虑 Tier 3。
