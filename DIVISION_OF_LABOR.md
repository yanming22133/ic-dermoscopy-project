# 分工认领表（Division of Labor — Self-Signup）
## Group 2 · IC DSI Summer School Project

> **说明**：本表只固定 Mengzhe Yang 的技术角色（代码与模型训练只能由其执行）。其余 5 个角色**不署名，请组员自行认领**。每个角色都有评分可见的产出，并能在最终 Q&A 中独立答辩——没有挂名。
>
> **认领方式**：在"认领人"一栏填你的名字即可。建议结合自身背景选（背景匹配度见各角色"适合背景"）。多人想认领同一角色时，由组长协调或拆分子任务。
>
> 认领截止：**7/22 晚**（D2 结束前），以便 7/23 各就各位。

---

## 角色总览

| # | 角色 | 认领人 | 适合背景 | 工作量 | 是否需写代码 |
|---|---|---|---|---|---|
| R0 | 技术/实验总负责 | **Mengzhe Yang**（固定） | 信息管理与信息系统 | 极高 | 是（全部） |
| R1 | 文献综述负责人 | _________（待认领） | Applied AI / CS / 数学 | 高 | 否 |
| R2 | 评估与指标负责人 | _________（待认领） | 数学 / 统计 / 数据科学 | 中高 | 否（可用 Excel/Sheets） |
| R3 | 数据与可视化负责人 | _________（待认领） | 大数据 / 数据分析 / DS | 中 | 否（绘图工具即可） |
| R4 | 临床内容 + 报告主笔 | _________（待认领） | 任何背景，英文写作强 | 中高 | 否 |
| R5 | 项目负责人 + 汇报设计 | _________（待认领） | 管理 / 经济 / 任何 | 中 | 否 |

---

## 各角色详情

### R0 · 技术/实验总负责 — Mengzhe Yang（固定）
- **产出**：全部代码——Task1/2/3/Bonus 模型训练、pipeline、一键推理脚本；产出原始结果（CSV/mask/log）放到共享文件夹喂给队友。
- **Q&A 答辩**：模型选型、训练细节、消融实验、bug 与排查。
- **约束**：单卡（RTX 4070 8GB）单 coder；7/29 前所有模型训完、推理脚本在 val 上验证。

### R1 · 文献综述负责人
- **产出**：阅读 `papers/` 文件夹 30+ 篇论文（已按 task 分类、标注星级，见 `papers/INDEX.md`）；撰写**文献综述**（**7/27 要交**）；论证模型选型依据；与技术负责共同把"为什么选 PEFT-SAM + Mamba 消融"讲清楚。
- **结对备份**：能读懂并一键跑通推理脚本（技术 Q&A 帮腔 + 单点故障备份）。
- **Q&A 答辩**："为什么选这些模型"——你读过的论文。
- **关键论文**：PEFT-MedSAM、MambaLiteUNet、SkinSAM、CKTG、MMed-RAG（见 INDEX 标 ★★★ 者）。
- **交付节点**：7/26 综述初稿，7/27 终稿提交。

### R2 · 评估与指标负责人
- **产出**：设计评估协议——Task1 用 IoU/Dice/**Hausdorff95**/边界 F-score，Task2 用逐属性 F1/AUPRC；从技术负责给的原始输出**建结果表**（baseline vs 改进 vs 消融）；做 error analysis（失败案例分析）；设计 Bonus retrieval impact 对比方案（with/without RAG）。
- **Q&A 答辩**：指标含义、逐属性表现、误差分析。
- **工具**：Excel/Google Sheets + 简单 Python 脚本（技术负责可帮跑）。
- **交付节点**：7/28 评估协议定稿，7/29 最终结果表。

### R3 · 数据与可视化负责人
- **产出**：EDA——类别分布、属性稀疏度统计、图像尺寸/质量分析；制作**报告 + PPT 的全部图表**（IoU 对比柱状图、混淆矩阵、分割样例对比、检索可视化、消融表）。所有图要美观、统一风格。
- **Q&A 答辩**：数据特性、图表解读。
- **工具**：matplotlib/seaborn 或 Excel；技术负责提供原始数据。
- **交付节点**：7/25 EDA 完成，7/28 图表初版，7/29 终版。

### R4 · 临床内容 + 报告主笔
- **产出**：5 个皮镜术语（pigment network, negative network, streaks, milia-like cysts, globules）的医学释义与诊断意义；撰写**≤12 页技术报告**（Methods/Results/Discussion/Future work 四段式）；与技术负责共同设计 Task3 报告的规则逻辑与一致性 checklist。
- **Q&A 答辩**：临床动机、术语解释、讨论与局限。
- **关键**：英文写作要扎实；临床内容准确（查文献 + 术语表）。
- **交付节点**：7/26 临床内容初稿，7/29 报告终稿。

### R5 · 项目负责人 + 汇报设计
- **产出**：项目时间线管理、每日 15 分钟 sync 协调、提交与复现性 checklist；**PPT 设计与叙事主线**；Q&A 题库（预判评委问题 + 准备答案）；整合各人产出成最终汇报。
- **Q&A 答辩**：项目流程、整体叙事、结论。
- **关键**：协调能力 + PPT 审美；掌控 15 分钟节奏。
- **交付节点**：7/28 PPT 初稿，7/30 终稿 + 排练。

---

## 15 分钟汇报分工（每人讲自己产出的部分）

| # | 模块 | 时长 | 主讲角色 |
|---|---|---|---|
| 1 | 题目、临床动机、5 术语 | 2:00 | R4 临床/报告 |
| 2 | 文献综述 | 2:30 | R1 文献 |
| 3 | 数据分析（分布/稀疏度） | 2:00 | R3 数据/可视化 |
| 4 | 方法：pipeline + Task1/2/3 + Bonus | 5:00 | **R0 技术（Mengzhe）** |
| 5 | 结果、评估、demo | 2:30 | R2 评估 |
| 6 | 结论、未来工作、致谢团队 | 1:00 | R5 项目/汇报 |
| | 合计 | 15:00 | |
| QA | 5 分钟全员站台 | 5:00 | 全员各自答本模块 |

---

## 协作机制

- **共享文件夹**：技术负责产出的 `results/`（CSV/mask/log）实时同步，R2/R3/R4 各取所需做表/图/正文，R5 整合 PPT。**代码与写作/图表互不阻塞**。
- **每日 sync**（15 分钟，建议每晚）：对齐三个 task 的输出接口（mask 格式、JSON schema）——接口不对齐是流水线最大返工风险。
- **接口契约**（7/22 前定死，严格对齐 tutorial + example_result）：
  - **Task1 mask**：`{image_id}.jpg`（或 png，待助教确认），二值（0/255）。指标：Dice / IoU / **95% Hausdorff**（tutorial p46 必交）。
  - **Task2**：`{image_id}/{attribute}.png` × 5，二值。presence 由 **mean(sigmoid(logits)) over lesion ROI** 派生（tutorial p40，非覆盖率）；status 阈值：p≥0.60 present / p≤0.40 absent / 中间 uncertain。
  - **Task3 JSON**：严格匹配 `example_result/task3/json/000001.json`（`image_id, split, model_version, attributes_order[5], outputs.presence.{attr}.{prob,status}`）。教程阈值硬编码：border_irregularity=perimeter²/(4π·area)，≥1.60 irregular；lesion_area_ratio=lesion_pixels/total_pixels，<0.08 small / 0.08–0.25 moderate / >0.25 large。**字段歧义**：scoring 提 recommendation 等，按 example 执行，7/22 问助教。
  - **Bonus CSV**：`query_image, neighbor_id, similarity`。**Audit Completeness**（全图都有结果、neighbor ID 全存）+ **Sanity**（无重复 image ID、K 一致）= 评分硬指标（tutorial p46）。CLIP 冻结特征提取，不微调；RAG 不抄邻居事实。
  - **预处理**（tutorial p35）：DullRazor 去毛 + Shades of Gray 颜色恒常 + 归一化 + Resize，所有 task 共用。
  - **Split**：确定性 80/10/10，固定种子（tutorial p56）。

---

## 认领回执（请填后回传）

| 角色 | 认领人 | 联系方式（可选） |
|---|---|---|
| R1 文献综述 | _________ | _________ |
| R2 评估指标 | _________ | _________ |
| R3 数据可视化 | _________ | _________ |
| R4 临床+报告 | _________ | _________ |
| R5 项目+汇报 | _________ | _________ |

> 认领截止 7/22 晚。有任何角色想拆分或合并，提出来一起商量。
