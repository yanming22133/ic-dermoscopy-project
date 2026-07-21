# IC 夏校项目方案（第二组）

10 天的深度学习项目（7/21–7/31），做皮肤镜图像的三件事：病灶分割、5 种属性检测、锚定报告，外加一个 CLIP 检索的 bonus。我们组 6 个人，目标冲 Best Overall 和 Best Segmentation 两个奖。

下面把要做什么、怎么做、谁来做、时间怎么排，一次讲清楚。

---

## 0. 总体思路

四条主线：

1. Task1（分割）和 Task2（属性）共用一套分割模型骨架。SegFormer-B2 当保底主力，先保证能跑出 IoU 0.87–0.90；PEFT-SAM 当升级候选，两个都训，谁在 val 上分高就用谁；再用 Mamba 做消融对比，用 SAM 做边界精修去冲 Best Seg 奖。
2. Task3（报告）用规则模板写死，不上神经网络——这样 20% 的一致性分能稳拿满。阈值直接用教程给的那套，不自创。
3. Bonus 用 CLIP（再加 DINOv2 做对比）做检索，模型冻结、不微调。重点是 audit 完整：每张图都有结果、neighbor ID 都存下来、不重复、K 值一致。RAG 只用来润色措辞，不抄邻居的事实。
4. 分 MVP 和 Stretch 两档。MVP（SegFormer T1 + 6 类 SegFormer T2 + 规则 T3 + CLIP Bonus）必须在 7/27 前跑通；PEFT-SAM、Mamba、DINOv2、SAM 精修、集成这些是 Stretch，等 MVP 通了再碰。MVP 没通之前，不碰任何 Stretch。

还有一条判断要贯穿始终：模型选型只是入场券（别人也会查同样的论文），真正能赢的是别翻车、三个 task 都不拉胯、Hausdorff 和 error analysis 做得深、汇报讲得好、代码能复现。

所有模型 7/29 前训完，一键推理脚本要在 val 上验证过——因为 7/30 测试集只给 6 小时跑推理，当天不能写代码。

---

## 1. 环境与算力

电脑是 ASUS ROG Zephyrus G16，显卡 RTX 4070 Laptop（8 GB 显存），内存约 31.4 GB。

有个坑要先说：PATH 里的 `python` 是 CPU 版（Python310），不能用。正确的环境是 `f:\anacondaenvs\pytorch\python.exe`，里面的 torch 是 2.11.0+cu128，CUDA 能用，能识别到 4070。VSCode 的解释器要手动选到这个路径。装包用对应的 pip：`f:\AnacondaEnvs\pytorch\Scripts\pip`。

已经装好的：timm 1.0.22、transformers 4.57.3、opencv 4.13、sklearn、OpenAI 的 clip、pandas。还要补装：albumentations、faiss-cpu、open_clip_torch，segmentation_models_pytorch 可选。

8 GB 显存的用法：全程开混合精度（AMP）+ 梯度累积，输入分辨率压到 480 或 512，单模型训练占 3–6 GB，不要同时加载多个模型。

---

## 2. 数据与划分

训练集 2700 张皮肤镜图，640×480，RGB。Task1 的标签是 `task1_gt/*_segmentation.png`，2700 张单通道二值图（值是 0/255）。Task2 的标签是 `task2_gt/*_attribute_{属性}.png`，5 个属性 × 2700 = 13500 张（也是 0/255）。5 个属性是：pigment_network、negative_network、streaks、milia_like_cyst、globules。测试集 7/30 13:00 才发，格式和训练集一样，但没有标签。

**预处理**（教程 p35 要求，皮肤镜专用）：先 DullRazor 去毛发伪影，再 Shades of Gray 做颜色恒常（统一不同设备的色温），然后归一化（zero mean / unit variance，或 ImageNet 均值方差），再 resize 到 480/512。训练前一定要可视化 image 和 mask 的对齐，杜绝错位（教程 p55）。

**数据增强**（albumentations，教程 p35）：旋转、翻转、颜色抖动、缩放、裁剪、弹性形变。Task1 和 Task2 要同步增强，保证 mask 还是对得上的。

**数据划分与泛化策略**（重点）：

2700 张按 80/10/10 切成 train / val / test-local，确定性划分、固定种子（教程 p56 要求确定性 split）。三个集各干各的：

- **train（80%，约 2160 张）**：训练用。
- **val（10%）**：模型选择用。早停、调超参、选 SegFormer 还是 PEFT-SAM、调 Task3 阈值，都反复在 val 上做。因为你反复看它，val 的分数会偏乐观。
- **test-local（10%）**：无偏估计用。只在 7/28–29 跑一次，看真实泛化能力。如果 test-local 比 val 低很多，说明过拟合了，赶紧修。这个集不参与任何调参。

关于"泛化到官方测试集"——官方测试集是从同一个数据集留出来的同分布内部测试，不是另一个医院的图，所以只要不严重过拟合，泛化基本有保障。最大化泛化的办法是：强增强 + 用预训练/基础模型（SegFormer 是 ImageNet 预训练，SAM 在海量数据上训过，迁移学习天然抗漂移）+ Shades of Gray/DullRazor 去域差异 + val 早停别过拟合 + 测试时 TTA。**官方测试集没有标签，绝不用它训练**，7/30 只对它跑推理。

要不要在 train+val 上重训最终模型？可选，但不建议。好处是多用 10% 数据可能多 0.3–0.5 点；坏处是重训后没有 val 做早停了，复杂度增加，6 小时窗口容不下意外。建议直接用 val 选出来的 checkpoint 当最终模型；如果 7/28 一切顺利、时间显存都富余，再考虑重训一个 train+val 候选，用 test-local 比一下择优。

诚实说一句：我们做不到外部验证（没有第二个带 5 属性标签的数据集现成可用），官方测试集是内部测试。这点要写进报告的 limitation 和 future work——评委爱看这种清醒的认知。

---

## 3. 技术方案

### 3.1 Task 1 · 病灶分割（冲 Best Seg，占 25%）

目标：输出二值病灶 mask，把 Dice、IoU、95% Hausdorff 三个指标都拉高（这三个教程 p46 都要求交，Hausdorff 不是可选的）。

2026 年的文献（deep-research 查的，23 条验证通过）显示，ISIC2018 上 SOTA 的 Dice 在 0.91–0.94、IoU 在 0.89–0.91，主流已经转向基础模型 PEFT 微调（PEFT-MedSAM 2026.06 拿到 IoU 0.8918）、Mamba（MambaLiteUNet 进了 CVPR 2026）、扩散模型。但模型选型只是入场券，执行可靠比追趋势更重要。

模型怎么配：

- **保底主力**：SegFormer-B2。D1–2 先跑通，保证 IoU 0.87–0.90 必达，是最终提交的候选。8 GB 能开 batch 16。
- **升级候选**：PEFT 微调 SAM（ViT-B），冻结图像编码器，只训 mask decoder，可选加 LoRA。对标 PEFT-MedSAM。它只有在 val 上超过 SegFormer 才替换主力，否则不用。8 GB 开 batch 2–4。
- **边界精修**：SAM 推理精修 + BiSeg-SAM 那种 DetailRefine 边界模块，专门降 Hausdorff。
- **消融**：UltraLBM-UNet（Mamba，只有 0.034M 参数），2025 SOTA 的轻量对比，几乎零成本。
- **推理增强**：TTA，翻转加旋转平均，白捡 1–2 点 IoU。

损失用 BCE + Dice，边界再上 Tversky 或 Boundary Loss。评估报 Dice、IoU、95% Hausdorff 加边界 F-score，逐图和全局都报。

决策规则很简单：SegFormer 和 PEFT-SAM 都训，在 val 上比分，谁高用谁；PEFT-SAM 不收敛就直接用 SegFormer，不赌时间线。

放弃的几个：nnU-Net（8 GB + 10 天太重）；扩散模型（训练慢、8 GB 风险高，只在未来工作里讨论）；SAM2（研究证实它在医学分割上不普遍强于 SAM，不值得换）；SAM3（deep-research 否决了，2026 年中不可用）。

### 3.2 Task 2 · 属性检测（占 25%）

目标：输出 5 个属性的 mask，外加每个属性的 presence（present / absent / uncertain）。

主路线是 6 类分割：用 SegFormer（和 Task1 同族骨架），num_classes=6（背景 + 5 属性），每个属性输出 per-pixel logits。presence 的概率怎么算？教程 p40 给了定义：对每个属性的 logits 做 sigmoid，然后在**病灶 ROI 里取均值**得 p_attr（病灶 ROI 就是 Task1 预测的病灶 mask 里的像素，这是教程推荐的做法）。注意是均值，不是覆盖率。status 阈值直接用教程的：p ≥ 0.60 是 present，p ≤ 0.40 是 absent，0.40 到 0.60 之间是 uncertain。

稀疏属性兜底：milia_like_cysts 和 streaks 区域特别小，如果召回太低，就加一个共享 backbone 的多标签分类头（5 个 sigmoid）来补 presence（教程 p35 的 dual cls head 就是这思路）；但 mask 证据还是用分割输出，保证报告里"证据对齐"。另外 CLIP 零样本属性探测（文本 prompt 匹配）可以做一个无监督的消融亮点。

损失用稀疏类加权 Dice + Focal，组合损失按教程 Loss = Loss_seg + 0.5·Loss_cls。评估逐属性报 F1、AUPRC、mask 的 Dice/IoU，稀疏类重点看 recall（教程要求逐属性单独报，别只报均值）。

### 3.3 Task 3 · 锚定报告（占 20%，看一致性）

目标：输出一个 JSON 和一段英文报告，5 个术语全出现，文本里的 status 和 JSON 完全一致，证据描述和 mask 对得上。

方案是确定性规则模板，阈值全用教程 p40 给的，不自创：

- 属性 status：p_attr ≥ 0.60 present / ≤ 0.40 absent / 0.40–0.60 uncertain。
- border_irregularity = perimeter² / (4π·area)，area 为 0 时记 0.0；≥ 1.60 是 irregular，< 1.60 是 regular。
- lesion_area_ratio = 病灶像素 / 总像素；< 0.08 small / 0.08–0.25 moderate / > 0.25 large。

输入是 Task1 的 mask（算 size 和 border）和 Task2 的 presence + mask（做证据描述）。输出 JSON 严格照 `example_result/task3/json/000001.json` 的格式（image_id、split、model_version、attributes_order[5]、outputs.presence.每个属性.{prob,status}）；文本用模板拼，5 术语全出现，status 和 JSON 严格一致，照着 example 那句"The lesion is moderate with irregular borders. Pigment network is present; ..."的写法。

一致性 checklist 自动校验（教程 p40）：5 术语齐全、文本 status 等于 JSON status、证据描述和 mask 覆盖一致、JSON 格式合规且数值范围合法且术语正确（教程 p46 的 Format Constraint）。

有个字段歧义要确认：教程 p46 的 scoring 提到 lesion_presence / attributes_detected / findings_summary / recommendation 这几个字段，但 example 的 JSON 只有 attributes_order / outputs.presence。我们按 example 执行，7/22 问助教到底要不要 recommendation 字段。

为什么不用神经网络生成报告？因为 Task3 没有训练数据，它就是 Task1/2 输出的一个确定性函数。上神经网络反而可能漏术语、改 status，危及那 20% 的一致性分。RAG 只做措辞增强，而且事后要校验，绝不抄邻居的事实（教程 p43 的要求）。

### 3.4 Bonus · CLIP 检索 + RAG（看 audit 完整性）

检索部分（教程 p41–42，模型冻结不微调）：用 CLIP ViT-B/32 把全库 2700 张训练图编码成向量（N×512 float32），建 FAISS 余弦索引，测试图来了取 Top-K 近邻。升级对比可以再加 DINOv2 ViT-B/14（密集视觉相似，医学检索常赢）。检索前先用 Task1 的 mask 把病灶裁出来再编码，召回质量会明显提升，这也把 Task1 和 Bonus 串起来了。输出严格照 `test_bonus_clip.csv`：query_image, neighbor_id, similarity。

评分重点要对齐教程 p46，别自创：一是 Audit Completeness，每张测试图都得有检索结果、neighbor ID 全存；二是 Sanity Check，不能有重复 image ID、K 值要一致、similarity 得合法。这两条是硬指标，全图覆盖、不重复、K 一致，比"证明检索有用"更关键。

RAG 措辞增强（教程 p43）：拿 Top-K 近邻的 GT 报告做模板聚合，但要点是 grounded——不抄邻居的事实，事实必须和当前图自己的 mask 一致。汇报里可以展示 with/without RAG 的对比当亮点，但打分还是看 audit/sanity。

---

### 3.5 文献证据（deep-research 2025–2026，23 条验证通过）

| 模块 | 文献 | 对方案的影响 |
|---|---|---|
| Task1 候选 | PEFT-MedSAM (2026.06) IoU 0.8918，超 U-Net 和 zero-shot；SkinSAM/BiSeg-SAM 同族 | PEFT-SAM 当升级候选，SegFormer 保底 |
| Task1 消融 | MambaLiteUNet (CVPR 2026) Dice 93.09%；UltraLBM-UNet 0.034M 参数 | 加 Mamba 轻量消融 |
| Task1 放弃 | 扩散训练慢；SAM2 不普遍强于 SAM；SAM3 被否决 | 明确放弃并说理由 |
| Task2 属性 | CKTG+GD-DDW (TNNLS 2025.08) EDRA AUC 88.6% | 当 SOTA 参照；我们用 mask 监督做 evidence-grounded 检测 |
| Bonus RAG | MMed-RAG (ICLR 2025)、RadAlign (MICCAI 2025)：CLIP+FAISS | 验证 CLIP+FAISS 路线对 |
| 评估 | 多数论文只报 Dice/IoU，Hausdorff 很少报 | 报 Hausdorff 是差异化（而且教程必交） |
| 基础模型可选 | DermINO (2025.08) 皮肤科基础模型 432K 图 | 权重能用就当检索 backbone，DINOv2 兜底 |

老实说，这些大多是 arXiv 预印本，只有 CKTG（TNNLS 2025）和 MambaLiteUNet（CVPR 2026）是确认录用的。各论文的 benchmark split 不一样，指标不能直接跨论文比。ISIC2018 的 SOTA Dice 大概聚在 0.91–0.94。

---

### 3.6 MVP / Stretch（管住 scope）

| 层级 | 内容 | 截止 |
|---|---|---|
| MVP（必达标底线） | SegFormer Task1 + 6 类 SegFormer Task2 + 规则 Task3 + CLIP Bonus | 7/27 前全跑通 |
| Stretch（MVP 通了才上） | PEFT-SAM、Mamba 消融、DINOv2 对比、SAM 边界精修、集成、TTA 扩展 | 7/28–29 |
| 铁律 | MVP 没通前不碰任何 Stretch；7/28 后冻结，不上 Stretch 新项 | — |

---

## 4. 交付物与格式（照 example_result）

```
submit/
├── task1/              # 病灶 mask，命名照 example：000001.jpg
├── task2/              # 每张图一个目录：000001/{属性}.png
├── task3/
│   ├── json/000001.json
│   └── Summary_reports_text.csv   # image_id, findings
├── bonus/
│   └── test_bonus_clip.csv        # query_image, neighbor_id, similarity（全图、不重复、K 一致）
└── code/               # 可复现，带 README、requirements、固定种子
```

技术报告不超过 12 页（不含参考文献），四段：Literature review / Methods / Results & discussion / Future work。代码也要交，可能被随机抽查复现性，所以种子固定、版本固定、有一键脚本 `run_inference.py`。

---

## 5. 分工（认领版见 DIVISION_OF_LABOR.md）

约束是：代码和模型训练只有 Mengzhe Yang 一个人做（单卡单 coder）。所以把"想、读、写、分析、画图"和"敲代码、跑模型"分开，另外 5 个人各自认领一个能在评分里看到、能在 Q&A 答上来的模块，并行推进。5 个角色都不署名，组员自己选，7/22 截止。

协作上解耦：Mengzhe 跑 GPU，产出 results/（CSV、mask、log）丢共享文件夹，评估、画图、写报告、写综述、做 PPT 的人各取所需并行，代码和写作互不阻塞。单点风险对冲：文献综述负责人（AI 背景）和 Mengzhe 结对，能跑通推理脚本，技术 Q&A 能帮腔，也是单点故障的备份。

15 分钟汇报每人讲自己产出的部分：①临床动机 2 分钟 → ②文献综述 2.5 分钟 → ③数据分析 2 分钟 → ④方法 + pipeline + Task1/2/3 + Bonus（Mengzhe）5 分钟 → ⑤结果评估 + demo 2.5 分钟 → ⑥结论 + 未来 1 分钟，一共 15 分钟；QA 5 分钟全员上台。

---

## 6. 时间表（10 天倒排，硬截止 7/30 19:00）

| 日期 | 阶段 | 任务 |
|---|---|---|
| 7/21–22 (D1-2) | Setup & Baseline | 环境定稿；数据管线 + 预处理（DullRazor/Shades of Gray）+ EDA；SegFormer Task1 baseline；文献综述起稿 |
| 7/23–24 (D3-4) | Improve Seg | SegFormer 调优（Dice+Tversky+TTA）；PEFT-SAM 升级候选；SAM 精修；Mamba 消融 |
| 7/25–26 (D5-6) | Attribute | Task2 6 类 SegFormer + ROI 均值 + 教程阈值；稀疏类分类头；Task3 规则报告 + 一致性校验；CLIP 索引 |
| 7/27 (D7) | Report + Bonus | 交文献综述；MVP 全跑通；DINOv2 对比；模板化 RAG；audit/sanity 检查 |
| 7/28 (D8) | Integration | 全流水线集成；一键推理脚本在 val 上端到端验证 + 计时；error analysis；冻结模型 |
| 7/29 (D9) | Finalize | 生成全部交付物样例；PPT 初稿；写报告；Stretch 收尾 |
| 7/30 (D10) | Submit | 13:00 测试集出 → 一键推理 → 19:00 提交；当晚练 pre |
| 7/31 | Present | 15 分钟 pre + 5 分钟 QA |

6 小时窗口铁律：7/29 必须用 val 集完整模拟一次"拿图 → 出全部交付物"并计时，7/30 当天只跑推理、不写代码。

---

## 7. 风险与应对

| 风险 | 概率 | 应对 |
|---|---|---|
| 7/30 推理窗口出 bug | 高 | 7/29 val 端到端验证 + 计时；预留 fallback 模型 |
| PEFT-SAM 不收敛 | 中 | SegFormer 保底；PEFT-SAM 只当升级候选，val 不赢就不用 |
| 稀疏属性召回低 | 中 | 分类头兜底；Focal + 加权 Dice |
| 显存 OOM | 中 | AMP + 梯度累积 + 降分辨率 + 降 batch |
| 接口不对齐返工 | 中 | 每天 sync 对齐 JSON/mask 格式 |
| scope 膨胀 | 中 | MVP/Stretch 铁律，MVP 没通不碰 Stretch |
| 一致性丢分 | 低 | 规则模板 + 教程阈值 + 自动校验 |
| 组员技术不均 | 中 | 技术活集中 1–2 人，非技术成员做文献/报告/PPT/可视化 |

---

## 8. 拿第一的论证

竞争逻辑前面说过：模型选型是入场券，赢面在执行可靠 + 全 task 不拉胯 + 分析深 + 汇报强。

Best Segmentation（专项奖，中高）：SegFormer 保底 IoU 0.87–0.90，PEFT-SAM 升级可能摸到 0.89–0.91，SAM 精修降 Hausdorff；Dice + IoU + Hausdorff 三个指标都报，加上多模型消融和 error analysis，工程完整度就出来了。

Best Overall（综合奖，中高，主攻方向）：三个 task 权重均衡没短板，Task3 一致性拿满，Bonus 的 audit/sanity 全过；我们团队结构就是为"全 task 不拉胯 + 汇报强"设计的——多数技术强的组汇报糙、报告薄，这是我们的结构性优势。

老实说不保证第一：名次取决于对手实力和测试集分布，这两个不可控。但方案在可控范围内把期望排名最大化了，可靠性高（SegFormer 保底 + 多 fallback）、技术深度足、汇报故事强。最大的不确定性是测试集分布漂移和对手算力，靠 val 贴近、TTA、SAM 精修、汇报质量去对冲。

---

## 9. 方案评估

### 9.1 符合 briefing + tutorial（逐条）

| 要求 | 来源 | 方案对应 | 符合 |
|---|---|---|---|
| Task1 分割 + Dice/IoU/Hausdorff | briefing p5, tutorial p46 | §3.1，三指标全报 | 是 |
| Task2 五属性，逐属性报指标，稀疏类处理 | briefing p6, tutorial p53 | §3.2 | 是 |
| Task2 presence = lesion ROI 上的 mean logits | tutorial p40 | §3.2，已改正（不是覆盖率） | 是 |
| Task3 JSON + 5 术语 + status 一致 + 证据对齐 | briefing p7, tutorial p40 | §3.3 | 是 |
| Task3 教程阈值（status/border/size） | tutorial p40 | §3.3，硬编码采用 | 是 |
| 预处理 DullRazor + Shades of Gray | tutorial p35 | §2，已补 | 是 |
| Bonus CLIP 检索 + 存 neighbor ID + audit/sanity | briefing p8, tutorial p41/46 | §3.4 | 是 |
| 交付物 masks/attr/report/bonus | briefing p9 | §4，对齐 example | 是 |
| 评分权重 seg25/attr25/report20/bonus | briefing p9 | 三 task 无短板 | 是 |
| 报告 ≤12 页四段式 | briefing p11 | §4 | 是 |
| 代码提交 + 复现性抽查 | briefing p9 | §4，固定种子 + 一键脚本 | 是 |
| 关键日期 | briefing p12 | §6 | 是 |
| 每组一套交付物 | briefing p9 | §5 | 是 |
| 确定性 train/val split | tutorial p56 | §2，固定种子 | 是 |

待确认三件事：Task1 mask 存成 jpg 是有损的，按样例执行，但要问助教能不能用 png；Task3 JSON 字段有歧义（scoring 提 recommendation 等，example 没有），按 example 执行并问助教；Bonus 权重百分比 PDF 没写，按高分项处理。

### 9.2 按时完成

可行，但 GPU 是硬瓶颈。MVP 大概 10 GPU 小时，Stretch 15–20 小时；10 天 × 每天 8–10 有效 GPU 小时 = 80–100 小时窗口，算力够。最大风险还是 7/30 那 6 小时窗口，已经用 7/29 的 val 演练对冲。按时交付概率中高（约 75%），前提是 MVP 7/27 前跑通、GPU 时间有序调度。

### 9.3 拿第一

| 奖项 | 可达性 | 依据 |
|---|---|---|
| Best Segmentation | 中高 | SegFormer 保底 + PEFT-SAM 升级 + SAM 精修 + 三指标 + 消融 |
| Best Overall | 中高（主攻） | 全 task 无短板 + 一致性拿满 + bonus audit/sanity + 强汇报（团队结构优势） |

劣势老实说：单 8 GB 笔记本 GPU（对手可能有工作站）；单 coder 单点；测试集分布未知。杠杆是：Hausdorff + error analysis 的深度、Bonus audit/sanity 全过 + CLIP/DINOv2 对比、报告里消融写全、汇报一句话讲清统一架构。

结论：方案可行，没有致命短板，每种失败模式都有 fallback。Best Overall 是更现实的冲击点；不保证第一，但在可控范围内把期望排名最大化了。
