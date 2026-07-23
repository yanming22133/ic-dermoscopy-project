# 项目状态文档（2026-07-24）
## Project Status — pick up here for the next conversation

---

## 一、项目目标

IC DSI Summer School (Group 2)。三个 Task + Bonus：
- **Task1** 病灶分割（25%）：PEFT-SAM ViT-B，目标 Dice 0.94+
- **Task2** 属性检测（25%）：SegFormer-B2 5通道多标签 + Focal + 边缘损失
- **Task3** 锚定报告（20%）：规则模板 + 一致性校验
- **Bonus** CLIP检索+RAG（~30%）：代码已就绪

**7/27** 交文献综述（Task1 皮肤分割） | **7/30** 13:00 测试集发布 → 19:00 交全部 | **7/31** 15分钟汇报+5分钟QA

---

## 二、当前进度

### Task1
| 模型 | 训练 val Dice | 推理 Dice | HD95 | 位置 |
|---|---|---|---|---|
| **B2 baseline** (SegFormer, 笔记本) | 0.9039 | 0.9039 | 24.75 | `outputs/task1_b2/best.pth` |
| **B3** (SegFormer, 笔记本) | 0.9019 | 0.9025 | 132.55 | 放弃，不如 B2 |
| **PEFT-SAM ViT-B** (4090) | **0.9453** (ep 45) | ⚠️ **待跑** | 32.36(训练eval) | AutoDL: `/root/autodl-tmp/ic-dermoscopy-project/outputs/task1_sam/best.pth` |

**Task1 当前状态**：PEFT-SAM 训练完成（50 epoch, best Dice 0.9453 @ epoch 45），best.pth 已保存。推理测了两轮都因为 bug 失败（full-image prompt → Dice 0.357；bbox 坐标未缩放 → Dice 0.000）。**模型本身没问题——训练 eval 时用 GT bbox 拿的 0.945，是推理 pipeline 的 SAM box prompt 两轮机制有 bug。**

**推理 bug 已修**（bbox 256→1024 缩放 + numpy import），AutoDL 上正重新跑推理。

### Task2
- 代码就绪：Focal Loss + 平衡采样 + 边缘损失(edge_loss 0.5) + 属性关系图后处理
- **待训练**：等 Task1 推理跑完或并行开另一个终端
- 训练命令：`python -u -m src.train_task2 --epochs 50 --batch 16 --size 512 --num_workers 8 --cosine_lr --balanced 1 --edge_loss 0.5 --patience 10 --out outputs/task2_b2`

### Task3
- 代码就绪：`src/report_task3.py`，规则模板 + 教程阈值硬编码 + 一致性 checklist
- CPU 冒烟测试通过
- **不需要训练**，拿到 Task1 mask + Task2 presence 后直接跑

### Bonus
- 代码就绪：`src/bonus_clip.py`（CLIP/DINOv2 双检索 + RAG + audit）
- **待建索引 + 检索**

---

## 三、代码版本

| 版本 | 入口 | 状态 |
|---|---|---|
| v1 SegFormer B2/B3 | `src/train_task1.py --model segformer --variant b2` | ✅ 完成 |
| v2 PEFT-SAM | `src/train_task1.py --model peft_sam` | ✅ 训练完成，推理在测 |
| v3 新模块 | `src/improvements/` (5个模块) | ⚠️ 代码已写，未接线 |
| v3 扩散/潜空间 | `src/diffusion_loss.py`, `src/diffusion_refine.py` | ⚠️ SD 权重未下 |
| Task2 | `src/train_task2.py` | ⚠️ 待训练 |
| Task3 | `src/report_task3.py` | ✅ 就绪 |
| Bonus | `src/bonus_clip.py` | ⚠️ 待建索引 |
| 一键推理 | `run_inference.py` | ⚠️ 待端到端测试 |

**GitHub**: `https://github.com/yanming22133/ic-dermoscopy-project`，最新 commit `f1e3092`

---

## 四、关键文件位置

| 用途 | 路径 |
|---|---|
| 方案（中文） | `PROJECT_PLAN.md` |
| 方案（英文） | `PROJECT_PLAN_EN.md` |
| 改进清单 | `IMPROVEMENT_LIST.md` |
| 分工表 | `DIVISION_OF_LABOR.md` + `DIVISION_OF_LABOR.xlsx` |
| 文献综述 PPT 要求 | `papers/lit_review_to7.27/` |
| Task1 文献综述论文 | `papers/lit_review_final/` (11篇核心) |
| 全部论文 | `papers/lit_review_full/` (44篇) |
| Task1 SOTA 论文 | `papers_sota/` |
| Task2 SOTA 论文 | `papers_sota/task2/` |
| 训练日志 Excel | `outputs/task1_sam_training_log.xlsx` |
| AutoDL 配置指南 | `AUTODL_SETUP.md` |
| 项目要求 Word | `Task_Requirements_Scoring.docx` |

---

## 五、AutoDL 连接

- SSH: `ssh -p 29034 root@connect.nmb1.seetacloud.com`（密码见 AutoDL 控制台）
- 项目路径: `/root/autodl-tmp/ic-dermoscopy-project/`
- 数据路径: `/root/autodl-tmp/ic-dermoscopy-project/summer_school_project_train/train/`
- 权重路径: `/root/autodl-tmp/ic-dermoscopy-project/pretrained/`
- 输出路径: `/root/autodl-tmp/ic-dermoscopy-project/outputs/`

**config.py 重要**: GitHub 上的 `config.py` 是 Windows 路径（`F:\Desktop\IC\project`），AutoDL 上被 sed 改成了 `/root/autodl-tmp/ic-dermoscopy-project`。上传任何 config.py 后都需要重新 sed。

---

## 六、当前正在跑

**AutoDL 终端1：Task1 SAM v3 训练**
```bash
python -u -m src.train_task1 --model peft_sam --epochs 50 --batch 4 --accum_steps 2 --size 1024 --num_workers 8 --boundary_loss --cosine_lr --freq_loss --edge_loss 0.3 --hd_score --patience 10 --out outputs/task1_sam_v3
```
含：freq_loss + Boundary + Tversky + edge_loss(Sobel) + HD95每轮 + 综合评分选最佳

**AutoDL 终端2：Task2 训练（待启动）**
```bash
cd /root/autodl-tmp/ic-dermoscopy-project
python -u -m src.train_task2 --epochs 50 --batch 16 --size 512 --num_workers 8 --cosine_lr --balanced 1 --edge_loss 0.5 --patience 10 --out outputs/task2_b2
```

---

## 六-B：上传文件清单

**Task1（v3 SAM 已在跑，以下已上传）：**
- `src/train_task1.py`
- `src/model_sam.py`（含 numpy import + 两轮 bbox 缩放修复）
- `src/improvements/`（整个文件夹：5个模块）
- `src/infer_task1.py`（含 boundary_smooth 接线）

**Task2（需上传以下到 AutoDL）：**
- `src/train_task2.py`（含 tqdm + edge_loss）
- `src/infer_task2.py`（含 attr_rules + ms_tta）
- `src/task2_attr_graph.py`

Task2 的 torch 版本问题：AutoDL 上 torch 2.5.1 不够，需先升级：
```bash
pip install torch==2.6.0 torchvision --index-url https://download.pytorch.org/whl/cu124
```

---

## 六-C：Task1 v3 跑完后

推理命令（SAM + 全部推理改进）：
```bash
python -u -m src.infer_task1 --ckpt outputs/task1_sam_v3/best.pth --split val \
  --tta 1 --postproc 1 --boundary_smooth 1
```

## 六-D：综合评分公式

`score = Dice × 0.7 + IoU × 0.4 − HD95 / 500`
满分约 1.1（Dice=1,IoU=1,HD95=0）。当前 v3 训练中按此公式选最佳 checkpoint。

---

## 七、已知问题和解决方案

| 问题 | 状态 |
|---|---|
| SAM 推理 full-image prompt → Dice 崩溃 | ✅ 已修（两轮 bbox 缩放） |
| SAM 推理 bbox 坐标未缩放 → Dice=0 | ✅ 已修 |
| `--ch_attn` 2通道崩溃 | ✅ 已修（attention.py max(1,ch//r)） |
| HD95 被 postproc 污染 | ✅ 已修（raw mask 算 HD95） |
| eval OOM (fp32) | ✅ 已修（eval 内 autocast） |
| B3 HD95=132 | ❌ B3 放弃，B2 为主 |
| SD 权重下载 | ⚠️ 先跳过 |

---

## 八、新对话快速恢复命令

新对话开始时，告诉我：
1. "读一下 STATUS.md 恢复上下文"
2. 当前 AutoDL 上 Task1 推理和 Task2 训练的状态
3. 继续下一步

**笔记本环境**: 只用 `f:\anacondaenvs\pytorch\python.exe`，别用 PATH 里的 `python`（那个是 CPU 版）。
