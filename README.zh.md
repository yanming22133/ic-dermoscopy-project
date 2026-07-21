# IC 夏校 — 皮肤镜深度学习项目

[English](README.md) | **中文**

病灶分割 + 5 属性检测 + 锚定报告 + CLIP 检索 bonus。
第二组 · Imperial College DSI 夏校 · 2026-07-21 → 2026-07-31。

## 仓库结构

```
.
├── src/                      # 代码
├── scripts/
│   ├── download_weights.py   # 下 SegFormer 权重
│   ├── download_data.py      # 数据准备说明
│   └── make_division_xlsx.py # 重新生成分工 Excel
├── PROJECT_PLAN.md           # 方案（中文）
├── PROJECT_PLAN_EN.md        # 方案（英文）
├── IMPROVEMENT_LIST.md       # 改进清单（Stretch）
├── DIVISION_OF_LABOR.md      # 分工（文本）
├── DIVISION_OF_LABOR.xlsx    # 分工（中英文 Excel）
├── requirements.txt
└── README.md
```

数据 / 权重 / 输出**都不进 git**（见 `.gitignore`）：10GB 训练数据和 134MB 权重由 `scripts/` 重新下载，`outputs/` 是生成物。

## 环境（重要）

只用这个 Python，别用 PATH 里的 `python`（那是 CPU 版）：

```
PY = f:\anacondaenvs\pytorch\python.exe   # torch 2.11.0+cu128, CUDA, RTX 4070 8GB
```

装依赖：`f:\AnacondaEnvs\pytorch\Scripts\pip install -r requirements.txt`
下权重：`%PY% scripts/download_weights.py`

## 数据划分

2700 张图确定性 80/10/10 切 train / val / test-local（固定种子 42）：
- **train** — 训练
- **val** — 选模型、早停（反复用，分数偏乐观）
- **test-local** — 最后跑一次，看无偏泛化

官方测试集 7/30 13:00 发布，无标签，**只跑推理，绝不训练**。

## Task 1 — 病灶分割

训练（主力 B2）：
```
%PY% -m src.train_task1 --variant b2 --epochs 50 --batch 8
```
显存不够：`--batch` 降到 4，或 `--variant b0`，或 `--size 384`。

冒烟测试（1 分钟内验证流程）：
```
%PY% -m src.train_task1 --variant b0 --epochs 1 --batch 2 --max_train 20 --max_val 20
```

每轮在 val 上报 Dice/IoU/HD95，保存 `outputs/task1_segformer/best.pth`。

推理 + 评估：
```
# val 上评估 + 存 mask
%PY% -m src.infer_task1 --ckpt outputs/task1_segformer/best.pth --split val --save_dir outputs/task1_val_masks
# 官方测试集（7/30）：只存 mask
%PY% -m src.infer_task1 --ckpt outputs/task1_segformer/best.pth --image_dir <测试集目录> --save_dir submit/task1
```

## 输出格式（对齐 example_result）

- Task1 mask：`{id}.jpg`，L 模式，0/255（jpg 有损，读取时 >127 还原二值）
- Task2：`{id}/{attr}.png`，L，0/255；attr 用单数 `milia_like_cyst`
- Task3 JSON：`{id}.json`；`attributes_order` 用复数 `milia_like_cysts`
- Bonus CSV：`query_image, neighbor_id, similarity`

命名陷阱：mask 文件名用 `milia_like_cyst`（单数），JSON 用 `milia_like_cysts`（复数）。

## 待写（还没写）

- `src/train_task2.py`（6 类分割）、`src/infer_task2.py`
- `src/report_task3.py`（规则报告 + 一致性校验）
- `src/bonus_clip.py`（CLIP/DINOv2 检索 + RAG）
- `run_inference.py`（7/30 一键端到端）
