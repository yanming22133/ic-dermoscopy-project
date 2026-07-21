# IC Summer School — Dermoscopy Deep Learning Project

**English** | [中文](README.zh.md)

Lesion segmentation + 5-attribute detection + anchored findings report + CLIP retrieval bonus.
Group 2 · Imperial College DSI Summer School · 2026-07-21 → 2026-07-31.

## Repo layout

```
.
├── src/                      # code
├── scripts/
│   ├── download_weights.py   # fetch SegFormer weights
│   ├── download_data.py      # data prep note
│   └── make_division_xlsx.py # regenerate the division-of-labor Excel
├── PROJECT_PLAN.md           # plan (Chinese)
├── PROJECT_PLAN_EN.md        # plan (English)
├── IMPROVEMENT_LIST.md       # improvement roadmap (Stretch tier)
├── DIVISION_OF_LABOR.md      # division of labor (text)
├── DIVISION_OF_LABOR.xlsx    # division of labor (bilingual Excel)
├── requirements.txt
└── README.md
```

Data / weights / outputs are **gitignored** (see `.gitignore`): the 10 GB training data and 134 MB weights are re-fetched via `scripts/`; `outputs/` is generated.

## Environment (important)

Only use this Python — not the `python` on PATH (that one is CPU-only):

```
PY = f:\anacondaenvs\pytorch\python.exe   # torch 2.11.0+cu128, CUDA, RTX 4070 8GB
```

Install deps: `f:\AnacondaEnvs\pytorch\Scripts\pip install -r requirements.txt`
Fetch weights: `%PY% scripts/download_weights.py`

## Data split

2700 images split deterministically 80/10/10 into train / val / test-local (fixed seed 42):
- **train** — training
- **val** — model selection & early stopping (used repeatedly, so optimistic)
- **test-local** — run once at the end for an unbiased generalization estimate

The official test set is released 7/30 13:00, has no labels, and is **inference-only — never trained on**.

## Task 1 — Lesion segmentation

Train (main, B2):
```
%PY% -m src.train_task1 --variant b2 --epochs 50 --batch 8
```
If OOM: drop `--batch` to 4, or `--variant b0`, or `--size 384`.

Smoke test (verify the pipeline in <1 min):
```
%PY% -m src.train_task1 --variant b0 --epochs 1 --batch 2 --max_train 20 --max_val 20
```

Each epoch reports val Dice/IoU/HD95 and saves `outputs/task1_segformer/best.pth`.

Inference + eval:
```
# eval on val + save masks
%PY% -m src.infer_task1 --ckpt outputs/task1_segformer/best.pth --split val --save_dir outputs/task1_val_masks
# official test set (7/30): masks only
%PY% -m src.infer_task1 --ckpt outputs/task1_segformer/best.pth --image_dir <test dir> --save_dir submit/task1
```

## Output format (aligned with example_result)

- Task1 mask: `{id}.jpg`, L mode, 0/255 (jpg is lossy; threshold >127 to recover binary on read)
- Task2: `{id}/{attr}.png`, L, 0/255; attr uses singular `milia_like_cyst`
- Task3 JSON: `{id}.json`; `attributes_order` uses plural `milia_like_cysts`
- Bonus CSV: `query_image, neighbor_id, similarity`

Naming trap: mask filenames use `milia_like_cyst` (singular), JSON uses `milia_like_cysts` (plural).

## TODO (not yet written)

- `src/train_task2.py` (6-class seg), `src/infer_task2.py`
- `src/report_task3.py` (rule template + consistency check)
- `src/bonus_clip.py` (CLIP/DINOv2 retrieval + RAG)
- `run_inference.py` (one-click end-to-end, for 7/30)
