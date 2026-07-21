"""Task3 锚定报告：从 Task1 病灶 mask + Task2 presence 生成 JSON + 文本报告，并做一致性校验。
Task3 anchored report: generate JSON + text report from Task1 lesion mask + Task2 presence,
and run a consistency check.

阈值全用 tutorial p40（硬编码）/ thresholds from tutorial p40 (hardcoded):
- size: lesion_area_ratio <0.08 small / 0.08-0.25 moderate / >0.25 large
- border: border_irregularity = perimeter²/(4π·area), >=1.60 irregular else regular
- status: 来自 Task2 presence（>=0.60 present / <=0.40 absent / 中间 uncertain）

输出 / Output:
- task3/json/{id}.json   (照 example schema，attributes_order 用复数 milia_like_cysts)
- Summary_reports_text.csv (image_id, findings)，findings 开头有空格（照 example）
- consistency_report.json (校验统计 / check stats)

用法 / Usage:
  f:\\anacondaenvs\\pytorch\\python.exe -m src.report_task3 \
      --task1_mask_dir outputs/task1_val_masks --presence outputs/task2_val/presence.json \
      --save_dir outputs/task3_val --split val --model_version "task1_best.pth, task2_best.pth"
"""
import os
import csv
import json
import argparse
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion

from .config import ATTRS_JSON, BORDER_IRREG, SIZE_SMALL, SIZE_LARGE
from .data import list_image_ids


def load_lesion_mask(iid, mask_dir):
    for ext in ('.jpg', '.png'):
        p = os.path.join(mask_dir, iid + ext)
        if os.path.exists(p):
            m = np.array(Image.open(p).convert('L'))
            return (m > 127).astype(np.uint8)
    return None


def lesion_features(mask):
    """从病灶 mask 算 size/border 特征（tutorial p40 公式）。
    Compute size/border features from lesion mask (tutorial p40 formulas)."""
    area = int(mask.sum())
    total = mask.size
    if area == 0:
        return 0.0, 'regular', 'small', 0.0
    eroded = binary_erosion(mask)
    perimeter = int((mask & ~eroded).sum())
    border_irreg = perimeter ** 2 / (4 * np.pi * area)
    border_cat = 'irregular' if border_irreg >= BORDER_IRREG else 'regular'
    area_ratio = area / total
    size_cat = 'small' if area_ratio < SIZE_SMALL else ('large' if area_ratio > SIZE_LARGE else 'moderate')
    return border_irreg, border_cat, size_cat, area_ratio


# 文本里每个属性的措辞（照 example：Pigment network is / Streaks are ...）/ wording per example
TEXT_FRAGMENTS = [
    ('Pigment network', 'is'),
    ('Negative network', 'is'),
    ('Streaks', 'are'),
    ('Milia-like cysts', 'are'),
    ('Globules', 'are'),
]


def build_text(size_cat, border_cat, presence):
    """照 example 拼报告文本，findings 开头留一个空格。
    Build report text per example; findings starts with one space."""
    s = f" The lesion is {size_cat} with {border_cat} borders."
    parts = []
    for (term, verb), attr in zip(TEXT_FRAGMENTS, ATTRS_JSON):
        status = presence[attr]['status']
        parts.append(f"{term} {verb} {status}")
    s += " " + "; ".join(parts) + "."
    return s


def build_json(iid, split, model_version, presence):
    return {
        "image_id": iid,
        "split": split,
        "model_version": model_version,
        "attributes_order": list(ATTRS_JSON),
        "outputs": {"presence": {a: presence[a] for a in ATTRS_JSON}},
    }


def consistency_check(iid, presence, mask, attr_mask_dir):
    """一致性校验（tutorial p40/p46）。返回问题列表。
    Consistency check. Returns list of issues."""
    issues = []
    # 1) 5 术语齐全 / 5 terms present
    for a in ATTRS_JSON:
        if a not in presence or 'status' not in presence[a] or 'prob' not in presence[a]:
            issues.append(f'{iid}: missing term/status/prob for {a}')
    # 2) status 与 prob 阈值一致 / status matches prob thresholds
    for a in ATTRS_JSON:
        if a in presence:
            p = presence[a]['prob']; st = presence[a]['status']
            exp = 'present' if p >= 0.60 else ('absent' if p <= 0.40 else 'uncertain')
            if st != exp:
                issues.append(f'{iid}: status {st} != expected {exp} for {a} (p={p})')
    # 3) 证据对齐（若有属性 mask）/ evidence alignment (if attr masks available)
    if attr_mask_dir is not None and mask is not None and mask.any():
        from .config import ATTRS_FILE
        roi = mask.astype(bool)
        for c, attr_file in enumerate(ATTRS_FILE):
            p = os.path.join(attr_mask_dir, iid, attr_file + '.png')
            if not os.path.exists(p):
                continue
            am = (np.array(Image.open(p).convert('L')) > 127).astype(bool)
            cov = float(am[roi].mean()) if roi.any() else 0.0
            st = presence[ATTRS_JSON[c]]['status']
            if st == 'present' and cov < 0.01:
                issues.append(f'{iid}: {attr_file} marked present but ROI coverage {cov:.3f}')
            elif st == 'absent' and cov > 0.05:
                issues.append(f'{iid}: {attr_file} marked absent but ROI coverage {cov:.3f}')
    return issues


def run(args):
    """主逻辑，供 main() 和 run_inference 调用 / main logic, called by main() and run_inference."""
    presence = json.load(open(args.presence))
    if args.ids:
        ids = [l.strip() for l in open(args.ids) if l.strip()]
    else:
        ids = list(presence.keys())

    json_dir = os.path.join(args.save_dir, 'json')
    os.makedirs(json_dir, exist_ok=True)
    csv_path = os.path.join(args.save_dir, 'Summary_reports_text.csv')
    all_issues = []

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['image_id', 'findings'])
        for iid in ids:
            pres = presence.get(iid)
            if pres is None:
                all_issues.append(f'{iid}: no presence entry')
                continue
            mask = load_lesion_mask(iid, args.task1_mask_dir)
            if mask is None:
                all_issues.append(f'{iid}: no Task1 mask')
                mask = np.zeros((480, 640), np.uint8)
            _, border_cat, size_cat, _ = lesion_features(mask)
            obj = build_json(iid, args.split, args.model_version, pres)
            json.dump(obj, open(os.path.join(json_dir, iid + '.json'), 'w'), indent=2)
            text = build_text(size_cat, border_cat, pres)
            w.writerow([iid, text])
            all_issues.extend(consistency_check(iid, pres, mask, args.attr_mask_dir))

    json.dump({'n_images': len(ids), 'n_issues': len(all_issues), 'issues': all_issues[:200]},
              open(os.path.join(args.save_dir, 'consistency_report.json'), 'w'), indent=2, ensure_ascii=False)
    print(f'{len(ids)} imgs -> {args.save_dir} | issues: {len(all_issues)}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task1_mask_dir', required=True)
    ap.add_argument('--presence', required=True, help='Task2 的 presence.json / Task2 presence.json')
    ap.add_argument('--save_dir', required=True)
    ap.add_argument('--split', default='test')
    ap.add_argument('--model_version', default='task1_best.pth, task2_best.pth')
    ap.add_argument('--ids', default=None, help='限定 id 列表文件，每行一个；默认用 presence 里的 / optional id list file')
    ap.add_argument('--attr_mask_dir', default=None, help='Task2 属性 mask 目录（做证据对齐用）/ Task2 attr mask dir for evidence check')
    run(ap.parse_args())


if __name__ == '__main__':
    main()
