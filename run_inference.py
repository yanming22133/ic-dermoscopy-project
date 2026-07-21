"""一键端到端推理（7/30 测试集发布后用）：Task1 -> Task2 -> Task3 -> Bonus。
One-click end-to-end inference (for the 7/30 test set): Task1 -> Task2 -> Task3 -> Bonus.

用法 / Usage:
  f:\\anacondaenvs\\pytorch\\python.exe run_inference.py \\
      --test_dir <测试集图目录> \\
      --task1_ckpt outputs/task1_segformer_b2/best.pth \\
      --task2_ckpt outputs/task2_segformer/best.pth \\
      --submit_dir submit

输出 / Output:
  submit/task1/{id}.jpg
  submit/task2/{id}/{attr}.png + presence.json
  submit/task3/json/{id}.json + Summary_reports_text.csv
  submit/bonus/test_bonus_clip.csv
"""
import os
import argparse
from types import SimpleNamespace

from src import infer_task1, infer_task2, report_task3, bonus_clip
from src.config import OUT_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--test_dir', required=True, help='官方测试集图目录 / official test image dir')
    ap.add_argument('--task1_ckpt', required=True)
    ap.add_argument('--task2_ckpt', required=True)
    ap.add_argument('--submit_dir', default='submit')
    ap.add_argument('--encoder', default='clip', choices=['clip', 'dinov2'])
    ap.add_argument('--k', type=int, default=3)
    ap.add_argument('--model_version', default='task1_best.pth, task2_best.pth')
    ap.add_argument('--do_preprocess', type=int, default=1)
    ap.add_argument('--tta', type=int, default=1, help='Tier1: 翻转 TTA 1/0 / flip TTA')
    ap.add_argument('--sam_refine', type=int, default=0, help='Tier1: SAM 边界精修 1/0 / SAM refine')
    ap.add_argument('--skip_bonus', action='store_true')
    args = ap.parse_args()

    sub = args.submit_dir
    t1_dir = os.path.join(sub, 'task1')
    t2_dir = os.path.join(sub, 'task2')
    t3_dir = os.path.join(sub, 'task3')
    bon_dir = os.path.join(sub, 'bonus')

    print('===== [1/4] Task1 病灶分割 / lesion segmentation =====', flush=True)
    infer_task1.infer(SimpleNamespace(
        ckpt=args.task1_ckpt, split='test', image_dir=args.test_dir,
        save_dir=t1_dir, do_preprocess=args.do_preprocess,
        tta=args.tta, sam_refine=args.sam_refine))

    print('===== [2/4] Task2 属性检测 / attribute detection =====', flush=True)
    infer_task2.infer(SimpleNamespace(
        ckpt=args.task2_ckpt, task1_mask_dir=t1_dir, split='test',
        image_dir=args.test_dir, save_dir=t2_dir, do_preprocess=args.do_preprocess,
        tta=args.tta))

    print('===== [3/4] Task3 锚定报告 / anchored report =====', flush=True)
    report_task3.run(SimpleNamespace(
        task1_mask_dir=t1_dir, presence=os.path.join(t2_dir, 'presence.json'),
        save_dir=t3_dir, split='test', model_version=args.model_version,
        ids=None, attr_mask_dir=t2_dir))

    if not args.skip_bonus:
        print('===== [4/4] Bonus CLIP 检索 / CLIP retrieval =====', flush=True)
        cache = os.path.join(OUT_DIR, f'bonus_{args.encoder}_embeds.npz')
        if not os.path.exists(cache):
            print('索引不存在，先建索引 / index missing, building...', flush=True)
            bonus_clip.cmd_build(SimpleNamespace(
                encoder=args.encoder, do_preprocess=args.do_preprocess,
                batch=32, rebuild=False))
        bonus_clip.cmd_retrieve(SimpleNamespace(
            encoder=args.encoder, image_dir=args.test_dir, task1_mask_dir=t1_dir,
            save_dir=bon_dir, k=args.k, do_preprocess=args.do_preprocess, batch=32))

    print('===== 完成 / done =====', flush=True)
    print(f'交付物在 / deliverables in: {sub}/', flush=True)


if __name__ == '__main__':
    main()
