# IC Summer School Project Plan (Group 2)

A 10-day deep learning project (7/21–7/31) on dermoscopy images: lesion segmentation, 5-attribute detection, and an anchored findings report, plus a CLIP retrieval bonus. We're a team of 6, aiming for Best Overall and Best Segmentation.

This doc covers what we're doing, how, who does what, and the timeline — all in one place.

---

## 0. The overall idea

Four main threads:

1. Task1 (segmentation) and Task2 (attributes) share one segmentation backbone. SegFormer-B2 is the safe main model — it guarantees we hit IoU 0.87–0.90. PEFT-SAM is the upgrade candidate: we train both and use whichever scores higher on val. We add Mamba as an ablation and use SAM for boundary refinement to push for Best Seg.
2. Task3 (report) is a fixed rule-based template, no neural network — this locks in the full 20% consistency score. The thresholds are the ones the tutorial gives, not invented by us.
3. Bonus uses CLIP (plus DINOv2 for comparison) for retrieval, with the model frozen (no fine-tuning). The priority is audit completeness: every image has results, every neighbor ID is saved, no duplicates, consistent K. RAG only polishes wording, it does not copy neighbor facts.
4. We split the work into MVP and Stretch. The MVP (SegFormer T1 + 6-class SegFormer T2 + rule T3 + CLIP Bonus) must be passing by 7/27. PEFT-SAM, Mamba, DINOv2, SAM refinement, ensembles are Stretch — we only touch them once the MVP works.

One judgement that runs through everything: model choice is just table stakes (other groups read the same papers). What actually wins is not crashing, having no weak task, going deep on Hausdorff and error analysis, presenting well, and shipping reproducible code.

All models must be trained by 7/29, and the one-click inference script must be validated on val — because the test set on 7/30 gives us only 6 hours for inference, and we can't write code that day.

---

## 1. Environment & compute

The machine is an ASUS ROG Zephyrus G16 with an RTX 4070 Laptop GPU (8 GB VRAM) and about 31.4 GB RAM.

One trap first: the `python` on PATH is a CPU-only build (Python310) and must not be used. The correct env is `f:\anacondaenvs\pytorch\python.exe`, which has torch 2.11.0+cu128, working CUDA, and sees the 4070. Set the VSCode interpreter to this path explicitly. To install packages, use the matching pip: `f:\AnacondaEnvs\pytorch\Scripts\pip`.

Already installed: timm 1.0.22, transformers 4.57.3, opencv 4.13, sklearn, OpenAI clip, pandas. Still to install: albumentations, faiss-cpu, open_clip_torch; segmentation_models_pytorch is optional.

With 8 GB VRAM: use mixed precision (AMP) + gradient accumulation throughout, keep input resolution at 480 or 512, single-model training takes 3–6 GB, and never load multiple models at once.

---

## 2. Data & splits

The training set is 2700 dermoscopy images, 640×480, RGB. Task1 labels are `task1_gt/*_segmentation.png`, 2700 single-channel binary masks (values 0/255). Task2 labels are `task2_gt/*_attribute_{attr}.png`, 5 attributes × 2700 = 13500 masks (also 0/255). The 5 attributes: pigment_network, negative_network, streaks, milia_like_cyst, globules. The test set is released 7/30 13:00, same format as train but with no labels.

**Preprocessing** (required by tutorial p35, dermoscopy-specific): first DullRazor to remove hair artifacts, then Shades of Gray for color constancy (to unify color temperature across devices), then normalization (zero mean / unit variance, or ImageNet stats), then resize to 480/512. Before training, always visualize image–mask alignment to catch any misalignment (tutorial p55).

**Augmentation** (albumentations, tutorial p35): rotation, flip, color jitter, scale, crop, elastic. Task1 and Task2 must be augmented in sync so the masks still line up.

**Data split & generalization** (important):

We split the 2700 images 80/10/10 into train / val / test-local, deterministically with a fixed seed (tutorial p56 requires a deterministic split). Each set has its own job:

- **train (80%, ~2160 images)**: for training.
- **val (10%)**: for model selection — early stopping, hyperparameters, choosing SegFormer vs PEFT-SAM, tuning Task3 thresholds. You look at it repeatedly, so val scores end up optimistic.
- **test-local (10%)**: for an unbiased estimate. Run it only once, on 7/28–29, to see real generalization. If test-local is much lower than val, we're overfitting and need to fix it before 7/30. This set is never used for tuning.

About "generalizing to the official test set" — that test set is held out from the same dataset, same distribution, not images from another hospital. So as long as we don't badly overfit, generalization is mostly fine. The way to maximize it: strong augmentation + pretrained/foundation backbones (SegFormer is ImageNet-pretrained, SAM trained on huge data — transfer learning resists domain shift naturally) + Shades of Gray / DullRazor to remove domain differences + early stopping on val to avoid overfitting + TTA at test time. **The official test set has no labels and is never used for training**; on 7/30 we only run inference on it.

Should we retrain the final model on train+val? It's optional and not recommended. The upside is using 10% more data for maybe 0.3–0.5 points; the downside is losing val for early stopping, added complexity, and the 6-hour window can't absorb surprises. Recommended: just use the val-selected checkpoint as the final model. If by 7/28 everything is smooth and there's spare time and VRAM, we can train a train+val candidate and compare on test-local, picking the better one.

Honest note: we can't do external validation (no second dataset with these 5 attribute labels is readily available), and the official test set is an internal test. This goes in the report's limitation and future work — judges like seeing that kind of honest awareness.

---

## 3. Technical approach

### 3.1 Task 1 · Lesion segmentation (pushing for Best Seg, 25%)

Goal: output a binary lesion mask and push up Dice, IoU, and 95% Hausdorff (all three are required by tutorial p46 — Hausdorff is not optional).

The 2026 literature (deep-research, 23 claims verified) shows ISIC2018 SOTA at Dice 0.91–0.94 and IoU 0.89–0.91, with the field moving to foundation-model PEFT (PEFT-MedSAM hit IoU 0.8918 in 2026.06), Mamba (MambaLiteUNet, CVPR 2026), and diffusion. But model choice is table stakes — reliable execution beats chasing trends.

How we configure the models:

- **Safe main**: SegFormer-B2. Get it running on D1–2, guarantee IoU 0.87–0.90, and it's a final-submission candidate. Batch 16 fits in 8 GB.
- **Upgrade candidate**: PEFT-finetuned SAM (ViT-B), frozen image encoder, train only the mask decoder, optional LoRA. Aligns with PEFT-MedSAM. It only replaces the main model if it beats SegFormer on val. Batch 2–4 in 8 GB.
- **Boundary refinement**: SAM inference refinement + a BiSeg-SAM-style DetailRefine boundary module, specifically to lower Hausdorff.
- **Ablation**: UltraLBM-UNet (Mamba, only 0.034M params), a lightweight 2025-SOTA comparison at near-zero cost.
- **Inference boost**: TTA, flip plus rotate averaging, free 1–2 IoU.

Loss is BCE + Dice, with Tversky or Boundary Loss for boundaries. Evaluation reports Dice, IoU, 95% Hausdorff, and boundary F-score, both per-image and global.

The decision rule is simple: train both SegFormer and PEFT-SAM, compare on val, use the winner. If PEFT-SAM doesn't converge, just use SegFormer — we don't bet the timeline on it.

What we discard: nnU-Net (too heavy for 8 GB + 10 days); diffusion (slow to train, risky on 8 GB, only discussed as future work); SAM2 (research shows it's not generally better than SAM for medical segmentation, not worth switching); SAM3 (refuted by deep-research, not available mid-2026).

### 3.2 Task 2 · Attribute detection (25%)

Goal: output 5 attribute masks plus a presence label (present / absent / uncertain) for each attribute.

The main route is 6-class segmentation: SegFormer (same family as Task1), num_classes=6 (background + 5 attributes), with per-pixel logits for each attribute. How is the presence probability computed? Tutorial p40 gives the definition: take each attribute's logits, apply sigmoid, then **take the mean over the lesion ROI** to get p_attr (the lesion ROI is the pixels inside the Task1 predicted lesion mask — this is the tutorial's recommended approach). Note it's a mean, not a coverage ratio. The status thresholds are the tutorial's: p ≥ 0.60 is present, p ≤ 0.40 is absent, 0.40–0.60 is uncertain.

Sparse-attribute fallback: milia_like_cysts and streaks have very small regions. If recall is too low, add a shared-backbone multi-label classification head (5 sigmoids) to back up presence (this is the dual cls head idea from tutorial p35); but mask evidence still comes from the segmentation output, to keep "evidence alignment" in the report. CLIP zero-shot attribute probing (text-prompt matching) can be an unsupervised ablation highlight.

Loss is weighted Dice + Focal for sparse classes, with the combined loss from the tutorial: Loss = Loss_seg + 0.5·Loss_cls. Evaluation reports per-attribute F1, AUPRC, and mask Dice/IoU, focusing on recall for sparse classes (the tutorial requires per-attribute reporting, not just an average).

### 3.3 Task 3 · Anchored report (20%, scored on consistency)

Goal: output a JSON and a short English report, with all 5 terms present, text statuses exactly matching the JSON, and evidence descriptions aligned with the masks.

The approach is a deterministic rule-based template, using the tutorial's thresholds (p40), not invented ones:

- Attribute status: p_attr ≥ 0.60 present / ≤ 0.40 absent / 0.40–0.60 uncertain.
- border_irregularity = perimeter² / (4π·area); if area is 0, set it to 0.0; ≥ 1.60 is irregular, < 1.60 is regular.
- lesion_area_ratio = lesion pixels / total pixels; < 0.08 small / 0.08–0.25 moderate / > 0.25 large.

Inputs are the Task1 mask (for size and border) and Task2 presence + mask (for evidence descriptions). The JSON output strictly follows `example_result/task3/json/000001.json` (image_id, split, model_version, attributes_order[5], outputs.presence.{attr}.{prob,status}); the text is template-concatenated, all 5 terms present, statuses strictly matching the JSON, in the style of the example: "The lesion is moderate with irregular borders. Pigment network is present; ...".

The consistency checklist is verified automatically (tutorial p40): all 5 terms present, text status equals JSON status, evidence description matches mask coverage, JSON schema compliant with valid numeric ranges and correct terminology (tutorial p46 Format Constraint).

There's a field discrepancy to confirm: tutorial p46 scoring mentions lesion_presence / attributes_detected / findings_summary / recommendation, but the example JSON only has attributes_order / outputs.presence. We follow the example and ask the TA on 7/22 whether a recommendation field is needed.

Why no neural network for the report? Task3 has no training data — it's a deterministic function of the Task1/2 outputs. A neural net could drop terms or change statuses and endanger the 20% consistency score. RAG only enhances wording and is checked afterward; it never copies neighbor facts (tutorial p43).

### 3.4 Bonus · CLIP retrieval + RAG (scored on audit completeness)

Retrieval (tutorial p41–42, model frozen, no fine-tuning): use CLIP ViT-B/32 to encode all 2700 training images into vectors (N×512 float32), build a FAISS cosine index, and for each test image take the Top-K neighbors. An upgrade comparison adds DINOv2 ViT-B/14 (dense visual similarity, often wins in medical retrieval). Before retrieval, crop the lesion with the Task1 mask and encode that — it markedly improves recall and ties Task1 to the Bonus. Output strictly follows `test_bonus_clip.csv`: query_image, neighbor_id, similarity.

The scoring priorities align with tutorial p46, not invented: first, Audit Completeness — every test image has results and all neighbor IDs are saved; second, Sanity Check — no duplicate image IDs, consistent K, valid similarities. These are the hard criteria — full coverage, no duplicates, consistent K matters more than "proving retrieval helps."

RAG wording augmentation (tutorial p43): aggregate the Top-K neighbors' GT reports with a template, but the key is grounded — no copying neighbor facts, facts must match the current image's own masks. The presentation can show a with/without RAG comparison as a highlight, but grading is on audit/sanity.

---

### 3.5 Literature evidence (deep-research 2025–2026, 23 claims verified)

| Module | Literature | Impact on plan |
|---|---|---|
| Task1 candidate | PEFT-MedSAM (2026.06) IoU 0.8918, beats U-Net and zero-shot; SkinSAM/BiSeg-SAM same family | PEFT-SAM as upgrade candidate, SegFormer safe |
| Task1 ablation | MambaLiteUNet (CVPR 2026) Dice 93.09%; UltraLBM-UNet 0.034M params | Add lightweight Mamba ablation |
| Task1 discarded | Diffusion slow; SAM2 not generally better; SAM3 refuted | Explicitly discarded with reasons |
| Task2 attributes | CKTG+GD-DDW (TNNLS 2025.08) EDRA AUC 88.6% | SOTA reference; we use mask supervision for evidence-grounded detection |
| Bonus RAG | MMed-RAG (ICLR 2025), RadAlign (MICCAI 2025): CLIP+FAISS | Validates CLIP+FAISS route |
| Evaluation | Most papers report only Dice/IoU, Hausdorff rarely | Reporting Hausdorff is differentiation (and tutorial-required) |
| Optional FM | DermINO (2025.08) dermatology foundation model, 432K images | Backbone if weights available; DINOv2 fallback |

Honestly, most of these are arXiv preprints; only CKTG (TNNLS 2025) and MambaLiteUNet (CVPR 2026) are confirmed accepted. Benchmark splits differ across papers, so metrics aren't directly comparable. ISIC2018 SOTA Dice clusters around 0.91–0.94.

---

### 3.6 MVP / Stretch (keeping scope in check)

| Tier | Content | Deadline |
|---|---|---|
| MVP (must-hit floor) | SegFormer Task1 + 6-class SegFormer Task2 + rule Task3 + CLIP Bonus | passing by 7/27 |
| Stretch (only after MVP works) | PEFT-SAM, Mamba ablation, DINOv2, SAM refinement, ensemble, TTA | 7/28–29 |
| Iron rule | no Stretch until MVP works; freeze after 7/28, no new Stretch | — |

---

## 4. Deliverables & format (following example_result)

```
submit/
├── task1/              # lesion masks, named like example: 000001.jpg
├── task2/              # one folder per image: 000001/{attribute}.png
├── task3/
│   ├── json/000001.json
│   └── Summary_reports_text.csv   # image_id, findings
├── bonus/
│   └── test_bonus_clip.csv        # query_image, neighbor_id, similarity (all images, no dup, consistent K)
└── code/               # reproducible, with README, requirements, fixed seed
```

The technical report is at most 12 pages (excluding references), four sections: Literature review / Methods / Results & discussion / Future work. Code is also submitted and may be randomly checked for reproducibility, so seeds are fixed, versions pinned, and there's a one-click `run_inference.py`.

---

## 5. Division of labor (signup version in DIVISION_OF_LABOR.md)

The constraint: code and model training are done by Mengzhe Yang alone (single GPU, single coder). So we split "thinking, reading, writing, analyzing, plotting" from "coding, running models" — the other five each claim a module that's visible in the grading and defensible in Q&A, advancing in parallel. The 5 roles are unnamed; members self-select by 7/22.

Collaboration is decoupled: Mengzhe runs the GPU and drops results/ (CSV, masks, logs) into a shared folder; the people doing evaluation, plots, report, lit review, and PPT each take what they need in parallel — code and writing never block each other. Single-point hedge: the literature lead (AI background) pairs with Mengzhe, can run the inference script, can back up technical Q&A, and is the fallback if Mengzhe is unavailable.

The 15-minute presentation has each person present their own output: ① clinical motivation 2 min → ② lit review 2.5 min → ③ data analysis 2 min → ④ methods + pipeline + Task1/2/3 + Bonus (Mengzhe) 5 min → ⑤ results + eval + demo 2.5 min → ⑥ conclusion + future 1 min, total 15 min; Q&A 5 min with everyone on stage.

---

## 6. Timeline (10-day countdown, hard deadline 7/30 19:00)

| Date | Phase | Tasks |
|---|---|---|
| 7/21–22 (D1-2) | Setup & Baseline | finalize env; data pipeline + preprocessing (DullRazor/Shades of Gray) + EDA; SegFormer Task1 baseline; lit review draft |
| 7/23–24 (D3-4) | Improve Seg | SegFormer tuning (Dice+Tversky+TTA); PEFT-SAM upgrade candidate; SAM refinement; Mamba ablation |
| 7/25–26 (D5-6) | Attribute | Task2 6-class SegFormer + ROI mean + tutorial thresholds; sparse-class head; Task3 rule report + consistency check; CLIP index |
| 7/27 (D7) | Report + Bonus | submit lit review; MVP fully passing; DINOv2 comparison; template RAG; audit/sanity check |
| 7/28 (D8) | Integration | full pipeline; one-click inference validated on val + timed; error analysis; freeze models |
| 7/29 (D9) | Finalize | generate all deliverable samples; PPT draft; report writing; Stretch wrap-up |
| 7/30 (D10) | Submit | 13:00 test set out → one-click inference → 19:00 submit; rehearse that evening |
| 7/31 | Present | 15-min pre + 5-min QA |

6-hour window rule: on 7/29, fully simulate "image → all deliverables" on val and time it; on 7/30 we only run inference, no coding.

---

## 7. Risks & mitigations

| Risk | Prob | Mitigation |
|---|---|---|
| Bug in 7/30 inference window | High | 7/29 val end-to-end + timing; fallback model |
| PEFT-SAM fails to converge | Medium | SegFormer safe main; PEFT-SAM only an upgrade, use only if it wins val |
| Low sparse-attribute recall | Medium | classification-head fallback; Focal + weighted Dice |
| VRAM OOM | Medium | AMP + grad accumulation + lower res + smaller batch |
| Interface misalignment rework | Medium | daily sync on JSON/mask format |
| Scope creep | Medium | MVP/Stretch iron rule, no Stretch until MVP works |
| Consistency score loss | Low | rule template + tutorial thresholds + auto-check |
| Uneven team skill | Medium | technical work in 1–2 people; non-technical do lit/report/PPT/viz |

---

## 8. The case for winning

The competitive logic was stated above: model choice is table stakes; the win comes from reliable execution + no weak task + deep analysis + strong presentation.

Best Segmentation (special award, medium-high): SegFormer guarantees IoU 0.87–0.90, PEFT-SAM may reach 0.89–0.91, SAM refinement lowers Hausdorff; reporting all three of Dice + IoU + Hausdorff plus multi-model ablation and error analysis shows engineering completeness.

Best Overall (combined award, medium-high, main target): the three tasks are balanced with no weak spot, Task3 consistency is maxed, Bonus audit/sanity all pass; our team structure is built for "no weak spots + strong presentation" — most tech-heavy teams present poorly and write thin reports, which is our structural edge.

Honestly, no guarantee of first place: ranking depends on opponents and the test-set distribution, both uncontrollable. But the plan maximizes expected ranking within controllable factors, with high reliability (SegFormer safe + multiple fallbacks), solid depth, and a strong narrative. The biggest uncertainties are test-distribution shift and opponent compute, hedged by val closeness, TTA, SAM refinement, and presentation quality.

---

## 9. Assessment

### 9.1 Conformity with briefing + tutorial (item by item)

| Requirement | Source | Plan | OK |
|---|---|---|---|
| Task1 seg + Dice/IoU/Hausdorff | briefing p5, tutorial p46 | §3.1, all three | yes |
| Task2 5-attr, per-attr metrics, sparse handling | briefing p6, tutorial p53 | §3.2 | yes |
| Task2 presence = mean logits over lesion ROI | tutorial p40 | §3.2 (corrected from coverage) | yes |
| Task3 JSON + 5 terms + status match + evidence align | briefing p7, tutorial p40 | §3.3 | yes |
| Task3 tutorial thresholds (status/border/size) | tutorial p40 | §3.3 hardcoded | yes |
| Preprocessing DullRazor + Shades of Gray | tutorial p35 | §2 added | yes |
| Bonus CLIP + neighbor ID + audit/sanity | briefing p8, tutorial p41/46 | §3.4 | yes |
| Deliverables masks/attr/report/bonus | briefing p9 | §4 aligned with example | yes |
| Scoring weights seg25/attr25/report20/bonus | briefing p9 | no weak spot | yes |
| Report ≤12 pages, 4 sections | briefing p11 | §4 | yes |
| Code submission + reproducibility check | briefing p9 | §4 fixed seed + one-click | yes |
| Key dates | briefing p12 | §6 | yes |
| One deliverable set per group | briefing p9 | §5 | yes |
| Deterministic train/val split | tutorial p56 | §2 fixed seed | yes |

Three things to confirm: Task1 masks are stored as jpg (lossy) — follow the example but ask the TA if png is allowed; Task3 JSON fields are ambiguous (scoring mentions recommendation etc., example doesn't) — follow the example and ask the TA; Bonus weight percentage isn't stated in the PDF — handled as a high-value item.

### 9.2 Timeliness

Feasible, but the GPU is the hard bottleneck. The MVP is about 10 GPU-hours, Stretch 15–20 hours; 10 days × 8–10 effective GPU-hours = an 80–100 hour window, so compute is enough. The biggest risk is still the 6-hour window on 7/30, hedged by the 7/29 val rehearsal. On-time probability is medium-high (about 75%), assuming the MVP passes by 7/27 and GPU time is scheduled.

### 9.3 Winning

| Award | Reachability | Basis |
|---|---|---|
| Best Segmentation | Medium-high | SegFormer safe + PEFT-SAM upgrade + SAM refine + three metrics + ablation |
| Best Overall | Medium-high (main target) | no weak spots + maxed consistency + bonus audit/sanity + strong presentation (team-structure edge) |

Honest disadvantages: single 8 GB laptop GPU (opponents may have workstations); single coder; unknown test distribution. The levers: Hausdorff + error-analysis depth, Bonus audit/sanity all passing + CLIP/DINOv2 comparison, full ablations in the report, a one-sentence unified-architecture narrative.

Conclusion: the plan is feasible, with no fatal flaw and a fallback for every failure mode. Best Overall is the more realistic target; first place isn't guaranteed, but expected ranking is maximized within controllable factors.
