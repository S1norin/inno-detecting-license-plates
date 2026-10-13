# Baseline evaluation report

Run: `eval-78d42c59ff22f6c9`  
Generated: 2026-10-13T12:58:42.305022+00:00

## Inputs

- dataset: `data/dataset` (val 700, test 800)
- detection: `baseline/evaluation/results/inputs/detection` model `yolo11s-license-plate`
- ocr: `baseline/evaluation/results/inputs/ocr` engine `{'name': 'easyocr', 'version': '1.7.2', 'languages': ['ch_sim', 'en'], 'device': 'cpu', 'quantize': False, 'decoder': 'greedy', 'character_confidence': 'max probability in each collapsed CTC run', 'contrast_retry': False, 'torch_version': '2.6.0+cpu', 'torchvision_version': '0.21.0+cpu', 'numpy_version': '2.1.3', 'adapter_sha256': 'be86130547425ce7522a1c6127a2383351f7ad4dbc449d08e774507f53115b93', 'ctc_sha256': '65f173cbd8c43b59c5f8b6d913821b720824894ed18f16f9c26c79dd2d565434'}`

## E1 — Detection (YOLO)

### val

| metric | value |
| --- | --- |
| image_count | 700 |
| ground_truth_count | 700 |
| tp | 629 |
| fp | 40 |
| fn | 71 |
| map50 | 0.9245 |
| precision | 0.9402 |
| recall | 0.8986 |
| f1 | 0.9189 |

**by size bucket**

| bucket | map50 | precision | recall | f1 |
| --- | --- | --- | --- | --- |
| small | 0.8807 | 0.9083 | 0.9083 | 0.9083 |
| medium | 0.9699 | 0.9518 | 0.9476 | 0.9497 |
| large | 0.9404 | 0.9652 | 0.8398 | 0.8981 |

**by tilt bin**

| bucket | map50 | precision | recall | f1 |
| --- | --- | --- | --- | --- |
| 0-9 | 0.9380 | 0.9609 | 0.8786 | 0.9179 |
| 10-19 | 0.9536 | 0.9608 | 0.9245 | 0.9423 |
| 20+ | 0.9222 | 0.9327 | 0.9014 | 0.9168 |

### test

| metric | value |
| --- | --- |
| image_count | 800 |
| ground_truth_count | 800 |
| tp | 715 |
| fp | 45 |
| fn | 85 |
| map50 | 0.9073 |
| precision | 0.9408 |
| recall | 0.8938 |
| f1 | 0.9167 |

**by size bucket**

| bucket | map50 | precision | recall | f1 |
| --- | --- | --- | --- | --- |
| small | 0.8453 | 0.8935 | 0.9004 | 0.8969 |
| medium | 0.9505 | 0.9731 | 0.9134 | 0.9423 |
| large | 0.9371 | 0.9578 | 0.8664 | 0.9098 |

**by tilt bin**

| bucket | map50 | precision | recall | f1 |
| --- | --- | --- | --- | --- |
| 0-9 | 0.9593 | 0.9867 | 0.8862 | 0.9338 |
| 10-19 | 0.9018 | 1.0000 | 0.8095 | 0.8947 |
| 20+ | 0.8954 | 0.9231 | 0.9053 | 0.9141 |

## E2 — OCR-only (GT crops)

Selected on validation: mode `none`, tau 0.5, score `mean`.

### val

| metric | value |
| --- | --- |
| tp | 22 |
| fp | 495 |
| fn | 183 |
| count | 700 |
| accepted | 517 |
| coverage | 0.7386 |
| character_accuracy | 0.3580 |
| plate_exact_match | 0.0043 |
| accepted_exact_accuracy | 0.0058 |
| precision | 0.0426 |
| recall | 0.1073 |
| f1 | 0.0609 |

### test

| metric | value |
| --- | --- |
| tp | 22 |
| fp | 562 |
| fn | 216 |
| count | 800 |
| accepted | 584 |
| coverage | 0.7300 |
| character_accuracy | 0.3474 |
| plate_exact_match | 0.0000 |
| accepted_exact_accuracy | 0.0000 |
| precision | 0.0377 |
| recall | 0.0924 |
| f1 | 0.0535 |

## E2b — End-to-end (predicted boxes)

Selected on validation: mode `none`, tau 0.5, score `mean`.

### val

| metric | value |
| --- | --- |
| tp | 14 |
| fp | 480 |
| fn | 206 |
| count | 700 |
| accepted | 494 |
| coverage | 0.7057 |
| character_accuracy | 0.3279 |
| plate_exact_match | 0.0014 |
| accepted_exact_accuracy | 0.0020 |
| precision | 0.0283 |
| recall | 0.0636 |
| f1 | 0.0392 |

### test

| metric | value |
| --- | --- |
| tp | 15 |
| fp | 547 |
| fn | 238 |
| count | 800 |
| accepted | 562 |
| coverage | 0.7025 |
| character_accuracy | 0.3240 |
| plate_exact_match | 0.0000 |
| accepted_exact_accuracy | 0.0000 |
| precision | 0.0267 |
| recall | 0.0593 |
| f1 | 0.0368 |

**delta vs. OCR-only (e2e - ocr_only)**

| split | char_acc | exact | acc_exact | coverage |
| --- | --- | --- | --- | --- |
| val | -0.0301 | -0.0029 | -0.0038 | -0.0329 |
| test | -0.0234 | 0.0000 | 0.0000 | -0.0275 |

## E3 — Preprocessing ablation (OCR-only, val, tau=0.5)

| mode | char_acc | exact | acc_exact | coverage | d_char | d_exact | d_cov |
| --- | --- | --- | --- | --- | --- | --- | --- |
| none | 0.3580 | 0.0043 | 0.0058 | 0.7386 | - | - | - |
| gray_contrast | 0.2693 | 0.0014 | 0.0025 | 0.5714 | -0.0886 | -0.0029 | -0.1671 |
| canny | 0.0328 | 0.0000 | 0.0000 | 0.1257 | -0.3252 | -0.0043 | -0.6129 |

## E6 — Accuracy/coverage at the selected threshold (test)

| protocol | tau | accuracy | coverage |
| --- | --- | --- | --- |
| ocr_only | 0.5000 | 0.0000 | 0.7300 |
| end_to_end | 0.5000 | 0.0000 | 0.7025 |

## Failure cases

### Detection — val (5 cases)

- `images/val/0ded363fecc2fc1f_s100.jpg` — miss (gt=1, pred=0, best_iou=0.0000, best_score=0.0000)
- `images/val/1561c3e94b8bf2fe_s100.jpg` — miss (gt=1, pred=0, best_iou=0.0000, best_score=0.0000)
- `images/val/201ead679535ab84_s100.jpg` — miss (gt=1, pred=0, best_iou=0.0000, best_score=0.0000)
- `images/val/2b81955186414d14_s100.jpg` — miss (gt=1, pred=0, best_iou=0.0000, best_score=0.0000)
- `images/val/33d83c935cf2d7a7_s100.jpg` — miss, multiple_detections, low_confidence (gt=1, pred=3, best_iou=0.0000, best_score=0.0000)

### Detection — test (5 cases)

- `images/test/031ea44e43389e54_s100.jpg` — miss (gt=1, pred=0, best_iou=0.0000, best_score=0.0000)
- `images/test/13a98853c1e6f43f_s100.jpg` — miss (gt=1, pred=0, best_iou=0.0000, best_score=0.0000)
- `images/test/3849398a5cfba4c4_s100.jpg` — miss (gt=1, pred=0, best_iou=0.0000, best_score=0.0000)
- `images/test/4751e26240077779_s100.jpg` — miss (gt=1, pred=0, best_iou=0.0000, best_score=0.0000)
- `images/test/4efe8ac3cfbfb80e_s100.jpg` — miss, multiple_detections, low_confidence (gt=1, pred=4, best_iou=0.0000, best_score=0.0000)

### Recognition — ocr_only (none, tau=0.5; 5 cases)

- `00add693a6114a31_s100` — abstained: gt=`皖AW7D77` text=`王` (status=read)
- `02d8de34c4583629_s100` — substitution, missing_chars: gt=`皖AL1T88` text=`TUIIN` (status=read)
- `0329c17c8a183430_s100` — abstained: gt=`皖A858W8` text=`_|` (status=read)
- `0422e4691631c37b_s100` — abstained: gt=`皖A21L99` text=`8` (status=read)
- `0a4c101d8d966db4_s100` — abstained: gt=`苏A9G6A1` text=`%L孙` (status=read)

### Recognition — end_to_end (none, tau=0.5; 5 cases)

- `9bc93bb6a1b0fb55_s100` — substitution, extra_chars: gt=`皖AC357U` text=`|皖4.63570055 15 {TUNII` (status=read)
- `e3793503646bf28e_s100` — substitution, extra_chars: gt=`皖AS0558` text=`慨450558CILRYC` (status=read)
- `3df9078f89084f5b_s100` — substitution, extra_chars: gt=`皖AES366` text=`ESBGGI既1` (status=read)
- `755884cc17626dac_s100` — substitution, extra_chars: gt=`皖AY8H01` text=`皖4 Y8H0 |035N` (status=read)
- `d9a728d9a1fcc0df_s100` — abstained: gt=`皖AX1U38` text=`170/3{皖』` (status=read)

## Timing probe

n=100, seed=42, warmup excluded per stage.

| stage | count | median ms | p95 ms | mean ms | total s | max RSS MB |
| --- | --- | --- | --- | --- | --- | --- |
| detection | 95 | 70.1981 | 102.2598 | 73.8096 | 7.0119 | 530.2383 |
| ocr_none | 95 | 203.8015 | 467.5495 | 237.7463 | 22.5859 | 1163.7344 |
| preprocess_none | 95 | 0.3702 | 0.8536 | 0.4176 | 0.0397 | 1163.7344 |
| ocr_gray_contrast | 95 | 208.0251 | 470.2822 | 238.9447 | 22.6997 | 1526.3945 |
| preprocess_gray_contrast | 95 | 0.4943 | 1.0946 | 0.5388 | 0.0512 | 1526.3945 |
| ocr_canny | 95 | 200.4197 | 428.2461 | 225.8520 | 21.4559 | 1570.6211 |
| preprocess_canny | 95 | 0.4546 | 1.0533 | 0.5172 | 0.0491 | 1570.6211 |

## Notes

- Detection and OCR metrics come from the frozen stage results; nothing was retrained or re-evaluated here.
- Recognition tilt bins are reported on validation only; the frozen test reports predate
  per-sample tilt records.
- Failure cases are ranked deterministically; recognition cases come from validation so the test split is
  untouched.
- The timing probe measured 100 sampled images (seed 42), not the full split; full-split timing is tracked as a follow-up.
