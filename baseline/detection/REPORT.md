# Pretrained YOLO11s baseline results

Evaluated on 12 October 2026 using the colleague's prepared CCPD dataset.
Its `metadata.csv` is byte-identical to the repository's tracked metadata.
Original scale only: 700 validation and 800 test images, one plate per image.
The 3,500-image training split was checked for fine-tuning readiness; no training
was performed. Settings were fixed before test evaluation.

Model: `morsetechlab/yolov11-license-plate-detection`, `license-plate-finetune-v1s.pt`.
Revision: `251a30d7daedca065f56e04b0af04052c907c68f`.
Weight SHA-256: `95e50c25ab7066dd0ca5aec18fa80349676db08697780d1149576461174d2381`.

Runtime: Python 3.12.10, Torch 2.6.0+cpu, Torchvision 0.21.0+cpu,
Ultralytics 8.3.162, NumPy 2.1.3, OpenCV 4.11.0. CPU: AMD Ryzen 7 7700X.
Inference: CPU, one thread, FP32, seed 42, batch 1, image size 640,
confidence floor 0.001, NMS IoU 0.7, max 300 predictions, square padding,
augmentation disabled. Exact dependency pins are committed.

All metrics below use matching IoU >= 0.5. Precision, recall, and F1 use
confidence >= 0.25; AP uses confidence-ranked predictions down to 0.001.

| Split | Images | mAP@0.5 | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Validation | 700 | 0.9245 | 0.9402 | 0.8986 | 0.9189 |
| Test | 800 | 0.9073 | 0.9408 | 0.8938 | 0.9167 |

Size buckets use train-derived plate-height thresholds: small <= 83 px,
medium > 83 and <= 112 px, large > 112 px.

| Size | Val images | Val mAP@0.5 | Val F1 | Test images | Test mAP@0.5 | Test F1 |
|---|---:|---:|---:|---:|---:|---:|
| Small | 240 | 0.8807 | 0.9083 | 261 | 0.8453 | 0.8969 |
| Medium | 229 | 0.9699 | 0.9497 | 277 | 0.9505 | 0.9423 |
| Large | 231 | 0.9404 | 0.8981 | 262 | 0.9371 | 0.9098 |

Tilt is the CCPD filename proxy `max(abs(horizontal), abs(vertical))`.

| Tilt | Val images | Val mAP@0.5 | Val F1 | Test images | Test mAP@0.5 | Test F1 |
|---|---:|---:|---:|---:|---:|---:|
| 0-9 | 140 | 0.9380 | 0.9179 | 167 | 0.9593 | 0.9338 |
| 10-19 | 53 | 0.9536 | 0.9423 | 63 | 0.9018 | 0.8947 |
| 20+ | 507 | 0.9222 | 0.9168 | 570 | 0.8954 | 0.9141 |

## Verification

- Fresh isolated setup succeeded, all required imports worked, and a second
  setup returned `installed: []` and `noop: true`.
- The downloaded model matched its manifest SHA-256. Corruption, interrupted
  downloads, and download-free cache access are covered by tests.
- Independent Ultralytics validation returned mAP@0.5 = 0.9234620594 versus
  custom 0.9245359408: difference 0.0010738814, below the required 0.01.
- Each split was repeated with unchanged settings. Prediction records and all
  overall/size/tilt numerical metrics were exactly identical between runs.
- Offline metric recomputation matched validation metrics. Prediction records
  cover every image, and difficulty-group counts sum to split totals.
- All 23 focused tests passed in `.venv-baseline`; Ruff passed.
- Training readiness validated the cached model and all 3,500 training images.

Local artifacts are under `results/baseline/` and `results/repeat/`, with
`predictions_{val,test}.json` and `metrics_{val,test}.json`. They remain gitignored;
this report preserves the compact results for review. JSON files include
input fingerprints, exact runtime settings, model digest, and timestamps.

These are results on the project's custom CCPD subset, not the official CCPD
benchmark split. Smaller plate detections have the lowest AP. The tilt proxy
and unequal bin counts should be considered when interpreting that breakdown.
See [README.md](README.md) for reproduction commands and metric definitions.
