# YOLO11s fine-tuning report

Fine-tuned the pinned YOLO11s license-plate detector on the original-scale
train split, compared it against the frozen pretrained baseline under the
shared detection harness, and re-ran the CRNN end-to-end recognition protocol
with the better detector. All runs used the pinned configurations; every
number below traces to a metrics JSON or training record listed in
[Artifacts](#artifacts).

## Training

The pinned pretrained detector (`yolo11s-license-plate`, start-weight
SHA-256 `95e50c25ab7066dd0ca5aec18fa80349676db08697780d1149576461174d2381`)
was fine-tuned with `models/yolo11_finetune/train_config.yaml`: 640 px
images, up to 100 epochs, patience 20, batch 16, seed 42, deterministic mode,
on the 3,500 original-scale train images (700 validation, no test).

- Machine: RTX 3090 Ti, CUDA 12.4, Python 3.12, torch 2.6.0+cu124,
  ultralytics 8.3.162 (pinned GPU environment).
- Completed 24 of 100 epochs (early stop: best epoch 4 + patience 20) in
  1,439.8 s (~24 min).
- Selection metric `metrics/mAP50(B)`, first maximum on ties; best
  validation mAP@0.5 **0.995** at epoch 4.
- Optimizer: Ultralytics `optimizer=auto` selected AdamW with lr 0.002
  (the config's `lr0: 0.01` is only the value used when the optimizer is
  not auto-selected); recorded in the training record.
- Final checkpoint `best.pt` SHA-256
  `0a4f080fe3749b8be145101468a0dd98455cf50117d2acd37575f33e0d261055`,
  matching `final_checkpoint_sha256` in the training record.

Deviations from the change plan: training ran on this repository's machine
after its GPU was freed (not a separate machine), and the 2-epoch smoke run
was skipped in favor of going straight to the full run.

## Detector comparison

Frozen baseline vs fine-tuned detector, identical inference settings
(CPU, single-threaded, imgsz 640, conf 0.001, iou 0.7, batch 1, seed 42)
under `baseline/detection/evaluate.py`. The fine-tuned run's Ultralytics
cross-check on validation agreed with the custom AP to 6e-07.

| Split | mAP@0.5 | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| val — frozen | 0.9245 | 0.9402 | 0.8986 | 0.9189 |
| val — fine-tuned | 0.9949 | 0.9549 | 0.9986 | 0.9763 |
| test — frozen | 0.9073 | 0.9408 | 0.8938 | 0.9167 |
| test — fine-tuned | 0.9949 | 0.9466 | 0.9975 | 0.9714 |

Per size bucket and tilt bin, the fine-tuned detector improves mAP@0.5 in
every scope (val deltas: small +0.1143, medium +0.0251, large +0.0546;
tilt 0-9 +0.0568, 10-19 +0.0414, 20+ +0.0727; test deltas: small +0.1497,
medium +0.0445, large +0.0578; tilt 0-9 +0.0354, 10-19 +0.0846, 20+ +0.0996).
Recall improves everywhere (val large +0.1558 is the largest); precision
dips in a few tilt bins because the fine-tuned detector emits more, lower
confidence detections. Full tables: `comparison_val.json` and
`comparison_test.json`.

The fine-tuned detector is the winner (validation mAP@0.5 0.9949 vs
0.9245) and is used for the end-to-end re-run.

## End-to-end re-run

The CRNN recognizer checkpoint was not present on this machine (gitignored
artifact of the CRNN change), so the recognizer was re-trained with the
fixed full configuration (50 epochs, batch 32, Adam lr 0.001, seed 42,
CPU, single-threaded, 3,500 train / 700 validation crops) in 4,154.9 s.
Selected epoch 39 (best validation character accuracy 0.7564); checkpoint
SHA-256 `92d1fbeae19a81ddf3ae498a40e4416ed619ad9667736c592ccbb899e59ca2cc`
from its training record.

The end-to-end protocol (`baseline/ocr/run.py --protocol end_to_end
--engine crnn`) ran on all three preprocessing modes over the fine-tuned
detector's validation predictions, then selected mode and threshold from
validation only:

| Mode | Accepted / 700 | Coverage | Char accuracy | Exact plates | Plate F1 |
|---|---:|---:|---:|---:|---:|
| none | 698 | 0.9971 | 0.7534 | 237 | 0.7121 |
| gray_contrast | 700 | 1.0000 | 0.7500 | 222 | 0.7144 |
| canny | 697 | 0.9957 | 0.4539 | 5 | 0.0900 |

The shared selection rule picked mode `none`, tau 0.50. The test split was
then run exactly once under that policy (one-shot reservation recorded):

| Metric | Frozen detector | Fine-tuned detector | Delta |
|---|---:|---:|---:|
| Coverage | 0.9762 | 1.0000 | +0.0238 |
| Character accuracy | 0.6561 | 0.7661 | +0.1100 |
| Exact plates | 133 / 800 | 296 / 800 | +163 |
| Precision | 0.3624 | 0.5837 | +0.2213 |
| Recall | 0.9371 | 1.0000 | +0.0629 |
| Plate F1 | 0.5226 | 0.7372 | +0.2146 |

The improvement is consistent across size buckets (F1: small 0.6371 →
0.8750, medium 0.6355 → 0.8312, large 0.2177 → 0.4073) and tilt bins (F1:
0-9 0.4749 → 0.7396, 10-19 0.1194 → 0.2500, 20+ 0.5696 → 0.7742). The
frozen-detector column is from `models/crnn/REPORT.md` (CRNN with frozen
YOLO predictions); the fine-tuned column is from
`metrics_test_none.json`.

## Reproducibility scope

All training and evaluation runs are reproducible on the same machine and
pinned environment: fixed seeds (42), deterministic mode, single-threaded
CPU inference, pinned package versions, and input digests recorded in every
artifact. Cross-machine byte-identity of checkpoints is not claimed.

## Artifacts

- Training: `results/finetune/weights/training_record.json`,
  `results/finetune/weights/best.pt`
- Detector evaluation: `baseline/detection/results/baseline/` (frozen),
  `baseline/detection/results/finetuned-val/` and `finetuned-test/`
  (fine-tuned), comparisons in `results/finetune/comparison_{val,test}.json`
- CRNN re-training: `models/crnn/results/full-training/training_record.json`,
  `checkpoint.pt`
- End-to-end: `models/crnn/results/evaluation-finetuned/end_to_end/`
  (`raw_val_*.json`, `metrics_val_*.json`, `selection.json`,
  `raw_test_none.json`, `metrics_test_none.json`,
  `test_inference_end_to_end.json`)

All of these are gitignored local outputs; the digests above make them
verifiable.
