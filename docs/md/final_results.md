# Final results: license plate detection and recognition on CCPD2019

Single source of truth for the numbers behind the two-page technical summary.
Every value below was regenerated on 2026-11-02 from the canonical weights
listed in [Weights and digests](#weights-and-digests); per-change
`REPORT.md` files remain the historical record of their experiments.

## System definitions

- **Frozen detector**: off-the-shelf YOLO11s (`yolo11s-license-plate`,
  Ultralytics) on the original-scale test images, no fine-tuning.
- **Fine-tuned detector**: the same YOLO11s backbone fine-tuned on the 3,500
  train images (24 epochs, best epoch 4, early stop patience 20).
- **CRNN recognizer**: 3.0M-parameter grayscale CRNN (CNN + bidirectional
  LSTM + CTC), 59 classes, trained 50 epochs (39 selected) on the 3,500
  original-scale train crops (700 validation), CPU, 1 thread, 4,154.9 s
  total on a Linux x86_64 machine.
- **EasyOCR baseline**: frozen off-the-shelf; numbers below are the committed
  frozen baseline (`baseline/ocr/REPORT.md`), not rerun.

Protocols: `ocr_only` recognizes ground-truth crops (isolates OCR from the
detector); `end_to_end` recognizes crops cut with the detector's own test
predictions (top-1 box per image).

## Dataset

CCPD2019 subset: 5,000 source photos -> 15,000 variants
(scales 1.0 / 0.75 / 0.5, 30 tilt angles), split before resize so all
variants of one photo stay in one split: train 3,500 / val 700 / test 800.
Size buckets from train only: small <= 83 px, medium <= 112 px, large > 112 px
(plate width). Tilt bins: 0-9 / 10-19 / 20+.

The primary metrics below use original-scale (1.0) images and crops; the
0.75/0.5 variants are evaluated in [Scale robustness](#scale-robustness).

## Weights and digests

| Weight | SHA-256 |
| --- | --- |
| Frozen YOLO11s | `95e50c25ab7066dd0ca5aec18fa80349676db08697780d1149576461174d2381` |
| Fine-tuned YOLO11s (`best.pt`) | `0a4f080fe3749b8be145101468a0dd98455cf50117d2acd37575f33e0d261055` |
| CRNN checkpoint (epoch 39) | `92d1fbeae19a81ddf3ae498a40e4416ed619ad9667736c592ccbb899e59ca2cc` |

All weights are fetched and digest-verified through
`baseline/common/models.json` (`python -m baseline.common.weights <name>`).
Host: Apple M3, CPU only, single-threaded inference, Ultralytics/YOLO settings
`imgsz=640, conf=0.001, iou=0.7`; CRNN greedy CTC decoding.

## Detection: final test metrics (800 images)

| Detector | mAP@0.5 | Precision | Recall | F1 |
| --- | --- | --- | --- | --- |
| Frozen YOLO11s | 0.9073 | 0.9408 | 0.8938 | 0.9167 |
| Fine-tuned YOLO11s | 0.9949 | 0.9466 | 0.9975 | 0.9714 |

Validation (for reference): frozen 0.9245 / 0.9402 / 0.8986 / 0.9189;
fine-tuned 0.9949 / 0.9549 / 0.9986 / 0.9763. The fine-tuned detector was
cross-checked with native Ultralytics validation (identical mAP to 1e-4).

Fine-tuned test by scope (mAP@0.5 / F1): small 0.9950 / 0.9647,
medium 0.9950 / 0.9753, large 0.9948 / 0.9739;
tilt 0-9 0.9947 / 0.9707, 10-19 0.9863 / 0.9254, 20+ 0.9950 / 0.9769.
The 10-19-degree bin is the weakest detection scope for both detectors.

Artifacts: `baseline/detection/results/finetuned-{val,test}-20261102/`,
`baseline/detection/results/review-macos/` (frozen),
`models/yolo11_finetune/results/finetune/comparison_{val,test}.json`.

## Recognition: final test metrics (800 samples, mode `none`, tau = 0.5)

| Engine | Protocol | Char accuracy | Exact plates | F1 |
| --- | --- | --- | --- | --- |
| EasyOCR (frozen) | ocr_only | 0.3474 | 0 / 800 | - |
| EasyOCR (frozen) | end_to_end | 0.3240 | 0 / 800 | - |
| CRNN | ocr_only | 0.7593 | 267 / 800 | 0.7271 |
| CRNN | end_to_end + frozen detector | 0.6710 | 148 / 800 | 0.5676 |
| CRNN | end_to_end + fine-tuned detector | 0.7661 | 296 / 800 | 0.7372 |

Preprocessing ablation (validation, char accuracy): `none` 0.7558,
`gray_contrast` 0.7469, `canny` 0.4575 - no preprocessing wins; Canny
destroys the glyph information the network needs. Selection policy
(accepted exact accuracy + coverage - 1) chose `none` / tau 0.50 for all
three protocol groups; test was run once per group after selection.

CRNN test by scope, ocr_only (char accuracy / exact rate): small 0.8790 /
0.4751, medium 0.8295 / 0.4152, large 0.5658 / 0.1069; tilt 0-9 0.7996 /
0.2335, 10-19 0.5193 / 0.0000, 20+ 0.7740 / 0.4000. Large plates and the
10-19-degree tilt bin are the dominant recognition failures.

Artifacts: `models/crnn/results/evaluation_canonical/{ocr_only,end_to_end,end_to_end_finetuned}/`.

## Scale robustness

The 0.75 and 0.5 variants (800 test images / 800 test crops each) were
evaluated with the same weights, settings, and selection policy
(`none` / tau 0.5 selected on the 700-sample validation split for every
protocol-scope group). No model was retrained.

Detection (test, 800 images per scale):

| Scale | Detector | mAP@0.5 | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- |
| 1.0 | Frozen YOLO11s | 0.9073 | 0.9408 | 0.8938 | 0.9167 |
| 0.75 | Frozen YOLO11s | 0.9218 | 0.9490 | 0.9062 | 0.9271 |
| 0.5 | Frozen YOLO11s | 0.9294 | 0.9504 | 0.9100 | 0.9298 |
| 1.0 | Fine-tuned YOLO11s | 0.9949 | 0.9466 | 0.9975 | 0.9714 |
| 0.75 | Fine-tuned YOLO11s | 0.9949 | 0.9489 | 0.9975 | 0.9726 |
| 0.5 | Fine-tuned YOLO11s | 0.9949 | 0.9444 | 0.9988 | 0.9708 |

Recognition (test, 800 crops per scale, mode `none`, tau 0.5):

| Scale | Protocol | Char accuracy | Exact plates | F1 |
| --- | --- | --- | --- | --- |
| 1.0 | ocr_only | 0.7593 | 267 / 800 | 0.7271 |
| 0.75 | ocr_only | 0.7613 | 265 / 800 | 0.7402 |
| 0.5 | ocr_only | 0.7610 | 268 / 800 | 0.7332 |
| 1.0 | end_to_end + fine-tuned | 0.7661 | 296 / 800 | 0.7372 |
| 0.75 | end_to_end + fine-tuned | 0.7629 | 297 / 800 | 0.7342 |
| 0.5 | end_to_end + fine-tuned | 0.7555 | 279 / 800 | 0.7200 |
| 1.0 | end_to_end + frozen | 0.6710 | 148 / 800 | 0.5676 |
| 0.75 | end_to_end + frozen | 0.6713 | 157 / 800 | 0.5599 |
| 0.5 | end_to_end + frozen | 0.6721 | 144 / 800 | 0.5520 |

Findings:

- Detection is flat across scales for both detectors (frozen mAP 0.907 ->
  0.930, fine-tuned 0.9949 at all three scales); the fine-tuned detector's
  F1 drops by only 0.0006 from 1.0 to 0.5.
- CRNN character accuracy is also flat (ocr_only 0.7593 -> 0.7610 -> 0.7610):
  the recognizer resizes every crop to 48x256, so absolute image scale is
  largely normalized away and what matters is the crop content.
- The quality loss appears in the per-size-bucket breakdown: at 0.5 scale,
  the medium and large buckets (former large/medium parent plates) drop to
  0.44 and 0.34 character accuracy respectively (end-to-end + fine-tuned)
  while the small bucket stays at ~0.80 - the same large-plate weakness seen
  at 1.0, now covering more of the test population. Buckets are per-variant:
  "large" at 0.5 means >112 px plate width at that scale.
- The frozen detector's end-to-end gap to ocr_only (~0.09 character accuracy)
  is constant across scales: detector quality is the limiting factor at every
  scale, and fine-tuning removes it at every scale.

Figure: [`figures/scale_robustness.png`](../figures/scale_robustness.png).

Artifacts: `baseline/detection/results/{frozen,finetuned}-{val,test}-s075-s050/`,
`models/crnn/results/scaled/{ocr_only_s075,ocr_only_s050,end_to_end_{finetuned,frozen}_{s075,s050}}/`.

## Runtime (Apple M3, CPU, 95 timed samples after 5 warmup, seed 42)

| Stage | Frozen detector | Fine-tuned detector |
| --- | --- | --- |
| Detection, median / p95 | 77.8 ms / 88.0 ms | 80.2 ms / 98.8 ms |
| CRNN recognition, median / p95 | 5.6 ms / 6.3 ms | 5.7 ms / 6.6 ms |
| Preprocessing, median | ~0.4 ms | ~0.4 ms |
| Peak RSS | ~508 MB | ~498 MB |

A full end-to-end image (detection + recognition of the top box) takes about
83-86 ms on CPU. Training runtimes: CRNN 4,154.9 s (~69 min, CPU, 1 thread,
50 epochs, Linux x86_64); fine-tuned YOLO 1,439.8 s (~24 min, RTX 3090 Ti).

Artifacts: `baseline/evaluation/results/final_{frozen,finetuned}/timing.json`.

## Failure analysis

Six representative failures from the regenerated test caches (sample IDs
recorded so the montages are reproducible):

| # | Sample | Stage | Observation | Likely cause |
| --- | --- | --- | --- | --- |
| 1 | `6f3c772ca888fa92_s100` | Frozen detector | Large plate, tilt 19 degrees, zero predictions at any score | Weak prior for large plate aspect ratios in the pretrained model; 10-19-degree bin is the frozen model's weakest tilt scope |
| 2 | `b996022cc9a5ee07_s100` | Frozen detector | Large plate, tilt 45 degrees, zero predictions | Same: extreme tilt + large size outside the pretrained distribution |
| 3 | `356b4ca635beeca7_s100` | Fine-tuned detector | 11 boxes: 1 true (0.84) + 10 false positives (0.18, 0.06, ...) | Fine-tuning removed the misses but left low-confidence false positives on cluttered scenes; a higher confidence cutoff would drop them |
| 4 | `588e616d9e7cec6c_s100` | CRNN (end-to-end + fine-tuned) | GT `苏E050R8`, predicted `皖AK81`; large, tilt 19 degrees | The model overfits small-plate statistics: wrong province char and wrong length on a large tilted plate |
| 5 | `261f3b723deb67ac_s100` | CRNN (OCR-only) | GT `京K06350`, predicted `皖AF4069`; large, tilt 29 degrees | Full hallucination with low per-character confidence (min 0.24): large plate glyphs fall outside the training distribution |
| 6 | `bfe070b4b5543916_s100` | CRNN (OCR-only) | GT `浙C7HV30`, predicted `皖A9W165`; medium, tilt 18 degrees | Province-char confusion plus plausible-looking digits; the 10-19-degree tilt bin has 0% exact match on test |

Cross-cutting pattern: detection failures concentrate on **large + tilted**
plates (the frozen model's blind spot, largely fixed by fine-tuning), while
recognition failures concentrate on the **same large + 10-19-degree tilt**
region, where the recognizer was effectively untrained. Both point at the
same data gap: the dataset is dominated by small, near-0-degree plates.

Figures: [`figures/failure_detection.png`](../figures/failure_detection.png),
[`figures/failure_ocr.png`](../figures/failure_ocr.png),
[`figures/baseline_comparison.png`](../figures/baseline_comparison.png).

## Reproducing these numbers

```bash
# weights (digest-verified)
python -m baseline.common.weights yolo11s-license-plate-finetuned
python -m baseline.common.weights crnn-full-training

# fine-tuned detection (val + test)
python -m baseline.detection.evaluate --split val --run-id finetuned-val-20261102 \
    --model baseline/weights/yolo11_finetune/best.pt --validate
python -m baseline.detection.evaluate --split test --run-id finetuned-test-20261102 \
    --model baseline/weights/yolo11_finetune/best.pt

# CRNN (validation -> selection -> one-shot test), per protocol
python -m baseline.ocr.run --protocol ocr_only --split val --mode none \
    --output models/crnn/results/evaluation_canonical/ocr_only/raw_val_none.json \
    --engine crnn
# (same for gray_contrast, canny; then end_to_end with --predictions,
#  end_to_end_finetuned with the fine-tuned predictions)
python -m baseline.ocr.evaluate select --protocol ocr_only --split val \
    --raw ... --output .../selection.json
python -m baseline.ocr.run --protocol ocr_only --split test --mode none \
    --selection .../selection.json --output .../raw_test_none.json --engine crnn
python -m baseline.ocr.evaluate test --protocol ocr_only --split test \
    --raw .../raw_test_none.json --output .../metrics_test_none.json

# consolidated table + timing probe (CRNN engine)
python -m baseline.evaluation.run --dataset data/dataset \
    --detection-results <detection dir> --ocr-results <ocr dir> \
    --out baseline/evaluation/results/final_finetuned \
    --detector-weights baseline/weights/yolo11_finetune/best.pt --probe-engine crnn

# scale robustness (directory-form config, scaled variants)
python -m baseline.detection.evaluate --split test --data data/dataset/dataset.yaml \
    --scales 0.75,0.5 --run-id frozen-test-s075-s050
python -m baseline.detection.evaluate --split test --data data/dataset/dataset.yaml \
    --scales 0.75,0.5 --run-id finetuned-test-s075-s050 \
    --model baseline/weights/yolo11_finetune/best.pt
python -m baseline.ocr.run --engine crnn --protocol ocr_only --split val --mode none \
    --scale 0.75 --output models/crnn/results/scaled/ocr_only_s075/raw_val_none.json
# (same for the other modes/scales; end_to_end adds --predictions with the
#  per-scale prediction files from the scaled detection runs)
```

## Known limitations

- EasyOCR baseline is frozen on an earlier machine; its absolute numbers are
  indicative, not comparable-run.
- Fine-tuned YOLO training was done on a GPU (RTX 3090 Ti); only inference
  was rerun locally.
- The timing probe measures CPU single-threaded latency, not throughput.
- The 10-19-degree tilt bin (63 test samples) and large plates remain the
  system's weak spots; a larger tilt/size-balanced training set is the
  obvious next step.
