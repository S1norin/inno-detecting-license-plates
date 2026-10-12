# Local integration of #4/#5 with #6/#7

The OCR branch includes Max's infrastructure commit `340f17f6` from PR #8 and
detector commit `d99a428f` from PR #9. The local merge had no conflicts.
No remote PR was merged or updated; the OCR branch remains local.

## Environment (#4)

`python -m baseline.ocr.setup` inspects the environment without installing.
After permission, `python -m baseline.ocr.setup --install` reuses the shared
environment installer with the union of detection and OCR pins. It rejects
version conflicts and preserves torch 2.6.0+cpu, torchvision 0.21.0+cpu,
NumPy 2.1.3, OpenCV 4.11.0.86 and matplotlib 3.10.3. EasyOCR is 1.7.2.

## EasyOCR weights still needed

Two files are required in `baseline/weights/easyocr/`:

| File | Upstream archive |
| --- | --- |
| `craft_mlt_25k.pth` | `https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/craft_mlt_25k.zip` |
| `zh_sim_g2.pth` | `https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/zh_sim_g2.zip` |

The current common manifest only registers YOLO. Download and extract the
two archives explicitly after permission, calculate each extracted `.pth`
file's actual SHA-256, and add entries to the shared manifest. Use plain
filenames (`zh_sim_g2.pth`, not `easyocr/zh_sim_g2.pth`) to conform to #4's
manifest validator. Never insert placeholder or invented digests.

The shared `weights.fetch()` currently handles direct weight files, not ZIP
extraction. Do not use an extracted-file hash to verify an archive. OCR reuses
`baseline.common.weights.verify()` for the extracted files, then constructs
EasyOCR with `download_enabled=False`. Hash verification happens before loading.

## Detector predictions (#5)

`--predictions baseline/detection/results/RUN/predictions_val.json` accepts
Max's native result format directly:

```text
provenance: schema_version, model_id, weight_sha256, split, inference,
            input_sha256.metadata, ...
images: [{image_path, boxes, scores, class_ids, ...}, ...]
```

The adapter preserves provenance, validates single-class pixel `xyxy` boxes,
requires every selected split image once, and rejects mismatched split or
dataset metadata fingerprints. It chooses the highest-score box, rounds
outwards and clamps to the image; empty/out-of-image crops count as no detection.
The validation-selected contract also freezes detector inference settings for
test. The earlier `schema_version` / `metadata` / `records` interchange schema
is still supported for external callers.

Max's prediction files are gitignored and absent from his commits. Obtain them
from his run, or run the detector locally after the prepared images and YOLO
weights are available. Metric summary tables alone are insufficient for OCR.

## Remaining work for real evaluation

1. Provide the prepared dataset with `images/` and `crops_gt/`; retain the
   existing train/val/test membership and metadata.
2. Install the shared baseline environment plus OCR dependencies after approval.
3. Obtain/extract/hash/register the two EasyOCR weights; smoke-test a val crop
   with downloads disabled.
4. Run all three validation modes for OCR-only and select its policy.
5. Supply detector validation predictions; run/select end-to-end validation.
6. Run test once per protocol with its selected mode/threshold and saved
   detector test predictions; generate metrics and accuracy/coverage plots.
7. Validate/export the training manifest for future CRNN experiments.

Real-model checks and final recognition metrics are still prerequisites for
completing #6/#7; synthetic integration tests are not accuracy measurements.
