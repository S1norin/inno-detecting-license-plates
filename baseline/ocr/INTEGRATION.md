# Local integration of #4/#5 with #6/#7

The OCR branch includes Max's infrastructure commit `340f17f6` from PR #8 and
detector commit `d99a428f` from PR #9. The local merge had no conflicts.
The OCR PR should target `feat/yolo11-detection-baseline` (PR #9).
Merge #8, then #9; retarget the OCR PR to `main` before its final merge.
The local integration did not merge or modify Max's remote PRs.

## Environment (#4)

`python -m baseline.ocr.setup` inspects the environment without installing.
After permission, `python -m baseline.ocr.setup --install` reuses the shared
environment installer with the union of detection and OCR pins. It rejects
version conflicts and preserves torch 2.6.0+cpu, torchvision 0.21.0+cpu,
NumPy 2.1.3, OpenCV 4.11.0.86 and matplotlib 3.10.3. EasyOCR is 1.7.2.

## EasyOCR weights

Two files are required in `baseline/weights/easyocr/`:

| File | Upstream archive |
| --- | --- |
| `craft_mlt_25k.pth` | `https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/craft_mlt_25k.zip` |
| `zh_sim_g2.pth` | `https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/zh_sim_g2.zip` |

Both archives and extracted files are now SHA-256 pinned in the common manifest,
after comparing their extracted contents to EasyOCR 1.7.2's upstream receipts.
The shared downloader supports `archive_member`, `archive_sha256` and
`cache_subdir` for these entries, preserving the direct-file YOLO path.
It extracts only the exact named member, verifies both digests and promotes
the file atomically. Corruption never becomes a usable cache entry.

Use `python -m baseline.common.weights fetch easyocr-craft` and
`python -m baseline.common.weights fetch easyocr-zh-sim-g2` after download
permission. OCR reuses the shared verifier before constructing EasyOCR with
`download_enabled=False`.

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

## Completed local evaluation

1. Prepared dataset supplied: all 15,000 images and 15,000 crops extracted,
   with byte-identical metadata and unchanged splits.
2. Shared environment installed and transitive OCR additions pinned;
   a second setup returned a no-op.
3. YOLO/EasyOCR weights verified; inference runs without downloads.
4. All three validation modes evaluated for both protocols on 700 originals;
   both policies select `none`, mean confidence and tau 0.50.
5. Detector predictions regenerated locally with Max's frozen settings;
   validation/test summary metrics reproduce his report.
6. Test evaluated once per OCR protocol on 800 originals using frozen policies.
   Metrics, raw caches and accuracy/coverage plots are saved locally.
7. Training manifest validated for all 10,500 crops; training-only charset has
   58 characters plus CTC blank. CRNN training was not part of this baseline.

See [REPORT.md](REPORT.md) for actual recognition results and limitations.
Raw artifacts remain in ignored `results/` folders; the compact report is tracked.
