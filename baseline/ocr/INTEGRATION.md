# Inputs needed from #4 and #5

This branch can be merged alongside the detector branch: it introduces only
`baseline/ocr/**` and `baseline/requirements-ocr.txt`. It does not implement or
modify `baseline/common/`, detection, shared requirements or the dataset.

## Shared environment and weights (#4)

Use one `.venv-baseline` and agree on the torch/torchvision pair before adding
`baseline/requirements-ocr.txt`. The OCR code never installs dependencies.

Two EasyOCR files are required:

| File | Upstream archive |
| --- | --- |
| `craft_mlt_25k.pth` | `https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/craft_mlt_25k.zip` |
| `zh_sim_g2.pth` | `https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/zh_sim_g2.zip` |

The archives contain the named weight files. The shared downloader must unpack
them and pin/verify the **extracted file's SHA-256**, not just the archive hash.
Upstream's MD5 receipts are not substituted for SHA-256. Do not insert invented
or placeholder digests. The OCR verifier reads entries of this form:

```text
name: easyocr-zh-sim-g2
url: upstream URL handled by the shared downloader
filename: easyocr/zh_sim_g2.pth
sha256: actual 64-character SHA-256 of the extracted .pth file
```

Accepted manifest containers: a list of entries, a name-to-entry mapping, or
either under a `models` key. Entries retain the planned `name`, `url`,
`filename`, `sha256` fields. OCR identifies its entries by filename and reads
the already extracted files in `baseline/weights/easyocr/`.

If #4 chooses another manifest/cache layout, supply `--weights-manifest` and
`--weights-directory` or adapt this small verifier. It requires no specific
`baseline.common` function names. The two OCR entries are intentionally not
added to a competing shared manifest in this branch; add them after actual
downloads and hashes are available.

## Saved detector predictions (#5)

OCR consumes a file, not a detector implementation API. Convert saved
predictions to this schema if the detector uses another representation:

```json
{
  "schema_version": 1,
  "metadata": {
    "model_id": "the detector's actual model identifier",
    "weight_sha256": "the detector's actual weight SHA-256",
    "box_format": "xyxy_pixels"
  },
  "records": [
    {
      "image_path": "images/val/ee22be0fe8e36f86_s100.jpg",
      "boxes": [[40.5, 120.0, 200.0, 170.0]],
      "scores": [0.9]
    }
  ]
}
```

Paths are relative to `data/dataset/` and must match `metadata.csv`.
Include every selected split image exactly once; represent no detections
with `boxes: []`, `scores: []`. Coordinates are original-image pixels in
`[x1, y1, x2, y2]` order, not normalized YOLO center/width/height.
For scale 1.0, include 700 val or 800 test images. Scaled experiments have
their own selected prediction files. OCR chooses the highest-score box
(first box breaks score ties), rounds outwards and clamps it to the image.
An empty/out-of-image crop counts as no detection.

The detector's own confidence/NMS settings should remain fixed between val
and test. Preserve them as extra metadata fields. The detector model id and
weight hash must match the validation-selected contract. Add a format adapter
once #5's actual output is available; no detector rerun is needed for conversion.

## After combining branches

1. Resolve shared dependency versions and register/extract/hash the two weights.
2. Make the prepared dataset images and crops available; retain existing splits.
3. Smoke-test EasyOCR on one validation crop with network access disabled.
4. Run all three validation modes for OCR-only; select its policy.
5. Adapt #5's validation predictions and run/select end-to-end validation.
6. Run each protocol on test once with its selected mode and threshold.
7. Export and validate the training manifest; generate plots and record metrics.

The real-model checks and final metrics remain prerequisites for completing
#6/#7. This branch supplies the implementation and independent checks.
