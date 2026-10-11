# CCPD: prepared data (Data & Preprocessing)

Generated from CCPD original image names. Original CCPD source: https://github.com/detectRecog/CCPD

## Counts
- Original images: 5000
- Splits (originals): {'train': 3500, 'val': 700, 'test': 800}
- Image scale variants: [1.0, 0.75, 0.5]
- Exported images / YOLO txt labels / GT crops: 15000 each
- Reproducibility seed: 42
- Thresholds for pixel-height buckets (trained only on train split): small <= 83.00; medium <= 112.00; large > 112.00

## Files
- images/{train,val,test}/*.jpg — source-resolution + resized images
- labels/{train,val,test}/*.txt — YOLO single-class normalized bbox (class 0)
- dataset.yaml — all three scales; use for explicit scale experiments
- dataset_original.yaml — only scale=1.0 for primary YOLO training/evaluation
- configure_paths.py — run once after moving this result folder: `python configure_paths.py`
- lists_original/*.txt — absolute paths to original-resolution exported images; regenerate after relocation
- crops_gt/{train,val,test}/*.png — crops based on GT rectangles for OCR-only evaluation
- ocr_{train,val,test}.csv — OCR targets and crop filenames, includes scale, parent_id, bucket
- metadata.csv — per-variant size, tilt, bbox and OCR ground truth
- original_metadata.csv — per original annotation and split
- train.txt, val.txt, test.txt — **original** source relative paths (not variant files)
- preprocessing.py — preprocess_plate(image, mode, rgb3=False)
- plots/*.png — distribution plots and examples
- run_config.json — parameters used in this run

## Reproduction
1. Open Data_Preprocessing.ipynb in the complete CCPD_Data_Preprocessing folder; the selected images and archive metadata are already included in source_data/.
2. Open this notebook and optionally edit the values in Step 0.
3. Install dependencies with `pip install -r requirements.txt` or let the notebook install missing packages, then Run All.
4. Keep a record of CCPD edition/source and parameters.

## Notes
- This re-samples a new split rather than using the official benchmark split. Results are comparable only with models using this same split.
- Splits are assigned before generating resize variants. Byte-identical source duplicates are rejected; near-duplicate photos with different JPEG encodings are not detected automatically.
- Tilt proxy = max(|horizontal tilt|, |vertical tilt|), computed from the CCPD filename.
- `preprocessing.py` returns BGR/grayscale by default, RGB with rgb3=True. OCR model author should agree on channel convention, input size, and mode.
- No model training or evaluation is performed by this notebook.
- Detection teammate: use `dataset_original.yaml` for the primary original-image experiment; `dataset.yaml` includes all three scales and changes the training distribution.
- OCR teammate: use `crops_gt/` and `ocr_train.csv`, `ocr_val.csv`, `ocr_test.csv` plus `preprocessing.py`.
- Pipeline/evaluation teammate: use `metadata.csv` to group by plate size and tilt; GT crops are for OCR-only evaluation.
