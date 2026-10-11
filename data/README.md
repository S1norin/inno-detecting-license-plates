# CCPD: Data and Preprocessing

The `dataset/` folder contains the completed dataset. You can start training
without running data preparation again. The notebook and source images are
included for reproducibility.

## Contents

- `dataset/` — prepared images, annotations, crops, metadata and figures.
- `Data_Preprocessing.ipynb` — reproducible data preparation notebook.
- `source_data/` — 5,000 unique source images and selection records.
- `preprocessing.py` — plate crop preprocessing for OCR.
- `requirements.txt` — dependency versions, including JupyterLab.
- `project_documents/` — project plan and course guidelines.
- `configure_paths.py` — updates YOLO paths after moving the folder.
- `Open_Notebook.cmd` — sets up a local environment and opens Jupyter on Windows.
- `select_dataset.py` — optionally extracts the same selection from the full archive.

## Getting started after extracting the ZIP

Use Python 3.13. Open a terminal in this folder and install dependencies:

```text
python -m pip install -r requirements.txt
python configure_paths.py
```

The second command updates the YOLO configurations and image lists for your
computer. Run it again whenever you move the folder. Keep the folder structure intact.

On Windows, `Open_Notebook.cmd` creates a local environment, installs dependencies
and opens the notebook. After setup, you can use that environment to update paths:

```text
.venv\Scripts\python.exe configure_paths.py
```

The full CCPD archive is not required to use or reproduce the included dataset.
YOLO and OCR model training code is provided separately by the corresponding teammates.

## License plate detection

Use `dataset/dataset_original.yaml` for the primary training and evaluation:
3,500 training images, 700 validation images and 800 test images, at the original scale.

`dataset/dataset.yaml` includes all three scales and is intended for explicitly
defined scale experiments. Its train/validation/test splits contain
10,500 / 2,100 / 2,400 variants.

Within `dataset/`, `images/` contains full photographs and `labels/` contains
normalized YOLO bounding boxes. Class `0` is `license_plate`.

## License plate recognition

Use `dataset/crops_gt/` with `dataset/ocr_train.csv`, `dataset/ocr_val.csv`
and `dataset/ocr_test.csv`. Each table records the crop path, target text,
scale, source image ID and size group. For an original-scale baseline,
filter the rows to `scale == 1.0`.

The independent preprocessing modes in `preprocessing.py` are:

- `none` — unchanged crop.
- `gray_contrast` — grayscale with CLAHE contrast enhancement.
- `canny` — edge detection with thresholds 100/200.

Canny can produce empty edge maps on low-contrast images. Tune preprocessing
parameters using training/validation data and keep them fixed for test evaluation.
Agree on input dimensions and channels with the OCR model author. The function
returns BGR/grayscale by default, or three-channel RGB with `rgb3=True`.

The crops use ground-truth bounding boxes and are suitable for evaluating OCR
separately. For end-to-end evaluation, use full test images and the detector's
predicted boxes.

## Evaluation and analysis

`dataset/metadata.csv` contains 15,000 variants; `dataset/original_metadata.csv`
contains 5,000 originals. The tables include split, parent ID, scale, dimensions,
tilt, plate text and bounding box coordinates.

`dataset/plots/` contains distribution comparisons, a bounding box/crop example
and preprocessing examples. `dataset/run_config.json` records preparation
parameters and environment versions. `dataset/data_checks.json` records the
completed checks of copied files, source checksums, labels and split consistency.

## Dataset decisions and limitations

- There are 5,000 unique source images with scales 1.0, 0.75 and 0.5, producing
  15,000 variants. Variants of one source are related observations.
- Splits are assigned before resizing. All variants of a source stay in the same split.
- Random seed: 42. This is a custom split, different from the official CCPD benchmark split.
- Byte-identical duplicates are excluded. Similar photographs with different file
  contents are not automatically excluded.
- Small/medium/large groups use plate height thresholds calculated from training
  originals only: small <= 83 px, medium > 83 and <= 112 px, large > 112 px.
  Each variant is also assigned a group using its resulting plate height.
- The counts in the project plan total 4,984. This dataset uses 3,500 / 700 / 800
  to total 5,000; document this decision in the team report.
- Eight duplicate photographs in the initial selection were replaced with unique
  images from the same size and angle groups. The replacement history is in
  `source_data/selection_config.json`.
- Data preparation checks do not measure detection or recognition accuracy.

## Reproducing the dataset

Open `Data_Preprocessing.ipynb` from this folder and select
**Restart Kernel → Run All**. If `dataset/` already contains files, the notebook
creates a separate `dataset_run_2/`, `dataset_run_3/`, etc., preserving previous results.

To optionally extract the exact source selection from a separately downloaded archive:

```text
python select_dataset.py --archive CCPD2019.tar.xz --selection-manifest source_data/selected_images.csv --output reproduced_source
```

Dataset source: [official CCPD repository](https://github.com/detectRecog/CCPD).
