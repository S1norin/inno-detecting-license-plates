# inno-detecting-license-plates

Team computer-vision project: detecting and recognizing Chinese license plates, based on the CCPD2019 benchmark.

- Detection: single-class YOLO bounding boxes (`license_plate`)
- Recognition: OCR on ground-truth plate crops, with three preprocessing ablations (`none`, `gray_contrast`, `canny`)

## Layout

| Path | Contents |
|---|---|
| `data/` | Data pipeline: preparation notebook, preprocessing module, source images (LFS), prepared dataset |
| `data/dataset/` | Ready-to-train dataset: YOLO labels, metadata, OCR tables, dataset configs, plots |
| `data/source_data/` | 5,000 unique CCPD2019 source images (Git LFS) plus selection records and checksums |
| `baseline/detection/`, `baseline/ocr/` | Model code (in progress) |
| `docs/` | Project plan and course guidelines (PDF) |

Start with [`data/README.md`](data/README.md) for dataset format, split policy, limitations, and per-teammate usage notes.

## Getting the data

The 5,000 source images are stored in Git LFS. The large derived images
(`data/dataset/images/`, `data/dataset/crops_gt/`, ~1.6 GB) are gitignored and
regenerated locally from the source images.

1. Install the [git-lfs](https://git-lfs.com) client once: `git lfs install`
2. Clone normally; LFS files are pulled automatically.
3. Regenerate the dataset images and crops (~6 min, Python 3.13 per `.python-version`):

   ```bash
   rm -rf data/dataset
   python -m pip install -r data/requirements.txt
   ```

   Then open `data/Data_Preprocessing.ipynb` **from the `data/` folder** and run
   **Restart Kernel → Run All**. The notebook refuses to write into a non-empty
   `dataset/` folder, hence the `rm -rf` above; it recreates all tracked files in
   that folder (labels, metadata, configs, plots).

4. Refresh absolute paths after cloning or moving the repo (the committed YOLO
   configs and image lists point at the author's machine):

   ```bash
   python data/configure_paths.py
   ```

### Reproducing the source selection from scratch

Optional. The full CCPD2019 archive is not included in this repo. To re-extract
the exact 5,000-image selection, download the CCPD2019 archive from the
[official CCPD repository](https://github.com/detectRecog/CCPD) or the
[Kaggle mirror](https://www.kaggle.com/datasets/binh234/ccpd2019), then:

```bash
python data/select_dataset.py --archive /path/to/CCPD2019.tar.xz \
  --selection-manifest data/source_data/selected_images.csv \
  --output /path/to/new_subset
```

Reruns verify extracted images against the SHA-256 receipts in
`data/source_data/subset_complete.json` and never overwrite existing files.
