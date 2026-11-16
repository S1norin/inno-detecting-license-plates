# inno-detecting-license-plates

Team computer-vision project: detecting and recognizing Chinese license plates, based on the CCPD2019 benchmark.

- Detection: single-class YOLO bounding boxes (`license_plate`)
- Recognition: OCR on ground-truth plate crops, with three preprocessing ablations (`none`, `gray_contrast`, `canny`)

The interactive demo is available at <https://toledo.sinorin.ru>.

## Layout

| Path | Contents |
|---|---|
| `data/` | Data pipeline: preparation notebook, preprocessing module, source images (LFS), prepared dataset |
| `data/dataset/` | Ready-to-train dataset: YOLO labels, metadata, OCR tables, dataset configs, plots |
| `data/source_data/` | 5,000 unique CCPD2019 source images (Git LFS) plus selection records and checksums |
| `baseline/` | Baselines and shared infrastructure: YOLO11s detection, EasyOCR recognition, combined evaluation, pinned environment and weights |
| `demo/` | Interactive Streamlit demo: dataset explorer, historical results, live inference |
| `openspec/` | Planning artifacts: capability specs in `specs/`, in-flight changes in `changes/`, completed ones in `changes/archive/` |
| `docs/` | Project plan and course guidelines (PDF), schedule, progress report |

Start with [`data/README.md`](data/README.md) for dataset format, split policy, limitations, and per-teammate usage notes.
For baseline results and reproduction: [`baseline/detection/README.md`](baseline/detection/README.md), [`baseline/ocr/README.md`](baseline/ocr/README.md), [`baseline/evaluation/README.md`](baseline/evaluation/README.md).

## Running the demo

The interactive demo is a Streamlit app in `demo/`: an Overview of the
research, a Dataset browser, live detection + recognition on the real
checkpoints (frozen and fine-tuned YOLO11s with CRNN or the EasyOCR
baseline), historical Experiments, and Failure Analysis. The steps below are
all that is needed on a fresh machine; [`demo/README.md`](demo/README.md) has
extra notes for edge cases.

1. Create the environment and install the demo dependencies (CPU PyTorch):

   ```bash
   uv venv --python 3.12 .venv
   uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cpu \
     'torch==2.6.0' 'torchvision==0.21.0'
   uv pip install --python .venv/bin/python \
     'streamlit==1.52.2' 'ultralytics==8.3.162' 'numpy==2.1.3' \
     'pandas==2.3.3' 'PyYAML==6.0.3' 'pytest==9.2.0' \
     'nbformat==5.11.1' 'nbclient==0.11.0' ipykernel matplotlib tqdm
   ```

2. Fetch and verify the model checkpoints (stored in gitignored
   `baseline/weights/`):

   ```bash
   .venv/bin/python -m baseline.common.weights fetch yolo11s-license-plate
   .venv/bin/python -m baseline.common.weights fetch yolo11s-license-plate-finetuned
   .venv/bin/python -m baseline.common.weights fetch crnn-full-training
   .venv/bin/python -m baseline.common.weights verify yolo11s-license-plate
   .venv/bin/python -m baseline.common.weights verify yolo11s-license-plate-finetuned
   .venv/bin/python -m baseline.common.weights verify crnn-full-training
   ```

3. Start the app from the repository root (requires a complete
   `data/dataset/`, see "Getting the data"):

   ```bash
   .venv/bin/python -m streamlit run demo/app.py \
     --server.address 127.0.0.1 --server.port 8501 --server.headless true
   ```

   Open http://127.0.0.1:8501 in a browser; stop the server with Ctrl+C.
   `127.0.0.1` is local-only — the app has no authentication, so do not bind
   `0.0.0.0` or expose it publicly.

Optional: the EasyOCR comparison engine needs
`uv pip install --python .venv/bin/python 'easyocr==1.7.2'` plus its pinned
offline weights in `baseline/weights/easyocr/` (see `demo/README.md`).
Without it the app runs and shows an install hint in the Detection and
Recognition section.

Note: do not use `uv run` for the demo — `pyproject.toml` declares no runtime
dependencies, so `uv` would sync an empty environment and wipe the demo
packages from `.venv`. Call `.venv/bin/python` directly.

## Getting the data

The 5,000 source images are stored in Git LFS. The large derived images
(`data/dataset/images/`, `data/dataset/crops_gt/`, ~1.6 GB) are gitignored and
regenerated locally from the source images.

1. Install the [git-lfs](https://git-lfs.com) client once: `git lfs install`
2. Clone normally; LFS files are pulled automatically.
3. Regenerate the dataset images and crops (~6 min, Python 3.12 per `.python-version`):

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
