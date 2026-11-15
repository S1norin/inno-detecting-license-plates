# PlateLab demo

Interactive research demo for the CCPD2019 license plate project. The
Streamlit app provides a dataset explorer, historical results and live CPU
inference with frozen and fine-tuned YOLO11s detectors and two OCR engines
(CRNN and the EasyOCR baseline). See `demo/DEMO_SCRIPT.md` for presentation
notes.

## Installation for a human

These instructions target Linux with Python 3.12, `uv` and Git LFS. The
commands download packages, source images and model weights. Check available
disk space and obtain permission before downloading. The prepared image and
crop directories are gitignored and are not included in a normal clone.

1. Clone the repository, enter it and check out the branch containing the demo:

   ```bash
   git clone <repository-url>
   cd inno-detecting-license-plates
   git checkout feat/streamlit-demo
   ```

2. Create a Python 3.12 environment and install the demo dependencies:

   ```bash
   uv venv --python 3.12 .venv
   uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cpu \
     'torch==2.6.0' 'torchvision==0.21.0'
   uv pip install --python .venv/bin/python \
     'streamlit==1.52.2' 'ultralytics==8.3.162' 'numpy==2.1.3' \
     'pandas==2.3.3' 'PyYAML==6.0.3' 'pytest==9.2.0' \
     'nbformat==5.11.1' 'nbclient==0.11.0' ipykernel matplotlib tqdm
   ```

   The demo uses OpenCV through Ultralytics' `opencv-python` dependency. The
   above is an installation recipe based on the existing runtime, not a tested
   lockfile or a guarantee of identical results on every machine. On other
   platforms select the appropriate PyTorch CPU wheels. `pyproject.toml` does
   not install the demo dependencies automatically.

3. Fetch the three checkpoints with the project's SHA-256-verifying fetcher:

   ```bash
   .venv/bin/python -m baseline.common.weights fetch yolo11s-license-plate
   .venv/bin/python -m baseline.common.weights fetch yolo11s-license-plate-finetuned
   .venv/bin/python -m baseline.common.weights fetch crnn-full-training
   .venv/bin/python -m baseline.common.weights verify yolo11s-license-plate
   .venv/bin/python -m baseline.common.weights verify yolo11s-license-plate-finetuned
   .venv/bin/python -m baseline.common.weights verify crnn-full-training
   ```

   Weights go under gitignored `baseline/weights/`. Do not substitute
   unverified checkpoints. The EasyOCR baseline additionally needs the
   pinned offline weights in `baseline/weights/easyocr/` (verified against
   `baseline/common/models.json`) and `easyocr==1.7.2` installed in the demo
   environment:

   ```bash
   uv pip install --python .venv/bin/python 'easyocr==1.7.2'
   ```

   If easyocr is missing, the app still runs and the Pipeline Inspector shows
   an install hint instead of EasyOCR results.

4. Prepare the data. If an existing complete `data/dataset/` already contains
   `images/`, `crops_gt/` and `metadata.csv`, preserve it and skip regeneration.
   Otherwise fetch the source JPEGs with Git LFS:

   ```bash
   git lfs install
   git lfs pull --include='data/source_data/images/**'
   ```

   Verify that a source file is a JPEG rather than an LFS pointer. Follow
   `data/README.md` and run `data/Data_Preprocessing.ipynb` from the `data/`
   directory using the project's prescribed dataset environment in
   `data/requirements.txt`. The notebook avoids overwriting a non-empty output
   directory and may create `dataset_run_N/` instead. Inspect and preserve
   existing tracked or generated data; never delete an existing dataset without
   explicit approval. The project instructions describe full regeneration and
   require the notebook's Step 8 assertions to pass.

5. If using the dataset with YOLO evaluation or training, refresh its
   machine-specific absolute paths:

   ```bash
   .venv/bin/python data/configure_paths.py
   ```

   This changes tracked generated files under `data/dataset/`. Inspect the diff
   and do not commit machine-specific paths. The Streamlit demo reads relative
   paths from metadata and does not require YOLO evaluation to be run.

6. Start locally and open the displayed URL:

   ```bash
   .venv/bin/python -m streamlit run demo/app.py \
     --server.address 127.0.0.1 --server.port 8501 --server.headless true
   ```

   `127.0.0.1` is accessible only on this machine. For private remote access,
   arrange a Tailscale-only binding or a private proxy after checking the local
   VPN, firewall and Tailscale configuration. Binding `0.0.0.0` exposes the
   port on all interfaces and is not Tailscale-only. This app has no separate
   application authentication. Never enable public exposure without approval.

7. Stop the local foreground server with Ctrl+C. Optional checks:

   ```bash
   .venv/bin/python -m pytest tests/test_demo.py -q
   .venv/bin/python demo/check_inference.py
   ```

   The inference check loads real models and uses CPU and memory. The Failure
   Analysis page automatically runs both detectors when opened or changed;
   Pipeline Inspector runs them when its button is pressed. Avoid these during
   training or other heavy work on a resource-constrained machine.

## Instructions for a coding agent

- Read repository `AGENTS.md`, `data/README.md` and this file before changing
  anything. Ask the operator before installing packages, downloading images or
  weights, or changing service/network settings.
- Do not modify source JPEGs or commit generated images, crops, weights,
  machine-specific dataset paths, secrets or environment files. Preserve any
  pre-existing user data and changes.
- Before starting the server on the machine described in this session, set and
  verify an independent timed stop for the service, as requested by the
  operator. A timer must target only the demo; do not change or restart Tailscale,
  VPN, firewall or routing. Ask before configuring persistent deployment.
- Check `systemctl --user is-active platelab` before starting a second server.
  A previously installed service may be enabled at boot; the repository does
  not install or manage it. A local-only server can be stopped with Ctrl+C.
- Verify the HTTP health endpoint and clearly distinguish that from a browser
  interaction test or live inference. Do not claim a clean installation was
  tested unless it was actually performed. Record any limitations.
- Keep canonical evaluation settings separate from UI thresholds. Use the
  project CRNN engine and `data/preprocessing.py`; do not silently download
  weights from model libraries.

## Code map

- `demo/app.py` — entry point, theme and navigation.
- `demo/dataset.py`, `demo/yolo.py`, `demo/crnn.py`, `demo/easy_ocr.py` — data
  and model adapters.
- `demo/weights_io.py`, `demo/viz.py` — weight checks and visualization.
- `demo/results.py` — historical metrics from `docs/md/final_results.md`.
- `demo/sections/` — the app sections.
