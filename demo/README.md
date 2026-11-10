# PlateLab demo

Interactive research demo for the CCPD2019 license plate project:
Streamlit app (`demo/app.py`) with live YOLO11s detection and CRNN
recognition on real checkpoints, plus a dataset explorer, experiment
review, failure analysis, and reproducibility view.

## Run

From the repository root:

```bash
.venv/bin/python -m streamlit run demo/app.py
```

## Contents

- `demo/app.py` — entry point, theme, navigation.
- `demo/config.py` — paths, pinned weight digests, canonical YOLO settings.
- `demo/dataset.py` — metadata access, sample loading, scale variants.
- `demo/yolo.py` — YOLO detector adapter (frozen + fine-tuned).
- `demo/crnn.py` — CRNN adapter over `models/crnn/engine.py` (canonical).
- `demo/weights_io.py` — SHA-256 weight verification.
- `demo/viz.py` — drawing helpers.
- `demo/results.py` — canonical metrics from `docs/md/final_results.md`.
- `demo/sections/` — page sections.
- `demo/check_inference.py` — vertical-slice inference check (CLI).

## Tests

```bash
.venv/bin/python -m pytest tests/test_demo.py -q
```
