# Contributing

Team project: license plate **detection** (YOLO) and **recognition** (OCR) on
CCPD2019. Dataset details live in [`data/README.md`](data/README.md); this file
covers setup and working agreements.

## Setup

1. Install the [git-lfs](https://git-lfs.com) client once: `git lfs install`
2. Clone the repo (LFS source images pull automatically).
3. Create the environment (Python 3.12 per `.python-version`; the dataset was
   originally built on 3.13, the notebook runs on either):

   ```bash
   uv venv
   uv pip install -r data/requirements.txt
   ```

4. Regenerate the dataset images and crops (~6 min; they are gitignored):

   ```bash
   rm -rf data/dataset
   ```

   then open `data/Data_Preprocessing.ipynb` **from the `data/` folder** and run
   **Restart Kernel → Run All**. The notebook refuses to write into a non-empty
   `dataset/`, hence the `rm -rf`; it recreates all tracked files in that folder.

5. Refresh absolute paths (the committed YOLO configs point at another machine):

   ```bash
   python data/configure_paths.py
   ```

## Working areas

| Area | Where | Notes |
|---|---|---|
| Detection | `baseline/detection/` | Primary config: `data/dataset/dataset_original.yaml` (original scale). Scale experiments: `data/dataset/dataset.yaml`. |
| Recognition | `baseline/ocr/` | Input: `data/dataset/crops_gt/` + `ocr_{train,val,test}.csv` + `data/dataset/preprocessing.py`. |
| Data pipeline | `data/` | Changes require rerunning the notebook; keep `data/README.md` in sync. |

## Data rules

- Never commit anything under `data/dataset/images/` or `data/dataset/crops_gt/`
  (gitignored, regenerable).
- Git LFS is reserved for `data/source_data/images/**`. Do not add other large
  binaries to LFS without team agreement (GitHub free tier: 1 GB LFS total).
- Treat `data/source_data/` as read-only input. Never modify or delete it.
- After cloning or moving the repo, run `python data/configure_paths.py` before
  any YOLO training/evaluation.

## Evaluation agreements

- Use the project's splits as-is (3,500 / 700 / 800 originals, seed 42). Do not
  resample or switch to the official CCPD benchmark split; results must stay
  comparable across teammates.
- Tune on train, select on validation. The test split is for final numbers only.
- For an original-scale OCR baseline, filter the OCR CSVs to `scale == 1.0`.
- Put model code, configs, and short experiment notes in your `baseline/`
  subfolder, not in `data/`.

## Code style

- Separate logical blocks with blank lines: after imports, between
  validation/setup/processing/output groups, and between phases inside loops
  and functions. Ruff does not enforce this, so keep it consistent manually.
- Keep comments and docstrings minimal; no decorative delimiters.

## Code quality

Ruff is the project linter (config in `pyproject.toml`):

```bash
uv sync             # installs dev tools (ruff) into .venv
uv run ruff check .
```

Run this before committing. The pipeline notebook is linted with relaxed style
rules (per-file-ignores) because it is a verified artifact in idiomatic
notebook style; keep bug detectors (F/B rules) clean everywhere.

## Branches, commits, and pull requests

`main` is always in a working state; do not push to it directly.

### Branches

One short-lived branch per model, experiment, or fix, created from `main`.
Name them `<type>/<kebab-case-description>`:

| Prefix | Use for |
|---|---|
| `feat/` | New models, features, evaluation scripts |
| `fix/` | Bug fixes |
| `data/` | Data pipeline changes (notebook, preprocessing, selection) |
| `docs/` | Documentation only |

Examples: `feat/yolo11-baseline`, `fix/crop-path-case`, `data/clahe-params`.
Delete the branch after merging.

### Commits

- Title line: imperative mood, `<type>: <summary>`, under 72 characters.
  Types: `feat`, `fix`, `data`, `docs`, `chore`.
- Body (optional): what changed and why, one point per line.
- Keep commits scoped: model code, data pipeline changes, and docs in
  separate commits/PRs.

Examples:

```
feat: add YOLO11n baseline detector with training script
fix: clamp scaled bboxes to image bounds in notebook export
data: switch OCR baseline to original scale only
docs: document size bucket thresholds in data README
```

### Pull requests

- One focused change per PR; small is better.
- Title follows the commit convention (`<type>: <summary>`).
- Description covers:
  - **What & why** — the change and its purpose.
  - **How it was verified** — commands run, notebook rerun, checks passed.
  - For data pipeline changes: state that `Data_Preprocessing.ipynb` was
    rerun end-to-end and its Step 8 integrity asserts passed.
  - For model changes: report validation metrics (test metrics only for
    final results), with the config used.
- Rebase on `main` before requesting review; merge only when the branch is in
  a working state.
