# AGENTS.md

Guidance for AI coding agents working in this repository.

## Project

License plate detection and recognition on CCPD2019 (team course project).
Detection is single-class YOLO bounding boxes; recognition is OCR on
ground-truth plate crops with three preprocessing ablations.
Human-facing docs: `README.md`, `CONTRIBUTING.md`, `data/README.md`.

## OpenSpec

- Planning artifacts live in `openspec/`: durable capability specs in
  `openspec/specs/`, in-flight changes (proposal, spec deltas, design, tasks)
  in `openspec/changes/`.
- New features and experiments start as a change (`/openspec-propose`), are
  implemented from its tasks (`/openspec-apply-change`), and are archived
  after merge (`/openspec-archive-change`).
- Read the active change's artifacts before implementing, keep them in sync
  when decisions change, and validate with `openspec validate <change>`.

## Environment

- Python 3.12 in `.venv` (uv project; `.python-version` pins 3.12).
- `data/requirements.txt` pins the exact package versions used to build the dataset.
- Do not install packages or download files (datasets, archives, models) without
  asking the user first.

## Data layout and hard rules

- `data/source_data/images/` — 5,000 seed images tracked with Git LFS. Read-only.
  Never modify, delete, or re-encode them.
- `data/dataset/images/` and `data/dataset/crops_gt/` — gitignored and
  regenerable. Never add them to git or LFS.
- Regenerate with: `rm -rf data/dataset`, run `data/Data_Preprocessing.ipynb`
  from the `data/` folder (it refuses a non-empty output dir and picks
  `dataset_run_N/` otherwise), then `python data/configure_paths.py`.
- `data/dataset/dataset_original.yaml` and `data/dataset/lists_original/*.txt`
  contain absolute paths from the machine that built the dataset. Regenerate
  them with `configure_paths.py` before any YOLO use.
- Do not change the split assignment, seed (42), split ratios, or scale factors
  (1.0 / 0.75 / 0.5) without an explicit user request. The
  split-before-resize property (all variants of one photo stay in one split)
  is the anti-leakage invariant; preserve it in any pipeline change.
- Size bucket thresholds (small <= 83 px, medium <= 112 px) are derived from
  the train split only. Keep them train-derived.
- Prefer editing the notebook (source of truth for the pipeline) over editing
  generated `data/dataset/` artifacts by hand.

## Jupyter notebooks

- Read the full `.ipynb` before editing.
- Edit programmatically with `nbformat` (load, modify cells, save), not raw
  JSON text edits.

## Code style

- No decorative or aggressive delimiters in code or docs.
- Keep comments and docstrings clean and minimal.
- Separate logical blocks with blank lines: after imports, between
  validation/setup/processing/output groups, and between phases inside loops
  and functions. Ruff does not enforce this; keep it consistent manually.
- Prefer small, scoped changes.
- Lint with `uv run ruff check .` (config in `pyproject.toml`); keep it clean
  before committing. Notebook style rules are relaxed via per-file-ignores —
  readability edits (blank lines) are fine, but never change notebook
  behavior.

## Useful references

- `python data/configure_paths.py` — rewrite YOLO YAMLs and image lists for
  the current machine.
- `data/preprocessing.py` — `preprocess_plate(image, mode, rgb3=False)` with
  mode in `none` / `gray_contrast` / `canny`; BGR/grayscale out by default.
- `data/dataset/metadata.csv` — per-variant links (image, label, crop, source,
  split, parent_id, scale, bbox, tilt, plate text).
- `data/dataset/run_config.json` — parameters and exact environment of the
  last successful dataset build.

## Branches, commits, pull requests

- Do not push directly to `main`; work on a branch and open a PR.
- Branch names: `feat/`, `fix/`, `data/`, `docs/` + kebab-case description
  (e.g., `feat/yolo11-baseline`). Data pipeline changes use `data/`.
- Commit titles: imperative, `<type>: <summary>` under 72 chars, types
  `feat`/`fix`/`data`/`docs`/`chore`; scope model, data, and docs changes
  separately.
- PR descriptions must state how the change was verified; data pipeline
  changes require a full notebook rerun with Step 8 asserts passing.
- Full conventions: see `CONTRIBUTING.md`.

## When training or evaluating models

- Detection: `data/dataset/dataset_original.yaml` for the primary
  original-scale experiment; `dataset.yaml` only for explicit scale experiments.
- Recognition: `crops_gt/` + `ocr_{train,val,test}.csv`; GT crops isolate OCR
  from detector quality. End-to-end evaluation uses full test images with the
  detector's predicted boxes.
- Tune on train, select on val, report test once.
- Put model code and experiment notes in `baseline/detection/` or
  `baseline/ocr/`.
