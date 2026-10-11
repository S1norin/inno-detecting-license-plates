# Contributing

Team project: license plate **detection** (YOLO) and **recognition** (OCR) on
CCPD2019. Dataset details live in [`data/README.md`](data/README.md); this file
covers setup and working agreements.

## Setup

1. Install the [git-lfs](https://git-lfs.com) client once: `git lfs install`
2. Clone the repo (LFS source images pull automatically).
3. Create the environment (Python 3.10 per `.python-version`; the dataset was
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

## Commits

- Work on a short-lived branch per model/experiment; merge to `main` when it
  works.
- Keep commits scoped: model code, data pipeline changes, and docs separately.
- After any data pipeline change, rerun the notebook end-to-end; its Step 8
  asserts are the dataset integrity checks.
