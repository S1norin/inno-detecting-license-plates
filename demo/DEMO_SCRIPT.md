# PlateLab — presentation guide

Suggested order and talking points for the 6 sections. Everything shown is
real: live inference on the real checkpoints, real dataset samples, and the
canonical saved results from `docs/md/final_results.md`.

## 1. Overview (30 s)

- One task, two stages: detect the plate, then read it.
- Three research questions: does fine-tuning help? does a plate-specific
  recognizer beat general OCR? where does the system fail?
- Point at the four metric cards and the stage table.

## 2. Dataset Explorer (1–2 min)

- Browse a real test image: green box = ground truth, the metadata row
  (bucket, tilt, text) comes from `metadata.csv`.
- Show the three scale variants of the same photo — and emphasize: all three
  stay in the same split (split before resize, anti-leakage invariant).
- Walk through the 5-step preparation (selection → split → variants →
  annotations → validation).
- Preprocessing ablation on the selected crop: three real outputs from
  `data/preprocessing.py`; Canny visibly destroys the glyphs, and the
  validation numbers agree (0.7558 / 0.7469 / 0.4575).

## 3. Pipeline Inspector (2–3 min) — the centerpiece

- Pick a sample (default: the documented failure `6f3c772c`), press
  **Run inference**.
- Show both detectors side by side: on a failure image the frozen model is
  empty while the fine-tuned one finds the plate.
- The crop the CRNN sees is the detector's own top-1 box (not the GT crop) —
  that is what makes it end-to-end.
- Try a few normal images; try the upload box with any street photo
  (no ground truth, purely a live demo).
- Mention the live CPU runtime printed at the bottom.

## 4. Experiments (1–2 min)

- Detection table: recall 0.8938 → 0.9975 is the whole story of
  fine-tuning.
- Recognition table: CRNN 0.7593 (OCR-only) vs EasyOCR 0.3474; end-to-end
  with the fine-tuned detector 0.7661 / 296 exact / F1 0.7372.
- Preprocessing bar: `none` wins on validation, test was run once after
  selection.
- Scale robustness: flat curves — the CRNN resizes to 48×256, so absolute
  scale is normalized away.
- Size/tilt sensitivity: large plates and the 10–19° bin are the weak spots.
- Runtime table: historical Apple M3 numbers, clearly labeled.

## 5. Failure Analysis (1–2 min)

- Select each case; the app re-runs the real models and prints a live check
  against the documented observation (all six reproduce).
- Emphasize the cross-cutting pattern: detection and recognition fail in the
  same region (large + tilted plates) — one data gap, two symptoms.

## 6. Team & Reproducibility (30 s)

- Four roles, one page.
- Weights: SHA-256 pinned, re-verified on every app start; digests shown
  live.
- Reproducibility: repo, canonical results file, evaluation commands in
  `final_results.md`, dataset invariants.

## Honest disclaimers to keep in mind

- EasyOCR numbers are a frozen baseline from an earlier environment
  (not a comparable rerun).
- Historical runtimes are from an Apple M3; live runtimes on this server
  are shown separately and labeled.
- The dataset is a custom 5,000-photo selection of CCPD2019, not the
  official benchmark split.
- This is a research demo, not a production ALPR system.
