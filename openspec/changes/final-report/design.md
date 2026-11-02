# Design

## Context

See proposal.md for motivation. State of the repository on `main`:

- All experimental work is merged. Five committed reports carry the
  numbers: `baseline/detection/REPORT.md`, `baseline/ocr/REPORT.md`,
  `baseline/evaluation/REPORT.md`, `models/crnn/REPORT.md`,
  `models/yolo11_finetune/REPORT.md`.
- This working copy (Apple M3, CPU-only) has the dataset, the frozen
  YOLO weights plus its val/test predictions and metrics, and one local
  CRNN checkpoint (`540cba37`, epoch 43) with a full evaluation
  (`models/crnn/results/evaluation/`).
- Missing locally: the fine-tuned detector weights and its evaluation
  artifacts, the canonical CRNN checkpoint (`92d1fbea`, epoch 39, pinned
  in `baseline/common/models.json`), the CRNN end-to-end caches built on
  fine-tuned predictions, all EasyOCR raw caches, and any consolidated
  final-results document.
- Both missing checkpoints are in the HuggingFace repo
  `Sinorin/inno-detecting-license-plates-weights`
  (`detection/yolo11_finetune/best.pt`, `ocr/crnn/checkpoint.pt`) with
  LFS OIDs equal to the digests recorded in the training reports.
- The shared harnesses already support everything needed:
  `baseline/detection/evaluate.py --model <path>`,
  `baseline/ocr/run.py --engine crnn --protocol {ocr_only,end_to_end}`,
  selection via `baseline/ocr/evaluate.py`, and the
  `baseline/evaluation` consolidation + timing probe (probe accepts
  `--detector-weights` but hardcodes the EasyOCR engine).

## Goals / Non-Goals

**Goals:**

- One consistent number set for the final report: every CRNN metric from
  the canonical `92d1fbea` checkpoint, every detector metric from the two
  pinned weight files, all produced by the existing harnesses.
- Fresh, provenance-complete artifacts (metrics/predictions/raw caches)
  in this working copy for the fine-tuned detector and all CRNN
  protocols, so the report tables trace to JSON files.
- Inference timing for frozen YOLO, fine-tuned YOLO, and CRNN on the
  same probe protocol; training times traced to the fetched training
  records.
- At least three representative failure examples per stage with likely
  causes, plus the comparison figures, consolidated in a `docs/`
  final-results document.

**Non-Goals:**

- No retraining of any model, no dataset changes, no EasyOCR re-run
  (frozen numbers stay as committed), no new recognition or detection
  architectures, no changes to the frozen EasyOCR protocol.
- No modification of the tracked per-change REPORT.md files; they remain
  historical records. The new `docs/` document is authoritative for the
  final report.

## Decisions

**D1 — Canonical CRNN checkpoint is `92d1fbea`.**
All regenerated CRNN evaluations use the manifest-pinned checkpoint
(epoch 39, val character accuracy 0.7564). It is the checkpoint the final
end-to-end result was produced with and the one a clean clone fetches, so
the reported system is exactly the reproducible system.
*Alternative considered:* keep both checkpoints' numbers and label them.
Rejected: the final report would then cite ocr-only and end-to-end
numbers from different models, which is exactly the inconsistency this
change removes. The old local checkpoint stays on disk, untouched.

**D2 — Fine-tuned weights enter through the manifest, not by hand copy.**
Add one entry to `baseline/common/models.json`:
name `yolo11s-license-plate-finetuned`, URL
`https://huggingface.co/Sinorin/inno-detecting-license-plates-weights/resolve/main/detection/yolo11_finetune/best.pt`,
filename `best.pt`, cache subdir `yolo11_finetune`, SHA-256
`0a4f080fe3749b8be145101468a0dd98455cf50117d2acd37575f33e0d261055`.
The downloader verifies the digest before use, so a moving `main` branch
fails closed. *Alternative considered:* `resolve/<commit>/...` like the
CRNN entry; the repo's commit API is not reachable, and the SHA pin
provides the same integrity guarantee, so `main` is used.
The two small `training_record.json` files are fetched alongside
(plain HTTPS GET into the same cache subdirs) so runtime numbers trace
to artifacts.

**D3 — Regenerate with the existing harness entry points, in dependency
order.**

1. Fetch weights (D2).
2. Fine-tuned detector: `evaluate.py --model … --validate` on val, then
   the one-shot test run. Outputs under
   `baseline/detection/results/finetuned-val|test/`, comparison JSONs
   under `models/yolo11_finetune/results/finetune/` (same layout as the
   original change).
3. CRNN validation passes with `92d1fbea`: `ocr_only` x3 modes,
   `end_to_end` x3 modes on frozen predictions, `end_to_end` x3 modes on
   fine-tuned predictions (9 runs, 700 crops each).
4. Selection per protocol from validation only (3 selections).
5. One test run per protocol under its selected policy (3 runs, 800
   crops each). A new checkpoint is a new policy, so fresh one-shot test
   runs are legitimate under the test-once rule — the same pattern the
   yolo11-finetune change used in group 5.
6. Timing probes (D4), failure extraction (D6), document and figures
   (D5).

Every step writes only to gitignored `results/` directories except the
final `docs/` artifacts.

**D4 — Extend the timing probe with a CRNN engine option.**
Add `--probe-engine {easyocr,crnn}` (default `easyocr`) and
`--crnn-checkpoint` to `baseline/evaluation/run.py`; `run_probe` in
`timing.py` builds the corresponding engine. EasyOCR stays the default so
the frozen baseline probe remains reproducible unchanged. Three probe
runs: frozen YOLO + EasyOCR (re-confirm the committed numbers on this
machine is optional — default is to reuse the committed probe and only
probe the new stages), fine-tuned YOLO + CRNN, and CRNN per mode.
*Alternative considered:* a standalone timing script. Rejected: the probe
already implements seeded sampling, warmup exclusion, median/p95, and
peak RSS per the baseline-evaluation spec; reusing it keeps one protocol.

**D5 — Consolidated output: `docs/md/final_results.md` + `docs/figures/`.**
The document contains, in the order the two-page summary needs them:
dataset and split summary; detection comparison (frozen vs fine-tuned,
overall + size + tilt); recognition comparison (EasyOCR ocr-only /
end-to-end, CRNN ocr-only / end-to-end frozen / end-to-end fine-tuned,
overall + size + tilt); preprocessing ablation table; runtime table
(training and inference, machine-labelled); failure analysis (per stage,
with image paths, GT, prediction, likely cause); and a reproducibility
section (weights digests, commands, environment). Figures:
(1) detection comparison bars, (2) recognition comparison (character
accuracy and F1 across the five configurations), (3) preprocessing
ablation, (4) size/tilt breakdown, (5) failure-case montage. Failure
images live under the gitignored `data/dataset/`; the document references
relative paths that resolve on any machine with the dataset built.

**D6 — Deterministic failure-case selection.**
Detection: misses (no predicted box at matching IoU) ranked by
best-score, plus one multi-detection case per detector. Recognition:
accepted reads with the largest normalized edit distance, plus one large-
plate case and one 10-19-degree tilt case (the two known-weak scopes).
Selection is a pure function of the regenerated JSON caches; selected
sample IDs are recorded in the document so the montage is reproducible.

## Risks / Trade-offs

- [Regenerated CRNN numbers differ slightly from `models/crnn/REPORT.md`
  (different checkpoint)] → Expected and intended; the new `docs/`
  document is authoritative, the old report stays as a historical record
  of the `540cba37` run.
- [Inference timings measured on Apple M3 differ from the committed
  Ryzen probe] → Every timing row is machine-labelled; the EasyOCR frozen
  probe number is cited as committed, not re-measured.
- [HuggingFace `main` moves and the digest check fails] → The downloader
  fails closed with the expected/actual digests; re-pin the URL commit
  then. No silent use of wrong weights.
- [Probe extension touches the baseline-evaluation package] → Minimal,
  additive flags with EasyOCR defaults; existing tests must stay green
  and a new test covers the CRNN path with a synthetic checkpoint.
- [Three one-shot test runs consume the test split again] → Legitimate
  under the test-once rule (new policy bindings), same pattern as the
  yolo11-finetune change; no policy is chosen after observing test
  output.
- [Download size / time] → ~31 MB of weights plus minutes of CPU
  inference (detection ~1,600 images, CRNN ~2,400 crops, probes ~300
  images); well within a single working session.

## Migration Plan

Linear, no deployment: fetch weights → run evaluations → probe →
document. Rollback is trivial: all regenerated artifacts are
gitignored, and the only tracked changes are one manifest entry, two
probe files, and the new `docs/` files — revertable by commit. Nothing
in `data/` is touched.

## Open Questions

None blocking. The exact figure set (4-5 figures) can be trimmed when
writing the document without affecting any other artifact.
