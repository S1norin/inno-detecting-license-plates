# Proposal

## Why

The Nov 3 results checkpoint and Nov 10 final submission need one fresh,
consistent set of final results: metrics for every tested approach,
comparison tables and figures, runtime measurements, and a failure analysis.
The committed per-change reports are complete, but two gaps remain. First,
the final system's artifacts (fine-tuned detector evaluation, CRNN
end-to-end with the fine-tuned detector, and their timings) exist only on
the GPU machine, not in this working copy. Second, the CRNN numbers in
`models/crnn/REPORT.md` come from a local 50-epoch run (epoch 43,
`540cba37`) while the final end-to-end numbers use the canonical shared
checkpoint (epoch 39, `92d1fbea`, pinned in `baseline/common/models.json`)
— the final report must not mix two checkpoints.

## What Changes

- Add a `yolo11s-license-plate-finetuned` entry to
  `baseline/common/models.json` (HuggingFace URL, SHA-256 `0a4f080f…`,
  matching the fine-tuning training record) and fetch it through the
  existing digest-verifying downloader.
- Fetch the canonical CRNN checkpoint (`92d1fbea`) and both training
  records (fine-tuned detector, CRNN) from the same HuggingFace repo, so
  runtime numbers trace to artifacts, not just report prose.
- Regenerate the fine-tuned detector evaluation (validation with the
  Ultralytics cross-check, then the one-shot test run) via
  `baseline/detection/evaluate.py --model`.
- Regenerate every CRNN evaluation with the canonical checkpoint:
  ocr-only and end-to-end (frozen and fine-tuned detector predictions) in
  none/gray_contrast/canny on validation, plus one test run per protocol
  under the validation-selected policy.
- Extend the `baseline/evaluation` timing probe so it can measure CRNN
  inference (today only EasyOCR) alongside the fine-tuned detector
  weights; run probes for frozen YOLO, fine-tuned YOLO, and CRNN.
- Extract failure cases (at least three per stage: detection misses for
  both detectors, CRNN recognition failures) from the regenerated
  prediction and raw caches, with the supporting images.
- Produce a consolidated final-results document under `docs/` — metrics
  tables for all approaches, comparison figures, a runtime table, and the
  failure analysis — as the source for the two-page technical summary.
- The EasyOCR baseline stays frozen: its committed numbers are reused
  verbatim; no re-run.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

None. The manifest entry and the probe option stay inside the existing
"hash-verified weight acquisition" and "sampled timing probe"
requirements; the rest is data regeneration and documentation. This
change sets `skip_specs: true`.

## Impact

- Tracked: `baseline/common/models.json` (one entry),
  `baseline/evaluation/run.py` + `timing.py` (probe engine option), a new
  `docs/` final-results document and its figures, README touches where
  the fine-tuned weights become fetchable.
- Gitignored: `baseline/weights/yolo11_finetune/best.pt`,
  `baseline/weights/crnn/checkpoint.pt`, both training records, and all
  regenerated `results/` artifacts.
- No dataset changes, no retraining, no test-based selection; the frozen
  EasyOCR numbers are untouched.
