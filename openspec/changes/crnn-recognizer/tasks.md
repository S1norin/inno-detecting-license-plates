# Tasks

## 1. Model and training data

- [ ] 1.1 Create branch `feat/crnn-recognizer` from `main` and verify `git branch --show-current` prints it
- [ ] 1.2 Add the top-level `models/` package and `models/crnn/` subpackage (`models/__init__.py`, `models/crnn/__init__.py`, nested `.gitignore` ignoring `results/`) with `model.py` (architecture per design D2, grayscale H48/W256 aspect-preserving right-pad input transform, checkpoint save/load carrying charset digest and config) and verify unit tests pass: forward on a random batch yields `(B, 16, 59)` logits, parameter count is ~1.1M, input transform on a synthetic crop yields `(1, 48, 256)`, checkpoint round-trips
- [ ] 1.3 Add training-data assembly to `models/crnn/train.py` (load scale-1.0 train samples, refuse to start unless the charset equals `build_charset(ocr_train.csv)`, build CTC targets and lengths) and verify a test on the real dataset confirms 3,500 samples, all at scale 1.0, all targets inside the charset

## 2. Training loop

- [ ] 2.1 Implement the deterministic CPU training loop in `train.py` (seed 42, 1 thread, deterministic mode, Adam lr 1e-3, batch 32, per-epoch greedy-decoded validation character accuracy, best-epoch selection with ties to the earlier epoch, checkpoint + `training_record.json` output) and verify a short 2-epoch run writes a record containing config, seed, thread count, library versions, dataset/charset digests, per-epoch validation metrics, selected epoch, and the checkpoint digest
- [ ] 2.2 Run a 1-epoch timing probe and fix the final epoch count, recorded in the training record and design follow-through (report), verifying the probe duration is noted in `models/crnn/REPORT.md`
- [ ] 2.3 Verify the determinism requirement: run the 2-epoch training twice from the same data and config and confirm both `checkpoint.pt` files have identical sha256 digests

## 3. Engine and runner integration

- [ ] 3.1 Implement `CRNNEngine` in `models/crnn/engine.py` (digest-verified checkpoint load, `identity`, `weight_digests`, `recognize` via softmax + the existing `decode_greedy`, 1-thread deterministic inference) and verify unit tests pass: construction rejects a mismatched sha256, `recognize` on real validation crops returns text with one finite confidence per emitted character, and `identity` carries the required fields
- [ ] 3.2 Add `--engine {easyocr,crnn}` (default `easyocr`) plus `--crnn-checkpoint` and `--crnn-expected-sha256` to `baseline/ocr/run.py` and verify the existing test suite still passes for the EasyOCR default while a CRNN inference pass over a few samples produces a cache that passes `validate_raw`
- [ ] 3.3 Update `baseline/ocr/README.md` with CRNN training and evaluation usage and verify the documented commands match the implemented CLIs
- [ ] 3.4 Update the model-code location line in `AGENTS.md` to point non-baseline model code to `models/` and verify the file names both `baseline/` and `models/` correctly

## 4. Full evaluation run and report

- [ ] 4.1 Run the full training with the fixed epoch count and verify `training_record.json` is complete and the best-epoch validation character accuracy is recorded
- [ ] 4.2 Run the six validation inference passes (ocr_only and end_to-end, each in none/gray_contrast/canny, end-to-end with the frozen YOLO predictions) and verify all six caches pass `validate_raw` with 700 samples each
- [ ] 4.3 Run `evaluate select` over the six validation caches and verify `selection.json` contains all three modes per protocol with the selected mode and tau
- [ ] 4.4 Run test inference once per protocol under the saved selection, then the `evaluate val/test` reports and the selection plot, and verify the test-once reservation exists and the reports include per-size-bucket and per-tilt-bin metrics
- [ ] 4.5 Write `models/crnn/REPORT.md` (config, timing probe, training curve summary, per-protocol metrics, CRNN-vs-EasyOCR comparison table using the existing baseline test numbers) and add the CRNN rows to `docs/progress_report.md` and `docs/progress_report.tex`, verifying every table value matches the saved report JSONs

## Workflow follow-up

- Archive the change after the 20 Oct second-approach milestone review is satisfied.
- Verify the archived result.
