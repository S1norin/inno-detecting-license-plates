# Tasks

## 1. Module setup and file moves

- [x] 1.1 Create the `models/yolo11_finetune/` package (`__init__.py`, gitignored `results/`) and verify `uv run ruff check .` stays clean with the new package present
- [x] 1.2 Move `train_ready.py` and `train_config.yaml` from `baseline/detection/` to `models/yolo11_finetune/`, update the config's output directory to `models/yolo11_finetune/results` and add the `patience` setting, and verify `./.venv-baseline/bin/python -m models.yolo11_finetune.train_ready` passes against the cached weights and train split
- [x] 1.3 Update `baseline/detection/README.md` so the fine-tuning-readiness section points at `models/yolo11_finetune/` and verify the retired baseline readiness command no longer appears in any doc
- [x] 1.4 Add `models/yolo11_finetune/requirements-gpu.txt` pinning the GPU training environment (ultralytics 8.3.162, Torch 2.6.0 CUDA wheels, torchvision, numpy, OpenCV, opencv-python-headless) and verify every non-Torch pin matches the baseline requirements versions

## 2. Training code

- [ ] 2.1 Implement `models/yolo11_finetune/train.py`: config-driven Ultralytics `train()` wrapper that starts from the digest-verified pretrained weights, trains on the original-scale train split, uses val-mAP checkpoint selection with early stopping, and writes a machine-readable training record (start-weight digest, full config, per-epoch val mAP@0.5, selected epoch, library versions, machine) next to the final checkpoint
- [ ] 2.2 Add unittest tests for `train.py` covering config validation (positive integers, patience), rejection of a start-weight digest mismatch, and the completeness of the training-record fields, and verify `./.venv-baseline/bin/python -m unittest discover -s models/yolo11_finetune/tests -v` passes
- [ ] 2.3 Verify the training entry point refuses to run on a machine without a GPU and `uv run ruff check .` passes

## 3. Remote training run (GPU machine)

- [ ] 3.1 Transfer the dataset (1.8GB zip), the pinned weights, and the code to the GPU machine; set up the pinned GPU environment, run `data/configure_paths.py`, and verify `python -m models.yolo11_finetune.train_ready` reports the expected 3,500 train images
- [ ] 3.2 Run a 2-epoch smoke training and verify it produces `best.pt` and a training record with 2 per-epoch entries and the expected digests
- [ ] 3.3 Run the full training (max 100 epochs, patience 20) and verify the training record shows all completed epochs, a selected epoch, and a final-checkpoint digest
- [ ] 3.4 Return the final checkpoint and training record to `models/yolo11_finetune/results/` and verify the SHA-256 of the returned checkpoint matches the digest in the training record

## 4. Evaluation and comparison

- [ ] 4.1 Add a `--model` path option to `baseline/detection/evaluate.py` (default unchanged: the pinned pretrained model; a supplied file's SHA-256 is recorded in the metrics JSON in place of the manifest check) and verify all existing `baseline/tests` pass and a new test covers digest recording for a supplied model
- [ ] 4.2 Evaluate the fine-tuned checkpoint on validation with `--validate` and on test, and verify the metrics JSONs exist, the independent cross-check agrees within 0.01, and both runs use the identical inference settings as the frozen baseline
- [ ] 4.3 Build the pretrained-vs-fine-tuned comparison (overall and per size bucket and tilt bin, mAP@0.5/precision/recall/F1) and verify the numbers match the two metrics JSONs

## 5. End-to-end re-run with the better detector

- [ ] 5.1 Select the detector with the higher validation mAP@0.5 and verify its validation and test prediction JSONs exist for the end-to-end protocol
- [ ] 5.2 Re-run the CRNN end-to-end protocol on validation for all three preprocessing modes via `baseline/ocr/run.py` and verify the mode/tau selection is made from validation results only
- [ ] 5.3 Run the end-to-end protocol on test exactly once with the selected mode/tau and verify the test metrics JSON exists
- [ ] 5.4 Compare the new end-to-end numbers against the previous end-to-end results produced by the frozen detector and verify both sets are reported side by side

## 6. Report and closeout

- [ ] 6.1 Write `models/yolo11_finetune/REPORT.md` covering the training (config, timing, selected epoch, digests), the detector comparison, the end-to-end re-run, and the same-machine reproducibility scope, and verify every reported number traces to a metrics or training-record file
- [ ] 6.2 Run the full verification pass: `uv run ruff check .`, the baseline unittest suite, and the CRNN unittest suite, and verify all pass
- [ ] 6.3 Validate the change with `openspec validate yolo11-finetune` and write the PR description stating how the change was verified, per `CONTRIBUTING.md`
