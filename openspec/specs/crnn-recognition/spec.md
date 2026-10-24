# crnn-recognition Specification

## Purpose

Plate-specific CRNN/CTC recognizer as the project's second recognition approach: deterministically trained on original-scale training crops and evaluated under the shared recognition protocols for comparison against the EasyOCR baseline.

## Requirements

### Requirement: Deterministic CPU training
The recognizer SHALL be trained on CPU only with a fixed seed, single thread, and deterministic algorithms, so that a repeated training run from the same data and configuration produces a bit-identical checkpoint.

#### Scenario: Repeated training
- **WHEN** training is run twice with the same data, charset, and configuration
- **THEN** both runs produce checkpoints with identical content digests

#### Scenario: No device selection
- **WHEN** the training entry point is invoked
- **THEN** it runs on the CPU and offers no device selection

### Requirement: Original-scale training data
Training SHALL use only the original-scale (1.0) training crops with the training-split charset; scaled variants (0.75, 0.5) SHALL NOT be used as training data and remain reserved for later scale experiments.

#### Scenario: Scale scope
- **WHEN** the training data is assembled
- **THEN** every training sample has scale 1.0 and its source image belongs to the train split

### Requirement: Validation-based epoch selection
The training checkpoint SHALL be selected on the validation split by character accuracy, with ties resolved to the earlier epoch; the test split SHALL be used only through the existing validate-then-test-once policy.

#### Scenario: Best epoch kept
- **WHEN** validation character accuracy peaks at epoch k and no later epoch exceeds it
- **THEN** the saved checkpoint is the one from epoch k

#### Scenario: Test untouched during selection
- **WHEN** epoch selection is performed
- **THEN** no test-split sample is read, and test evaluation later runs once under the validation-selected threshold

### Requirement: Shared engine contract
The recognizer SHALL expose an engine compatible with the shared recognition harness: a static identity description, a checkpoint content digest, and a recognition function that returns the plate text with one confidence value per emitted character.

#### Scenario: Inference through the runner
- **WHEN** the OCR inference runner is asked to use the CRNN engine
- **THEN** every sample in the split produces a standard raw-result record with text and per-character confidences

#### Scenario: Checkpoint pinned
- **WHEN** the CRNN engine is constructed
- **THEN** it records the checkpoint's content digest in its weight digests and refuses a checkpoint whose digest does not match the expected one

### Requirement: Evaluation under both protocols
The recognizer SHALL be evaluated through the unmodified shared harness under both the OCR-only and end-to-end protocols, for all three preprocessing modes on validation, with metrics reported per protocol, size bucket, and tilt bin.

#### Scenario: Both protocols
- **WHEN** the CRNN evaluation is completed
- **THEN** character accuracy, plate exact-match rate, and edit-distance P/R/F1 exist for the OCR-only and end-to-end protocols on validation and test

#### Scenario: Mode ablation on validation
- **WHEN** validation inference is complete
- **THEN** results exist for all three preprocessing modes for each protocol

### Requirement: Traceable training record
A training record SHALL capture the full configuration (architecture sizes, input geometry, optimizer settings, epoch count, batch size), the seed, thread count, library versions, dataset and charset digests, per-epoch validation metrics, and the selected epoch, stored alongside the checkpoint.

#### Scenario: Record complete
- **WHEN** training finishes
- **THEN** a machine-readable training record exists containing all fields above plus the content digest of the saved checkpoint

### Requirement: Baseline comparison
The recognition report SHALL present CRNN metrics alongside the EasyOCR baseline metrics (character accuracy, exact match rate, P/R/F1) for both protocols on the test split, so the second-approach comparison is directly readable.

#### Scenario: Comparison table
- **WHEN** the report is generated after test evaluation
- **THEN** it contains a per-protocol table with CRNN and EasyOCR rows and the same metric columns
