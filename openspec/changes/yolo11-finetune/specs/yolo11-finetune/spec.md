# Spec Delta

## Purpose

Fine-tuned YOLO11s plate detector for the project's E1 comparison: trained from the pinned pretrained baseline on the original-scale train split, evaluated under the shared detection harness against the frozen baseline, and used for the end-to-end recognition re-run.

## ADDED Requirements

### Requirement: Fine-tuning from the pinned pretrained detector
The fine-tuned detector SHALL be trained starting from the pinned, digest-verified pretrained YOLO11s checkpoint, on the 3,500 original-scale (1.0) train-split images only. Scaled variants (0.75, 0.5) and any validation or test image SHALL NOT be used as training data.

#### Scenario: Pinned starting weights
- **WHEN** training is started
- **THEN** it loads the cached pretrained checkpoint whose content digest matches the pinned value, and refuses a checkpoint whose digest does not match

#### Scenario: Training data scope
- **WHEN** the training data is assembled
- **THEN** every training sample has scale 1.0 and belongs to the train split, and no validation or test image path appears in the training input

### Requirement: Validation-based checkpoint selection with early stopping
The training checkpoint SHALL be selected on the validation split by mAP@0.5, with early stopping after a fixed number of epochs without improvement. The test split SHALL be used only through the existing validate-then-test-once policy.

#### Scenario: Best validation checkpoint kept
- **WHEN** validation mAP@0.5 peaks at epoch k and no later epoch exceeds it
- **THEN** the saved final checkpoint is the one from epoch k

#### Scenario: Early stopping
- **WHEN** validation mAP@0.5 does not improve for the configured patience
- **THEN** training stops before the maximum epoch count

#### Scenario: Test untouched during selection
- **WHEN** checkpoint selection is performed
- **THEN** no test-split image is read, and test evaluation later runs exactly once

### Requirement: Traceable training record
A machine-readable training record SHALL be written alongside the final checkpoint capturing the starting-weight digest, the full training configuration (image size, maximum epochs, patience, batch size, learning rate, seed), per-epoch validation mAP@0.5, the selected epoch, the library versions, and the training machine.

#### Scenario: Record complete
- **WHEN** training finishes
- **THEN** the training record exists containing all fields above plus the content digest of the saved final checkpoint

### Requirement: Comparison against the frozen baseline
The final checkpoint SHALL be evaluated on the validation and test splits by the shared detection harness using the same inference settings as the frozen baseline, and SHALL be compared against the frozen baseline overall and per plate-size bucket and tilt bin using mAP@0.5, precision, recall, and F1.

#### Scenario: Identical inference settings
- **WHEN** the fine-tuned checkpoint is evaluated
- **THEN** inference uses the same image size, confidence floor, NMS IoU, seed, and precision as the frozen baseline evaluation, so the two result sets are directly comparable

#### Scenario: Independent validation cross-check
- **WHEN** validation is evaluated with the independent validator pass
- **THEN** the overall mAP@0.5 agrees with the harness computation within 0.01

#### Scenario: Comparison table
- **WHEN** the comparison report is generated
- **THEN** it contains per-split tables with pretrained and fine-tuned rows, the same metric columns, and the per-size and per-tilt breakdowns for both

### Requirement: End-to-end re-run with the better detector
The detector with the higher validation mAP@0.5 SHALL be used for the end-to-end recognition re-run with the CRNN recognizer: preprocessing mode and confidence threshold selected on validation, test reported once, and results compared against the previous end-to-end numbers produced by the frozen detector.

#### Scenario: Winner selected on validation
- **WHEN** the end-to-end re-run is prepared
- **THEN** the detector used is the one with the higher validation mAP@0.5, decided before test evaluation

#### Scenario: Single test pass
- **WHEN** the end-to-end re-run completes
- **THEN** test-split metrics exist for exactly one run and are reported alongside the previous end-to-end test numbers

### Requirement: Reproducibility scope
Training SHALL run with a fixed seed on a single machine with a pinned environment. Reproducibility SHALL be claimed only for identical machine and environment, and reported numbers SHALL use the same fixed precision as the rest of the project.

#### Scenario: Same-machine re-run
- **WHEN** training is repeated on the same machine with the same pinned environment and seed
- **THEN** the reported metrics match to the reported precision

#### Scenario: No cross-machine claim
- **WHEN** the report describes reproducibility
- **THEN** it states that reproducibility is expected for the same machine and environment and makes no cross-machine guarantee
