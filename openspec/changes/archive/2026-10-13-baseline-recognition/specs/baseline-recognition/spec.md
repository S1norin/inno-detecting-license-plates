# Spec Delta

## Purpose

Recognition baseline for the license plate project: reads plate text from
crops under both the OCR-only (ground-truth crops) and end-to-end (detector
crops) protocols, applies a confidence-based read/abstain policy, and
produces reproducible per-protocol metrics.

## ADDED Requirements

### Requirement: OCR-only protocol
The baseline SHALL recognize plate text directly from ground-truth crops listed in `ocr_{train,val,test}.csv`, so detection errors do not influence recognition results.

#### Scenario: Split recognized
- **WHEN** the OCR-only protocol is run on a split
- **THEN** every crop listed for that split produces a recognized plate string with per-character confidences

### Requirement: End-to-end protocol
The baseline SHALL recognize plate text end-to-end: run the baseline detector on full images, crop from the predicted boxes, and OCR the crops.

#### Scenario: End-to-end run
- **WHEN** the end-to-end protocol is run on a split
- **THEN** each image is processed by the detector, its best predicted box is cropped, and the crop is OCR'd

#### Scenario: No detection
- **WHEN** the detector produces no box for an image
- **THEN** that image is recorded as unread (abstained) and the run continues without failing

### Requirement: Preprocessing ablation
The baseline SHALL support the preprocessing modes none, gray+contrast, and Canny, applied to crops immediately before OCR, with results reported per mode.

#### Scenario: Mode applied
- **WHEN** the baseline runs with the Canny mode
- **THEN** every crop is Canny-preprocessed before being passed to the OCR engine

#### Scenario: Ablation complete
- **WHEN** the ablation is run for a split
- **THEN** recognition results exist for all three modes

### Requirement: Per-character confidence and plate score
Each recognition result SHALL carry per-character confidences, and each plate SHALL be assigned a confidence score computed as the mean of its character probabilities by default, with product and sum-of-log-probabilities available as alternative score functions.

#### Scenario: Score present
- **WHEN** a plate is recognized
- **THEN** its result includes per-character confidences and a plate-level score

#### Scenario: Alternative score function
- **WHEN** the baseline is configured with an alternative score function
- **THEN** plate scores are computed with that function and the chosen function is recorded in the output

### Requirement: Read-abstain confidence policy
A threshold on the plate score SHALL decide "read" vs. "abstain"; the threshold SHALL be selected on the validation split and the test split SHALL be evaluated once with that threshold.

#### Scenario: Threshold selection on validation
- **WHEN** the confidence policy is calibrated
- **THEN** an accuracy-coverage curve (accuracy among non-abstained images vs. coverage) is produced from the validation split and a threshold is selected from it

#### Scenario: Test evaluated once
- **WHEN** the test split is evaluated under the confidence policy
- **THEN** the threshold chosen on validation is used and test read/abstain results are reported for that single threshold

### Requirement: Recognition metrics
The baseline SHALL report character accuracy, plate exact-match rate, and precision/recall/F1 derived from edit distance between predicted and ground-truth plate strings, per split, per protocol, and per size bucket.

#### Scenario: Metrics per protocol
- **WHEN** both protocols are run for a split
- **THEN** character accuracy, exact-match rate, and edit-distance-based P/R/F1 are available separately for the OCR-only and end-to-end protocols

#### Scenario: Per-bucket metrics
- **WHEN** recognition metrics are produced
- **THEN** they are available per size bucket for the evaluated split

### Requirement: Traceable metric output
Recognition results SHALL be written as machine-readable JSON records identifying the OCR engine, the weight digests, the protocol, the preprocessing mode, the score function, the threshold, the split, and the run timestamp.

#### Scenario: Metrics file content
- **WHEN** a recognition run completes
- **THEN** a JSON metrics file exists containing all of the identifying fields above plus the reported metrics

### Requirement: CRNN training readiness
The change SHALL provide the character set of the training data (Chinese province characters, Latin letters, digits) and an export of the training crops with their plate strings, so a later change can train a CRNN without further data preparation.

#### Scenario: Character set complete
- **WHEN** the character-set module is loaded
- **THEN** it contains every distinct character appearing in the training split's plate strings and no others

#### Scenario: Training data export
- **WHEN** the training-data export is run
- **THEN** a file mapping each training crop to its plate string is produced and usable as CRNN input
