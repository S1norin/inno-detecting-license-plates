# baseline-evaluation Specification

## Purpose
Consolidates the frozen detection and recognition baseline results into one
reproducible evaluation artifact set: a comparison table covering all planned
experiments, failure cases with likely causes, and a sampled timing probe, so
the report, demo, and any future model comparison start from the same numbers.

## Requirements

### Requirement: Consolidated comparison table

The evaluation run SHALL produce a single machine-readable table covering,
for both the validation and test splits: detection mAP@0.5, precision,
recall and F1; OCR-only recognition metrics (character accuracy, exact match,
precision, recall, F1, coverage); end-to-end recognition metrics with the
delta versus OCR-only; the preprocessing ablation (none / gray+contrast /
Canny) with per-step delta versus the no-preprocessing mode; per-plate-size
bucket breakdowns for both stages; per-tilt-bin breakdowns for both stages;
and accuracy-coverage values at the validation-selected confidence threshold.

#### Scenario: Table built from existing frozen results

- **WHEN** a run is started and the detection metrics/predictions and the
  recognition validation and test result files for both protocols exist
- **THEN** the run produces the consolidated table with every experiment row
  (E1 through E6) present for the available splits, without performing any
  new metrics inference

#### Scenario: Missing input fails the run

- **WHEN** any required detection or recognition result file is missing
- **THEN** the run fails before writing any output and names the missing
  input; it never substitutes, skips, or re-derives a missing stage

### Requirement: Frozen results are read-only

The evaluation run SHALL treat all existing detection and recognition result
files as immutable inputs. Test-split recognition metrics SHALL come only
from the previously frozen test reports; the run MUST NOT re-run
recognition inference on the test split or alter any input file.

#### Scenario: Repeated runs are identical

- **WHEN** two runs are started against the same inputs
- **THEN** both produce byte-identical table, failure-case, and report
  contents (excluding timestamps and run identifiers), and no input file is
  modified

#### Scenario: Test metrics come from the frozen report

- **WHEN** the consolidated table includes test-split recognition rows
- **THEN** those values are read from the frozen test report files and the
  run performs no recognition inference on test images

### Requirement: Failure case dumps

The run SHALL dump at least 3 detection failure cases and at least 3
recognition failure cases. Each detection case SHALL include the image
identifier, ground-truth box, predicted boxes with scores, and an
auto-tagged likely cause (for example: missed, low-IoU match, multiple
detections). Each recognition case SHALL include the sample identifier,
ground-truth text, raw read, confidence, and an auto-tagged likely cause.
Recognition examples SHALL be drawn from validation results, not test.

#### Scenario: Detections without a matching box are failures

- **WHEN** a validation or test image has no predicted box reaching the
  matching IoU threshold
- **THEN** it is eligible as a detection failure case tagged as a miss

#### Scenario: Recognition failures come from validation

- **WHEN** the recognition failure dump is produced
- **THEN** every case references a validation sample and none references a
  test sample

#### Scenario: Minimum case count

- **WHEN** the failure dumps are written
- **THEN** each contains at least 3 cases, ordered by severity

### Requirement: Sampled timing probe

The run SHALL measure per-stage latency and peak memory on a fixed, seeded
sample of N images (default 100). Stages SHALL include detection inference,
recognition inference for each preprocessing mode, and the preprocessing
itself. The probe SHALL report per-stage median and p95 latency in
milliseconds and peak memory, together with the sample size and seed. The
probe MUST NOT modify any frozen result, and re-running the probe on the
same sample MUST NOT affect reported metrics.

#### Scenario: Same seed selects the same sample

- **WHEN** two probe runs use the same seed and sample size
- **THEN** both measure the identical set of images

#### Scenario: Probe output is self-describing

- **WHEN** the timing report is written
- **THEN** it records per-stage median and p95 latency, peak memory, sample
  size, seed, and the inference settings used

#### Scenario: Probe is safe to repeat

- **WHEN** a probe is re-run after a completed run
- **THEN** the previously written consolidated table and frozen inputs are
  unchanged

### Requirement: Report and figure artifacts

The run SHALL write a tracked report document containing the consolidated
table with a short interpretation, and exactly one committed figure
summarizing the baseline evaluation for the two-page technical summary.
Raw per-run JSON artifacts (table, failures, timing) SHALL be written to a
gitignored results directory.

#### Scenario: A run produces the full artifact set

- **WHEN** a run completes successfully
- **THEN** the results directory contains the table, failure-case, and
  timing JSON files, the report document exists with the table, and the
  committed figure file exists

#### Scenario: Raw results stay out of version control

- **WHEN** a run completes
- **THEN** files under the evaluation results directory are not tracked in
  git, while the report document and figure are

### Requirement: Path-based input convention

Detection and recognition inputs SHALL be supplied as paths and SHALL be
validated against the expected dataset (split, scale, and dataset
fingerprint recorded in their provenance) before use. Inputs whose
provenance does not match the dataset being evaluated SHALL be rejected.

#### Scenario: Provenance mismatch is rejected

- **WHEN** a supplied detection result file names a different dataset
  fingerprint or split than the run targets
- **THEN** the run fails and names the mismatched field

#### Scenario: Alternate results directory plugs in

- **WHEN** a run is given a results directory produced by a different model
  run with matching dataset provenance
- **THEN** the harness accepts it and builds the table from it without code
  changes
