# baseline-detection Specification

## Purpose
Detection baseline for the license plate project: evaluates the pinned
pretrained YOLOv11s plate detector on the prepared CCPD dataset and produces
reproducible metrics overall and per plate-size and tilt condition, as the
reference point for all later fine-tuning and end-to-end comparisons.

## Requirements

### Requirement: Pinned detector evaluation
The detection baseline SHALL evaluate the pinned pretrained YOLOv11s license plate detector on the validation and test splits of the prepared dataset, using only original-scale (1.0) images for the primary evaluation.

#### Scenario: Test split evaluation
- **WHEN** the detection baseline is run on the test split
- **THEN** every test image receives predicted plate boxes that are matched against the dataset ground-truth labels

#### Scenario: Primary evaluation uses original scale
- **WHEN** the primary evaluation is run
- **THEN** only scale 1.0 images are evaluated, per the original-scale dataset configuration

### Requirement: Detection metrics
The baseline SHALL report mAP@0.5 and F1 at IoU 0.5 for each evaluated split, alongside overall precision and recall at the matched threshold.

#### Scenario: Validation metrics
- **WHEN** the baseline finishes evaluating the validation split
- **THEN** mAP@0.5, F1@0.5, precision, and recall for that split are present in the metrics output

#### Scenario: Test metrics
- **WHEN** the baseline finishes evaluating the test split
- **THEN** mAP@0.5, F1@0.5, precision, and recall for the test split are present in the metrics output

### Requirement: Size and tilt breakdowns
Metrics SHALL be reported per plate size bucket (small, medium, large, using the dataset's train-derived thresholds) and per tilt bin, with bucket and tilt taken from the dataset metadata.

#### Scenario: Per-bucket metrics
- **WHEN** the baseline metrics are produced
- **THEN** mAP@0.5 and F1@0.5 are available separately for each size bucket present in the evaluated split

#### Scenario: Per-tilt-bin metrics
- **WHEN** the baseline metrics are produced
- **THEN** detection metrics are available per tilt bin for the evaluated split

### Requirement: Traceable metric output
Evaluation results SHALL be written as machine-readable JSON records that identify the model, the weight file's sha256, the dataset configuration, the split, and the evaluation timestamp, so any reported number can be traced to exact artifacts.

#### Scenario: Metrics file content
- **WHEN** an evaluation run completes
- **THEN** a JSON metrics file exists containing the model identifier, weight sha256, dataset configuration used, split name, timestamp, and all reported metrics

#### Scenario: Deterministic re-run
- **WHEN** the evaluation is re-run on the same machine with the same weights
- **THEN** the reported metrics are identical

### Requirement: Fine-tuning readiness
The change SHALL leave the detector and dataset in a state where the same model can be fine-tuned on the train split by a later change without any further dataset preparation.

#### Scenario: Train configuration available
- **WHEN** a later fine-tuning change starts
- **THEN** a training configuration pointing at the train split is available and usable with the cached, hash-verified weights
