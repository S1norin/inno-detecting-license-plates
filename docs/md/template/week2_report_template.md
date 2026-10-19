# Progress report: <short title, e.g. "Plate-specific recognizer">

[Code: <repo URL>](<repo URL>)

Status as of <DD Month YYYY>, for the 20 October "second approach" milestone.

**Contributors**

| Name | Email |
|---|---|
| Mikhail Krylov | m.krylov@innopolis.university |
| Maximilian Mifsud Bonici | m.mifsudbonici@innopolis.university |
| Andrei Polevoi | a.polevoi@innopolis.university |
| Evgenii Khovalyg | e.khovalyg@innopolis.university |

## Milestone status

<One short paragraph: what the second approach is, that it works, and how it
compares to the week-1 baseline. Name the weak stage and what changed it.>

| Stage | Model | Status |
|---|---|---|
| Dataset | <...> | complete |
| Detection baseline (E1) | <...> | complete |
| Recognition baseline (E2) | <...> | complete |
| Second approach (E2/E2b) | <...> | <complete / in progress> |
| Preprocessing ablation (E3) | <...> | <complete / in progress> |
| Size and tilt sensitivity (E4/E5) | <...> | <complete / in progress> |
| Confidence threshold (E6) | <...> | <complete / in progress> |
| <next experiment> | <...> | <remaining / planned> |

## 1. Second approach

### 1.1 What changed

<Two to four sentences: what was replaced or added relative to the week-1
baseline, and why (link to the week-1 observation that motivated it).>

### 1.2 Model

<Architecture summary: backbone, layers, output head, character set,
input size, and where the code lives (e.g. `models/<name>/`).>

### 1.3 Training setup

| Setting | Value |
|---|---|
| Data | <e.g. 3,500 original-scale train crops> |
| Optimizer / LR | <...> |
| Batch size / epochs | <...> |
| Image size | <...> |
| Seed / determinism | <...> |
| Hardware | <...> |
| Runtime | <...> |

Table 1: Training setup.

### 1.4 Evaluation protocol

<One paragraph: state that the same evaluation harness as week 1 is used,
the two protocols (ocr_only, end_to_end), the confidence metric, and that
mode + threshold were selected on validation only, with test reported once.>

## 2. Results

### 2.1 Detection (frozen baseline)

| Split | mAP@0.5 | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Validation | <...> | <...> | <...> | <...> |
| Test | <...> | <...> | <...> | <...> |

Table 2: Detection results (unchanged from week 1).

### 2.2 Recognition: baseline vs second approach

| Protocol | Engine | Coverage | Char. accuracy | Exact plates | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| ocr_only | <baseline> | <...> | <...> | <...> | <...> | <...> | <...> |
| ocr_only | <second approach> | <...> | <...> | <...> | <...> | <...> | <...> |
| end_to_end | <baseline> | <...> | <...> | <...> | <...> | <...> | <...> |
| end_to_end | <second approach> | <...> | <...> | <...> | <...> | <...> | <...> |

Table 3: Test-split recognition results, validation-selected policy.

### 2.3 Ablations

#### Preprocessing (E3)

| Mode | Coverage | Char. accuracy |
|---|---:|---:|
| none | <...> | <...> |
| gray_contrast | <...> | <...> |
| canny | <...> | <...> |

Table 4: Validation-split preprocessing ablation.

#### Size buckets and tilt bins (E4/E5)

| Bucket / bin | Char. accuracy (ocr_only) | Char. accuracy (end_to_end) |
|---|---:|---:|
| <small / bin> | <...> | <...> |
| <medium / bin> | <...> | <...> |
| <large / bin> | <...> | <...> |

Table 5: Test-split recognition by plate size and tilt bin.

#### Confidence threshold (E6)

<One to two sentences: accuracy-coverage sweep range, the selected tau on
validation, and the selected policy for each protocol.>

## 3. Observations

- <Main improvement, with numbers.>
- <Where the second approach is still weak, with numbers.>
- <Largest end-to-end gap / detector floor, if relevant.>
- <Any ablation surprise.>

## 4. Remaining work

- <Next experiment and why.>
- <Planned improvement, if any.>
- <Final-submission items and deadline.>

## Limitations

- <Custom split / benchmark comparability, if applicable.>
- <Any proxy labels or approximation used.>
- <Any timing/sampling caveats.>

## References

- <Dataset format and usage: data/README.md>
- <Baseline details: baseline/<stage>/README.md>
- <Second-approach details and reproduction: models/<name>/REPORT.md>
- Repository: <repo URL>
