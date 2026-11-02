# baseline-detection Delta

## ADDED Requirements

### Requirement: Scaled-variant evaluation
The detection baseline SHALL support evaluating explicit scale subsets of
the dataset variants (0.75 and 0.5) through an additive command-line option,
while the primary evaluation remains original-scale (1.0) only. Scaled runs
SHALL use the same matching, metric, breakdown, and traceability rules as the
primary evaluation, with bucket and tilt taken from each variant's own
metadata row.

#### Scenario: Scaled test split evaluation
- **WHEN** the detection baseline is run on the test split with a scale subset other than 1.0
- **THEN** every variant image of the requested scales receives predicted plate boxes matched against its own ground-truth label, and metrics are reported overall and per size bucket and tilt bin

#### Scenario: Primary evaluation unchanged by default
- **WHEN** the detection baseline is run without the scale option
- **THEN** only scale 1.0 images are evaluated, exactly as before

#### Scenario: Scaled run provenance
- **WHEN** a scaled evaluation run completes
- **THEN** its JSON metrics output records the evaluated scale subset alongside the model, weight sha256, dataset configuration, split, and timestamp

#### Scenario: Unknown scale refused
- **WHEN** the scale option names a scale that is not one of the dataset's variant scales
- **THEN** the run is refused with a clear error before any inference
