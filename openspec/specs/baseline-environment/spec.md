# baseline-environment Specification

## Purpose
Self-managed, pinned environment for all baseline model stages: baseline
scripts install their exact declared dependencies into a dedicated venv and
acquire model weights verified by sha256, so every baseline runs on the same
software and the same model artifacts on any machine.

## Requirements

### Requirement: Pinned dependency installation
Baseline entry points SHALL install their declared dependencies at exact
pinned versions into a dedicated baseline virtual environment, creating it
when missing and repairing it when a pinned version is not satisfied.

#### Scenario: First run on a clean machine
- **WHEN** a baseline script runs and the baseline venv does not exist
- **THEN** the script creates the venv, installs all pinned dependencies for that stage, and proceeds without manual steps

#### Scenario: Satisfied environment
- **WHEN** a baseline script runs and the venv already satisfies every pin
- **THEN** no installation happens and the script proceeds immediately

#### Scenario: Version drift
- **WHEN** a baseline script runs and a package in the venv does not match its pinned version
- **THEN** the script installs the pinned version before proceeding

### Requirement: Manifest-driven reproducibility
The dependencies of each baseline stage SHALL be declared in a machine-readable pinned manifest stored in the repository, so the environment is reconstructible from a clean clone.

#### Scenario: Rebuild from clone
- **WHEN** the repository is cloned on a new machine and a baseline script is run
- **THEN** the resulting environment contains exactly the pinned versions named in that stage's manifest

### Requirement: Hash-verified weight acquisition
Model weights SHALL be downloaded from pinned URLs and verified against a recorded sha256 digest before any use; weights are cached locally after a successful verification.

#### Scenario: Cached weights
- **WHEN** the weight file is already cached and its sha256 matches the manifest
- **THEN** the cache is used without a network download

#### Scenario: Missing weights
- **WHEN** the weight file is not cached
- **THEN** it is downloaded from the pinned URL, its sha256 is verified, and it is stored in the cache

#### Scenario: Hash mismatch
- **WHEN** a weight file (cached or freshly downloaded) does not match the recorded sha256
- **THEN** the script fails with an explicit error naming the expected and actual digest and MUST NOT use the weights

### Requirement: Isolation from the data environment
Baseline environment setup MUST NOT modify the project's main virtual environment, `data/requirements.txt`, or any file under `data/`.

#### Scenario: Setup leaves data env untouched
- **WHEN** a baseline script completes environment setup
- **THEN** the main project venv and `data/requirements.txt` are unchanged
