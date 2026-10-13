"""Loading and provenance validation of the frozen stage results.

The evaluation stack consumes three inputs: the dataset directory
(metadata + OCR CSVs), one detection results directory, and one OCR results
directory. Every file is checked for internal validity and for provenance
consistency (split, scale, dataset fingerprints, model identity) so that a
results directory from a future model plugs in only when it is genuinely
compatible with the dataset it claims to evaluate.
"""

import csv
from dataclasses import dataclass
from pathlib import Path

from baseline.ocr.io import MODES, PROTOCOLS, dataset_path, digest, read_json, validate_raw

SPLITS = ("val", "test")
SCALE = 1.0


class InputError(ValueError):
    """A required input file is missing or its provenance does not match."""


@dataclass(frozen=True)
class DatasetInputs:
    root: Path
    digests: dict
    sample_ids: dict
    sample_count: dict
    tilt: dict
    rows: dict


@dataclass(frozen=True)
class DetectionInputs:
    directory: Path
    model_id: str
    digests: dict
    predictions: dict
    metrics: dict


@dataclass(frozen=True)
class OcrInputs:
    directory: Path
    engine: str
    digests: dict
    raw: dict
    selection: dict
    test_report: dict
    selected_mode: dict
    selected_tau: dict


def _check_file(directory, name):
    path = Path(directory) / name
    if not path.is_file():
        raise InputError(f"Missing required file: {path}")
    return path


def required_files(dataset, detection, ocr):
    names = ["metadata.csv", "ocr_val.csv", "ocr_test.csv"]
    files = [(Path(dataset) / name, name) for name in names]
    for split in SPLITS:
        for kind in ("predictions", "metrics"):
            name = f"{kind}_{split}.json"
            files.append((Path(detection) / name, name))
    for protocol in PROTOCOLS:
        for mode in MODES:
            name = f"raw_{protocol}_val_{mode}.json"
            files.append((Path(ocr) / protocol / name, name))
        for name in (f"selection_{protocol}_val.json", f"report_{protocol}_test.json"):
            files.append((Path(ocr) / protocol / name, name))
    return files


def _read_table(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _sha256(path):
    from hashlib import file_digest

    with Path(path).open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def load_dataset(root):
    root = Path(root)
    if not root.is_dir():
        raise InputError(f"Dataset directory does not exist: {root}")
    names = ("metadata.csv", "ocr_val.csv", "ocr_test.csv")
    paths = {name: _check_file(root, name) for name in names}

    metadata = {}
    for row in _read_table(paths["metadata.csv"]):
        key = (row["parent_id"], float(row["scale"]))
        if key in metadata:
            raise InputError(f"Duplicate metadata key: {key}")
        metadata[key] = row

    sample_ids = {}
    for split in SPLITS:
        seen = set()
        for row in _read_table(paths[f"ocr_{split}.csv"]):
            if row["split"] != split:
                raise InputError(f"Row with split={row['split']} in ocr_{split}.csv")
            key = (row["parent_id"], float(row["scale"]))
            if float(row["scale"]) != SCALE:
                continue
            if key not in metadata:
                raise InputError(f"OCR sample missing from metadata: {key}")
            if key in seen:
                raise InputError(f"Duplicate OCR sample: {key}")
            seen.add(key)
        sample_ids[split] = sorted(f"{parent_id}_s100" for parent_id, _scale in seen)

    tilt = {key: float(row["tilt_magnitude"]) for key, row in metadata.items()}
    digests = {name: _sha256(path) for name, path in paths.items()}
    return DatasetInputs(
        root=root,
        digests=digests,
        sample_ids=sample_ids,
        sample_count={split: len(ids) for split, ids in sample_ids.items()},
        tilt=tilt,
        rows=metadata,
    )


def _provenance_field(artifact, name, field, split):
    provenance = artifact.get("provenance", artifact)
    if not isinstance(provenance, dict):
        raise InputError(f"{name}: missing provenance block")
    value = provenance.get(field)
    if value is None:
        raise InputError(f"{name}: provenance missing field '{field}'")
    if field == "split" and value != split:
        raise InputError(f"{name}: provenance split is {value!r}, expected {split!r}")
    if field == "scale" and float(value) != SCALE:
        raise InputError(f"{name}: provenance scale is {value!r}, expected {SCALE!r}")
    return value


def load_detection(directory, dataset):
    directory = Path(directory)
    artifacts = {}
    for split in SPLITS:
        for kind in ("predictions", "metrics"):
            name = f"{kind}_{split}.json"
            artifacts[(kind, split)] = read_json(_check_file(directory, name))

    for (kind, split), artifact in artifacts.items():
        name = f"{kind}_{split}.json"
        _provenance_field(artifact, name, "split", split)
    models = {
        _provenance_field(artifact, f"{kind}_{split}.json", "model_id", split)
        for (kind, split), artifact in artifacts.items()
    }
    if len(models) != 1:
        raise InputError(f"Detection files disagree on model: {sorted(models)}")
    model_id = models.pop()

    predictions = {}
    for split in SPLITS:
        records = artifacts[("predictions", split)].get("images")
        if not isinstance(records, list) or not records:
            raise InputError(f"predictions_{split}.json: empty or missing 'images'")
        ids = set()
        for record in records:
            relative = record.get("image_path")
            if not relative:
                raise InputError(f"predictions_{split}.json: record without image_path")
            try:
                dataset_path(dataset.root, relative)
            except ValueError as error:
                raise InputError(f"predictions_{split}.json: {error}") from error
            if not record.get("ground_truth"):
                raise InputError(f"predictions_{split}.json: empty ground truth for {relative}")
            parent_id = Path(relative).stem.rsplit("_s", 1)[0]
            ids.add(f"{parent_id}_s100")
        expected = set(dataset.sample_ids[split])
        if ids != expected:
            missing = sorted(expected - ids)[:5]
            extra = sorted(ids - expected)[:5]
            raise InputError(
                f"predictions_{split}.json: sample set differs from the dataset "
                f"(missing e.g. {missing}, unexpected e.g. {extra})"
            )
        predictions[split] = records

    metrics = {split: artifacts[("metrics", split)] for split in SPLITS}
    for split in SPLITS:
        for block in ("overall", "by_size", "by_tilt"):
            if block not in metrics[split]:
                raise InputError(f"metrics_{split}.json: missing block '{block}'")

    digests = {}
    for (kind, split) in artifacts:
        digests[f"{kind}_{split}.json"] = _sha256(directory / f"{kind}_{split}.json")
    return DetectionInputs(
        directory=directory, model_id=model_id, digests=digests, predictions=predictions, metrics=metrics
    )


def _check_digests(artifact_digests, dataset, name):
    # Artifacts digest only the tables of their own split; every digest they
    # declare must match the current dataset.
    if "metadata.csv" not in artifact_digests:
        raise InputError(f"{name}: dataset_digests is missing metadata.csv")
    for field, actual in artifact_digests.items():
        expected = dataset.digests.get(field)
        if expected is None:
            raise InputError(f"{name}: unknown dataset digest field {field!r}")
        if actual != expected:
            raise InputError(f"{name}: dataset digest for {field} does not match the dataset")


def _ocr_file(directory, protocol, name):
    return _check_file(Path(directory) / protocol, name)


def load_ocr(directory, dataset, detection):
    directory = Path(directory)
    raw = {}
    engine = None
    for protocol in PROTOCOLS:
        for mode in MODES:
            name = f"raw_{protocol}_val_{mode}.json"
            artifact = read_json(_ocr_file(directory, protocol, name))
            try:
                validate_raw(artifact)
            except ValueError as error:
                raise InputError(f"{name}: {error}") from error
            metadata = artifact["metadata"]
            if metadata["protocol"] != protocol:
                raise InputError(f"{name}: protocol is {metadata['protocol']!r}, expected {protocol!r}")
            if metadata["split"] != "val" or metadata["mode"] != mode:
                raise InputError(f"{name}: split/mode does not match the file name")
            if float(metadata["scale"]) != SCALE:
                raise InputError(f"{name}: scale is {metadata['scale']!r}, expected {SCALE!r}")
            if metadata["sample_count"] != dataset.sample_count["val"]:
                raise InputError(
                    f"{name}: sample_count is {metadata['sample_count']}, "
                    f"dataset has {dataset.sample_count['val']} val samples"
                )
            _check_digests(metadata["dataset_digests"], dataset, name)
            if engine is None:
                engine = metadata["engine"]
            elif metadata["engine"] != engine:
                raise InputError(f"{name}: engine is {metadata['engine']!r}, expected {engine!r}")
            if protocol == "end_to_end":
                detector = metadata.get("detector", {})
                if detector.get("model_id") != detection.model_id:
                    raise InputError(
                        f"{name}: detector model is {detector.get('model_id')!r}, "
                        f"detection results use {detection.model_id!r}"
                    )
            raw[(protocol, mode)] = artifact

    selection = {}
    test_report = {}
    selected_mode = {}
    selected_tau = {}
    for protocol in PROTOCOLS:
        sel_name = f"selection_{protocol}_val.json"
        sel = read_json(_ocr_file(directory, protocol, sel_name))
        if sel.get("schema_version") != 1 or sel.get("selection_split") != "val":
            raise InputError(f"{sel_name}: unsupported selection schema")
        policy = sel["protocols"][protocol]
        if set(policy["candidates"]) != set(MODES):
            raise InputError(f"{sel_name}: candidates must cover all three modes")
        for mode in MODES:
            candidate = policy["candidates"][mode]
            local = directory / protocol / f"raw_{protocol}_val_{mode}.json"
            if candidate["raw_sha256"] != digest(local):
                raise InputError(f"{sel_name}: saved raw digest for {mode} does not match the local cache")
        selected_mode[protocol] = policy["mode"]
        selected_tau[protocol] = policy["tau"]
        selection[protocol] = policy

        rep_name = f"report_{protocol}_test.json"
        report = read_json(_ocr_file(directory, protocol, rep_name))
        metadata = report["metadata"]
        if metadata["split"] != "test" or metadata["mode"] != policy["mode"]:
            raise InputError(f"{rep_name}: split/mode does not match the validation selection")
        if metadata.get("tau") != policy["tau"]:
            raise InputError(f"{rep_name}: tau does not match the validation selection")
        _check_digests(metadata["dataset_digests"], dataset, rep_name)
        test_report[protocol] = report

    digests = {}
    for protocol in PROTOCOLS:
        for mode in MODES:
            name = f"raw_{protocol}_val_{mode}.json"
            digests[f"{protocol}/{name}"] = _sha256(directory / protocol / name)
        for name in (f"selection_{protocol}_val.json", f"report_{protocol}_test.json"):
            digests[f"{protocol}/{name}"] = _sha256(directory / protocol / name)
    return OcrInputs(
        directory=directory,
        engine=engine,
        digests=digests,
        raw=raw,
        selection=selection,
        test_report=test_report,
        selected_mode=selected_mode,
        selected_tau=selected_tau,
    )


def load_all(dataset_root, detection_dir, ocr_dir):
    missing = [str(path) for path, _name in required_files(dataset_root, detection_dir, ocr_dir) if not path.is_file()]
    if missing:
        raise InputError("Missing required input files: " + ", ".join(missing))
    dataset = load_dataset(dataset_root)
    detection = load_detection(detection_dir, dataset)
    ocr = load_ocr(ocr_dir, dataset, detection)
    return dataset, detection, ocr
