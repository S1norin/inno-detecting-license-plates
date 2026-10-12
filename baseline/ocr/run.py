"""CPU OCR inference with resumable caches and a validation policy for test."""

import argparse
import json
from pathlib import Path

from . import end_to_end, ocr_only
from .engine import EasyOCREngine
from .evaluate import load_policy, model_contract
from .io import (
    MODES,
    PROTOCOLS,
    SCALES,
    digest,
    load_samples,
    read_json,
    timestamp,
    validate_raw,
    write_json,
)


def reserve_test(selection_path, protocol, output):
    reservation = Path(selection_path).parent / f"test_inference_{protocol}.json"
    identity = {"output": str(Path(output).resolve()), "selection_sha256": digest(selection_path)}
    try:
        with reservation.open("x", encoding="utf-8") as stream:
            json.dump(identity, stream)
    except FileExistsError:
        if read_json(reservation) != identity:
            raise ValueError("Test inference already reserved; resume its existing cache and policy") from None
        if not Path(output).exists():
            raise ValueError("Reserved test cache is missing; do not rerun test to tune the model") from None


def run_inference(args, engine):
    import cv2

    from data.preprocessing import preprocess_plate

    samples = load_samples(args.dataset, args.split, args.scale)
    adapter = ocr_only if args.protocol == "ocr_only" else end_to_end
    predictions = detector = None
    if args.protocol == "end_to_end":
        if not args.predictions:
            raise ValueError("End-to-end inference requires --predictions")
        detector, predictions = end_to_end.load_predictions(args.predictions, samples)

    metadata = {
        "protocol": args.protocol,
        "split": args.split,
        "mode": args.mode,
        "scale": args.scale,
        "sample_count": len(samples),
        "engine": engine.identity,
        "weight_digests": engine.weight_digests,
        "dataset_root": str(args.dataset.resolve()),
        "preprocessing_sha256": digest(Path(__file__).resolve().parents[2] / "data" / "preprocessing.py"),
        "dataset_digests": {
            f"ocr_{args.split}.csv": digest(args.dataset / f"ocr_{args.split}.csv"),
            "metadata.csv": digest(args.dataset / "metadata.csv"),
        },
    }
    if detector:
        metadata.update(detector=detector, predictions_sha256=digest(args.predictions))

    signature = metadata.copy()
    metadata["timestamp"] = timestamp()
    expected = {sample["sample_id"]: sample for sample in samples}
    paths = {key: adapter.input_path(args.dataset, sample) for key, sample in expected.items()}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} input images; first: {missing[0]}")

    cache = {"schema_version": 1, "metadata": metadata, "records": []}
    completed = {}
    if args.output.exists():
        cache = validate_raw(read_json(args.output), complete=False)
        previous = {key: value for key, value in cache["metadata"].items() if key != "timestamp"}
        if previous != signature:
            raise ValueError("Cache parameters/weights/dataset differ; use a new validation output")
        completed = {record["sample_id"]: record for record in cache["records"]}
        for key, record in completed.items():
            if key not in expected or any(record[field] != value for field, value in expected[key].items()):
                raise ValueError("Cache sample differs from the current dataset")
            if record["input_sha256"] != digest(paths[key]):
                raise ValueError(f"Cached input image changed: {key}")

    if args.split == "test":
        if not args.selection:
            raise ValueError("Test inference requires a saved --selection from validation")
        policy = load_policy(read_json(args.selection), args.protocol)
        if args.mode != policy["mode"] or model_contract(metadata) != policy["contract"]:
            raise ValueError("Test inference must use validation-selected model, dataset, scale and mode")

    if not args.output.exists():
        write_json(args.output, cache)
    if args.split == "test":
        reserve_test(args.selection, args.protocol, args.output)

    for sample in samples:
        key = sample["sample_id"]
        if key in completed:
            continue
        path = paths[key]
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Cannot decode input image: {path}")

        crop, extra = adapter.prepare_image(image, predictions[sample["image_path"]] if predictions else sample)
        result = {"text": "", "char_confidences": []}
        if crop is not None:
            result = engine.recognize(preprocess_plate(crop, mode=args.mode))
        record = {
            **sample,
            **result,
            "status": "read" if result["text"] else "unread",
            "input_sha256": digest(path),
            **extra,
        }
        completed[key] = record
        cache["records"].append(record)
        validate_raw(cache, complete=False)
        write_json(args.output, cache)
        if len(completed) % 25 == 0 or len(completed) == len(samples):
            print(f"{args.protocol}/{args.split}/{args.mode}: {len(completed)}/{len(samples)}", flush=True)

    validate_raw(cache)
    return cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=PROTOCOLS, required=True)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--dataset", type=Path, default=Path("data/dataset"))
    parser.add_argument("--scale", type=float, choices=SCALES, default=1.0)
    parser.add_argument("--weights-directory", type=Path, default=Path("baseline/weights/easyocr"))
    parser.add_argument("--weights-manifest", type=Path, default=Path("baseline/common/models.json"))
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    engine = EasyOCREngine(args.weights_directory, args.weights_manifest)
    run_inference(args, engine)


if __name__ == "__main__":
    main()
