"""Detector-output adapter: highest-score box, clamped crop, explicit no-read."""

import math

from .io import dataset_path, read_json


def load_predictions(path, samples, *, split=None, metadata_sha256=None):
    """Read #5's native result or the original interchange schema."""
    document = read_json(path)
    if "provenance" in document:
        provenance = document["provenance"]
        if provenance.get("schema_version") != 1:
            raise ValueError("Unsupported detector provenance schema")
        for record in document["images"]:
            if len(record["class_ids"]) != len(record["boxes"]) or any(c != 0 for c in record["class_ids"]):
                raise ValueError("End-to-end OCR requires single-class plate predictions")
        document = {
            "schema_version": 1,
            "metadata": {**provenance, "box_format": "xyxy_pixels", "source_format": "detection_baseline_v1"},
            "records": document["images"],
        }

    if document.get("schema_version") != 1 or document["metadata"].get("box_format") != "xyxy_pixels":
        raise ValueError("Predictions require schema_version=1 and box_format=xyxy_pixels")
    if not document["metadata"].get("weight_sha256") or not document["metadata"].get("model_id"):
        raise ValueError("Detector model id and weight hash are required")
    metadata = document["metadata"]
    if split is not None and metadata.get("split", split) != split:
        raise ValueError("Detector predictions belong to a different split")
    if (
        metadata_sha256 is not None
        and "input_sha256" in metadata
        and metadata["input_sha256"]["metadata"] != metadata_sha256
    ):
        raise ValueError("Detector predictions were made from different dataset metadata")

    records = {}
    for record in document["records"]:
        key = record["image_path"].replace("\\", "/")
        if key in records:
            raise ValueError(f"Duplicate detector image: {key}")
        if len(record["boxes"]) != len(record["scores"]):
            raise ValueError(f"Detector boxes/scores mismatch: {key}")
        for box, score in zip(record["boxes"], record["scores"], strict=True):
            if len(box) != 4 or any(not math.isfinite(x) for x in box):
                raise ValueError(f"Invalid detector coordinates: {key}")
            if box[2] <= box[0] or box[3] <= box[1] or not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError(f"Invalid detector box/score: {key}")
        records[key] = record

    expected = {sample["image_path"] for sample in samples}
    if set(records) != expected:
        raise ValueError("Detector predictions must contain every selected split image exactly once")
    return metadata, records


def input_path(dataset, sample):
    return dataset_path(dataset, sample["image_path"])


def select_box(boxes, scores, width, height):
    if not boxes:
        return None
    index = max(range(len(scores)), key=lambda i: (scores[i], -i))
    x1, y1, x2, y2 = boxes[index]
    box = (
        max(0, min(width, math.floor(x1))),
        max(0, min(height, math.floor(y1))),
        max(0, min(width, math.ceil(x2))),
        max(0, min(height, math.ceil(y2))),
    )
    if box[2] <= box[0] or box[3] <= box[1]:
        return None
    return box, scores[index]


def prepare_image(image, prediction):
    selected = select_box(prediction["boxes"], prediction["scores"], image.shape[1], image.shape[0])
    if selected is None:
        return None, {"status": "no_detection", "detector_box": None, "detector_score": None}

    box, score = selected
    x1, y1, x2, y2 = box
    return image[y1:y2, x1:x2], {"detector_box": list(box), "detector_score": score}
