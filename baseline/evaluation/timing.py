"""Sampled timing probe: the only part of the evaluation stack that runs inference.

A deterministic subset of original-scale images (val + test) is pushed through
the detector and the OCR engine per preprocessing mode. The first images are a
warmup and are excluded from the statistics. Heavy imports (torch, ultralytics,
easyocr, cv2) live inside ``run_probe`` so that ``--no-probe`` stays fully
offline and stdlib-only.
"""

import os
import platform
import random
import statistics

WARMUP = 5


def sample_images(dataset, n, seed):
    """Sample original-scale samples from val and test, deterministically."""
    pool = []
    for split in ("val", "test"):
        for sample_id in dataset.sample_ids[split]:
            parent_id = sample_id.rsplit("_s", 1)[0]
            row = dataset.rows[(parent_id, 1.0)]
            pool.append(
                {
                    "sample_id": sample_id,
                    "split": split,
                    "image_path": row["image_path"],
                    "crop_path": row["crop_path"],
                }
            )
    rng = random.Random(seed)
    return rng.sample(pool, min(n, len(pool)))


def stage_stats(times_ms):
    if not times_ms:
        raise ValueError("No timings to summarize")
    ordered = sorted(times_ms)
    index = min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))
    return {
        "count": len(times_ms),
        "median_ms": statistics.median(times_ms),
        "p95_ms": ordered[index],
        "mean_ms": statistics.fmean(times_ms),
        "total_s": sum(times_ms) / 1000.0,
    }


def max_rss_mb():
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _environment():
    from importlib.metadata import version

    environment = {"python": platform.python_version(), "cpu_count": os.cpu_count()}
    for name in ("torch", "ultralytics", "easyocr", "opencv-python", "numpy"):
        try:
            environment[name] = version(name)
        except Exception:
            environment[name] = None
    return environment


def run_probe(dataset, detector_weights, easyocr_weights, manifest, n, seed, timestamp):
    import time

    import cv2
    from ultralytics import YOLO

    from baseline.ocr.engine import EasyOCREngine
    from baseline.ocr.io import MODES
    from data.preprocessing import preprocess_plate

    samples = sample_images(dataset, n, seed)
    if len(samples) <= WARMUP:
        raise ValueError(f"Probe needs more than {WARMUP} samples; got {len(samples)}")
    stages = {}

    def measure(name, fn):
        for sample in samples[:WARMUP]:
            fn(sample)
        times = []
        for sample in samples[WARMUP:]:
            start = time.perf_counter()
            fn(sample)
            times.append((time.perf_counter() - start) * 1000.0)
        stats = stage_stats(times)
        stats["warmup"] = WARMUP
        stats["max_rss_mb_after"] = max_rss_mb()
        stages[name] = stats

    # Frozen detection inference settings (baseline/detection/evaluate.py).
    settings = {"imgsz": 640, "conf": 0.001, "iou": 0.7, "max_det": 300,
                "device": "cpu", "batch": 1, "half": False, "augment": False, "rect": False}
    model = YOLO(str(detector_weights))
    measure("detection", lambda sample: model(str(dataset.root / sample["image_path"]), **settings, verbose=False))

    engine = EasyOCREngine(easyocr_weights, manifest)
    for mode in MODES:
        def recognize(sample, mode=mode):
            image = cv2.imread(str(dataset.root / sample["crop_path"]))
            if image is None:
                raise ValueError(f"Cannot decode crop: {sample['crop_path']}")
            engine.recognize(preprocess_plate(image, mode=mode))

        measure(f"ocr_{mode}", recognize)

        def preprocess(sample, mode=mode):
            image = cv2.imread(str(dataset.root / sample["crop_path"]))
            if image is None:
                raise ValueError(f"Cannot decode crop: {sample['crop_path']}")
            preprocess_plate(image, mode=mode)

        measure(f"preprocess_{mode}", preprocess)

    return {
        "n": n,
        "seed": seed,
        "samples": samples,
        "environment": _environment(),
        "stages": stages,
        "timestamp": timestamp,
    }
