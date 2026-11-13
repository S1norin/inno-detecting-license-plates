"""Vertical-slice check: real YOLO + CRNN inference on real dataset images.

Run: .venv/bin/python demo/check_inference.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


from demo.config import YOLO_CONF_FOR_AP  # noqa: E402
from demo.crnn import CRNNRecognizer  # noqa: E402
from demo.dataset import _row_to_sample, load_image, load_metadata  # noqa: E402
from demo.yolo import get_detectors  # noqa: E402


def main():
    frame = load_metadata()
    # A normal test image and a documented detection-failure image
    ids = ["261f3b723deb67ac_s100", "6f3c772ca888fa92_s100"]
    samples = {}
    for sample_id in ids:
        sub = frame[frame["image_path"].str.endswith(f"{sample_id}.jpg")]
        samples[sample_id] = _row_to_sample(sub.iloc[0])

    detectors = get_detectors()
    crnn = CRNNRecognizer()
    print(f"CRNN charset: {len(crnn.characters) - 1} chars, epoch "
          f"{crnn.identity['selected_epoch']}, params {crnn.identity['parameter_count']}")

    failures = 0
    for sample_id, sample in samples.items():
        print(f"\n=== {sample_id} (GT: {sample.plate_text}, {sample.size_bucket}, tilt {sample.tilt_magnitude}) ===")
        image = load_image(sample.image_path)
        assert image is not None, f"missing image {sample.image_path}"
        for key, det in detectors.items():
            res = det.detect(image, conf=YOLO_CONF_FOR_AP)
            top = res.detections[0] if res.detections else None
            if top is not None:
                desc = (
                    f"top=({top.confidence:.2f}, {top.x1:.0f},{top.y1:.0f},"
                    f"{top.x2:.0f},{top.y2:.0f})"
                )
            else:
                desc = "no detections"
            print(f"  YOLO {key:9s}: {len(res.detections):2d} boxes, {res.elapsed_ms:6.0f} ms, {desc}")
            if top is not None:
                from demo.viz import crop_box

                crop = crop_box(image, (top.x1, top.y1, top.x2, top.y2))
                recog = crnn.recognize(crop)
                print(f"    CRNN on top-1 crop: `{recog.text}` ({recog.elapsed_ms:.0f} ms)")
        crop_gt = load_image(sample.crop_path)
        recog = crnn.recognize(crop_gt)
        match = "MATCH" if recog.text == sample.plate_text else "diff"
        print(f"  CRNN on GT crop: `{recog.text}` vs GT `{sample.plate_text}` [{match}]")

    # Determinism spot check: same GT crop twice
    crop = load_image(samples["261f3b723deb67ac_s100"].crop_path)
    a = crnn.recognize(crop).text
    b = crnn.recognize(crop).text
    assert a == b, "CRNN inference not deterministic"
    print(f"\nDeterminism check OK: `{a}` == `{b}`")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
