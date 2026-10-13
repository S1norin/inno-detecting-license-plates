"""Command-line entry point for the baseline evaluation stack.

Consumes the frozen stage results (detection predictions/metrics and OCR
caches/selection/test reports) plus the dataset directory, and writes one
run folder with table.json, failures.json, an optional timing.json,
REPORT.md, and a summary figure.
"""

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from .failures import build_failures
from .inputs import load_all
from .report import build_report, make_figure
from .table import make_document
from .timing import run_probe

WEIGHTS_MANIFEST_DEFAULT = Path("baseline/common/models.json")
EASYOCR_WEIGHTS_DEFAULT = Path("baseline/weights/easyocr")


def build_parser():
    parser = argparse.ArgumentParser(
        prog="python -m baseline.evaluation.run",
        description="Consolidate frozen baseline stage results into one report.",
    )
    parser.add_argument("--dataset", type=Path, required=True,
                        help="Dataset directory containing metadata.csv and ocr_{val,test}.csv.")
    parser.add_argument("--detection-results", type=Path, required=True,
                        help="Detection results directory with predictions/metrics_{val,test}.json.")
    parser.add_argument("--ocr-results", type=Path, required=True,
                        help="OCR results directory with ocr_only/ and end_to_end/ protocol folders.")
    parser.add_argument("--out", type=Path, required=True,
                        help="Output directory for this evaluation run (created if missing).")
    parser.add_argument("--no-probe", action="store_true",
                        help="Skip the timing probe; the run then stays fully offline.")
    parser.add_argument("--probe-n", type=int, default=100,
                        help="Number of images for the timing probe (default: 100).")
    parser.add_argument("--probe-seed", type=int, default=42,
                        help="Seed for the timing probe image sample (default: 42).")
    parser.add_argument("--detector-weights", type=Path,
                        default=Path("baseline/weights/license-plate-finetune-v1s.pt"),
                        help="YOLO weights for the timing probe only.")
    parser.add_argument("--easyocr-weights", type=Path, default=EASYOCR_WEIGHTS_DEFAULT,
                        help="Offline EasyOCR weights directory for the timing probe only.")
    parser.add_argument("--weights-manifest", type=Path, default=WEIGHTS_MANIFEST_DEFAULT,
                        help="Weight manifest for probe verification.")
    parser.add_argument("--timestamp", type=str, default=None,
                        help="Override the run timestamp (tests).")
    return parser


def _write_json(path, document):
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def run(args):
    dataset, detection, ocr = load_all(args.dataset, args.detection_results, args.ocr_results)
    timestamp = args.timestamp or datetime.now(UTC).isoformat()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    table_doc = make_document(dataset, detection, ocr, timestamp)
    _write_json(out / "table.json", table_doc)

    failures = build_failures(dataset, detection, ocr)
    _write_json(out / "failures.json", failures)

    timing = None
    if not args.no_probe:
        timing = run_probe(
            dataset,
            args.detector_weights,
            args.easyocr_weights,
            args.weights_manifest,
            args.probe_n,
            args.probe_seed,
            timestamp,
        )
        _write_json(out / "timing.json", timing)

    (out / "REPORT.md").write_text(build_report(table_doc, failures, timing), encoding="utf-8")
    if not make_figure(table_doc, out / "figures" / "baseline_comparison.png"):
        print("matplotlib is not available; the summary figure was skipped", file=sys.stderr)
    return out


def main(argv=None):
    args = build_parser().parse_args(argv)
    out = run(args)
    print(f"evaluation run written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
