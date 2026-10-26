"""Compare frozen baseline and fine-tuned detection metrics JSONs."""

import argparse
import json
from pathlib import Path

from baseline.detection.metrics import SIZE_GROUPS, TILT_GROUPS

METRIC_FIELDS = ("map50", "precision", "recall", "f1")
SCOPES = [("overall", "overall", None)] \
    + [(f"size/{group}", "by_size", group) for group in SIZE_GROUPS] \
    + [(f"tilt/{group}", "by_tilt", group) for group in TILT_GROUPS]
REQUIRED_KEYS = ("model_id", "weight_sha256", "split", "overall", "by_size", "by_tilt")


def load_metrics(path):
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    missing = [key for key in REQUIRED_KEYS if key not in document]
    if missing:
        raise ValueError(f"Metrics JSON missing {', '.join(missing)}: {path}")
    return document


def build_comparison(pretrained, finetuned):
    if pretrained["split"] != finetuned["split"]:
        raise ValueError(f"Split mismatch: {pretrained['split']} vs {finetuned['split']}")
    comparison = {}
    for key, section, scope in SCOPES:
        base = pretrained[section] if scope is None else pretrained[section][scope]
        tuned = finetuned[section] if scope is None else finetuned[section][scope]
        if base["image_count"] != tuned["image_count"]:
            raise ValueError(f"Image count differs for {key}: {base['image_count']} vs {tuned['image_count']}")
        comparison[key] = {
            "image_count": base["image_count"],
            "pretrained": {field: base[field] for field in METRIC_FIELDS},
            "finetuned": {field: tuned[field] for field in METRIC_FIELDS},
            "delta": {field: tuned[field] - base[field] for field in METRIC_FIELDS},
        }
    return comparison


def provenance(document, source):
    return {"model_id": document["model_id"], "weight_sha256": document["weight_sha256"],
            "source": str(source), "split": document["split"]}


def markdown_table(comparison):
    lines = ["| scope | images | metric | pretrained | finetuned | delta |",
             "|---|---|---|---|---|---|"]
    for key, row in comparison.items():
        for field in METRIC_FIELDS:
            lines.append(f"| {key} | {row['image_count']} | {field} "
                         f"| {row['pretrained'][field]:.4f} | {row['finetuned'][field]:.4f} "
                         f"| {row['delta'][field]:+.4f} |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--finetuned", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pretrained = load_metrics(args.pretrained)
    finetuned = load_metrics(args.finetuned)
    comparison = build_comparison(pretrained, finetuned)
    document = {"schema_version": 1,
                "pretrained": provenance(pretrained, args.pretrained),
                "finetuned": provenance(finetuned, args.finetuned),
                "comparison": comparison}
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(document, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(markdown_table(comparison))
    print(f"\nComparison written to {args.output}")


if __name__ == "__main__":
    main()
