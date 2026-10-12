"""Select on validation caches, then report test once without rerunning OCR."""

import argparse
from pathlib import Path

from .confidence import SCORE_FUNCTIONS, sweep
from .io import MODES, digest, read_json, timestamp, validate_raw, write_json
from .metrics import evaluate


def model_contract(metadata):
    contract = {field: metadata[field] for field in ("protocol", "scale", "engine", "weight_digests")}
    contract["metadata_sha256"] = metadata["dataset_digests"]["metadata.csv"]
    if "preprocessing_sha256" in metadata:
        contract["preprocessing_sha256"] = metadata["preprocessing_sha256"]
    if metadata["protocol"] == "end_to_end":
        contract["detector"] = {field: metadata["detector"][field] for field in ("model_id", "weight_sha256")}
        if "inference" in metadata["detector"]:
            contract["detector"]["inference"] = metadata["detector"]["inference"]
    return contract


def select_validation(paths, function="mean"):
    groups = {}
    for path in paths:
        raw = validate_raw(read_json(path))
        metadata = raw["metadata"]
        if metadata["split"] != "val":
            raise ValueError("Thresholds and preprocessing must be selected on val only")
        protocol = metadata["protocol"]
        mode = metadata["mode"]
        candidates = groups.setdefault(protocol, {})
        if mode in candidates:
            raise ValueError(f"Duplicate mode for {protocol}: {mode}")

        policy = sweep(raw["records"], function)
        candidates[mode] = {
            "mode": mode,
            "raw_path": str(Path(path).resolve()),
            "raw_sha256": digest(path),
            "contract": model_contract(metadata),
            "validation_csv_sha256": metadata["dataset_digests"]["ocr_val.csv"],
            "sample_ids": sorted(record["sample_id"] for record in raw["records"]),
            **policy,
            "metrics": evaluate(raw["records"], policy["selected_tau"], function),
        }

    if not groups:
        raise ValueError("Provide validation caches for at least one protocol")
    protocols = {}
    for protocol, candidates in groups.items():
        if set(candidates) != set(MODES):
            raise ValueError(f"Validation selection needs all three modes for {protocol}")
        reference = candidates[MODES[0]]
        for candidate in candidates.values():
            for field in ("contract", "validation_csv_sha256", "sample_ids"):
                if candidate[field] != reference[field]:
                    raise ValueError(f"Validation modes differ in {field}")

        best = max(
            candidates.values(),
            key=lambda candidate: (
                max(point["objective"] for point in candidate["curve"]),
                -candidate["selected_tau"],
                -MODES.index(candidate["mode"]),
            ),
        )
        protocols[protocol] = {
            "mode": best["mode"],
            "tau": best["selected_tau"],
            "score_function": function,
            "contract": best["contract"],
            "candidates": candidates,
        }

    return {
        "schema_version": 1,
        "selection_split": "val",
        "timestamp": timestamp(),
        "rule": "maximize accepted exact accuracy + coverage - 1; ties: lower tau, then mode order",
        "protocols": protocols,
    }


def load_policy(selection, protocol):
    if selection.get("schema_version") != 1 or selection.get("selection_split") != "val":
        raise ValueError("A validation-selected policy is required")
    policy = selection["protocols"][protocol]
    if policy["mode"] not in MODES or policy["score_function"] not in SCORE_FUNCTIONS:
        raise ValueError("Invalid selected mode or score function")
    if not 0.50 <= policy["tau"] <= 0.99:
        raise ValueError("Selected tau is outside the validation grid")
    return policy


def evaluate_test(raw, selection):
    validate_raw(raw)
    metadata = raw["metadata"]
    if metadata["split"] != "test":
        raise ValueError("Final evaluation requires a test cache")
    policy = load_policy(selection, metadata["protocol"])
    if metadata["mode"] != policy["mode"] or model_contract(metadata) != policy["contract"]:
        raise ValueError("Test model, dataset metadata, scale or mode differs from validation selection")
    return {
        "schema_version": 1,
        "metadata": {
            **metadata,
            "timestamp": timestamp(),
            "score_function": policy["score_function"],
            "tau": policy["tau"],
            "selection_timestamp": selection["timestamp"],
        },
        "metrics": evaluate(raw["records"], policy["tau"], policy["score_function"]),
    }


def plot_selection(selection, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7, 5))
    for protocol, policy in selection["protocols"].items():
        for mode, candidate in policy["candidates"].items():
            curve = candidate["curve"]
            axis.plot([p["coverage"] for p in curve], [p["accuracy"] for p in curve], label=f"{protocol}: {mode}")
    axis.set(xlabel="Coverage", ylabel="Exact accuracy among accepted plates", xlim=(0, 1), ylim=(0, 1))
    axis.legend(fontsize=8)
    figure.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    select = commands.add_parser("select")
    select.add_argument("--raw", type=Path, nargs="+", required=True)
    select.add_argument("--score-function", choices=SCORE_FUNCTIONS, default="mean")
    select.add_argument("--output", type=Path, required=True)
    test = commands.add_parser("test")
    test.add_argument("--raw", type=Path, required=True)
    test.add_argument("--selection", type=Path, required=True)
    test.add_argument("--output", type=Path, required=True)
    plot = commands.add_parser("plot")
    plot.add_argument("--selection", type=Path, required=True)
    plot.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "plot":
        plot_selection(read_json(args.selection), args.output)
    else:
        if args.output.exists():
            raise FileExistsError(f"Refusing to overwrite a saved selection/report: {args.output}")
        if args.command == "select":
            write_json(args.output, select_validation(args.raw, args.score_function))
        else:
            report = evaluate_test(read_json(args.raw), read_json(args.selection))
            report["metadata"].update(raw_sha256=digest(args.raw), selection_sha256=digest(args.selection))
            write_json(args.output, report)


if __name__ == "__main__":
    main()
