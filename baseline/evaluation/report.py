"""Deterministic REPORT.md and summary figure generated from the run data.

Both artifacts are pure functions of (table, failures, timing): rerunning on
the same inputs with the same timestamp reproduces them byte for byte.
"""

from .table import ABLATION_TAU


def _fmt(value):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _metrics_table(overall):
    lines = ["| metric | value |", "| --- | --- |"]
    for key in ("image_count", "ground_truth_count", "tp", "fp", "fn",
                "map50", "count", "accepted", "coverage",
                "character_accuracy", "plate_exact_match",
                "accepted_exact_accuracy", "precision", "recall", "f1"):
        if key in overall:
            lines.append(f"| {key} | {_fmt(overall[key])} |")
    return "\n".join(lines)


def _size_tilt_row(name, row):
    return f"| {name} | {_fmt(row['map50'])} | {_fmt(row['precision'])} | {_fmt(row['recall'])} | {_fmt(row['f1'])} |"


def _detection_section(table):
    lines = ["## E1 — Detection (YOLO)", ""]
    header = "| bucket | map50 | precision | recall | f1 |"
    sub_header = "| --- | --- | --- | --- | --- |"
    for split in ("val", "test"):
        section = table["e1_detection"][split]
        lines += [f"### {split}", "", _metrics_table(section["overall"]), ""]
        lines += ["**by size bucket**", "", header, sub_header]
        for bucket, row in section["by_size"].items():
            lines.append(_size_tilt_row(bucket, row))
        lines += ["", "**by tilt bin**", "", header, sub_header]
        for bin_name, row in section["by_tilt"].items():
            lines.append(_size_tilt_row(bin_name, row))
        lines.append("")
    return lines


def _ocr_section(title, section, with_delta):
    lines = [f"## {title}", ""]
    selected = section["selected"]
    selected_note = (f"Selected on validation: mode `{selected['mode']}`, tau {selected['tau']}, "
                     f"score `{selected['score_function']}`.")
    lines.append(selected_note)
    lines += ["", "### val", "", _metrics_table(section["val"]["overall"]), ""]
    lines += ["### test", "", _metrics_table(section["test"]["overall"]), ""]
    if with_delta:
        lines += ["**delta vs. OCR-only (e2e - ocr_only)**", "",
                  "| split | char_acc | exact | acc_exact | coverage |",
                  "| --- | --- | --- | --- | --- |"]
        for split in ("val", "test"):
            delta = section["delta_vs_ocr_only"][split]
            lines.append(f"| {split} | {_fmt(delta['character_accuracy'])} | {_fmt(delta['plate_exact_match'])} "
                         f"| {_fmt(delta['accepted_exact_accuracy'])} | {_fmt(delta['coverage'])} |")
        lines.append("")
    return lines


def _ablation_section(table):
    lines = [f"## E3 — Preprocessing ablation (OCR-only, val, tau={ABLATION_TAU})", "",
             "| mode | char_acc | exact | acc_exact | coverage | d_char | d_exact | d_cov |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for mode, section in table["e3_ablation"].items():
        overall = section["overall"]
        delta = section["delta_vs_none"]
        if delta is None:
            delta_text = ["-", "-", "-"]
        else:
            delta_text = [_fmt(delta["character_accuracy"]),
                          _fmt(delta["plate_exact_match"]),
                          _fmt(delta["coverage"])]
        row = (f"| {mode} | {_fmt(overall['character_accuracy'])} | {_fmt(overall['plate_exact_match'])} "
               f"| {_fmt(overall['accepted_exact_accuracy'])} | {_fmt(overall['coverage'])} "
               f"| {delta_text[0]} | {delta_text[1]} | {delta_text[2]} |")
        lines.append(row)
    lines.append("")
    return lines


def _threshold_section(table):
    lines = ["## E6 — Accuracy/coverage at the selected threshold (test)", "",
             "| protocol | tau | accuracy | coverage |", "| --- | --- | --- | --- |"]
    for protocol, row in table["e6_threshold"].items():
        lines.append(f"| {protocol} | {_fmt(row['tau'])} | {_fmt(row['accuracy'])} | {_fmt(row['coverage'])} |")
    lines.append("")
    return lines


def _failure_section(failures):
    lines = ["## Failure cases", ""]
    for split in ("val", "test"):
        cases = failures["detection"][split]
        lines += [f"### Detection — {split} ({len(cases)} cases)", ""]
        if not cases:
            lines.append("No failures in this split.")
        for case in cases:
            details = case["details"]
            lines.append(f"- `{case['image_path']}` — {', '.join(case['tags'])} "
                         f"(gt={details['ground_truth_count']}, pred={details['prediction_count']}, "
                         f"best_iou={_fmt(details['best_iou'])}, best_score={_fmt(details['best_score'])})")
        lines.append("")
    for protocol in ("ocr_only", "end_to_end"):
        section = failures["recognition"][protocol]
        header = (f"### Recognition — {protocol} ({section['mode']}, tau={section['tau']}; "
                  f"{len(section['cases'])} cases)")
        lines += [header, ""]
        if not section["cases"]:
            lines.append("No failures.")
        for case in section["cases"]:
            details = case["details"]
            lines.append(f"- `{case['sample_id']}` — {', '.join(case['tags'])}: "
                         f"gt=`{details['ground_truth']}` text=`{details['text']}` (status={details['status']})")
        lines.append("")
    return lines


def _timing_section(timing):
    lines = ["## Timing probe", "",
             f"n={timing['n']}, seed={timing['seed']}, warmup excluded per stage.", "",
             "| stage | count | median ms | p95 ms | mean ms | total s | max RSS MB |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for name, stats in timing["stages"].items():
        lines.append(f"| {name} | {stats['count']} | {_fmt(stats['median_ms'])} | {_fmt(stats['p95_ms'])} "
                     f"| {_fmt(stats['mean_ms'])} | {_fmt(stats['total_s'])} | {_fmt(stats['max_rss_mb_after'])} |")
    lines.append("")
    return lines


def build_report(table_doc, failures, timing):
    table = table_doc["table"]
    inputs = table_doc["inputs"]
    lines = [
        "# Baseline evaluation report",
        "",
        f"Run: `{table_doc['run_id']}`  ",
        f"Generated: {table_doc['timestamp']}",
        "",
        "## Inputs",
        "",
        f"- dataset: `{inputs['dataset']['path']}` (val {inputs['dataset']['sample_count']['val']}, "
        f"test {inputs['dataset']['sample_count']['test']})",
        f"- detection: `{inputs['detection']['path']}` model `{inputs['detection']['model']}`",
        f"- ocr: `{inputs['ocr']['path']}` engine `{inputs['ocr']['engine']}`",
        "",
    ]
    lines += _detection_section(table)
    lines += _ocr_section("E2 — OCR-only (GT crops)", table["e2_ocr_only"], with_delta=False)
    lines += _ocr_section("E2b — End-to-end (predicted boxes)", table["e2b_end_to_end"], with_delta=True)
    lines += _ablation_section(table)
    lines += _threshold_section(table)
    lines += _failure_section(failures)
    if timing:
        lines += _timing_section(timing)
    lines += [
        "## Notes",
        "",
        "- Detection and OCR metrics come from the frozen stage results; nothing was retrained or re-evaluated here.",
        "- Recognition tilt bins are reported on validation only; the frozen test reports predate",
        "  per-sample tilt records.",
        "- Failure cases are ranked deterministically; recognition cases come from validation so the test split is",
        "  untouched.",
    ]
    if timing:
        lines.append(f"- The timing probe measured {timing['n']} sampled images (seed {timing['seed']}), "
                     "not the full split; full-split timing is tracked as a follow-up.")
    else:
        lines.append("- Run without `--no-probe` to add the sampled timing probe.")
    lines.append("")
    return "\n".join(lines)


def make_figure(table_doc, path):
    """Write the summary figure; returns False when matplotlib is unavailable."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)

    table = table_doc["table"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    ax = axes[0][0]
    metrics = ("map50", "precision", "recall", "f1")
    splits = ("val", "test")
    width = 0.35
    for i, split in enumerate(splits):
        values = [table["e1_detection"][split]["overall"][m] for m in metrics]
        ax.bar([x + i * width for x in range(len(metrics))], values, width, label=split)
    ax.set_xticks([x + width / 2 for x in range(len(metrics))], metrics)
    ax.set_ylim(0, 1.1)
    ax.set_title("Detection (YOLO)")
    ax.legend()

    ax = axes[0][1]
    modes = list(table["e3_ablation"])
    char_acc = [table["e3_ablation"][m]["overall"]["character_accuracy"] for m in modes]
    exact = [table["e3_ablation"][m]["overall"]["accepted_exact_accuracy"] for m in modes]
    ax.bar([x - 0.2 for x in range(len(modes))], char_acc, 0.4, label="character accuracy")
    ax.bar([x + 0.2 for x in range(len(modes))], exact, 0.4, label="accepted exact accuracy")
    ax.set_xticks(range(len(modes)), modes)
    ax.set_ylim(0, 1.1)
    ax.set_title("Preprocessing ablation (val, tau=0.5)")
    ax.legend()

    ax = axes[1][0]
    protocols = ("e2_ocr_only", "e2b_end_to_end")
    labels = ["OCR-only", "End-to-end"]
    for key, label in zip(protocols, labels, strict=True):
        section = table[key]
        val = section["val"]["overall"]
        test = section["test"]["overall"]
        ax.plot([val["coverage"], test["coverage"]],
                [val["accepted_exact_accuracy"], test["accepted_exact_accuracy"]],
                marker="o", label=label)
        ax.scatter([val["coverage"]], [val["accepted_exact_accuracy"]])
    ax.set_xlabel("coverage")
    ax.set_ylabel("accepted exact accuracy")
    ax.set_title("Coverage vs accuracy (selected tau)")
    ax.legend()

    ax = axes[1][1]
    buckets = list(table["e1_detection"]["test"]["by_size"])
    values = [table["e1_detection"]["test"]["by_size"][b]["f1"] for b in buckets]
    ax.bar(buckets, values)
    ax.set_ylim(0, 1.1)
    ax.set_title("Detection F1 by size bucket (test)")

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True
