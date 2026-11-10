"""Experiments: detection/OCR comparisons, ablations, scale robustness,
size/tilt sensitivity, confidence policy. All values from canonical saved
artifacts (docs/md/final_results.md), labeled by split and protocol."""

import pandas as pd
import streamlit as st

from .. import results


def render():
    st.markdown(
        f"""
# Experiments & Model Comparison

Every value below is a **saved historical measurement** from
`{results.SOURCE}` — not a live computation. Splits and protocols are labeled
on each block; validation and test numbers are never mixed.
"""
    )

    st.subheader("Detection: frozen vs fine-tuned YOLO11s (test, 800 images)")
    df = pd.DataFrame(
        [
            {"Detector": "Frozen YOLO11s", **{k.capitalize(): v for k, v in results.DETECTION_TEST["frozen"].items()}},
            {
                "Detector": "Fine-tuned YOLO11s",
                **{k.capitalize(): v for k, v in results.DETECTION_TEST["finetuned"].items()},
            },
        ]
    ).set_index("Detector")
    st.dataframe(df, width='stretch')
    st.markdown(
        """
The large gain comes from **recall** (0.894 → 0.998): the frozen detector
produced no predictions for 51 test images; the fine-tuned one removed those
complete misses. Precision stays roughly flat.
"""
    )

    st.subheader("Recognition: baseline vs plate-specific CRNN (test, 800 plates, mode `none`, τ = 0.50)")
    df = (
        pd.DataFrame(results.RECOGNITION_TEST)[["engine", "protocol", "char_acc", "exact", "f1"]]
        .set_index(["engine", "protocol"])
    )
    df.columns = ["Char accuracy", "Exact plates / 800", "Plate F1"]
    st.dataframe(df, width='stretch')
    st.caption(
        "EasyOCR rows are the frozen historical baseline from an earlier "
        "environment (not a fresh same-environment rerun). `ocr_only` uses "
        "ground-truth crops; `end_to_end` uses the detector's own top-1 crop. "
        "Improving the detector raised end-to-end accuracy (0.671 → 0.766)."
    )

    st.subheader("Preprocessing ablation (validation char accuracy)")
    st.bar_chart(pd.Series(results.PREPROCESSING_VAL), height=220)
    st.markdown(
        "Selected on validation: `none` wins. Canny removes the glyph "
        "information the CRNN needs (0.4575). These are validation numbers — "
        "the selection policy, not test results."
    )

    st.subheader("Scale robustness (test, same weights, no retraining)")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Detection mAP@0.5 by scale**")
        det = pd.DataFrame(
            {
                "Frozen": [results.SCALE_DETECTION["frozen"][s][0] for s in (1.0, 0.75, 0.5)],
                "Fine-tuned": [results.SCALE_DETECTION["finetuned"][s][0] for s in (1.0, 0.75, 0.5)],
            },
            index=[f"{s:g}" for s in (1.0, 0.75, 0.5)],
        )
        st.line_chart(det, height=240)
    with c2:
        st.markdown("**CRNN char accuracy by scale**")
        scales = (1.0, 0.75, 0.5)
        rec = pd.DataFrame(
            {
                "OCR-only": [results.SCALE_RECOGNITION["ocr_only"][s][0] for s in scales],
                "E2E + fine-tuned": [
                    results.SCALE_RECOGNITION["end_to_end + fine-tuned"][s][0] for s in scales
                ],
                "E2E + frozen": [
                    results.SCALE_RECOGNITION["end_to_end + frozen"][s][0] for s in scales
                ],
            },
            index=[f"{s:g}" for s in scales],
        )
        st.line_chart(rec, height=240)
    st.caption(
        "Both models were evaluated at all three scales with identical weights "
        "— controlled scale experiments, not separate training runs. The CRNN "
        "resizes every crop to 48×256, so absolute scale is largely normalized "
        "away; the per-size-bucket quality loss at 0.5 comes from plate pixels, "
        "not model retraining."
    )

    st.subheader("Size & tilt sensitivity (CRNN, test, OCR-only)")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**By plate size bucket** (char accuracy / exact rate)")
        size = pd.DataFrame(
            {
                "Char accuracy": [results.CRNN_OCRONLY_SCOPES["size"][k][0] for k in ("small", "medium", "large")],
                "Exact rate": [results.CRNN_OCRONLY_SCOPES["size"][k][1] for k in ("small", "medium", "large")],
            },
            index=["small", "medium", "large"],
        )
        st.bar_chart(size, height=240)
    with c2:
        st.markdown("**By tilt bin** (char accuracy / exact rate)")
        tilt = pd.DataFrame(
            {
                "Char accuracy": [results.CRNN_OCRONLY_SCOPES["tilt"][k][0] for k in ("0-9", "10-19", "20+")],
                "Exact rate": [results.CRNN_OCRONLY_SCOPES["tilt"][k][1] for k in ("0-9", "10-19", "20+")],
            },
            index=["0-9°", "10-19°", "20+°"],
        )
        st.bar_chart(tilt, height=240)
    st.caption(
        "Large plates and the 10–19° tilt bin are the dominant recognition "
        "failures; detection is weakest in the same tilt bin for both detectors."
    )

    st.subheader("Confidence policy")
    st.markdown(
        """
The recognizer emits a per-character confidence (max probability in each
collapsed CTC run). A policy of threshold **τ = 0.50** — selected on the 700
validation samples by accepted-exact-accuracy + coverage − 1 — accepts a
reading only when all character confidences are above τ, otherwise the system
abstains. The test split was then run **once** under that policy.

A single-image confidence value shown in the Pipeline Inspector is a live
per-sample measurement; it does not reproduce this test-set policy curve.
"""
    )

    st.subheader("Runtime (historical: Apple M3, CPU, single-threaded)")
    rt = results.RUNTIME_M3
    frozen_det = f"{rt['frozen']['detection_median_ms']} / {rt['frozen']['detection_p95_ms']} ms"
    finetuned_det = f"{rt['finetuned']['detection_median_ms']} / {rt['finetuned']['detection_p95_ms']} ms"
    st.markdown(
        f"""
| Stage | Frozen | Fine-tuned |
|---|---:|---:|
| Detection median / p95 | {frozen_det} | {finetuned_det} |
| CRNN median | {rt['crnn_median_ms']} ms | {rt['crnn_median_ms']} ms |
| Full end-to-end | ~{rt['end_to_end_ms']} ms | ~{rt['end_to_end_ms']} ms |

These are historical measurements on the stated hardware. Live runtimes on
this server are shown in the Pipeline Inspector and labeled as such.
"""
    )
