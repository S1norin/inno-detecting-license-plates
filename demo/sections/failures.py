"""Failure analysis: the six documented cases, reproduced with live inference.

Each case shows the real image, ground-truth box, live model output, and a
check against the documented observation. Detection cases use the canonical
conf 0.001 floor (the setting under which the documented observations were
made); recognition cases run the real CRNN on the appropriate crop.
"""

import streamlit as st

from data.preprocessing import preprocess_plate

from .. import results
from ..crnn import CRNNRecognizer
from ..dataset import load_image, load_metadata
from ..viz import COLOR_FINETUNED, COLOR_FROZEN, COLOR_GT, crop_box, draw_box, draw_detections
from ..yolo import get_detectors

CANONICAL_CONF = 0.001


@st.cache_resource
def _metadata():
    return load_metadata()


@st.cache_resource
def _crnn() -> CRNNRecognizer:
    return CRNNRecognizer()


def _find(sample_id: str):
    frame = _metadata()
    sub = frame[frame["image_path"].str.endswith(f"{sample_id}.jpg")]
    if sub.empty:
        return None
    from ..dataset import _row_to_sample

    return _row_to_sample(sub.iloc[0])


def _check(case: dict, live: dict) -> tuple[bool, str]:
    """Compare live inference against the documented observation."""
    if case["stage"] == "detection":
        key = case["detector"]
        n = len(live["dets"][key])
        if case["title"].startswith("Missed"):
            ok = n == 0
            return ok, (
                f"Live check: frozen detector produced {n} prediction(s) at conf 0.001 — "
                + ("matches the documented zero predictions." if ok else "differs from the documented observation.")
            )
        ok = n >= 1
        top = live["dets"][key][0].confidence if n else None
        detail = f"Live check: fine-tuned detector produced {n} box(es) at conf 0.001"
        if top is not None:
            detail += f", top confidence {top:.2f}"
        detail += " — " + (
            "consistent with the documented 1 true + low-confidence false positives."
            if ok
            else "differs from the documented observation."
        )
        return ok, detail
    # recognition
    pred = live["recog"].text
    documented_pred = case["observation"].split("predicted ")[1].split(";")[0].split(", ")[0]
    ok = pred == documented_pred
    return ok, (
        f"Live CRNN output `{pred}` vs documented `{documented_pred}` — "
        + ("identical (deterministic CPU inference, same checkpoint)."
           if ok
           else "differs; the documented value comes from the saved canonical cache.")
    )


def _run_case(case: dict, sample) -> dict:
    image = load_image(sample.image_path)
    detectors = get_detectors()
    live = {"dets": {}, "recog": None, "crop": None}
    for key in ("frozen", "finetuned"):
        live["dets"][key] = detectors[key].detect(image, conf=CANONICAL_CONF).detections
    if case["stage"] == "recognition":
        if case.get("protocol") == "ocr_only":
            crop = load_image(sample.crop_path)
        else:
            top = live["dets"]["finetuned"][0] if live["dets"]["finetuned"] else None
            crop = crop_box(image, (top.x1, top.y1, top.x2, top.y2)) if top else None
        live["crop"] = crop
        if crop is not None:
            live["recog"] = _crnn().recognize(preprocess_plate(crop, "none"))
    return live


def _show_case(case: dict):
    sample = _find(case["sample_id"])
    if sample is None:
        st.warning(f"Sample {case['sample_id']} not found in the metadata.")
        return

    with st.spinner("Running live inference for this case…"):
        live = _run_case(case, sample)
    ok, detail = _check(case, live)

    image = load_image(sample.image_path)
    st.columns(2)[0].markdown(
        f"**{case['title']}**  \n`{case['sample_id']}` — {sample.size_bucket} plate, "
        f"{sample.plate_pixel_height} px, tilt {sample.tilt_magnitude}°"
    )
    st.columns(2)[1].markdown(
        f"Ground truth: **`{sample.plate_text}`**"
    )

    c1, c2 = st.columns(2)
    shown = draw_box(image, sample.box, COLOR_GT, "GT")
    if case["stage"] == "detection":
        key = case["detector"]
        color = COLOR_FROZEN if key == "frozen" else COLOR_FINETUNED
        label = "frozen" if key == "frozen" else "fine-tuned"
        shown = draw_detections(shown, live["dets"][key], color, max_labels=5)
        c1.image(shown, width='stretch', caption=f"GT box + live {label} predictions (conf 0.001)")
    else:
        c1.image(shown, width='stretch', caption="GT box")
    crop = live.get("crop")
    if crop is not None:
        crop_caption = (
            "Crop fed to the CRNN (GT crop, OCR-only)"
            if case.get("protocol") == "ocr_only"
            else "Crop fed to the CRNN (top-1 detector box)"
        )
        c2.image(crop, width='stretch', caption=crop_caption)
    if live.get("recog") is not None:
        c2.markdown(f"**Live CRNN:** `{live['recog'].text}`  ")
        c2.caption("per-char conf: " + ", ".join(f"{v:.2f}" for v in live["recog"].char_confidences))

    (st.success if ok else st.warning)(detail)
    st.markdown(f"**Documented observation:** {case['observation']}")
    st.markdown(f"**Likely cause:** {case['cause']}")
    st.markdown(f"**Remaining limitation:** {case['limitation']}")


def render():
    st.markdown(
        """
# Failure Analysis

Six representative failures from the final report's test-split analysis.
Selecting a case **re-runs the real models on the real image** and checks the
live output against the documented observation. Failure analysis is part of
the research: it localizes the data gap (large, tilted plates) that both
stages share.
"""
    )
    options = [f"{i+1}. {c['title']} — `{c['sample_id']}`" for i, c in enumerate(results.FAILURE_CASES)]
    choice = st.radio("Failure case", options, label_visibility="collapsed")
    case = results.FAILURE_CASES[int(choice.split(".")[0]) - 1]
    _show_case(case)

    st.divider()
    st.markdown(
        """
**Cross-cutting pattern.** Detection failures concentrate on large + tilted
plates (the frozen model's blind spot, largely fixed by fine-tuning);
recognition failures concentrate on the same region, where the recognizer was
effectively untrained. Both point at the same data gap: the dataset is
dominated by small, near-0-degree plates.
"""
    )
