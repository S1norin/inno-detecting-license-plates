"""Live Pipeline Inspector: real YOLO + CRNN inference on a selected image."""

import time

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from data.preprocessing import preprocess_plate

from ..crnn import CRNNRecognizer
from ..dataset import Sample, _row_to_sample, load_image, load_metadata
from ..viz import COLOR_FINETUNED, COLOR_FROZEN, crop_box, draw_detections, gray_to_bgr
from ..yolo import YOLODetector, get_detectors


@st.cache_resource
def _frame() -> pd.DataFrame:
    return load_metadata()


@st.cache_resource
def _crnn() -> CRNNRecognizer:
    return CRNNRecognizer()


def _pick_sample(split: str, scale: float, query: str) -> Sample | None:
    frame = _frame()
    sub = frame[(frame["split"] == split) & (frame["scale"] == scale)]
    if query:
        sub = sub[sub["image_path"].str.contains(query, case=False, na=False)]
    if sub.empty:
        return None
    return _row_to_sample(sub.iloc[0])


def _run_all(image, detectors: dict[str, YOLODetector], crnn: CRNNRecognizer,
             conf: float, mode: str) -> dict:
    """One real inference pass: both detectors + CRNN on the finetuned top-1."""
    started = time.perf_counter()
    out = {}
    for key, det in detectors.items():
        out[key] = det.detect(image, conf=conf)
    top_key = "finetuned"
    top = out[top_key].detections[0] if out[top_key].detections else None
    crop = crop_box(image, (top.x1, top.y1, top.x2, top.y2)) if top else None
    recognition = crnn.recognize(preprocess_plate(crop, mode)) if crop is not None else None
    out["top_key"] = top_key
    out["top"] = top
    out["crop"] = crop
    out["recognition"] = recognition
    out["total_ms"] = (time.perf_counter() - started) * 1000.0
    return out


def render():
    st.markdown(
        """
# Live Pipeline Inspector

Real inference on the real checkpoints (SHA-256 verified at load). Everything
under **Run** is a live model prediction; ground-truth annotations are shown
separately and labeled.
"""
    )

    st.subheader("1 — Input")
    col_a, col_b, col_c = st.columns([2, 1, 2])
    split = col_a.selectbox("Split", ["test", "val", "train"], index=0)
    scale = col_b.selectbox("Scale", [1.0, 0.75, 0.5], index=0, format_func=lambda s: f"{s:g}")
    query = col_c.text_input("Sample ID / plate text filter", "6f3c772c")

    uploaded = st.file_uploader(
        "…or upload your own photo (ground truth unavailable)",
        type=["jpg", "jpeg", "png", "webp"],
    )

    sample: Sample | None = _pick_sample(split, scale, query.strip())
    image = None
    if uploaded is not None:
        raw = uploaded.getvalue()
        if len(raw) > 8 * 1024 * 1024:
            st.error("Upload too large (max 8 MB).")
            return
        image = _decode_upload(raw)
        if image is None:
            st.error("Could not decode the uploaded image.")
            return
        st.image(image, width=560, caption=f"Uploaded: {uploaded.name} (no ground truth)")
    elif sample is not None:
        image = load_image(sample.image_path)
        if image is None:
            st.warning(f"Image file missing: {sample.image_path.name}")
            return
        st.image(image, width=560,
                 caption=f"{sample.sample_id} — {sample.split}, scale {sample.scale:g}, "
                         f"plate {sample.plate_pixel_height} px, text `{sample.plate_text}`")
    else:
        st.info("No sample matches the filter.")
        return

    st.subheader("2 — Detection")
    col_d, col_e, col_f = st.columns([3, 1, 2])
    conf = col_d.slider("Confidence floor (display only)", 0.01, 0.9, 0.25, 0.01)
    mode = col_f.selectbox("OCR preprocessing (experimental mode)",
                           ["none", "gray_contrast", "canny"], index=0,
                           help="Applied to the crop before the CRNN; see the ablation in Experiments.")
    run = col_e.button("Run inference", type="primary")
    st.caption(
        "Canonical metrics used conf 0.001 for mAP and fixed settings "
        "(imgsz 640, NMS IoU 0.7, CPU). Moving this slider changes only the "
        "boxes you see here, not the reported numbers."
    )

    if not run:
        st.info("Press **Run inference** to execute the real models.")
        return

    with st.spinner("Running YOLO + CRNN…"):
        detectors = get_detectors()
        res = _run_all(image, detectors, _crnn(), conf, mode)

    st.subheader("Detectors side by side")
    c1, c2 = st.columns(2)
    for col, key, color, label in (
        (c1, "frozen", COLOR_FROZEN, "Frozen YOLO11s (pretrained)"),
        (c2, "finetuned", COLOR_FINETUNED, "Fine-tuned YOLO11s"),
    ):
        det_res = res[key]
        col.markdown(f"**{label}** — {len(det_res.detections)} box(es), {det_res.elapsed_ms:.0f} ms")
        col.image(
            draw_detections(image, det_res.detections, color),
            width=340,
        )
        if not det_res.detections:
            col.warning("No detection at this confidence floor.")

    st.subheader("3 — Crop (top-1 from the fine-tuned detector)")
    top = res["top"]
    if top is None:
        st.warning("The fine-tuned detector found no box — nothing to recognize.")
        return
    crop = res["crop"]
    c1, c2 = st.columns(2)
    c1.image(crop, width='content',
             caption=f"Predicted crop, conf {top.confidence:.2f} (this is what the CRNN sees)")
    if sample is not None:
        gt_crop = load_image(sample.crop_path)
        if gt_crop is not None:
            c2.image(gt_crop, width='content',
                     caption="Ground-truth crop (from the annotation box)")
    else:
        c2.caption("No ground truth for uploaded images.")

    st.subheader("4 — Preprocessing ablation (experimental alternatives)")
    cols = st.columns(3)
    for col, mode in zip(cols, ["none", "gray_contrast", "canny"], strict=True):
        col.image(gray_to_bgr(preprocess_plate(crop, mode)), width=200, caption=mode)

    st.subheader("5 — CRNN recognition")
    recog = res["recognition"]
    if recog is not None:
        c1, c2, c3 = st.columns([2, 1, 1])
        c1.markdown(f"**Predicted:** `{recog.text}`" + ("" if recog.text else " *(empty)*"))
        if sample is not None:
            exact = recog.text == sample.plate_text
            c2.markdown(f"**Ground truth:** `{sample.plate_text}`")
            c3.markdown("**Exact match:** " + ("✅" if exact else "❌"))
        c1.caption(
            "Per-character confidences: "
            + (", ".join(f"{v:.2f}" for v in recog.char_confidences) or "—")
            + f" · {recog.elapsed_ms:.1f} ms · greedy CTC decoding"
        )

    st.subheader("6 — What happened")
    total = res["total_ms"]
    lines = [
        f"- **Detection:** fine-tuned detector returned {len(res['finetuned'].detections)} box(es) "
        f"(frozen: {len(res['frozen'].detections)}); top box confidence "
        f"{top.confidence:.2f}, {res['finetuned'].elapsed_ms:.0f} ms.",
        f"- **Recognition:** CRNN read `{recog.text}` "
        f"({recog.elapsed_ms:.1f} ms) with preprocessing `{mode}`.",
    ]
    if sample is not None:
        ok = recog.text == sample.plate_text
        lines.append(
            f"- **Against ground truth** (`{sample.plate_text}`): "
            + ("exact match." if ok else "mismatch — see the failure analysis section.")
        )
    lines.append(f"- **Total live runtime on this machine:** {total:.0f} ms (CPU, both detectors + CRNN).")
    st.markdown("\n".join(lines))

    st.divider()
    st.caption(
        "Live runtime measured on this server (CPU). The historical benchmark in "
        "Experiments is from a single-threaded Apple M3 run."
    )


def _decode_upload(raw: bytes):
    arr = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return image
