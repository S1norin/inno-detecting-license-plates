"""Live Pipeline Inspector: real YOLO + CRNN/EasyOCR inference on a selected image."""

import time

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from data.preprocessing import preprocess_plate

from ..crnn import CRNNRecognizer
from ..dataset import Sample, _row_to_sample, load_image, load_metadata, variants_of
from ..easy_ocr import EasyOCRReader
from ..viz import COLOR_FINETUNED, COLOR_FROZEN, crop_box, draw_detections, gray_to_bgr
from ..yolo import YOLODetector, get_detectors

ENGINES = ("CRNN", "EasyOCR")
SCALES = (1.0, 0.75, 0.5)


@st.cache_resource
def _frame() -> pd.DataFrame:
    return load_metadata()


@st.cache_resource
def _crnn() -> CRNNRecognizer:
    return CRNNRecognizer()


@st.cache_resource
def _easyocr() -> EasyOCRReader:
    return EasyOCRReader()


def _pick_sample(split: str, scale: float, query: str) -> Sample | None:
    frame = _frame()
    sub = frame[(frame["split"] == split) & (frame["scale"] == scale)]
    if query:
        matches = sub["image_path"].str.contains(query, case=False, na=False, regex=False)
        matches |= sub["plate_text"].str.contains(query, case=False, na=False, regex=False)
        sub = sub[matches]
    if sub.empty:
        return None
    return _row_to_sample(sub.iloc[0])


def _reader(engine: str):
    if engine == "EasyOCR":
        try:
            return _easyocr()
        except Exception as exc:
            st.error(
                f"EasyOCR is unavailable on this server ({type(exc).__name__}: {exc}). "
                "Install it with `uv pip install -p .venv-baseline/bin/python easyocr==1.7.2` "
                "and restart the plate-demo service."
            )
            return None
    return _crnn()


def _top_crop(image, det_result):
    top = det_result.detections[0] if det_result.detections else None
    crop = crop_box(image, (top.x1, top.y1, top.x2, top.y2)) if top else None
    return top, crop


def _run_all(image, detectors: dict[str, YOLODetector], reader, conf: float, mode: str) -> dict:
    """One real inference pass: both detectors + the selected OCR engine on each top-1 crop."""
    started = time.perf_counter()
    out = {"dets": {}, "tops": {}, "crops": {}, "processed": {}, "recs": {}}
    for key, det in detectors.items():
        result = det.detect(image, conf=conf)
        out["dets"][key] = result
        top, crop = _top_crop(image, result)
        out["tops"][key] = top
        out["crops"][key] = crop
        processed = preprocess_plate(crop, mode) if crop is not None else None
        out["processed"][key] = processed
        out["recs"][key] = reader.recognize(processed) if processed is not None else None
    out["total_ms"] = (time.perf_counter() - started) * 1000.0
    return out


def _scale_images(sample: Sample | None, image, uploaded: bool) -> dict[float, np.ndarray]:
    """The same photo at all three scales (dataset variants, or live resizes for uploads)."""
    if uploaded:
        h, w = image.shape[:2]
        out = {}
        for s in SCALES:
            if s == 1.0:
                out[s] = image
            else:
                out[s] = cv2.resize(image, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
        return out
    if sample is None:
        return {}
    variants = variants_of(_frame(), sample.parent_id)
    out = {}
    for s in SCALES:
        variant = variants.get(s)
        if variant is None:
            continue
        img = load_image(variant.image_path)
        if img is not None:
            out[s] = img
    return out


def _show_reading(col, rec, engine: str, sample: Sample | None, crop, processed, mode: str,
                  top, label: str):
    col.markdown(f"**{label}**")
    if top is None:
        col.warning("No detection at this confidence floor — nothing to recognize.")
        return
    if crop is not None:
        if processed is not None and mode != "none":
            c_raw, c_proc = col.columns(2)
            c_raw.image(crop, width=170, channels="BGR",
                        caption=f"Predicted crop (conf {top.confidence:.2f})")
            c_proc.image(gray_to_bgr(processed), width=170, channels="BGR",
                         caption=f"After `{mode}` — what the {engine} sees")
        else:
            col.image(crop, width=200, channels="BGR",
                      caption=f"Predicted crop (conf {top.confidence:.2f})")
    if rec is not None:
        col.markdown(f"**{engine} reads:** `{rec.text}`" + ("" if rec.text else " *(empty)*"))
        if sample is not None:
            exact = rec.text == sample.plate_text
            col.markdown(f"Ground truth: `{sample.plate_text}` — " + ("✅ exact" if exact else "❌ mismatch"))
        col.caption(
            "per-char conf: " + (", ".join(f"{v:.2f}" for v in rec.char_confidences) or "—")
            + f" · {rec.elapsed_ms:.1f} ms"
        )


def _run_all_scales(images: dict[float, np.ndarray], detectors, reader, conf: float, mode: str) -> dict:
    """Run the fine-tuned detector + the selected engine on every scale of the photo."""
    out = {}
    for s, img in images.items():
        result = detectors["finetuned"].detect(img, conf=conf)
        top, crop = _top_crop(img, result)
        processed = preprocess_plate(crop, mode) if crop is not None else None
        rec = reader.recognize(processed) if processed is not None else None
        out[s] = {"image": img, "dets": result, "top": top, "rec": rec}
    return out


def render():
    st.markdown(
        """
# Live Pipeline Inspector

Real inference on the real checkpoints (SHA-256 verified at load). Everything
under **Run** is a live model prediction; ground-truth annotations are shown
separately and labeled. This demo displays the raw reading and does not apply
the historical confidence gate or abstention policy.

The OCR engine selectbox lets you compare the presentation configurations:
**EasyOCR** (the general-purpose baseline) vs **CRNN** (the plate-specific
recognizer), each reading the crops of both detectors.
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
        sample = None
        raw = uploaded.getvalue()
        if len(raw) > 8 * 1024 * 1024:
            st.error("Upload too large (max 8 MB).")
            return
        image = _decode_upload(raw)
        if image is None:
            st.error("Could not decode the uploaded image.")
            return
        st.image(image, width=560, channels="BGR", caption=f"Uploaded: {uploaded.name} (no ground truth)")
    elif sample is not None:
        image = load_image(sample.image_path)
        if image is None:
            st.warning(f"Image file missing: {sample.image_path.name}")
            return
        st.image(image, width=560, channels="BGR",
                 caption=f"{sample.sample_id} — {sample.split}, scale {sample.scale:g}, "
                         f"plate {sample.plate_pixel_height} px, text `{sample.plate_text}`")
    else:
        st.info("No sample matches the filter.")
        return

    st.subheader("2 — Detection")
    col_d, col_e, col_f, col_g = st.columns([3, 1, 2, 2])
    conf = col_d.slider("Detection confidence threshold", 0.01, 0.9, 0.25, 0.01)
    engine = col_e.selectbox("OCR engine", list(ENGINES), index=0,
                             help="EasyOCR: general-purpose baseline OCR. CRNN: the plate-specific recognizer.")
    mode = col_f.selectbox("OCR preprocessing (experimental mode)",
                           ["none", "gray_contrast", "canny"], index=0,
                           help="Applied to the crop before OCR; see the ablation in Experiments.")
    run = col_g.button("Run inference", type="primary")
    st.caption(
        "Canonical metrics used conf 0.001 for mAP and fixed settings "
        "(imgsz 640, NMS IoU 0.7, CPU). This threshold changes live detection "
        "and the crop sent to OCR, but not the reported numbers."
    )

    if not run:
        st.info("Press **Run inference** to execute the real models.")
        return

    reader = _reader(engine)
    if reader is None:
        return

    with st.spinner(f"Running YOLO + {engine}…"):
        detectors = get_detectors()
        res = _run_all(image, detectors, reader, conf, mode)

    st.subheader("Detectors side by side")
    c1, c2 = st.columns(2)
    for col, key, color, label in (
        (c1, "frozen", COLOR_FROZEN, "Frozen YOLO11s (pretrained)"),
        (c2, "finetuned", COLOR_FINETUNED, "Fine-tuned YOLO11s"),
    ):
        det_res = res["dets"][key]
        col.markdown(f"**{label}** — {len(det_res.detections)} box(es), {det_res.elapsed_ms:.0f} ms")
        col.image(
            draw_detections(image, det_res.detections, color),
            width=340,
            channels="BGR",
        )
        if not det_res.detections:
            col.warning("No detection at this confidence floor.")

    st.subheader(f"3 — Crops & readings ({engine})")
    c1, c2 = st.columns(2)
    _show_reading(c1, res["recs"]["frozen"], engine, sample, res["crops"]["frozen"],
                  res["processed"]["frozen"], mode, res["tops"]["frozen"],
                  "Frozen YOLO11s crop")
    _show_reading(c2, res["recs"]["finetuned"], engine, sample, res["crops"]["finetuned"],
                  res["processed"]["finetuned"], mode, res["tops"]["finetuned"],
                  "Fine-tuned YOLO11s crop")
    if engine == "EasyOCR":
        st.caption(
            "EasyOCR runs the canonical offline baseline engine (easyocr 1.7.2, ch_sim + en, "
            "pinned offline weights) — a fresh live measurement on this server, not the "
            "historical canonical row shown in Experiments."
        )

    top = res["tops"]["finetuned"]
    crop = res["crops"]["finetuned"]
    if sample is not None and top is not None:
        gt_crop = load_image(sample.crop_path)
        if gt_crop is not None:
            st.image(gt_crop, width=220, channels="BGR",
                     caption="Ground-truth crop (from the annotation box)")
    elif sample is None:
        st.caption("No ground truth for uploaded images.")

    if crop is not None:
        st.markdown("**Preprocessing modes on the fine-tuned crop:**")
        cols = st.columns(3)
        for col, m in zip(cols, ["none", "gray_contrast", "canny"], strict=True):
            col.image(gray_to_bgr(preprocess_plate(crop, m)), width=200, channels="BGR", caption=m)

    st.subheader("4 — Same photo, all scales (fine-tuned models)")
    scale_run = st.button("Run on scales 1.0 / 0.75 / 0.5")
    if scale_run:
        images = _scale_images(sample, image, uploaded is not None)
        if not images:
            st.warning("No scale variants available for this image.")
        else:
            with st.spinner(f"Running the fine-tuned pipeline on {len(images)} scales…"):
                scale_res = _run_all_scales(images, detectors, reader, conf, mode)
            cols = st.columns(len(scale_res))
            for col, s in zip(cols, sorted(scale_res, reverse=True), strict=True):
                entry = scale_res[s]
                col.markdown(f"**Scale {s:g}**")
                col.image(
                    draw_detections(entry["image"], entry["dets"].detections, COLOR_FINETUNED),
                    width=240,
                    channels="BGR",
                )
                if entry["rec"] is not None:
                    col.markdown(f"`{entry['rec'].text}`" + ("" if entry["rec"].text else " *(empty)*"))
                    if sample is not None:
                        col.caption(f"GT: `{sample.plate_text}`")
                else:
                    col.warning("No box at this scale.")
            st.caption(
                "Live fine-tuned YOLO11s + "
                f"{engine} on the same photo at three scales"
                + (" (dataset scale variants)." if sample is not None else " (live resizes of your upload).")
            )

    st.subheader("5 — What happened")
    total = res["total_ms"]
    rec = res["recs"]["finetuned"]
    lines = [
        f"- **Detection:** fine-tuned detector returned {len(res['dets']['finetuned'].detections)} box(es) "
        f"(frozen: {len(res['dets']['frozen'].detections)}); top box confidence "
        f"{top.confidence:.2f}, {res['dets']['finetuned'].elapsed_ms:.0f} ms." if top is not None else
        "- **Detection:** no fine-tuned box at this confidence floor.",
        f"- **Recognition:** {engine} read `{rec.text}` "
        f"({rec.elapsed_ms:.1f} ms) with preprocessing `{mode}`." if rec is not None else
        f"- **Recognition:** {engine} had no valid crop.",
    ]
    if sample is not None and rec is not None:
        ok = rec.text == sample.plate_text
        lines.append(
            f"- **Against ground truth** (`{sample.plate_text}`): "
            + ("exact match." if ok else "mismatch — see the failure analysis section.")
        )
    lines.append(f"- **Total live runtime on this machine:** {total:.0f} ms (CPU, both detectors + {engine}).")
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
