"""Dataset: summary, prep workflow, sample browser, scale variants,
preprocessing ablation (real data/preprocessing.py implementation)."""

import math

import pandas as pd
import streamlit as st

from data.preprocessing import preprocess_plate

from ..dataset import Sample, _row_to_sample, load_image, load_metadata, thumbnail, variants_of
from ..viz import COLOR_GT, draw_box, gray_to_bgr

PER_PAGE = 12


@st.cache_resource
def _frame() -> pd.DataFrame:
    return load_metadata()


def _show_sample(sample: Sample):
    image = load_image(sample.image_path)
    if image is None:
        st.warning(f"Image file missing: {sample.image_path.name}")
        return
    st.image(draw_box(image, sample.box, COLOR_GT, "ground truth"), channels="BGR")
    crop = load_image(sample.crop_path)
    if crop is not None:
        st.image(crop, channels="BGR", caption="Ground-truth crop")


def render():
    st.markdown(
        """
# Dataset

The data team's pipeline: a 5,000-photo stratified selection from CCPD2019,
split **before** resizing so every scale variant of a photo stays in one
split (anti-leakage invariant), then YOLO labels and ground-truth plate crops
are exported for training.
"""
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Source photos", "5,000")
    c2.metric("Train / Val / Test", "3,500 / 700 / 800")
    c3.metric("Scale variants", "15,000")
    c4.metric("Split seed", "42")

    st.caption(
        "Size buckets (plate pixel height, train-derived): small ≤ 83 px, "
        "medium ≤ 112 px, large > 112 px. Tilt values are encoded in the CCPD "
        "file names, not independently calibrated angles."
    )

    st.subheader("How the dataset was prepared")
    st.markdown(
        """
1. **Selection** — proportional stratified sampling from CCPD2019 on bounding-box
   height and the two tilt values encoded in file names, preserving the source
   distribution. 8 byte-identical duplicates were replaced with unique images
   (replacement records kept).
2. **Splitting** — 70 / 14 / 16 % train / val / test, seed 42, done **before**
   resizing so scaled variants never cross splits.
3. **Variants** — each photo is exported at scales 1.0, 0.75, 0.5 (15,000 images).
4. **Annotations** — single-class normalized YOLO boxes (class 0 = license plate)
   plus ground-truth rectangular crops and OCR text labels.
5. **Validation** — data checks, metadata CSVs, and example visualizations are
   exported with every build; a SHA-256 receipt guards the source images.

This is a custom selection, not the official CCPD benchmark split.
"""
    )

    st.subheader("Samples")
    col_a, col_b = st.columns(2)
    split = col_a.selectbox("Split", ["test", "val", "train"], index=0)
    scale = col_b.selectbox("Scale", [1.0, 0.75, 0.5], index=0, format_func=lambda s: f"{s:g}")

    frame = _frame()
    sub = frame[(frame["split"] == split) & (frame["scale"] == scale)]
    if sub.empty:
        st.info("No samples in this split.")
        return

    n_pages = max(1, math.ceil(len(sub) / PER_PAGE))
    page = st.slider("Page", 1, n_pages, 1, key=f"page_{split}_{scale:g}")
    window = sub.iloc[(page - 1) * PER_PAGE : page * PER_PAGE]

    grid = st.columns(4)
    for i, (path, row) in enumerate(window.iterrows()):
        col = grid[i % 4]
        col.image(thumbnail(path), channels="BGR", width="stretch")
        if col.button(row["plate_text"], key=f"pick_{path}", help=path):
            st.session_state["picked"] = path
            st.rerun()

    picked = st.session_state.get("picked")
    if picked is not None and picked in sub.index:
        sample = _row_to_sample(sub.loc[picked])
    else:
        sample = _row_to_sample(window.iloc[0])

    meta = st.columns(6)
    meta[0].markdown(f"**{sample.split}**\n`{sample.sample_id}`")
    meta[1].markdown(f"scale **{sample.scale:g}**")
    meta[2].markdown(f"plate **{sample.plate_pixel_height} px** ({sample.size_bucket})")
    meta[3].markdown(f"tilt **{sample.tilt_magnitude}°** (h {sample.tilt_horizontal} / v {sample.tilt_vertical})")
    meta[4].markdown(f"image **{sample.width}×{sample.height}**")
    meta[5].markdown(f"text **`{sample.plate_text}`**")

    _show_sample(sample)

    st.subheader("Scale variants of the same photo")
    st.caption("All variants inherit the source photo's split.")
    variants = variants_of(frame, sample.parent_id)
    ordered = sorted(variants.values(), key=lambda v: v.scale, reverse=True)
    cols = st.columns([v.scale for v in ordered])
    for col, s in zip(cols, ordered, strict=True):
        img = load_image(s.image_path)
        if img is not None:
            col.image(draw_box(img, s.box, COLOR_GT, f"scale {s.scale:g}"), channels="BGR")
            col.caption(f"{s.width}×{s.height}, plate {s.plate_pixel_height} px ({s.size_bucket})")

    st.subheader("Preprocessing ablation")
    st.caption(
        "Three experimental alternatives from `data/preprocessing.py` — "
        "`none`, grayscale + CLAHE (clip 2.0, 8×8), Canny (100/200). They are "
        "alternatives, not a sequential pipeline; the CRNN always applies its "
        "own 48×256 input transform afterwards."
    )
    crop = load_image(sample.crop_path)
    if crop is not None:
        cols = st.columns(3)
        for col, mode in zip(cols, ["none", "gray_contrast", "canny"], strict=True):
            out = preprocess_plate(crop, mode)
            col.image(gray_to_bgr(out), width=200, channels="BGR", caption=mode)
