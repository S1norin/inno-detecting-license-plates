"""Team contributions and reproducibility view."""

import platform

import streamlit as st

from ..config import DIGEST_CRNN, DIGEST_FINETUNED, DIGEST_FROZEN, WEIGHT_CRNN, WEIGHT_FINETUNED, WEIGHT_FROZEN

CONTRIBUTIONS = [
    ("Andrei Polevoi", "Data & Preprocessing",
     "Dataset selection and stratification, split construction, duplicate and "
     "leakage checks, label and crop export, preprocessing module and ablations."),
    ("Maximilian Mifsud Bonici", "Detection",
     "YOLO11s baseline, fine-tuning on the train split, detection metrics, detection failure analysis."),
    ("Evgenii Khovalyg", "Recognition",
     "CRNN recognizer (CNN + BiLSTM + CTC), EasyOCR baseline comparison, recognition evaluation."),
    ("Mikhail Krylov", "Pipeline & Evaluation",
     "Pipeline integration, confidence policy, evaluation harness, runtime "
     "measurements, sensitivity analysis, reporting and demo coordination."),
]


def _weight_status(path, digest):
    from ..weights_io import sha256_of

    try:
        actual = sha256_of(path)
        mark = "✅ verified" if actual == digest else "⚠️ digest mismatch"
    except FileNotFoundError:
        mark = "❌ missing"
    return mark


def render():
    st.markdown("# Team & Reproducibility")

    st.subheader("Contributions (per the project plan)")
    for name, role, scope in CONTRIBUTIONS:
        with st.expander(f"{name} — {role}"):
            st.write(scope)

    st.subheader("Reproducibility")
    st.markdown(
        """
- **Source repository:** `github.com/S1norin/inno-detecting-license-plates`
  (this deployment runs from that checkout).
- **Weights:** Hugging Face `Sinorin/inno-detecting-license-plates-weights`
  (our fine-tuned YOLO11s + CRNN) and `morsetechlab/yolov11-license-plate-detection`
  (frozen YOLO11s). All weights are SHA-256 pinned in `baseline/common/models.json`
  and checked when models are loaded or this page is rendered.
- **Canonical results:** `docs/md/final_results.md` (2026-11-02) — the single
  source for every metric shown in Experiments.
- **Protocols:** `ocr_only` = ground-truth crops; `end_to_end` = detector's
  top-1 crop. Selection (preprocessing mode, confidence τ) on validation;
  test run once.
- **Dataset:** 5,000 CCPD2019 photos → 15,000 variants, split before resize,
  seed 42; buckets train-derived. Custom selection, not the official split.
"""
    )

    st.markdown("**Weight digests on this machine (checked at render time):**")
    for label, path, digest in (
        ("Frozen YOLO11s", WEIGHT_FROZEN, DIGEST_FROZEN),
        ("Fine-tuned YOLO11s", WEIGHT_FINETUNED, DIGEST_FINETUNED),
        ("CRNN (epoch 39)", WEIGHT_CRNN, DIGEST_CRNN),
    ):
        st.markdown(f"- {label}: `{str(path).replace(str(path.parents[2]), '')}` — {_weight_status(path, digest)}")

    st.markdown(
        f"""
**Running environment (live):** Python {platform.python_version()},
{platform.system()} {platform.release()}, torch/ultralytics/streamlit as
installed in the project venv. Detection and recognition run on CPU with the
canonical settings (imgsz 640, NMS IoU 0.7, single-threaded CRNN).
"""
    )

    st.caption(
        "Historical results (Experiments section) were measured on an Apple M3 "
        "and an RTX 3090 Ti per the final report; live predictions in the "
        "Pipeline Inspector and Failure Analysis run on this server."
    )
