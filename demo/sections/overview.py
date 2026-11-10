"""Overview: research objective, architecture, dataset summary, key results."""

import streamlit as st

from .. import results


def render():
    st.markdown(
        """
# PlateLab — License Plate Detection & Recognition

**Task.** Given a traffic photograph, (1) detect the license plate and (2)
recognize its characters (Chinese province character, Latin letters, digits)
on the CCPD2019 benchmark subset.

**Research questions.** How much does fine-tuning the detector help? How does a
plate-specific recognizer compare to general-purpose OCR? Where does the
two-stage system fail — and why?
"""
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Source photos", "5,000")
    c2.metric("Image variants", "15,000")
    c3.metric("Test split", "800 images")
    c4.metric("Final plate F1 (test)", f"{results.RECOGNITION_TEST[-1]['f1']:.3f}")

    st.subheader("Two-stage architecture")
    st.markdown(
        """
| Stage | Model | Input | Output |
|---|---|---|---|
| **1 — Detection** | YOLO11s (frozen or fine-tuned) | full photo, 640 px | boxes + confidences |
| **2 — Recognition** | CRNN (CNN + BiLSTM + CTC, ~3 M params) | top-1 crop, 48×256 gray | text + per-char confidences |

The pipeline recognizes the **highest-confidence predicted box** per image.
Two evaluation protocols are kept separate throughout:
*ocr-only* (recognizer gets ground-truth crops) and *end-to-end*
(recognizer gets the detector's own crop).
""")

    st.subheader("Headline results (test split, 800 images)")
    st.markdown(
        f"""
- Detection: fine-tuned YOLO11s mAP@0.5 **{results.DETECTION_TEST['finetuned']['map50']:.4f}**
  vs frozen **{results.DETECTION_TEST['frozen']['map50']:.4f}** (recall
  {results.DETECTION_TEST['frozen']['recall']:.3f} → {results.DETECTION_TEST['finetuned']['recall']:.3f}).
- Recognition: CRNN end-to-end character accuracy **{results.RECOGNITION_TEST[-1]['char_acc']:.4f}**
  and **{results.RECOGNITION_TEST[-1]['exact']}/800** exact plates, vs
  EasyOCR baseline **{results.RECOGNITION_TEST[0]['char_acc']:.4f}** / 0 exact.

All numbers: {results.SOURCE}. Try the live pipeline in the
**Pipeline Inspector** section.
"""
    )
