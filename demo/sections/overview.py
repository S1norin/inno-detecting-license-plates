"""Overview: research objective, architecture, dataset summary, key results."""

import streamlit as st

from .. import results


def render():
    st.markdown(
        """
# PlateLab — License Plate Detection & Recognition

**Task.** Given a traffic photograph, on the CCPD2019 benchmark subset:
- detect the license plate
- recognize its characters (Chinese province character, Latin letters, digits)

**Research questions.**
- How much does fine-tuning the detector help?
- How does a plate-specific recognizer compare to general-purpose OCR?
- Where does the two-stage system fail — and why?
"""
    )

    st.subheader("Data preparation")
    facts = results.DATASET_FACTS
    splits = facts["splits"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Train", f"{splits['train']:,} photos")
    c2.metric("Validation", f"{splits['val']:,} photos")
    c3.metric("Test", f"{splits['test']:,} photos")
    c4.metric("Scale variants", " / ".join(str(s) for s in facts["scales"]))

    st.markdown(
        """
Sampling is stratified by plate size and tilt angle, with quotas proportional
to stratum size, so the subset preserves the source distribution (height bins
≈ 75 / 96 / 127 px, tilt ≈ 16 / 90 / 92). Each photo is rendered at three
scales, giving 15,000 variants (10,500 / 2,100 / 2,400 across train /
validation / test). Splits are assigned before scale variation, so all
variants of one photo stay in the same split.
"""
    )

    st.subheader("Architecture")
    st.markdown(
        """
<div class="diagram-flow" role="img"
     aria-label="Photo goes to YOLO11s, then the detected plate crop goes to CRNN, producing text.">
  <span class="diagram-step">Traffic photo</span><span class="diagram-arrow" aria-hidden="true">→</span>
  <span class="diagram-step">YOLO11s<br><small>detect plate</small></span>
  <span class="diagram-arrow" aria-hidden="true">→</span>
  <span class="diagram-step">Top box<br><small>crop plate</small></span>
  <span class="diagram-arrow" aria-hidden="true">→</span>
  <span class="diagram-step">CRNN<br><small>read characters</small></span>
  <span class="diagram-arrow" aria-hidden="true">→</span>
  <span class="diagram-step">Recognized text</span>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown(
        """
| Stage | Model | Input | Output |
|---|---|---|---|
| Detection | YOLO11s (frozen or fine-tuned) | full photo, 640 px | boxes + confidences |
| Recognition | CRNN (CNN + BiLSTM + CTC, ~3 M params) | top-1 crop, 48×256 gray | text + per-char confidences |

The pipeline recognizes the **highest-confidence predicted box** per image.
Two evaluation protocols are kept separate throughout:
*ocr-only* (recognizer gets ground-truth crops) and *end-to-end*
(recognizer gets the detector's own crop).
"""
    )

    st.subheader("Results")
    st.caption(f"Test split, {splits['test']:,} original-scale images.")
    frozen = results.DETECTION_TEST["frozen"]
    finetuned = results.DETECTION_TEST["finetuned"]
    n_test = f"{splits['test']:,}"

    def row(config, det, rec):
        det_cells = "— | —" if det is None else f"{det['map50']:.4f} | {det['f1']:.4f}"
        return (
            f"| {config} | {det_cells} | {rec['char_acc']:.4f} "
            f"| {rec['exact']} / {n_test} |"
        )

    rows = [
        row("Frozen YOLO11s + EasyOCR", frozen, results.RECOGNITION_TEST[1]),
        row("CRNN, GT crops (OCR-only)", None, results.RECOGNITION_TEST[2]),
        row("CRNN + frozen YOLO11s", frozen, results.RECOGNITION_TEST[3]),
        row("CRNN + fine-tuned YOLO11s", finetuned, results.RECOGNITION_TEST[4]),
    ]
    table = (
        "| Config | mAP@0.5 | F1 | Char. acc. | Exact plates |\n"
        "|---|---|---|---|---|\n"
        + "\n".join(rows)
    )
    st.markdown(table)
    st.markdown("Try the live pipeline in the **Detection and Recognition** section.")
