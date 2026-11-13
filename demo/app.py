"""PlateLab — interactive research demo for the CCPD2019 license plate project.

Run from the repository root:

    .venv/bin/python -m streamlit run demo/app.py
"""

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from demo.sections import dataset_explorer, experiments, failures, overview, pipeline, team  # noqa: E402

st.set_page_config(
    page_title="PlateLab — License Plate Detection & Recognition",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
html, body, [data-testid="stBlockContainer"] {
    background-color: #0e1117;
    color: #e6e9ef;
}
h1, h2, h3 { color: #f2f5fa; font-weight: 650; }
h1 { border-bottom: 2px solid #2b6cb0; padding-bottom: 0.3rem; }
.stMetric { background: #171c26; border: 1px solid #2a3242; border-radius: 10px; padding: 10px 14px; }
.stMetric label { color: #9aa7bd !important; }
.stCaption, [data-testid="stCaptionContainer"] { color: #8b96ab; }
div[data-testid="stSidebar"] { background-color: #12161f; }
.stButton > button { background-color: #2b6cb0; color: white; border: none; border-radius: 8px; }
.stButton > button:hover { background-color: #3d82c9; }
.stAlert { border-radius: 8px; }
[data-testid="stExpander"] { background: #171c26; border: 1px solid #2a3242; border-radius: 8px; }
.diagram-flow { display: flex; flex-wrap: wrap; align-items: center; gap: 0.5rem; margin: 1rem 0; }
.diagram-step { background: #171c26; border: 1px solid #40516c; border-radius: 8px;
    padding: 0.6rem 0.8rem; font-size: 0.95rem; }
.diagram-arrow { color: #9aa7bd; font-size: 1.2rem; }
[data-testid="stImage"] img {
    border-radius: 6px;
    max-height: min(55vh, 480px);
    width: auto;
    max-width: 100%;
    object-fit: contain;
}
</style>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## PlateLab")
    st.caption("License plate detection & recognition on CCPD2019")
    section = st.radio(
        "Navigate",
        [
            "Overview",
            "Dataset Explorer",
            "Pipeline Inspector",
            "Experiments",
            "Failure Analysis",
            "Team & Reproducibility",
        ],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption(
        "A university CV project demo. Real checkpoints, real data, "
        "documented results. Not a production ALPR system."
    )

if section == "Overview":
    overview.render()
elif section == "Dataset Explorer":
    dataset_explorer.render()
elif section == "Pipeline Inspector":
    pipeline.render()
elif section == "Experiments":
    experiments.render()
elif section == "Failure Analysis":
    failures.render()
elif section == "Team & Reproducibility":
    team.render()
