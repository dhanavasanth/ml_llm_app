"""Diabetes tester page — predicts the progression score and the diabetic class.

Runs either standalone (`streamlit run diabetic.py`) or as a page inside main.py.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_DIR = Path(__file__).parent / "models"
MODEL_PATH = MODEL_DIR / "diabetes_model.joblib"
META_PATH = MODEL_DIR / "model_metadata.json"

try:
    st.set_page_config(page_title="Diabetes Tester", page_icon="🧪", layout="wide")
except Exception:
    pass  # already configured by main.py


@st.cache_resource(show_spinner="Loading model…")
def load_artifacts():
    if not MODEL_PATH.exists() or not META_PATH.exists():
        return None, None
    return joblib.load(MODEL_PATH), json.loads(META_PATH.read_text())


def classify(score: float, meta: dict) -> dict:
    for band in meta["bands"]:
        if band["max"] is None or score < band["max"]:
            return band
    return meta["bands"][-1]


def band_style(band: dict) -> tuple[str, str]:
    """Return (emoji, colour) for a risk band."""
    return {
        "Low risk": ("🟢", "#2e7d32"),
        "Borderline": ("🟡", "#ed6c02"),
        "High risk": ("🔴", "#c62828"),
    }.get(band["name"], ("⚪", "#555555"))


model, meta = load_artifacts()

st.title("🧪 Diabetes Progression Tester")
st.caption(
    "Enter the patient's clinical measurements to predict the one-year diabetes "
    "progression score and its risk classification."
)

if model is None:
    st.error(
        "Model artefacts not found. Run every cell of **training.ipynb** first — "
        "it writes `models/diabetes_model.joblib` and `models/model_metadata.json`."
    )
    st.stop()

# ── sidebar: model card ────────────────────────────────────────────────────────
with st.sidebar:
    st.subheader("Model card")
    st.write(f"**Estimator** · {meta['model_type']}")
    st.write(f"**Dataset** · {meta['dataset']}")
    st.write(f"**Samples** · {meta['trained_on_samples']}")
    m = meta["metrics"]
    st.metric("Test R²", f"{m['r2']:.3f}")
    st.metric("RMSE", f"{m['rmse']:.1f}")
    st.metric("Class accuracy", f"{meta['classification_accuracy']:.1%}")
    st.caption(
        f"Diabetic when the predicted progression score ≥ **{meta['threshold']:.0f}** "
        "(a teaching heuristic on a toy dataset, not a clinical cut-off)."
    )

# ── input form ─────────────────────────────────────────────────────────────────
features = meta["features"]

with st.form("patient_form"):
    st.subheader("Patient measurements")

    left, right = st.columns(2)
    values: dict[str, float] = {}

    for i, feat in enumerate(features):
        col = left if i % 2 == 0 else right
        name, label = feat["name"], feat["label"]

        if name == "sex":
            choice = col.radio(
                label, ["Male", "Female"],
                index=0, horizontal=True, key="in_sex",
            )
            values[name] = 1.0 if choice == "Male" else 2.0
        else:
            # widen the slider slightly beyond the training range so real inputs fit
            span = feat["max"] - feat["min"]
            values[name] = col.slider(
                label,
                min_value=round(feat["min"] - 0.1 * span, 1),
                max_value=round(feat["max"] + 0.1 * span, 1),
                value=float(feat["mean"]),
                step=float(feat["step"]),
                key=f"in_{name}",
            )

    submitted = st.form_submit_button("Predict", type="primary", width="stretch")

# ── prediction ─────────────────────────────────────────────────────────────────
if submitted:
    row = pd.DataFrame([[values[f["name"]] for f in features]],
                       columns=[f["name"] for f in features])
    score = float(model.predict(row)[0])
    band = classify(score, meta)

    st.session_state["diagnosis"] = {
        "score": round(score, 1),
        "band": band["name"],
        "diabetic": bool(band["diabetic"]),
        "threshold": meta["threshold"],
        "inputs": {f["label"]: values[f["name"]] for f in features},
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }

if "diagnosis" in st.session_state:
    d = st.session_state["diagnosis"]
    band = next(b for b in meta["bands"] if b["name"] == d["band"])
    emoji, colour = band_style(band)

    st.divider()
    st.subheader("Result")

    c1, c2, c3 = st.columns(3)
    c1.metric("Progression score", f"{d['score']:.1f}",
              delta=f"{d['score'] - meta['threshold']:+.1f} vs threshold",
              delta_color="inverse")
    c2.metric("Risk band", f"{emoji} {d['band']}")
    c3.metric("Classification", "Diabetic" if d["diabetic"] else "Not diabetic")

    lo, hi = meta["target_min"], meta["target_max"]
    st.progress(min(max((d["score"] - lo) / (hi - lo), 0.0), 1.0))
    st.caption(f"Scale: {lo:.0f} (lowest progression) → {hi:.0f} (highest progression)")

    if d["diabetic"]:
        st.error(
            f"**Diabetic — high progression risk.** The predicted score of "
            f"{d['score']:.1f} is above the {d['threshold']:.0f} threshold.",
            icon="🔴",
        )
    elif d["band"] == "Borderline":
        st.warning(
            f"**Not diabetic, but borderline.** A score of {d['score']:.1f} sits in the "
            "pre-diabetic band — worth monitoring.",
            icon="🟡",
        )
    else:
        st.success(
            f"**Not diabetic — low risk.** A score of {d['score']:.1f} is in the low band.",
            icon="🟢",
        )

    with st.expander("Values used for this prediction"):
        st.dataframe(
            pd.DataFrame(d["inputs"].items(), columns=["Measurement", "Value"]),
            hide_index=True, width="stretch",
        )

    st.info(
        "➡️ Head to the **Prescriber Chatbot** page for diet and medication guidance "
        "tailored to this result.",
        icon="💬",
    )

st.caption(
    "⚕️ Educational demo built on a scikit-learn toy dataset. Not a medical device and "
    "not a substitute for professional diagnosis."
)
