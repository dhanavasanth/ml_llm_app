"""DiabetiCare — multi-page Streamlit app.

    streamlit run main.py

Pages
  1. Home              — what the app is and who it's for (this file)
  2. Diabetes Tester   — diabetic.py
  3. Prescriber Chatbot— llm.py
"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="DiabetiCare",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

META_PATH = Path(__file__).parent / "models" / "model_metadata.json"


def home() -> None:
    st.title("🩺 DiabetiCare")
    st.subheader("Screen your diabetes risk, then talk it through with an AI prescriber.")

    st.markdown(
        """
DiabetiCare turns a clinical blood panel into two things a patient can actually use: a **predicted
one-year diabetes progression score** with a clear diabetic / not-diabetic call, and a **conversation**
about what to eat, how to move, and which treatments a doctor would typically weigh up for that result.
"""
    )

    st.divider()

    st.header("How it works")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### 1️⃣ Enter your numbers")
        st.write(
            "Ten routine measurements — age, sex, BMI, blood pressure and six serum readings — "
            "on the **Diabetes Tester** page."
        )
    with c2:
        st.markdown("#### 2️⃣ Get a score and a class")
        st.write(
            "A linear-regression model trained on the scikit-learn diabetes dataset predicts a "
            "progression score, then bands it into low risk, borderline or high risk."
        )
    with c3:
        st.markdown("#### 3️⃣ Ask the prescriber")
        st.write(
            "The **Prescriber Chatbot** reads your result and answers questions about diet, "
            "exercise, monitoring and medication classes — grounded in your own numbers."
        )

    st.divider()

    st.header("Who it's for")
    left, right = st.columns(2)
    with left:
        st.markdown(
            """
- **Patients and carers** wanting a plain-language read on a blood panel.
- **Clinic front desks** doing a first-pass triage before a consultation.
- **Wellness programmes** tracking how lifestyle changes move the score.
"""
        )
    with right:
        st.markdown(
            """
- **ML learners** seeing an end-to-end path from a toy dataset to a deployed app.
- **Educators** demonstrating regression, thresholding and LLM grounding in one place.
- **Developers** wanting a compact template for scikit-learn + Streamlit + Gemini.
"""
        )

    st.divider()

    st.header("Under the hood")
    tech, model_card = st.columns([1, 1])
    with tech:
        st.markdown(
            """
| Layer | Choice |
|---|---|
| Dataset | `sklearn.datasets.load_diabetes` (442 patients, unscaled) |
| Model | `StandardScaler` → `LinearRegression` pipeline |
| Training | `training.ipynb`, artefacts in `models/` |
| UI | Streamlit multi-page app |
| Chatbot | Google **Gemini 2.5** via `google-genai` |
"""
        )
    with model_card:
        if META_PATH.exists():
            meta = json.loads(META_PATH.read_text())
            m = meta["metrics"]
            a, b, c = st.columns(3)
            a.metric("Test R²", f"{m['r2']:.3f}")
            b.metric("RMSE", f"{m['rmse']:.1f}")
            c.metric("Class accuracy", f"{meta['classification_accuracy']:.1%}")
            st.caption(
                f"Trained on {meta['trained_on_samples']} patients. A predicted score of "
                f"**{meta['threshold']:.0f}** or above is classified diabetic."
            )
        else:
            st.warning(
                "No trained model found. Run every cell of **training.ipynb** to create "
                "`models/diabetes_model.joblib`.",
                icon="⚠️",
            )

    st.divider()

    st.header("Get started")
    g1, g2 = st.columns(2)
    g1.page_link("diabetic.py", label="**Run the Diabetes Tester**", icon="🧪")
    g2.page_link("llm.py", label="**Open the Prescriber Chatbot**", icon="💬")
    st.caption("Tip: run the tester first — the chatbot grounds its advice in that result.")

    st.divider()
    st.warning(
        "**Medical disclaimer.** DiabetiCare is an educational demo built on a public toy dataset. "
        "Its score is not a diagnosis, its chatbot does not issue prescriptions, and neither "
        "replaces a qualified clinician.",
        icon="⚕️",
    )


pages = [
    st.Page(home, title="Home", icon="🏠", default=True),
    st.Page("diabetic.py", title="Diabetes Tester", icon="🧪"),
    st.Page("llm.py", title="Prescriber Chatbot", icon="💬"),
]

with st.sidebar:
    st.markdown("### 🩺 DiabetiCare")
    st.caption("Predict · Classify · Prescribe")

st.navigation(pages).run()
