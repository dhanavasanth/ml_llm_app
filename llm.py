"""Prescriber chatbot page — Gemini 2.5 acting as a diabetes-care advisor.

Reads the latest prediction from the Diabetes Tester page (st.session_state["diagnosis"])
and grounds every answer in that result.

Runs either standalone (`streamlit run llm.py`) or as a page inside main.py.
"""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

GEMINI_MODELS = ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.5-flash-lite"]

try:
    st.set_page_config(page_title="Prescriber Chatbot", page_icon="💬", layout="wide")
except Exception:
    pass  # already configured by main.py


PLACEHOLDERS = {"", "your_gemini_api_key_here", "your_key_here", "changeme"}


def get_api_key() -> str | None:
    """Return a usable key, treating the .env placeholder as 'not configured'."""
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        try:
            key = st.secrets.get("GEMINI_API_KEY")
        except Exception:
            key = None
    if key is None or key.strip().lower() in PLACEHOLDERS:
        return None
    return key.strip()


@st.cache_resource(show_spinner=False)
def get_client(api_key: str):
    from google import genai

    return genai.Client(api_key=api_key)


def build_persona(diagnosis: dict | None) -> str:
    persona = """You are Dr. Gemini, an experienced diabetologist and clinical nutritionist.
You advise a single patient whose diabetes risk has just been assessed by a machine-learning
screening tool.

How you respond:
- Speak directly to the patient — warm, plain language, no jargon dumps. Expand any clinical term you use.
- Ground every answer in the patient's specific screening result below. Reference their numbers.
- Structure advice under clear headings, typically: Diet, Physical activity, Monitoring,
  Medication considerations, When to see a doctor. Skip headings that don't apply to the question.
- For food guidance be concrete: name meals, portion sizes and swaps, and account for common
  Indian/South-Asian staples unless the patient says otherwise.
- For medication, explain the *classes* a doctor typically considers (e.g. metformin as first line,
  SGLT2 inhibitors, GLP-1 agonists) and what each does — never issue a prescription, a dose, or tell
  the patient to start or stop anything on their own. Always route the final decision to their
  treating physician.
- Keep answers focused: around 250-400 words unless the patient asks for more.
- Close anything clinical with a one-line reminder that this is educational guidance, not a diagnosis.
- If the patient describes red-flag symptoms (severe hypoglycaemia, chest pain, confusion, vomiting,
  very high blood sugar), tell them to seek urgent medical care first, before any other advice.
"""

    if diagnosis:
        readings = "\n".join(f"  - {k}: {v}" for k, v in diagnosis["inputs"].items())
        persona += f"""
--- SCREENING RESULT FOR THIS PATIENT (from the Diabetes Tester, {diagnosis['timestamp']}) ---
Predicted one-year diabetes progression score: {diagnosis['score']} (scale 25-346)
Risk band: {diagnosis['band']}
Classification: {"DIABETIC — high progression risk" if diagnosis['diabetic']
                 else "NOT DIABETIC at the screening threshold"}
Threshold used: a score of {diagnosis['threshold']:.0f} or above is flagged diabetic.

Measurements the patient entered:
{readings}
--- END OF SCREENING RESULT ---

Open the conversation by acknowledging this result in one or two sentences, then answer what is asked.
The score comes from a linear-regression model on the scikit-learn toy diabetes dataset, so treat it as
a screening signal rather than a diagnosis, and say so if the patient over-reads it.
"""
    else:
        persona += """
The patient has not run the Diabetes Tester yet. Give general, safe diabetes-prevention guidance and
invite them to run the screening on the Diabetes Tester page for tailored advice.
"""
    return persona


def stream_reply(client, model_name: str, system_instruction: str, history: list[dict]):
    """Stream Gemini's answer. `history` ends with the user turn to answer."""
    from google.genai import types

    prior = [
        types.Content(
            role="model" if m["role"] == "assistant" else "user",
            parts=[types.Part.from_text(text=m["content"])],
        )
        for m in history[:-1]
    ]
    chat = client.chats.create(
        model=model_name,
        history=prior,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.6,
            max_output_tokens=2048,
        ),
    )
    for chunk in chat.send_message_stream(history[-1]["content"]):
        if chunk.text:
            yield chunk.text


# ── page ───────────────────────────────────────────────────────────────────────
st.title("💬 Diabetes Prescriber Chatbot")
st.caption("Gemini 2.5, acting as a diabetes-care advisor for your screening result.")

diagnosis = st.session_state.get("diagnosis")
api_key = get_api_key()

with st.sidebar:
    st.subheader("Chat settings")
    model_name = st.selectbox("Gemini model", GEMINI_MODELS, index=0)
    if st.button("Clear conversation", width="stretch"):
        st.session_state.pop("chat_history", None)
        st.rerun()

    st.divider()
    st.subheader("Your screening result")
    if diagnosis:
        st.metric("Progression score", f"{diagnosis['score']:.1f}")
        st.write(f"**Band** · {diagnosis['band']}")
        st.write(f"**Class** · {'Diabetic' if diagnosis['diabetic'] else 'Not diabetic'}")
        st.caption(f"Tested {diagnosis['timestamp']}")
    else:
        st.info("No result yet — run the Diabetes Tester page first.")

if not api_key:
    st.error(
        "No Gemini API key found. Add `GEMINI_API_KEY=your_key_here` to the **.env** file in "
        "the project root, then restart the app.\n\n"
        "Get a key at https://aistudio.google.com/apikey",
        icon="🔑",
    )
    st.stop()

if not diagnosis:
    st.warning(
        "You haven't run a prediction yet, so advice will be generic. Visit the "
        "**Diabetes Tester** page to get guidance tailored to your numbers.",
        icon="🧪",
    )

client = get_client(api_key)
history = st.session_state.setdefault("chat_history", [])

if not history:
    starters = [
        "What should my daily meal plan look like?",
        "Which medications would a doctor typically consider for me?",
        "What lifestyle changes will move my score the most?",
        "Explain my result in simple terms.",
    ]
    st.write("**Try asking:**")
    cols = st.columns(2)
    for i, s in enumerate(starters):
        if cols[i % 2].button(s, width="stretch", key=f"starter_{i}"):
            st.session_state["pending_prompt"] = s
            st.rerun()

for msg in history:
    with st.chat_message(msg["role"], avatar="⚕️" if msg["role"] == "assistant" else None):
        st.markdown(msg["content"])

prompt = st.chat_input("Ask about diet, medication or lifestyle…")
prompt = st.session_state.pop("pending_prompt", None) or prompt

if prompt:
    history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="⚕️"):
        try:
            reply = st.write_stream(
                stream_reply(client, model_name, build_persona(diagnosis), history)
            )
            history.append({"role": "assistant", "content": reply})
        except Exception as exc:  # network / quota / bad key
            st.error(f"Gemini request failed: {exc}")
            history.pop()

st.caption(
    "⚕️ Educational guidance generated by an AI model. It is not a prescription and does not "
    "replace your doctor."
)
