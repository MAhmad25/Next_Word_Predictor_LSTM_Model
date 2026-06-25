from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import joblib
import numpy as np
import streamlit as st
import os
import logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.ERROR)


st.set_page_config(
    page_title="Next Word Predictor",
    page_icon="✦",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">

<style>
:root {
  --primary:         #f54e00;
  --primary-active:  #d04200;
  --ink:             #26251e;
  --body:            #5a5852;
  --body-strong:     #26251e;
  --muted:           #807d72;
  --muted-soft:      #a09c92;
  --hairline:        #e6e5e0;
  --hairline-soft:   #efeee8;
  --hairline-strong: #cfcdc4;
  --canvas:          #f7f7f4;
  --canvas-soft:     #fafaf7;
  --surface-card:    #ffffff;
  --surface-strong:  #e6e5e0;
  --on-primary:      #ffffff;
  --timeline-done:   #c08532;
  --semantic-error:  #cf2d56;
  --font-display: 'Inter', system-ui, 'Helvetica Neue', Arial, sans-serif;
  --font-mono:    'JetBrains Mono', 'Fira Code', monospace;
}

html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
  background-color: #ffffff !important;
  font-family: var(--font-display) !important;
  color: var(--ink) !important;
}

/* gradient div injected via st.markdown */
.nwp-gradient {
  position: fixed;
  left: 50%;
  width: 280vw;
  height: 280vw;
  top: calc(34svh + 85vw);
  clip-path: inset(0 0 33.33% 0);
  filter: blur(40px);
  background-image: radial-gradient(
    130vw,
    rgba(0,0,0,0) 54.8%,
    rgb(245, 78, 0,0.5) 63%,
    rgb(245, 78, 0,0.2) 76%,
    rgb(66,144,217) 95%,
    rgb(28,8,67) 99%,
    rgba(0,0,0,0) 100%
  );
  transform: translate(-50%, -50%) scale(1);
  z-index: 0;
  pointer-events: none;
}

#MainMenu, footer, header,
[data-testid="stDecoration"],
[data-testid="stToolbar"] { visibility: hidden; display: none; }

.block-container {
  max-width: 760px !important;
  padding: 0 24px 80px !important;
  margin: 0 auto !important;
}
             .center{
      display: flex;
  flex-direction: column;
  justify-content: center; 
  align-items: center;
            width:100%;
            }

/* hero */
.hero-eyebrow {
  font-size: 11px; font-weight: 600; letter-spacing: 0.88px;
  text-transform: uppercase; color: var(--muted); margin-bottom: 16px;
}
.hero-title {
  font-size: 56px; font-weight: 400; line-height: 1.1; width: 100%;
  letter-spacing: -1.68px; color: var(--ink); 
}
            .txt{
            color: var(--primary);
            }
.accent { color: var(--primary); font-weight: 600;
  background: rgba(245,78,0,0.07); border-radius: 4px; padding: 1px 5px; }
.hero-sub {
  font-size: 16px; line-height: 1.6; color: var(--ink);
  max-width: 520px; margin-bottom: 48px;
}

/* textarea */
[data-testid="stTextArea"] textarea {
  font-family: var(--font-mono) !important;
  font-size: 14px !important;
  background: var(--canvas-soft) !important;
  border: 1px solid var(--hairline-strong) !important;
  border-radius: 8px !important;
  color: var(--ink) !important;
  caret-color: var(--primary) !important;
  padding: 14px 16px !important;
  line-height: 1.6 !important;
  resize: vertical !important;
}
[data-testid="stTextArea"] textarea:focus {
  border-color: var(--primary) !important;
  box-shadow: 0 0 0 3px rgba(245,78,0,0.08) !important;
}
[data-testid="stTextArea"] label {
  font-size: 13px !important; font-weight: 600 !important;
  color: var(--body-strong) !important; margin-bottom: 8px !important;
}

/* button */
[data-testid="stButton"] > button {
  background: var(--primary) !important;
  color: var(--on-primary) !important;
  font-family: var(--font-display) !important;
  font-size: 14px !important; font-weight: 500 !important;
  border-radius: 8px !important; border: none !important;
  padding: 10px 24px !important; height: 44px !important;
  transition: background 0.15s ease !important; width: 100% !important;
}
[data-testid="stButton"] > button:hover,
[data-testid="stButton"] > button:active {
  background: var(--primary-active) !important;
}

/* timeline */
.timeline-row {
  display: flex; align-items: center; gap: 8px;
  flex-wrap: wrap; margin-bottom: 16px;
}
.tpill {
  font-size: 11px; font-weight: 600; letter-spacing: 0.88px;
  text-transform: uppercase; border-radius: 9999px;
  padding: 4px 12px; color: var(--ink);
}
.tpill.thinking { background: #dfa88f; }
.tpill.reading  { background: #9fbbe0; }
.tpill.editing  { background: #c0a8dd; }
.tpill.done     { background: var(--timeline-done); color: #fff; }

/* result */
.result-card {
  background: var(--surface-card); border-radius: 12px;
  border: 1px solid var(--hairline); padding: 28px; margin-top: 24px;
}
.result-title {
  font-size: 13px; font-weight: 600; letter-spacing: 0.88px;
  text-transform: uppercase; color: var(--muted); margin-bottom: 20px;
}
.result-text {
  font-family: var(--font-mono); font-size: 15px; line-height: 1.7;
  color: var(--ink); background: var(--canvas-soft); border-radius: 8px;
  padding: 18px 20px; border: 1px solid var(--hairline-soft);
  word-break: break-word;
}
.result-text .predicted-word {
  color: var(--primary); font-weight: 600;
  background: rgba(245,78,0,0.07); border-radius: 4px; padding: 1px 5px;
}

/* footer */
.cursor-footer {
  margin-top: 80px; padding-top: 32px;
  border-top: 1px solid var(--hairline);
  display: flex; align-items: center; justify-content: space-between;
}
.footer-left  { font-size: 13px; color: var(--muted); }
.footer-right {
  font-size: 11px; font-weight: 600; letter-spacing: 0.88px;
  text-transform: uppercase; color: var(--muted-soft);
}
</style>
""", unsafe_allow_html=True)

# ── gradient div ──────────────────────────────────────────────────────────────
st.markdown('<div class="nwp-gradient"></div>', unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_artifacts():
    return load_model("model.keras"), joblib.load("tokenizer.pkl")


WINDOW_SIZE = 6
TEMPERATURE = 0.3

try:
    model, tokenizer = load_artifacts()
    vocab_size = len(tokenizer.word_index) + 1
    model_loaded = True
except Exception as e:
    model_loaded = False
    load_error = str(e)


def predict_next_word(seed_text: str) -> str:
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences(
        [token_list], maxlen=WINDOW_SIZE, padding="post")
    proba = model.predict(token_list, verbose=0)[0]

    proba[0] = 0
    oov = tokenizer.word_index.get("<OOV>", -1)
    if oov != -1:
        proba[oov] = 0

    proba = np.asarray(proba, dtype="float64")
    proba = np.log(proba + 1e-7) / TEMPERATURE
    proba = np.exp(proba)
    proba = proba / np.sum(proba)

    idx = np.random.choice(len(proba), p=proba)
    return next((w for w, i in tokenizer.word_index.items() if i == idx), "")


st.markdown("""
<div class="center"><h1 class="hero-title">Predict the <span class="txt">next word</span></h1>
<p class="hero-sub">
  A bi-layer <span class="accent">input to sequence </span>LSTM  trained on Shakespeare's <em>Hamlet</em>  
  <br> Type any seed phrase and watch the model complete your thought.
</p></div>
""", unsafe_allow_html=True)


seed_text = st.text_area(
    "Seed phrase",
    value="To be or not to be",
    height=110,
    placeholder="Enter a sentence or phrase…",
)
predict_btn = st.button("✦  Predict", use_container_width=True)

# ── inference ─────────────────────────────────────────────────────────────────
if predict_btn:
    if not model_loaded:
        st.error(f"⚠ Could not load model: {load_error}")
    elif not seed_text.strip():
        st.warning("Please enter a seed phrase before predicting.")
    else:
        st.markdown("""
        <div class="timeline-row">
          <span class="tpill thinking">Thinking</span>
          <span class="tpill reading">Tokenising</span>
          <span class="tpill editing">Predicting</span>
        </div>
        """, unsafe_allow_html=True)

        with st.spinner("Predicting…"):
            try:
                word = predict_next_word(seed_text.strip())
                seed_clean = seed_text.strip()

                st.markdown(
                    '<div class="timeline-row">'
                    '<span class="tpill done">Done</span></div>',
                    unsafe_allow_html=True,
                )
                st.markdown(f"""
                <div class="result-card">
                  <div class="result-title">Output</div>
                  <div class="result-text">
                    {seed_clean} <span class="predicted-word">{word}</span>
                  </div>
                </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Prediction error: {e}")

# ── footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="cursor-footer">
  <span class="footer-left">Built with TensorFlow · Keras · Streamlit</span>
  <span class="footer-right">LSTM · Next Word Predictor</span>
</div>
""", unsafe_allow_html=True)
