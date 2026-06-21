import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

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
  --primary:          #f54e00;
  --primary-active:   #d04200;
  --ink:              #26251e;
  --body:             #5a5852;
  --body-strong:      #26251e;
  --muted:            #807d72;
  --muted-soft:       #a09c92;
  --hairline:         #e6e5e0;
  --hairline-soft:    #efeee8;
  --hairline-strong:  #cfcdc4;
  --canvas:           #f7f7f4;
  --canvas-soft:      #fafaf7;
  --surface-card:     #ffffff;
  --surface-strong:   #e6e5e0;
  --on-primary:       #ffffff;
  --timeline-thinking:#dfa88f;
  --timeline-grep:    #9fc9a2;
  --timeline-read:    #9fbbe0;
  --timeline-edit:    #c0a8dd;
  --timeline-done:    #c08532;
  --semantic-error:   #cf2d56;
  --semantic-success: #1f8a65;
  --font-display: 'Inter', system-ui, 'Helvetica Neue', Arial, sans-serif;
  --font-mono:    'JetBrains Mono', 'Fira Code', monospace;
}

/* ── Reset / Global ── */
html, body, [data-testid="stAppViewContainer"],
[data-testid="stApp"] {
  background-color: var(--canvas) !important;
  font-family: var(--font-display) !important;
  color: var(--ink) !important;
}

/* hide default streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }
[data-testid="stToolbar"] { display: none; }

/* ── Main content container ── */
.block-container {
  max-width: 760px !important;
  padding: 0 24px 80px !important;
  margin: 0 auto !important;
}

/* ── Top Nav ── */
.cursor-nav {
  display: flex;
  align-items: center;
  justify-content: space-between;
  height: 64px;
  border-bottom: 1px solid var(--hairline);
  margin-bottom: 80px;
  padding: 0 4px;
}
.cursor-nav-logo {
  font-family: var(--font-display);
  font-size: 16px;
  font-weight: 600;
  color: var(--ink);
  letter-spacing: -0.3px;
  display: flex;
  align-items: center;
  gap: 8px;
}
.cursor-nav-logo span.orange { color: var(--primary); }
.cursor-nav-badge {
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.88px;
  text-transform: uppercase;
  background: var(--surface-strong);
  color: var(--ink);
  border-radius: 9999px;
  padding: 4px 10px;
}

/* ── Hero section ── */
.hero-eyebrow {
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.88px;
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: 16px;
}
.hero-title {
  font-family: var(--font-display);
  font-size: 56px;
  font-weight: 400;
  line-height: 1.1;
  letter-spacing: -1.68px;
  color: var(--ink);
  margin-bottom: 20px;
}
.hero-title .accent { color: var(--primary); }
.hero-sub {
  font-size: 16px;
  font-weight: 400;
  line-height: 1.6;
  color: var(--body);
  max-width: 520px;
  margin-bottom: 48px;
}

/* ── Input card / IDE pane ── */
.ide-card {
  background: var(--surface-card);
  border-radius: 12px;
  border: 1px solid var(--hairline);
  overflow: hidden;
  margin-bottom: 24px;
}
.ide-card-header {
  background: var(--canvas-soft);
  border-bottom: 1px solid var(--hairline);
  padding: 12px 20px;
  display: flex;
  align-items: center;
  gap: 8px;
}
.ide-dot {
  width: 10px; height: 10px;
  border-radius: 50%;
  background: var(--hairline-strong);
}
.ide-dot.red   { background: #ff5f57; }
.ide-dot.yellow{ background: #febc2e; }
.ide-dot.green { background: #28c840; }
.ide-label {
  font-family: var(--font-mono);
  font-size: 12px;
  color: var(--muted);
  margin-left: auto;
}
.ide-card-body {
  padding: 24px 28px;
}

/* ── Override Streamlit text_area ── */
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
  font-size: 13px !important;
  font-weight: 600 !important;
  color: var(--body-strong) !important;
  margin-bottom: 8px !important;
}

/* ── Override Streamlit slider ── */
[data-testid="stSlider"] label {
  font-size: 13px !important;
  font-weight: 600 !important;
  color: var(--body-strong) !important;
}
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
  background: var(--primary) !important;
}

/* ── Override Streamlit number_input ── */
[data-testid="stNumberInput"] label {
  font-size: 13px !important;
  font-weight: 600 !important;
  color: var(--body-strong) !important;
}
[data-testid="stNumberInput"] input {
  font-family: var(--font-mono) !important;
  font-size: 14px !important;
  border-radius: 8px !important;
  border: 1px solid var(--hairline-strong) !important;
  background: var(--canvas-soft) !important;
  color: var(--ink) !important;
}

/* ── Primary CTA button ── */
[data-testid="stButton"] > button {
  background: var(--primary) !important;
  color: var(--on-primary) !important;
  font-family: var(--font-display) !important;
  font-size: 14px !important;
  font-weight: 500 !important;
  border-radius: 8px !important;
  border: none !important;
  padding: 10px 24px !important;
  height: 44px !important;
  letter-spacing: 0 !important;
  transition: background 0.15s ease !important;
  width: 100% !important;
}
[data-testid="stButton"] > button:hover {
  background: var(--primary-active) !important;
}
[data-testid="stButton"] > button:active {
  background: var(--primary-active) !important;
}

/* ── Result output card ── */
.result-card {
  background: var(--surface-card);
  border-radius: 12px;
  border: 1px solid var(--hairline);
  padding: 28px;
  margin-top: 24px;
}
.result-header {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 20px;
}
.result-title {
  font-size: 13px;
  font-weight: 600;
  letter-spacing: 0.88px;
  text-transform: uppercase;
  color: var(--muted);
}
.result-text {
  font-family: var(--font-mono);
  font-size: 15px;
  line-height: 1.7;
  color: var(--ink);
  background: var(--canvas-soft);
  border-radius: 8px;
  padding: 18px 20px;
  border: 1px solid var(--hairline-soft);
  word-break: break-word;
}
.result-text .predicted-word {
  color: var(--primary);
  font-weight: 600;
  background: rgba(245,78,0,0.07);
  border-radius: 4px;
  padding: 1px 5px;
}

/* ── Timeline pills ── */
.timeline-row {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
  margin-bottom: 16px;
}
.tpill {
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.88px;
  text-transform: uppercase;
  border-radius: 9999px;
  padding: 4px 12px;
  color: var(--ink);
}
.tpill.thinking { background: var(--timeline-thinking); }
.tpill.reading  { background: var(--timeline-read); }
.tpill.editing  { background: var(--timeline-edit); }
.tpill.done     { background: var(--timeline-done); color: #fff; }

/* ── Metrics row ── */
.metrics-row {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 12px;
  margin-top: 20px;
}
.metric-cell {
  background: var(--canvas-soft);
  border: 1px solid var(--hairline);
  border-radius: 8px;
  padding: 16px;
  text-align: center;
}
.metric-value {
  font-family: var(--font-mono);
  font-size: 22px;
  font-weight: 400;
  color: var(--ink);
  letter-spacing: -0.5px;
}
.metric-label {
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.88px;
  text-transform: uppercase;
  color: var(--muted);
  margin-top: 4px;
}

/* ── Section divider ── */
.divider {
  border: none;
  border-top: 1px solid var(--hairline);
  margin: 48px 0;
}

/* ── Footer ── */
.cursor-footer {
  margin-top: 80px;
  padding-top: 32px;
  border-top: 1px solid var(--hairline);
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.footer-left {
  font-size: 13px;
  color: var(--muted);
}
.footer-right {
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.88px;
  text-transform: uppercase;
  color: var(--muted-soft);
}

/* ── Error / info alerts ── */
[data-testid="stAlert"] {
  border-radius: 8px !important;
  border: 1px solid var(--hairline-strong) !important;
  background: var(--canvas-soft) !important;
}
</style>
""", unsafe_allow_html=True)



@st.cache_resource(show_spinner=False)
def load_artifacts():
    model     = load_model("model.keras")
    tokenizer = joblib.load("tokenizer.pkl")
    return model, tokenizer




WINDOW_SIZE = 6   

def predict_next_word(seed_text: str, n_words: int, temperature: float,
                      model, tokenizer) -> tuple[str, list[str]]:

    current_text   = seed_text
    predicted_words: list[str] = []

    for _ in range(n_words):
        token_list = tokenizer.texts_to_sequences([current_text])[0]
        token_list = pad_sequences([token_list], maxlen=WINDOW_SIZE, padding="post")

        predicted_proba = model.predict(token_list, verbose=0)[0]

        predicted_proba[0] = 0
        oov_index = tokenizer.word_index.get("<OOV>", -1)
        if oov_index != -1:
            predicted_proba[oov_index] = 0

        
        predicted_proba = np.asarray(predicted_proba, dtype="float64")
        predicted_proba = np.log(predicted_proba + 1e-7) / temperature
        exp_preds       = np.exp(predicted_proba)
        predicted_proba = exp_preds / np.sum(exp_preds)

        predicted_index = np.random.choice(len(predicted_proba), p=predicted_proba)

        output_word = ""
        for word, idx in tokenizer.word_index.items():
            if idx == predicted_index:
                output_word = word
                break

        if output_word:
            current_text   += " " + output_word
            predicted_words.append(output_word)
        else:
            break

    return current_text, predicted_words



st.markdown("""
<div class="cursor-nav">
  <div class="cursor-nav-logo">
    <span class="orange">✦</span> NextWord
  </div>
  <span class="cursor-nav-badge">LSTM · Shakespeare</span>
</div>
""", unsafe_allow_html=True)





st.markdown("""
<p class="hero-eyebrow">Deep Learning · NLP</p>
<h1 class="hero-title">Predict the<br><span class="accent">next word.</span></h1>
<p class="hero-sub">
  A bi-layer LSTM trained on Shakespeare's <em>Hamlet</em> — 
  type any seed phrase and watch the model complete your thought.
</p>
""", unsafe_allow_html=True)




with st.spinner("Loading model…"):
    try:
        model, tokenizer = load_artifacts()
        vocab_size = len(tokenizer.word_index) + 1
        model_loaded = True
    except Exception as e:
        model_loaded = False
        load_error   = str(e)




with st.container():
    seed_text = st.text_area(
        "Seed phrase",
        value="To be or not to be",
        height=110,
        placeholder="Enter a sentence or phrase…",
        help="The model uses the last 6 words as context.",
    )

    col1, col2 = st.columns(2)
    with col1:
        n_words = st.number_input(
            "Words to generate",
            min_value=1, max_value=20, value=1, step=1,
        )
    with col2:
        temperature = st.slider(
            "Temperature",
            min_value=0.1, max_value=2.0, value=0.3, step=0.05,
        )
        st.markdown(
            '<div style="display:flex;justify-content:space-between;'
            'font-size:11px;font-weight:600;letter-spacing:0.6px;'
            'text-transform:uppercase;color:var(--muted);margin-top:-10px;">'
            '<span>Deterministic</span><span>Creative</span></div>',
            unsafe_allow_html=True,
        )

    predict_btn = st.button("✦  Predict", use_container_width=True)





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

        with st.spinner("Running inference…"):
            try:
                full_text, new_words = predict_next_word(
                    seed_text.strip(), int(n_words), float(temperature),
                    model, tokenizer,
                )


                st.markdown('<div class="timeline-row"><span class="tpill done">Done</span></div>',
                            unsafe_allow_html=True)

  
                seed_clean   = seed_text.strip()
                tail_section = full_text[len(seed_clean):]
                tail_html    = ""
                for w in tail_section.split():
                    tail_html += f' <span class="predicted-word">{w}</span>'

                st.markdown(f"""
                <div class="result-card">
                  <div class="result-header">
                    <span class="result-title">Output</span>
                  </div>
                  <div class="result-text">{seed_clean}{tail_html}</div>
                  <div class="metrics-row">
                    <div class="metric-cell">
                      <div class="metric-value">{len(new_words)}</div>
                      <div class="metric-label">Words generated</div>
                    </div>
                    <div class="metric-cell">
                      <div class="metric-value">{temperature}</div>
                      <div class="metric-label">Temperature</div>
                    </div>
                    <div class="metric-cell">
                      <div class="metric-value">{vocab_size:,}</div>
                      <div class="metric-label">Vocab size</div>
                    </div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Prediction error: {e}")



st.markdown("""
<div class="cursor-footer">
  <span class="footer-left">Built with TensorFlow · Keras · Streamlit</span>
  <span class="footer-right">LSTM · Next Word Predictor</span>
</div>
""", unsafe_allow_html=True)