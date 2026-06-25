<div align="center">

# ✦ Next Word Predictor

### Bi-layer LSTM · Shakespeare's Hamlet · Streamlit

A sequence model that learns the language of Hamlet and predicts what comes next —  
built end-to-end from raw corpus ingestion to a polished web interface.

![Python](https://img.shields.io/badge/Python-3.10–3.13-3776AB?style=flat-square&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20+-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-f54e00?style=flat-square)

</div>

---

<img src="https://i.ibb.co/1JwbpFQj/718-1x-shots-so.png" alt="718 1x shots so" border="0">

## Overview

This project trains a next-word language model on Shakespeare's _Hamlet_ using a stacked LSTM architecture. Given any seed phrase, the model predicts the most contextually plausible continuation — one word at a time — using temperature-controlled sampling. The entire pipeline — data collection, preprocessing, training, inference, and UI — lives in this single repository.

---

## Pipeline

```
NLTK Gutenberg  →  Preprocessing  →  Tokenisation  →  Sequence Windows
     →  Train / Test Split  →  LSTM Training  →  Inference  →  Streamlit UI
```

---

## 1 · Data Collection

The corpus is sourced directly from **NLTK's Gutenberg corpus** — no manual download required.

```python
import nltk
nltk.download("gutenberg")

from nltk.corpus import gutenberg
data = gutenberg.raw("shakespeare-hamlet.txt")
```

The raw text is written to `hamlet.txt` and reloaded in lowercase to normalise capitalisation across the entire vocabulary.

```python
with open("hamlet.txt", "r", encoding="utf-8") as file:
    txt = file.read().lower()
```

> _Why Hamlet?_ It has a rich, varied vocabulary with poetic structure — ideal for a sequence model to learn contextual dependencies.

---

## 2 · Preprocessing & Tokenisation

### Tokeniser

A Keras `Tokenizer` is fitted on the full lowercased text with an `<OOV>` (out-of-vocabulary) token reserved for unseen words at inference time.

```python
from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts([txt])
total_words = len(tokenizer.word_index) + 1
```

### Sliding Window Sequences

The token stream is split into overlapping **7-token windows** (6 input tokens → 1 target token). This is the context the model sees during both training and inference.

```
Window size = 6
[ w1, w2, w3, w4, w5, w6 ]  →  target: w7
[ w2, w3, w4, w5, w6, w7 ]  →  target: w8
...
```

```python
window_size = 6
sequences = []
for i in range(window_size, len(input_sequence)):
    seq = input_sequence[i - window_size : i + 1]
    sequences.append(seq)
```

### Feature / Label Split

Each padded sequence is split into `X` (the 6-token context) and `y` (the target word), which is one-hot encoded across the full vocabulary.

```python
X = padded_sequences[:, :-1]          # shape: (N, 6)
y = padded_sequences[:, -1]           # shape: (N,)
y = to_categorical(y, num_classes=total_words)
```

---

## 3 · Train / Test Split

An 80 / 20 stratified split with a fixed random seed keeps results reproducible.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

---

## 4 · Model Architecture

A two-layer stacked LSTM with an embedding front-end and softmax output over the full vocabulary.

```
Input (6 tokens)
    │
    ▼
Embedding(vocab_size → 128, mask_zero=True)
    │
    ▼
LSTM(128, return_sequences=True)
    │
    ▼
Dropout(0.2)
    │
    ▼
LSTM(64, return_sequences=False)
    │
    ▼
Dense(vocab_size, activation='softmax')
```

| Component     | Detail                                   |
| ------------- | ---------------------------------------- |
| Embedding dim | 128                                      |
| LSTM layer 1  | 128 units, returns full sequence         |
| Dropout       | 0.2 (regularisation between LSTM layers) |
| LSTM layer 2  | 64 units, returns final hidden state     |
| Output        | Softmax over entire vocabulary           |
| Loss          | Categorical cross-entropy                |
| Optimiser     | Adam                                     |

```python
model = Sequential([
    Embedding(input_dim=total_words, output_dim=128,
              mask_zero=True, input_shape=(6,)),
    LSTM(128, return_sequences=True),
    Dropout(0.2),
    LSTM(64, return_sequences=False),
    Dense(total_words, activation='softmax'),
])

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
```

**`mask_zero=True`** on the embedding layer tells the LSTM to ignore padding tokens, so padded positions don't contribute to gradient updates.

---

## 5 · Training

The model trains for up to 50 epochs with **Early Stopping** (patience = 5) monitoring validation loss and restoring the best weights automatically.

```python
from tensorflow.keras.callbacks import EarlyStopping

stop_callback = EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    epochs=50,
    validation_data=(X_test, y_test),
    callbacks=[stop_callback],
    verbose=1,
)
```

Early stopping prevents overfitting to Hamlet's idiosyncratic syntax while keeping the best generalising checkpoint.

---

## 6 · Inference — Temperature Sampling

The prediction function takes a seed phrase and autoregressively generates `n_words` tokens. Raw softmax probabilities are re-scaled by a **temperature** parameter before sampling:

| Temperature       | Behaviour                                                                 |
| ----------------- | ------------------------------------------------------------------------- |
| `0.1`             | Near-deterministic — picks the highest-probability word almost every time |
| `0.3` _(default)_ | Confident but with some variation                                         |
| `1.0`             | Proportional sampling — the model's raw distribution                      |
| `2.0`             | Highly creative / unpredictable — low-probability words get a real chance |

```python
# Temperature scaling
predicted_proba = np.log(predicted_proba + 1e-7) / temperature
exp_preds       = np.exp(predicted_proba)
predicted_proba = exp_preds / np.sum(exp_preds)

predicted_index = np.random.choice(len(predicted_proba), p=predicted_proba)
```

`<OOV>` tokens and index-0 are suppressed before sampling so the model never outputs a placeholder.

---

## 7 · Saving Artifacts

```python
import joblib

model.save("model.keras")           # full model — architecture + weights
joblib.dump(tokenizer, "tokenizer.pkl")  # fitted tokeniser — vocab + indices
```

Both artifacts are loaded at Streamlit startup via `@st.cache_resource` so the model is only deserialised once per session.

---

## 8 · Streamlit Frontend

The UI is built to the **Cursor design system** — warm cream canvas, Inter + JetBrains Mono typography, and Cursor Orange (`#f54e00`) as the single brand accent.

**Features**

- Monospace seed-text area with an orange blinking caret
- Words-to-generate counter (1–20)
- Temperature slider with **Deterministic ↔ Creative** inline labels
- Inference timeline pills (Thinking → Tokenising → Predicting → Done)
- Output card with predicted words highlighted in orange
- Run-time metrics row: words generated · temperature · vocabulary size

---

## Project Structure

```
your_project/
├── frontend.py          # Streamlit application
├── model.keras          # Trained Keras model
├── tokenizer.pkl        # Fitted Keras tokenizer
├── requirements.txt     # Pinned dependencies
├── hamlet.txt           # Raw corpus (generated by notebook)
└── README.md
```

---

## Quickstart

```bash
# 1 — clone / copy the project folder
# 2 — create a virtual environment

# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate

# 3 — install dependencies
pip install -r requirements.txt

# 4 — launch
streamlit run frontend.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Requirements

```
tensorflow>=2.20.0
streamlit>=1.35.0
numpy>=1.26.0
scikit-learn>=1.4.0
joblib>=1.3.0
```

> TensorFlow 2.20+ officially supports **Python 3.10 – 3.13**.

---

## Tech Stack

| Layer          | Library                            |
| -------------- | ---------------------------------- |
| Corpus         | NLTK Gutenberg                     |
| Preprocessing  | Keras `Tokenizer`, `pad_sequences` |
| Modelling      | TensorFlow / Keras (LSTM)          |
| Training split | scikit-learn                       |
| Serialisation  | `model.keras` + `joblib`           |
| Frontend       | Streamlit                          |
| Fonts          | Inter · JetBrains Mono             |

---

<div align="center">
  <sub>Built with TensorFlow · Keras · Streamlit &nbsp;·&nbsp; Trained on Shakespeare's <em>Hamlet</em></sub>
</div>
