"""LRI-Text demonstration app. Research-only demo, NOT a production sentiment
monitor: confidence scores are model outputs, not certainty, and the model was
fine-tuned on a small labelled sample (see research_methodology.md for scope).
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config  # noqa: E402

st.set_page_config(page_title="LRI-Text Demo", layout="centered")


@st.cache_data
def available_checkpoints(cfg: dict) -> list[str]:
    ckpt_dir = Path(PROJECT_ROOT) / cfg["paths"]["checkpoints"]
    if not ckpt_dir.exists():
        return []
    return sorted([p.name for p in ckpt_dir.iterdir() if p.is_dir()])


@st.cache_resource
def load_model(checkpoint_dir: str):
    path = Path(PROJECT_ROOT) / "checkpoints" / checkpoint_dir
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)
    model.eval()
    return model, tokenizer


def predict(text: str, model, tokenizer, label_names: list[str]):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    return {label_names[i]: float(probs[i]) for i in range(len(label_names))}


def main():
    cfg = load_config()
    st.title("LRI-Text: Hindi/Bengali Sentiment Demo")
    st.caption(
        "Research demo only — NOT a production sentiment monitor. The model was "
        "fine-tuned on a small labelled sample under a controlled low-resource "
        "study; confidence scores are model probabilities, not certainty. "
        "See research_methodology.md for the full experimental scope and limitations."
    )

    language = st.selectbox("Language", ["hindi", "bengali"])
    checkpoints = [c for c in available_checkpoints(cfg) if c.startswith(language)]
    if not checkpoints:
        st.warning(f"No trained checkpoints found for {language} yet. Run src/training/trainer.py first.")
        return

    checkpoint_dir = st.selectbox("Trained model", checkpoints)
    text = st.text_area("Enter text", height=120)

    if st.button("Predict") and text.strip():
        model, tokenizer = load_model(checkpoint_dir)
        label_names = cfg["canonical_label_names"]
        probs = predict(text, model, tokenizer, label_names)
        pred_label = max(probs, key=probs.get)

        st.subheader(f"Predicted: {pred_label}")
        st.bar_chart(pd.Series(probs))
        st.caption(f"Model: {checkpoint_dir} — these are softmax probabilities, not calibrated confidence.")


if __name__ == "__main__":
    main()
