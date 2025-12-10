import torch
import numpy as np
import pandas as pd
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ================== PAGE CONFIG ==================
st.set_page_config(
    page_title="Financial Sentiment Classifier",
    layout="centered"
)

# ================== CONFIG ==================
MODEL_DIR = "/home/abirh/fin_sentiment_model"
LABEL_NAMES = ["positive", "neutral", "negative"]

MAX_LEN = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource
def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(device)
    model.eval()
    return tokenizer, model


tokenizer, model = load_model_and_tokenizer()

# ================== CUSTOM STYLES ==================
st.markdown(
    """
    <style>
    .sentiment-badge {
        display: inline-block;
        padding: 0.40rem 0.85rem;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.95rem;
        margin-top: 0.5rem;
    }
    .sentiment-negative {
        background-color: #f5d6d6;
        color: #7a0000;
    }
    .sentiment-neutral {
        background-color: #e2e2e2;
        color: #4a4a4a;
    }
    .sentiment-positive {
        background-color: #d6f5df;
        color: #0b5722;
    }
    .prob-container {
        border-radius: 10px;
        padding: 1rem 1.2rem;
        background-color: #f7f7f9;
        margin-top: 1rem;
    }
    .footer-note {
        font-size: 0.8rem;
        color: #777777;
        margin-top: 2rem;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ================== SIDEBAR ==================
st.sidebar.title("Model Details")
st.sidebar.write("Fine-tuned **DistilBERT** model for financial sentiment classification.")
st.sidebar.write("Labels: negative, neutral, positive")

st.sidebar.markdown("---")
st.sidebar.caption("You may select an example headline to auto-fill the input.")

example_headlines = [
    "agilent technologies announces pricing       million senior notes",
    "agilent technologies inc ceo president michael mcmullen sold     million shares,",
    "strong cloud growth propel alibaba baba earnings.",
    "agilent eps beats misses revenue",
    "agilent earnings revenues surpass estimates",
    "agilent watch positive drug development"
]

example_choice = st.sidebar.selectbox(
    "Example headline:",
    ["(None)"] + example_headlines,
)

# ================== MAIN APP ==================
st.title("Financial Headline Sentiment Classifier")
st.write(
    "This tool analyzes financial news headlines using a fine-tuned DistilBERT model "
    "and predicts whether the sentiment is **negative**, **neutral**, or **positive**."
)

if example_choice != "(None)":
    default_text = example_choice
else:
    default_text = ""

headline = st.text_input(
    "Headline",
    value=default_text,
    placeholder="Enter a financial news headline...",
)

col1, col2 = st.columns([1, 2])
with col1:
    predict_clicked = st.button("Predict")
with col2:
    clear_clicked = st.button("Clear Input")

if clear_clicked:
    st.info("Input field cleared. You may enter a new headline.")

if predict_clicked:
    if not headline.strip():
        st.warning("Please enter a headline first.")
    else:
        # Tokenize
        inputs = tokenizer(
            headline,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        pred_idx = int(np.argmax(probs))
        pred_label = LABEL_NAMES[pred_idx]
        pred_prob = float(probs[pred_idx])

        # ================== DISPLAY RESULTS ==================
        st.markdown("### Prediction")

        if pred_label == "negative":
            badge_class = "sentiment-negative"
        elif pred_label == "neutral":
            badge_class = "sentiment-neutral"
        else:
            badge_class = "sentiment-positive"

        st.markdown(
            f'<span class="sentiment-badge {badge_class}">'
            f'{pred_label.capitalize()} — {pred_prob:.2%} confidence'
            f"</span>",
            unsafe_allow_html=True,
        )

        # Probabilities section
        st.markdown("#### Class Probabilities")
        prob_df = pd.DataFrame(
            {"sentiment": LABEL_NAMES, "probability": probs}
        ).set_index("sentiment")

        st.markdown('<div class="prob-container">', unsafe_allow_html=True)
        st.bar_chart(prob_df)
        st.markdown("</div>", unsafe_allow_html=True)

        # Echo the input
        st.markdown("#### Input Analyzed")
        st.info(headline)

# ================== FOOTER ==================
st.markdown(
    '<div class="footer-note">Financial sentiment analysis demo — DistilBERT model.</div>',
    unsafe_allow_html=True,
)
