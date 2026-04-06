# app.py
import streamlit as st
import torch
import json
import gdown
import os
import re
import emoji
import numpy as np
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from emoticon_fix import emoticon_fix

# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = "distilbert_emotion_model.pth"
MODEL_URL = "https://drive.google.com/uc?id=1x_OONKXfGx0I7uPzGzYOdR_5HOUVVdQy"
LABEL_MAPPING_PATH = "label_mapping.json"
MAX_LENGTH = 256

# -----------------------------
# LOAD LABEL MAPPING
# -----------------------------
with open(LABEL_MAPPING_PATH, "r") as f:
    label_mapping = json.load(f)
id_to_label = {v: k for k, v in label_mapping.items()}

# -----------------------------
# DOWNLOAD MODEL IF NOT PRESENT
# -----------------------------
if not os.path.exists(MODEL_PATH):
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# -----------------------------
# LOAD TOKENIZER & MODEL
# -----------------------------
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
num_labels = len(label_mapping)

model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased',
    num_labels=num_labels
)
model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# -----------------------------
# PREPROCESSING FUNCTION
# -----------------------------
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    text = emoji.demojize(text)
    text = emoticon_fix(text)
    text = re.sub(r'@\w+', '', text)
    return text

# -----------------------------
# STRESS MAPPING AND UTILS
# -----------------------------
stress_mapping = {
    'Suicidal': 5,
    'Depressed': 4,
    'Anxious': 3,
    'Frustrated': 2,
    'Others': 1
}

def normalize_score(score, min_score=1, max_score=5):
    return (score - min_score) / (max_score - min_score) * 10

def get_stress_level(score):
    if score <= 4:
        return "Low"
    elif score <= 7:
        return "Medium"
    else:
        return "High"

# -----------------------------
# PREDICTION FUNCTION
# -----------------------------
def predict(text):
    text = preprocess_text(text)
    encoding = tokenizer(
        [text],
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        pred_id = torch.argmax(logits, dim=-1).item()

    label = id_to_label[pred_id]
    score = stress_mapping[label]
    normalized = normalize_score(score)
    level = get_stress_level(normalized)

    return label, score, normalized, level

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.title("Community Stress Tracker AI")
st.write(
    "Analyze textual posts to detect emotional state and estimate stress levels at individual or community level."
)

# Multi-post input
texts_input = st.text_area(
    "Enter your posts (one per line):",
    placeholder="Write each post on a new line..."
).split("\n")
texts = [t.strip() for t in texts_input if t.strip()]

user_scores = []

if texts:
    st.subheader("Individual Predictions")
    for i, text in enumerate(texts, 1):
        label, score, norm, level = predict(text)
        st.markdown(f"**Post {i}:** {text}")
        st.write(f"Prediction: {label}")
        st.write(f"Stress Score: {score}")
        st.write(f"Normalized Score: {norm:.2f}")
        st.write(f"Stress Level: {level}")
        st.markdown("---")
        user_scores.append(score)

    # Aggregated community stress
    avg_score = np.mean(user_scores)
    norm_avg = normalize_score(avg_score)
    final_level = get_stress_level(norm_avg)

    st.subheader("Aggregated Stress (Community Level)")
    st.write(f"Normalized Score: {norm_avg:.2f}")
    st.write(f"Overall Stress Level: {final_level}")
