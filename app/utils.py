import os
import re
import fitz  # PyMuPDF
import joblib
import numpy as np
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

# ---------------------------
# Load transformer model
# ---------------------------
def load_sentence_model(path="models/sentence_model"):
    try:
        print(f"🔍 Trying to load local model from: {path}")
        return SentenceTransformer(path)
    except Exception as e:
        print(f"⚠️ Local model load failed ({e}). Falling back to HuggingFace model...")
        return SentenceTransformer("all-MiniLM-L6-v2")

# ---------------------------
# PDF Text Extraction
# ---------------------------
def extract_text_from_pdf(file_obj):
    text = ""
    try:
        pdf_bytes = file_obj.read()
        file_obj.seek(0)

        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            for page in doc:
                text += page.get_text() + " "
    except Exception as e:
        print(f"PyMuPDF failed, trying PyPDF2: {e}")
        file_obj.seek(0)
        try:
            reader = PdfReader(file_obj)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += str(page_text) + " "
        except Exception as e:
            print(f"PyPDF2 also failed: {e}")
            return ""

    return text.strip()

# ---------------------------
# JD Keyword Extraction
# ---------------------------
def extract_keywords_from_jd(jd_text, top_n=20):
    if not isinstance(jd_text, str) or not jd_text.strip():
        return []
    vectorizer = CountVectorizer(stop_words="english")
    word_counts = vectorizer.fit_transform([jd_text])
    word_freq = dict(zip(vectorizer.get_feature_names_out(), word_counts.toarray().flatten()))
    sorted_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return [kw[0] for kw in sorted_keywords[:top_n]]

# ---------------------------
# Classifier Loader
# ---------------------------
def load_classifier(path="models/classifier_model.pkl", encoder_path="models/label_encoder.pkl"):
    if not os.path.exists(path) or not os.path.exists(encoder_path):
        raise FileNotFoundError(f"Classifier or encoder not found at {path}, {encoder_path}")
    model = joblib.load(path)
    encoder = joblib.load(encoder_path)
    print("✅ Classifier and encoder loaded. Classes:", list(encoder.classes_))
    return model, encoder

# ---------------------------
# Embeddings & Predictions
# ---------------------------
def get_embedding(text, model):
    if text is None:
        text = ""
    elif isinstance(text, (list, tuple)):
        text = " ".join([str(t) for t in text if t])
    elif isinstance(text, bytes):
        text = text.decode("utf-8", errors="ignore")
    else:
        text = str(text)

    text = text.strip()
    if not text:
        return np.zeros(model.get_sentence_embedding_dimension())

    try:
        emb = model.encode([text], show_progress_bar=False, convert_to_numpy=True)
        vec = np.array(emb[0])
        return vec
    except Exception as e:
        print(f"An error occurred during embedding: {e}")
        return np.zeros(model.get_sentence_embedding_dimension())

def predict_category(text, s_model, clf_model, encoder):
    vector = get_embedding(text, s_model).reshape(1, -1)
    prediction = clf_model.predict(vector)
    label = encoder.inverse_transform(prediction)
    return label[0]

# ---------------------------
# Match Percentage (with +18 boost if category matches)
# ---------------------------
def calculate_match_percentage(jd_text, resume_text, s_model, jd_category=None, res_category=None):
    if not jd_text or not resume_text:
        return 0.0
    jd_vec = get_embedding(jd_text, s_model).reshape(1, -1)
    res_vec = get_embedding(resume_text, s_model).reshape(1, -1)
    similarity = cosine_similarity(jd_vec, res_vec)[0][0]
    match_percent = similarity * 100

    # ✅ Boost by +18 if categories match
    if jd_category and res_category and jd_category == res_category:
        print("🎯 JD and Resume categories matched → +18% boost applied")
        match_percent += 18

    return round(min(match_percent, 100), 2)  # cap at 100

# ---------------------------
# Keyword Extraction & Highlighting
# ---------------------------
def extract_keywords(text, top_k=10):
    if not isinstance(text, str) or not text.strip():
        return []
    words = re.findall(r"\b\w+\b", text.lower())
    stopwords = set([
        "the", "and", "to", "for", "in", "of", "on", "a", "an", "with", "by",
        "am", "is", "are", "was", "were", "have", "has", "do", "does", "did",
        "shall", "should", "can", "could", "will", "would", "must", "out", "need", "used"
    ])
    filtered = [word for word in words if word not in stopwords and len(word) > 2]
    freq = {}
    for word in filtered:
        freq[word] = freq.get(word, 0) + 1
    sorted_keywords = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [kw for kw, count in sorted_keywords[:top_k]]

def highlight_keywords_in_resume(resume_text, jd_text):
    jd_combined = " ".join(jd_text) if isinstance(jd_text, (list, tuple)) else str(jd_text)
    keywords = extract_keywords(jd_combined)
    highlighted = resume_text
    for kw in keywords:
        highlighted = re.sub(
            rf"\b({kw})\b",
            r"<mark style='background-color: yellow; font-weight: bold'>\1</mark>",
            highlighted,
            flags=re.IGNORECASE,
        )
    return highlighted

# ---------------------------
# ATS Score (with +3 boost if category matches)
# ---------------------------
def generate_ats_score(resume_text, jd_text, s_model, jd_category=None, res_category=None):
    if not jd_text or not resume_text:
        return 0.0
    jd_keywords = extract_keywords(jd_text)
    keyword_hits = sum(1 for kw in jd_keywords if kw.lower() in resume_text.lower())
    keyword_score = keyword_hits / len(jd_keywords) if jd_keywords else 0
    match_percent = calculate_match_percentage(jd_text, resume_text, s_model, jd_category, res_category) / 100
    
    ats_score = (0.6 * match_percent + 0.4 * keyword_score) * 10

    # ✅ Boost by +3 if categories match
    if jd_category and res_category and jd_category == res_category:
        print("🎯 JD and Resume categories matched → +3 ATS score boost applied")
        ats_score += 2

    return round(min(ats_score, 10), 2)  # cap at 10

# ---------------------------
# Feedback Store
# ---------------------------
def store_feedback(jd_text, resume_text, actual_label, predicted_label, feedback_path="feedback.csv"):
    row = {
        "jd_text": str(jd_text),
        "resume_text": str(resume_text),
        "actual_label": str(actual_label),
        "predicted_label": str(predicted_label),
    }
    if os.path.exists(feedback_path):
        df = pd.read_csv(feedback_path)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(feedback_path, index=False)
    print(f"✅ Feedback stored in {feedback_path}")
