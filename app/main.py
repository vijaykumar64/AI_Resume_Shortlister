import streamlit as st
import os
import pandas as pd

from utils import (
    extract_text_from_pdf,
    load_sentence_model,
    load_classifier,
    predict_category,
    calculate_match_percentage,
    generate_ats_score
)

# ------------------------------
# Streamlit UI Customization
# ------------------------------
st.set_page_config(page_title="AI Resume Matcher", layout="wide")

st.markdown("""
    <style>
    #MainMenu, footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# ------------------------------
# Title Section
# ------------------------------
st.header("🧠 AI-Powered Resume Shortlisting System")
st.markdown("Upload resumes and a job description to find the best matches 🔍")

# ------------------------------
# Load Models
# ------------------------------
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "models", "sentence_model")
CLASSIFIER_PATH = os.path.join(BASE_DIR, "models", "classifier_model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "models", "label_encoder.pkl")

s_model = load_sentence_model(MODEL_PATH)
clf_model, encoder = load_classifier(CLASSIFIER_PATH, ENCODER_PATH)

# ------------------------------
# JD Input
# ------------------------------
jd_input_type = st.radio("📝 Choose Job Description Input Method:", ["Upload PDF", "Paste Text"])
jd_text = ""

if jd_input_type == "Upload PDF":
    jd_file = st.file_uploader("📄 Upload Job Description (PDF)", type=["pdf"], key="jd_file")
    if jd_file:
        jd_text = extract_text_from_pdf(jd_file)
elif jd_input_type == "Paste Text":
    jd_text = st.text_area("📝 Paste Job Description Text Here", height=300)

# ------------------------------
# Resume Upload
# ------------------------------
resumes = st.file_uploader("📂 Upload Resumes (PDFs)", accept_multiple_files=True, type=["pdf"], key="resume_upload")

# ------------------------------
# Helper
# ------------------------------
def get_confidence_label(percentage):
    if percentage >= 70:
        return "🟢 Strong Fit for this JD"
    elif percentage >= 50:
        return "⚠️ Partial Fit – Improve Keywords"
    else:
        return "❌ Resume needs optimization"

# ------------------------------
# Process Button
# ------------------------------
if st.button("🔍 Find Best Matches"):
    if jd_text and resumes:
        jd_category = predict_category(jd_text, s_model, clf_model, encoder)

        results = []
        for resume in resumes:
            res_text = extract_text_from_pdf(resume)
            res_category = predict_category(res_text, s_model, clf_model, encoder)
            match = "✅ Match" if res_category == jd_category else "🔴 Not Match"

            # ✅ Now passing jd_category & res_category
            match_percent = calculate_match_percentage(jd_text, res_text, s_model, jd_category, res_category)
            confidence = get_confidence_label(match_percent)
            ats_score = generate_ats_score(res_text, jd_text, s_model, jd_category, res_category)

            results.append({
                "Resume Name": resume.name,
                "Predicted Category": res_category,
                "Match With JD": match,
                "Match %": match_percent,
                "Confidence": confidence,
                "ATS Score": ats_score
            })

        if results:
            df = pd.DataFrame(results).sort_values(by="Match %", ascending=False)
            st.success(f"🎯 JD Predicted Category: **{jd_category}**")
            st.dataframe(df, use_container_width=True)

            # Save and download
            df.to_csv("output/top_resume_matches.csv", index=False)
            st.download_button("📥 Download Result CSV", df.to_csv(index=False), "top_resume_matches.csv", "text/csv")
    else:
        st.error("⚠️ Please upload/paste JD and upload at least one resume.")
