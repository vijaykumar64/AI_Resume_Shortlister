# app/train_classifier.py
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils.multiclass import unique_labels
from sentence_transformers import SentenceTransformer

# ----------------------------
# Load Sentence Model
# ----------------------------
s_model = SentenceTransformer("all-MiniLM-L6-v2")  # same model used in utils.py
embedding_dim = s_model.get_sentence_embedding_dimension()
print(f"✅ Loaded Sentence Model. Embedding dim = {embedding_dim}")

# ----------------------------
# Load Dataset (CSV or Inline Examples)
# ----------------------------
dataset_path = os.path.join(os.path.dirname(__file__), "jd_dataset.csv")

if os.path.exists(dataset_path):
    print(f"📂 Loading dataset from {dataset_path}")
    df = pd.read_csv(dataset_path)

    if "jd_text" not in df.columns or "category" not in df.columns:
        raise ValueError("Dataset must contain 'jd_text' and 'category' columns")

    texts = df["jd_text"].astype(str).tolist()
    labels = df["category"].astype(str).tolist()
else:
    print("⚠️ No dataset found, using small inline examples...")
    texts = [
        "Strong Python, ML, NLP, TensorFlow experience",
        "Frontend with React, JavaScript, CSS",
        "Backend developer Node.js, databases",
    ]
    labels = ["ML Engineer", "Frontend Developer", "Backend Developer"]

# ----------------------------
# Encode Labels
# ----------------------------
encoder = LabelEncoder()
y = encoder.fit_transform(labels)
print("✅ Classes:", list(encoder.classes_))

# ----------------------------
# Create Embeddings
# ----------------------------
X = s_model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
print("👉 Embedding matrix shape:", X.shape)

# ----------------------------
# Train/Test Split
# ----------------------------
test_size = max(0.2, len(set(y)) / len(y))  # ensure enough test samples per class
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42, stratify=y if len(set(y)) > 1 else None
)

print(f"📊 Train size: {len(X_train)}, Test size: {len(X_test)}")

# ----------------------------
# Train Classifier
# ----------------------------
clf = LogisticRegression(max_iter=3000)
clf.fit(X_train, y_train)

# ----------------------------
# Evaluate
# ----------------------------
y_pred = clf.predict(X_test)
labels_in_data = unique_labels(y_test, y_pred)

print("\n📊 Classification Report:\n")
print(classification_report(
    y_test,
    y_pred,
    labels=labels_in_data,
    target_names=encoder.classes_[labels_in_data]
))

# ----------------------------
# Save Models
# ----------------------------
model_dir = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(model_dir, exist_ok=True)

clf_path = os.path.join(model_dir, "classifier_model.pkl")
enc_path = os.path.join(model_dir, "label_encoder.pkl")

joblib.dump(clf, clf_path)
joblib.dump(encoder, enc_path)

print("\n✅ Classifier and encoder saved in app/models/")
print("📂", clf_path)
print("📂", enc_path)
print("Available Categories:", list(encoder.classes_))
