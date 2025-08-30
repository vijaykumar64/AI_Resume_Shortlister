# app/train_classifier.py
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer
import os

# Load sentence model
s_model = SentenceTransformer("all-MiniLM-L6-v2")  # or your local sentence_model path

# Example mini training data
texts = [
    "Strong Python, ML, NLP, TensorFlow experience",
    "Frontend with React, JavaScript, CSS",
    "Backend developer Node.js, databases",
]
labels = ["ML Engineer", "Frontend", "Backend"]

# Encode labels
encoder = LabelEncoder()
y = encoder.fit_transform(labels)

# Encode texts
X = [s_model.encode(t) for t in texts]

# Train classifier
clf = LogisticRegression(max_iter=1000)
clf.fit(X, y)

# Save models
model_dir = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(model_dir, exist_ok=True)

joblib.dump(clf, os.path.join(model_dir, "classifier_model.pkl"))
joblib.dump(encoder, os.path.join(model_dir, "label_encoder.pkl"))

print("✅ Classifier and encoder saved in app/models/")
