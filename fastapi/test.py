import re
import pickle
import mlflow
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ── AWS MLflow Config ─────────────────────────────────────────────────────────

MLFLOW_TRACKING_URI = "http://ec2-13-50-5-147.eu-north-1.compute.amazonaws.com:5000"


# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess_comment(comment: str) -> str:
    try:
        comment = comment.lower().strip()
        comment = __import__('re').sub(r'\n', ' ', comment)
        comment = __import__('re').sub(r'[^A-Za-z0-9\s!?.,]', '', comment)
        stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
        comment = ' '.join([w for w in comment.split() if w not in stop_words])
        lemmatizer = WordNetLemmatizer()
        comment = ' '.join([lemmatizer.lemmatize(w) for w in comment.split()])
        return comment
    except Exception as e:
        print(f"Error preprocessing comment: {e}")
        return comment


# ── Model Loading from AWS MLflow ─────────────────────────────────────────────

def load_model_from_mlflow(model_name: str, model_version: str, vectorizer_path: str):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"models:/{model_name}/{model_version}"
    model = mlflow.pyfunc.load_model(model_uri)
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer


# ── Local Predict Test ────────────────────────────────────────────────────────

def predict(model, vectorizer):
    comments = ["I love this product!", "This is the worst experience."]

    try:
        preprocessed = [preprocess_comment(c) for c in comments]
        transformed = vectorizer.transform(preprocessed).toarray()
        predictions = model.predict(transformed).tolist()
    except Exception as e:
        print(f"Prediction error: {e}")
        return

    response = [{"comment": c, "sentiment": s} for c, s in zip(comments, predictions)]
    print(response)


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Update model_name and model_version to match your MLflow registry
    model, vectorizer = load_model_from_mlflow(
        model_name="my_model",
        model_version="1",
        vectorizer_path="./tfidf_vectorizer.pkl"
    )
    predict(model, vectorizer)