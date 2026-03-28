import sys
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

# ── Mock model & vectorizer ───────────────────────────────────────────────────

mock_model = MagicMock()
mock_model.predict.return_value = np.array([1, 0, -1])

mock_vectorizer = MagicMock()
mock_vectorizer.transform.return_value = MagicMock(
    toarray=lambda: np.zeros((3, 10))
)
mock_vectorizer.get_feature_names_out.return_value = [
    f"word_{i}" for i in range(10)
]

# ── Inject fake mlflow BEFORE import ──────────────────────────────────────────

mock_mlflow = MagicMock()
mock_mlflow.pyfunc.load_model.return_value = mock_model

sys.modules["mlflow"] = mock_mlflow
sys.modules["mlflow.tracking"] = MagicMock()

# ── IMPORTANT: Patch load_model BEFORE importing app ──────────────────────────

with patch("backend.main.load_model_from_mlflow",
           return_value=(mock_model, mock_vectorizer)):

    from fastapi.testclient import TestClient
    from backend.main import app

client = TestClient(app)

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_comments():
    return [
        "This video is amazing!",
        "It was okay I guess",
        "Terrible, waste of time"
    ]

@pytest.fixture
def sample_comments_with_timestamps():
    return [
        {"text": "This video is amazing!", "timestamp": "2024-01-15T10:00:00Z"},
        {"text": "It was okay I guess",    "timestamp": "2024-02-20T12:00:00Z"},
        {"text": "Terrible, waste of time","timestamp": "2024-03-10T08:00:00Z"},
    ]

@pytest.fixture
def sample_sentiment_counts():
    return {"1": 50, "0": 30, "-1": 20}

@pytest.fixture
def sample_sentiment_data():
    return [
        {"sentiment": 1,  "timestamp": "2024-01-15T10:00:00Z"},
        {"sentiment": 0,  "timestamp": "2024-02-20T12:00:00Z"},
        {"sentiment": -1, "timestamp": "2024-03-10T08:00:00Z"},
    ]

# ── GET / ─────────────────────────────────────────────────────────────────────

def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"]

# ── POST /predict ─────────────────────────────────────────────────────────────

def test_predict_success(sample_comments):
    response = client.post("/predict", json={"comments": sample_comments})
    assert response.status_code == 200
    assert len(response.json()) == 3

def test_predict_empty_comments():
    response = client.post("/predict", json={"comments": []})
    assert response.status_code == 400

def test_predict_single_comment():
    mock_model.predict.return_value = np.array([1])
    mock_vectorizer.transform.return_value = MagicMock(
        toarray=lambda: np.zeros((1, 10))
    )

    response = client.post("/predict", json={"comments": ["Great video!"]})
    assert response.status_code == 200
    assert len(response.json()) == 1

# ── POST /predict_with_timestamps ─────────────────────────────────────────────

def test_predict_with_timestamps(sample_comments_with_timestamps):
    response = client.post(
        "/predict_with_timestamps",
        json={"comments": sample_comments_with_timestamps}
    )
    assert response.status_code == 200
    assert "timestamp" in response.json()[0]

# ── POST /generate_chart ──────────────────────────────────────────────────────

def test_generate_chart(sample_sentiment_counts):
    response = client.post(
        "/generate_chart",
        json={"sentiment_counts": sample_sentiment_counts}
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"

# ── POST /generate_wordcloud ──────────────────────────────────────────────────

def test_generate_wordcloud(sample_comments):
    response = client.post(
        "/generate_wordcloud",
        json={"comments": sample_comments}
    )
    assert response.status_code == 200

# ── POST /generate_trend_graph ────────────────────────────────────────────────

def test_generate_trend_graph(sample_sentiment_data):
    response = client.post(
        "/generate_trend_graph",
        json={"sentiment_data": sample_sentiment_data}
    )
    assert response.status_code == 200

# ── Edge cases ────────────────────────────────────────────────────────────────

def test_special_characters():
    response = client.post(
        "/predict",
        json={"comments": ["🔥🔥 Amazing!!! @test #wow"]}
    )
    assert response.status_code == 200

def test_empty_string():
    response = client.post("/predict", json={"comments": [""]})
    assert response.status_code == 200