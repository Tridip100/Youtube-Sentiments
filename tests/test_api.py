import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd


# ── Mock model & vectorizer before importing app ──────────────────────────────

mock_model = MagicMock()
mock_model.predict.return_value = np.array([1, 0, -1])

mock_vectorizer = MagicMock()
mock_vectorizer.transform.return_value = MagicMock(toarray=lambda: np.zeros((3, 10)))
mock_vectorizer.get_feature_names_out.return_value = [f"word_{i}" for i in range(10)]

with patch("main.load_model_from_mlflow", return_value=(mock_model, mock_vectorizer)):
    from fastapi.main import app

client = TestClient(app)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_comments():
    return ["This video is amazing!", "It was okay I guess", "Terrible, waste of time"]

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
    assert response.json() == {"message": "Welcome to the FastAPI Sentiment Analysis API"}


# ── POST /predict ─────────────────────────────────────────────────────────────

def test_predict_success(sample_comments):
    mock_vectorizer.transform.return_value = MagicMock(toarray=lambda: np.zeros((3, 10)))
    mock_model.predict.return_value = np.array([1, 0, -1])

    response = client.post("/predict", json={"comments": sample_comments})
    assert response.status_code == 200

    data = response.json()
    assert len(data) == 3
    assert "comment" in data[0]
    assert "sentiment" in data[0]

def test_predict_empty_comments():
    response = client.post("/predict", json={"comments": []})
    assert response.status_code == 400
    assert "No comments provided" in response.json()["detail"]

def test_predict_single_comment():
    mock_vectorizer.transform.return_value = MagicMock(toarray=lambda: np.zeros((1, 10)))
    mock_model.predict.return_value = np.array([1])

    response = client.post("/predict", json={"comments": ["Great video!"]})
    assert response.status_code == 200
    assert len(response.json()) == 1

def test_predict_returns_correct_structure(sample_comments):
    mock_vectorizer.transform.return_value = MagicMock(toarray=lambda: np.zeros((3, 10)))
    mock_model.predict.return_value = np.array([1, 0, -1])

    response = client.post("/predict", json={"comments": sample_comments})
    data = response.json()
    for item in data:
        assert "comment" in item
        assert "sentiment" in item
        assert item["comment"] in sample_comments


# ── POST /predict_with_timestamps ─────────────────────────────────────────────

def test_predict_with_timestamps_success(sample_comments_with_timestamps):
    mock_vectorizer.transform.return_value = MagicMock(toarray=lambda: np.zeros((3, 10)))
    mock_model.predict.return_value = np.array([1, 0, -1])

    response = client.post("/predict_with_timestamps",
                           json={"comments": sample_comments_with_timestamps})
    assert response.status_code == 200

    data = response.json()
    assert len(data) == 3
    assert "comment"   in data[0]
    assert "sentiment" in data[0]
    assert "timestamp" in data[0]

def test_predict_with_timestamps_empty():
    response = client.post("/predict_with_timestamps", json={"comments": []})
    assert response.status_code == 400
    assert "No comments provided" in response.json()["detail"]

def test_predict_with_timestamps_preserves_timestamps(sample_comments_with_timestamps):
    mock_vectorizer.transform.return_value = MagicMock(toarray=lambda: np.zeros((3, 10)))
    mock_model.predict.return_value = np.array([1, 0, -1])

    response = client.post("/predict_with_timestamps",
                           json={"comments": sample_comments_with_timestamps})
    data = response.json()
    returned_timestamps = [item["timestamp"] for item in data]
    original_timestamps = [c["timestamp"] for c in sample_comments_with_timestamps]
    assert returned_timestamps == original_timestamps


# ── POST /generate_chart ──────────────────────────────────────────────────────

def test_generate_chart_success(sample_sentiment_counts):
    response = client.post("/generate_chart",
                           json={"sentiment_counts": sample_sentiment_counts})
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert len(response.content) > 0

def test_generate_chart_empty():
    response = client.post("/generate_chart", json={"sentiment_counts": {}})
    assert response.status_code == 400

def test_generate_chart_all_zero():
    response = client.post("/generate_chart",
                           json={"sentiment_counts": {"1": 0, "0": 0, "-1": 0}})
    assert response.status_code == 400

def test_generate_chart_partial_counts():
    # Only positive comments — should still work
    response = client.post("/generate_chart",
                           json={"sentiment_counts": {"1": 100}})
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"


# ── POST /generate_wordcloud ──────────────────────────────────────────────────

def test_generate_wordcloud_success(sample_comments):
    response = client.post("/generate_wordcloud",
                           json={"comments": sample_comments})
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert len(response.content) > 0

def test_generate_wordcloud_empty():
    response = client.post("/generate_wordcloud", json={"comments": []})
    assert response.status_code == 400
    assert "No comments provided" in response.json()["detail"]

def test_generate_wordcloud_large_input():
    comments = ["great video very interesting content"] * 100
    response = client.post("/generate_wordcloud", json={"comments": comments})
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"


# ── POST /generate_trend_graph ────────────────────────────────────────────────

def test_generate_trend_graph_success(sample_sentiment_data):
    response = client.post("/generate_trend_graph",
                           json={"sentiment_data": sample_sentiment_data})
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert len(response.content) > 0

def test_generate_trend_graph_empty():
    response = client.post("/generate_trend_graph", json={"sentiment_data": []})
    assert response.status_code == 400
    assert "No sentiment data provided" in response.json()["detail"]

def test_generate_trend_graph_single_month():
    data = [
        {"sentiment": 1,  "timestamp": "2024-01-01T10:00:00Z"},
        {"sentiment": -1, "timestamp": "2024-01-15T12:00:00Z"},
        {"sentiment": 0,  "timestamp": "2024-01-20T08:00:00Z"},
    ]
    response = client.post("/generate_trend_graph", json={"sentiment_data": data})
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"

def test_generate_trend_graph_all_positive():
    data = [{"sentiment": 1, "timestamp": f"2024-0{i+1}-01T10:00:00Z"} for i in range(3)]
    response = client.post("/generate_trend_graph", json={"sentiment_data": data})
    assert response.status_code == 200


# ── Preprocessing edge cases ──────────────────────────────────────────────────

def test_predict_handles_special_characters():
    mock_vectorizer.transform.return_value = MagicMock(toarray=lambda: np.zeros((1, 10)))
    mock_model.predict.return_value = np.array([1])

    response = client.post("/predict",
                           json={"comments": ["Great video!!! 🔥🔥 #amazing @channel"]})
    assert response.status_code == 200

def test_predict_handles_empty_string():
    mock_vectorizer.transform.return_value = MagicMock(toarray=lambda: np.zeros((1, 10)))
    mock_model.predict.return_value = np.array([0])

    response = client.post("/predict", json={"comments": [""]})
    assert response.status_code == 200

def test_predict_handles_very_long_comment():
    mock_vectorizer.transform.return_value = MagicMock(toarray=lambda: np.zeros((1, 10)))
    mock_model.predict.return_value = np.array([1])

    long_comment = "this is a great video " * 200
    response = client.post("/predict", json={"comments": [long_comment]})
    assert response.status_code == 200