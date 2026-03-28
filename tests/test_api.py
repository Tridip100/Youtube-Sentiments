from fastapi.testclient import TestClient
from fastapi.main import app

client = TestClient(app)

def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the FastAPI Sentiment Analysis API"}

def test_predict_empty():
    response = client.post("/predict", json={"comments": []})
    assert response.status_code == 400

def test_predict_with_timestamps_empty():
    response = client.post("/predict_with_timestamps", json={"comments": []})
    assert response.status_code == 400