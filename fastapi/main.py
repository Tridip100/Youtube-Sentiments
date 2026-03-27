import matplotlib
matplotlib.use('Agg')

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List
import io
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from wordcloud import WordCloud
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import mlflow
from mlflow.tracking import MlflowClient
import pickle

# ── AWS MLflow Config ─────────────────────────────────────────────────────────

MLFLOW_TRACKING_URI = "http://ec2-13-50-5-147.eu-north-1.compute.amazonaws.com:5000"

app = FastAPI(title="Sentiment Analysis API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic Models ───────────────────────────────────────────────────────────

class CommentItem(BaseModel):
    text: str
    timestamp: str

class PredictRequest(BaseModel):
    comments: List[str]

class PredictWithTimestampsRequest(BaseModel):
    comments: List[CommentItem]

class SentimentCountsRequest(BaseModel):
    sentiment_counts: dict

class WordCloudRequest(BaseModel):
    comments: List[str]

class SentimentDataItem(BaseModel):
    sentiment: int
    timestamp: str

class TrendRequest(BaseModel):
    sentiment_data: List[SentimentDataItem]


# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess_comment(comment: str) -> str:
    try:
        comment = comment.lower().strip()
        comment = re.sub(r'\n', ' ', comment)
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)
        stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
        comment = ' '.join([w for w in comment.split() if w not in stop_words])
        lemmatizer = WordNetLemmatizer()
        comment = ' '.join([lemmatizer.lemmatize(w) for w in comment.split()])
        return comment
    except Exception as e:
        print(f"Error preprocessing: {e}")
        return comment


# ── Model Loading from AWS MLflow ─────────────────────────────────────────────

def load_model_from_mlflow(model_name: str, model_version: str, vectorizer_path: str):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"models:/{model_name}/{model_version}"
    model = mlflow.pyfunc.load_model(model_uri)
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

# Update model_name and model_version to match your MLflow registry
model, vectorizer = load_model_from_mlflow(
    model_name="my_model",
    model_version="1",
    vectorizer_path="../tfidf_vectorizer.pkl"
)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def home():
    return {"message": "Welcome to the FastAPI Sentiment Analysis API"}


@app.post("/predict")
def predict(body: PredictRequest):
    if not body.comments:
        raise HTTPException(status_code=400, detail="No comments provided")
    try:
        preprocessed = [preprocess_comment(c) for c in body.comments]
        transformed = vectorizer.transform(preprocessed)
        feature_names = vectorizer.get_feature_names_out()
        df_input = pd.DataFrame(transformed.toarray(), columns=feature_names)
        predictions = model.predict(df_input).tolist()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    return [{"comment": c, "sentiment": s} for c, s in zip(body.comments, predictions)]


@app.post("/predict_with_timestamps")
def predict_with_timestamps(body: PredictWithTimestampsRequest):
    if not body.comments:
        raise HTTPException(status_code=400, detail="No comments provided")
    try:
        texts = [item.text for item in body.comments]
        timestamps = [item.timestamp for item in body.comments]
        preprocessed = [preprocess_comment(t) for t in texts]
        transformed = vectorizer.transform(preprocessed)
        feature_names = vectorizer.get_feature_names_out()
        df_input = pd.DataFrame(transformed.toarray(), columns=feature_names)
        predictions = [str(p) for p in model.predict(df_input).tolist()]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    return [{"comment": c, "sentiment": s, "timestamp": t}
            for c, s, t in zip(texts, predictions, timestamps)]


@app.post("/generate_chart")
def generate_chart(body: SentimentCountsRequest):
    sc = body.sentiment_counts
    if not sc:
        raise HTTPException(status_code=400, detail="No sentiment counts provided")
    sizes = [int(sc.get('1', 0)), int(sc.get('0', 0)), int(sc.get('-1', 0))]
    if sum(sizes) == 0:
        raise HTTPException(status_code=400, detail="Sentiment counts sum to zero")

    labels = ['Positive', 'Neutral', 'Negative']
    colors = ['#36A2EB', '#C9CBCF', '#FF6384']
    plt.figure(figsize=(6, 6))
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
            startangle=140, textprops={'color': 'w'})
    plt.axis('equal')
    img_io = io.BytesIO()
    plt.savefig(img_io, format='PNG', transparent=True)
    img_io.seek(0)
    plt.close()
    return StreamingResponse(img_io, media_type='image/png')


@app.post("/generate_wordcloud")
def generate_wordcloud(body: WordCloudRequest):
    if not body.comments:
        raise HTTPException(status_code=400, detail="No comments provided")
    preprocessed = [preprocess_comment(c) for c in body.comments]
    text = ' '.join(preprocessed)
    wc = WordCloud(width=800, height=400, background_color='black',
                   colormap='Blues', stopwords=set(stopwords.words('english')),
                   collocations=False).generate(text)
    img_io = io.BytesIO()
    wc.to_image().save(img_io, format='PNG')
    img_io.seek(0)
    return StreamingResponse(img_io, media_type='image/png')


@app.post("/generate_trend_graph")
def generate_trend_graph(body: TrendRequest):
    if not body.sentiment_data:
        raise HTTPException(status_code=400, detail="No sentiment data provided")
    df = pd.DataFrame([item.dict() for item in body.sentiment_data])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    df['sentiment'] = df['sentiment'].astype(int)

    sentiment_labels = {-1: 'Negative', 0: 'Neutral', 1: 'Positive'}
    monthly_counts = df.resample('M')['sentiment'].value_counts().unstack(fill_value=0)
    monthly_totals = monthly_counts.sum(axis=1)
    monthly_pct = (monthly_counts.T / monthly_totals).T * 100
    for val in [-1, 0, 1]:
        if val not in monthly_pct.columns:
            monthly_pct[val] = 0
    monthly_pct = monthly_pct[[-1, 0, 1]]

    colors = {-1: 'red', 0: 'gray', 1: 'green'}
    plt.figure(figsize=(12, 6))
    for val in [-1, 0, 1]:
        plt.plot(monthly_pct.index, monthly_pct[val], marker='o', linestyle='-',
                 label=sentiment_labels[val], color=colors[val])
    plt.title('Monthly Sentiment Percentage Over Time')
    plt.xlabel('Month')
    plt.ylabel('Percentage of Comments (%)')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=12))
    plt.legend()
    plt.tight_layout()
    img_io = io.BytesIO()
    plt.savefig(img_io, format='PNG')
    img_io.seek(0)
    plt.close()
    return StreamingResponse(img_io, media_type='image/png')