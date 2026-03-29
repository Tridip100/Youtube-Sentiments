# ── Base image ────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# ── System deps ───────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Install Python dependencies ───────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Download NLTK data ────────────────────────────────────────────────────────
RUN python -c "import nltk; \
    nltk.download('stopwords'); \
    nltk.download('wordnet'); \
    nltk.download('omw-1.4')"

# ── Copy application code ─────────────────────────────────────────────────────
COPY . .

# ── Expose port ───────────────────────────────────────────────────────────────
EXPOSE 8000

# ── Healthcheck ───────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
  CMD curl -f http://localhost:8000/ || exit 1

# ── Start FastAPI ─────────────────────────────────────────────────────────────
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]