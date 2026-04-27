# ============================================================
# Ingestion Pipeline — Dockerfile for Cloud Run Job
#
# Runs run_pipeline.py to parse PNGs, embed, and upsert to Qdrant.
# PNGs are read from GCS (set GCS_BUCKET env var at deploy time).
# Secrets injected as env vars: QDRANT_URL, QDRANT_API_KEY,
#   VLLM_API_BASE, GCS_BUCKET
# ============================================================
FROM python:3.11-slim

WORKDIR /app

# System deps required by Docling (poppler) and Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY src/ ./src/
COPY config/ ./config/
COPY pyproject.toml README.md ./

RUN pip install --no-cache-dir -e .

CMD ["python", "-m", "src.run_pipeline"]
