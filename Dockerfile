# ──────────────────────────────────────────────────────────────────────────────
# Dockerfile - ml-service (Python FastAPI)
# ──────────────────────────────────────────────────────────────────────────────
# build context: smart-room-access-backend/ (root repo di CI/CD)
#
# model ONNX tidak di-commit ke repo (terlalu besar, ~192MB total)
# model di-download dari GCS saat docker build:
#   - Cloud Build service account otomatis punya akses GCS di project yang sama
#   - untuk build lokal: pastikan sudah `gcloud auth application-default login`
#
# model sudah ada di bucket: gs://ml-models-iot/
# ──────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# install system deps: opencv deps & curl (untuk download model)
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# install python deps (layer cache-friendly, copy requirements dulu)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# download model ONNX dari GCS saat build (karena bucket public, cukup pakai curl)
ARG GCS_BUCKET=ml-models-iot
RUN mkdir -p /app/models && \
    curl -o /app/models/det_10g.onnx   https://storage.googleapis.com/${GCS_BUCKET}/det_10g.onnx && \
    curl -o /app/models/w600k_r50.onnx https://storage.googleapis.com/${GCS_BUCKET}/w600k_r50.onnx && \
    curl -o /app/models/genderage.onnx https://storage.googleapis.com/${GCS_BUCKET}/genderage.onnx

# copy source ml-service
COPY app     ./app
COPY main.py ./main.py

# jalankan sebagai non-root
RUN addgroup --system --gid 1001 mlgroup && \
    adduser  --system --uid 1001 mluser  --ingroup mlgroup && \
    chown -R mluser:mlgroup /app
USER mluser

EXPOSE 8001

ENV MODEL_DIR=/app/models
ENV PORT=8001

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8001/health')" || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]

