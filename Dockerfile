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

# install system deps: google-cloud-sdk (gsutil) + opencv deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        gnupg \
        libglib2.0-0 \
        libgomp1 \
    && echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
       | tee /etc/apt/sources.list.d/google-cloud-sdk.list \
    && curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
       | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg \
    && apt-get update && apt-get install -y --no-install-recommends \
        google-cloud-cli \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# install python deps (layer cache-friendly, copy requirements dulu)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# download model ONNX dari GCS saat build
# ARG GCS_BUCKET supaya bisa di-override: --build-arg GCS_BUCKET=nama-bucket
ARG GCS_BUCKET=ml-models-iot
RUN mkdir -p /app/models && \
    gsutil -m cp \
      gs://${GCS_BUCKET}/det_10g.onnx \
      gs://${GCS_BUCKET}/w600k_r50.onnx \
      gs://${GCS_BUCKET}/genderage.onnx \
      /app/models/

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

