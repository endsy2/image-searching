# syntax=docker/dockerfile:1
# Fashion product search with CLIP - CPU-only image.
#
# The dataset (642 MB) and the built artifacts (209 MB) stay out of the image;
# docker-compose.yml bind-mounts them at run time. Only the code and the Python
# environment live here. Milvus is a separate image, brought up by the same
# compose file; this one only holds pymilvus, the client.
#
#   docker compose build
#   docker compose up

FROM python:3.13-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/hf-cache

WORKDIR /app


# --- dependencies -----------------------------------------------------------
FROM base AS deps

# torch must come from the CPU wheel index; the default PyPI wheel pulls in the
# CUDA runtime (several GB) that this image has no use for. torchvision is
# deliberately absent - src/encoder.py uses the PIL/NumPy processor.
COPY requirements.txt ./
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.0 \
 && pip install -r requirements.txt


# --- model weights ----------------------------------------------------------
# Baked in so the container needs no network at run time (HF_HUB_OFFLINE=1).
# Build with --build-arg PREFETCH_MODEL=false to skip it and download into the
# mounted cache on first use instead.
FROM deps AS weights

ARG PREFETCH_MODEL=true
ARG MODEL_ID=openai/clip-vit-base-patch32
RUN if [ "$PREFETCH_MODEL" = "true" ]; then \
      python -c "from huggingface_hub import snapshot_download; snapshot_download('$MODEL_ID', allow_patterns=['*.json', '*.txt', 'pytorch_model.bin'])"; \
    fi


# --- runtime ----------------------------------------------------------------
FROM weights AS runtime

# Source only. dataset/, artifacts/ and evaluation/ arrive as bind mounts.
COPY src/ ./src/
COPY tests/ ./tests/
COPY static/ ./static/
COPY docker/healthcheck.py ./docker/healthcheck.py
COPY app.py README.md CLIP_PROJECT_PLAN.md ./

# config.find_root() walks up looking for dataset/styles.csv, so the mount
# points must exist for the lookup to land on /app.
RUN mkdir -p /app/dataset /app/artifacts /app/evaluation \
 && useradd --create-home --uid 10001 fashion \
 && chown -R fashion:fashion /app

USER fashion
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
  CMD ["python", "/app/docker/healthcheck.py"]

CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
