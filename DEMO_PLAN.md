# 3-Minute Demo Plan — Fashion Product Search with CLIP

Covers **FastAPI**, **Docker Compose** (Milvus vector store), and **MLflow** (experiment tracking).

## Verification status (checked on this machine, 2026-10-31)

| Component | Status | Evidence |
| --- | --- | --- |
| FastAPI | ✅ works | `/health` → `ready:true` (500 products, numpy backend); `/search/text` → 200 with ranked results |
| MLflow | ✅ works | `python -m src.track_mlflow` logged a run (11 params, 23 metrics) to a local store; `mlflow` 3.17.2 |
| Docker Compose | ⚠️ partially | Both compose files are **valid**; `etcd` + `milvus` images pulled; but the **minio image failed to pull** (quay.io `401 UNAUTHORIZED`, Docker Hub `access denied`) — a registry/network issue on this machine, **not** a config problem. The full Milvus stack could not be brought up end-to-end here. |

> **Bottom line:** FastAPI and MLflow are demo-ready now. Docker Compose commands are correct and the files validate, but you must **pre-pull the minio image on a stable connection** before the live demo (see Pre-flight). If minio still won't pull, use the FastAPI + MLflow parts and present the Docker Compose commands as the orchestration story.

Everything below runs from the project root. The Python interpreter is the project venv: `./.venv/bin/python`.

---

## Pre-flight (one-time, BEFORE the demo — needs a stable connection)

Do these once so the live demo is fast and reliable.

```bash
cd "<project-root>"

# 1. Sample embeddings (already built; ~8 s to rebuild if needed)
./.venv/bin/python -m src.build_embeddings \
  --catalog "$(pwd)/artifacts/catalog_sample_500.csv" \
  --out artifacts/embeddings_sample --batch-size 64

# 2. Pre-pull the Milvus stack images (the minio one is the flaky one — retry if it fails)
docker compose -f deploy/milvus/docker-compose.yml pull

# 3. mlflow is already installed in the venv (pip install mlflow)
```

Confirm the images are present before you go live:
```bash
docker image inspect quay.io/minio/minio:RELEASE.2024-12-18T13-15-44Z >/dev/null && echo "minio OK"
```

---

## The 3-minute demo

### Part 1 — FastAPI (0:00–1:00) — "the search app"

**Terminal A:**
```bash
FASHION_ARTIFACTS="$(pwd)/artifacts/embeddings_sample" \
FASHION_CATALOG="$(pwd)/artifacts/catalog_sample_500.csv" \
./.venv/bin/python -m uvicorn app:app --port 8000
```

**Say:** "This is a FastAPI service. It loads a frozen CLIP model and a unified vector table — 500 products here, 44,441 in production — once at startup."

**Show (browser → http://127.0.0.1:8000):**
- Type `navy blue sports shoes` → top-10 grid with rank badges + per-modality score breakdown.
- Switch to the **Image search** tab, upload a product photo → similar products.
- Open the "breakdown" on a card to show the image/title scores being fused.

**Optional one-liner (if you prefer terminal over browser):**
```bash
curl -s localhost:8000/health
curl -s -X POST localhost:8000/search/text -H 'Content-Type: application/json' \
  -d '{"query":"red handbag","top_k":5}'
```

---

### Part 2 — Docker Compose + Milvus (1:00–2:00) — "the vector store, orchestrated"

**Say:** "The same vector table can live in Milvus instead of in-process NumPy. Docker Compose brings up the whole store — etcd, MinIO, and Milvus — and the app switches backends with one env var."

**Terminal B:**
```bash
# Bring up the Milvus stack (fast if you pre-pulled in Pre-flight)
docker compose -f deploy/milvus/docker-compose.yml up -d
docker compose -f deploy/milvus/docker-compose.yml ps        # show all "healthy"

# Load the sample vectors into the collection (a few seconds)
./.venv/bin/python -m src.milvus_import --sample \
  --catalog "$(pwd)/artifacts/catalog_sample_500.csv"

# Start the app pointed at Milvus
FASHION_BACKEND=milvus MILVUS_URI=http://127.0.0.1:19530 \
FASHION_ARTIFACTS="$(pwd)/artifacts/embeddings_sample" \
FASHION_CATALOG="$(pwd)/artifacts/catalog_sample_500.csv" \
./.venv/bin/python -m uvicorn app:app --port 8001
```

**Show:**
```bash
curl -s localhost:8001/health | ./.venv/bin/python -m json.tool
```
Point at `"backend": "milvus"` — the exact same search now runs server-side in Milvus.

> **Note:** `--catalog` is required because the sample catalog lives at `artifacts/catalog_sample_500.csv`, not the default `artifacts/data_preparation/` path.
> **Fallback if minio won't start:** keep Part 1's app (numpy backend) and present the `docker compose up`/`ps` commands as the orchestration story; the app's `/health` already reports which backend is live.

---

### Part 3 — MLflow (2:00–3:00) — "the experiment record"

**Say:** "Every build is reproducible and tracked. This logs the model identity, catalog fingerprint, indexing timings, and the measured modality gap as an MLflow run."

**Terminal C:**
```bash
./.venv/bin/python -m src.track_mlflow \
  --artifacts artifacts/embeddings_sample --run-name demo-run

# Open the MLflow UI (reads the local store the CLI just wrote to)
./.venv/bin/python -m mlflow ui --port 5000
```

**Show (browser → http://127.0.0.1:5000):**
- Experiment `fashion-clip-search` → the `demo-run` run.
- **Params:** `model_id`, `embedding_dim`, `batch_size`, `catalog_fingerprint`, …
- **Metrics:** `index.total_index_seconds`, `rows.image`, `rows.title`, `modality_gap.*`, `spread_ratio`, …

---

## Teardown (after the demo)

```bash
# stop the apps (Ctrl+C in each terminal), then:
docker compose -f deploy/milvus/docker-compose.yml down     # keep Milvus data
# mlflow ui stops with Ctrl+C
```

## Gotchas / honest caveats
- **minio image pull** is the one external dependency that failed on this machine (registry 401 / access-denied). Pre-pull it on a good connection; retry if needed. Everything else is self-contained.
- The demo uses the **500-product sample** for speed. The production numbers (44,441 products, 88,875 vectors) are in `evaluation/report.md` if you want to quote them.
- MLflow 3.x writes to a local SQLite store (`mlflow.db`) by default; `mlflow ui` reads it. No server needed.
- Ports used: app 8000 (Part 1), 8001 (Part 2), MLflow UI 5000. Change if any are taken.
