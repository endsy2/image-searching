# Project Analysis Report — Multimodal Fashion Product Search with CLIP

**Project path:** `/Users/macbook/Documents/ITC8/Techpreneur Bootcam/Activity/Group 5/image-searching`
**Report generated:** 2026-10-30
**Branch:** `development` (2 commits: `d7a250f first commit`, `2d3ec57 init`)
**Status:** Built, running, and measured. The one outstanding item is the human relevance-judging pass (no Precision@5 exists yet).

---

## 1. Executive Summary

This is a self-contained, production-shaped **multimodal search application** over the Kaggle *Fashion Product Images (Small)* dataset. Given a **text description** ("navy blue sports shoes") or an **uploaded image**, it returns the most similar products from a 44,441-product catalog, with images and metadata, through a FastAPI backend and a vanilla-JS web UI.

The core idea is a **single shared vector space**. OpenAI's frozen CLIP `ViT-B/32` was pretrained so that a photo and the text describing it land close together. That means **one query vector can be compared against both product photos and product titles with the same dot product**. There is **no training step**: both encoders are frozen, and all the engineering is in building, validating, and searching the vector table around them.

| Property | Value |
| --- | --- |
| Model | `openai/clip-vit-base-patch32`, frozen, `eval()` + `torch.no_grad()` |
| Vector dim | 512, L2-normalized `float32` |
| Similarity | cosine = plain dot product (vectors are unit-norm) |
| Index | Milvus collection (HNSW + COSINE) **or** exact brute-force NumPy matmul |
| Backend switch | `FASHION_BACKEND`: `auto` (default), `milvus`, `numpy` |
| Catalog | 44,441 products |
| Vector table | 88,875 rows = 44,441 `image` + 44,434 `title` |
| Device | CPU (CUDA used automatically when available) |
| Tests | 85 (no GPU, weights, dataset, or Milvus required) |

The project is unusually well-documented and unusually careful about **correctness invariants** (atomic publishing, manifest-based artifact refusal, "missing modality is `None`, never `0.0`", filters-before-truncation). It reads like a reference implementation rather than a quick hack.

---

## 2. Repository Layout

```
image-searching/
├── README.md                  # How to run it (310 lines)
├── ARCHITECTURE.md            # End-to-end design + invariants (631 lines)
├── CLAUDE.md                  # Guidance for Claude Code (111 lines)
├── CLIP_PROJECT_PLAN.md       # Authoritative spec + completion checklist (312 lines)
├── requirements.txt           # Pinned, verified versions
├── app.py                     # FastAPI backend (164 lines)
├── fashion_data_cleaning.ipynb# Canonical, audited data-prep path (1 MB, 22 cells)
├── Dockerfile                 # 4-stage CPU-only build
├── docker-compose.yml         # App + Milvus stack (179 lines)
├── .gitignore / .dockerignore
│
├── src/                       # The pipeline (12 modules, ~2,100 lines)
│   ├── config.py              # Paths, model identity, limits
│   ├── prepare_data.py        # Condensed re-implementation of the notebook
│   ├── encoder.py             # CLIP load, encode_images/encode_texts, validation
│   ├── verify_encoder.py      # Step-2 sanity checks + batch benchmark
│   ├── build_embeddings.py    # Builds image + title rows into one matrix
│   ├── store.py               # Unified table: validate, publish atomically, load
│   ├── milvus_store.py        # Milvus backend: schema, import, server-side search
│   ├── milvus_import.py       # CLI: load a built table into a collection
│   ├── fusion.py              # weighted sum, RRF, raw sum, single-modality refs
│   ├── search.py              # Retrieval, grouping, filters, exclusions
│   ├── baseline.py            # TF-IDF keyword baseline over search_text
│   └── evaluate.py            # pool / score / images / timing
│
├── static/                    # Vanilla-JS frontend, no build step
│   ├── index.html             # Text/image tabs, options, result grid
│   ├── app.js                 # 246 lines, ES modules
│   └── styles.css             # 179 lines
│
├── tests/                     # 85 tests, fake encoder, fake Milvus client
│   ├── helpers.py             # Synthetic catalogs, fake encoders, fixtures
│   ├── conftest.py
│   ├── test_store.py          # 14 tests
│   ├── test_search.py         # 18 tests
│   ├── test_fusion.py         # 16 tests
│   ├── test_api.py            # 15 tests
│   ├── test_evaluate.py       # 9 tests
│   └── test_milvus.py         # 13 tests
│
├── evaluation/
│   ├── queries.csv            # 32 queries: 12 development + 20 final_test
│   ├── judgments_development.csv   # 204 pooled pairs, 0 judged
│   ├── judgments_final_test.csv    # 326 pooled pairs, 0 judged
│   ├── image_queries.csv    # 20 image queries, pooled, unjudged
│   ├── measurements.json     # Modality gap, strategy overlap, metadata agreement
│   ├── timings.json          # Indexing, latency, cold start
│   └── report.md             # Measured results + honest weaknesses
│
├── artifacts/                 # Built outputs (large binaries gitignored)
│   ├── data_preparation/      # catalog_clean.csv, audit CSVs, figures
│   ├── embeddings/            # vectors.npy + rows.parquet + manifest.json
│   └── embeddings_sample/     # 500-product dev build
│
├── deploy/milvus/             # Standalone Milvus compose file
└── docker/healthcheck.py      # Container healthcheck
```

**Note on this checkout:** the large binaries are gitignored by design. `artifacts/embeddings/` and `artifacts/embeddings_sample/` contain only their `manifest.json` (the provenance record); `vectors.npy` and `rows.parquet` are not present and must be rebuilt. `dataset/` (642 MB of Kaggle data) is also absent and must be downloaded from Kaggle. The manifests are committed deliberately — they are the evidence behind the reported numbers.

---

## 3. Technology Stack

| Layer | Choice | Notes |
| --- | --- | --- |
| Language | Python 3.13.5 | `src/` and `tests/` are packages; run as `python -m src.<module>` |
| ML | torch 2.14.0+cpu, transformers 4.57.6 | CLIP via `transformers.CLIPModel` |
| Numerics | numpy 2.5.0, pandas 2.3.3, pyarrow 25.0.1 | Parquet for the row table |
| Imaging | Pillow 12.2.0 | `use_fast=False` so torchvision is **not** required |
| Vector DB | pymilvus 3.0.2 (optional) | Falls back to NumPy when absent |
| API | fastapi 0.141.1, uvicorn 0.53.0, python-multipart 0.0.32 | |
| Testing | pytest 8.4.2, httpx 0.28.1 | |
| Frontend | Vanilla JS (ES modules) | No build step, no framework |
| Container | python:3.13-slim, 4-stage Dockerfile | CPU-only, non-root (uid 10001) |
| Infra | Milvus v2.5.4, etcd v3.5.18, MinIO | Standalone, loopback-bound |

**Deliberate omissions:** torchvision (only needed for the fast image processor), any ANN library (exact matmul is fast enough at 44k), any frontend framework/build tooling.

---

## 4. Architecture & Data Flow

```
dataset/styles.csv ──┐
dataset/images/   ───┤
                     ▼
     fashion_data_cleaning.ipynb          (canonical data prep)
     src/prepare_data.py                  (condensed alternative)
                     ▼
     artifacts/data_preparation/catalog_clean.csv
                     │
                     ▼
     src/build_embeddings.py  (uses src/encoder.py, publishes via src/store.py)
                     ▼
     artifacts/embeddings/
       vectors.npy + rows.parquet + manifest.json
                     │
                     ▼
     src/store.py  ◄────── loads ──── src/search.py ──► src/fusion.py
                     │                              ▲
                     │  src/milvus_import.py        │  one of two backends,
                     └──────────► Milvus ───────────┘  chosen by FASHION_BACKEND
                                  src/milvus_store.py
                                                │
                          ┌─────────────────────┴───────────────┐
                          ▼                                     ▼
                      app.py (FastAPI)                   src/evaluate.py
                      static/ (vanilla JS UI)            src/baseline.py (TF-IDF)
```

### The unified vector table (the central design decision)

All embeddings sit in **one** vector column. Row `i` of `vectors.npy` is described by row `i` of `rows.parquet`; that shared row order is the **only** link between a vector and its product, so anything writing one writes the other in the same pass.

| File | Contents |
| --- | --- |
| `artifacts/embeddings/vectors.npy` | `float32` `(88875, 512)`, every row unit-norm |
| `artifacts/embeddings/rows.parquet` | `product_id`, `embedding_type`, `name`, `image_path` |
| `artifacts/embeddings/manifest.json` | model identity, versions, catalog fingerprint, counts, timings |

The schema maps one-to-one onto a vector database: one collection, one vector field, `embedding_type` as a scalar field. It is deliberately **not** two collections — splitting the modalities would push the modality gap onto whatever queried them.

### Search pipeline

```
query ──► CLIP vector ──► score the ENTIRE unified table (one matmul)
      ──► apply exclusions + metadata filters
      ──► top-K per modality ──► union by product_id
      ──► read BOTH modality scores for every candidate
      ──► fuse ──► top-k unique products
```

Text and image queries differ only in which encoder produces the query vector. After that, the pipeline is identical — which is precisely what the shared CLIP space buys.

---

## 5. Module-by-Module Analysis

### 5.1 `src/config.py` (53 lines)
Centralizes paths and constants. `find_root()` walks up the directory tree until it finds `dataset/styles.csv`, so every module resolves catalog-relative paths identically. Defines model identity (`MODEL_ID`, `MODEL_SOURCE`, `EMBED_DIM=512`), the 77-token CLIP context window, retrieval defaults (`DEFAULT_TOP_K=10`, `MAX_TOP_K=100`, `DEFAULT_CANDIDATE_K=100`, `RRF_K=60`), and upload limits (10 MB, 500 chars).

### 5.2 `src/encoder.py` (171 lines)
`ClipEncoder` wraps `CLIPModel` + `CLIPProcessor`. Loaded once per process (`get_encoder` is `lru_cache`d), put in `eval()`, every parameter `requires_grad_(False)`.

Two loading flags are deliberate:
- `use_safetensors=False` — the cached checkpoint is the `.bin`.
- `use_fast=False` — the fast image processor would require torchvision, which is intentionally not installed. Preprocessing stays on PIL/NumPy.

Both entry points return `(B, 512)` unit-norm `float32`, so cosine similarity is a plain dot product everywhere downstream.

**`EncoderIdentity`** is a frozen record of everything that must match between a built artifact and a running process: `model_id, model_source, dim, image_size, image_mean, image_std, context_length`. It is written into the manifest at build time and compared at load time; a mismatch is a hard refusal, not a warning.

**Query validation vs. catalog truncation** — handled differently on purpose:
- Catalog titles are truncated silently (they are data; a long title losing its tail is acceptable).
- User queries go through `validate_text`, which normalizes whitespace, rejects blanks, rejects >500 chars, and rejects >75 content tokens with a message that says how long the query was and what the limit is. A silent truncation would mean the user searched for something other than what they typed. This raises `QueryValidationError`, which the API maps to **422**.

### 5.3 `src/store.py` (298 lines) — the trust boundary
This module never trusts what it just wrote or what it just read.

- **`validate()`** checks: matrix is 2-D `float32`; row counts match; every value finite; every vector unit-norm within `1e-4`; `embedding_type` contains only known values; no `(product_id, embedding_type)` pair repeats; image-row count equals the expected product count.
- **`publish()`** writes into a temp directory beside the target, with the **manifest written last** so a half-built directory can never look complete. It then reloads what actually landed on disk, re-validates it, byte-compares the reloaded matrix against the built one, and only then `os.replace`s the directory into place.
- **`load()`** compares the manifest against the running encoder's `EncoderIdentity` and the catalog fingerprint on disk. Any difference raises `ArtifactError` listing each mismatch. Searching a table built with different preprocessing would return quietly wrong results, so this is a refusal rather than a warning.
- **`UnifiedStore.__post_init__`** precomputes what search needs on every query: `type_index[modality]` (row positions per modality), `row_of[modality][product_id]` (O(1) product→row lookup), and `product_meta`. `score_all(q)` is then a single `vectors @ q` over all 88,875 rows.
- **`search_modalities()`** is the retrieval contract shared with Milvus: top-K per modality via `argpartition` (not a full sort), then the union by product_id, returning both modality scores for every candidate. A modality a product genuinely lacks is `None`, never `0.0`.

### 5.4 `src/build_embeddings.py` (210 lines)
Every product contributes up to two rows to one matrix: an `image` row (photo through the vision encoder) and a `title` row (`productDisplayName` through the text encoder). Seven products have a blank title and carry an image row only; that absence is recorded in the manifest's `failures` list and becomes a `None` score downstream — **never a silent zero**. A product whose image fails to load is excluded entirely (no title-only row for a product with no image row). The build times the image and title passes separately and records both.

### 5.5 `src/fusion.py` (173 lines) — the one genuinely subtle piece of modeling
**The modality gap.** CLIP places text near text more tightly than text near images. Measured on this catalog:

| Query→ | mean | std |
| --- | --- | --- |
| title vectors | 0.468 | 0.104 |
| image vectors | 0.171 | 0.035 |

The **mean offset is harmless** — a constant shift cannot reorder candidates. What distorts a naive sum is the difference in **spread**: a ratio of ≈3.09. In a raw 50/50 sum, the title modality therefore exerts roughly three times the ranking influence it was nominally given.

Five strategies behind one `fuse()` interface:

| Strategy | Behavior |
| --- | --- |
| `weighted_sum` | **Default.** Per-modality min-max normalization over the candidate pool, then a convex combination |
| `rrf` | Reciprocal Rank Fusion, `1/(k + rank)` with `k = 60`; scale-free by construction |
| `raw_weighted_sum` | The naive reference, kept so the report can demonstrate why the other two exist |
| `image_only` / `title_only` | Single-modality ablations |

A degenerate pool (every candidate scoring identically on a modality) normalizes to a constant 1.0 — that modality carries no ranking information, so it must not amplify floating-point noise into an ordering. Under the weighted sums, a missing modality's weight is **redistributed** onto the modalities the product does have, so a title-less product is ranked on its image score rather than penalized for a gap in the catalog. Every result carries a `breakdown` dict so any ranking in the UI can be explained rather than taken on faith.

### 5.6 `src/search.py` (286 lines)
`SearchEngine` holds the encoder and the unified vector table for the life of the process. Key details:
- **Backend selection** (`_open_store`): Milvus when running, NumPy otherwise. Only "Milvus is not there" (`MilvusUnavailable`) falls back. A collection that exists but does not match raises `ArtifactError` either way — serving it would return quietly wrong results.
- **Filters are applied before truncation.** `_allowed_ids()` resolves metadata filters into an allowed ID set against the *full* catalog. Filtering the top-k after the fact would return 2 results where 10 exist.
- **Candidates are collected per modality, then unioned**, so one modality can't crowd the other out of the pool.
- **Every candidate is scored on both modalities** straight out of the already-computed full score vector — no re-encoding, no second pass.
- **Image serving is not user-controlled path input.** `image_path_for()` looks the path up by product ID and asserts the resolved path is inside `dataset/images/`, so `/images/{id}` cannot be walked out of the image directory.
- **Duplicate exclusion.** `exclude_product_id` removes the query item and, by default, every product in its `image_duplicate_group` (the 1,431-product byte-identical problem).

### 5.7 `src/milvus_store.py` (347 lines)
The same table in a Milvus collection. Both backends answer exactly one question (`search_modalities`), so nothing downstream knows which is in use.

| Field | |
| --- | --- |
| `row_id` | `INT64` primary key — the row's position in `rows.parquet` |
| `vector` | `FLOAT_VECTOR(512)`, HNSW, COSINE |
| `product_id` | `INT64` |
| `embedding_type` | `VARCHAR` — `"image"` or `"title"` |
| `articleType`, `baseColour`, `gender`, `masterCategory`, `subCategory` | `VARCHAR` |

The metadata columns are in the collection so filters run *inside* the search rather than after it. One search per modality, then the union — the same shape as the NumPy path. A candidate that only one search found has the other modality's vector fetched (a few hundred rows at most) and scored locally, so both scores are always real. Identity still comes from the manifest: `open_store` runs the same `manifest_mismatches` check plus a row count, and raises `ArtifactError` on any difference.

### 5.8 `src/baseline.py` (78 lines)
`KeywordBaseline` is a from-scratch TF-IDF bag-of-words scorer over `search_text` (sublinear TF, smoothed IDF, L2-normalized document vectors, inverted postings). Deliberately plain: the point is to show what lexical matching achieves on the same catalog under the same filters, not to be a competitive search engine.

### 5.9 `src/evaluate.py` (343 lines)
Four subcommands, because **judging is a human step that sits in the middle**:
- `pool --split development` — runs all five CLIP strategies plus the TF-IDF baseline at depth 5, unions results per query, **shuffles** the rows, hides the producing system in `_found_by`, leaves `grade` blank. Re-running preserves existing grades unless `--force`.
- `score --split development` — reads the sheet back, reports mean Precision@5. A blank cell means *unjudged*, never 0; `score` returns `None` for any query whose top 5 is not fully judged and excludes it from every system's mean.
- `images --n 20` — image-query inspection sheet (metadata agreement, not relevance).
- `timing` — indexing / warm latency / cold start.

### 5.10 `app.py` (164 lines)
The encoder and vector table load **once** at startup in a worker thread (`asyncio.to_thread` inside the `lifespan` context manager). If the artifacts are missing or mismatched the app still starts, `/health` reports `ready: false` with a user-safe reason, and the search endpoints return 503 — a bad artifact never becomes a stack trace in the browser. `app.state.engine_factory` is the test seam: tests swap in a fake engine, so the API tests need neither weights nor the dataset.

| Endpoint | Purpose |
| --- | --- |
| `POST /search/text` | `{query, top_k, strategy, image_weight, filters}` → ranked results |
| `POST /search/image` | multipart upload + `top_k`, `strategy`, `image_weight`, `exclude_product_id` |
| `GET /images/{product_id}` | Catalog JPEG, `Cache-Control: max-age=86400` |
| `GET /filters` | Distinct values for each filter column |
| `GET /health` | Readiness, model identity, device, row counts, `built_at`, startup seconds |
| `/` | `static/` mounted with `html=True` |

Error mapping: `QueryValidationError` → **422**; missing engine → **503**; unknown product → **404**. Uploads capped at 10 MB; decode failures become a user-safe 422.

### 5.11 Frontend (`static/`)
Plain ES modules, no build step. A text/image tab pair, a strategy selector, a top-k input, an image-weight slider, three metadata filters, and a result grid that surfaces the score breakdown. State handling: loading spinner, clear empty state, error state for upload/validation failures.

---

## 6. Data Pipeline (Stage 1)

Two implementations exist and **the notebook is the source of truth**:
- `fashion_data_cleaning.ipynb` → `artifacts/data_preparation/`. Produces every audit CSV, the EDA figures, `audit_summary.json`, asserts record conservation across stages, and excludes ambiguous IDs.
- `src/prepare_data.py` → `artifacts/catalog.csv`. A condensed re-implementation that **omits the ambiguous-ID exclusion and all audit exports**. It is not what downstream code reads.

Downstream code reads `artifacts/data_preparation/catalog_clean.csv` (44,441 rows) or `catalog_sample_500.csv` (fixed `SEED = 42` development sample).

**What cleaning has to survive:** `styles.csv` contains unquoted commas inside `productDisplayName`. Parsing uses `csv.reader(..., strict=True)` on `utf-8-sig` and asserts the header against a fixed `EXPECTED_SCHEMA`. Rows with more than 10 fields are repaired by keeping the first 9 and rejoining the rest into the name (22 such rows); rows with fewer than 10 fields are quarantined, never silently dropped.

**Record conservation (from `audit_summary.json`):**
- 44,446 source records → 44,441 catalog products.
- 22 repaired CSV records, 0 quarantined short records.
- 5 dropped rows = metadata with no readable image.
- 0 unreadable images, 0 ambiguous image IDs, 0 images without metadata.
- **1,431 products share image bytes** with another product.
- 99.95% of images are 60×80 pixels.

**Catalog contract** (load-bearing consequences):
1. Images are always opened as `ImageOps.exif_transpose(Image.open(p)).convert('RGB')` — the exact path the audit validated readability on.
2. Because 1,431 products are byte-identical twins, **image-to-image evaluation must exclude the query item and its duplicate group**, or the top results are trivially the same photo again.

---

## 7. Measured Performance (CPU, 44,441 products, batch size 64)

From `evaluation/timings.json` and `evaluation/measurements.json`:

| Metric | Value |
| --- | --- |
| Full index build | 2,486 s ≈ 41 min (1,478 s images + 1,008 s titles) |
| Image encoding throughput | 30.1 images/s |
| Cold startup | 8.84 s |
| Warm text query | mean 37.1 ms, median 36.2 ms, p95 41.7 ms (n=32) |
| Warm image query | mean 56.2 ms, median 52.4 ms, p95 88.0 ms (n=20) |
| Modality-gap spread ratio | 3.09 |

Brute-force search over 88,875 × 512 floats is ~45 MB of matmul per query — a few milliseconds. Query latency is dominated by *encoding*, not retrieval, which is why no ANN index is warranted at this catalog size.

**Milvus vs. NumPy agreement** (32 evaluation queries, `candidate_k = 100`):
- Mean top-10 overlap with the NumPy backend: **1.000**
- Queries returning an identical ordering: **32 / 32**
- Largest `final_score` difference: 1e-6 (float rounding)
- Warm text query: 62.5 / 74.6 ms (NumPy: 51.9 / 54.1 ms)

Milvus costs about 11 ms per query here and buys nothing back at 44k products — the NumPy matmul is already only a few milliseconds. It earns its place an order of magnitude further up, where the table no longer fits in memory.

**Strategy overlap** (top-10, averaged over 32 queries) — the `raw_weighted_sum` row is the key finding: despite equal nominal weights, it shares **67%** of its top 10 with pure title search and only **13%** with pure image search. The naive fusion is, in practice, title search wearing a fusion label. The normalized weighted sum sits between the two single-modality systems (0.27 image / 0.40 title) — genuinely blending rather than being captured by one side.

---

## 8. Testing

**85 tests**, and none of them need a GPU, the CLIP weights, the dataset, or Milvus.

| File | Tests | Covers |
| --- | --- | --- |
| `test_store.py` | 14 | Validation rules, atomic publish, manifest mismatch refusal |
| `test_search.py` | 18 | Candidate collection, filters-before-truncation, exclusions, path safety |
| `test_fusion.py` | 16 | All five strategies, missing modalities, weight redistribution, degenerate pools |
| `test_api.py` | 15 | Endpoints, 422/503/404 mapping, readiness, via `engine_factory` |
| `test_evaluate.py` | 9 | Pooling, blank-is-unjudged, Precision@5 abstention |
| `test_milvus.py` | 13 | Search expressions, the two-modality union, backend selection, via a fake client |
| `helpers.py` | — | Synthetic catalogs, fake encoders, artifact fixtures |

The suite uses a `FakeEncoder` (deterministic hash-to-direction vectors) and a synthetic 6-product catalog with a blank-title product and a duplicate pair, so it exercises the real build path without any external dependency.

---

## 9. Deployment

**`Dockerfile`** is a four-stage CPU-only build (`base` → `deps` → `weights` → `runtime`). torch comes from the CPU wheel index (the PyPI wheel drags in a multi-GB CUDA runtime this image has no use for), torchvision is deliberately absent, and the CLIP weights are prefetched into the image so the container runs with `HF_HUB_OFFLINE=1` and no network. It runs as a non-root user (uid 10001) with a healthcheck that allows a 180 s start period.

**`docker-compose.yml`** bind-mounts rather than bakes: the 642 MB dataset and 209 MB artifacts stay outside the image, so a rebuilt embedding table appears without rebuilding the image. The `app` service mounts `artifacts/` **read-only** so a running server can never race a build. One-shot jobs (`build-embeddings`, `evaluate`, `verify-encoder`, `tests`, `milvus-import`) sit behind the `tools` profile. Milvus keeps its data in `./deploy/milvus/volumes` (a bind mount that survives `down -v`).

> **Note:** `deploy/milvus/docker-compose.yml` is the same three services under their own project name and writes to the same `./deploy/milvus/volumes`. Run one or the other, never both at once.

---

## 10. Invariants — the things that silently break if violated

From `ARCHITECTURE.md` §13:

1. **Row order is the only vector→product link.** `vectors.npy` row *i* ↔ `rows.parquet` row *i*. Write both in the same pass, always.
2. **`id` is the stable product identifier** across catalog, embeddings, artifacts, and evaluation sheets.
3. **Vectors are unit-norm `float32`**, validated at publish and load. Cosine is a dot product only because of this.
4. **Artifacts carry a manifest, and search refuses a mismatch.** Model identity, preprocessing identity, and catalog fingerprint must all agree.
5. **Publish only after validating the reload.** Manifest written last.
6. **A missing modality is `None`, never `0.0`**, at every layer.
7. **Never sum raw cross-modality scores** without per-modality normalization or a rank-based fusion — the ≈3× spread ratio is real and measured.
8. **Filters apply before truncation**, not to the top-k.
9. **Image-to-image queries exclude the query item and its duplicate group.**
10. **Over-long queries are rejected, not truncated.** Blank queries and invalid uploads get understandable messages.
11. **A blank grade cell is unjudged, never 0**, and a metric is never reported over a partially judged top-5.
12. **Never fabricate judgments or measured numbers**, and never present metadata agreement as relevance.
13. **Both backends answer `search_modalities` identically.** Retrieval changes go behind that method, never into `search.py`, or the two drift apart silently.
14. **A Milvus collection that does not match the manifest is refused**, not served and not quietly replaced by the NumPy path.

---

## 11. Known Limitations

- **Images are 60×80**, upsampled to 224×224 for CLIP. Fine patterns, textures, and brand text are unrecoverable. This is measured and reported, not tuned away.
- **Titles are brand-heavy** ("Nike Sahara Team India Fanwear Round Neck Jersey"), so the title modality rewards brand and category matches and says little about colour or visual style. The two modalities are genuinely complementary.
- **1,431 products share image bytes** with another product. Image-to-image evaluation excludes the query item and its exact duplicates.
- **Metadata labels are filter and evaluation aids**, not proof that a requested visual attribute is actually present in the photo.
- **Milvus is exercised, not needed.** At 44,441 products the exact NumPy matmul is a few milliseconds and Milvus adds ~11 ms per query. The collection is there because it is the shape this grows into, not because it is faster today.
- **Relevance is unmeasured.** The pooled judging sheets are complete and blind, but ungraded, so no Precision@5 comparison against the baseline exists yet.
- **Observed failure mode: attribute bleed.** `D01 navy blue sports shoes` returns ten items all correctly Navy Blue, but every one is `articleType = Casual Shoes`, not `Sports Shoes`. The colour and the word "shoes" dominate; the usage qualifier is lost.
- Fine-tuning, deployment, authentication, and personalization are out of scope.

---

## 12. Observations & Discrepancies Found During Analysis

These are things I noticed while reading the code and docs that are worth flagging:

1. **Test count is inconsistent across docs.** `README.md` (line 308) and the actual test files say **85 tests** (14+18+16+15+9+13 = 85). But `CLAUDE.md` (line 11) and `evaluation/report.md` (line 242) both say **72 tests**. The README and the code are correct; `CLAUDE.md` and `report.md` are stale.

2. **Two data-prep paths can diverge.** `src/prepare_data.py` omits the ambiguous-ID exclusion and the audit exports that the notebook produces. Nothing downstream reads it (the embedding build reads `artifacts/data_preparation/catalog_clean.csv`), but if cleaning logic changes, both must change or one must be deleted. This is flagged as "the main hazard in this repo" in the plan.

3. **Dormant dual-folder image resolution.** `prepare_data.py` still lists `archive/images` alongside `dataset/images`; `archive/` does not exist in this checkout. The code is dormant, not dead — the plan says to leave it until the audit step is formally closed.

4. **Large binaries are not in this checkout.** `vectors.npy`, `rows.parquet`, the audit CSVs, and the entire `dataset/` are gitignored. Only the manifests and `audit_summary.json` are committed. A fresh clone cannot run the app until the dataset is downloaded from Kaggle and the embeddings are rebuilt (`python -m src.build_embeddings`, ~41 min on CPU).

5. **The headline result is pending.** The assignment's core question — does fusion beat a keyword baseline — cannot be answered until the 204 + 326 pooled pairs are hand-graded. The report is honest about this and does not estimate it.

6. **`prepare_data.py` fills blank titles with a fabricated name** (`"Unnamed product <id>"`), whereas the canonical notebook path and the embedding build treat a blank title as "no title row" (a `None` score, never a fabricated title). This is another facet of the divergence in point 2 — the condensed path would embed a made-up string the canonical path deliberately avoids.

---

## 13. Strengths

- **Exceptional documentation.** Four complementary docs (README, ARCHITECTURE, CLAUDE, PLAN) plus a measured evaluation report. Every number is reproducible by a named command.
- **Correctness-first design.** Atomic publishing, manifest-based artifact refusal, "missing modality is `None` never `0.0`", filters-before-truncation, path-safe image serving. The code treats its own outputs as untrusted.
- **Honest evaluation.** The report explicitly labels metadata agreement as *not* a relevance metric, refuses to report Recall@k (the denominator is unknown), and states plainly that the headline Precision@5 is pending human judgment.
- **Backend-agnostic architecture.** One retrieval contract (`search_modalities`) lets NumPy and Milvus be swapped without touching fusion, the API, or the evaluation.
- **Zero-friction tests.** 85 tests run with no GPU, weights, dataset, or Milvus, via a fake encoder and synthetic catalog.
- **Lean dependencies.** No torchvision, no ANN library, no frontend build step.

## 14. Recommendations

1. **Grade the judging sheets** to produce the headline Precision@5 and the fusion-strategy comparison. This is the single outstanding deliverable.
2. **Reconcile or delete `src/prepare_data.py`** to eliminate the silent-divergence hazard (and its fabricated-blank-title behavior).
3. **Fix the stale test count** in `CLAUDE.md` and `evaluation/report.md` (72 → 85).
4. **Add UI screenshots** (the build environment had no browser).
5. **Verify a clean-checkout rebuild** reproduces the reported numbers (the plan's final checklist item, still unchecked).

---

*Report generated by static analysis of the repository. No code was executed; all figures are taken from the project's own committed artifacts (`manifest.json`, `timings.json`, `measurements.json`, `audit_summary.json`) and documentation.*
