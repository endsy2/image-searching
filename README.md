# Multimodal Fashion Product Search with CLIP

Search 44,441 fashion products by **text description** or by **uploaded image**, using
frozen OpenAI CLIP `ViT-B/32`. Every product is embedded twice — once from its photo
through the vision encoder, once from its title through the text encoder — and both
kinds of vector live in **one unified table**, tagged by `embedding_type`. A query is
encoded into the same 512-d space, scored against the whole table, grouped back by
product, fused across the two modalities, and returned as the top 10 unique products.

There is no training step. All ranking is cosine similarity between L2-normalized
frozen embeddings.

```
[Query: text or image]
        │
        ▼
[CLIP query embedding]  512-d, unit-norm
        │
        ▼
[Score the unified vector table]   ── one dot product over 88,875 rows
   ├── embedding_type = "image"
   └── embedding_type = "title"
        │
        ▼
[Group by product_id → (S_image, S_title)]
        │
        ▼
[Fusion & reranking]   normalized weighted sum · RRF · raw sum
        │
        ▼
[Top 10 unique products] ──► [Web UI]
```

## Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or a CUDA build
pip install -r requirements.txt
```

The CLIP weights (~600 MB) download from the Hugging Face hub on first use and are
cached in `~/.cache/huggingface`. After that everything runs offline; set
`HF_HUB_OFFLINE=1` to guarantee it.

The dataset is [Fashion Product Images (Small)](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small)
from Kaggle, expected at `dataset/styles.csv` and `dataset/images/`. It is not
redistributed with this project; download it from Kaggle under its own licence.

Run everything from this directory (`week3/`).

## Commands

```powershell
# 1. Data preparation (already done; artifacts are committed under artifacts/data_preparation/)
jupyter lab fashion_data_cleaning.ipynb     # the canonical, fully audited path

# 2. Verify the encoder and benchmark a batch before committing to a full build
python -m src.verify_encoder --limit 120 --batch-size 32

# 3. Build the unified vector table
python -m src.build_embeddings --sample            # 500 products, ~30 s, for development
python -m src.build_embeddings --batch-size 64     # full catalog, ~30 min on CPU

# 4. Optional: put the vectors in Milvus (see "Milvus" below)
docker compose -f deploy/milvus/docker-compose.yml up -d
python -m src.milvus_import --recreate

# 5. Launch the app  (http://127.0.0.1:8000)
python -m uvicorn app:app --reload

# 6. Evaluation
python -m src.evaluate pool   --split development  # build the blind judging sheet
#    ... fill in the `grade` column by hand ...
python -m src.evaluate score  --split development  # mean Precision@5 per system
python -m src.evaluate images --n 20               # image-query inspection sheet
python -m src.evaluate timing                      # indexing, latency, cold start

# 7. Tests (no GPU, no model weights, no dataset, no Milvus needed)
python -m pytest -q
```

To serve the 500-product development build instead of the full one:

```powershell
$env:FASHION_ARTIFACTS = "artifacts/embeddings_sample"
$env:FASHION_CATALOG   = "artifacts/data_preparation/catalog_sample_500.csv"
python -m uvicorn app:app --reload
```

## The unified vector table

All embeddings sit in a single vector column. Row `i` of `vectors.npy` is described by
row `i` of `rows.parquet`; that shared order is the *only* link between a vector and its
product, so anything writing one writes the other in the same pass.

| file | contents |
| --- | --- |
| `artifacts/embeddings/vectors.npy` | `float32` `(N, 512)`, every row unit-norm |
| `artifacts/embeddings/rows.parquet` | `product_id`, `embedding_type`, `name`, `image_path` |
| `artifacts/embeddings/manifest.json` | model identity, versions, catalog fingerprint, counts, timings |

Exact NumPy search over ~89k rows is a single matrix-vector product, so no ANN index is
used. The schema maps one-to-one onto a vector database later: one collection, one
vector field, `embedding_type` as a scalar field. It is deliberately not two collections.

`store.load` refuses artifacts whose manifest model identity, preprocessing identity, or
catalog fingerprint does not match the running configuration, so a stale build cannot be
served silently.

## Milvus

Search runs on one of two backends, and they return the same thing:

| backend | what it is |
| --- | --- |
| `milvus` | The vector table lives in a Milvus collection; searching, filtering and excluding all happen server-side |
| `numpy` | The original exact brute-force matmul over `vectors.npy`, in-process |

`FASHION_BACKEND` picks one — `auto` (the default) uses Milvus when it is running
and falls back to NumPy when it is not. Asking for `milvus` explicitly never falls
back. A collection that exists but does **not** match the running encoder and
catalog is refused by either setting: serving it would return quietly wrong
results, which is worse than not serving at all.

```powershell
docker compose -f deploy/milvus/docker-compose.yml up -d   # Milvus on 127.0.0.1:19530
python -m src.milvus_import --sample --collection fashion_clip_sample   # 997 rows, seconds
python -m src.milvus_import --recreate                     # the full 88,875 rows

$env:FASHION_BACKEND = "milvus"; python -m uvicorn app:app --reload
```

`GET /health` reports which backend is live, and why, under `backend` and
`backend_note`.

Nothing is encoded during the import: it reads the `vectors.npy` / `rows.parquet`
that `src.build_embeddings` already produced and copies them in. The artifacts
directory stays the source of truth — its `manifest.json` is what the collection
is checked against on load — so rebuild the vectors first when the model or the
catalog changes, then re-import with `--recreate`.

### The collection

One collection holds the whole unified table, exactly as the NumPy artifacts do.
Two collections would have made the modality gap somebody else's problem.

| field | |
| --- | --- |
| `row_id` | `INT64` primary key — the row's position in `rows.parquet` |
| `vector` | `FLOAT_VECTOR(512)`, **HNSW** index, **COSINE** metric |
| `product_id` | `INT64` |
| `embedding_type` | `VARCHAR` — `"image"` or `"title"` |
| `articleType`, `baseColour`, `gender`, `masterCategory`, `subCategory` | `VARCHAR` |

The metadata columns are in the collection so that filters run *inside* the
search, keeping the rule that filters apply before truncation rather than to the
top-k. One search per modality, then the union — the same shape as the NumPy
path, so neither modality can crowd the other out of the candidate pool. A
candidate that only one search found has the other modality's vector fetched and
scored, so both scores are always real; a modality a product genuinely lacks is
still `None`, never `0.0`.

### Measured agreement

HNSW is an approximate index, so agreement is something to check rather than
assume. Over the 32 evaluation queries on the full 44,441-product catalog, with
`candidate_k = 100`:

| | |
| --- | --- |
| Mean top-10 overlap with the NumPy backend | **1.000** |
| Queries returning an identical ordering | **32 / 32** |
| Largest `final_score` difference | 1e-6 (float rounding) |
| Warm text query, mean / p95 | 62.5 / 74.6 ms (NumPy: 51.9 / 54.1 ms) |

Milvus costs about 11 ms per query here and buys nothing back at 44k products —
the NumPy matmul is already only a few milliseconds. It earns its place an order
of magnitude further up, where the table no longer fits in memory and the index
stops being brute force. Reproduce the comparison by running the same queries
through `SearchEngine(backend='numpy')` and `SearchEngine(backend='milvus')`.

## Fusion, and why it is not a plain sum

Each candidate product carries `S_image` and `S_title`. These are **not on a comparable
scale**. Measured on this catalog with `python -m src.verify_encoder`, a text query
scores:

| against | mean cosine | std |
| --- | --- | --- |
| title vectors | 0.458 | 0.115 |
| image vectors | 0.182 | 0.038 |

The difference in *means* is harmless on its own — a constant offset shifts every
candidate equally and cannot reorder them. What distorts a raw sum is the difference in
**spread**: the title modality varies three times as widely, so in a raw 50/50 sum it
exerts roughly three times the ranking influence it was given.

Three strategies are implemented behind one interface and selectable at query time:

- **`weighted_sum`** *(default)* — min-max normalize each modality **over the candidate
  pool**, then take a convex combination. Restores the intended weighting.
- **`rrf`** — Reciprocal Rank Fusion, `Σ 1/(60 + rank)`. Scale-free by construction, at
  the cost of discarding score margins.
- **`raw_weighted_sum`** — the naive reference, kept so the evaluation can show the
  difference rather than assert it.

`image_only` and `title_only` are also available as single-modality references.

A modality a product genuinely lacks is recorded as `null`, **never zero**. Seven catalog
products have a blank `productDisplayName`; they get an image row and no title row, and
under the weighted sums their title weight is redistributed onto their image score rather
than penalising them for a gap in the catalog.

## API

| Endpoint | Request |
| --- | --- |
| `POST /search/text` | `{"query": "navy blue sports shoes", "top_k": 10, "strategy": "weighted_sum", "image_weight": 0.5, "filters": {"baseColour": "Navy Blue"}}` |
| `POST /search/image` | `multipart/form-data` file upload; `top_k`, `strategy`, `image_weight`, `exclude_product_id` as query params |
| `GET /images/{product_id}` | thumbnail, resolved through the catalog and path-checked, never from raw user input |
| `GET /filters` | available article types, colours, and genders |
| `GET /health` | readiness, row counts, model identity, cold-start time |

```json
{
  "results": [
    {
      "rank": 1,
      "product_id": 1163,
      "name": "Nike Sahara Team India Fanwear Round Neck Jersey",
      "image_url": "/images/1163",
      "final_score": 0.8123,
      "breakdown": {"strategy": "weighted_sum", "image": 0.2910, "title": 0.6336,
                    "normalized": {"image": 0.84, "title": 0.79},
                    "weights": {"image": 0.5, "title": 0.5}},
      "metadata": {"articleType": "Tshirts", "baseColour": "Blue", "gender": "Men"}
    }
  ],
  "count": 10, "top_k": 10, "strategy": "weighted_sum",
  "candidate_pool": 187, "filters": {}, "query": "...", "query_type": "text",
  "took_ms": 41.2
}
```

Errors are structured JSON with a user-safe message and no traceback. Blank queries,
queries over CLIP's 77-token context window, unreadable uploads, oversized uploads, and
unknown strategies are all 422. If the artifacts are missing or mismatched the app still
starts, `/health` returns 503 with the reason, and both search endpoints return 503.

## Evaluation

`evaluation/queries.csv` holds 32 text queries: 12 `development` and 20 `final_test`.
**Tune on the development split only**; the final split stays untouched until the end.

`pool` collects the top 5 of every system (all five CLIP strategies plus the keyword
baseline over `search_text`) into one shuffled sheet with the producing system hidden, so
grading is blind. Written relevance rule:

- **2 — relevant**: correct article type *and* every stated attribute matches.
- **1 — partial**: correct article type but a stated attribute is wrong or absent, or a
  near-miss type.
- **0 — irrelevant**: wrong article type.

Precision@5 counts grade 2 only. **A blank cell means unjudged, never 0** — `score`
returns `None` for any query whose top 5 is not fully judged and excludes it from every
system's mean, so no system is flattered by a partially graded sheet. Nothing in this
repository fabricates a grade.

Recall@k is deliberately not reported: the labels are nowhere near complete enough over
44,441 products to support it.

## Known limitations

- **Images are 60×80**, upsampled to 224×224 for CLIP. Fine patterns, textures, and brand
  text are unrecoverable. This is measured and reported, not tuned away.
- **Titles are brand-heavy** ("Nike Sahara Team India Fanwear Round Neck Jersey"), so the
  title modality rewards brand and category matches and says little about colour or
  visual style. The two modalities are genuinely complementary.
- **1,431 products share image bytes** with another product. Image-to-image evaluation
  excludes the query item *and* its exact duplicates (`exclude_product_id` does both by
  default).
- **Metadata labels are filter and evaluation aids**, not proof that a requested visual
  attribute is actually present in the photo.
- **Milvus is exercised, not needed.** At 44,441 products the exact NumPy matmul
  is a few milliseconds and Milvus adds ~11 ms per query. The collection is
  there because it is the shape this grows into, not because it is faster today.
- Fine-tuning, deployment, authentication, and personalization are out of scope.

## Layout

```
src/config.py            paths, model identity, limits
src/prepare_data.py      condensed re-implementation of the cleaning notebook
src/encoder.py           CLIP load, encode_images / encode_texts, query validation
src/verify_encoder.py    step-2 checks and the batch benchmark
src/store.py             the unified table: validate, publish atomically, load
src/milvus_store.py      Milvus backend: schema, import, server-side search
src/milvus_import.py     CLI: load a built table into a collection
src/build_embeddings.py  builds image + title rows into one matrix
src/fusion.py            weighted sum, RRF, raw sum, single-modality references
src/search.py            retrieval, grouping by product_id, filters, exclusions
src/baseline.py          TF-IDF keyword baseline over search_text
src/evaluate.py          pool / score / images / timing
app.py                   FastAPI
static/                  vanilla-JS frontend, no build step
tests/                   85 tests, fake encoder, fake Milvus client, synthetic catalog
```
# image-searching
