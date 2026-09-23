# Architecture — Fashion Product Search with CLIP

How this project works, end to end: what each module owns, what data flows between
them, and which invariants hold the whole thing together.

Companion documents: `CLIP_PROJECT_PLAN.md` (the authoritative spec),
`README.md` (how to run it), `evaluation/report.md` (measured results).

---

## 1. What the system does

Given **a text description** ("navy blue sports shoes") or **a reference image**
(an uploaded photo), return the most similar products from a 44,441-product
fashion catalog, with their images and metadata.

The core idea is a single shared vector space. OpenAI's CLIP `ViT-B/32` was
pretrained so that a photo and the text describing it land close together. That
means **one query vector can be compared against both product photos and product
titles with the same dot product**. There is no training step in this project:
the encoders are frozen, and all of the engineering is in building, validating,
and searching the vector table around them.

| Property | Value |
| --- | --- |
| Model | `openai/clip-vit-base-patch32`, frozen, `eval()` + `torch.no_grad()` |
| Vector dim | 512, L2-normalized `float32` |
| Similarity | cosine = plain dot product (vectors are unit-norm) |
| Index | Milvus collection (HNSW + COSINE), or the exact brute-force NumPy matmul |
| Backend | `FASHION_BACKEND`: `auto` (Milvus if running, else NumPy), `milvus`, `numpy` |
| Catalog | 44,441 products |
| Vector table | 88,875 rows = 44,441 image + 44,434 title |
| Device | CPU (CUDA used automatically when available) |

---

## 2. Component map

```
dataset/styles.csv ──┐
dataset/images/   ───┤
                     ▼
        fashion_data_cleaning.ipynb          (canonical data prep)
        src/prepare_data.py                  (condensed alternative)
                     ▼
        artifacts/data_preparation/catalog_clean.csv
                     │
                     ├──────────────────────────────┐
                     ▼                              │
        src/build_embeddings.py                     │
              uses src/encoder.py                   │
              publishes via src/store.py            │
                     ▼                              │
        artifacts/embeddings/                       │
          vectors.npy + rows.parquet + manifest.json│
                     │                              │
                     ▼                              ▼
              src/store.py  ◄────── loads ──── src/search.py ──► src/fusion.py
                     │                              ▲
                     │  src/milvus_import.py        │  one of the two backends,
                     └──────────► Milvus ───────────┘  chosen by FASHION_BACKEND
                                  src/milvus_store.py
                                                    │
                          ┌─────────────────────────┴───────────────┐
                          ▼                                         ▼
                      app.py (FastAPI)                     src/evaluate.py
                      static/ (vanilla JS UI)              src/baseline.py (TF-IDF)
                                                           evaluation/*.csv, report.md
```

### Module responsibilities

| Module | Owns |
| --- | --- |
| `src/config.py` | Project root discovery, paths, model identity, retrieval limits |
| `src/encoder.py` | Loading CLIP, device selection, `encode_texts` / `encode_images`, query validation |
| `src/store.py` | The unified vector table: validate, publish atomically, load, refuse mismatches |
| `src/milvus_store.py` | The same table in Milvus: collection schema, import, server-side search |
| `src/milvus_import.py` | CLI that loads a built vector table into a collection |
| `src/build_embeddings.py` | Batch-encode the catalog into one matrix + row table + manifest |
| `src/search.py` | Query → vector → score → filter → candidates → group → fuse → results |
| `src/fusion.py` | Combining `S_image` and `S_title` into one ranking score |
| `src/baseline.py` | TF-IDF keyword baseline over `search_text`, for comparison |
| `src/evaluate.py` | Pooling, blind judging sheets, Precision@5, image queries, timings |
| `src/verify_encoder.py` | Step-2 sanity check of the frozen encoder on the 500-product sample |
| `app.py` | FastAPI: lifespan loading, `/search/*`, `/images/{id}`, `/filters`, `/health` |
| `static/` | Vanilla-JS frontend (no build step) |

`src/` and `tests/` are packages. Everything runs as `python -m src.<module>` from
the project root, which `config.find_root()` locates by walking up until it finds
`dataset/styles.csv`.

---

## 3. Stage 1 — Data preparation

Two implementations exist and **the notebook is the source of truth**:

- `fashion_data_cleaning.ipynb` → `artifacts/data_preparation/`. Produces every
  audit CSV, the EDA figures, `audit_summary.json`, asserts record conservation
  across stages, and excludes ambiguous IDs (one product ID resolving to two
  different image hashes).
- `src/prepare_data.py` → `artifacts/catalog.csv`. A condensed re-implementation
  that omits the ambiguous-ID exclusion and all audit exports. It is not what
  downstream code reads.

Downstream code reads `artifacts/data_preparation/catalog_clean.csv` (44,441 rows)
or `catalog_sample_500.csv` (fixed `SEED = 42` development sample).

### What cleaning has to survive

`styles.csv` contains unquoted commas inside `productDisplayName`. Parsing uses
`csv.reader(..., strict=True)` on `utf-8-sig` and asserts the header against a
fixed `EXPECTED_SCHEMA`. Rows with more than 10 fields are repaired by keeping
the first 9 and rejoining the rest into the name (22 such rows); rows with fewer
than 10 fields are quarantined into an audit CSV, never silently dropped.

44,446 source records → 44,441 catalog products. The 5 dropped rows are metadata
with no readable image (`metadata_without_usable_image.csv`). No images lacked
metadata, and no image failed to decode.

### Catalog contract

The ten source columns plus derived ones that later stages depend on:

| Column | Role |
| --- | --- |
| `id` | The join key and the stable product identifier through every artifact |
| `image_path` | Posix, **relative to the project root** (`dataset/images/1673.jpg`); resolve as `ROOT / image_path` |
| `sha256`, `bytes`, `width`, `height`, `mode` | Image audit facts |
| `search_text` | `productDisplayName \| gender \| masterCategory \| subCategory \| articleType \| baseColour \| usage` — the keyword baseline's field |
| `image_duplicate_group` | Image SHA-256. **1,431 products share image bytes** with another product |
| `source_line`, `<column>_was_missing` | Provenance and fill flags |

Two consequences are load-bearing:

1. Images are always opened as `ImageOps.exif_transpose(Image.open(p)).convert('RGB')`
   — the exact path the audit validated readability on.
2. Because 1,431 products are byte-identical twins, **image-to-image evaluation
   must exclude the query item *and* its duplicate group**, or the top results
   are trivially the same photo again.

Images are tiny: 99.95% are 60×80 pixels, which CLIP upsamples to 224×224. Fine
patterns and brand text are physically unrecoverable. This is reported in the
evaluation rather than tuned away.

---

## 4. Stage 2 — The encoder (`src/encoder.py`)

`ClipEncoder` wraps `CLIPModel` + `CLIPProcessor`. It is loaded once per process
(`get_encoder` is `lru_cache`d), put in `eval()`, and every parameter gets
`requires_grad_(False)`.

Two loading flags are deliberate, not incidental:

- `use_safetensors=False` — the cached checkpoint is the `.bin`.
- `use_fast=False` — the fast image processor would require torchvision, which
  is intentionally not installed. Preprocessing stays on PIL/NumPy.

Both entry points return `(B, 512)` unit-norm `float32`, so cosine similarity is
a plain dot product everywhere downstream:

- `encode_texts(texts)` — tokenized to `max_length=77`, padded, truncated.
- `encode_images(images)` — the checkpoint's own transform; never hand-rolled.

### `EncoderIdentity` — the compatibility key

The checkpoint name alone does not pin preprocessing, so the encoder exposes a
frozen record of everything that must match between a built artifact and a
running process:

```
model_id, model_source, dim, image_size, image_mean, image_std, context_length
```

This is written into the manifest at build time and compared at load time.
A mismatch is a hard refusal, not a warning (§6).

### Query validation vs. catalog truncation

CLIP's text context window is 77 tokens (75 usable content tokens). The two
sides are handled differently on purpose:

- **Catalog titles** are truncated silently — they are data, and a long title
  losing its tail is acceptable.
- **User queries** go through `validate_text`, which normalizes whitespace,
  rejects blanks, rejects >500 characters, and rejects >75 content tokens with a
  message that says how long the query was and what the limit is. A silent
  truncation would mean the user searched for something other than what they
  typed. This raises `QueryValidationError`, which the API maps to **422**.

---

## 5. Stage 3 — Building the unified vector table (`src/build_embeddings.py`)

Every product contributes up to two rows to **one** matrix:

- an `image` row — the product photo through the vision encoder;
- a `title` row — `productDisplayName` through the text encoder.

Seven products have a blank title and therefore carry an image row only. That
absence is recorded in the manifest's `failures` list as `blank productDisplayName`
and becomes a `None` score downstream — **never a silent zero**.

A product whose image fails to load is excluded entirely: the build tracks the
set of successfully encoded IDs and refuses to emit a title-only row for a
product with no image row.

The build times the image pass and title pass separately and records both, which
is what `evaluation/timings.json` reports as indexing cost.

### The artifact set — `artifacts/embeddings/`

| File | Contents |
| --- | --- |
| `vectors.npy` | `(88875, 512)` `float32`, unit-norm |
| `rows.parquet` | 88,875 rows: `product_id`, `embedding_type`, `name`, `image_path` |
| `manifest.json` | Model identity, catalog fingerprint, counts, timings, environment, failures |

**Row order is the only vector→product linkage.** Row *i* of `vectors.npy` is
described by row *i* of `rows.parquet`. Nothing else connects them, so anything
writing one writes the other in the same pass.

`artifacts/embeddings_sample/` is the same structure built from the 500-product
development sample (~30 s, for fast iteration).

### The manifest

Recorded at build time:

```json
{
  "schema_version": 1,
  "encoder": { "model_id": "...", "dim": 512, "image_size": 224,
               "image_mean": [...], "image_std": [...], "context_length": 77 },
  "catalog_fingerprint": "2cf622d6…",
  "catalog_products": 44441,
  "row_counts": { "image": 44441, "title": 44434 },
  "total_rows": 88875,
  "device": "cpu", "batch_size": 64,
  "timings": { "total_index_seconds": 2486.12, "images_per_second": 30.06 },
  "environment": { "python": "3.13.5", "torch": "2.14.0+cpu", ... },
  "failures": [ … ], "failure_count": 7
}
```

`catalog_fingerprint` is a SHA-256 over the product-ID array *in catalog order*,
so re-cleaning the data into a different order or product set invalidates the
artifacts rather than silently misaligning them.

---

## 6. Stage 4 — The store (`src/store.py`)

This module is the trust boundary of the whole pipeline. It never trusts what it
just wrote or what it just read.

### Validation

`validate()` is run before publishing, after reloading from staging, and again at
load time. It checks that the matrix is 2-D `float32`; that row counts match;
that every value is finite; that every vector is unit-norm within `1e-4`; that
`embedding_type` contains only known values; that no `(product_id, embedding_type)`
pair repeats; and, when the expected product count is known, that the image-row
count equals it.

### Atomic publishing

`publish()` writes into a temp directory beside the target, with the **manifest
written last** so a half-built directory can never look complete. It then reloads
what actually landed on disk, re-validates it, byte-compares the reloaded matrix
against the built one, and only then `os.replace`s the directory into place.

### Refusing mismatched artifacts

`load()` compares the manifest against the running encoder's `EncoderIdentity`
(`model_id`, `dim`, `image_size`, `context_length`, `image_mean`, `image_std`)
and against the catalog fingerprint on disk. Any difference raises `ArtifactError`
listing each mismatch and telling the operator to rebuild. Searching a table
built with different preprocessing would return quietly wrong results, so this is
a refusal rather than a warning.

### The in-memory view

`UnifiedStore.__post_init__` precomputes what search needs on every query:

- `type_index[modality]` — the row positions belonging to each modality;
- `row_of[modality][product_id]` — the O(1) product→row lookup;
- `product_meta` — one metadata row per product, indexed by ID.

`score_all(q)` is then a single `vectors @ q` over all 88,875 rows.

### The Milvus backend (`src/milvus_store.py`)

The same table, in a Milvus collection instead of a NumPy array. Both backends
answer exactly one question, so nothing downstream knows which is in use:

```python
search_modalities(query_vector, candidate_k, filters, excluded)
    -> (candidate product ids in discovery order,
        {product_id: {"image": score|None, "title": score|None}})
```

`UnifiedStore` answers it with one matmul and two `argpartition`s. The Milvus
store answers it with one search per modality, then the union — the same shape,
so neither modality can crowd the other out of the pool. A candidate that only
one of the two searches found has the other modality's vector fetched (a few
hundred rows at most) and scored locally, so both scores are always real, and a
modality a product genuinely lacks is still `None`, never `0.0`.

| field | |
| --- | --- |
| `row_id` | `INT64` primary key — the row's position in `rows.parquet` |
| `vector` | `FLOAT_VECTOR(512)`, HNSW, COSINE |
| `product_id` | `INT64` |
| `embedding_type` | `VARCHAR` — `"image"` or `"title"` |
| `articleType`, `baseColour`, `gender`, `masterCategory`, `subCategory` | `VARCHAR` |

**One collection, not two.** Image rows and title rows sit together exactly as
they do in `vectors.npy`. Splitting them would push the modality gap of §8 onto
whatever queried them.

**The metadata columns are in the collection** so a filter becomes part of the
search expression rather than a pass over the results. That is the §7 rule —
filters before truncation — expressed in Milvus's own terms. Exclusions ride the
same expression as `product_id not in [...]`.

**Identity still comes from the manifest.** `open_store` runs the same
`manifest_mismatches` check as `load`, plus a row count against
`manifest["total_rows"]`, and raises `ArtifactError` on any difference. The
import reads the built artifacts and copies them; it never encodes, so the
artifacts directory remains the single source of truth.

**Only absence falls back.** `MilvusUnavailable` (no server, no collection, no
`pymilvus`) makes `auto` use NumPy and records why in `check_ready()["backend_note"]`.
A collection that exists but does not match is an `ArtifactError` under every
setting — wrong results are worse than no results.

**Measured, not assumed.** HNSW is approximate, so agreement was checked rather
than asserted: over the 32 evaluation queries on the full catalog, Milvus and the
NumPy backend returned identical top-10 orderings for 32/32 queries, with score
differences at float rounding (1e-6). Milvus costs ~11 ms per query (62.5 ms mean
against 51.9 ms) and buys nothing back at 44k products; it is the shape this
grows into, not a speedup today.

---

## 7. Stage 5 — Search (`src/search.py`)

```
query ──► CLIP vector ──► score the ENTIRE unified table (one matmul)
      ──► apply exclusions + metadata filters
      ──► top-K per modality ──► union by product_id
      ──► read BOTH modality scores for every candidate
      ──► fuse ──► top-k unique products
```

Text and image queries differ only in which encoder produces the query vector.
After that, the pipeline is identical — which is precisely what the shared CLIP
space buys.

The middle three steps are the backend's (`search_modalities`, §6). Everything 
else in this module — validation, exclusions, grouping, fusion, the response —
runs the same whether the vectors are in NumPy or in Milvus.

### Details that matter

**Filters are applied before truncation.** `_allowed_ids()` resolves the metadata
filters (`articleType`, `baseColour`, `gender`, `masterCategory`, `subCategory`)
into an allowed ID set against the *full* catalog, and candidate selection honors
it. Filtering the top-k after the fact would return 2 results where 10 exist.

**Candidates are collected per modality, then unioned.** Taking the global top-K
over the mixed table would let one modality crowd out the other. Each modality
contributes its own top `candidate_k` (default 100), via `argpartition` rather
than a full sort.

**Every candidate is then scored on both modalities.** Whichever modality pulled
a product into the pool, both `S_image` and `S_title` are read straight out of
the already-computed full score vector — no re-encoding, no second pass. A
product that genuinely has no title row gets `None`, not `0.0`.

**Image serving is not user-controlled path input.** `image_path_for()` looks the
path up in the store by product ID and then asserts the resolved path is inside
`dataset/images/`, so `/images/{id}` cannot be walked out of the image directory.

**Duplicate exclusion.** `exclude_product_id` removes the query item and, by
default, every product in its `image_duplicate_group` — the 1,431-product
byte-identical problem from §3.

---

## 8. Stage 6 — Fusion (`src/fusion.py`)

This is where the one genuinely subtle piece of modeling lives.

### The problem: the modality gap

CLIP places text near text more tightly than text near images. Measured on this
catalog (`python -m src.verify_encoder`, recorded in `evaluation/measurements.json`):

| Query→ | mean | std |
| --- | --- | --- |
| title vectors | 0.468 | 0.104 |
| image vectors | 0.171 | 0.035 |

The **mean offset is harmless** — a constant shift cannot reorder candidates.
What distorts a naive sum is the difference in **spread**: a ratio of ≈3.09. In a
raw 50/50 sum, the title modality therefore exerts roughly three times the
ranking influence it was nominally given.

### The strategies

All five sit behind one `fuse()` interface:

| Strategy | Behavior |
| --- | --- |
| `weighted_sum` | **Default.** Per-modality min-max normalization *over the candidate pool*, then a convex combination |
| `rrf` | Reciprocal Rank Fusion, `1/(k + rank)` with `k = 60`; scale-free by construction |
| `raw_weighted_sum` | The naive reference, kept so the report can demonstrate why the other two exist |
| `image_only` / `title_only` | Single-modality ablations |

A degenerate pool (every candidate scoring identically on a modality) normalizes
to a constant 1.0 — that modality carries no ranking information, so it must not
amplify floating-point noise into an ordering.

### Missing modalities

Under the weighted sums, a missing modality's weight is **redistributed** onto
the modalities the product does have (`_normalized_weights`), so a title-less
product is ranked on its image score rather than penalized for a gap in the
catalog. Under RRF it simply contributes no rank term.

Every result carries a `breakdown` dict with the raw per-modality scores, the
normalized values (or the ranks, under RRF), and the effective weights — so any
ranking in the UI can be explained rather than taken on faith.

---

## 9. Stage 7 — The API (`app.py`)

The encoder and the vector table load **once**, at startup, in a worker thread
(`asyncio.to_thread` inside the `lifespan` context manager). Requests never
reload them. If the artifacts are missing or mismatched the app still starts,
`/health` reports `ready: false` with a user-safe reason, and the search
endpoints return 503 — a bad artifact never becomes a stack trace in the browser.

`app.state.engine_factory` is the test seam: `tests/` swaps in a fake engine, so
the API tests need neither weights nor the dataset.

| Endpoint | Purpose |
| --- | --- |
| `POST /search/text` | `{query, top_k, strategy, image_weight, filters}` → ranked results |
| `POST /search/image` | multipart upload + `top_k`, `strategy`, `image_weight`, `exclude_product_id` |
| `GET /images/{product_id}` | Catalog JPEG, `Cache-Control: max-age=86400` |
| `GET /filters` | Distinct values for each filter column, to populate the UI |
| `GET /health` | Readiness, model identity, device, row counts, `built_at`, startup seconds |
| `/` | `static/` mounted with `html=True` |

Blocking work (`encode`, the matmul) is dispatched with `asyncio.to_thread` so
the event loop is never held by a synchronous CPU-bound call.

Error mapping: `QueryValidationError` → **422** with a human-readable message;
missing engine → **503**; unknown product → **404**. Uploads are capped at 10 MB
and decoded through `open_upload`, which turns any decode failure into a
user-safe 422 rather than a raw Pillow exception.

The frontend (`static/index.html` + `app.js` + `styles.css`) is plain ES modules
with no build step: a text/image tab pair, a strategy selector, a top-k input, an
image-weight slider, three metadata filters, and a result grid that surfaces the
score breakdown.

---

## 10. Stage 8 — Evaluation (`src/evaluate.py`, `src/baseline.py`)

Four subcommands, because **judging is a human step that sits in the middle**:

```
pool   --split development   # build the blind judging sheet
score  --split development   # read it back, report Precision@5
images --n 20                # image-query inspection sheet
timing                       # indexing / warm latency / cold start
```

### Pooling and blind judging

`pool` runs all five CLIP strategies plus the TF-IDF baseline at depth 5, unions
their results per query, **shuffles** the rows, and hides the producing system in
a `_found_by` column. `grade` is left blank. Re-running preserves existing grades
unless `--force` is passed.

The written relevance rule, applied uniformly:

- **2 = relevant** — the requested article type AND every stated attribute matches
- **1 = partial** — correct article type but an attribute is wrong/absent, or a near-miss type
- **0 = irrelevant** — wrong article type

Precision@5 counts **grade 2 only**. A blank cell means *unjudged*, never 0, and
`score` returns `None` for any query whose top 5 is not fully judged, excluding it
from every system's mean rather than scoring it optimistically. Recall@k is not
reported — the labels are not complete enough to support it.

### The baseline

`KeywordBaseline` is a from-scratch TF-IDF bag-of-words scorer over `search_text`
(sublinear TF, smoothed IDF, L2-normalized document vectors, inverted postings).
It is deliberately plain: the point is to show what lexical matching achieves on
the same catalog under the same filters, not to be a competitive search engine.

### Queries

`evaluation/queries.csv` holds 32 queries split into `development` and
`final_test`. **Tune on development only**; the final set stays untouched.

### Current state

| Artifact | Status |
| --- | --- |
| `evaluation/timings.json` | Measured |
| `evaluation/measurements.json` | Measured (modality gap, strategy overlap, metadata agreement) |
| `evaluation/image_queries.csv` | Pooled, `grade` blank |
| `evaluation/judgments_development.csv` | 204 pooled pairs, **0 judged** |
| `evaluation/judgments_final_test.csv` | 326 pooled pairs, **0 judged** |

**No Precision@5 exists yet.** Those grade cells are a human step and must never
be filled by inference. The metadata-agreement rates reported by
`evaluate images` (e.g. 76–83% article-type agreement) are *metadata agreement*,
not judged relevance — a catalog label is a filter and eval aid, not proof that a
requested visual attribute is present.

### Measured performance (CPU, 44,441 products, batch size 64)

| Metric | Value |
| --- | --- |
| Full index build | 2,486 s ≈ 41 min (1,478 s images + 1,008 s titles) |
| Image encoding throughput | 30.1 images/s |
| Cold startup | 8.84 s |
| Warm text query | mean 37.1 ms, median 36.2 ms, p95 41.7 ms |
| Warm image query | mean 56.2 ms, median 52.4 ms, p95 88.0 ms |

Brute-force search over 88,875 × 512 floats is ~45 MB of matmul per query — a few
milliseconds. Query latency is dominated by *encoding*, not retrieval, which is
why no ANN index is warranted at this catalog size.

---

## 11. Tests

85 tests, and none of them need a GPU, the CLIP weights, the dataset, or Milvus.

| File | Covers |
| --- | --- |
| `tests/test_store.py` | Validation rules, atomic publish, manifest mismatch refusal |
| `tests/test_search.py` | Candidate collection, filters-before-truncation, exclusions, path safety |
| `tests/test_fusion.py` | All five strategies, missing modalities, weight redistribution, degenerate pools |
| `tests/test_api.py` | Endpoints, 422/503/404 mapping, readiness, via `engine_factory` |
| `tests/test_evaluate.py` | Pooling, blank-is-unjudged, Precision@5 abstention |
| `tests/test_milvus.py` | Search expressions, the two-modality union, backend selection, via a fake client |
| `tests/helpers.py` | Synthetic catalogs, fake encoders, artifact fixtures |

Run with `python -m pytest -q`.

---

## 12. Deployment

`Dockerfile` is a four-stage CPU-only build (`base` → `deps` → `weights` →
`runtime`). torch comes from the CPU wheel index (the PyPI wheel drags in a
multi-GB CUDA runtime this image has no use for), torchvision is deliberately
absent, and the CLIP weights are prefetched into the image so the container runs
with `HF_HUB_OFFLINE=1` and no network. It runs as a non-root user (uid 10001)
with a healthcheck that allows a 180 s start period.

`docker-compose.yml` bind-mounts rather than bakes: the 642 MB dataset and 209 MB
artifacts stay outside the image, so a rebuilt embedding table appears without
rebuilding the image. The `app` service mounts `artifacts/` **read-only** so a
running server can never race a build. One-shot jobs (`build-embeddings`,
`evaluate`, `verify-encoder`, `tests`) sit behind the `tools` profile and take
their module's own flags.

---

## 13. Invariants — the things that silently break if violated

1. **Row order is the only vector→product link.** `vectors.npy` row *i* ↔
   `rows.parquet` row *i*. Write both in the same pass, always.
2. **`id` is the stable product identifier** across catalog, embeddings, artifacts,
   and evaluation sheets.
3. **Vectors are unit-norm `float32`**, validated at publish and load. Cosine is a
   dot product only because of this.
4. **Artifacts carry a manifest, and search refuses a mismatch.** Model identity,
   preprocessing identity, and catalog fingerprint must all agree.
5. **Publish only after validating the reload.** Manifest written last.
6. **A missing modality is `None`, never `0.0`**, at every layer.
7. **Never sum raw cross-modality scores** without per-modality normalization or
   a rank-based fusion — the ≈3× spread ratio is real and measured.
8. **Filters apply before truncation**, not to the top-k.
9. **Image-to-image queries exclude the query item and its duplicate group.**
10. **Over-long queries are rejected, not truncated.** Blank queries and invalid
    uploads get understandable messages.
11. **A blank grade cell is unjudged, never 0**, and a metric is never reported
    over a partially judged top-5.
12. **Never fabricate judgments or measured numbers**, and never present metadata
    agreement as relevance.
13. **Both backends answer `search_modalities` identically.** Retrieval changes go
    behind that method, never into `search.py`, or the two drift apart silently.
14. **A Milvus collection that does not match the manifest is refused**, not served
    and not quietly replaced by the NumPy path.

---

## 14. Known limitations

- **Source images are 60×80.** CLIP upsamples to 224×224; fine patterns, textures,
  and brand text cannot be recovered. Report this; do not tune around it.
- **Brute-force search is exact but linear** in catalog size. Fine at 44k
  products on CPU; an ANN index would be needed an order of magnitude higher.
- **Milvus is exercised, not needed yet.** It costs ~11 ms per query at this
  catalog size and returns the same ranking. Its value is the growth path, and
  the agreement figures above are what justify calling the two interchangeable.
- **Relevance is unmeasured.** The pooled judging sheets are complete and blind,
  but ungraded, so no Precision@5 comparison against the baseline exists yet.
- **Two data-prep paths can diverge.** `src/prepare_data.py` omits the
  ambiguous-ID exclusion and the audit exports that the notebook produces. If
  cleaning logic changes, change both or delete one.
- **Dormant dual-folder image resolution.** `IMAGE_DIRS` still lists
  `archive/images` alongside `dataset/images`; `archive/` does not exist in this
  checkout. The code is dormant, not dead — leave it until the plan's audit step
  is formally closed.
