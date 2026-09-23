# Plan: Multimodal Fashion Product Search with CLIP

## 1. Goal

Build an end-to-end multimodal search application over the Kaggle *Fashion Product Images (Small)* dataset using frozen OpenAI CLIP `ViT-B/32`.

Two fields of every product are embedded into the **same shared CLIP latent space**:

- the **product image** (`dataset/images/<id>.jpg`), via the CLIP vision encoder;
- the **product title** (`productDisplayName`), via the CLIP text encoder.

Both kinds of vector live in **one unified vector table**, each row tagged with an `embedding_type` of `"image"` or `"title"`. A user query — raw text or an uploaded image — is encoded into the same space, scored against the whole unified table, grouped back by `product_id`, fused across modalities, and returned as the **top 10 unique products** through a FastAPI backend and a web UI.

There is **no training step**. All ranking comes from cosine similarity between L2-normalized frozen embeddings.

### Pipeline

```text
[Query: text or uploaded image]
            │
            ▼
   [CLIP query embedding]  (512-d, L2-normalized)
            │
            ▼
 [Candidate retrieval from the unified vector table]
   ├── rows where embedding_type = "image"
   └── rows where embedding_type = "title"
            │
            ▼
   [Group by product_id → (S_image, S_title)]
            │
            ▼
   [Fusion & reranking: weighted sum / RRF]
            │
            ▼
   [Top 10 unique products] ──► [Web interface]
```

### Status — built, running, and measured

**Steps 1-6, 8 and 9 are complete. Step 7 is complete except for the human judging pass.**

Built and verified on 2026-09-22, CPU only (Windows 11, Python 3.13.5, torch 2.14.0+cpu):

- **Unified vector table**: 88,875 rows in one matrix - 44,441 `image` + 44,434 `title`, 512-d, all finite and unit-norm. The 7 missing title rows are the 7 products with a blank `productDisplayName`; they carry an image row and a `null` title score, never a fabricated title. Build took 41.4 min at 30.1 images/s.
- **Retrieval, fusion, API, UI**: all five strategies run; both endpoints verified over HTTP on the full catalog; every error path returns a user-safe 4xx/503 with no traceback.
- **Measured**: indexing 2,486 s, cold start 8.8 s, warm text query 37 ms (p95 42), warm image query 56 ms (p95 88). Modality-gap spread ratio **3.09**. See `evaluation/report.md`.
- **Tests**: 72 passing without GPU, model weights, or the dataset.

**Outstanding:** the relevance grades. `evaluation/judgments_development.csv` (204 pooled pairs) and `judgments_final_test.csv` (326 pairs) are built, blind and complete but unjudged, so no Precision@5 is reported. Also outstanding: UI screenshots (no browser was available in the build environment).

`src/prepare_data.py` still diverges from the canonical notebook (it omits the ambiguous-ID exclusion and the audit exports). Nothing downstream reads it - the embedding build reads `artifacts/data_preparation/catalog_clean.csv` - but it should be reconciled or deleted.

## 2. Deliverables

- `requirements.txt` with pinned, verified versions, and a README with reproducible setup and run commands.
- A validated catalog (done) plus a **unified embedding artifact set**: one vector matrix holding both image and title vectors, one aligned row table carrying `product_id`, `embedding_type`, `name`, `image_path`, and one manifest.
- A retrieval module implementing candidate retrieval, per-`product_id` grouping, and a configurable fusion/reranking strategy.
- A FastAPI backend exposing `POST /search/text`, `POST /search/image`, a static image route, and a health endpoint.
- A web interface with a text/image toggle, a responsive result grid of the top 10 cards, loading and empty states.
- An evaluation report with measured Precision@5 against a keyword baseline, a fusion-strategy comparison, and measured timings.
- A focused test suite under `tests/`.

## 3. Technical approach

**Model.** Frozen pretrained CLIP `ViT-B/32`, 512-d output for both encoders. Use the preprocessing transform and tokenizer shipped with the checkpoint; never hand-roll them. Run under `model.eval()` and `torch.no_grad()`. Support CUDA when available, CPU otherwise. Record which package supplies CLIP (`openai/CLIP` or `open_clip`/HF `transformers`) and pin it — the checkpoint name alone does not identify the preprocessing.

**Normalization.** Every stored and query vector is L2-normalized to unit length, so cosine similarity is a plain dot product and the whole unified table can be scored with a single matrix-vector product.

**Storage — the unified vector table.** All embeddings reside in **one** vector column. Canonical form on disk:

| file | contents |
| --- | --- |
| `artifacts/embeddings/vectors.npy` | `float32` array of shape `(N, 512)`, every row unit-norm |
| `artifacts/embeddings/rows.parquet` | row `i` describes `vectors.npy[i]`: `product_id` (int64), `embedding_type` (`"image"`/`"title"`), `name`, `image_path` |
| `artifacts/embeddings/manifest.json` | model identity, package versions, preprocessing identity, catalog fingerprint, per-type counts, `dim`, build timestamp, device, batch size |

For the full catalog `N = 2 × 44,441 = 88,882` rows, about 182 MB — exact NumPy search over that is a sub-100 ms matrix product on CPU, so **no ANN index is needed**. The row table is the "metadata per vector entry" the brief asks for; `embedding_type` is a real column that the retrieval code filters on, not an implicit split into two files. If the table is later moved into a vector database (Milvus, Qdrant, Chroma), the schema carries over one-to-one: a single collection, one vector field, `embedding_type` as a scalar field. Do not split into two collections.

**Row-order contract.** `vectors.npy` row order and `rows.parquet` row order are the same and are the only linkage between a vector and its product. Any code that writes one writes the other in the same pass.

## 4. Implementation sequence

### Step 1 — Data audit and catalog preparation ✅ complete

Done in `fashion_data_cleaning.ipynb`. Established facts that constrain later steps:

- 44,446 source CSV records → 44,441 catalog products. The 5 drops are metadata with no readable image (`metadata_without_usable_image.csv`). No image lacked metadata; no image failed to decode.
- `styles.csv` has unquoted commas inside `productDisplayName`; 22 over-long rows are repaired by keeping the first 9 fields and rejoining the rest into the name. Short rows are quarantined, never silently dropped.
- `id` is the join key between metadata and image filename stem and is the stable product identifier through every artifact.
- `image_path` is relative to the project root, posix-style; resolve as `ROOT / image_path`. Always load with `ImageOps.exif_transpose(Image.open(p)).convert("RGB")`.
- **1,431 products share image bytes with another product** (`image_duplicate_group` is the image SHA-256). Image-to-image evaluation must exclude the query item *and* its exact duplicates.
- 99.95% of images are 60×80. CLIP upsamples to 224×224, so fine patterns and brand text are unrecoverable — report this rather than tuning around it.

**Remaining work for this step:** confirm the title field is clean enough to embed. `productDisplayName` is the primary title; reject or flag blank titles, collapse whitespace, and record the count of `name_was_missing` products, which must not get a title vector under a fabricated name.

**Done when:** every retained product has an unambiguous ID, a readable image, and a non-empty title, and all exclusions are counted in the audit. (Met except the title check.)

### Step 2 — Verify CLIP on the 500-product sample

- Create the venv, install torch/CLIP/pandas/numpy/Pillow/FastAPI, and pin the versions that actually work on this machine.
- `src/encoder.py`: one module owning model load, device selection, `encode_images(list[PIL.Image]) -> (B, 512)` and `encode_texts(list[str]) -> (B, 512)`, both normalized. Model loads once and is cached.
- Verify on `catalog_sample_500.csv`: embeddings are 512-d, finite, unit-norm to within `1e-5`, and that image and title vectors of the *same* product are more similar to each other than to a random product on average.
- Sanity queries: "a navy blue shirt", "black casual shoes", "a silver watch" against the sample, inspecting both the image-side and the title-side ranking separately.
- Benchmark a batch to pick a batch size and estimate full-catalog build time on this hardware.

**Done when:** both encoders run on the available device and sample queries return sensible ranked results from each modality independently.

### Step 3 — Build the unified embedding artifacts

- `src/build_embeddings.py` encodes, for each catalog product, its image **and** its title, and appends both as separate rows of the unified table with the correct `embedding_type`.
- Stream images in batches; never load 44k images at once. Log and record any encode failure and drop that product's rows rather than leaving a half-populated product.
- Support `--sample` to build from `catalog_sample_500.csv` for development, writing to a separate artifact directory so the full build is never half-overwritten.
- Write to a temporary directory, then validate before publishing: the file reloads, `len(rows) == len(vectors)`, per-type counts match the catalog size, every vector is finite and unit-norm, every `product_id` appears exactly once per `embedding_type`, and every `image_path` resolves.
- Write the manifest last. Publish by atomic rename.
- Record indexing duration, device, catalog size, and batch size into the manifest.

**Done when:** artifacts reload, counts and IDs align, all vectors are valid, and the manifest describes the build.

### Step 4 — Retrieval, fusion, and reranking

`src/search.py` implements the core of the assignment.

**Query encoding.** Text queries: reject blank/whitespace-only input; CLIP's context window is **77 tokens**, so measure token length explicitly and reject or explicitly truncate with a warning rather than letting it silently cut off. Image queries: validate the upload is a decodable image, convert to RGB, apply the model's preprocessing.

**Candidate retrieval.** Score the query vector against the entire unified matrix in one dot product. Split the scores by `embedding_type` and take the top `K` per modality (`K ≈ 100`, configurable and ≥ `top_k`) as the candidate pool. The union of the two candidate sets, grouped by `product_id`, is the rerank input.

**Per-product modality scores.** For each candidate `product_id`:
- `S_image` = cosine(query, that product's image vector)
- `S_title` = cosine(query, that product's title vector)

Both are looked up directly from the full score vector, so a product that entered the pool through only one modality still gets its other score — the candidate pool decides *who* is reranked, not what information they are reranked with. A product genuinely missing a modality gets a recorded `None`, never a silent zero.

**The modality-gap problem (important).** In CLIP, a text query is systematically closer to *text* vectors than to *image* vectors — the two score distributions have different means and spreads. Adding raw `S_image` and `S_title` therefore lets whichever modality happens to sit on the higher scale dominate. Handle it explicitly, and say in the report which was chosen:
1. **Per-modality normalization then weighted sum** (default): z-score or min-max each modality's scores *over the candidate pool*, then `S_final = w_img · Ŝ_image + w_title · Ŝ_title` with `w_img + w_title = 1`.
2. **Reciprocal Rank Fusion**: `S_final = Σ_m 1 / (k + rank_m(product))` with `k = 60`. Scale-free by construction, so it sidesteps the gap entirely, at the cost of discarding score margins.
3. **Raw weighted sum**: kept as the naive reference to show *why* the other two exist.

Implement all three behind one interface selected by a `strategy` parameter. Default weights start at `w_img = w_title = 0.5`; asymmetric defaults per query modality (text queries leaning on titles, image queries leaning on images) are a hypothesis to **test on development queries**, not to assume.

**Final selection.** Sort by `S_final`, deduplicate by `product_id`, return the top 10 with the score breakdown attached. Support configurable `top_k` capped by the eligible catalog size. Apply metadata filters (article type, colour, gender) *before* truncation, and return a clean empty result when no candidate survives. For a catalog image used as a query, offer exclusion of the query product and its `image_duplicate_group` siblings.

**Artifact safety.** Refuse to serve artifacts whose manifest model identity, preprocessing identity, or catalog fingerprint does not match the running configuration.

**Done when:** both query modalities return 10 unique products with a full score breakdown, all three fusion strategies run, and every edge case produces an understandable message rather than a traceback.

### Step 5 — Backend API

`app.py` — FastAPI. The model and artifacts load **once** at startup (lifespan), not per request.

| Endpoint | Request | Notes |
| --- | --- | --- |
| `POST /search/text` | `{"query": "navy blue sports shoes", "top_k": 10}` | 422 on blank or over-length query |
| `POST /search/image` | `multipart/form-data` file upload | 422 on undecodable or oversized upload |
| `GET /images/{product_id}` | — | serves the thumbnail from `dataset/images/`, path-validated against the catalog, never from raw user input |
| `GET /health` | — | reports whether artifacts loaded, item counts, model identity |

Response item shape:

```json
{
  "rank": 1,
  "product_id": 1163,
  "name": "Nike Sahara Team India Fanwear Round Neck Jersey",
  "image_url": "/images/1163",
  "final_score": 0.8123,
  "breakdown": {"image": 0.7910, "title": 0.8336, "strategy": "weighted_sum"}
}
```

Return a top-level object carrying `results`, the `strategy` used, `top_k`, and query timing. Keep errors as structured JSON with a user-safe message — no tokens, no raw tracebacks. Cap upload size and request rate at the app layer.

**Done when:** both endpoints return 10 correctly ranked unique items with breakdowns, and invalid inputs produce clear 4xx responses.

### Step 6 — Web interface

A static vanilla-JS frontend served by FastAPI (`static/index.html`, `styles.css`, `app.js`). No build step.

- A tab/toggle switching **Text search** and **Image search**; text box with a search button and Enter-to-submit; a file picker with a local preview for the image tab.
- Optional filter controls for article type, colour, and gender, and a top-k control.
- Responsive card grid of the top 10: thumbnail, product name, **rank badge `#1`–`#10`**, final combined score, and the per-modality breakdown (collapsed or on hover).
- A fusion-strategy selector, so the comparison from step 7 is demonstrable live.
- State handling: loading spinner during the request, a clear empty state when nothing matches, and an error state for upload/validation failures.

**Done when:** a user can launch the app locally and complete both search workflows end to end without editing code.

### Step 7 — Evaluation

- Prepare **≥ 30 text queries** spanning article type, colour, and attribute combinations. Split into a development set and a **final set that stays untouched until the end**. Tune fusion weights, `K`, and strategy on development queries only.
- Inspect the eligible catalog before finalizing queries so "product does not exist" is distinguishable from "retrieval failed".
- **Baseline:** keyword search over `search_text` (`productDisplayName | gender | masterCategory | subCategory | articleType | baseColour | usage`) on the same catalog and the same filters.
- Judge the **pooled** top-5 of CLIP and the baseline under a written relevance rule, blind to which system produced each item. Report **mean Precision@5** with per-query results. **Do not report Recall@k** — the labels are not complete enough.
- **Fusion comparison:** report Precision@5 for image-only, title-only, raw weighted sum, normalized weighted sum, and RRF on the *same* judged pool. This is the core experimental result of the assignment.
- **≥ 20 image queries:** exclude the query product and its exact duplicates, then assess whether results match product type and visual attributes.
- Measure **indexing time**, **warm query latency**, and **cold startup** separately, recording device, catalog size, and batch size.
- Never fabricate judgments or measured numbers. Metadata labels are filter and evaluation aids, not proof that a requested visual attribute is present.

**Done when:** `evaluation/report.md` contains measured results, the baseline comparison, the fusion-strategy table, timings, and an honest account of weaknesses.

### Step 8 — Tests

`tests/` is a package; import as `from tests.helpers import ...` and run with `python -m pytest`. Use a tiny synthetic catalog and a fake encoder so the suite needs neither the CLIP weights nor the real dataset.

- Unified table: row-order alignment between `vectors.npy` and `rows.parquet`; every product has exactly one row per `embedding_type`.
- Fusion: known score inputs produce the expected ranking under each strategy; per-modality normalization is computed over the candidate pool; a missing modality is not silently treated as zero.
- Top-k: results are unique by `product_id`, length ≤ `top_k`, sorted descending by `final_score`.
- Filters applied before truncation; empty result when filters exclude everything.
- Self-exclusion removes the query product and its duplicate group.
- Validation: blank query, over-77-token query, undecodable upload, and manifest mismatch each raise the right typed error.
- API: both endpoints against a stubbed search engine, asserting the response shape and 4xx paths.

**Done when:** the suite passes without Docker, GPU, model weights, or the full dataset.

### Step 9 — Package and submit

- README: installation, model download, data preparation, embedding build, app launch, evaluation commands — written after those commands exist and verified by running them.
- Screenshots of both search modes and several reproducible example queries.
- Explain CLIP, cosine ranking, the unified-table design, the fusion strategy chosen and why, dataset exclusions, and observed failure cases.
- Record dataset provenance and usage terms before redistributing any images.
- Rebuild from the documented setup on a clean checkout and confirm the reported evaluation reproduces.

**Done when:** another person can rebuild the artifacts, launch the app, and reproduce the reported numbers.

## 5. File structure

```text
week3/
  CLIP_PROJECT_PLAN.md
  CLAUDE.md
  README.md
  requirements.txt
  fashion_data_cleaning.ipynb      # canonical data-prep path (done)
  dataset/
    styles.csv
    images/                        # 44,441 jpgs
  src/
    __init__.py
    prepare_data.py                # condensed re-implementation of the notebook
    encoder.py                     # CLIP load, encode_images, encode_texts
    build_embeddings.py            # writes the unified vector table
    store.py                       # load/validate the unified table + manifest
    search.py                      # retrieval, grouping, fusion, reranking
    evaluate.py                    # Precision@5, baseline, fusion comparison, timings
  app.py                           # FastAPI
  static/
    index.html
    styles.css
    app.js
  artifacts/
    data_preparation/              # notebook outputs (done)
    embeddings/
      vectors.npy
      rows.parquet
      manifest.json
    embeddings_sample/             # same shape, 500-product dev build
  evaluation/
    queries.csv                    # development + final split
    judgments.csv
    report.md
  tests/
    __init__.py
    helpers.py
    test_store.py
    test_fusion.py
    test_search.py
    test_api.py
```

Run everything from `week3/` as `python -m src.<module>`; `src/` and `tests/` are packages.

## 6. Schedule

| Session | Work | Checkpoint |
| --- | --- | --- |
| 1 | Environment, `encoder.py`, verify on the 500-product sample | Both encoders produce valid unit-norm vectors; sample queries look sane |
| 2 | `build_embeddings.py` + `store.py`, full unified table | Artifacts reload and validate; per-type counts match |
| 3 | `search.py`: retrieval, grouping, all three fusion strategies | Top 10 unique products with score breakdown from both query modalities |
| 4 | `app.py` + web UI | Complete local demo, both tabs, both states |
| 5 | Evaluation, tests, README, screenshots | Measured report and reproducible deliverables |

Not a hardware estimate — estimate the full embedding build from the session-1 batch benchmark.

## 7. Scope controls and known limitations

- **Frozen encoders only.** No fine-tuning; it would need a separate split, training procedure, and evaluation plan.
- **Titles are short and templated.** `productDisplayName` is often brand-heavy ("Nike Sahara Team India Fanwear Round Neck Jersey"), so the title modality rewards brand and category matches and says little about colour or visual style. Expect the fusion to be genuinely complementary and report where each modality wins.
- **Images are 60×80**, upsampled to 224×224. Fine patterns, textures, and brand text are unrecoverable. Document this measured limitation instead of tuning around it.
- **1,431 products share image bytes.** They are not errors; they must be excluded from image-query evaluation and deduplicated thoughtfully in results.
- **Metadata labels do not prove visual attributes.** They are filter and evaluation aids only.
- Deployment, user accounts, authentication, and personalized recommendations are out of scope.
- The two data-prep paths (`fashion_data_cleaning.ipynb` and `src/prepare_data.py`) currently diverge. Reconcile or delete one before building embeddings — silent divergence is the main hazard in this repo.

## 8. Completion checklist

- [x] Catalog audit and exclusions documented.
- [x] Title field validated; blank-title products handled explicitly (7 found, image-row only).
- [x] `requirements.txt` pinned and verified; CLIP source recorded (`transformers.CLIPModel`).
- [x] Encoder verified on the sample; batch size benchmarked (`python -m src.verify_encoder`).
- [x] Unified vector table built: 88,875 rows in one vector column, tagged by `embedding_type`.
- [x] Artifacts validated on reload (counts, ID alignment, unit norms) and described by a manifest.
- [x] Candidate retrieval + grouping by `product_id` implemented.
- [x] All five strategies implemented and selectable (3 fusion + 2 single-modality references).
- [x] Top 10 unique products returned with a per-modality breakdown.
- [x] `POST /search/text` and `POST /search/image` working with validation and clear errors.
- [x] Web UI: mode toggle, result grid, rank badges, scores, loading and empty states.
- [ ] **Precision@5 measured against the keyword baseline** - sheets pooled and blind, grades pending.
- [ ] **Fusion strategies compared on the same judged pool** - pending the same grades.
- [x] 20 image queries run with duplicate exclusion (metadata agreement measured; grades pending).
- [x] Indexing time, warm latency, and cold startup recorded with device and batch size.
- [x] Tests pass without model weights or the full dataset (72 tests).
- [x] README and evaluation report complete. [ ] Screenshots pending.
- [ ] Documented workflow reproduced on a clean checkout.
