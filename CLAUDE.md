# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`week3/` is a self-contained project: **fashion product search with CLIP**. Given a text description or a reference image, it should return the most similar products from the Kaggle fashion-product dataset (`dataset/styles.csv` + `dataset/images/`) with their images and metadata. The design uses frozen pretrained CLIP `ViT-B/32` — both encoders, L2-normalized vectors, cosine ranking, exact NumPy search — with no training step. `CLIP_PROJECT_PLAN.md` is the authoritative spec: read it before starting new work, and keep its completion checklist in sync.

The parent folder's `CLAUDE.md` describes a different, unrelated project (a Milvus lyrics search engine). Nothing in it applies here.

**Status: built and running.** The full pipeline exists - encoder, unified embedding build, fusion, search, FastAPI app, static UI, evaluation harness, 72 tests, `requirements.txt`, README. The unified vector table (`artifacts/embeddings/`) holds 88,875 rows: 44,441 `image` + 44,434 `title` (7 products have a blank title and carry an image row only). Built 2026-09-22 on CPU; see `evaluation/report.md` for measured timings.

The one thing outstanding is the **relevance judging pass**: `evaluation/judgments_{development,final_test}.csv` are pooled, blind and complete but every `grade` cell is blank, so no Precision@5 exists yet. Never fill those cells by inference - they are a human step.

Environment: Python 3.13.5, torch 2.14.0+cpu, transformers 4.57.6, pandas 2.3.3, Pillow 12.2.0, no GPU.

## Commands

Run everything from `week3/` (the project root, detected by the presence of `dataset/styles.csv`).

```powershell
jupyter lab fashion_data_cleaning.ipynb   # the canonical, fully audited data-prep path
python -m src.verify_encoder --limit 120 --batch-size 32
python -m src.build_embeddings --sample            # 500 products, ~30 s
python -m src.build_embeddings --batch-size 64     # full catalog, ~41 min on CPU
python -m uvicorn app:app --reload                 # http://127.0.0.1:8000
docker compose -f deploy/milvus/docker-compose.yml up -d   # Milvus on 127.0.0.1:19530
python -m src.milvus_import --sample --collection fashion_clip_sample
python -m src.milvus_import --recreate             # full table, 88,875 rows
python -m src.evaluate pool --split development     # then grade by hand
python -m src.evaluate score --split development
python -m src.evaluate images --n 20
python -m src.evaluate timing
python -m pytest -q                                # 85 tests, no GPU/weights/dataset/Milvus
```

The CLIP weights load from the Hugging Face cache; `src/encoder.py` passes
`use_safetensors=False` (the cached checkpoint is the `.bin`) and `use_fast=False`
(the fast image processor would require torchvision, which is not installed).
Set `HF_HUB_OFFLINE=1` to guarantee no network access.

A full image audit re-reads and SHA-256-hashes ~44k files, so it takes minutes; prefer reading the existing artifacts over re-running it.

## Two data-prep paths, one of which is canonical

`fashion_data_cleaning.ipynb` and `src/prepare_data.py` implement the *same* pipeline, but they are not equivalent and they write to different places:

- The **notebook is the source of truth**. It writes to `artifacts/data_preparation/`, produces every audit CSV, the EDA/dimension figures, and `audit_summary.json`, asserts record conservation across stages, and excludes ambiguous IDs (one product ID resolving to two different image hashes).
- `src/prepare_data.py` is a condensed re-implementation that writes `artifacts/catalog.csv` and `artifacts/catalog_sample_500.csv` (neither currently exists on disk). It **omits the ambiguous-ID exclusion and all audit exports**.

Downstream code should read `artifacts/data_preparation/catalog_clean.csv` (full, 44,441 rows) or `catalog_sample_500.csv` (the fixed `SEED = 42` development sample). If you change cleaning logic, change both paths or delete one — silent divergence between them is the main hazard here.

## Catalog contract

`catalog_clean.csv` carries the ten source columns plus: `source_line` (physical CSV line, for provenance), one `<column>_was_missing` boolean per filled categorical field plus `name_was_missing`, `image_path`, `sha256`, `bytes`, `width`, `height`, `mode`, `search_text`, `image_duplicate_group`, `aspect_ratio`, `file_size_kb`.

- `id` is the join key between metadata and image filename stem, and must stay the stable product identifier through embeddings and artifacts. Any saved embedding matrix must be paired with product IDs in exactly the same row order.
- `image_path` is **relative to the project root**, posix-style (e.g. `dataset/images/1673.jpg`). Resolve it as `ROOT / image_path`.
- Always load images as `ImageOps.exif_transpose(Image.open(p)).convert('RGB')` — the audit validated readability that way.
- `search_text` is `productDisplayName | gender | masterCategory | subCategory | articleType | baseColour | usage`, and is intended for the keyword baseline and for text-side experiments.
- `image_duplicate_group` is the image SHA-256. **1,431 products share image bytes with another product**, which is why image-to-image evaluation must exclude the query item *and* its exact duplicates.

## Data facts that constrain the work

- 44,446 source CSV records → 44,441 catalog products. The 5 dropped rows are metadata with no readable image (`metadata_without_usable_image.csv`); no images lacked metadata and no image failed to decode.
- `styles.csv` contains unquoted commas inside `productDisplayName`. Rows with more than 10 fields are repaired by keeping the first 9 fields and rejoining the rest into the name (22 such rows); rows with fewer than 10 fields are quarantined, never silently dropped. Parsing uses `csv.reader(..., strict=True)` on `utf-8-sig` and asserts the header matches `EXPECTED_SCHEMA`.
- IDs must match `[0-9]{1,18}` and be positive; `year` is validated into 1900–2100 and stored as nullable `Int64`. Missing categoricals are filled with `'Unknown'` and flagged, never dropped.
- `IMAGE_DIRS` still lists `archive/images` alongside `dataset/images`, with `dataset/` winning on conflict. **`archive/` does not exist in this checkout** — all 44,441 images came from `dataset/images`. The dual-folder resolution code is dormant, not dead; leave it unless the plan's audit step is formally closed out.
- Images are tiny: 99.95% are 60×80. CLIP upsamples them to 224×224, so expect fine patterns and brand text to be unrecoverable — say so in the evaluation rather than tuning it away.


## Two search backends, one contract

`store.UnifiedStore` (exact NumPy matmul) and `milvus_store.MilvusVectorStore` both
answer `search_modalities(query_vector, candidate_k, filters, excluded)` with
candidate product ids in discovery order plus **both** modality scores for each.
Everything downstream — fusion, the API, the evaluation — is backend-agnostic and
must stay that way; put retrieval changes behind that method, not in `search.py`.

- `FASHION_BACKEND` is `auto` (Milvus if running, else NumPy), `milvus`, or `numpy`.
  Only "Milvus is not there" falls back — `MilvusUnavailable`. A collection that
  exists but does not match the running encoder and catalog raises `ArtifactError`
  and is never served, because wrong results are worse than no results.
- One collection holds image and title rows together, told apart by
  `embedding_type`, with the five filter columns stored alongside so filters run
  inside the search. Never split the modalities into two collections.
- The import reads the built `vectors.npy` / `rows.parquet`; it never encodes.
  `artifacts/embeddings/manifest.json` remains the identity the collection is
  checked against, so rebuild first, then re-import with `--recreate`.
- Unit tests run against a fake client (`tests/test_milvus.py`) and need no server.
  Agreement with the NumPy backend was measured on the full catalog — identical
  top-10 ordering on all 32 evaluation queries — and is recorded in the README.
  Re-measure it rather than restating it if retrieval changes.

## Evaluation rules (from the plan)

Prepare ≥30 text queries spanning article type, colour, and attribute combinations, and split them into a development set and a final set that stays untouched until the end. Tune on development queries only. Compare CLIP against a keyword baseline over `search_text` on the same catalog and filters, judge the **pooled** top-5 of both systems under a written relevance rule, and report mean Precision@5. Do not report Recall@k — the labels are not complete enough to support it. Never fabricate judgments or measured numbers; metadata labels are filter/eval aids, not proof that a requested visual attribute is present. Measure indexing time, warm query latency, and cold startup separately, recording device, catalog size, and batch size.

## Conventions for the code still to be written

The plan's target layout is `src/{prepare_data,encoder,build_embeddings,store,search,evaluate}.py`, `app.py` (FastAPI) with a static vanilla-JS frontend under `static/`, `artifacts/embeddings/` for the unified vector table (`vectors.npy` + `rows.parquet` + `manifest.json`), `evaluation/` for queries, judgments, and the report, and `tests/`.

- Both modalities go in **one** vector table. Each product contributes an image-encoder row and a title-encoder row, distinguished by an `embedding_type` column (`"image"`/`"title"`), not by separate files or collections. `vectors.npy` row order and `rows.parquet` row order are the same and are the only vector→product linkage.
- Retrieval scores the query against the whole table, groups hits by `product_id`, and fuses `S_image`/`S_title` into one score. CLIP has a modality gap (a text query scores systematically higher against title vectors than image vectors), so normalize per modality over the candidate pool before any weighted sum, or use RRF. A missing modality is never a silent zero.

`src/` and `tests/` are packages; import as `from src.x import y` / `from tests.helpers import ...` and run modules as `python -m src.<module>`.

- Build artifacts into a temporary location and publish only after validating that they reload, that row counts match, that IDs align, and that all vectors are finite and unit-norm.
- Every artifact set needs a manifest recording model identity, package versions, preprocessing identity, catalog fingerprint, item count, and embedding dimension. Search must refuse artifacts whose manifest does not match the configured model and catalog.
- CLIP's text context window is 77 tokens. Handle over-long queries explicitly rather than letting them truncate silently, and reject blank queries and invalid uploads with understandable messages.
- Support CUDA when available and CPU otherwise; run inference under `eval()` and `torch.no_grad()`.
