# Verification Report — 2026-10-31

Audited the "built-and-working" claim of this repository. Read-only verification first (Phase 1), findings reported (Phase 2), then minimal fixes applied after confirmation (Phase 3).

## Summary
- **Tests:** 85 passed / 85 total (identical before and after)
- **Modules importing cleanly:** 13/13 `src.*` + `app` (was 12/13 before the `track_mlflow` move)
- **CLIs responding to `--help`:** 5/5 (`build_embeddings`, `milvus_import`, `evaluate`, `verify_encoder`, `track_mlflow`)
- **Known issues fixed:** 4 / 5
  - Fixed: #2 (test-count drift), #3 (`prepare_data` DEPRECATED banner), #4 (`prepare_data` archive comment), #5 (catch-all — the `track_mlflow` broken import/CLI, via confirmed Option A)
  - Not fixed: #1 (`track_mlflow` `manifest.get('dim')`) — **false positive**, see below
- **Known issues deferred:**
  - #1 `track_mlflow.py` `dim` — deferred because it is **not a bug** (the manifest has a top-level `dim` key; the line is correct as-is)
  - `src.evaluate timing` traceback on missing artifacts — deferred (not in the known-issues list; not confirmed for fixing)
- **Runtime checks skipped (missing artifacts/dataset):**
  - `python -m src.build_embeddings --sample` — skipped; `artifacts/data_preparation/catalog_sample_500.csv` does not exist

## Changes made
Each entry: `<file>:<line>` — description — which check it fixes.

1. `CLAUDE.md:11` — "72 tests" → "85 tests" — doc drift vs. code (d)
2. `evaluation/report.md:185` — "# 72 tests" → "# 85 tests" — doc drift vs. code (d)
3. `evaluation/report.md:242` — "**72 tests pass**" → "**85 tests pass**" — doc drift vs. code (d)
4. `CLIP_PROJECT_PLAN.md:310` — "(72 tests)" → "(85 tests)" — doc drift vs. code (d)
5. `src/prepare_data.py:1-7` — prepended `DEPRECATED` docstring banner (no behavior change) — known issue #3
6. `src/prepare_data.py:144` — added one-line `archive/` dormant comment above `image_dirs` (no behavior change) — known issue #4
7. `track_mlflow.py` → `src/track_mlflow.py` — moved into the `src` package; removed the stray trailing `w` (former line 92). `from . import config` kept as-is (valid once inside the package) — broken import (a) + broken command (b)

## Checks that passed
- `python -m py_compile src/*.py app.py track_mlflow.py` — all clean
- `import src.<module>` for all 12 original `src` modules + `import app` — clean
- `python -m pytest -q` — **85 passed**
- `--help` (exit 0, prints usage): `src.build_embeddings`, `src.milvus_import`, `src.evaluate`, `src.verify_encoder`
- `static/app.js` fetch endpoints exactly match `app.py` routes (no drift)
- `FASHION_BACKEND` consistent everywhere (14 occurrences, no `_URL`-style drift); every documented env var is real
- Every third-party import in `src/` + `app.py` is pinned in `requirements.txt` (pymilvus optional; its absence is handled by the NumPy fallback — verified live in `/health`)
- Manifest schema in `ARCHITECTURE.md` §5 matches what `build_embeddings.py` writes and `store.py`/`milvus_store.py` read
- Milvus collection fields `product_id`/`embedding_type` match `rows.parquet`; scalar filter fields come from the catalog (by design)
- Every `config.py` constant is used (`TEXT_CONTEXT_LENGTH` used internally to derive `MAX_TEXT_TOKENS`)
- `python -m uvicorn app:app` starts; `/health` returns `ready:false` with a user-safe message when vectors are missing (no crash)
- `ruff` — not installed, skipped (not installed, per instructions)

## Checks that failed and could not be fixed minimally
- None. Every failing check was either fixed (above) or deliberately deferred (below). The one structural fix (`track_mlflow` move) was explicitly pre-approved as Option A before it was made.

## Observations, no action taken
- **`track_mlflow.py` was an undocumented orphan.** Not referenced in any doc; `mlflow` is not in `requirements.txt`. It is now correctly placed in `src/` and its `--help` works, but it is still undocumented and `mlflow` is still unpinned (it is imported lazily inside `main()`, so `--help` needs no mlflow).
- **`src.evaluate timing` dumps a raw `FileNotFoundError` traceback** when `vectors.npy` is absent (the fresh-clone state: manifest present, vectors gitignored). The app handles the same condition gracefully (503 + message); the CLI does not. A minimal fix (try/except around the `SearchEngine(...)` in `command_timing`) was **not** applied — it is not in the known-issues list and was not confirmed.
- **`pydantic`** is imported in `app.py` but not explicitly pinned in `requirements.txt` (it is a hard transitive dependency of `fastapi`, so it is always present — not broken).
- **`FASHION_DEVICE`** and **`MILVUS_TOKEN`** are read in code but not documented (minor; not drift).
- **Working-tree changes NOT made by this audit** (flagged for awareness, left untouched):
  - `docker-compose.yml` — its 18-line header comment block is removed in the working tree. This audit did not edit that file; the change pre-dates it.
  - `artifacts/embeddings_sample/manifest.json` — reflects the 500-product sample build run in a prior session (records macOS / Python 3.14 / torch 2.15 instead of the committed Windows / 3.13 / 2.14). Not an audit change.

## Recommended follow-ups (not done today)
- Grade `evaluation/judgments_{development,final_test}.csv` to produce the headline Precision@5 (the one genuinely outstanding deliverable).
- Add UI screenshots (none captured).
- Reproduce the documented workflow on a clean checkout.
- Document `src/track_mlflow.py` in README/ARCHITECTURE and add `mlflow` to `requirements.txt` as an optional dep (or confirm it is intentionally out of scope).
- Decide whether `src.evaluate timing` (and the other `evaluate` subcommands) should convert missing-artifact errors into a user-safe message like the app does.
- Reconcile or delete `src/prepare_data.py` (now marked `DEPRECATED`) to remove the silent-divergence hazard.

---

## Final answers

**1. Every file changed (with line-count delta):**

| File | Delta | Note |
| --- | --- | --- |
| `CLAUDE.md` | 1 line modified (net 0) | 72 → 85 |
| `CLIP_PROJECT_PLAN.md` | 1 line modified (net 0) | 72 → 85 |
| `evaluation/report.md` | 2 lines modified (net 0) | 72 → 85 (×2) |
| `src/prepare_data.py` | +4 lines (5 ins, 1 del) | DEPRECATED banner + archive comment |
| `src/track_mlflow.py` | new file, 91 lines | moved from root `track_mlflow.py` (92 lines); −1 for the stray `w` |

Not changed by this audit but present in the working tree (flagged, untouched): `docker-compose.yml` (−18 header-comment lines, pre-existing), `artifacts/embeddings_sample/manifest.json` (prior-session sample build).

**2. Test count before and after:** **85 → 85** (identical). No test files were touched.

**3. Any check that failed and required more than a minimal fix:** The `track_mlflow` fix was a **file move** (structural, beyond a pure line edit) plus a one-line deletion. It exceeded a "line edit" but was **explicitly pre-approved as Option A** before being made, so it did not require stopping to ask. No other fix exceeded the 5-line / 2-file budget.
