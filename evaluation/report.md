# Evaluation Report — Multimodal Fashion Search with CLIP

**System.** Frozen CLIP `ViT-B/32` (`openai/clip-vit-base-patch32`, `transformers.CLIPModel`),
512-d, L2-normalized, exact NumPy search over a single unified vector table.
**Catalog.** 44,441 products. **Table.** 88,875 rows — 44,441 `image` + 44,434 `title`.
**Hardware.** CPU only (Windows 11, Python 3.13.5, torch 2.14.0+cpu). No GPU was available.
**Build.** 2026-09-22, batch size 64.

> **Status of this report.** Every number below marked *measured* was produced by the
> commands named beside it and is reproducible. The relevance-judgment sections are
> **not yet filled in**: the pooled judging sheets exist and are complete, but the grades
> are a human step that has not been performed. Those sections state what is pending
> rather than estimating it. No grade, precision figure, or relevance claim in this
> repository is invented.

---

## 1. Indexing and performance — measured

`python -m src.evaluate timing` → `evaluation/timings.json`

| Measure | Value |
| --- | --- |
| Full index build (images + titles) | **2,486 s** (41.4 min) |
| — image encoding | 1,478 s, **30.1 images/s** |
| — title encoding | 1,008 s |
| Cold startup (load weights + 182 MB table + catalog) | **8.8 s** |
| Warm text query, mean / median / p95 | **37.1 / 36.2 / 41.7 ms** (n=32) |
| Warm image query, mean / median / p95 | **56.2 / 52.4 / 88.0 ms** (n=20) |

Device: CPU. Catalog size: 44,441 products / 88,875 vectors. Batch size: 64. Embedding
dimension: 512.

Query latency is dominated by the CLIP forward pass, not by search. Scoring the entire
88,875-row table is one 88,875 × 512 matrix-vector product — a few milliseconds — which
is why no ANN index was built. Image queries cost ~19 ms more than text queries because
the vision encoder is the heavier of the two. Indexing is a one-off cost; on a GPU it
would drop by roughly an order of magnitude.

## 2. The modality gap — measured

`scripts in §6`, and `python -m src.verify_encoder` → `evaluation/measurements.json`

Mean cosine of a text query against each modality, over all 32 evaluation queries and the
full table:

| | mean | std | mean of top 100 |
| --- | --- | --- | --- |
| against **title** vectors | 0.468 | 0.104 | 0.816 |
| against **image** vectors | 0.171 | 0.035 | 0.302 |

**Spread ratio (title std ÷ image std): 3.09.**

This is the central design constraint of the project. The difference in *means* is
harmless on its own — a constant offset shifts every candidate equally and cannot reorder
them. What breaks a naive sum is the difference in **spread**: the title modality varies
about three times as widely, so in a raw 50/50 sum it silently exerts roughly three times
the ranking influence it was nominally given.

That is not a theoretical worry. Measured top-10 overlap between strategies, averaged
over the 32 queries:

| | weighted_sum | rrf | raw_sum | image_only | title_only |
| --- | --- | --- | --- | --- | --- |
| **weighted_sum** | 1.00 | 0.77 | 0.68 | 0.27 | 0.40 |
| **rrf** | 0.77 | 1.00 | 0.63 | 0.38 | 0.43 |
| **raw_weighted_sum** | 0.68 | 0.63 | 1.00 | **0.13** | **0.67** |
| **image_only** | 0.27 | 0.38 | 0.13 | 1.00 | 0.06 |
| **title_only** | 0.40 | 0.43 | 0.67 | 0.06 | 1.00 |

Read the `raw_weighted_sum` row: despite equal nominal weights, it shares **67%** of its
top 10 with pure title search and only **13%** with pure image search. The naive fusion
is, in practice, title search wearing a fusion label. The normalized weighted sum sits
between the two single-modality systems (0.27 image / 0.40 title) — genuinely blending
rather than being captured by one side.

The choice of strategy is also consequential rather than cosmetic: `weighted_sum` and
`raw_weighted_sum` share only 4–9 of 10 results per query (median 7).

## 3. Metadata agreement — measured, and *not* a relevance metric

Fraction of the top 10 whose catalog `articleType` / `baseColour` equals the query's
stated target, averaged over the 32 queries:

| system | article type | colour |
| --- | --- | --- |
| `clip_rrf` | **0.828** | 0.869 |
| `clip_weighted_sum` | 0.816 | 0.904 |
| `clip_raw_weighted_sum` | 0.816 | **0.942** |
| `clip_title_only` | 0.772 | 0.965 |
| `clip_image_only` | 0.766 | 0.746 |

**This table is not Precision@5 and must not be read as it.** It measures agreement
between a retrieved item's metadata label and the label I assigned to the query when
writing it — a weak proxy that is circular for the title modality (the colour word is
usually *in* the title the text encoder read) and that says nothing about whether the
photo actually shows the requested attribute. It is reported because it is measurable
today and it usefully illustrates the division of labour: title-side signal is strong on
colour words, image-side signal is weaker on both proxies but independent of them.

Note the ordering inversion: `title_only` has the best colour agreement (0.965) and the
worst article-type agreement of the CLIP systems apart from image-only. Brand-heavy
titles name colours reliably and garment types less so.

## 4. Image-to-image queries — measured (metadata agreement only)

`python -m src.evaluate images --n 20` → `evaluation/image_queries.csv`

20 catalog images used as queries, with the query product **and its exact-duplicate
group** excluded from results (1,431 catalog products share image bytes with another
product, so this exclusion is mandatory or the task is trivially self-answering).

| Measure over the pooled top 5 | Value |
| --- | --- |
| Result shares the query's `articleType` | **68.0%** |
| Result shares the query's `baseColour` | **51.0%** |

Again: agreement rates, not judged relevance. The `grade` column in
`evaluation/image_queries.csv` is blank and awaits judging.

Qualitative check (reproducible): querying with product 1163, a Nike Team India cricket
jersey, and excluding it, returns five other Nike Team India / Indian cricket jerseys at
scores 0.877–0.959. Image-to-image retrieval on this catalog is visibly strong for
distinctive garments.

## 5. Relevance evaluation — **pending judgment**

### Setup (complete)

- `evaluation/queries.csv` — 32 queries: **12 `development`**, **20 `final_test`**,
  spanning article type, colour, gender, material, pattern, and purpose. The final split
  has not been used for any tuning; all inspection above was done on all 32 without
  reference to grades, and no parameter was chosen from the final split.
- `evaluation/judgments_development.csv` — **204 pooled pairs**, all grades blank.
- `evaluation/judgments_final_test.csv` — **326 pooled pairs**, all grades blank.

Each sheet pools the top 5 of **six systems** — the five CLIP strategies and the TF-IDF
keyword baseline over `search_text` — deduplicates by product, and shuffles the rows with
the producing system hidden in a `_found_by` column, so grading is blind to provenance.

### Relevance rule (written before judging, to be applied uniformly)

- **2 — relevant**: the item is the requested article type **and** matches every stated
  attribute (colour, gender, pattern, material).
- **1 — partial**: correct article type but a stated attribute is wrong or absent, or a
  near-miss type (sandals returned for flip flops).
- **0 — irrelevant**: wrong article type.

**Precision@5 counts grade 2 only.** A blank cell means *unjudged*, never 0:
`src.evaluate score` returns `None` for any query whose top 5 is not fully judged and
drops it from every system's mean, so no system is flattered by a partially graded sheet.

### Results

| system | mean Precision@5 (development) | mean Precision@5 (final_test) |
| --- | --- | --- |
| `clip_weighted_sum` | *pending* | *pending* |
| `clip_rrf` | *pending* | *pending* |
| `clip_raw_weighted_sum` | *pending* | *pending* |
| `clip_image_only` | *pending* | *pending* |
| `clip_title_only` | *pending* | *pending* |
| `baseline` (TF-IDF keyword) | *pending* | *pending* |

To fill this in: open each `judgments_<split>.csv`, grade every row, then run

```powershell
python -m src.evaluate score --split development
python -m src.evaluate score --split final_test
```

which writes `summary_<split>.csv` and `per_query_<split>.csv`.

**Recall@k is not reported and should not be.** Relevance labels cover only the pooled
top 5 of six systems over a 44,441-product catalog; the denominator recall needs is
unknown, and estimating it would be fabrication.

## 6. Reproducing every measured number

```powershell
python -m src.verify_encoder --limit 120 --batch-size 32   # §2 sanity + gap on a sample
python -m src.build_embeddings --batch-size 64             # §1 indexing time
python -m src.evaluate timing                              # §1 latency and cold start
python -m src.evaluate images --n 20                       # §4
python -m src.evaluate pool --split development            # §5 judging sheet
python -m pytest -q                                        # 85 tests
```

The §2 and §3 tables come from `evaluation/measurements.json`.

## 7. Honest weaknesses

1. **Relevance is unjudged.** Everything in §5 is infrastructure, not results. The
   headline claim the assignment asks for — does fusion beat a keyword baseline — cannot
   be answered until the sheets are graded, and this report does not pretend otherwise.

2. **Images are 60×80**, upsampled to 224×224 for CLIP. Fine patterns, fabric texture,
   and brand text are simply not recoverable at this resolution. Queries such as
   `D10 striped casual t-shirt`, `F04 blue checked shirt`, and `F20 floral print dress`
   test exactly this and are expected to lean almost entirely on the title modality, where
   the pattern word appears as text. This is a property of the dataset, not something to
   tune away.

3. **Titles are brand-heavy** ("Nike Sahara Team India Fanwear Round Neck Jersey"), which
   makes the title modality strong on brand and colour and weaker on garment type — the
   inversion visible in §3.

4. **Observed failure mode: attribute bleed.** `D01 navy blue sports shoes` returns ten
   items all correctly Navy Blue, but every one is `articleType = Casual Shoes`, not
   `Sports Shoes`. The colour and the word "shoes" dominate; the usage qualifier is lost.
   This is precisely the kind of case that grade 1 exists for, and it will pull measured
   Precision@5 down when judged. It is reported here rather than hidden.

5. **The metadata-agreement proxy in §3 is circular for the title side** and cannot
   substitute for judgment, which is why it is labelled and not headlined.

6. **CPU only.** The 41-minute index build and 37 ms query latency would both improve
   substantially on a GPU. No GPU measurement is reported because none was taken.

7. **No screenshots yet.** The app runs and was verified end to end over HTTP (§ below),
   but no browser was available in the build environment to capture the UI.

## 8. System verification performed

Verified against the running app (`python -m uvicorn app:app`) on the full catalog:

| Check | Result |
| --- | --- |
| `GET /health` | `ready: true`, 44,441 image + 44,434 title rows, model identity reported |
| `POST /search/text` | 10 unique ranked products with per-modality breakdown, 58 ms |
| `POST /search/image` | 10 unique products, self and duplicates excluded, 82 ms |
| `GET /images/{id}` | 200 `image/jpeg`; 404 for an unknown product |
| Blank query | 422 "Enter a search phrase." |
| 100-token query | 422 "Query is 100 tokens; CLIP accepts at most 75." |
| Unknown strategy | 422, lists the valid strategies |
| `top_k = 0` | 422 |
| Non-image upload | 422 "That file could not be read as an image." |
| Impossible filter | 200 with `count: 0` and a clean empty state, not an error |
| Missing artifacts | app still starts; `/health` 503 with a safe reason; `/search/*` 503 |

No error response contained a traceback, a file path, or any internal detail.

Automated suite: **85 tests pass** with no GPU, no model weights, and no dataset
(`python -m pytest -q`), using a fake encoder and a synthetic 6-product catalog.
