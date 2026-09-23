"""Evaluation harness: pooling, scoring, image queries, and timings.

Four subcommands, because judging is a human step that sits in the middle:

    python -m src.evaluate pool   --split development   # build the judging sheet
    python -m src.evaluate score  --split development   # read it back, report P@5
    python -m src.evaluate images --n 20                # image-query inspection sheet
    python -m src.evaluate timing                       # indexing/latency/cold start

``pool`` writes ``evaluation/judgments_<split>.csv`` with one row per
(query, product) pair drawn from the **pooled** top-5 of every system, in a
shuffled order and with the producing system hidden, so grading is blind. The
``grade`` column is left blank. A blank cell means *unjudged*, never 0, and
``score`` refuses to report a metric for a query whose top 5 are not fully
judged. Nothing in this module ever invents a grade.

Relevance rule (write it down before judging, apply it uniformly):

    2 = relevant   - the item is the requested article type AND matches every
                     stated attribute (colour, gender, pattern, material).
    1 = partial    - the correct article type but a stated attribute is wrong
                     or absent, or a near-miss type (sandals for flip flops).
    0 = irrelevant - the wrong article type.

Precision@5 counts grade 2 only.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .baseline import KeywordBaseline
from .search import SearchEngine

SYSTEMS = {
    'clip_weighted_sum': {'strategy': 'weighted_sum'},
    'clip_rrf': {'strategy': 'rrf'},
    'clip_raw_weighted_sum': {'strategy': 'raw_weighted_sum'},
    'clip_image_only': {'strategy': 'image_only'},
    'clip_title_only': {'strategy': 'title_only'},
}
POOL_DEPTH = 5
SEED = 42


def load_queries(split: str | None = None) -> pd.DataFrame:
    queries = pd.read_csv(config.EVALUATION / 'queries.csv')
    if split:
        queries = queries[queries['split'] == split]
        if queries.empty:
            raise SystemExit(f'No queries with split={split!r}.')
    return queries.reset_index(drop=True)


def judgments_path(split: str) -> Path:
    return config.EVALUATION / f'judgments_{split}.csv'


# -- pool ----------------------------------------------------------------


def command_pool(args) -> int:
    engine = SearchEngine(artifacts_dir=args.artifacts, catalog_path=args.catalog)
    baseline = KeywordBaseline(engine.catalog)
    queries = load_queries(args.split)

    rows: list[dict] = []
    for query in queries.itertuples():
        pooled: dict[int, dict] = {}

        for system, kwargs in SYSTEMS.items():
            for item in engine.search_text(query.query, top_k=POOL_DEPTH, **kwargs)['results']:
                entry = pooled.setdefault(item['product_id'], {
                    'product_id': item['product_id'], 'name': item['name'], 'systems': []})
                entry['systems'].append(f"{system}@{item['rank']}")

        for item in baseline.search(query.query, top_k=POOL_DEPTH):
            entry = pooled.setdefault(item['product_id'], {
                'product_id': item['product_id'], 'name': item['name'], 'systems': []})
            entry['systems'].append(f"baseline@{len(entry['systems']) + 1}")

        candidates = list(pooled.values())
        # Shuffle so the sheet carries no ordering cue about which system found what.
        random.Random(SEED + hash(query.query_id) % 1000).shuffle(candidates)
        for candidate in candidates:
            metadata = engine._meta_lookup.get(candidate['product_id'], {})
            rows.append({
                'query_id': query.query_id,
                'query': query.query,
                'product_id': candidate['product_id'],
                'name': candidate['name'],
                'articleType': metadata.get('articleType'),
                'baseColour': metadata.get('baseColour'),
                'gender': metadata.get('gender'),
                'image_url': f"/images/{candidate['product_id']}",
                'grade': '',
                '_found_by': ';'.join(sorted(candidate['systems'])),
            })

    target = judgments_path(args.split)
    if target.exists() and not args.force:
        existing = pd.read_csv(target, dtype={'grade': 'string'})
        merged = pd.DataFrame(rows).merge(
            existing[['query_id', 'product_id', 'grade']].rename(columns={'grade': '_old'}),
            on=['query_id', 'product_id'], how='left')
        merged['grade'] = merged['_old'].fillna('')
        rows = merged.drop(columns=['_old']).to_dict('records')
        kept = sum(1 for r in rows if str(r['grade']).strip())
        print(f'Preserved {kept} existing grades. Pass --force to start over.')

    pd.DataFrame(rows).to_csv(target, index=False)
    unjudged = sum(1 for r in rows if not str(r['grade']).strip())
    print(f'Wrote {len(rows)} pooled pairs for {len(queries)} queries to {target}')
    print(f'{unjudged} still need a grade. Fill the "grade" column with 0, 1, or 2.')
    print('Leave a cell blank only if you have not judged it; blank is never treated as 0.')
    return 0


# -- score ---------------------------------------------------------------


def _precision_at_5(ranked: list[int], grades: dict[int, int]) -> float | None:
    """Fraction of the top 5 graded 2. None if any of the top 5 is unjudged."""
    top = ranked[:5]
    if not top:
        return 0.0
    if any(pid not in grades for pid in top):
        return None
    return sum(1 for pid in top if grades[pid] == 2) / len(top)


def command_score(args) -> int:
    target = judgments_path(args.split)
    if not target.exists():
        raise SystemExit(f'{target} does not exist. Run: python -m src.evaluate pool '
                         f'--split {args.split}')
    sheet = pd.read_csv(target, dtype={'grade': 'string'})
    sheet['grade'] = sheet['grade'].astype('string').str.strip()

    graded = sheet[sheet['grade'].notna() & (sheet['grade'] != '')]
    bad = set(graded['grade'].unique()) - {'0', '1', '2'}
    if bad:
        raise SystemExit(f'Grades must be 0, 1, or 2. Found: {sorted(bad)}')
    grades_by_query = {
        qid: {int(r.product_id): int(r.grade) for r in group.itertuples()}
        for qid, group in graded.groupby('query_id')
    }

    total = len(sheet)
    judged = len(graded)
    print(f'Judged {judged}/{total} pooled pairs ({judged / total:.0%}) in {target.name}')
    if judged == 0:
        print('\nNothing has been judged yet, so no precision can be reported.')
        print('Fill in the grade column and run this again.')
        return 1

    engine = SearchEngine(artifacts_dir=args.artifacts, catalog_path=args.catalog)
    baseline = KeywordBaseline(engine.catalog)
    queries = load_queries(args.split)

    records: list[dict] = []
    for query in queries.itertuples():
        grades = grades_by_query.get(query.query_id, {})
        systems: dict[str, list[int]] = {}
        for system, kwargs in SYSTEMS.items():
            systems[system] = [r['product_id'] for r in
                               engine.search_text(query.query, top_k=5, **kwargs)['results']]
        systems['baseline'] = [r['product_id'] for r in baseline.search(query.query, top_k=5)]
        for system, ranked in systems.items():
            records.append({'query_id': query.query_id, 'query': query.query,
                            'system': system, 'precision_at_5': _precision_at_5(ranked, grades)})

    per_query = pd.DataFrame(records)
    per_query.to_csv(config.EVALUATION / f'per_query_{args.split}.csv', index=False)

    summary = (per_query.groupby('system')['precision_at_5']
               .agg(mean_precision_at_5='mean', queries_scored='count')
               .sort_values('mean_precision_at_5', ascending=False))
    summary['queries_total'] = len(queries)
    summary['mean_precision_at_5'] = summary['mean_precision_at_5'].round(4)

    print(f'\nMean Precision@5 ({args.split}, grade 2 only, fully judged queries only)\n')
    print(summary.to_string())
    incomplete = per_query[per_query['precision_at_5'].isna()]['query_id'].nunique()
    if incomplete:
        print(f'\n{incomplete} queries are excluded from every system because at least '
              f'one of their top 5 is unjudged. Judge them for a complete comparison.')
    summary.to_csv(config.EVALUATION / f'summary_{args.split}.csv')
    return 0


# -- image queries -------------------------------------------------------


def command_images(args) -> int:
    """Run N catalog images as queries, excluding the item and its exact duplicates."""
    engine = SearchEngine(artifacts_dir=args.artifacts, catalog_path=args.catalog)
    from .encoder import load_image

    rng = np.random.default_rng(SEED)
    pool = engine.catalog.sample(n=args.n, random_state=SEED)

    rows = []
    for product in pool.itertuples():
        image = load_image(config.ROOT / product.image_path)
        payload = engine.search_image(image, top_k=5, exclude_product_id=int(product.id))
        for item in payload['results']:
            metadata = engine._meta_lookup.get(item['product_id'], {})
            rows.append({
                'query_product_id': int(product.id),
                'query_name': product.productDisplayName,
                'query_articleType': product.articleType,
                'query_baseColour': product.baseColour,
                'rank': item['rank'],
                'result_product_id': item['product_id'],
                'result_name': item['name'],
                'result_articleType': metadata.get('articleType'),
                'result_baseColour': metadata.get('baseColour'),
                'final_score': item['final_score'],
                'same_article_type': metadata.get('articleType') == product.articleType,
                'same_colour': metadata.get('baseColour') == product.baseColour,
                'grade': '',
            })

    frame = pd.DataFrame(rows)
    target = config.EVALUATION / 'image_queries.csv'
    frame.to_csv(target, index=False)

    # These two rates are metadata agreement, not relevance judgments.
    print(f'Wrote {len(frame)} rows for {args.n} image queries to {target}')
    print(f'Article-type agreement in top 5: {frame["same_article_type"].mean():.1%}')
    print(f'Colour agreement in top 5:       {frame["same_colour"].mean():.1%}')
    print('\nThese are metadata agreement rates, not judged relevance. Fill the '
          '"grade" column to judge them properly.')
    return 0


# -- timing --------------------------------------------------------------


def command_timing(args) -> int:
    import platform

    cold_started = time.perf_counter()
    engine = SearchEngine(artifacts_dir=args.artifacts, catalog_path=args.catalog)
    cold_seconds = time.perf_counter() - cold_started

    queries = load_queries()['query'].tolist()
    # Warm up so the first-call lazy paths are not counted as query latency.
    for query in queries[:3]:
        engine.search_text(query, top_k=10)

    text_ms, image_ms = [], []
    for query in queries:
        started = time.perf_counter()
        engine.search_text(query, top_k=10)
        text_ms.append((time.perf_counter() - started) * 1000)

    from .encoder import load_image
    sample = engine.catalog.sample(n=min(20, len(engine.catalog)), random_state=SEED)
    images = [load_image(config.ROOT / p) for p in sample['image_path']]
    for image in images:
        started = time.perf_counter()
        engine.search_image(image, top_k=10)
        image_ms.append((time.perf_counter() - started) * 1000)

    manifest = engine.store.manifest
    report = {
        'device': engine.encoder.device,
        'platform': platform.platform(),
        'python': platform.python_version(),
        'catalog_products': int(len(engine.store.product_ids)),
        'unified_table_rows': int(len(engine.store.rows)),
        'row_counts': engine.store.counts(),
        'embedding_dim': engine.store.dim,
        'indexing': {
            'batch_size': manifest.get('batch_size'),
            'total_index_seconds': manifest.get('timings', {}).get('total_index_seconds'),
            'image_encode_seconds': manifest.get('timings', {}).get('image_encode_seconds'),
            'title_encode_seconds': manifest.get('timings', {}).get('title_encode_seconds'),
            'images_per_second': manifest.get('timings', {}).get('images_per_second'),
        },
        'cold_startup_seconds': round(cold_seconds, 2),
        'warm_text_query_ms': {
            'n': len(text_ms),
            'mean': round(float(np.mean(text_ms)), 1),
            'median': round(float(np.median(text_ms)), 1),
            'p95': round(float(np.percentile(text_ms, 95)), 1),
        },
        'warm_image_query_ms': {
            'n': len(image_ms),
            'mean': round(float(np.mean(image_ms)), 1),
            'median': round(float(np.median(image_ms)), 1),
            'p95': round(float(np.percentile(image_ms, 95)), 1),
        },
    }
    target = config.EVALUATION / 'timings.json'
    target.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    print(f'\nWritten to {target}')
    return 0


# -- cli -----------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--artifacts', type=Path, default=config.EMBEDDINGS_FULL)
    parser.add_argument('--catalog', type=Path, default=config.CATALOG_FULL)
    sub = parser.add_subparsers(dest='command', required=True)

    pool = sub.add_parser('pool', help='build the blind judging sheet')
    pool.add_argument('--split', default='development')
    pool.add_argument('--force', action='store_true', help='discard existing grades')
    pool.set_defaults(func=command_pool)

    score = sub.add_parser('score', help='report Precision@5 from the judging sheet')
    score.add_argument('--split', default='development')
    score.set_defaults(func=command_score)

    images = sub.add_parser('images', help='run catalog images as queries')
    images.add_argument('--n', type=int, default=20)
    images.set_defaults(func=command_images)

    timing = sub.add_parser('timing', help='measure indexing, latency, and cold start')
    timing.set_defaults(func=command_timing)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
