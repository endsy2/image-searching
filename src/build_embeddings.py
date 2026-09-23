"""Build the unified vector table.

Each catalog product contributes up to two rows to ONE vector matrix:

  * an ``image`` row   - the product photo through the CLIP vision encoder
  * a ``title`` row    - ``productDisplayName`` through the CLIP text encoder

A product with no usable title gets an image row only; its title score is then a
recorded ``None`` downstream, never a silent zero.

    python -m src.build_embeddings --sample
    python -m src.build_embeddings --batch-size 64
"""
from __future__ import annotations

import argparse
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, store
from .encoder import ClipEncoder, load_image

DEFAULT_BATCH_SIZE = 64


def load_catalog(path: Path) -> pd.DataFrame:
    catalog = pd.read_csv(path, dtype={'id': 'int64'}, keep_default_na=True)
    catalog['id'] = catalog['id'].astype('int64')
    if catalog['id'].duplicated().any():
        raise ValueError('Catalog contains duplicate product ids.')
    catalog['title'] = (
        catalog['productDisplayName'].fillna('').astype(str)
        .str.replace(r'\s+', ' ', regex=True).str.strip()
    )
    return catalog.reset_index(drop=True)


def _batches(count: int, size: int):
    for start in range(0, count, size):
        yield start, min(start + size, count)


def build(catalog: pd.DataFrame, encoder: ClipEncoder, batch_size: int,
          progress_every: int = 20) -> tuple[np.ndarray, pd.DataFrame, dict]:
    """Encode every image and title into one matrix plus its aligned row table."""
    dim = encoder.identity.dim
    records: list[dict] = []
    blocks: list[np.ndarray] = []
    failures: list[dict] = []

    # --- image rows -----------------------------------------------------
    paths = [config.ROOT / p for p in catalog['image_path']]
    ids = catalog['id'].tolist()
    titles = catalog['title'].tolist()

    started = time.perf_counter()
    encoded_ids: list[int] = []
    for batch_no, (lo, hi) in enumerate(_batches(len(catalog), batch_size)):
        images, kept = [], []
        for offset in range(lo, hi):
            try:
                images.append(load_image(paths[offset]))
            except Exception as error:  # unreadable or corrupt on this run
                failures.append({'id': ids[offset], 'stage': 'image',
                                 'error': f'{type(error).__name__}: {error}'})
                continue
            kept.append(offset)
        if not kept:
            continue
        blocks.append(encoder.encode_images(images))
        for offset in kept:
            encoded_ids.append(ids[offset])
            records.append({
                'product_id': ids[offset],
                'embedding_type': 'image',
                'name': titles[offset],
                'image_path': catalog['image_path'].iloc[offset],
            })
        if batch_no % progress_every == 0:
            done = len(encoded_ids)
            rate = done / max(time.perf_counter() - started, 1e-9)
            print(f'  images {done}/{len(catalog)}  ({rate:.1f}/s)', flush=True)
    image_seconds = time.perf_counter() - started

    # A product that failed to encode must not keep a title-only row.
    usable = set(encoded_ids)

    # --- title rows -----------------------------------------------------
    title_rows = [
        (pid, title, path)
        for pid, title, path in zip(ids, titles, catalog['image_path'])
        if pid in usable and title
    ]
    skipped_titles = sorted(
        pid for pid, title in zip(ids, titles) if pid in usable and not title
    )
    for pid in skipped_titles:
        failures.append({'id': pid, 'stage': 'title', 'error': 'blank productDisplayName'})

    started = time.perf_counter()
    for batch_no, (lo, hi) in enumerate(_batches(len(title_rows), batch_size)):
        chunk = title_rows[lo:hi]
        blocks.append(encoder.encode_texts([t for _, t, _ in chunk]))
        for pid, title, path in chunk:
            records.append({
                'product_id': pid,
                'embedding_type': 'title',
                'name': title,
                'image_path': path,
            })
        if batch_no % progress_every == 0:
            print(f'  titles {min(hi, len(title_rows))}/{len(title_rows)}', flush=True)
    title_seconds = time.perf_counter() - started

    vectors = (np.vstack(blocks).astype(np.float32) if blocks
               else np.zeros((0, dim), dtype=np.float32))
    rows = pd.DataFrame.from_records(records, columns=store.ROW_COLUMNS)
    rows['product_id'] = rows['product_id'].astype('int64')

    timings = {
        'image_encode_seconds': round(image_seconds, 2),
        'title_encode_seconds': round(title_seconds, 2),
        'total_index_seconds': round(image_seconds + title_seconds, 2),
        'images_per_second': round(len(encoded_ids) / max(image_seconds, 1e-9), 2),
    }
    return vectors, rows, {'timings': timings, 'failures': failures,
                           'products_encoded': len(usable),
                           'titles_skipped_blank': len(skipped_titles)}


def build_manifest(catalog: pd.DataFrame, encoder: ClipEncoder, rows: pd.DataFrame,
                   batch_size: int, report: dict, catalog_path: Path) -> dict:
    import torch
    import transformers

    counts = rows['embedding_type'].value_counts().to_dict()
    return {
        'schema_version': 1,
        'built_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'encoder': encoder.identity.as_dict(),
        'catalog_path': str(catalog_path.relative_to(config.ROOT).as_posix()),
        'catalog_fingerprint': store.catalog_fingerprint(catalog),
        'catalog_products': int(len(catalog)),
        'products_embedded': int(report['products_encoded']),
        'titles_skipped_blank': int(report['titles_skipped_blank']),
        'row_counts': {k: int(v) for k, v in counts.items()},
        'total_rows': int(len(rows)),
        'dim': int(encoder.identity.dim),
        'device': encoder.device,
        'batch_size': int(batch_size),
        'timings': report['timings'],
        'environment': {
            'python': platform.python_version(),
            'platform': platform.platform(),
            'torch': torch.__version__,
            'transformers': transformers.__version__,
            'numpy': np.__version__,
            'pandas': pd.__version__,
        },
        'failures': report['failures'][:200],
        'failure_count': len(report['failures']),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sample', action='store_true',
                        help='build from catalog_sample_500.csv into artifacts/embeddings_sample')
    parser.add_argument('--catalog', type=Path, default=None)
    parser.add_argument('--out', type=Path, default=None)
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument('--device', default=None)
    parser.add_argument('--limit', type=int, default=None,
                        help='encode only the first N products (benchmarking)')
    args = parser.parse_args(argv)

    catalog_path = args.catalog or (config.CATALOG_SAMPLE if args.sample else config.CATALOG_FULL)
    out_dir = args.out or (config.EMBEDDINGS_SAMPLE if args.sample else config.EMBEDDINGS_FULL)

    catalog = load_catalog(catalog_path)
    if args.limit:
        catalog = catalog.head(args.limit).reset_index(drop=True)

    print(f'Catalog : {catalog_path} ({len(catalog)} products)')
    encoder = ClipEncoder(device=args.device)
    print(f'Encoder : {encoder.model_id} on {encoder.device}, dim={encoder.identity.dim}')
    print(f'Output  : {out_dir}')

    vectors, rows, report = build(catalog, encoder, args.batch_size)
    manifest = build_manifest(catalog, encoder, rows, args.batch_size, report, catalog_path)

    store.publish(out_dir, vectors, rows, manifest,
                  expected_products=report['products_encoded'])

    print(f"\nPublished {len(rows)} rows ({manifest['row_counts']}) to {out_dir}")
    print(f"Index time {report['timings']['total_index_seconds']}s "
          f"({report['timings']['images_per_second']} images/s on {encoder.device})")
    if report['failures']:
        print(f"Recorded {len(report['failures'])} encode exclusions in the manifest.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
