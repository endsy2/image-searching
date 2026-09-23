"""Load a built vector table into Milvus.

    docker compose -f deploy/milvus/docker-compose.yml up -d
    python -m src.milvus_import --sample          # the 500-product build, ~1k rows
    python -m src.milvus_import --recreate        # the full build, 88,875 rows

Nothing is encoded here: this reads ``vectors.npy`` / ``rows.parquet`` produced by
``src.build_embeddings`` and copies them into a collection. Rebuild the vectors
first if the catalog or the model changed, then re-import with ``--recreate``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from . import config, milvus_store


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--artifacts', type=Path, default=None,
                        help='Embedding artifacts directory (default: artifacts/embeddings)')
    parser.add_argument('--catalog', type=Path, default=None,
                        help='Catalog CSV supplying the filter columns')
    parser.add_argument('--sample', action='store_true',
                        help='Use the 500-product development build')
    parser.add_argument('--collection', default=None,
                        help=f'Collection name (default: {milvus_store.DEFAULT_COLLECTION})')
    parser.add_argument('--uri', default=None,
                        help=f'Milvus URI (default: {milvus_store.DEFAULT_URI})')
    parser.add_argument('--batch-size', type=int, default=1000)
    parser.add_argument('--recreate', action='store_true',
                        help='Drop the collection first if it already exists')
    args = parser.parse_args(argv)

    artifacts = args.artifacts or (
        config.EMBEDDINGS_SAMPLE if args.sample else config.EMBEDDINGS_FULL)
    catalog_path = args.catalog or (
        config.CATALOG_SAMPLE if args.sample else config.CATALOG_FULL)

    if not (artifacts / 'manifest.json').exists():
        print(f'No embedding artifacts at {artifacts}. Build them first with: '
              f'python -m src.build_embeddings', file=sys.stderr)
        return 1
    if not catalog_path.exists():
        print(f'No catalog at {catalog_path}.', file=sys.stderr)
        return 1

    catalog = pd.read_csv(catalog_path, dtype={'id': 'int64'})
    try:
        summary = milvus_store.import_artifacts(
            artifacts_dir=artifacts,
            catalog=catalog,
            uri=args.uri,
            collection=args.collection,
            batch_size=args.batch_size,
            recreate=args.recreate,
        )
    except (milvus_store.MilvusUnavailable, RuntimeError) as error:
        print(str(error), file=sys.stderr)
        return 1

    print(f"\nServe it with:  FASHION_BACKEND=milvus "
          f"MILVUS_COLLECTION={summary['collection']} "
          f"python -m uvicorn app:app")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
