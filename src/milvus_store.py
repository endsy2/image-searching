"""Milvus backend for the unified vector table.

The exact NumPy matmul in ``store.UnifiedStore`` and this module answer the same
question and return the same thing (``search_modalities``): candidate product
ids in discovery order, plus BOTH modality scores for each of them. Everything
downstream -- fusion, filters, the API, the evaluation -- is unchanged.

One collection holds the whole unified table, exactly as the NumPy artifacts do:
image rows and title rows together, told apart by ``embedding_type``. Splitting
them into two collections would make the modality gap someone else's problem.

    collection "fashion_clip"
      row_id          INT64 primary  -- the row's position in rows.parquet
      vector          FLOAT_VECTOR(512), HNSW + COSINE
      product_id      INT64
      embedding_type  VARCHAR  "image" | "title"
      articleType, baseColour, gender, masterCategory, subCategory  VARCHAR

The metadata columns live in the collection so filters run *inside* the search
rather than after it. Identity still comes from the build manifest next to the
vectors: model, preprocessing, and catalog fingerprint must match, or the store
refuses rather than serving quietly wrong results.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, store

DEFAULT_URI = 'http://127.0.0.1:19530'
DEFAULT_COLLECTION = 'fashion_clip'
CONNECT_TIMEOUT = 5.0

# Catalog columns stored alongside the vectors so filters apply before truncation.
SCALAR_FIELDS = ('articleType', 'baseColour', 'gender', 'masterCategory', 'subCategory')
SCALAR_MAX_LENGTH = 128
UNKNOWN = 'Unknown'

METRIC = 'COSINE'
INDEX_TYPE = 'HNSW'
INDEX_PARAMS = {'M': 16, 'efConstruction': 200}


class MilvusUnavailable(RuntimeError):
    """Milvus is not reachable, or holds no collection yet.

    This is the *fall back to NumPy* signal. A collection that exists but does
    not match the running encoder raises ``store.ArtifactError`` instead, which
    is a hard refusal: serving it would return quietly wrong results.
    """


def settings(uri: str | None = None, collection: str | None = None) -> tuple[str, str]:
    """Where Milvus is and which collection to use. Arguments win over the environment."""
    return (
        uri or os.environ.get('MILVUS_URI') or DEFAULT_URI,
        collection or os.environ.get('MILVUS_COLLECTION') or DEFAULT_COLLECTION,
    )


def connect(uri: str | None = None, token: str | None = None, timeout: float = CONNECT_TIMEOUT):
    """Open a client, turning every "it isn't there" into MilvusUnavailable."""
    try:
        from pymilvus import MilvusClient
    except ImportError as error:  # pragma: no cover - depends on the install
        raise MilvusUnavailable(
            'pymilvus is not installed. Install it with: pip install pymilvus'
        ) from error

    target = settings(uri)[0]
    try:
        client = MilvusClient(uri=target, token=token or os.environ.get('MILVUS_TOKEN') or '',
                              timeout=timeout)
        client.list_collections()  # cheap round trip: fail now, not mid-query
    except MilvusUnavailable:
        raise
    except Exception as error:  # noqa: BLE001 - any transport failure means "not running"
        raise MilvusUnavailable(
            f'Cannot reach Milvus at {target} ({type(error).__name__}). '
            f'Start it with: docker compose -f deploy/milvus/docker-compose.yml up -d'
        ) from error
    return client


# -- schema and import ---------------------------------------------------


def build_schema(client, dim: int = config.EMBED_DIM):
    """The collection contract, in one place so import and search cannot drift."""
    from pymilvus import DataType

    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field('row_id', DataType.INT64, is_primary=True)
    schema.add_field('vector', DataType.FLOAT_VECTOR, dim=dim)
    schema.add_field('product_id', DataType.INT64)
    schema.add_field('embedding_type', DataType.VARCHAR, max_length=16)
    for column in SCALAR_FIELDS:
        schema.add_field(column, DataType.VARCHAR, max_length=SCALAR_MAX_LENGTH)
    return schema


def index_params(client):
    params = client.prepare_index_params()
    params.add_index(field_name='vector', index_type=INDEX_TYPE, metric_type=METRIC,
                     params=INDEX_PARAMS)
    return params


def _scalar_frame(rows: pd.DataFrame, catalog: pd.DataFrame | None) -> pd.DataFrame:
    """Attach the filter columns to each row, defaulting to 'Unknown'."""
    frame = rows[['product_id', 'embedding_type']].copy()
    for column in SCALAR_FIELDS:
        if catalog is not None and column in catalog.columns:
            lookup = catalog.set_index('id')[column]
            values = frame['product_id'].map(lookup)
            frame[column] = values.fillna(UNKNOWN).astype(str).str.slice(0, SCALAR_MAX_LENGTH)
        else:
            frame[column] = UNKNOWN
    return frame


def import_artifacts(artifacts_dir: Path, catalog: pd.DataFrame | None = None,
                     uri: str | None = None, collection: str | None = None,
                     batch_size: int = 1000, recreate: bool = False,
                     progress=print) -> dict:
    """Load a built vector table into Milvus. The artifacts on disk stay the source."""
    artifacts_dir = Path(artifacts_dir)
    vectors = np.load(artifacts_dir / store.VECTORS_FILE)
    rows = pd.read_parquet(artifacts_dir / store.ROWS_FILE)
    manifest = json.loads((artifacts_dir / store.MANIFEST_FILE).read_text(encoding='utf-8'))
    store.validate(vectors, rows)  # never import what we would refuse to serve

    client = connect(uri)
    name = settings(uri, collection)[1]

    if client.has_collection(name):
        if not recreate:
            raise RuntimeError(
                f'Collection {name!r} already exists. Re-import with --recreate, '
                f'or choose another name with --collection.'
            )
        progress(f'Dropping existing collection {name!r}.')
        client.drop_collection(name)

    progress(f'Creating collection {name!r} ({len(rows):,} rows, dim {vectors.shape[1]}).')
    client.create_collection(collection_name=name, schema=build_schema(client, vectors.shape[1]),
                             index_params=index_params(client))

    scalars = _scalar_frame(rows, catalog)
    records = scalars.to_dict('records')
    for start in range(0, len(records), batch_size):
        chunk = records[start:start + batch_size]
        client.insert(collection_name=name, data=[
            {
                'row_id': start + offset,
                'vector': vectors[start + offset].tolist(),
                'product_id': int(record['product_id']),
                'embedding_type': str(record['embedding_type']),
                **{column: str(record[column]) for column in SCALAR_FIELDS},
            }
            for offset, record in enumerate(chunk)
        ])
        progress(f'  inserted {min(start + batch_size, len(records)):,} / {len(records):,}')

    client.flush(name)
    client.load_collection(name)
    imported = _count(client, name)
    if imported != len(rows):
        raise RuntimeError(
            f'Milvus holds {imported} rows but the artifacts have {len(rows)}. '
            f'The import did not complete; re-run it with --recreate.'
        )
    progress(f'Done. {imported:,} rows in {name!r}.')
    return {'collection': name, 'rows': imported, 'dim': int(vectors.shape[1]),
            'manifest': manifest}


def _count(client, name: str, expression: str = '') -> int:
    result = client.query(collection_name=name, filter=expression, output_fields=['count(*)'])
    return int(result[0]['count(*)']) if result else 0


# -- serving -------------------------------------------------------------


def _quote(value: str) -> str:
    return '"' + str(value).replace('\\', '\\\\').replace('"', '\\"') + '"'


def build_filter(modality: str, filters: dict | None, excluded) -> str:
    """The search expression: modality, metadata filters, exclusions -- all server-side."""
    clauses = [f'embedding_type == {_quote(modality)}']
    for column, wanted in (filters or {}).items():
        if wanted in (None, '', 'All') or column not in SCALAR_FIELDS:
            continue
        values = wanted if isinstance(wanted, (list, tuple, set)) else [wanted]
        rendered = ', '.join(_quote(v) for v in values)
        clauses.append(f'{column} in [{rendered}]')
    if excluded:
        rendered = ', '.join(str(int(v)) for v in sorted(excluded))
        clauses.append(f'product_id not in [{rendered}]')
    return ' and '.join(clauses)


class MilvusVectorStore:
    """The Milvus-backed unified table, with the same surface search.py needs."""

    backend = 'milvus'

    def __init__(self, client, collection: str, manifest: dict, catalog: pd.DataFrame):
        self.client = client
        self.collection = collection
        self.manifest = manifest
        self.catalog = catalog
        self._counts = {
            modality: _count(client, collection, f'embedding_type == {_quote(modality)}')
            for modality in config.EMBEDDING_TYPES
        }
        meta = catalog.set_index('id')
        self.product_meta = pd.DataFrame({
            'name': meta['productDisplayName'].fillna('').astype(str),
            'image_path': meta['image_path'].astype(str),
        })
        self.product_ids = self.product_meta.index.to_numpy()

    @property
    def dim(self) -> int:
        return int(self.manifest.get('encoder', {}).get('dim', config.EMBED_DIM))

    def counts(self) -> dict[str, int]:
        return dict(self._counts)

    def search_modalities(self, query_vector: np.ndarray, candidate_k: int,
                          filters: dict | None = None,
                          excluded: set[int] | frozenset = frozenset(),
                          ) -> tuple[list[int], dict[int, dict[str, float | None]]]:
        query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        if query.shape[0] != self.dim:
            raise store.ArtifactError(
                f'Query dimension {query.shape[0]} does not match the collection '
                f'dimension {self.dim}.'
            )
        payload = query.tolist()
        candidate_k = int(candidate_k)
        search_params = {'metric_type': METRIC, 'params': {'ef': max(candidate_k * 2, 64)}}

        # One search per modality, then the union -- the same shape as the NumPy
        # path, so neither modality can crowd the other out of the pool.
        ordered: list[int] = []
        seen: set[int] = set()
        found: dict[str, dict[int, float]] = {}
        for modality in config.EMBEDDING_TYPES:
            hits = self.client.search(
                collection_name=self.collection,
                data=[payload],
                anns_field='vector',
                filter=build_filter(modality, filters, excluded),
                limit=candidate_k,
                output_fields=['product_id'],
                search_params=search_params,
            )
            scores: dict[int, float] = {}
            for hit in (hits[0] if hits else []):
                entity = hit.get('entity') or hit
                product_id = int(entity['product_id'])
                scores.setdefault(product_id, float(hit['distance']))
                if product_id not in seen:
                    seen.add(product_id)
                    ordered.append(product_id)
            found[modality] = scores

        # A product pulled in by one modality still needs the other's score.
        # Fetch only those vectors -- at most a few hundred rows -- and dot them.
        for modality in config.EMBEDDING_TYPES:
            missing = [pid for pid in ordered if pid not in found[modality]]
            if not missing:
                continue
            found[modality].update(self._scores_for(modality, missing, query))

        by_product = {
            pid: {modality: found[modality].get(pid) for modality in config.EMBEDDING_TYPES}
            for pid in ordered
        }
        return ordered, by_product

    def _scores_for(self, modality: str, product_ids: list[int],
                    query: np.ndarray) -> dict[int, float]:
        """Cosine of the query against specific rows, for candidates the search missed."""
        rendered = ', '.join(str(int(p)) for p in product_ids)
        rows = self.client.query(
            collection_name=self.collection,
            filter=f'embedding_type == {_quote(modality)} and product_id in [{rendered}]',
            output_fields=['product_id', 'vector'],
        )
        scores: dict[int, float] = {}
        for row in rows:
            vector = np.asarray(row['vector'], dtype=np.float32).reshape(-1)
            scores[int(row['product_id'])] = float(vector @ query)
        return scores


def open_store(artifacts_dir: Path, catalog: pd.DataFrame | None,
               identity_dict: dict | None = None, uri: str | None = None,
               collection: str | None = None) -> MilvusVectorStore:
    """Connect, check the collection matches this encoder and catalog, and load it."""
    if catalog is None:
        raise MilvusUnavailable(
            'The Milvus backend needs the catalog CSV for product names and images.'
        )
    client = connect(uri)
    name = settings(uri, collection)[1]
    if not client.has_collection(name):
        raise MilvusUnavailable(
            f'Milvus has no collection {name!r}. Import one with: '
            f'python -m src.milvus_import'
        )

    manifest_path = Path(artifacts_dir) / store.MANIFEST_FILE
    if not manifest_path.exists():
        raise store.ArtifactError(
            f'No build manifest at {manifest_path}. The Milvus collection is served '
            f'only alongside the manifest of the build it came from.'
        )
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))

    problems = store.manifest_mismatches(
        manifest, identity_dict, store.catalog_fingerprint(catalog)
    )
    rows = _count(client, name)
    if manifest.get('total_rows') not in (None, rows):
        problems.append(
            f"total_rows: manifest has {manifest.get('total_rows')}, "
            f'collection {name!r} holds {rows}'
        )
    if problems:
        raise store.ArtifactError(
            'The Milvus collection does not match the running configuration. '
            'Re-import it.\n  - ' + '\n  - '.join(problems)
        )

    client.load_collection(name)
    return MilvusVectorStore(client, name, manifest, catalog)
