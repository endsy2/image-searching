"""The unified vector table: load, validate, and publish.

All embeddings — image and title alike — live in ONE vector matrix. Row ``i`` of
``vectors.npy`` is described by row ``i`` of ``rows.parquet``, which carries the
``embedding_type`` tag (``"image"`` or ``"title"``) plus the product metadata.
That shared row order is the only linkage between a vector and its product, so
anything writing one writes the other in the same pass.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

VECTORS_FILE = 'vectors.npy'
ROWS_FILE = 'rows.parquet'
MANIFEST_FILE = 'manifest.json'

ROW_COLUMNS = ['product_id', 'embedding_type', 'name', 'image_path']

UNIT_NORM_TOLERANCE = 1e-4


class ArtifactError(RuntimeError):
    """Artifacts are missing, corrupt, or do not match the running configuration."""


def catalog_fingerprint(catalog: pd.DataFrame) -> str:
    """Stable hash of the catalog identity: which products, in which order."""
    import hashlib

    ids = catalog['id'].astype('int64').to_numpy()
    digest = hashlib.sha256()
    digest.update(str(len(ids)).encode())
    digest.update(ids.tobytes())
    return digest.hexdigest()



def allowed_ids(catalog: pd.DataFrame | None, filters: dict | None) -> set[int] | None:
    """Resolve metadata filters into an allowed product-id set over the FULL catalog.

    Returning ``None`` means "no filter". Filters are resolved here, before any
    truncation, because filtering a top-k after the fact would return 2 results
    where 10 exist. Column names are validated by the caller.
    """
    if not filters or catalog is None:
        return None
    mask = pd.Series(True, index=catalog.index)
    used = False
    for column, wanted in filters.items():
        if wanted in (None, '', 'All') or column not in catalog.columns:
            continue
        values = wanted if isinstance(wanted, (list, tuple, set)) else [wanted]
        mask &= catalog[column].astype(str).isin([str(v) for v in values])
        used = True
    if not used:
        return None
    return {int(v) for v in catalog.loc[mask, 'id']}


@dataclass
class UnifiedStore:
    """An in-memory view of the unified vector table."""

    vectors: np.ndarray           # (N, dim) float32, unit-norm
    rows: pd.DataFrame            # N rows, ROW_COLUMNS
    manifest: dict
    catalog: pd.DataFrame | None = None

    # Row indices split by modality, and product -> row lookups.
    def __post_init__(self) -> None:
        self.type_index: dict[str, np.ndarray] = {}
        self.row_of: dict[str, dict[int, int]] = {}
        types = self.rows['embedding_type'].to_numpy()
        product_ids = self.rows['product_id'].to_numpy()
        for embedding_type in config.EMBEDDING_TYPES:
            positions = np.flatnonzero(types == embedding_type)
            self.type_index[embedding_type] = positions
            self.row_of[embedding_type] = {
                int(product_ids[p]): int(p) for p in positions
            }
        first_row = self.rows.drop_duplicates('product_id').set_index('product_id')
        self.product_meta = first_row[['name', 'image_path']]
        self.product_ids = self.product_meta.index.to_numpy()

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[1])

    def counts(self) -> dict[str, int]:
        return {t: int(len(idx)) for t, idx in self.type_index.items()}

    def score_all(self, query_vector: np.ndarray) -> np.ndarray:
        """Cosine similarity of the query against every row of the unified table."""
        query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        if query.shape[0] != self.dim:
            raise ArtifactError(
                f'Query dimension {query.shape[0]} does not match the stored '
                f'dimension {self.dim}.'
            )
        return self.vectors @ query

    # -- the retrieval contract ------------------------------------------
    #
    # Every backend (this one, and MilvusVectorStore) answers exactly this:
    # candidate product ids in discovery order, plus BOTH modality scores for
    # each of them. A modality a product genuinely lacks is None, never 0.0.

    def search_modalities(self, query_vector: np.ndarray, candidate_k: int,
                          filters: dict | None = None,
                          excluded: set[int] | frozenset = frozenset(),
                          ) -> tuple[list[int], dict[int, dict[str, float | None]]]:
        scores = self.score_all(query_vector)
        allowed = allowed_ids(self.catalog, filters)

        # Top-K per modality, then the union. A global top-K over the mixed
        # table would let one modality crowd the other out of the pool.
        ordered: list[int] = []
        seen: set[int] = set()
        product_ids = self.rows['product_id'].to_numpy()
        for modality in config.EMBEDDING_TYPES:
            positions = self.type_index[modality]
            if positions.size == 0:
                continue
            modality_ids = product_ids[positions]
            keep = np.ones(positions.shape, dtype=bool)
            if excluded:
                keep &= ~np.isin(modality_ids, list(excluded))
            if allowed is not None:
                keep &= np.isin(modality_ids, list(allowed))
            if not keep.any():
                continue
            kept_scores = scores[positions][keep]
            kept_ids = modality_ids[keep]

            take = min(int(candidate_k), kept_scores.size)
            top = np.argpartition(-kept_scores, take - 1)[:take]
            for pid in kept_ids[top[np.argsort(-kept_scores[top])]]:
                pid = int(pid)
                if pid not in seen:
                    seen.add(pid)
                    ordered.append(pid)

        # Both modality scores come straight out of the score vector already
        # computed -- no second pass, no re-encoding.
        by_product = {
            pid: {
                modality: (float(scores[self.row_of[modality][pid]])
                           if pid in self.row_of[modality] else None)
                for modality in config.EMBEDDING_TYPES
            }
            for pid in ordered
        }
        return ordered, by_product


# -- validation ----------------------------------------------------------


def validate(vectors: np.ndarray, rows: pd.DataFrame, expected_products: int | None = None) -> None:
    """Every invariant the build must satisfy before it may be published."""
    if vectors.ndim != 2:
        raise ArtifactError(f'vectors must be 2-D, got shape {vectors.shape}.')
    if vectors.dtype != np.float32:
        raise ArtifactError(f'vectors must be float32, got {vectors.dtype}.')
    if len(rows) != len(vectors):
        raise ArtifactError(
            f'Row table has {len(rows)} rows but the matrix has {len(vectors)}.'
        )
    if len(rows) == 0:
        raise ArtifactError('The unified table is empty.')

    missing = [c for c in ROW_COLUMNS if c not in rows.columns]
    if missing:
        raise ArtifactError(f'Row table is missing columns: {missing}.')

    if not np.isfinite(vectors).all():
        raise ArtifactError('Some stored vectors contain NaN or infinity.')
    norms = np.linalg.norm(vectors, axis=1)
    worst = float(np.max(np.abs(norms - 1.0)))
    if worst > UNIT_NORM_TOLERANCE:
        raise ArtifactError(
            f'Some stored vectors are not unit-norm (worst deviation {worst:.2e}).'
        )

    unknown = set(rows['embedding_type'].unique()) - set(config.EMBEDDING_TYPES)
    if unknown:
        raise ArtifactError(f'Unknown embedding_type values: {sorted(unknown)}.')

    duplicated = rows.duplicated(['product_id', 'embedding_type'])
    if duplicated.any():
        example = rows.loc[duplicated, ['product_id', 'embedding_type']].head(3)
        raise ArtifactError(
            f'A product has more than one row of the same embedding_type:\n{example}'
        )

    image_rows = rows[rows['embedding_type'] == 'image']
    if expected_products is not None and len(image_rows) != expected_products:
        raise ArtifactError(
            f'Expected {expected_products} image rows, found {len(image_rows)}.'
        )


def manifest_mismatches(manifest: dict, identity_dict: dict | None,
                         fingerprint: str | None) -> list[str]:
    problems: list[str] = []
    if identity_dict is not None:
        stored = manifest.get('encoder', {})
        for key in ('model_id', 'dim', 'image_size', 'context_length'):
            if stored.get(key) != identity_dict.get(key):
                problems.append(
                    f'{key}: artifact has {stored.get(key)!r}, '
                    f'running encoder has {identity_dict.get(key)!r}'
                )
        for key in ('image_mean', 'image_std'):
            left = [round(float(v), 6) for v in stored.get(key, [])]
            right = [round(float(v), 6) for v in identity_dict.get(key, [])]
            if left != right:
                problems.append(f'{key}: artifact has {left}, running encoder has {right}')
    if fingerprint is not None and manifest.get('catalog_fingerprint') != fingerprint:
        problems.append(
            f"catalog_fingerprint: artifact has "
            f"{manifest.get('catalog_fingerprint')!r}, catalog on disk has {fingerprint!r}"
        )
    return problems


# -- publishing ----------------------------------------------------------


def publish(target: Path, vectors: np.ndarray, rows: pd.DataFrame, manifest: dict,
            expected_products: int | None = None) -> Path:
    """Write to a temporary directory, validate the reload, then swap into place."""
    target = Path(target)
    validate(vectors, rows, expected_products=expected_products)

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f'.{target.name}.staging.', dir=target.parent))
    try:
        np.save(staging / VECTORS_FILE, vectors)
        rows[ROW_COLUMNS].reset_index(drop=True).to_parquet(staging / ROWS_FILE, index=False)
        # Manifest is written last so a half-built directory can never look complete.
        (staging / MANIFEST_FILE).write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding='utf-8'
        )

        # Validate what actually landed on disk, not what we think we wrote.
        reloaded_vectors = np.load(staging / VECTORS_FILE)
        reloaded_rows = pd.read_parquet(staging / ROWS_FILE)
        validate(reloaded_vectors, reloaded_rows, expected_products=expected_products)
        if not np.array_equal(reloaded_vectors, vectors):
            raise ArtifactError('Reloaded vectors differ from the built vectors.')

        if target.exists():
            retired = target.with_name(target.name + '.previous')
            shutil.rmtree(retired, ignore_errors=True)
            os.replace(target, retired)
            shutil.rmtree(retired, ignore_errors=True)
        os.replace(staging, target)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return target


# -- loading -------------------------------------------------------------


def load(directory: Path, identity_dict: dict | None = None,
         catalog: pd.DataFrame | None = None) -> UnifiedStore:
    """Load and validate the unified table, refusing mismatched artifacts."""
    directory = Path(directory)
    if not (directory / MANIFEST_FILE).exists():
        raise ArtifactError(
            f'No embedding artifacts at {directory}. '
            f'Build them with: python -m src.build_embeddings'
        )
    manifest = json.loads((directory / MANIFEST_FILE).read_text(encoding='utf-8'))
    vectors = np.load(directory / VECTORS_FILE)
    rows = pd.read_parquet(directory / ROWS_FILE)
    validate(vectors, rows)

    fingerprint = catalog_fingerprint(catalog) if catalog is not None else None
    problems = manifest_mismatches(manifest, identity_dict, fingerprint)
    if problems:
        raise ArtifactError(
            'Embedding artifacts do not match the running configuration. Rebuild '
            'them.\n  - ' + '\n  - '.join(problems)
        )
    return UnifiedStore(vectors=vectors, rows=rows, manifest=manifest, catalog=catalog)
