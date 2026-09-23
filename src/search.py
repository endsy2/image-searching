"""Query processing: retrieval from the unified table, grouping, fusion, reranking.

    query -> CLIP vector -> score the whole unified table -> top-K per modality
          -> group by product_id -> (S_image, S_title) -> fuse -> top-k unique products
"""
from __future__ import annotations

import io
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps, UnidentifiedImageError

from . import config, fusion, store
from .encoder import ClipEncoder, QueryValidationError, get_encoder
from .store import ArtifactError

FILTER_COLUMNS = ('articleType', 'baseColour', 'gender', 'masterCategory', 'subCategory')

# 'auto' prefers Milvus and falls back to the exact NumPy table when Milvus
# is not running. FASHION_BACKEND overrides it.
BACKENDS = ('auto', 'milvus', 'numpy')


@dataclass
class SearchResult:
    rank: int
    product_id: int
    name: str
    image_url: str
    final_score: float
    breakdown: dict
    metadata: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            'rank': self.rank,
            'product_id': self.product_id,
            'name': self.name,
            'image_url': self.image_url,
            'final_score': round(self.final_score, 6),
            'breakdown': self.breakdown,
            'metadata': self.metadata,
        }


def open_upload(data: bytes) -> Image.Image:
    """Decode an uploaded image, with user-safe errors."""
    if not data:
        raise QueryValidationError('The uploaded file is empty.')
    if len(data) > config.MAX_UPLOAD_BYTES:
        megabytes = config.MAX_UPLOAD_BYTES / (1024 * 1024)
        raise QueryValidationError(f'Image is too large. The limit is {megabytes:.0f} MB.')
    try:
        with Image.open(io.BytesIO(data)) as handle:
            handle.load()
            return ImageOps.exif_transpose(handle).convert('RGB')
    except (UnidentifiedImageError, OSError, ValueError):
        raise QueryValidationError(
            'That file could not be read as an image. Upload a JPEG or PNG.'
        ) from None


class SearchEngine:
    """Holds the encoder and the unified vector table for the life of the process."""

    def __init__(self, artifacts_dir: Path | None = None, catalog_path: Path | None = None,
                 device: str | None = None, encoder: ClipEncoder | None = None,
                 backend: str | None = None):
        self.artifacts_dir = Path(artifacts_dir or config.EMBEDDINGS_FULL)
        self.catalog_path = Path(catalog_path or config.CATALOG_FULL)
        self.encoder = encoder or get_encoder(device=device)
        self.requested_backend = (backend or os.environ.get('FASHION_BACKEND') or 'auto').lower()
        if self.requested_backend not in BACKENDS:
            raise ArtifactError(
                f'Unknown backend {self.requested_backend!r}. '
                f'Choose from: {", ".join(BACKENDS)}.'
            )

        catalog = None
        if self.catalog_path.exists():
            catalog = pd.read_csv(self.catalog_path, dtype={'id': 'int64'})
            catalog['id'] = catalog['id'].astype('int64')
        self.store, self.backend, self.backend_note = self._open_store(catalog)
        self.catalog = catalog
        self._meta_lookup = self._build_meta_lookup(catalog)
        self._duplicate_groups = self._build_duplicate_groups(catalog)

    # -- backend selection -----------------------------------------------

    def _open_store(self, catalog: pd.DataFrame | None):
        """Milvus when it is running, the exact NumPy table otherwise.

        Only "Milvus is not there" falls back. A collection that exists but does
        not match this encoder and catalog raises ArtifactError either way --
        serving it would return quietly wrong results.
        """
        note = None
        if self.requested_backend in ('auto', 'milvus'):
            from . import milvus_store
            try:
                backing = milvus_store.open_store(
                    artifacts_dir=self.artifacts_dir,
                    catalog=catalog,
                    identity_dict=self.encoder.identity.as_dict(),
                )
                return backing, 'milvus', None
            except milvus_store.MilvusUnavailable as error:
                if self.requested_backend == 'milvus':
                    raise ArtifactError(str(error)) from error
                note = f'Milvus not used: {error}'

        backing = store.load(
            self.artifacts_dir,
            identity_dict=self.encoder.identity.as_dict(),
            catalog=catalog,
        )
        return backing, 'numpy', note

    # -- setup helpers ---------------------------------------------------

    @staticmethod
    def _build_meta_lookup(catalog: pd.DataFrame | None) -> dict[int, dict]:
        if catalog is None:
            return {}
        columns = [c for c in FILTER_COLUMNS if c in catalog.columns]
        subset = catalog.set_index('id')[columns]
        return {int(k): {c: (None if pd.isna(v[c]) else str(v[c])) for c in columns}
                for k, v in subset.to_dict('index').items()}

    @staticmethod
    def _build_duplicate_groups(catalog: pd.DataFrame | None) -> dict[int, list[int]]:
        """product_id -> the products sharing its exact image bytes (including itself)."""
        if catalog is None or 'image_duplicate_group' not in catalog.columns:
            return {}
        grouped = catalog.groupby('image_duplicate_group')['id'].apply(list)
        mapping: dict[int, list[int]] = {}
        for members in grouped:
            if len(members) < 2:
                continue
            members = [int(m) for m in members]
            for member in members:
                mapping[member] = members
        return mapping

    def filter_options(self) -> dict[str, list[str]]:
        if self.catalog is None:
            return {}
        options = {}
        for column in FILTER_COLUMNS:
            if column in self.catalog.columns:
                values = sorted({str(v) for v in self.catalog[column].dropna().unique()})
                options[column] = values
        return options

    def check_ready(self) -> dict:
        return {
            'ready': True,
            'model': self.encoder.identity.as_dict(),
            'device': self.encoder.device,
            'backend': self.backend,
            'backend_note': self.backend_note,
            'artifacts_dir': str(self.artifacts_dir),
            'row_counts': self.store.counts(),
            'products': int(len(self.store.product_ids)),
            'built_at': self.store.manifest.get('built_at'),
        }

    # -- query entry points ----------------------------------------------

    def search_text(self, query: str, **kwargs) -> dict:
        cleaned = self.encoder.validate_text(query)
        started = time.perf_counter()
        vector = self.encoder.encode_texts([cleaned])[0]
        payload = self._search(vector, **kwargs)
        payload['query'] = cleaned
        payload['query_type'] = 'text'
        payload['took_ms'] = round((time.perf_counter() - started) * 1000, 2)
        return payload

    def search_image(self, image: Image.Image, **kwargs) -> dict:
        started = time.perf_counter()
        vector = self.encoder.encode_images([image])[0]
        payload = self._search(vector, **kwargs)
        payload['query'] = None
        payload['query_type'] = 'image'
        payload['took_ms'] = round((time.perf_counter() - started) * 1000, 2)
        return payload

    # -- the pipeline ----------------------------------------------------

    def _search(self, query_vector: np.ndarray, top_k: int = config.DEFAULT_TOP_K,
                strategy: str = 'weighted_sum', weights: dict[str, float] | None = None,
                candidate_k: int = config.DEFAULT_CANDIDATE_K,
                filters: dict[str, str] | None = None,
                exclude_product_id: int | None = None,
                exclude_duplicates: bool = True) -> dict:
        top_k = self._validate_top_k(top_k)
        candidate_k = max(int(candidate_k), top_k)

        excluded = self._excluded_ids(exclude_product_id, exclude_duplicates)
        self._check_filters(filters)

        # Candidate retrieval: top-K per modality, then the union by product_id.
        # Whichever backend is in use answers with BOTH modality scores for every
        # candidate, whichever modality pulled it into the pool.
        candidate_ids, modality_scores = self.store.search_modalities(
            query_vector, candidate_k=candidate_k, filters=filters, excluded=excluded,
        )
        if not candidate_ids:
            return self._empty(top_k, strategy, filters)

        candidates = [
            fusion.Candidate(product_id=pid, scores=modality_scores[pid])
            for pid in candidate_ids
        ]

        fused = fusion.fuse(candidates, strategy=strategy, weights=weights)
        results = [
            SearchResult(
                rank=position + 1,
                product_id=int(item.product_id),
                name=str(self.store.product_meta.at[item.product_id, 'name']),
                image_url=f'/images/{int(item.product_id)}',
                final_score=item.final_score,
                breakdown=item.breakdown,
                metadata=self._meta_lookup.get(int(item.product_id), {}),
            )
            for position, item in enumerate(fused[:top_k])
        ]
        return {
            'results': [r.as_dict() for r in results],
            'count': len(results),
            'top_k': top_k,
            'strategy': strategy,
            'candidate_pool': len(candidate_ids),
            'filters': filters or {},
        }

    @staticmethod
    def _validate_top_k(top_k: int) -> int:
        try:
            value = int(top_k)
        except (TypeError, ValueError):
            raise QueryValidationError('top_k must be a whole number.') from None
        if value < 1:
            raise QueryValidationError('top_k must be at least 1.')
        return min(value, config.MAX_TOP_K)

    def _excluded_ids(self, product_id: int | None, exclude_duplicates: bool) -> set[int]:
        if product_id is None:
            return set()
        excluded = {int(product_id)}
        if exclude_duplicates:
            excluded.update(self._duplicate_groups.get(int(product_id), []))
        return excluded

    def _check_filters(self, filters: dict[str, str] | None) -> None:
        """Reject unknown filter columns here; each backend then applies the filters
        itself, always BEFORE truncation rather than to the top-k."""
        for column, wanted in (filters or {}).items():
            if wanted in (None, '', 'All'):
                continue
            known = self.catalog is not None and column in self.catalog.columns
            if not known and column not in FILTER_COLUMNS:
                raise QueryValidationError(f'Unknown filter {column!r}.')

    @staticmethod
    def _empty(top_k: int, strategy: str, filters: dict | None) -> dict:
        return {'results': [], 'count': 0, 'top_k': top_k, 'strategy': strategy,
                'candidate_pool': 0, 'filters': filters or {}}

    def image_path_for(self, product_id: int) -> Path:
        """Resolve a catalog image path, validated against the catalog, never user input."""
        pid = int(product_id)
        if pid not in self.store.product_meta.index:
            raise KeyError(pid)
        relative = str(self.store.product_meta.at[pid, 'image_path'])
        resolved = (config.ROOT / relative).resolve()
        if not resolved.is_relative_to(config.IMAGES_DIR.resolve()):
            raise KeyError(pid)
        return resolved
