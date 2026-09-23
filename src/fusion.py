"""Combining and reranking strategies.

A candidate product carries one score per modality, ``S_image`` and ``S_title``.
These are *not* on a comparable scale. In CLIP a text query sits systematically
closer to text vectors than to image vectors - the modality gap - so the two
score distributions differ in both mean and spread. The mean offset is harmless
on its own, since a constant shift cannot reorder candidates; what actually
distorts a raw sum is the difference in *spread*. Measured on this catalog
(``python -m src.verify_encoder``), a text query scores 0.46 +/- 0.11 against
title vectors and 0.18 +/- 0.04 against image vectors, so in a raw 50/50 sum the
title modality exerts roughly three times the ranking influence it was given.

Three strategies are provided, all behind one interface:

``weighted_sum``       per-modality min-max normalization over the candidate pool,
                       then a convex combination. The default.
``rrf``                Reciprocal Rank Fusion, scale-free by construction.
``raw_weighted_sum``   the naive reference, kept so the report can show why the
                       other two exist.

A modality a product genuinely lacks is ``None``, never zero. Under the weighted
sums its weight is redistributed onto the modalities the product does have, so a
title-less product is ranked on its image score rather than penalised for a gap
in the catalog. Under RRF it simply contributes no rank term.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import config

STRATEGIES = ('weighted_sum', 'rrf', 'raw_weighted_sum', 'image_only', 'title_only')


@dataclass
class Candidate:
    """One product under consideration, with its per-modality similarities."""

    product_id: int
    scores: dict[str, float | None] = field(default_factory=dict)

    def score(self, modality: str) -> float | None:
        value = self.scores.get(modality)
        return None if value is None else float(value)


@dataclass
class FusedResult:
    product_id: int
    final_score: float
    breakdown: dict


def _minmax(values: list[float]) -> dict:
    """Min-max parameters over the candidate pool. Degenerate pools map to 1.0."""
    if not values:
        return {'min': 0.0, 'span': 0.0}
    low, high = min(values), max(values)
    return {'min': low, 'span': high - low}


def _apply_minmax(value: float, params: dict) -> float:
    if params['span'] <= 1e-12:
        # Every candidate scored identically on this modality: it carries no
        # ranking information, so give them all the same normalized score.
        return 1.0
    return (value - params['min']) / params['span']


def _normalized_weights(present: list[str], weights: dict[str, float]) -> dict[str, float]:
    """Redistribute weight across the modalities a product actually has."""
    total = sum(weights.get(m, 0.0) for m in present)
    if total <= 0:
        return {m: 1.0 / len(present) for m in present} if present else {}
    return {m: weights.get(m, 0.0) / total for m in present}


def fuse(candidates: list[Candidate], strategy: str = 'weighted_sum',
         weights: dict[str, float] | None = None,
         rrf_k: int = config.RRF_K) -> list[FusedResult]:
    """Combine per-modality scores into one ranking score per product.

    Returns results sorted by ``final_score`` descending, one entry per product.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f'Unknown fusion strategy {strategy!r}. Choose from {STRATEGIES}.')
    if not candidates:
        return []

    weights = dict(weights or {m: 1.0 / len(config.EMBEDDING_TYPES)
                               for m in config.EMBEDDING_TYPES})

    if strategy in ('image_only', 'title_only'):
        modality = 'image' if strategy == 'image_only' else 'title'
        results = []
        for candidate in candidates:
            value = candidate.score(modality)
            if value is None:
                continue
            results.append(FusedResult(
                candidate.product_id, value,
                {'strategy': strategy, **candidate.scores},
            ))
        return sorted(results, key=lambda r: (-r.final_score, r.product_id))

    if strategy == 'rrf':
        return _fuse_rrf(candidates, rrf_k)

    normalize = strategy == 'weighted_sum'
    params = {}
    if normalize:
        for modality in config.EMBEDDING_TYPES:
            present = [c.score(modality) for c in candidates]
            params[modality] = _minmax([v for v in present if v is not None])

    results = []
    for candidate in candidates:
        present = [m for m in config.EMBEDDING_TYPES if candidate.score(m) is not None]
        if not present:
            continue
        effective = _normalized_weights(present, weights)
        components: dict[str, float] = {}
        total = 0.0
        for modality in present:
            raw = candidate.score(modality)
            value = _apply_minmax(raw, params[modality]) if normalize else raw
            components[modality] = round(value, 6)
            total += effective[modality] * value
        results.append(FusedResult(
            candidate.product_id, float(total),
            {
                'strategy': strategy,
                **{m: candidate.score(m) for m in config.EMBEDDING_TYPES},
                'normalized': components if normalize else None,
                'weights': {m: round(w, 4) for m, w in effective.items()},
            },
        ))
    return sorted(results, key=lambda r: (-r.final_score, r.product_id))


def _fuse_rrf(candidates: list[Candidate], rrf_k: int) -> list[FusedResult]:
    ranks: dict[str, dict[int, int]] = {}
    for modality in config.EMBEDDING_TYPES:
        scored = [(c.product_id, c.score(modality)) for c in candidates]
        scored = [(pid, s) for pid, s in scored if s is not None]
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        ranks[modality] = {pid: position + 1 for position, (pid, _) in enumerate(scored)}

    results = []
    for candidate in candidates:
        contributions: dict[str, float] = {}
        total = 0.0
        for modality in config.EMBEDDING_TYPES:
            rank = ranks[modality].get(candidate.product_id)
            if rank is None:
                continue
            contribution = 1.0 / (rrf_k + rank)
            contributions[modality] = round(contribution, 8)
            total += contribution
        if not contributions:
            continue
        results.append(FusedResult(
            candidate.product_id, float(total),
            {
                'strategy': 'rrf',
                **{m: candidate.score(m) for m in config.EMBEDDING_TYPES},
                'ranks': {m: ranks[m].get(candidate.product_id) for m in config.EMBEDDING_TYPES},
                'rrf_k': rrf_k,
            },
        ))
    return sorted(results, key=lambda r: (-r.final_score, r.product_id))
