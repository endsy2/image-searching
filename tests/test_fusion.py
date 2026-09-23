"""Fusion and reranking behaviour on known score inputs."""
from __future__ import annotations

import pytest

from src import config
from src.fusion import Candidate, fuse


def candidates(*pairs):
    return [Candidate(product_id=pid, scores={'image': img, 'title': ttl})
            for pid, img, ttl in pairs]


def order(results):
    return [r.product_id for r in results]


def test_raw_weighted_sum_is_a_plain_average():
    results = fuse(candidates((1, 0.8, 0.2), (2, 0.4, 0.4)), strategy='raw_weighted_sum')
    assert order(results) == [1, 2]
    assert results[0].final_score == pytest.approx(0.5)
    assert results[1].final_score == pytest.approx(0.4)


def test_raw_sum_lets_the_wider_spread_modality_dominate():
    """The modality gap in miniature.

    What breaks a raw sum is not the difference in *means* - a constant offset
    shifts every candidate equally and cannot reorder them - but the difference
    in *spread*. Here the title scores range over 0.40 while the image scores
    range over 0.18, so under a raw 50/50 sum the title modality carries roughly
    twice the ranking influence it was given. Product 1, the clear image winner,
    comes last. Normalizing each modality over the pool restores the intended
    50/50 and puts it first.
    """
    pool = candidates((1, 0.30, 0.30), (2, 0.16, 0.50), (3, 0.12, 0.70))
    assert order(fuse(pool, strategy='raw_weighted_sum')) == [3, 2, 1]
    assert order(fuse(pool, strategy='weighted_sum'))[0] == 1


def test_weighted_sum_normalizes_over_the_candidate_pool():
    pool = candidates((1, 0.9, 0.1), (2, 0.5, 0.5), (3, 0.1, 0.9))
    results = {r.product_id: r for r in fuse(pool, strategy='weighted_sum')}
    # The pool min maps to 0 and the pool max to 1 on each modality separately.
    assert results[1].breakdown['normalized']['image'] == pytest.approx(1.0)
    assert results[3].breakdown['normalized']['image'] == pytest.approx(0.0)
    assert results[1].breakdown['normalized']['title'] == pytest.approx(0.0)
    assert results[3].breakdown['normalized']['title'] == pytest.approx(1.0)


def test_normalization_is_pool_relative_not_global():
    """Removing a candidate changes the pool, so it changes the normalized values."""
    wide = candidates((1, 0.9, 0.5), (2, 0.5, 0.5), (3, 0.1, 0.5))
    narrow = candidates((1, 0.9, 0.5), (2, 0.5, 0.5))
    first = {r.product_id: r for r in fuse(wide, strategy='weighted_sum')}
    second = {r.product_id: r for r in fuse(narrow, strategy='weighted_sum')}
    assert first[2].breakdown['normalized']['image'] == pytest.approx(0.5)
    assert second[2].breakdown['normalized']['image'] == pytest.approx(0.0)


def test_degenerate_modality_carries_no_ranking_information():
    pool = candidates((1, 0.3, 0.7), (2, 0.3, 0.2))
    results = {r.product_id: r for r in fuse(pool, strategy='weighted_sum')}
    assert results[1].breakdown['normalized']['image'] == pytest.approx(1.0)
    assert results[2].breakdown['normalized']['image'] == pytest.approx(1.0)
    assert order(fuse(pool, strategy='weighted_sum')) == [1, 2]


def test_weights_shift_the_ranking():
    pool = candidates((1, 0.9, 0.1), (2, 0.1, 0.9))
    assert order(fuse(pool, 'weighted_sum', weights={'image': 0.9, 'title': 0.1})) == [1, 2]
    assert order(fuse(pool, 'weighted_sum', weights={'image': 0.1, 'title': 0.9})) == [2, 1]


def test_missing_modality_is_not_a_silent_zero():
    """A title-less product is ranked on its image score, not penalised to zero."""
    pool = [
        Candidate(1, {'image': 0.9, 'title': None}),
        Candidate(2, {'image': 0.2, 'title': 0.9}),
    ]
    results = {r.product_id: r for r in fuse(pool, strategy='weighted_sum')}
    # Product 1's image weight is redistributed to 1.0 rather than halved.
    assert results[1].breakdown['weights'] == {'image': 1.0}
    assert results[1].final_score == pytest.approx(1.0)
    assert results[1].breakdown['title'] is None
    assert order(fuse(pool, strategy='weighted_sum')) == [1, 2]


def test_a_candidate_with_no_modality_at_all_is_dropped():
    pool = [Candidate(1, {'image': None, 'title': None}), Candidate(2, {'image': 0.5, 'title': 0.5})]
    assert order(fuse(pool, strategy='weighted_sum')) == [2]


def test_rrf_is_scale_free():
    """RRF depends only on rank order, so rescaling one modality changes nothing."""
    pool = candidates((1, 0.9, 0.1), (2, 0.5, 0.5), (3, 0.1, 0.9))
    rescaled = candidates((1, 0.009, 0.1), (2, 0.005, 0.5), (3, 0.001, 0.9))
    assert order(fuse(pool, strategy='rrf')) == order(fuse(rescaled, strategy='rrf'))


def test_rrf_scores_match_the_formula():
    results = {r.product_id: r for r in fuse(candidates((1, 0.9, 0.9), (2, 0.1, 0.1)),
                                             strategy='rrf')}
    k = config.RRF_K
    assert results[1].final_score == pytest.approx(2 / (k + 1))
    assert results[2].final_score == pytest.approx(2 / (k + 2))
    assert results[1].breakdown['ranks'] == {'image': 1, 'title': 1}


def test_rrf_skips_a_missing_modality_rather_than_ranking_it_last():
    pool = [Candidate(1, {'image': 0.9, 'title': None}), Candidate(2, {'image': 0.1, 'title': 0.9})]
    results = {r.product_id: r for r in fuse(pool, strategy='rrf')}
    assert results[1].breakdown['ranks']['title'] is None
    assert results[1].final_score == pytest.approx(1 / (config.RRF_K + 1))


def test_single_modality_strategies_use_only_that_modality():
    pool = candidates((1, 0.9, 0.1), (2, 0.1, 0.9))
    assert order(fuse(pool, strategy='image_only')) == [1, 2]
    assert order(fuse(pool, strategy='title_only')) == [2, 1]


def test_image_only_drops_products_without_that_modality():
    pool = [Candidate(1, {'image': None, 'title': 0.9}), Candidate(2, {'image': 0.5, 'title': 0.1})]
    assert order(fuse(pool, strategy='image_only')) == [2]


def test_results_are_sorted_descending_and_ties_are_deterministic():
    results = fuse(candidates((7, 0.5, 0.5), (3, 0.5, 0.5)), strategy='raw_weighted_sum')
    assert order(results) == [3, 7]
    scores = [r.final_score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_unknown_strategy_is_rejected():
    with pytest.raises(ValueError, match='Unknown fusion strategy'):
        fuse(candidates((1, 0.5, 0.5)), strategy='magic')


def test_empty_pool_returns_no_results():
    assert fuse([], strategy='weighted_sum') == []
