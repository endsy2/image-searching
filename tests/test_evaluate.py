"""Evaluation arithmetic and the keyword baseline."""
from __future__ import annotations

import pandas as pd
import pytest

from src.baseline import KeywordBaseline, tokenize
from src.evaluate import _precision_at_5


def test_precision_counts_only_grade_two():
    grades = {1: 2, 2: 2, 3: 1, 4: 0, 5: 0}
    assert _precision_at_5([1, 2, 3, 4, 5], grades) == pytest.approx(0.4)


def test_precision_is_none_when_any_of_the_top_five_is_unjudged():
    """A blank cell means unjudged, never 0 - the query is skipped, not scored low."""
    grades = {1: 2, 2: 2, 3: 2, 4: 2}
    assert _precision_at_5([1, 2, 3, 4, 5], grades) is None


def test_precision_uses_only_the_top_five():
    grades = {1: 2, 2: 2, 3: 2, 4: 2, 5: 2}
    assert _precision_at_5([1, 2, 3, 4, 5, 6, 7], grades) == pytest.approx(1.0)


def test_precision_of_an_empty_result_list_is_zero():
    assert _precision_at_5([], {}) == 0.0


def test_precision_of_a_short_result_list_divides_by_its_length():
    assert _precision_at_5([1, 2], {1: 2, 2: 0}) == pytest.approx(0.5)


def test_tokenizer_lowercases_and_drops_punctuation():
    assert tokenize("Men's Navy-Blue Shirt!") == ['men', 's', 'navy', 'blue', 'shirt']


def test_baseline_ranks_an_exact_phrase_match_first(catalog):
    baseline = KeywordBaseline(catalog)
    target = catalog.iloc[1]
    results = baseline.search(str(target['productDisplayName']), top_k=3)
    assert results[0]['product_id'] == int(target['id'])


def test_baseline_returns_nothing_for_out_of_vocabulary_terms(catalog):
    baseline = KeywordBaseline(catalog)
    assert baseline.search('xyzzy plugh', top_k=5) == []


def test_baseline_respects_the_allowed_set(catalog):
    baseline = KeywordBaseline(catalog)
    allowed = {int(catalog['id'].iloc[-1])}
    results = baseline.search('Brand', top_k=5, allowed=allowed)
    assert {r['product_id'] for r in results} <= allowed
