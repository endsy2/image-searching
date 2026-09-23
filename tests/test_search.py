"""End-to-end retrieval: grouping, top-k, filters, exclusions, validation."""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from src import config
from src.encoder import QueryValidationError
from src.search import SearchEngine, open_upload
from tests.helpers import jpeg_bytes


def test_text_search_returns_unique_products_with_a_breakdown(engine):
    payload = engine.search_text('blue tshirt', top_k=3)
    ids = [r['product_id'] for r in payload['results']]
    assert len(ids) == len(set(ids))
    assert len(ids) <= 3
    assert payload['query_type'] == 'text'
    assert payload['results'][0]['rank'] == 1
    assert set(payload['results'][0]['breakdown']) >= {'image', 'title', 'strategy'}


def test_ranks_are_consecutive_and_scores_descend(engine):
    results = engine.search_text('red shoes', top_k=5)['results']
    assert [r['rank'] for r in results] == list(range(1, len(results) + 1))
    scores = [r['final_score'] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_image_search_works_from_an_upload(engine):
    image = open_upload(jpeg_bytes())
    payload = engine.search_image(image, top_k=4)
    assert payload['query_type'] == 'image'
    assert payload['query'] is None
    assert 0 < len(payload['results']) <= 4


def test_both_modality_scores_are_present_for_every_candidate(engine, catalog):
    """A product pulled in by one modality is still scored on the other."""
    titled = set(catalog.loc[catalog['productDisplayName'] != '', 'id'])
    for item in engine.search_text('black watch', top_k=6)['results']:
        breakdown = item['breakdown']
        assert breakdown['image'] is not None
        if item['product_id'] in titled:
            assert breakdown['title'] is not None


def test_a_product_without_a_title_reports_none_not_zero(engine, catalog):
    blank = set(catalog.loc[catalog['productDisplayName'] == '', 'id'])
    payload = engine.search_text('anything at all', top_k=len(catalog))
    found = [r for r in payload['results'] if r['product_id'] in blank]
    assert found, 'the title-less product should still be retrievable by image'
    assert found[0]['breakdown']['title'] is None


def test_top_k_is_capped_and_validated(engine):
    assert len(engine.search_text('shoes', top_k=1)['results']) == 1
    huge = engine.search_text('shoes', top_k=config.MAX_TOP_K + 50)
    assert huge['top_k'] == config.MAX_TOP_K
    with pytest.raises(QueryValidationError):
        engine.search_text('shoes', top_k=0)


def test_filters_are_applied_before_truncation(engine, catalog):
    wanted = catalog['articleType'].iloc[1]
    eligible = set(catalog.loc[catalog['articleType'] == wanted, 'id'])
    payload = engine.search_text('something', top_k=10, filters={'articleType': wanted})
    assert payload['results'], 'the filter should not empty a populated catalog'
    assert {r['product_id'] for r in payload['results']} <= eligible


def test_impossible_filter_gives_a_clean_empty_state(engine):
    payload = engine.search_text('shoes', filters={'baseColour': 'Chartreuse'})
    assert payload['results'] == []
    assert payload['count'] == 0
    assert payload['candidate_pool'] == 0


def test_unknown_filter_column_is_rejected(engine):
    with pytest.raises(QueryValidationError, match='Unknown filter'):
        engine.search_text('shoes', filters={'nonsense': 'x'})


def test_self_exclusion_removes_the_query_product_and_its_duplicates(engine, catalog, fake_root):
    from src.encoder import load_image

    duplicates = sorted(catalog.loc[catalog['image_duplicate_group'] == 'shared', 'id'])
    assert len(duplicates) >= 2
    query_id = duplicates[0]
    image = load_image(fake_root / catalog.loc[catalog['id'] == query_id, 'image_path'].iloc[0])

    included = engine.search_image(image, top_k=10)['results']
    assert query_id in {r['product_id'] for r in included}

    excluded = engine.search_image(image, top_k=10, exclude_product_id=query_id)['results']
    returned = {r['product_id'] for r in excluded}
    assert not returned & set(duplicates)


def test_exclusion_can_keep_duplicates_when_asked(engine, catalog, fake_root):
    from src.encoder import load_image

    duplicates = sorted(catalog.loc[catalog['image_duplicate_group'] == 'shared', 'id'])
    query_id, sibling = duplicates[0], duplicates[1]
    image = load_image(fake_root / catalog.loc[catalog['id'] == query_id, 'image_path'].iloc[0])
    results = engine.search_image(image, top_k=10, exclude_product_id=query_id,
                                  exclude_duplicates=False)['results']
    ids = {r['product_id'] for r in results}
    assert query_id not in ids
    assert sibling in ids


def test_strategies_can_disagree_but_all_run(engine):
    for strategy in ('weighted_sum', 'rrf', 'raw_weighted_sum', 'image_only', 'title_only'):
        payload = engine.search_text('blue', top_k=4, strategy=strategy)
        assert payload['strategy'] == strategy
        assert payload['results']


def test_blank_and_overlong_queries_are_rejected(engine):
    for bad in ('', '   ', None):
        with pytest.raises(QueryValidationError):
            engine.search_text(bad)
    with pytest.raises(QueryValidationError, match='tokens'):
        engine.search_text('shoe ' * 200)


def test_open_upload_rejects_junk_and_empty_files():
    with pytest.raises(QueryValidationError, match='could not be read'):
        open_upload(b'this is not an image')
    with pytest.raises(QueryValidationError, match='empty'):
        open_upload(b'')


def test_open_upload_rejects_oversized_files():
    with pytest.raises(QueryValidationError, match='too large'):
        open_upload(b'x' * (config.MAX_UPLOAD_BYTES + 1))


def test_open_upload_returns_rgb():
    image = open_upload(jpeg_bytes())
    assert image.mode == 'RGB'


def test_image_path_resolution_stays_inside_the_image_directory(engine, catalog):
    path = engine.image_path_for(int(catalog['id'].iloc[0]))
    assert path.exists()
    with pytest.raises(KeyError):
        engine.image_path_for(999999)


def test_check_ready_reports_the_unified_table(engine, catalog):
    info = engine.check_ready()
    assert info['ready'] is True
    assert info['row_counts']['image'] == len(catalog)
    assert info['products'] == len(catalog)
