"""The Milvus backend, against a fake client: no Docker, no server, no network.

These cover the parts that are ours rather than Milvus's -- the search
expression, the union of the two modality searches, and the fallback rule --
so the suite still runs anywhere. Agreement with the exact NumPy table is
checked against a real server; see README, "Milvus".
"""
from __future__ import annotations

import numpy as np
import pytest

from src import config, milvus_store, store
from src.milvus_store import MilvusUnavailable, MilvusVectorStore, build_filter
from src.search import SearchEngine
from tests.helpers import build_artifacts, make_catalog


class FakeClient:
    """An in-memory stand-in with just the MilvusClient methods the store calls."""

    def __init__(self, vectors: np.ndarray, rows, catalog):
        self.vectors = vectors
        self.rows = rows.reset_index(drop=True)
        self.catalog = catalog.set_index('id')
        self.searches: list[str] = []
        self.loaded: list[str] = []

    # -- the surface MilvusVectorStore uses ------------------------------

    def query(self, collection_name, filter='', output_fields=None, **kwargs):
        selected = self._select(filter)
        if output_fields and 'count(*)' in output_fields:
            return [{'count(*)': len(selected)}]
        return [
            {'product_id': int(self.rows.at[i, 'product_id']),
             'vector': self.vectors[i].tolist()}
            for i in selected
        ]

    def search(self, collection_name, data, filter='', limit=10, **kwargs):
        self.searches.append(filter)
        query = np.asarray(data[0], dtype=np.float32)
        scored = sorted(
            ((float(self.vectors[i] @ query), i) for i in self._select(filter)),
            reverse=True,
        )[:limit]
        return [[{'id': i, 'distance': score,
                  'entity': {'product_id': int(self.rows.at[i, 'product_id'])}}
                 for score, i in scored]]

    def has_collection(self, name):
        return True

    def load_collection(self, name, **kwargs):
        self.loaded.append(name)

    # -- a deliberately tiny expression evaluator ------------------------

    def _select(self, expression: str) -> list[int]:
        keep = []
        for i in range(len(self.rows)):
            product_id = int(self.rows.at[i, 'product_id'])
            row = {'embedding_type': str(self.rows.at[i, 'embedding_type'])}
            for column in milvus_store.SCALAR_FIELDS:
                value = self.catalog.at[product_id, column] if column in self.catalog else None
                row[column] = milvus_store.UNKNOWN if value is None else str(value)
            if self._matches(expression, row, product_id):
                keep.append(i)
        return keep

    @staticmethod
    def _matches(expression: str, row: dict, product_id: int) -> bool:
        for clause in expression.split(' and '):
            clause = clause.strip()
            if not clause:
                continue
            if clause.startswith('product_id not in ['):
                blocked = clause[clause.index('[') + 1:clause.rindex(']')]
                if product_id in {int(v) for v in blocked.split(',') if v.strip()}:
                    return False
            elif clause.startswith('product_id in ['):
                wanted = clause[clause.index('[') + 1:clause.rindex(']')]
                if product_id not in {int(v) for v in wanted.split(',') if v.strip()}:
                    return False
            elif ' in [' in clause:
                column = clause.split(' in [')[0].strip()
                values = {v.strip().strip('"')
                          for v in clause[clause.index('[') + 1:clause.rindex(']')].split(',')}
                if row.get(column) not in values:
                    return False
            elif ' == ' in clause:
                column, value = clause.split(' == ')
                if row.get(column.strip()) != value.strip().strip('"'):
                    return False
        return True


@pytest.fixture
def milvus_store_pair(fake_root, catalog, encoder):
    """The same synthetic build, behind both backends."""
    artifacts = build_artifacts(fake_root, catalog, encoder)
    numpy_backed = store.load(artifacts, identity_dict=encoder.identity.as_dict(),
                              catalog=catalog)
    client = FakeClient(numpy_backed.vectors, numpy_backed.rows, catalog)
    milvus_backed = MilvusVectorStore(client, 'test_collection',
                                      numpy_backed.manifest, catalog)
    return numpy_backed, milvus_backed, client


# -- the search expression ----------------------------------------------


def test_filter_pins_the_modality():
    assert build_filter('image', None, set()) == 'embedding_type == "image"'


def test_filter_carries_metadata_and_exclusions():
    expression = build_filter('title', {'gender': 'Men', 'baseColour': ''}, {7, 3})
    assert 'embedding_type == "title"' in expression
    assert 'gender in ["Men"]' in expression
    assert 'baseColour' not in expression          # blank means "no filter"
    assert 'product_id not in [3, 7]' in expression


def test_filter_escapes_quotes_rather_than_closing_the_expression():
    expression = build_filter('image', {'articleType': 'a" or "1'}, set())
    assert '\\"' in expression


def test_unknown_filter_columns_are_dropped_not_sent():
    assert 'nonsense' not in build_filter('image', {'nonsense': 'x'}, set())


# -- retrieval -----------------------------------------------------------


def test_both_backends_return_the_same_candidates(milvus_store_pair, encoder):
    numpy_backed, milvus_backed, _ = milvus_store_pair
    query = encoder.encode_texts(['blue shirt'])[0]

    numpy_ids, numpy_scores = numpy_backed.search_modalities(query, candidate_k=50)
    milvus_ids, milvus_scores = milvus_backed.search_modalities(query, candidate_k=50)

    assert set(numpy_ids) == set(milvus_ids)
    for product_id in numpy_ids:
        for modality in config.EMBEDDING_TYPES:
            left = numpy_scores[product_id][modality]
            right = milvus_scores[product_id][modality]
            if left is None:
                assert right is None
            else:
                assert right == pytest.approx(left, abs=1e-5)


def test_a_missing_modality_is_none_not_zero(milvus_store_pair, catalog, encoder):
    _, milvus_backed, _ = milvus_store_pair
    blank = {int(i) for i in catalog.loc[catalog['productDisplayName'] == '', 'id']}
    assert blank, 'the synthetic catalog should include a product with no title'
    _, scores = milvus_backed.search_modalities(
        encoder.encode_texts(['anything'])[0], candidate_k=100)
    for product_id in blank & set(scores):
        assert scores[product_id]['title'] is None
        assert scores[product_id]['image'] is not None


def test_each_modality_is_searched_separately(milvus_store_pair, encoder):
    """A global top-K over the mixed table would let one modality crowd the other out."""
    _, milvus_backed, client = milvus_store_pair
    milvus_backed.search_modalities(encoder.encode_texts(['red dress'])[0], candidate_k=5)
    assert len(client.searches) == len(config.EMBEDDING_TYPES)
    for modality in config.EMBEDDING_TYPES:
        assert any(f'embedding_type == "{modality}"' in s for s in client.searches)


def test_excluded_products_never_come_back(milvus_store_pair, encoder):
    numpy_backed, milvus_backed, _ = milvus_store_pair
    excluded = {int(numpy_backed.product_ids[0])}
    ids, _ = milvus_backed.search_modalities(
        encoder.encode_texts(['shoes'])[0], candidate_k=100, excluded=excluded)
    assert excluded.isdisjoint(ids)


def test_a_wrong_sized_query_is_refused(milvus_store_pair):
    _, milvus_backed, _ = milvus_store_pair
    with pytest.raises(store.ArtifactError):
        milvus_backed.search_modalities(np.zeros(3, dtype=np.float32), candidate_k=5)


# -- backend selection ---------------------------------------------------


def test_auto_falls_back_to_numpy_when_milvus_is_down(fake_root, catalog, encoder,
                                                      artifacts, monkeypatch):
    def refuse(*args, **kwargs):
        raise MilvusUnavailable('Cannot reach Milvus at http://127.0.0.1:19530')

    monkeypatch.setattr(milvus_store, 'open_store', refuse)
    engine = SearchEngine(artifacts_dir=artifacts, catalog_path=fake_root / 'catalog.csv',
                          encoder=encoder, backend='auto')
    assert engine.backend == 'numpy'
    assert 'Milvus not used' in engine.backend_note
    assert engine.search_text('blue shirt', top_k=3)['results']


def test_asking_for_milvus_explicitly_does_not_fall_back(fake_root, catalog, encoder,
                                                         artifacts, monkeypatch):
    def refuse(*args, **kwargs):
        raise MilvusUnavailable('Cannot reach Milvus')

    monkeypatch.setattr(milvus_store, 'open_store', refuse)
    with pytest.raises(store.ArtifactError, match='Cannot reach Milvus'):
        SearchEngine(artifacts_dir=artifacts, catalog_path=fake_root / 'catalog.csv',
                     encoder=encoder, backend='milvus')


def test_a_mismatched_collection_is_never_served_quietly(fake_root, catalog, encoder,
                                                         artifacts, monkeypatch):
    """A mismatch is a refusal, not a fallback: the results would be wrong."""
    def mismatch(*args, **kwargs):
        raise store.ArtifactError('collection does not match the running configuration')

    monkeypatch.setattr(milvus_store, 'open_store', mismatch)
    with pytest.raises(store.ArtifactError, match='does not match'):
        SearchEngine(artifacts_dir=artifacts, catalog_path=fake_root / 'catalog.csv',
                     encoder=encoder, backend='auto')


def test_an_unknown_backend_name_is_rejected(fake_root, artifacts, encoder):
    with pytest.raises(store.ArtifactError, match='Unknown backend'):
        SearchEngine(artifacts_dir=artifacts, catalog_path=fake_root / 'catalog.csv',
                     encoder=encoder, backend='sqlite')
