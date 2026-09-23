"""The unified vector table: row alignment, per-type coverage, manifest gating."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src import config, store
from src.store import ArtifactError
from tests.helpers import build_artifacts, unit


def test_both_modalities_live_in_one_matrix(artifacts):
    loaded = store.load(artifacts)
    assert set(loaded.rows['embedding_type'].unique()) == {'image', 'title'}
    assert len(loaded.vectors) == len(loaded.rows)
    # One table, not two files: image and title rows are interleaved in the same array.
    assert loaded.vectors.shape[1] == loaded.manifest['dim']


def test_row_order_is_the_vector_to_product_link(artifacts):
    loaded = store.load(artifacts)
    vectors = np.load(artifacts / store.VECTORS_FILE)
    rows = pd.read_parquet(artifacts / store.ROWS_FILE)
    assert len(rows) == len(vectors)
    for embedding_type, positions in loaded.type_index.items():
        for position in positions:
            product_id = int(rows['product_id'].iloc[position])
            assert loaded.row_of[embedding_type][product_id] == position


def test_every_product_has_exactly_one_row_per_type(artifacts, catalog):
    loaded = store.load(artifacts)
    counts = loaded.rows.groupby(['product_id', 'embedding_type']).size()
    assert counts.max() == 1
    assert loaded.counts()['image'] == len(catalog)


def test_blank_title_product_gets_no_title_row(artifacts, catalog):
    """A product with no usable title keeps its image row and is not given a fake one."""
    loaded = store.load(artifacts)
    blank = catalog.loc[catalog['productDisplayName'] == '', 'id'].tolist()
    assert blank, 'fixture should contain a blank-title product'
    for product_id in blank:
        assert product_id in loaded.row_of['image']
        assert product_id not in loaded.row_of['title']
    assert loaded.manifest['titles_skipped_blank'] == len(blank)


def test_all_stored_vectors_are_finite_and_unit_norm(artifacts):
    loaded = store.load(artifacts)
    assert np.isfinite(loaded.vectors).all()
    norms = np.linalg.norm(loaded.vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=store.UNIT_NORM_TOLERANCE)


def test_validate_rejects_misaligned_rows():
    vectors = np.vstack([unit([1, 0]), unit([0, 1])]).astype(np.float32)
    rows = pd.DataFrame([{'product_id': 1, 'embedding_type': 'image',
                          'name': 'a', 'image_path': 'x.jpg'}])
    with pytest.raises(ArtifactError, match='1 rows but the matrix has 2'):
        store.validate(vectors, rows)


def test_validate_rejects_non_unit_vectors():
    vectors = np.array([[3.0, 4.0]], dtype=np.float32)
    rows = pd.DataFrame([{'product_id': 1, 'embedding_type': 'image',
                          'name': 'a', 'image_path': 'x.jpg'}])
    with pytest.raises(ArtifactError, match='unit-norm'):
        store.validate(vectors, rows)


def test_validate_rejects_duplicate_product_type_pairs():
    vectors = np.vstack([unit([1, 0]), unit([0, 1])]).astype(np.float32)
    rows = pd.DataFrame([
        {'product_id': 1, 'embedding_type': 'image', 'name': 'a', 'image_path': 'x.jpg'},
        {'product_id': 1, 'embedding_type': 'image', 'name': 'a', 'image_path': 'x.jpg'},
    ])
    with pytest.raises(ArtifactError, match='more than one row'):
        store.validate(vectors, rows)


def test_load_refuses_a_different_model(artifacts, encoder):
    identity = encoder.identity.as_dict()
    identity['model_id'] = 'some/other-model'
    with pytest.raises(ArtifactError, match='model_id'):
        store.load(artifacts, identity_dict=identity)


def test_load_refuses_a_different_catalog(artifacts, catalog, encoder):
    other = catalog.iloc[:3].copy()
    with pytest.raises(ArtifactError, match='catalog_fingerprint'):
        store.load(artifacts, identity_dict=encoder.identity.as_dict(), catalog=other)


def test_load_accepts_the_matching_configuration(artifacts, catalog, encoder):
    loaded = store.load(artifacts, identity_dict=encoder.identity.as_dict(), catalog=catalog)
    assert loaded.counts()['image'] == len(catalog)


def test_missing_artifacts_give_a_build_instruction(tmp_path):
    with pytest.raises(ArtifactError, match='build_embeddings'):
        store.load(tmp_path / 'nope')


def test_publish_is_atomic_and_leaves_no_staging_dirs(fake_root, catalog, encoder):
    target = build_artifacts(fake_root, catalog, encoder)
    leftovers = [p for p in target.parent.iterdir() if p.name.startswith('.')]
    assert leftovers == []
    assert json.loads((target / store.MANIFEST_FILE).read_text())['schema_version'] == 1


def test_catalog_fingerprint_tracks_membership(catalog):
    first = store.catalog_fingerprint(catalog)
    assert first == store.catalog_fingerprint(catalog.copy())
    assert first != store.catalog_fingerprint(catalog.iloc[:-1])
