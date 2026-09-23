"""Fixtures that relocate the project root into a tmp dir with a synthetic catalog."""
from __future__ import annotations

from pathlib import Path

import pytest

from src import config
from tests.helpers import FakeEncoder, build_artifacts, make_catalog


@pytest.fixture
def fake_root(tmp_path, monkeypatch) -> Path:
    """A throwaway project root; config paths point inside it for the test."""
    monkeypatch.setattr(config, 'ROOT', tmp_path)
    monkeypatch.setattr(config, 'IMAGES_DIR', tmp_path / 'dataset' / 'images')
    return tmp_path


@pytest.fixture
def encoder() -> FakeEncoder:
    return FakeEncoder()


@pytest.fixture
def catalog(fake_root):
    return make_catalog(fake_root)


@pytest.fixture
def artifacts(fake_root, catalog, encoder) -> Path:
    return build_artifacts(fake_root, catalog, encoder)


@pytest.fixture(autouse=True)
def _default_backend(monkeypatch):
    """Unit tests run against the in-process NumPy table: no Docker, no Milvus.

    tests/test_milvus.py exercises the Milvus backend with a fake client.
    """
    monkeypatch.setenv('FASHION_BACKEND', 'numpy')


@pytest.fixture
def engine(fake_root, catalog, artifacts, encoder):
    from src.search import SearchEngine

    return SearchEngine(artifacts_dir=artifacts, catalog_path=fake_root / 'catalog.csv',
                        encoder=encoder, backend='numpy')
