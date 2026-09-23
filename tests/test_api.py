"""API contract: response shape, error codes, and the not-ready path."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.store import ArtifactError
from tests.helpers import jpeg_bytes


@pytest.fixture
def client(engine, monkeypatch):
    import app as app_module

    app_module.app.state.engine_factory = lambda: engine
    with TestClient(app_module.app) as test_client:
        yield test_client


@pytest.fixture
def broken_client(monkeypatch):
    import app as app_module

    def explode():
        raise ArtifactError('No embedding artifacts. Build them with: python -m src.build_embeddings')

    app_module.app.state.engine_factory = explode
    with TestClient(app_module.app, raise_server_exceptions=False) as test_client:
        yield test_client


def test_health_reports_the_loaded_table(client):
    body = client.get('/health').json()
    assert body['ready'] is True
    assert set(body['row_counts']) == {'image', 'title'}
    assert 'weighted_sum' in body['strategies']


def test_text_search_response_shape(client):
    response = client.post('/search/text', json={'query': 'blue tshirt', 'top_k': 3})
    assert response.status_code == 200
    body = response.json()
    assert body['strategy'] == 'weighted_sum'
    assert 0 < len(body['results']) <= 3
    item = body['results'][0]
    assert set(item) >= {'rank', 'product_id', 'name', 'image_url', 'final_score', 'breakdown'}
    assert item['rank'] == 1
    assert item['image_url'] == f"/images/{item['product_id']}"
    assert 'took_ms' in body


def test_text_search_results_are_unique_products(client):
    body = client.post('/search/text', json={'query': 'shoes', 'top_k': 6}).json()
    ids = [r['product_id'] for r in body['results']]
    assert len(ids) == len(set(ids))


def test_blank_query_is_422(client):
    response = client.post('/search/text', json={'query': '   '})
    assert response.status_code == 422
    assert 'search phrase' in response.json()['detail']


def test_overlong_query_is_422(client):
    response = client.post('/search/text', json={'query': 'shoe ' * 200})
    assert response.status_code == 422
    assert 'tokens' in response.json()['detail']


def test_unknown_strategy_is_422(client):
    response = client.post('/search/text', json={'query': 'shoes', 'strategy': 'magic'})
    assert response.status_code == 422
    assert 'Unknown strategy' in response.json()['detail']


def test_top_k_out_of_range_is_422(client):
    assert client.post('/search/text', json={'query': 'x', 'top_k': 0}).status_code == 422
    assert client.post('/search/text', json={'query': 'x', 'top_k': 5000}).status_code == 422


def test_image_search_accepts_an_upload(client):
    response = client.post('/search/image',
                           files={'file': ('query.jpg', jpeg_bytes(), 'image/jpeg')},
                           params={'top_k': 4})
    assert response.status_code == 200
    body = response.json()
    assert body['query_type'] == 'image'
    assert 0 < len(body['results']) <= 4


def test_image_search_rejects_a_non_image(client):
    response = client.post('/search/image',
                           files={'file': ('bad.jpg', b'not an image', 'image/jpeg')})
    assert response.status_code == 422
    assert 'could not be read' in response.json()['detail']


def test_image_search_requires_a_file(client):
    assert client.post('/search/image').status_code == 422


def test_image_route_serves_a_catalog_thumbnail(client, catalog):
    product_id = int(catalog['id'].iloc[0])
    response = client.get(f'/images/{product_id}')
    assert response.status_code == 200
    assert response.headers['content-type'] == 'image/jpeg'


def test_image_route_404s_for_an_unknown_product(client):
    assert client.get('/images/999999').status_code == 404


def test_filters_endpoint_lists_catalog_values(client, catalog):
    options = client.get('/filters').json()
    assert set(options['articleType']) == set(catalog['articleType'].astype(str))


def test_not_ready_health_is_503_with_a_safe_reason(broken_client):
    response = broken_client.get('/health')
    assert response.status_code == 503
    body = response.json()
    assert body['ready'] is False
    assert 'build_embeddings' in body['reason']
    assert 'Traceback' not in body['reason']


def test_not_ready_search_is_503(broken_client):
    assert broken_client.post('/search/text', json={'query': 'shoes'}).status_code == 503
    response = broken_client.post(
        '/search/image', files={'file': ('q.jpg', jpeg_bytes(), 'image/jpeg')})
    assert response.status_code == 503
