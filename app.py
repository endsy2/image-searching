"""FastAPI backend for multimodal fashion search.

The encoder and the unified vector table load once, at startup, in a worker
thread; requests never reload them. If the artifacts are missing or mismatched
the app still starts, ``/health`` reports ``ready: false`` with a user-safe
reason, and the search endpoints return 503.

    python -m uvicorn app:app --reload
"""
from __future__ import annotations

import asyncio
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src import config, fusion
from src.encoder import QueryValidationError
from src.search import SearchEngine, open_upload
from src.store import ArtifactError

ARTIFACTS_DIR = Path(os.environ.get('FASHION_ARTIFACTS', config.EMBEDDINGS_FULL))
CATALOG_PATH = Path(os.environ.get('FASHION_CATALOG', config.CATALOG_FULL))
DEVICE = os.environ.get('FASHION_DEVICE') or None


def default_engine_factory() -> SearchEngine:
    return SearchEngine(artifacts_dir=ARTIFACTS_DIR, catalog_path=CATALOG_PATH, device=DEVICE)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.engine = None
    app.state.error = None
    app.state.startup_seconds = None
    started = time.perf_counter()
    try:
        app.state.engine = await asyncio.to_thread(app.state.engine_factory)
        app.state.startup_seconds = round(time.perf_counter() - started, 2)
    except ArtifactError as error:
        app.state.error = str(error)
    except Exception as error:  # noqa: BLE001 - surfaced as a safe message
        app.state.error = f'{type(error).__name__}: {error}'
    yield


app = FastAPI(title='Fashion Product Search (CLIP)', version='1.0', lifespan=lifespan)
app.state.engine_factory = default_engine_factory


def require_engine(request: Request) -> SearchEngine:
    engine = getattr(request.app.state, 'engine', None)
    if engine is None:
        raise HTTPException(
            status_code=503,
            detail=request.app.state.error or 'Search artifacts are not loaded yet.',
        )
    return engine


@app.exception_handler(QueryValidationError)
async def _validation_handler(request: Request, error: QueryValidationError):
    return JSONResponse(status_code=422, content={'detail': str(error)})


class TextSearchRequest(BaseModel):
    query: str = Field(..., description='Free-text description of the product')
    top_k: int = Field(config.DEFAULT_TOP_K, ge=1, le=config.MAX_TOP_K)
    strategy: str = Field('weighted_sum')
    image_weight: float | None = Field(None, ge=0.0, le=1.0)
    filters: dict[str, str] | None = None


def _weights(image_weight: float | None) -> dict[str, float] | None:
    if image_weight is None:
        return None
    return {'image': image_weight, 'title': 1.0 - image_weight}


def _check_strategy(strategy: str) -> str:
    if strategy not in fusion.STRATEGIES:
        raise QueryValidationError(
            f'Unknown strategy {strategy!r}. Choose from: {", ".join(fusion.STRATEGIES)}.'
        )
    return strategy


@app.post('/search/text')
async def search_text(payload: TextSearchRequest, request: Request):
    engine = require_engine(request)
    _check_strategy(payload.strategy)
    return await asyncio.to_thread(
        engine.search_text,
        payload.query,
        top_k=payload.top_k,
        strategy=payload.strategy,
        weights=_weights(payload.image_weight),
        filters=payload.filters,
    )


@app.post('/search/image')
async def search_image(
    request: Request,
    file: UploadFile = File(...),
    top_k: int = config.DEFAULT_TOP_K,
    strategy: str = 'weighted_sum',
    image_weight: float | None = None,
    exclude_product_id: int | None = None,
):
    engine = require_engine(request)
    _check_strategy(strategy)
    data = await file.read()
    image = open_upload(data)
    return await asyncio.to_thread(
        engine.search_image,
        image,
        top_k=top_k,
        strategy=strategy,
        weights=_weights(image_weight),
        exclude_product_id=exclude_product_id,
    )


@app.get('/images/{product_id}')
def product_image(product_id: int, request: Request):
    engine = require_engine(request)
    try:
        path = engine.image_path_for(product_id)
    except KeyError:
        raise HTTPException(status_code=404, detail='No such product.') from None
    if not path.exists():
        raise HTTPException(status_code=404, detail='Image file is missing.')
    return FileResponse(path, media_type='image/jpeg',
                        headers={'Cache-Control': 'public, max-age=86400'})


@app.get('/filters')
def filters(request: Request):
    return require_engine(request).filter_options()


@app.get('/health')
def health(request: Request):
    engine = getattr(request.app.state, 'engine', None)
    if engine is None:
        return JSONResponse(status_code=503, content={
            'ready': False,
            'reason': request.app.state.error or 'Artifacts not loaded.',
        })
    info = engine.check_ready()
    info['startup_seconds'] = request.app.state.startup_seconds
    info['strategies'] = list(fusion.STRATEGIES)
    return info


if config.STATIC.exists():
    app.mount('/', StaticFiles(directory=config.STATIC, html=True), name='static')
