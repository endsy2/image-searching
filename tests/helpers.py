"""Synthetic catalogs, fake encoders, and built artifacts for the test suite.

Nothing here touches the CLIP weights, the real dataset, or the network.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src import store
from src.encoder import EncoderIdentity, QueryValidationError

DIM = 8
MAX_TOKENS = 75


def unit(values) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


class FakeEncoder:
    """Deterministic stand-in for ClipEncoder with the same public surface."""

    def __init__(self, dim: int = DIM, device: str = 'cpu'):
        self.device = device
        self.model_id = 'fake/clip'
        self.identity = EncoderIdentity(
            model_id='fake/clip',
            model_source='tests.helpers.FakeEncoder',
            dim=dim,
            image_size=224,
            image_mean=(0.5, 0.5, 0.5),
            image_std=(0.5, 0.5, 0.5),
            context_length=77,
        )

    # Hash text/images into a stable direction so rankings are reproducible.
    def _vector(self, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return unit(rng.normal(size=self.identity.dim))

    def count_tokens(self, text: str) -> int:
        return len(str(text).split())

    def validate_text(self, text: str) -> str:
        if text is None:
            raise QueryValidationError('Enter a search phrase.')
        cleaned = ' '.join(str(text).split())
        if not cleaned:
            raise QueryValidationError('Enter a search phrase.')
        if self.count_tokens(cleaned) > MAX_TOKENS:
            raise QueryValidationError(
                f'Query is {self.count_tokens(cleaned)} tokens; CLIP accepts at most '
                f'{MAX_TOKENS}.'
            )
        return cleaned

    def encode_texts(self, texts):
        if not len(texts):
            return np.zeros((0, self.identity.dim), dtype=np.float32)
        return np.vstack([self._vector(abs(hash(t)) % (2 ** 31)) for t in texts]).astype(np.float32)

    def encode_images(self, images):
        if not len(images):
            return np.zeros((0, self.identity.dim), dtype=np.float32)
        return np.vstack([
            self._vector(int(np.asarray(im.convert('L')).sum()) % (2 ** 31)) for im in images
        ]).astype(np.float32)


def make_catalog(root: Path, n: int = 6, blank_title_ids=(5,),
                 duplicate_pair=(1, 2)) -> pd.DataFrame:
    """Write n tiny jpgs under root/dataset/images and return a catalog frame."""
    images_dir = root / 'dataset' / 'images'
    images_dir.mkdir(parents=True, exist_ok=True)
    (root / 'dataset' / 'styles.csv').write_text('id\n', encoding='utf-8')

    colours = ['Blue', 'Black', 'Red', 'White', 'Green', 'Navy Blue']
    types = ['Tshirts', 'Shoes', 'Watches', 'Handbags', 'Jeans', 'Shirts']
    records = []
    for index in range(n):
        pid = index + 1
        Image.new('RGB', (60, 80), (index * 30 % 256, 80, 160)).save(images_dir / f'{pid}.jpg')
        title = '' if pid in blank_title_ids else f'Brand {types[index % len(types)]} {colours[index % len(colours)]}'
        records.append({
            'id': pid,
            'gender': ['Men', 'Women'][index % 2],
            'masterCategory': 'Apparel',
            'subCategory': 'Topwear',
            'articleType': types[index % len(types)],
            'baseColour': colours[index % len(colours)],
            'season': 'Summer',
            'year': 2012,
            'usage': 'Casual',
            'productDisplayName': title,
            'image_path': f'dataset/images/{pid}.jpg',
            'image_duplicate_group': (
                'shared' if pid in duplicate_pair else f'hash-{pid}'
            ),
            'search_text': f'{title} {types[index % len(types)]} {colours[index % len(colours)]}',
        })
    catalog = pd.DataFrame(records)
    catalog.to_csv(root / 'catalog.csv', index=False)
    return catalog


def build_artifacts(root: Path, catalog: pd.DataFrame, encoder: FakeEncoder,
                    out: Path | None = None) -> Path:
    """Run the real build path against the fake encoder and publish artifacts."""
    from src.build_embeddings import build, build_manifest

    prepared = catalog.copy()
    prepared['title'] = prepared['productDisplayName'].fillna('').astype(str).str.strip()
    # build() resolves image paths against config.ROOT, so callers must patch it.
    vectors, rows, report = build(prepared, encoder, batch_size=4)
    manifest = build_manifest(prepared, encoder, rows, 4, report, root / 'catalog.csv')
    target = out or (root / 'artifacts' / 'embeddings')
    return store.publish(target, vectors, rows, manifest,
                         expected_products=report['products_encoded'])


def jpeg_bytes(size=(60, 80), colour=(20, 90, 200)) -> bytes:
    import io

    buffer = io.BytesIO()
    Image.new('RGB', size, colour).save(buffer, format='JPEG')
    return buffer.getvalue()
