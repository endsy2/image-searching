"""Frozen CLIP encoder.

Owns model loading, device selection, and the two encode entry points. Both
return L2-normalized ``float32`` vectors, so cosine similarity is a plain dot
product everywhere downstream.

The preprocessing transform and tokenizer ship with the checkpoint and are never
hand-rolled: ``CLIPProcessor`` supplies both.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image, ImageOps

from . import config


class QueryValidationError(ValueError):
    """Raised for user input that cannot be encoded. Message is user-safe."""


@dataclass(frozen=True)
class EncoderIdentity:
    """Everything that must match between a built artifact and a running encoder."""

    model_id: str
    model_source: str
    dim: int
    image_size: int
    image_mean: tuple[float, ...]
    image_std: tuple[float, ...]
    context_length: int

    def as_dict(self) -> dict:
        return {
            'model_id': self.model_id,
            'model_source': self.model_source,
            'dim': self.dim,
            'image_size': self.image_size,
            'image_mean': list(self.image_mean),
            'image_std': list(self.image_std),
            'context_length': self.context_length,
        }


def select_device(requested: str | None = None) -> str:
    import torch

    if requested:
        return requested
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def load_image(path: str | Path) -> Image.Image:
    """Load one catalog image exactly the way the data audit validated it."""
    with Image.open(path) as handle:
        return ImageOps.exif_transpose(handle).convert('RGB')


class ClipEncoder:
    """A loaded CLIP model plus its processor, in eval mode, no gradients."""

    def __init__(self, model_id: str = config.MODEL_ID, device: str | None = None,
                 use_safetensors: bool = False):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.torch = torch
        self.model_id = model_id
        self.device = select_device(device)
        # use_safetensors=False keeps the load working from the cached .bin weights;
        # use_fast=False keeps preprocessing on PIL/NumPy so torchvision is not required.
        self.model = CLIPModel.from_pretrained(
            model_id, use_safetensors=use_safetensors
        ).to(self.device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.processor = CLIPProcessor.from_pretrained(model_id, use_fast=False)
        self.tokenizer = self.processor.tokenizer
        self.image_processor = self.processor.image_processor

        crop = self.image_processor.crop_size
        self.identity = EncoderIdentity(
            model_id=model_id,
            model_source=config.MODEL_SOURCE,
            dim=int(self.model.config.projection_dim),
            image_size=int(crop['height']),
            image_mean=tuple(float(v) for v in self.image_processor.image_mean),
            image_std=tuple(float(v) for v in self.image_processor.image_std),
            context_length=int(self.model.config.text_config.max_position_embeddings),
        )

    # -- text ------------------------------------------------------------

    def count_tokens(self, text: str) -> int:
        """Content tokens, excluding the BOS/EOS sentinels the tokenizer adds."""
        ids = self.tokenizer(text, add_special_tokens=False)['input_ids']
        return len(ids)

    def validate_text(self, text: str) -> str:
        """Normalize and range-check a user query. Raises QueryValidationError."""
        if text is None:
            raise QueryValidationError('Enter a search phrase.')
        cleaned = ' '.join(str(text).split())
        if not cleaned:
            raise QueryValidationError('Enter a search phrase.')
        if len(cleaned) > config.MAX_QUERY_CHARS:
            raise QueryValidationError(
                f'Query is too long ({len(cleaned)} characters). '
                f'Keep it under {config.MAX_QUERY_CHARS}.'
            )
        tokens = self.count_tokens(cleaned)
        if tokens > config.MAX_TEXT_TOKENS:
            raise QueryValidationError(
                f'Query is {tokens} tokens; CLIP accepts at most '
                f'{config.MAX_TEXT_TOKENS}. Shorten it to a few descriptive words.'
            )
        return cleaned

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Encode already-validated strings. Returns (B, dim) unit-norm float32.

        Catalog titles are truncated at the context length rather than rejected;
        user queries are validated separately by ``validate_text`` so that an
        over-long query is an explicit error instead of a silent cut-off.
        """
        if len(texts) == 0:
            return np.zeros((0, self.identity.dim), dtype=np.float32)
        batch = self.tokenizer(
            list(texts),
            padding='max_length',
            max_length=self.identity.context_length,
            truncation=True,
            return_tensors='pt',
        ).to(self.device)
        with self.torch.no_grad():
            features = self.model.get_text_features(**batch)
        return _normalize(features.cpu().numpy())

    # -- images ----------------------------------------------------------

    def encode_images(self, images: Sequence[Image.Image]) -> np.ndarray:
        """Encode PIL images. Returns (B, dim) unit-norm float32."""
        if len(images) == 0:
            return np.zeros((0, self.identity.dim), dtype=np.float32)
        batch = self.image_processor(images=list(images), return_tensors='pt').to(self.device)
        with self.torch.no_grad():
            features = self.model.get_image_features(**batch)
        return _normalize(features.cpu().numpy())

    def encode_image_paths(self, paths: Iterable[str | Path]) -> np.ndarray:
        return self.encode_images([load_image(p) for p in paths])


def _normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    # A zero vector cannot be normalized; leave it and let validation reject it.
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


@lru_cache(maxsize=2)
def get_encoder(model_id: str = config.MODEL_ID, device: str | None = None) -> ClipEncoder:
    """Process-wide cached encoder so the weights load once."""
    return ClipEncoder(model_id=model_id, device=device)
