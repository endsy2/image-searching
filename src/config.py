"""Shared paths and constants for the CLIP fashion search project.

The project root is detected by the presence of ``dataset/styles.csv``, so every
module resolves catalog-relative paths (``image_path``) the same way.
"""
from pathlib import Path


def find_root(start: Path | None = None) -> Path:
    """Walk upwards until the directory holding ``dataset/styles.csv`` is found."""
    here = (start or Path(__file__).resolve().parent)
    for candidate in [here, *here.parents]:
        if (candidate / 'dataset' / 'styles.csv').exists():
            return candidate
    # Fall back to the package parent so imports still work without the dataset.
    return Path(__file__).resolve().parent.parent


ROOT = find_root()

ARTIFACTS = ROOT / 'artifacts'
DATA_PREP = ARTIFACTS / 'data_preparation'
CATALOG_FULL = DATA_PREP / 'catalog_clean.csv'
CATALOG_SAMPLE = DATA_PREP / 'catalog_sample_500.csv'

EMBEDDINGS_FULL = ARTIFACTS / 'embeddings'
EMBEDDINGS_SAMPLE = ARTIFACTS / 'embeddings_sample'

EVALUATION = ROOT / 'evaluation'
STATIC = ROOT / 'static'
IMAGES_DIR = ROOT / 'dataset' / 'images'

# Model identity. The checkpoint name alone does not pin the preprocessing, so
# the loading package is recorded in the manifest alongside it.
MODEL_ID = 'openai/clip-vit-base-patch32'
MODEL_SOURCE = 'transformers.CLIPModel'
EMBED_DIM = 512

# CLIP's text context window, including the BOS/EOS sentinels.
TEXT_CONTEXT_LENGTH = 77
MAX_TEXT_TOKENS = TEXT_CONTEXT_LENGTH - 2  # usable content tokens

EMBEDDING_TYPES = ('image', 'title')

# Retrieval defaults.
DEFAULT_TOP_K = 10
MAX_TOP_K = 100
DEFAULT_CANDIDATE_K = 100
RRF_K = 60

# Upload limits for the API.
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_QUERY_CHARS = 500
