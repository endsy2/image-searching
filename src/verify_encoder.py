"""Step 2 of the plan: verify the frozen encoder on the 500-product sample.

Checks that both encoders run on the available device, that their output is
512-d, finite and unit-norm, that a product's own image and title vectors are
closer to each other than to a random product's, and benchmarks a batch so the
full-catalog build time can be estimated.

    python -m src.verify_encoder [--limit 200] [--batch-size 32]
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

from . import config
from .build_embeddings import load_catalog
from .encoder import ClipEncoder, QueryValidationError, load_image

SANITY_QUERIES = [
    'a navy blue shirt',
    'black casual shoes',
    'a silver watch',
    'a red handbag',
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--device', default=None)
    args = parser.parse_args(argv)

    catalog = load_catalog(config.CATALOG_SAMPLE).head(args.limit).reset_index(drop=True)
    catalog = catalog[catalog['title'] != ''].reset_index(drop=True)
    print(f'Sample  : {len(catalog)} products from {config.CATALOG_SAMPLE.name}')

    started = time.perf_counter()
    encoder = ClipEncoder(device=args.device)
    load_seconds = time.perf_counter() - started
    print(f'Encoder : {encoder.model_id} on {encoder.device} '
          f'(cold load {load_seconds:.1f}s, dim {encoder.identity.dim})')

    # --- encode ---------------------------------------------------------
    images, titles = [], catalog['title'].tolist()
    for path in catalog['image_path']:
        images.append(load_image(config.ROOT / path))

    started = time.perf_counter()
    image_vectors = np.vstack([
        encoder.encode_images(images[i:i + args.batch_size])
        for i in range(0, len(images), args.batch_size)
    ])
    image_seconds = time.perf_counter() - started

    started = time.perf_counter()
    title_vectors = np.vstack([
        encoder.encode_texts(titles[i:i + args.batch_size])
        for i in range(0, len(titles), args.batch_size)
    ])
    title_seconds = time.perf_counter() - started

    # --- structural checks ----------------------------------------------
    checks: list[tuple[str, bool, str]] = []
    for name, matrix in (('image', image_vectors), ('title', title_vectors)):
        checks.append((f'{name}: dim == {config.EMBED_DIM}',
                       matrix.shape[1] == config.EMBED_DIM, str(matrix.shape)))
        checks.append((f'{name}: all finite', bool(np.isfinite(matrix).all()), ''))
        norms = np.linalg.norm(matrix, axis=1)
        worst = float(np.max(np.abs(norms - 1.0)))
        checks.append((f'{name}: unit-norm within 1e-5', worst < 1e-5, f'worst {worst:.2e}'))

    # --- cross-modal alignment -------------------------------------------
    # A product's own title should be closer to its own image than to a random
    # product's image. This is the cheapest end-to-end sanity check there is.
    self_similarity = float(np.mean(np.sum(image_vectors * title_vectors, axis=1)))
    rng = np.random.default_rng(42)
    shuffled = rng.permutation(len(catalog))
    random_similarity = float(np.mean(np.sum(image_vectors * title_vectors[shuffled], axis=1)))
    matched = np.sum(image_vectors * title_vectors, axis=1)
    mismatched = np.sum(image_vectors * title_vectors[shuffled], axis=1)
    win_rate = float(np.mean(matched > mismatched))
    checks.append(('own title beats a random title on average',
                   self_similarity > random_similarity,
                   f'self {self_similarity:.4f} vs random {random_similarity:.4f}'))
    checks.append(('own title wins per-product in >80% of cases', win_rate > 0.80,
                   f'{win_rate:.1%}'))

    # --- the modality gap, measured ---------------------------------------
    query_vectors = encoder.encode_texts(SANITY_QUERIES)
    to_images = query_vectors @ image_vectors.T
    to_titles = query_vectors @ title_vectors.T
    gap = {
        'text_query_vs_image_vectors_mean': round(float(to_images.mean()), 4),
        'text_query_vs_title_vectors_mean': round(float(to_titles.mean()), 4),
        'text_query_vs_image_vectors_std': round(float(to_images.std()), 4),
        'text_query_vs_title_vectors_std': round(float(to_titles.std()), 4),
    }

    # --- validation behaviour ---------------------------------------------
    for label, value in (('blank query', '   '), ('over-long query', 'shoe ' * 100)):
        try:
            encoder.validate_text(value)
            checks.append((f'{label} rejected', False, 'accepted!'))
        except QueryValidationError as error:
            checks.append((f'{label} rejected', True, str(error)[:60]))

    # --- report -----------------------------------------------------------
    print('\nChecks')
    all_passed = True
    for name, passed, detail in checks:
        all_passed &= passed
        print(f'  [{"ok" if passed else "FAIL"}] {name}' + (f'  ({detail})' if detail else ''))

    rate = len(images) / image_seconds
    estimate = 44441 / rate
    print(f'\nThroughput: {rate:.1f} images/s, {len(titles) / title_seconds:.1f} titles/s '
          f'(batch {args.batch_size}, {encoder.device})')
    print(f'Estimated full-catalog image encode: {estimate / 60:.1f} min for 44,441 products')
    print(f'\nModality gap (mean cosine of a text query against each modality):')
    print('  ' + json.dumps(gap))

    print('\nSanity queries (top 3 by each modality separately):')
    for index, query in enumerate(SANITY_QUERIES):
        print(f'  "{query}"')
        for modality, scores in (('image', to_images[index]), ('title', to_titles[index])):
            top = np.argsort(-scores)[:3]
            names = ' | '.join(f'{catalog["title"].iloc[t][:38]} ({scores[t]:.3f})' for t in top)
            print(f'    {modality:<5} {names}')

    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
