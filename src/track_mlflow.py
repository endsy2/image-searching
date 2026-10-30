"""Log the project's measured results to MLflow.

There is no training step (frozen CLIP ViT-B/32), so the "experiment" is the
embedding build and its evaluation: parameters come from the embedding manifest,
metrics from the manifest timings and evaluation/{timings,measurements}.json.
Nothing is recomputed or invented - this only records what is already on disk.

    python -m src.track_mlflow                         # ./mlruns (file store)
    python -m src.track_mlflow --uri http://127.0.0.1:5000
    mlflow ui                                          # browse the runs

MLFLOW_TRACKING_URI is honoured when --uri is not given.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import config

EXPERIMENT = 'fashion-clip-search'


def _flat(prefix: str, value, out: dict) -> None:
    """Flatten nested dicts into numeric metrics, skipping non-numbers."""
    if isinstance(value, dict):
        for key, inner in value.items():
            _flat(f'{prefix}.{key}' if prefix else str(key), inner, out)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        out[prefix] = float(value)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--uri', default=None, help='MLflow tracking server (default: ./mlruns)')
    parser.add_argument('--artifacts', type=Path, default=config.EMBEDDINGS_FULL)
    parser.add_argument('--run-name', default=None)
    args = parser.parse_args(argv)

    import mlflow

    manifest = _load(args.artifacts / 'manifest.json')
    if not manifest:
        print(f'No manifest in {args.artifacts}; build embeddings first.', file=sys.stderr)
        return 1
    eval_dir = config.ROOT / 'evaluation'
    timings = _load(eval_dir / 'timings.json')
    measurements = _load(eval_dir / 'measurements.json')

    if args.uri:
        mlflow.set_tracking_uri(args.uri)
    mlflow.set_experiment(EXPERIMENT)

    encoder = manifest.get('encoder', {})
    env = manifest.get('environment', {})
    params = {
        'model_id': encoder.get('model_id'),
        'embedding_dim': manifest.get('dim'),
        'batch_size': manifest.get('batch_size'),
        'device': manifest.get('device'),
        'catalog_products': manifest.get('catalog_products'),
        'catalog_fingerprint': manifest.get('catalog_fingerprint', '')[:16],
        'image_size': encoder.get('image_size'),
        'context_length': encoder.get('context_length'),
        'torch': env.get('torch'),
        'transformers': env.get('transformers'),
        'training': 'none (frozen pretrained CLIP)',
    }

    metrics: dict[str, float] = {}
    _flat('index', manifest.get('timings', {}), metrics)
    _flat('rows', manifest.get('row_counts', {}), metrics)
    metrics['failure_count'] = float(manifest.get('failure_count', 0))
    metrics['cold_startup_seconds'] = float(timings.get('cold_startup_seconds', 0)) or 0.0
    _flat('warm_text_query_ms', timings.get('warm_text_query_ms', {}), metrics)
    _flat('warm_image_query_ms', timings.get('warm_image_query_ms', {}), metrics)
    _flat('modality_gap', measurements.get('modality_gap', {}), metrics)
    if 'spread_ratio' in measurements:
        metrics['spread_ratio'] = float(measurements['spread_ratio'])
    if not metrics['cold_startup_seconds']:
        del metrics['cold_startup_seconds']

    with mlflow.start_run(run_name=args.run_name or f"clip-{manifest.get('built_at', 'build')}"):
        mlflow.log_params({k: v for k, v in params.items() if v is not None})
        mlflow.log_metrics(metrics)
        for path in (args.artifacts / 'manifest.json', eval_dir / 'timings.json',
                     eval_dir / 'measurements.json', eval_dir / 'report.md'):
            if path.exists():
                mlflow.log_artifact(str(path))
        run = mlflow.active_run().info.run_id
    print(f'Logged run {run}: {len(params)} params, {len(metrics)} metrics '
          f'to {mlflow.get_tracking_uri()}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
