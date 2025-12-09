"""Gradient Boosting model for workout classification."""

import argparse
import json
import os
from typing import Dict, List, Tuple

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .sklearn_utils import load_dataset, DEFAULT_DATA_PATH

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DEFAULT_MODEL_PATH = os.path.join(ROOT_DIR, 'gradient_boosting_model.joblib')
DEFAULT_LABEL_MAP_PATH = os.path.join(ROOT_DIR, 'gboost_label_map.json')


def build_pipeline(seed: int) -> Pipeline:
    """Build a gradient boosting pipeline with standard scaling."""
    model = GradientBoostingClassifier(random_state=seed, n_estimators=200, learning_rate=0.05)
    return Pipeline([
        ('scaler', StandardScaler()),
        ('model', model),
    ])


def train(data_path: str = DEFAULT_DATA_PATH, label_map_path: str = DEFAULT_LABEL_MAP_PATH, model_path: str = DEFAULT_MODEL_PATH, val_split: float = 0.2, seed: int = 42) -> Dict[str, float]:
    """Train gradient boosting model on workout data."""
    X, y, label_map = load_dataset(data_path, label_map_path)

    if val_split > 0:
        try:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=val_split, random_state=seed, stratify=y
            )
        except ValueError:
            # Fall back to unstratified split for small datasets
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=val_split, random_state=seed, stratify=None
            )
    else:
        X_train, X_val, y_train, y_val = X, None, y, None

    pipeline = build_pipeline(seed)
    pipeline.fit(X_train, y_train)

    metrics = {}
    if X_val is not None and len(X_val) > 0:
        val_probs = pipeline.predict_proba(X_val)
        val_preds = pipeline.predict(X_val)
        metrics['val_accuracy'] = float(accuracy_score(y_val, val_preds))
        try:
            metrics['val_log_loss'] = float(log_loss(y_val, val_probs))
        except ValueError:
            metrics['val_log_loss'] = float(log_loss(y_val, val_probs, labels=list(range(val_probs.shape[1]))))

    joblib.dump(pipeline, model_path)
    with open(label_map_path, 'w') as f:
        json.dump(label_map, f)

    metrics['model_path'] = model_path
    metrics['label_map_path'] = label_map_path
    return metrics


def predict(features: List[float], model_path: str, label_map_path: str = DEFAULT_LABEL_MAP_PATH, topk: int = 3) -> List[Tuple[str, float]]:
    """Predict workout class from accelerometer/gyroscope features."""
    pipeline = joblib.load(model_path)

    inv_label_map = None
    try:
        with open(label_map_path, 'r') as f:
            label_map = json.load(f)
            inv_label_map = {v: k for k, v in label_map.items()}
    except Exception:
        pass

    probs = pipeline.predict_proba(np.array([features]))[0]
    topk = min(topk, len(probs))
    indices = np.argsort(probs)[::-1][:topk]

    results = []
    for idx in indices:
        label = inv_label_map[idx] if inv_label_map is not None and idx in inv_label_map else f'class_{idx}'
        results.append((label, float(probs[idx])))
    return results


def _print_metrics(metrics: Dict[str, float]) -> None:
    """Pretty-print training metrics."""
    printable = {k: v for k, v in metrics.items() if k not in {'model_path', 'label_map_path'}}
    print(json.dumps(printable, indent=2))
    print(f"Model saved to: {metrics.get('model_path')}")
    print(f"Label map saved to: {metrics.get('label_map_path')}")


def main():
    parser = argparse.ArgumentParser(description='Train or predict with gradient boosting for workout classification')
    subparsers = parser.add_subparsers(dest='command', required=True)

    train_parser = subparsers.add_parser('train', help='Train gradient boosting model')
    train_parser.add_argument('--data', default=DEFAULT_DATA_PATH)
    train_parser.add_argument('--label-map', dest='label_map', default=DEFAULT_LABEL_MAP_PATH)
    train_parser.add_argument('--model-path', dest='model_path', default=DEFAULT_MODEL_PATH)
    train_parser.add_argument('--val-split', dest='val_split', type=float, default=0.2)
    train_parser.add_argument('--seed', type=int, default=42)

    predict_parser = subparsers.add_parser('predict', help='Predict with trained gradient boosting model')
    predict_parser.add_argument('--features', nargs=6, type=float, help='Six numeric features', required=False)
    predict_parser.add_argument('--model-path', dest='model_path', default=DEFAULT_MODEL_PATH)
    predict_parser.add_argument('--label-map', dest='label_map', default=DEFAULT_LABEL_MAP_PATH)
    predict_parser.add_argument('--topk', type=int, default=3)

    args = parser.parse_args()

    if args.command == 'train':
        metrics = train(
            data_path=args.data,
            label_map_path=args.label_map,
            model_path=args.model_path,
            val_split=args.val_split,
            seed=args.seed,
        )
        _print_metrics(metrics)
    elif args.command == 'predict':
        sample = args.features if args.features is not None else [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        results = predict(sample, model_path=args.model_path, label_map_path=args.label_map, topk=args.topk)
        for label, prob in results:
            print(f"{label}: {prob:.4f}")


if __name__ == '__main__':
    main()
