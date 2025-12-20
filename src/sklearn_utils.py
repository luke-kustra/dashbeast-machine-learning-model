"""Shared utilities for scikit-learn ML models."""

import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DEFAULT_DATA_PATH = os.path.join(ROOT_DIR, 'data', 'workouts.csv')


def _resolve_label_map(raw_labels: List[str], label_map_path: str) -> Dict[str, int]:
    """Use existing label map if it covers the labels; otherwise build a new one."""
    labels_unique = sorted(set(raw_labels))
    if os.path.exists(label_map_path):
        try:
            with open(label_map_path, 'r') as f:
                existing = json.load(f)
            if set(existing.keys()) == set(labels_unique):
                return existing
        except Exception:
            pass
    return {label: idx for idx, label in enumerate(labels_unique)}


def load_dataset(file_path: str = DEFAULT_DATA_PATH, label_map_path: str = None) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """Load and parse workout data from CSV.

    Schema-aware behavior (mirrors src.data_loader):
    - If columns Ax, Ay, Az, Gx, Gy, Gz exist, they are used as features in that order.
    - If ExerciseName exists, it is used as the label.
    - Otherwise, defaults to: last column is label; all other columns are features.
    """
    df = pd.read_csv(file_path)

    # Determine label column
    if 'ExerciseName' in df.columns:
        label_col = 'ExerciseName'
    else:
        label_col = df.columns[-1]

    # Determine feature columns
    preferred_feature_cols = ['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz']
    if all(col in df.columns for col in preferred_feature_cols):
        feature_cols = preferred_feature_cols
    else:
        feature_cols = [c for c in df.columns if c != label_col]

    raw_labels = df[label_col].tolist()
    if label_map_path:
        label_map = _resolve_label_map(raw_labels, label_map_path)
    else:
        label_map = {label: idx for idx, label in enumerate(sorted(set(raw_labels)))}

    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = np.array([label_map[label] for label in raw_labels], dtype=np.int64)
    return X, y, label_map
