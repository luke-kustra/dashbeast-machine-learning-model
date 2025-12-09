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
    """Load and parse workout data from CSV."""
    df = pd.read_csv(file_path)
    label_col = df.columns[-1]
    raw_labels = df[label_col].tolist()
    label_map = _resolve_label_map(raw_labels, label_map_path) if label_map_path else {label: idx for idx, label in enumerate(sorted(set(raw_labels)))}
    X = df.iloc[:, :-1].to_numpy(dtype=np.float32)
    y = np.array([label_map[label] for label in raw_labels], dtype=np.int64)
    return X, y, label_map
