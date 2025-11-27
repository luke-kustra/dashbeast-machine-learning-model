import os
import json
import torch
from src.data_loader import get_dataloaders, save_scaler, load_scaler


def test_get_dataloaders_basic(tmp_path):
    # use the repo data file
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_path = os.path.join(repo_root, 'data', 'workouts.csv')

    train_loader, val_loader, label_map, scaler = get_dataloaders(data_path, batch_size=4, val_split=0.2)

    # basic assertions
    assert train_loader is not None
    assert isinstance(label_map, dict)
    assert 'mean' in scaler and 'std' in scaler

    # saved scaler loads back
    fpath = tmp_path / 'scaler.json'
    save_scaler(scaler, str(fpath))
    loaded = load_scaler(str(fpath))
    assert loaded['mean'] == scaler['mean']


def test_get_dataloaders_with_provided_label_map(tmp_path):
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_path = os.path.join(repo_root, 'data', 'workouts.csv')

    # build provided label map from existing labels
    _, _, label_map_default, _ = get_dataloaders(data_path, batch_size=4, val_split=0.2)
    provided_path = tmp_path / 'label_map.json'
    with open(provided_path, 'w') as f:
        json.dump(label_map_default, f)

    train_loader, val_loader, label_map, scaler = get_dataloaders(data_path, batch_size=4, val_split=0.2, provided_label_map=str(provided_path))
    assert label_map == label_map_default
