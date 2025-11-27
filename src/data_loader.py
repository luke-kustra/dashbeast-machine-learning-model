
import pandas as pd
import json
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split


def _compute_scaler(features):
    """Compute mean and std for features (per-column). Returns dict."""
    mean = features.mean(axis=0).tolist()
    std = features.std(axis=0).tolist()
    # Avoid zero std
    std = [s if s > 1e-6 else 1.0 for s in std]
    return {"mean": mean, "std": std}


def _apply_scaler(features, scaler):
    mean = torch.tensor(scaler["mean"], dtype=torch.float32)
    std = torch.tensor(scaler["std"], dtype=torch.float32)
    return (features - mean) / std


def get_dataloaders(file_path, batch_size=32, val_split=0.2, seed=42, provided_label_map=None):
    """
    Loads the workout data from a CSV file and returns train and validation DataLoaders,
    a label map, and a scaler (mean/std) computed from the training set.

    Args:
        file_path (str): The path to the CSV file.
        batch_size (int): The batch size for the DataLoader.
        val_split (float): Fraction of data to hold out for validation (0.0-1.0).
        seed (int): Random seed for deterministic splits.

    Returns:
        (train_loader, val_loader, label_map, scaler)
    """
    df = pd.read_csv(file_path)

    # Use the last column as the label column name
    label_col = df.columns[-1]

    # If a provided_label_map is given, use it (enforces consistent mapping). Otherwise build from data.
    if provided_label_map is not None:
        # provided_label_map may be a dict or a filepath
        if isinstance(provided_label_map, str):
            with open(provided_label_map, 'r') as f:
                provided_label_map = json.load(f)
        label_map = provided_label_map
        # Verify all labels in dataset exist in provided map
        missing = set(df[label_col].unique()) - set(label_map.keys())
        if missing:
            raise ValueError(f"Labels present in data but missing from provided label_map: {missing}")
    else:
        unique_labels = sorted(df[label_col].unique())
        label_map = {label: i for i, label in enumerate(unique_labels)}

    # Separate features and labels
    features = df.iloc[:, :-1].values.astype('float32')
    labels = df.iloc[:, -1].map(label_map).values.astype('int64')

    # Convert to PyTorch tensors
    features = torch.tensor(features, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)

    dataset = TensorDataset(features, labels)

    if val_split <= 0.0:
        train_dataset = dataset
        val_dataset = None
    else:
        total = len(dataset)
        val_size = int(total * val_split)
        train_size = total - val_size
        generator = torch.Generator().manual_seed(seed)
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    # Compute scaler on training features
    if val_split <= 0.0:
        train_features = torch.stack([x for x, _ in train_dataset])
    else:
        train_features = torch.stack([x for x, _ in train_dataset])

    scaler = _compute_scaler(train_features.numpy())

    # Apply scaler
    def _rescale_dataset(ds):
        feats = torch.stack([x for x, _ in ds])
        labs = torch.tensor([y for _, y in ds], dtype=torch.long)
        feats = _apply_scaler(feats, scaler)
        return TensorDataset(feats, labs)

    train_dataset = _rescale_dataset(train_dataset)
    val_loader = None
    if val_dataset is not None:
        val_dataset = _rescale_dataset(val_dataset)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader, label_map, scaler


def save_scaler(scaler, path):
    with open(path, 'w') as f:
        json.dump(scaler, f)


def load_scaler(path):
    with open(path, 'r') as f:
        return json.load(f)
