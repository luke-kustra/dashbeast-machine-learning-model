import argparse
import json
import torch
import torch.nn.functional as F
try:
    # When run as a package (python -m src.predict)
    from .model import WorkoutClassifier
    from .data_loader import load_scaler
except Exception:
    # Fallback for running as a script (python src/predict.py)
    from model import WorkoutClassifier
    from data_loader import load_scaler


def load_label_map(path):
    with open(path, 'r') as f:
        return json.load(f)


def apply_scaler_to_tensor(tensor, scaler):
    mean = torch.tensor(scaler['mean'], dtype=torch.float32)
    std = torch.tensor(scaler['std'], dtype=torch.float32)
    return (tensor - mean) / std


def predict(input_list, model_path='C:/Users/luke0/dashbeast/workout_model.pth', label_map_path='C:/Users/luke0/dashbeast/label_map.json', scaler_path='C:/Users/luke0/dashbeast/scaler.json', topk=3):
    # Load state_dict first to infer the number of output classes
    raw = torch.load(model_path, map_location='cpu')
    if isinstance(raw, dict) and 'model_state_dict' in raw:
        state_dict = raw['model_state_dict']
    else:
        state_dict = raw

    # infer num_classes from the output layer weight if available
    if 'output_layer.weight' in state_dict:
        num_classes = state_dict['output_layer.weight'].shape[0]
    else:
        # fallback to label_map length if weight not found
        try:
            label_map_tmp = load_label_map(label_map_path)
            num_classes = len(label_map_tmp)
        except Exception:
            num_classes = None

    if num_classes is None:
        raise RuntimeError('Could not infer number of classes from checkpoint or label_map')

    model = WorkoutClassifier(num_features=6, num_classes=num_classes)
    model.load_state_dict(state_dict)
    model.eval()

    # If checkpoint contains a label_map, prefer it
    ckpt_label_map = None
    if isinstance(raw, dict) and 'label_map' in raw:
        ckpt_label_map = raw['label_map']

    # Load external label_map if present
    try:
        external_label_map = load_label_map(label_map_path)
    except Exception:
        external_label_map = None

    # Decide which label_map to use: checkpoint -> external -> fallback
    chosen_label_map = ckpt_label_map or external_label_map
    if chosen_label_map is None or len(chosen_label_map) != num_classes:
        inv_label_map = {i: f'class_{i}' for i in range(num_classes)}
    else:
        inv_label_map = {v: k for k, v in chosen_label_map.items()}

    scaler = None
    try:
        scaler = load_scaler(scaler_path)
    except Exception:
        pass

    tensor = torch.tensor([input_list], dtype=torch.float32)
    if scaler is not None:
        tensor = apply_scaler_to_tensor(tensor, scaler)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0)

    topk = min(topk, probs.size(0))
    values, indices = torch.topk(probs, topk)
    results = [(inv_label_map[int(idx)], float(val)) for idx, val in zip(indices.tolist(), values.tolist())]
    return results


def main():
    parser = argparse.ArgumentParser(description='Predict workout from 6 features')
    parser.add_argument('--features', nargs=6, type=float, help='Six numeric features', required=False)
    parser.add_argument('--topk', type=int, default=1, help='Show top K predictions')
    args = parser.parse_args()

    if args.features is None:
        # default sample
        sample = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    else:
        sample = args.features

    results = predict(sample, topk=args.topk)
    for label, prob in results:
        print(f'{label}: {prob:.4f}')


if __name__ == '__main__':
    main()
