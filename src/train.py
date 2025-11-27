
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
try:
    # package-relative imports when run as module
    from .data_loader import get_dataloaders, save_scaler
    from .model import WorkoutClassifier
except Exception:
    # fallback when running as script from src directory
    from data_loader import get_dataloaders, save_scaler
    from model import WorkoutClassifier

# Hyperparameters
NUM_EPOCHS = 20
LEARNING_RATE = 0.001
BATCH_SIZE = 8
VAL_SPLIT = 0.2

DATA_PATH = 'C:/Users/luke0/dashbeast/data/workouts.csv'
MODEL_PATH = 'C:/Users/luke0/dashbeast/workout_model.pth'
CHECKPOINT_PATH = 'C:/Users/luke0/dashbeast/workout_checkpoint.pth'
SCALER_PATH = 'C:/Users/luke0/dashbeast/scaler.json'
LABEL_MAP_PATH = 'C:/Users/luke0/dashbeast/label_map.json'


def evaluate(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    loss_accum = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for features, labels in data_loader:
            features = features.to(device)
            labels = labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss_accum += loss.item() * labels.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    avg_loss = loss_accum / total if total > 0 else 0.0
    acc = correct / total if total > 0 else 0.0
    return avg_loss, acc


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load external label_map if present to enforce mapping
    external_label_map = None
    try:
        with open(LABEL_MAP_PATH, 'r') as f:
            external_label_map = json.load(f)
    except Exception:
        external_label_map = None

    # Load data (pass provided label_map to ensure consistent class ordering)
    train_loader, val_loader, label_map, scaler = get_dataloaders(DATA_PATH, batch_size=BATCH_SIZE, val_split=VAL_SPLIT, provided_label_map=external_label_map)

    # Initialize model, loss, and optimizer
    model = WorkoutClassifier(num_features=6, num_classes=len(label_map)).to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float('inf')

    # Training loop
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0
        total = 0
        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(features)
            loss = loss_function(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * labels.size(0)
            total += labels.size(0)

        avg_train_loss = epoch_loss / total if total > 0 else 0.0

        # Validation
        if val_loader is not None:
            val_loss, val_acc = evaluate(model, val_loader, device)
            print(f'Epoch [{epoch+1}/{NUM_EPOCHS}] Train Loss: {avg_train_loss:.4f} Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}')
        else:
            print(f'Epoch [{epoch+1}/{NUM_EPOCHS}] Train Loss: {avg_train_loss:.4f}')

        # Save best model by validation loss
        if val_loader is not None and val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'label_map': label_map,
            }, CHECKPOINT_PATH)

    # Save final model and artifacts
    torch.save(model.state_dict(), MODEL_PATH)
    save_scaler(scaler, SCALER_PATH)
    with open(LABEL_MAP_PATH, 'w') as f:
        json.dump(label_map, f)

    print('Finished Training')


if __name__ == '__main__':
    main()
