# Dashbeast

This repository contains a small PyTorch classifier (`WorkoutClassifier`) to predict workout types from 6 sensor features (accelerometer + gyroscope), plus scikit-learn baselines: logistic regression and gradient boosting. Using Python 3.11.9

## Overview

**What it does**
- Classifies workout types (pushup, squat, jumping_jack, bench press, bicep curl, deadlift, lunges, plank, burpees, mountain climbers) from accelerometer and gyroscope sensor data.
- Uses a 3-layer feedforward neural network trained with PyTorch.
- Includes feature normalization (per-feature scaling) and train/validation splits for robust training.
- Saves the trained model, scaler parameters, and label mappings for reproducible inference.

**Key features**
- **PyTorch neural network:** 3-layer feedforward classifier with feature normalization and train/validation split.
- **Logistic regression baseline:** simple, interpretable linear model with scikit-learn.
- **Gradient boosting baseline:** ensemble method with excellent performance characteristics.
- **Training pipelines:** each model supports external label maps, train/validation split, and automatic metric computation.
- **Inference CLI:** accepts 6 sensor features, returns top-K workout predictions with probabilities.
- **Reproducibility:** saves the scaler and label map with each checkpoint so inference uses identical preprocessing as training.
- **Tests:** unit tests for the dataloader, PyTorch predict CLI, logistic regression, and gradient boosting.

**Tech stack**
- PyTorch (neural network model and training)
- scikit-learn (logistic regression and gradient boosting)
- pandas (data loading)
- joblib (model serialization for scikit-learn)
- pytest (testing)
- Python 3.8–3.11

## Quick Demo

To demonstrate the project to your boss, follow these steps (5 minutes):

**1) Setup (1 minute)**

```powershell
# Clone and enter the repo
git clone https://github.com/luke-kustra/dashbeast-machine-learning-model.git
cd dashbeast-machine-learning-model

# Create and activate venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
python -m pip install --upgrade pip
python -m pip install -r src\requirements.txt
```

**2) Train the model (2 minutes)**

```powershell
python -m src.train
```

You'll see output like:
```
Epoch [1/20] Train Loss: 2.3112 Val Loss: 2.3259 Val Acc: 0.0000
Epoch [2/20] Train Loss: 2.2912 Val Loss: 2.3397 Val Acc: 0.0000
...
Epoch [20/20] Train Loss: 1.9928 Val Loss: 2.6217 Val Acc: 0.0000
Finished Training
```

This generates:
- `workout_model.pth` — trained model weights
- `workout_checkpoint.pth` — best checkpoint (with label map)
- `scaler.json` — feature normalization (mean/std)
- `label_map.json` — workout label mappings

**3) Run predictions (1 minute)**

```powershell
# Predict with example sensor data (accelerometer + gyroscope)
python -m src.predict --features 0.1 0.2 0.3 0.4 0.5 0.6 --topk 3
```

Output:
```
class_1: 0.3395
class_0: 0.3340
class_2: 0.3266
```

Or with the default sample:
```powershell
python -m src.predict
```

## Running Each ML Model

### PyTorch Neural Network

**Train:**
```powershell
python -m src.train
```

**Predict:**
```powershell
python -m src.predict --features 0.1 0.2 0.3 0.4 0.5 0.6 --topk 3
```

### Logistic Regression

**Train:**
```powershell
python -m src.logreg train
```

Saves model to `logreg_model.joblib` and label map to `logreg_label_map.json`.

**Predict:**
```powershell
python -m src.logreg predict --features 0.1 0.2 0.3 0.4 0.5 0.6 --topk 3
```

### Gradient Boosting

**Train:**
```powershell
python -m src.gboost train
```

Saves model to `gradient_boosting_model.joblib` and label map to `gboost_label_map.json`.

**Predict:**
```powershell
python -m src.gboost predict --features 0.1 0.2 0.3 0.4 0.5 0.6 --topk 3
```

## Model Files Structure

After training all three models, your project will have:

```
dashbeast/
├── workout_model.pth              # PyTorch model weights
├── workout_checkpoint.pth         # PyTorch best checkpoint
├── scaler.json                    # PyTorch scaler (mean/std)
├── logreg_model.joblib            # Logistic regression model
├── logreg_label_map.json          # Logistic regression label map
├── gradient_boosting_model.joblib # Gradient boosting model
├── gboost_label_map.json          # Gradient boosting label map
└── data/
    └── workouts.csv               # Training data
```

Each model is independent with its own label map, allowing them to be trained and used separately.

**4) Run tests (1 minute)**

```powershell
python -m pip install pytest
python -m pytest -q
```

Expected output:
```
5 passed in 13.9s
```

## Architecture

**Model structure** (`src/model.py`)
```
Input (6 features)
  ↓
Linear(6 → 32) + ReLU
  ↓
Linear(32 → 16) + ReLU
  ↓
Linear(16 → num_classes)
  ↓
Output (logits)
```

**Training pipeline** (`src/train.py`)
1. Loads `data/workouts.csv`
2. Applies feature normalization (mean/std scaling)
3. Splits into train (80%) / validation (20%)
4. Trains for 20 epochs with Adam optimizer
5. Saves best checkpoint based on validation loss
6. Outputs `scaler.json` + `label_map.json` for reproducible inference

**Inference pipeline** (`src/predict.py`)
1. Loads the saved model + scaler
2. Normalizes input features (using saved scaler)
3. Runs forward pass through the network
4. Applies softmax to get probabilities
5. Returns top-K predictions with confidence scores

## Key Improvements Made

- **Feature normalization:** per-feature mean/std scaling prevents features from dominating due to different units/ranges.
- **Train/validation split:** 80/20 split with deterministic seed ensures reproducible training and validation metrics.
- **Checkpoint saving:** best model saved based on validation loss; checkpoint contains the label map for consistency.
- **Scaler serialization:** `scaler.json` ensures inference applies the exact same normalization as training.
- **Flexible label mapping:** external `label_map.json` supported — allows enforcing specific label orderings across train/predict.
- **Probabilistic predictions:** returns softmax probabilities instead of hard class predictions; top-K support for confidence ranking.

Quick start


**Prerequisites**
- Python 3.8–3.11 (64-bit) installed on your system. Download from [python.org](https://www.python.org/downloads/windows/) or install via `winget install Python.Python.3.11`.

**Setup (one-time)**

1) Clone the repository and navigate to the project root:

```powershell
git clone https://github.com/YOUR_USERNAME/dashbeast-machine-learning-model.git
cd dashbeast-machine-learning-model
```

2) Create and activate a virtual environment (Windows PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3) Upgrade pip and install dependencies:

```powershell
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r src\requirements.txt
```

**Run the project**

4) Train the model (generates `workout_model.pth`, `scaler.json`, and `label_map.json`):

```powershell
python -m src.train
```

5) Predict from six features (CLI):

```powershell
# With custom features
python -m src.predict --features 0.1 0.2 0.3 0.4 0.5 0.6 --topk 3

# Or use the default sample
python -m src.predict
```

## Testing

### Run All Tests

The repository includes a `pytest` suite that tests all three ML models:

```powershell
# Activate venv
.\.venv\Scripts\Activate.ps1

# Install pytest (if missing)
python -m pip install pytest

# Run all tests
python -m pytest -v
```

Expected output:
```text
5 passed in ~15s
```

### Test Individual Models

**Test only logistic regression:**
```powershell
python -m pytest tests/test_sklearn_models.py::test_logistic_regression_train_and_predict -v
```

For manual testing:
```powershell
python -m src.logreg train 
python -m src.logreg predict --features 0.1 0.2 0.3 0.4 0.5 0.6 --topk 3   
```
the 6 numbers in the middle is sample data the user will input by working out.

**Test only gradient boosting:**
```powershell
python -m pytest tests/test_sklearn_models.py::test_gradient_boosting_train_and_predict -v
```
For manual testing:
```powershell
python -m src.gboost train
python -m src.gboost predict --features 0.23 0.55 1.31 -3.31 0.78 -0.99 --topk 3
```

**Test only PyTorch neural network:**
```powershell
python -m pytest tests/test_predict.py -v
```
For Manual Testing
```powershell
 python -m src.train
 python -m src.gboost predict --features 0.23 0.55 1.31 -3.31 0.78 -0.99 --topk 3
 ```
 
### Benchmark All Models

Compare all three models on speed and accuracy:

```powershell
python benchmark.py
```

Sample output:
```text
======================================================================
BENCHMARK RESULTS
======================================================================
Model                     |   Time (s) |   Accuracy
----------------------------------------------------------------------
neural_network            |      2.478 |     1.0000
logistic_regression       |      0.021 |     1.0000
gradient_boosting         |      1.075 |     0.9667

======================================================================
RECOMMENDATION
======================================================================
RECOMMENDED MODEL: LOGISTIC_REGRESSION
  - Accuracy: 1.0000
  - Training Time: 0.021s

Justification:
  ✓ Achieves 100% accuracy with exceptional training speed (20ms)
  ✓ ~100x faster than neural network with same accuracy
  ✓ Simple, interpretable, and perfect for small structured sensor data
```

The benchmark trains all three models on `data/workouts_multiclass.csv` and provides a recommendation based on accuracy and speed.

Troubleshooting

- If tests fail with `ModuleNotFoundError` for `torch` or `pandas`, make sure you activated the venv that has those packages installed. If activation is unreliable, run pytest with the venv python explicitly:

```powershell
& "C:\Users\luke0\dashbeast\.venv\Scripts\python.exe" -m pytest -q
```

- If `pip` cannot find a compatible `torch` wheel (wheel/ABI mismatch), recreate the venv with an official Windows CPython (3.8–3.11) or use Conda. Example (winget + venv):

```powershell
winget install --exact --id Python.Python.3.11 -e
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch
python -m pip install -r src\requirements.txt
```