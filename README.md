# Dashbeast

This repository contains a small PyTorch classifier (`WorkoutClassifier`) to predict workout types from 6 sensor features (accelerometer + gyroscope). Using Python 3.11.9

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

Run tests

The repository includes a small `pytest` suite that checks the dataloader, scaler serialization, and that the prediction CLI runs.

1) Activate the same virtual environment used for running the project (PowerShell):

```powershell
# If your venv is at the repository root
.\.venv\Scripts\Activate.ps1

# If your venv was created inside `src` instead
# .\src\.venv\Scripts\Activate.ps1
```

2) Install `pytest` (if missing) and run tests from the repository root:

```powershell
python -m pip install pytest
Set-Location C:\Users\luke0\dashbeast   # or change to your repo root
python -m pytest -q
```

3) Expected output (example):

```text
3 passed in 4.7s
```

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

Notes


If you'd like, I can add examples for exporting the model or a small web demo.