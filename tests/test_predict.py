import os
import json
import subprocess
import sys


def test_predict_runs():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    pyexe = sys.executable
    predict_script = os.path.join(repo_root, 'src', 'predict.py')

    # run predict with default sample (no --features) to ensure it executes
    proc = subprocess.run([pyexe, predict_script], capture_output=True, text=True)
    assert proc.returncode == 0
    # output should contain something like 'class_' or a label
    assert proc.stdout.strip() != ''
