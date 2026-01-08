#!/bin/bash
set -e  # Exit on error

echo "=== Starting build ==="
echo "Python version: $(python --version)"
echo "Pip version: $(pip --version)"

echo "=== Upgrading pip, setuptools, wheel ==="
pip install --upgrade pip setuptools wheel

echo "=== Installing MediaPipe explicitly ==="
pip install mediapipe==0.10.31 || {
    echo "ERROR: MediaPipe installation failed!"
    echo "Trying alternative installation method..."
    pip install --no-cache-dir mediapipe==0.10.31 || {
        echo "ERROR: MediaPipe still failed to install"
        exit 1
    }
}

echo "=== Verifying MediaPipe installation ==="
python -c "import mediapipe as mp; print(f'MediaPipe version: {mp.__version__}'); print(f'Has solutions: {hasattr(mp, \"solutions\")}')" || {
    echo "ERROR: MediaPipe import failed after installation"
    exit 1
}

echo "=== Installing other requirements ==="
pip install -r requirements.txt

echo "=== Build complete ==="
