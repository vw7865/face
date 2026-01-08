#!/bin/bash
set -e  # Exit on error

echo "=== Starting build ==="
echo "Python version: $(python --version)"
echo "Pip version: $(pip --version)"
echo "Platform: $(uname -a)"

echo "=== Upgrading pip, setuptools, wheel ==="
pip install --upgrade pip setuptools wheel

echo "=== Installing system dependencies (if needed) ==="
# MediaPipe might need these
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq || true
apt-get install -y -qq libglib2.0-0 libgomp1 || echo "System deps optional"

echo "=== Installing numpy first (MediaPipe dependency) ==="
pip install "numpy>=1.24.0,<2.0.0"

echo "=== Installing MediaPipe 0.10.21 (with solutions API) ==="
pip install --verbose mediapipe==0.10.21 2>&1 | tee /tmp/mediapipe_install.log || {
    echo "=== MediaPipe install failed, checking logs ==="
    cat /tmp/mediapipe_install.log
    echo "ERROR: MediaPipe installation failed"
    exit 1
}

echo "=== Verifying MediaPipe installation ==="
python -c "
import sys
try:
    import mediapipe as mp
    print(f'SUCCESS: MediaPipe imported')
    print(f'MediaPipe version: {getattr(mp, \"__version__\", \"unknown\")}')
    print(f'Has solutions: {hasattr(mp, \"solutions\")}')
    if hasattr(mp, 'solutions'):
        print('MediaPipe solutions module is available')
    sys.exit(0)
except ImportError as e:
    print(f'WARNING: MediaPipe not installed: {e}')
    print('App will use mock results')
    sys.exit(0)
except Exception as e:
    print(f'ERROR: MediaPipe import error: {e}')
    sys.exit(0)
"

echo "=== Installing other requirements ==="
pip install -r requirements.txt

echo "=== Build complete ==="
