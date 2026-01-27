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

echo "=== Checking for model files ==="
if [ -d "models" ]; then
    echo "Models directory exists"
    ls -lh models/*.pt 2>/dev/null || echo "No .pt model files found in models/"
else
    echo "⚠️  WARNING: models/ directory not found"
    echo "   Creating models directory..."
    mkdir -p models
fi

# Download beauty-classifier model if not present
GOOGLE_DRIVE_FILE_ID="${GOOGLE_DRIVE_FILE_ID:-1ehgqY0s9HsK_K-qHx1LSl3fipVBXa9Rp}"
if [ ! -f "models/attractiveness_classifier.pt" ]; then
    echo "=== Beauty-classifier model not found ==="
    echo "   Attempting to download from Google Drive (id=$GOOGLE_DRIVE_FILE_ID)..."
    wget --no-check-certificate "https://drive.google.com/uc?export=download&id=$GOOGLE_DRIVE_FILE_ID" -O models/attractiveness_classifier.pt 2>/dev/null || true
    # Large files: Google often returns HTML virus-scan page; try gdown if file is small/missing
    SIZE=$(wc -c < "models/attractiveness_classifier.pt" 2>/dev/null || echo 0)
    if [ ! -s "models/attractiveness_classifier.pt" ] || [ "$SIZE" -lt 50000000 ]; then
        echo "   Trying gdown for large-file support..."
        pip install -q gdown 2>/dev/null && gdown "https://drive.google.com/uc?id=$GOOGLE_DRIVE_FILE_ID" -O models/attractiveness_classifier.pt --fuzzy 2>/dev/null || true
    fi
    # Option 2: Dropbox (uncomment and add your link)
    # DROPBOX_LINK="https://dl.dropboxusercontent.com/s/YOUR_FILE_ID/attractiveness_classifier.pt"
    # if [ -n "$DROPBOX_LINK" ]; then
    #     echo "Downloading from Dropbox..."
    #     wget "$DROPBOX_LINK" -O models/attractiveness_classifier.pt || echo "Download failed"
    # fi
    
    if [ ! -f "models/attractiveness_classifier.pt" ]; then
        echo "⚠️  Beauty-classifier model not available"
        echo "   App will work with FaceStats only (still fully functional!)"
    else
        echo "✅ Beauty-classifier model downloaded successfully"
    fi
else
    echo "✅ Beauty-classifier model already present"
fi

echo "=== Build complete ==="
