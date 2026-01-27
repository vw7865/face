#!/bin/bash
set -e  # Exit on error

echo "=== Starting optimized build ==="
echo "Python version: $(python --version)"
echo "Pip version: $(pip --version)"

# Enable pip cache for faster installs
export PIP_CACHE_DIR=/tmp/pip-cache
mkdir -p $PIP_CACHE_DIR

echo "=== Upgrading pip, setuptools, wheel ==="
pip install --upgrade --cache-dir=$PIP_CACHE_DIR pip setuptools wheel

echo "=== Installing system dependencies (if needed) ==="
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq 2>/dev/null || true
apt-get install -y -qq libglib2.0-0 libgomp1 2>/dev/null || echo "System deps optional"

echo "=== Installing all requirements in one go (faster, uses cache) ==="
# Install everything at once - pip is smart about dependencies
# This avoids duplicate downloads and uses pip's dependency resolver
pip install --cache-dir=$PIP_CACHE_DIR -r requirements.txt

echo "=== Verifying MediaPipe installation ==="
python -c "
import sys
try:
    import mediapipe as mp
    print(f'✅ MediaPipe {getattr(mp, \"__version__\", \"unknown\")} imported')
    print(f'✅ Has solutions: {hasattr(mp, \"solutions\")}')
    sys.exit(0)
except Exception as e:
    print(f'⚠️  MediaPipe check: {e}')
    sys.exit(0)
"

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
