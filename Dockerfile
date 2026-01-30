FROM python:3.11-slim

WORKDIR /app

# Retry deployment

# Install system dependencies for OpenCV and MediaPipe
# OpenCV needs libGL even with headless version in some cases
# Debian 12 uses libgl1 instead of libgl1-mesa-glx
# Also install wget/unzip for downloading model files
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Upgrade pip first for faster installs
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies in stages to optimize caching
# Install lighter dependencies first
RUN pip install --no-cache-dir \
    "setuptools>=65.0.0" \
    wheel \
    "Flask==3.0.0" \
    "flask-cors==4.0.0" \
    "numpy>=1.24.0,<1.25.0" \
    "Pillow>=10.0.0" \
    "gunicorn==21.2.0" \
    "requests>=2.31.0" \
    "fal-client>=0.4.0" \
    "openai>=1.0.0"

# Install heavy ML dependencies
RUN pip install --no-cache-dir \
    "opencv-python-headless>=4.8.0" \
    "mediapipe==0.10.21" \
    "scikit-learn>=1.3.0"

# Install PyTorch (CPU-only, smaller and faster)
RUN pip install --no-cache-dir \
    "torch>=2.0.0" \
    "torchvision>=0.15.0" \
    --index-url https://download.pytorch.org/whl/cpu

# Install TensorFlow 2.13.x (for AttractiveNet MobileNetV2 model)
# MUST use 2.13.x - newer versions (2.16+) use Keras 3 which is incompatible with old .h5 models
RUN pip install --no-cache-dir "tensorflow==2.13.1"

# Install deepface (depends on torch)
RUN pip install --no-cache-dir "deepface>=0.0.79"

# CRITICAL: Force NumPy to 1.24.x AFTER all installs
# Some packages (mediapipe, deepface) may upgrade NumPy to 2.x which breaks TensorFlow 2.13
RUN pip install --no-cache-dir "numpy==1.24.4" --force-reinstall && \
    python -c "import numpy; print(f'✅ NumPy version: {numpy.__version__}')"

# Copy application and configuration files
COPY app.py .
COPY runtime.txt* .
COPY railway.json* .
COPY start.sh .
RUN chmod +x start.sh

# Copy models from repository (includes AttractiveNet MobileNetV2)
# AttractiveNet is the PRIMARY model - outputs 1-5 range with excellent discrimination
COPY models/ ./models/

# Verify AttractiveNet model exists
RUN ls -lh ./models/ && \
    test -f ./models/attractivenet_mnv2.h5 && \
    echo "✅ AttractiveNet model found ($(stat -c%s ./models/attractivenet_mnv2.h5 2>/dev/null || stat -f%z ./models/attractivenet_mnv2.h5) bytes)" || \
    echo "⚠️ AttractiveNet model not found - check git LFS or download manually"

# Expose port
EXPOSE 5000

# Run with startup script (better logging and error handling)
# Railway should use startCommand from railway.json, but this is fallback
CMD ["./start.sh"]

