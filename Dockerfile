FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
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

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

COPY requirements.txt .

# Install base dependencies FIRST
RUN pip install --no-cache-dir \
    "Flask==3.0.0" \
    "flask-cors==4.0.0" \
    "Pillow>=10.0.0" \
    "gunicorn==21.2.0" \
    "requests>=2.31.0" \
    "fal-client>=0.4.0" \
    "openai>=1.0.0"

# Install scikit-learn (needs numpy, will get numpy 2.x initially - that's ok)
RUN pip install --no-cache-dir "scikit-learn>=1.3.0"

# Install PyTorch CPU-only
RUN pip install --no-cache-dir \
    "torch>=2.0.0" \
    "torchvision>=0.15.0" \
    --index-url https://download.pytorch.org/whl/cpu

# ============================================================================
# CRITICAL: TensorFlow 2.15.0 + Keras 2.15.0 for .h5 model compatibility
# TF 2.16+ uses Keras 3.x which CANNOT load old .h5 models
# ============================================================================
RUN pip install --no-cache-dir \
    "numpy==1.26.4" \
    "protobuf==3.20.3" \
    "tensorflow==2.15.0" \
    "keras==2.15.0"

# Install OpenCV headless (compatible with NumPy 1.x)
RUN pip install --no-cache-dir "opencv-python-headless==4.8.1.78"

# Install mediapipe with --no-deps (prevents it from pulling opencv-contrib which needs numpy 2.x)
RUN pip install --no-cache-dir --no-deps "mediapipe==0.10.9"

# Install mediapipe's required dependencies manually (excluding opencv and numpy)
RUN pip install --no-cache-dir \
    "attrs>=19.1.0" \
    "matplotlib" \
    "absl-py"

# ============================================================================
# NO DEEPFACE - it requires retina-face which upgrades TensorFlow to 2.20.0
# Gender detection will be unavailable but attractiveness scoring will work
# ============================================================================

# Install pandas for data handling (used by various components)
RUN pip install --no-cache-dir "pandas>=0.23.4"

# FINAL LOCK: Force the exact versions we need
# This MUST be the last pip install to override any transitive dependency changes
RUN pip install --no-cache-dir --force-reinstall \
    "numpy==1.26.4" \
    "protobuf==3.20.3"

# ============================================================================
# VERIFY: All imports work AND correct versions are installed
# ============================================================================
RUN python -c "\
import sys; \
print('=== VERIFICATION ==='); \
import numpy; \
assert numpy.__version__ == '1.26.4', f'Wrong NumPy: {numpy.__version__}'; \
print(f'✅ NumPy: {numpy.__version__}'); \
import tensorflow as tf; \
assert tf.__version__ == '2.15.0', f'Wrong TensorFlow: {tf.__version__}'; \
print(f'✅ TensorFlow: {tf.__version__}'); \
import keras; \
assert keras.__version__ == '2.15.0', f'Wrong Keras: {keras.__version__}'; \
print(f'✅ Keras: {keras.__version__}'); \
import cv2; print(f'✅ OpenCV: {cv2.__version__}'); \
import mediapipe; print(f'✅ MediaPipe: {mediapipe.__version__}'); \
print('=== ALL CHECKS PASSED ==='); \
"

# Copy application files
COPY app.py .
COPY runtime.txt* .
COPY railway.json* .
COPY start.sh .
RUN chmod +x start.sh

# Copy models
COPY models/ ./models/

# Verify model exists and can be loaded
RUN ls -lh ./models/ && \
    test -f ./models/attractivenet_mnv2.h5 && \
    echo "✅ Model file found" && \
    python -c "\
from tensorflow import keras; \
model = keras.models.load_model('./models/attractivenet_mnv2.h5'); \
print(f'✅ Model loaded successfully: {model.input_shape} -> {model.output_shape}'); \
"

EXPOSE 5000

CMD ["./start.sh"]
