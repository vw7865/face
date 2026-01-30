FROM python:3.11-slim

WORKDIR /app

# Cache bust v4: Fix deepface upgrading TensorFlow

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

# Copy requirements for caching
COPY requirements.txt .

# Install base dependencies
RUN pip install --no-cache-dir \
    "setuptools>=65.0.0" \
    wheel \
    "Flask==3.0.0" \
    "flask-cors==4.0.0" \
    "Pillow>=10.0.0" \
    "gunicorn==21.2.0" \
    "requests>=2.31.0" \
    "fal-client>=0.4.0" \
    "openai>=1.0.0"

# Install OpenCV and scikit-learn (without mediapipe yet)
RUN pip install --no-cache-dir \
    "opencv-python-headless==4.9.0.80" \
    "scikit-learn>=1.3.0"

# Install PyTorch (CPU-only)
RUN pip install --no-cache-dir \
    "torch>=2.0.0" \
    "torchvision>=0.15.0" \
    --index-url https://download.pytorch.org/whl/cpu

# Install deepface (will install its own TensorFlow - we'll override it)
RUN pip install --no-cache-dir "deepface==0.0.79"

# CRITICAL: Force the correct versions AFTER deepface
# TensorFlow 2.15.0 is the LAST version with Keras 2.x (compatible with old .h5 models)
# protobuf 3.20.3 is required by MediaPipe
RUN pip install --no-cache-dir --force-reinstall \
    "tensorflow==2.15.0" \
    "keras==2.15.0" \
    "protobuf==3.20.3" \
    "numpy==1.26.4"

# Now install MediaPipe (requires protobuf<4)
RUN pip install --no-cache-dir "mediapipe==0.10.9"

# Verify all packages are correct versions
RUN python -c "\
import sys; \
print('Verifying package versions...'); \
import numpy; print(f'NumPy: {numpy.__version__}'); assert numpy.__version__ == '1.26.4', 'Wrong NumPy'; \
import tensorflow as tf; print(f'TensorFlow: {tf.__version__}'); assert tf.__version__ == '2.15.0', 'Wrong TensorFlow'; \
import keras; print(f'Keras: {keras.__version__}'); assert keras.__version__ == '2.15.0', 'Wrong Keras'; \
import google.protobuf; print(f'Protobuf: {google.protobuf.__version__}'); \
import cv2; print(f'OpenCV: {cv2.__version__}'); \
import mediapipe; print(f'MediaPipe: {mediapipe.__version__}'); \
print('✅ All versions correct!'); \
"

# Copy application files
COPY app.py .
COPY runtime.txt* .
COPY railway.json* .
COPY start.sh .
RUN chmod +x start.sh

# Copy models (includes AttractiveNet MobileNetV2)
COPY models/ ./models/

# Verify model exists
RUN ls -lh ./models/ && \
    test -f ./models/attractivenet_mnv2.h5 && \
    echo "✅ AttractiveNet model found ($(stat -c%s ./models/attractivenet_mnv2.h5 2>/dev/null || stat -f%z ./models/attractivenet_mnv2.h5) bytes)" || \
    echo "⚠️ AttractiveNet model not found"

EXPOSE 5000

CMD ["./start.sh"]
