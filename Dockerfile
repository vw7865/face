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

# Install base dependencies
RUN pip install --no-cache-dir \
    "Flask==3.0.0" \
    "flask-cors==4.0.0" \
    "Pillow>=10.0.0" \
    "gunicorn==21.2.0" \
    "requests>=2.31.0" \
    "fal-client>=0.4.0" \
    "openai>=1.0.0"

# Install scikit-learn
RUN pip install --no-cache-dir "scikit-learn>=1.3.0"

# Install PyTorch (CPU-only)
RUN pip install --no-cache-dir \
    "torch>=2.0.0" \
    "torchvision>=0.15.0" \
    --index-url https://download.pytorch.org/whl/cpu

# Install TensorFlow 2.15.0 FIRST (last version with Keras 2.x)
RUN pip install --no-cache-dir \
    "tensorflow==2.15.0" \
    "keras==2.15.0" \
    "protobuf==3.20.3" \
    "numpy==1.26.4"

# Install OpenCV that works with NumPy 1.x
RUN pip install --no-cache-dir "opencv-python-headless==4.8.1.78"

# Install mediapipe with --no-deps to prevent it from upgrading numpy
RUN pip install --no-cache-dir --no-deps "mediapipe==0.10.9"

# Install mediapipe's other dependencies manually (excluding numpy and opencv)
RUN pip install --no-cache-dir \
    "attrs>=19.1.0" \
    "matplotlib" \
    "sounddevice>=0.4.4"

# Install deepface with --no-deps to prevent TensorFlow upgrade
RUN pip install --no-cache-dir --no-deps "deepface==0.0.79"

# Install deepface dependencies manually (excluding tensorflow, keras, numpy)
RUN pip install --no-cache-dir \
    "pandas>=0.23.4" \
    "gdown>=3.10.1" \
    "mtcnn>=0.1.0" \
    "retina-face>=0.0.1" \
    "fire>=0.4.0"

# FINAL: Force correct NumPy version
RUN pip install --no-cache-dir --force-reinstall "numpy==1.26.4"

# Verify all imports work
RUN python -c "\
import sys; \
print('Testing imports...'); \
import numpy; print(f'NumPy: {numpy.__version__}'); \
import tensorflow as tf; print(f'TensorFlow: {tf.__version__}'); \
from tensorflow import keras; print(f'Keras: {keras.__version__}'); \
import cv2; print(f'OpenCV: {cv2.__version__}'); \
import mediapipe; print(f'MediaPipe: {mediapipe.__version__}'); \
print('All imports OK!'); \
"

# Copy application files
COPY app.py .
COPY runtime.txt* .
COPY railway.json* .
COPY start.sh .
RUN chmod +x start.sh

# Copy models
COPY models/ ./models/

# Verify model exists
RUN ls -lh ./models/ && \
    test -f ./models/attractivenet_mnv2.h5 && \
    echo "Model found" || echo "Model NOT found"

EXPOSE 5000

CMD ["./start.sh"]
