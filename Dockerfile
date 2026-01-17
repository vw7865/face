FROM python:3.11-slim

WORKDIR /app

# Retry deployment

# Install system dependencies for OpenCV and MediaPipe
# OpenCV needs libGL even with headless version in some cases
# Debian 12 uses libgl1 instead of libgl1-mesa-glx
# Also install wget for downloading model files
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
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
    "numpy>=1.24.0,<2.0.0" \
    "Pillow>=10.0.0" \
    "gunicorn==21.2.0" \
    "requests>=2.31.0" \
    "fal-client>=0.4.0"

# Install heavy ML dependencies
RUN pip install --no-cache-dir \
    "opencv-python-headless>=4.8.0" \
    "mediapipe==0.10.21" \
    "scikit-learn>=1.3.0" \
    "polars>=0.19.0"

# Install PyTorch (CPU-only, smaller and faster)
RUN pip install --no-cache-dir \
    "torch>=2.0.0" \
    "torchvision>=0.15.0" \
    --index-url https://download.pytorch.org/whl/cpu

# Install transformers and deepface last (they depend on torch)
RUN pip install --no-cache-dir \
    "transformers>=4.30.0" \
    "deepface>=0.0.79"

# Copy application and configuration files
COPY app.py .
COPY runtime.txt* .
COPY railway.json* .
COPY start.sh .
RUN chmod +x start.sh

# Create models directory and download FaceStats model
# Do this in one step to minimize layers
RUN mkdir -p ./models && \
    wget -q --timeout=30 --tries=3 \
    https://raw.githubusercontent.com/jayklarin/FaceStats/main/models/attractiveness_regressor.pt \
    -O ./models/attractiveness_regressor.pt && \
    ls -lh ./models/attractiveness_regressor.pt && \
    test -f ./models/attractiveness_regressor.pt && \
    test -s ./models/attractiveness_regressor.pt || \
    (echo "❌ ERROR: FaceStats model download failed!" && exit 1)

# Download beauty-classifier model (94 MB - too large for GitHub, needs cloud storage)
# Option 1: Download from Google Drive (uncomment and add your file ID)
# RUN wget --no-check-certificate "https://drive.google.com/uc?export=download&id=YOUR_FILE_ID" -O ./models/attractiveness_classifier.pt || echo "Beauty-classifier model download failed"
# Option 2: Download from Dropbox (uncomment and add your link)
# RUN wget "https://dl.dropboxusercontent.com/s/YOUR_FILE_ID/attractiveness_classifier.pt" -O ./models/attractiveness_classifier.pt || echo "Beauty-classifier model download failed"
# Note: For now, beauty-classifier is optional - app works with FaceStats only

# Note: Model files are downloaded from GitHub above (FaceStats model)
# If you have additional local model files, uncomment the line below:
# COPY models/ ./models/

# Expose port
EXPOSE 5000

# Run with startup script (better logging and error handling)
# Railway should use startCommand from railway.json, but this is fallback
CMD ["./start.sh"]

