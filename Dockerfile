FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for OpenCV and MediaPipe
# OpenCV needs libGL even with headless version in some cases
# Debian 12 uses libgl1 instead of libgl1-mesa-glx
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY app.py .

# Copy configuration files
COPY requirements.txt .
COPY runtime.txt* .
COPY railway.json* .
COPY start.sh .
RUN chmod +x start.sh

# Create models directory (app will handle missing models gracefully)
RUN mkdir -p ./models

# Download FaceStats model from GitHub (since Git LFS isn't working on Railway)
# The model is 774 KB and available directly from the FaceStats repo
RUN wget -q https://github.com/jayklarin/FaceStats/raw/main/models/attractiveness_regressor.pt -O ./models/attractiveness_regressor.pt || echo "Model download failed - will try to use local copy"

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

