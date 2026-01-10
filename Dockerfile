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

# Copy model files (will fail if models/ doesn't exist - that's OK, app handles it)
# If models don't exist in repo, they'll be downloaded during build or runtime
COPY models/ ./models/

# Expose port
EXPOSE 5000

# Run with startup script (better logging and error handling)
# Railway should use startCommand from railway.json, but this is fallback
CMD ["./start.sh"]

