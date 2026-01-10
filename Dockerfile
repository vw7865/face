FROM python:3.11-slim

WORKDIR /app

# Install minimal system dependencies
# MediaPipe should work without OpenGL in headless mode
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
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

# Create models directory (app will handle missing models gracefully)
RUN mkdir -p ./models

# Copy model files (will fail if models/ doesn't exist - that's OK, app handles it)
# If models don't exist in repo, they'll be downloaded during build or runtime
COPY models/ ./models/

# Expose port
EXPOSE 5000

# Run with gunicorn (Railway sets PORT automatically)
# Use shell form to allow variable expansion
CMD ["sh", "-c", "gunicorn app:app --bind 0.0.0.0:${PORT:-5000} --timeout 300 --workers 1 --preload"]

