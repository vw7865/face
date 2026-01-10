#!/bin/bash
set -e

echo "🚀 Starting LooksMax AI Backend..."
echo "PORT: ${PORT:-NOT SET}"
echo "Working directory: $(pwd)"
echo "Python version: $(python --version)"
echo "Files in /app:"
ls -la /app/ | head -10

echo ""
echo "🔍 Checking if app.py exists and is readable..."
if [ -f "app.py" ]; then
    echo "✅ app.py exists"
    echo "📄 First 5 lines of app.py:"
    head -5 app.py
else
    echo "❌ app.py NOT FOUND!"
    exit 1
fi

echo ""
echo "🔍 Testing Python import..."
python -c "import app; print('✅ app module imported successfully')" || {
    echo "❌ Failed to import app module"
    exit 1
}

echo ""
echo "🚀 Starting gunicorn..."
exec gunicorn app:app \
    --bind "0.0.0.0:${PORT:-5000}" \
    --timeout 300 \
    --workers 1 \
    --preload \
    --log-level info \
    --access-logfile - \
    --error-logfile -

