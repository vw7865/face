#!/bin/bash
# Force output to stdout/stderr (Railway captures these)
exec 1>&2  # Redirect stdout to stderr so Railway sees it

set -e

echo "🚀 Starting LooksMax AI Backend..." >&2
echo "PORT: ${PORT:-NOT SET}" >&2
echo "Working directory: $(pwd)" >&2
echo "Python version: $(python --version)" >&2

echo "" >&2
echo "🔍 Checking if app.py exists..." >&2
if [ -f "app.py" ]; then
    echo "✅ app.py exists" >&2
else
    echo "❌ app.py NOT FOUND!" >&2
    exit 1
fi

echo "" >&2
echo "🔍 Testing Python import..." >&2
python -c "import app; print('✅ app module imported successfully')" 2>&1 || {
    echo "❌ Failed to import app module" >&2
    python -c "import app" 2>&1 || true  # Show full error
    exit 1
}

echo "" >&2
echo "🚀 Starting gunicorn on port ${PORT:-5000}..." >&2
echo "Command: gunicorn app:app --bind 0.0.0.0:${PORT:-5000} --timeout 300 --workers 1 --preload" >&2
echo "" >&2

# Use exec to replace shell with gunicorn (so Railway sees it as the main process)
# All gunicorn output goes to stdout/stderr which Railway captures
exec gunicorn app:app \
    --bind "0.0.0.0:${PORT:-5000}" \
    --timeout 300 \
    --workers 1 \
    --preload \
    --log-level info \
    --access-logfile - \
    --error-logfile -

