# LooksMax AI Backend

Face analysis API using MediaPipe Face Mesh.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run locally:
```bash
python app.py
```

3. Test endpoint:
```bash
curl http://localhost:5000/health
```

## Render Deployment

**IMPORTANT**: Update your Render build command to:
```bash
pip install --upgrade pip setuptools wheel && pip install --force-reinstall --no-cache-dir mediapipe==0.10.31 && pip install -r requirements.txt
```

This ensures MediaPipe installs correctly on Render's platform.

## API Endpoint

**POST** `/api/analyze-face`

**Form Data:**
- `front_image`: Image file
- `side_image`: Image file
- `gender`: "Male" or "Female"

**Response:**
JSON with all calculated metrics (see app.py for structure)

## Environment Variables

- `PORT`: Server port (default: 5000)
