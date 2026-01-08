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

## Deployment

### Using Docker:
```bash
docker build -t looksmax-backend .
docker run -p 5000:5000 looksmax-backend
```

### Using Heroku:
```bash
heroku create your-app-name
git push heroku main
```

### Using AWS/GCP:
- Use the Dockerfile to build and deploy to container services
- Or use serverless functions (AWS Lambda, Google Cloud Functions)

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

