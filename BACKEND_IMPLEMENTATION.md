# Backend Implementation Guide for Face Rating

This document outlines how to implement the backend API that processes face images using MediaPipe Face Mesh and calculates all the facial metrics.

## Architecture

```
iOS App → Backend API → MediaPipe Face Mesh → Metric Calculations → JSON Response
```

## Backend Requirements

### 1. Technology Stack
- **Python** (recommended for MediaPipe)
- **Flask/FastAPI** for API endpoints
- **MediaPipe** for face mesh detection
- **NumPy** for geometric calculations
- **OpenCV** for image processing

### 2. API Endpoint

**POST** `/api/analyze-face`

**Request:**
- `front_image`: Multipart file upload (JPEG/PNG)
- `side_image`: Multipart file upload (JPEG/PNG)
- `gender`: String ("Male" or "Female")

**Response:**
```json
{
  "overall": {
    "psl": 85.5,
    "potential": 88.2
  },
  "eyes": {
    "orbitalDepth": 78.3,
    "canthalTilt": 82.1,
    "eyebrowDensity": 75.6,
    "eyelashDensity": 80.4,
    "eyelidExposure": 77.8,
    "underEyeHealth": 79.2
  },
  "midface": {
    "cheekbones": 83.5,
    "maxilla": 81.2,
    "nose": 79.8,
    "ipd": 76.4,
    "fwhr": 84.1,
    "compactness": 80.7
  },
  "lowerThird": {
    "lips": 77.3,
    "mandible": 85.6,
    "gonialAngle": 82.4,
    "ramus": 79.1,
    "hyoidSkinTightness": 78.9,
    "jawWidth": 83.2
  },
  "upperThird": {
    "norwoodStage": 88.5,
    "foreheadProjection": 81.3,
    "hairlineRecession": 79.7,
    "hairThinning": 82.6,
    "hairlineDensity": 85.1,
    "foreheadSlope": 80.4
  },
  "miscellaneous": {
    "skin": 84.2,
    "harmony": 86.7,
    "symmetry": 88.1,
    "neckWidth": 82.3,
    "bloat": 79.5,
    "boneMass": 83.8
  },
  "ascensionDate": "2026-04-07T00:00:00Z"
}
```

## Implementation Steps

### Step 1: Setup MediaPipe

```python
import mediapipe as mp
import cv2
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)
```

### Step 2: Process Images

```python
def process_image(image_bytes):
    # Convert bytes to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Get face mesh landmarks
    results = face_mesh.process(image_rgb)
    
    if not results.multi_face_landmarks:
        return None
    
    # Extract 468 landmarks (x, y, z)
    landmarks = results.multi_face_landmarks[0].landmark
    points = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    
    return points
```

### Step 3: Calculate Metrics

Use the formulas provided in the user's requirements. Key helper functions:

```python
def euclidean_distance(p1, p2):
    return np.linalg.norm(p1 - p2)

def angle_between_vectors(v1, v2):
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

def normalize_by_ipd(distance, ipd):
    return distance / ipd
```

### Step 4: Score Conversion (0-100)

For each metric, convert raw values to 0-100 scores:

```python
def score_metric(value, ideal_range, method='gaussian'):
    """
    Convert raw metric value to 0-100 score.
    
    Args:
        value: Raw metric value
        ideal_range: (min, max) tuple for ideal range
        method: 'gaussian' or 'linear'
    """
    if method == 'gaussian':
        center = (ideal_range[0] + ideal_range[1]) / 2
        std = (ideal_range[1] - ideal_range[0]) / 4
        score = 100 * np.exp(-0.5 * ((value - center) / std) ** 2)
        return np.clip(score, 0, 100)
    else:
        # Linear mapping
        if value < ideal_range[0]:
            return max(0, 100 * (value / ideal_range[0]))
        elif value > ideal_range[1]:
            return max(0, 100 - 100 * ((value - ideal_range[1]) / ideal_range[1]))
        else:
            return 100
```

### Step 5: Landmark Indices Reference

You'll need to map anatomical points to MediaPipe landmark indices. Key landmarks:

- **Eyes**: 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246
- **Nose**: 1, 2, 4, 5, 6, 19, 20, 94, 125, 141, 235, 236, 3, 51, 48, 115, 131, 134, 102, 49, 220, 305, 290, 305, 289, 439, 455, 310, 311, 312, 13, 82, 81, 80, 78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308
- **Jaw**: 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323
- **Mouth**: 61, 146, 91, 181, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318
- **Forehead**: 10, 151, 337, 299, 333, 298, 301, 368, 264, 447, 366, 401, 435, 410, 454, 356, 389

**Note**: Use MediaPipe's official documentation for exact indices: https://google.github.io/mediapipe/solutions/face_mesh.html

### Step 6: Example Metric Calculation

```python
def calculate_canthal_tilt(landmarks, gender='Male'):
    # Get eye corner landmarks
    left_outer = landmarks[33]  # Left eye outer corner
    left_inner = landmarks[133]  # Left eye inner corner
    right_outer = landmarks[263]  # Right eye outer corner
    right_inner = landmarks[362]  # Right eye inner corner
    
    # Calculate tilt for left eye
    v_left = left_inner - left_outer
    tilt_left = np.degrees(np.arctan2(v_left[1], v_left[0]))
    
    # Calculate tilt for right eye
    v_right = right_inner - right_outer
    tilt_right = np.degrees(np.arctan2(v_right[1], v_right[0]))
    
    # Average
    tilt = (tilt_left + tilt_right) / 2
    
    # Score (ideal range 5-12° for males)
    ideal_min, ideal_max = (5, 12) if gender == 'Male' else (3, 10)
    score = score_metric(tilt, (ideal_min, ideal_max), method='gaussian')
    
    return score
```

### Step 7: API Endpoint Implementation

```python
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/api/analyze-face', methods=['POST'])
def analyze_face():
    try:
        front_image = request.files['front_image']
        side_image = request.files['side_image']
        gender = request.form.get('gender', 'Male')
        
        # Process images
        front_landmarks = process_image(front_image.read())
        side_landmarks = process_image(side_image.read())
        
        if front_landmarks is None or side_landmarks is None:
            return jsonify({'error': 'Face not detected'}), 400
        
        # Calculate all metrics
        results = {
            'overall': calculate_overall(front_landmarks, side_landmarks, gender),
            'eyes': calculate_eyes(front_landmarks, side_landmarks, gender),
            'midface': calculate_midface(front_landmarks, side_landmarks, gender),
            'lowerThird': calculate_lower_third(front_landmarks, side_landmarks, gender),
            'upperThird': calculate_upper_third(front_landmarks, side_landmarks, gender),
            'miscellaneous': calculate_miscellaneous(front_landmarks, side_landmarks, gender),
            'ascensionDate': calculate_ascension_date()  # Based on improvement potential
        }
        
        return jsonify(results), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

## Next Steps

1. **Update FaceAnalysisService.swift** with your actual backend URL
2. **Implement the backend** using the structure above
3. **Test with sample images** to calibrate scoring ranges
4. **Deploy backend** to a cloud service (AWS, Google Cloud, Heroku, etc.)
5. **Update iOS app** to handle the real API responses

## Additional Notes

- **Privacy**: Ensure images are deleted after processing (30 days as per privacy policy)
- **Performance**: Consider caching results for the same user
- **Validation**: Add input validation for image size, format, etc.
- **Error Handling**: Implement proper error responses for edge cases
- **Security**: Add authentication/rate limiting for production







