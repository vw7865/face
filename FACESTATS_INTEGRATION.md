# FaceStats Attractiveness Scoring Integration

## Overview

This document outlines how to integrate the FaceStats attractiveness scoring system into the LooksmaxAI backend.

## FaceStats Approach

Based on [jayklarin/FaceStats](https://github.com/jayklarin/FaceStats), the system uses:
- **CLIP embeddings** for robust face representation (handles pose/lighting well)
- **MLP regressor** for continuous attractiveness scoring (0-100)
- **sklearn classifiers** for gender/ethnicity (already using DeepFace for gender)

## Integration Steps

### 1. Clone FaceStats Repository

```bash
cd backend
git clone https://github.com/jayklarin/FaceStats.git facestats
```

### 2. Download Pre-trained Model

The FaceStats repo includes:
- `models/attractiveness_regressor.pt` - Trained MLP regressor
- CLIP model will be downloaded automatically via transformers

### 3. Implement Attractiveness Scoring

The `calculate_attractiveness_score()` function in `app.py` needs to:

1. **Load CLIP Model**:
```python
from transformers import CLIPProcessor, CLIPModel

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
```

2. **Extract CLIP Embedding**:
```python
# Convert image to PIL Image
from PIL import Image
pil_image = Image.fromarray(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))

# Get CLIP embedding
inputs = clip_processor(images=pil_image, return_tensors="pt")
with torch.no_grad():
    image_features = clip_model.get_image_features(**inputs)
    embedding = image_features[0].numpy()  # Shape: (512,)
```

3. **Load MLP Regressor**:
```python
import joblib
import torch.nn as nn

# Load the trained model
# You'll need to copy the model architecture from FaceStats
attractiveness_model = joblib.load("facestats/models/attractiveness_regressor.pt")
```

4. **Predict Attractiveness**:
```python
# Predict score (0-100)
score = attractiveness_model.predict([embedding])[0]
score = float(np.clip(score, 0.0, 100.0))
```

### 4. Model Architecture

The MLP regressor from FaceStats typically has:
- Input: CLIP embedding (512 dimensions)
- Hidden layers: 2-3 fully connected layers
- Output: Single value (0-100 attractiveness score)

### 5. Integration with Current System

The attractiveness score is combined with geometric PSL:
- **60% geometric PSL** (current MediaPipe-based metrics)
- **40% holistic attractiveness** (FaceStats CLIP+MLP)

This gives a balanced score that considers both:
- Facial structure (geometric measurements)
- Overall attractiveness (learned from data)

## Benefits

1. **More Robust**: CLIP handles pose/lighting variations better than geometric metrics alone
2. **Data-Driven**: MLP regressor learned from actual attractiveness ratings
3. **Modern**: Uses state-of-the-art vision-language model (CLIP)
4. **Continuous Score**: Outputs 0-100 instead of discrete categories

## Current Status

- ✅ Dependencies added (torch, transformers, scikit-learn)
- ✅ Placeholder function created
- ⏳ Need to clone FaceStats repo and load models
- ⏳ Need to implement full CLIP embedding extraction
- ⏳ Need to integrate MLP regressor prediction

## Next Steps

1. Clone FaceStats repository
2. Copy model files to backend
3. Implement full `calculate_attractiveness_score()` function
4. Test with sample images
5. Deploy to Render

## Notes

- Models stay on backend (no .mlmodel conversion needed)
- CLIP model will be downloaded on first use (~150MB)
- MLP regressor is lightweight (~few MB)
- Inference time: ~200-500ms per image (GPU recommended for production)

