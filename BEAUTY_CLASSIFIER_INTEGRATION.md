# Beauty-Classifier Integration Guide

## Overview

This document outlines how to integrate the beauty-classifier model from [okurki/beauty-classifier](https://github.com/okurki/beauty-classifier) as a secondary attractiveness scorer for calibration and stability.

## Model Details

- **Architecture**: ResNet-50 CNN
- **Training Dataset**: SCUT-FBP5500 (5,500 frontal faces with human-rated attractiveness scores)
- **Output Scale**: 1-5 (continuous)
- **Purpose**: Secondary/calibration scorer to cross-check with FaceStats
- **Integration Effort**: Low (just load model and predict)

## SCUT-FBP5500 Dataset

- **Size**: 5,500 frontal faces (350×350px)
- **Diversity**: 
  - 2,000 Asian females
  - 2,000 Asian males
  - 750 Caucasian females
  - 750 Caucasian males
- **Labels**: Attractiveness scores (1.0-5.0) from human ratings
- **Gold Standard**: Well-established benchmark dataset for facial attractiveness

## Integration Steps

### 1. Clone Beauty-Classifier Repository

```bash
cd backend
git clone https://github.com/okurki/beauty-classifier.git beauty-classifier
```

### 2. Download Pre-trained Model

The beauty-classifier repo should have:
- `models/` directory with trained ResNet-50 model
- Model file typically named something like `beauty_classifier_resnet50.pt` or similar

### 3. Implement Scoring Function

The `calculate_beauty_classifier_score()` function in `app.py` needs to:

1. **Load Pre-trained ResNet-50 Model**:
```python
import torch
import torchvision.models as models
import torchvision.transforms as transforms

# Load the model
model = torch.load("beauty-classifier/models/beauty_classifier_resnet50.pt", map_location='cpu')
model.eval()
```

2. **Define Image Preprocessing**:
```python
# Match SCUT-FBP5500 preprocessing (350x350, ImageNet normalization)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((350, 350)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

3. **Preprocess and Predict**:
```python
# Convert BGR to RGB
rgb_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

# Apply transforms
image_tensor = transform(rgb_image)

# Predict (1-5 scale)
with torch.no_grad():
    prediction = model(image_tensor.unsqueeze(0))
    score_5 = prediction.item()  # 1-5 scale
```

4. **Convert to 0-100 Scale**:
```python
# Convert 1-5 scale to 0-100 scale
score_100 = (score_5 - 1) / 4 * 100
score_100 = float(np.clip(score_100, 0.0, 100.0))
return score_100
```

## Combined Scoring Strategy

The system now uses **multiple attractiveness models** for stability:

1. **FaceStats (CLIP + MLP)**: Modern, robust to pose/lighting
2. **Beauty-classifier (ResNet-50)**: Anchored in gold-standard dataset
3. **Geometric PSL**: MediaPipe-based facial measurements

### Final PSL Calculation

```
if attractiveness_scores_available:
    avg_attractiveness = average(faceStats_score, beauty_classifier_score)
    psl = 0.6 * geometric_psl + 0.4 * avg_attractiveness
else:
    psl = geometric_psl
```

### Benefits of Multi-Model Approach

1. **Stability**: Averaging reduces variance and outliers
2. **Calibration**: Beauty-classifier provides gold-standard anchor
3. **Robustness**: Different models catch different aspects
4. **Reliability**: If one model fails, others can still provide scores

## Model Loading (Lazy Loading)

For production, implement lazy loading to avoid startup delays:

```python
_beauty_model = None
_beauty_transform = None

def get_beauty_model():
    global _beauty_model, _beauty_transform
    if _beauty_model is None:
        _beauty_model = torch.load("models/beauty_classifier_resnet50.pt", map_location='cpu')
        _beauty_model.eval()
        _beauty_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((350, 350)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    return _beauty_model, _beauty_transform
```

## Current Status

- ✅ Dependencies added (torch, torchvision)
- ✅ Placeholder function created
- ✅ Integration with FaceStats averaging logic
- ⏳ Need to clone beauty-classifier repo
- ⏳ Need to locate/download model file
- ⏳ Need to implement full prediction pipeline

## Next Steps

1. Clone beauty-classifier repository
2. Locate trained model file
3. Implement full `calculate_beauty_classifier_score()` function
4. Test with sample images
5. Deploy to Render

## Notes

- Model stays on backend (no .mlmodel conversion needed)
- ResNet-50 is lightweight (~100MB)
- Inference time: ~50-100ms per image (CPU), ~10-20ms (GPU)
- 1-5 scale conversion: `(score - 1) / 4 * 100` maps to 0-100
- Works alongside FaceStats for ensemble scoring

