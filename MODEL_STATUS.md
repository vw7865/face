# Model Files Status

## ✅ FaceStats Model - FOUND

The FaceStats attractiveness model is available in multiple locations:

- ✅ `models/attractiveness_regressor.pt` (0.8 MB) - **Primary location**
- ✅ `facestats/models/attractiveness_regressor.pt` (0.8 MB)
- ✅ `facestats/src/models/attractiveness_regressor.pt` (0.6 MB)

**Status**: Ready to use! The code will check `models/` folder first.

## ❌ Beauty-Classifier Model - NOT FOUND

The beauty-classifier model file is not present:

- ❌ `models/attractiveness_classifier.pt` - NOT FOUND
- ❌ `beauty-classifier/models/attractiveness_classifier.pt` - NOT FOUND
- ✅ `beauty-classifier/models/attractiveness_classifier.pt.dvc` - EXISTS (DVC metadata only)

**Status**: Model file needs to be downloaded.

### How to Download Beauty-Classifier Model

**Option 1: Using DVC (Recommended)**
```bash
cd backend/beauty-classifier
dvc pull models/attractiveness_classifier.pt.dvc
```

**Option 2: Manual Download**
1. The model file is ~98.5 MB
2. Download from DVC remote storage
3. Place at: `backend/models/attractiveness_classifier.pt` or `backend/beauty-classifier/models/attractiveness_classifier.pt`

**Note**: The beauty-classifier integration will gracefully return `None` if the model is not found, and the system will fall back to using only FaceStats for attractiveness scoring.

## Current Integration Status

- ✅ **FaceStats**: Fully integrated and ready (model file present)
- ⚠️ **Beauty-Classifier**: Code integrated, but model file missing (will fall back to FaceStats only)

## Ensemble Scoring

When both models are available:
- Both scores are averaged: `(facestats_score + beauty_score) / 2`
- Then combined with geometric PSL: `0.6 * geometric_psl + 0.4 * attractiveness_score`

When only FaceStats is available:
- Uses FaceStats score directly
- Combined with geometric PSL: `0.6 * geometric_psl + 0.4 * facestats_score`

The system will work with just FaceStats, but having both models provides more robust and stable scoring.

