# Deploy to Render - Step by Step Guide

## Prerequisites

✅ All code is ready
✅ Models are in `models/` folder:
   - `attractiveness_regressor.pt` (0.8 MB) - FaceStats
   - `attractiveness_classifier.pt` (94 MB) - Beauty-classifier

## Important: Model Files

The beauty-classifier model (94 MB) is too large for GitHub. You have two options:

### Option 1: Use Git LFS (Recommended)
```bash
cd backend
git lfs install
git lfs track "models/*.pt"
git add .gitattributes models/attractiveness_classifier.pt
git commit -m "Add model files via Git LFS"
git push origin main
```

### Option 2: Download Models During Build
Add to `build.sh` to download models from a cloud storage (S3, Google Drive, etc.)

## Deployment Steps

### 1. Push Code to GitHub
```bash
cd backend
git add .
git commit -m "Ready for Render deployment"
git push origin main
```

### 2. Deploy on Render

1. **Go to Render Dashboard**: https://dashboard.render.com
2. **Create New Web Service**:
   - Connect your GitHub repository
   - Select the repository: `vw7865/face` (or your repo name)
   - Choose branch: `main`
   - Root directory: `backend`

3. **Configure Build Settings**:
   - **Build Command**: `./build.sh` (or `pip install -r requirements.txt`)
   - **Start Command**: `gunicorn app:app --bind 0.0.0.0:$PORT`
   - **Environment**: Python 3
   - **Python Version**: 3.11.9 (specified in `runtime.txt`)

4. **Environment Variables** (if needed):
   - No special env vars required for basic deployment

5. **Advanced Settings**:
   - **Auto-Deploy**: Yes (deploys on every push)
   - **Health Check Path**: `/health`

### 3. Verify Deployment

Once deployed, test the endpoints:

```bash
# Health check
curl https://your-app-name.onrender.com/health

# Should return:
# {
#   "status": "healthy",
#   "mediapipe_installed": true,
#   "deepface_available": true,
#   "attractiveness_scoring_available": true,
#   "beauty_classifier_available": true
# }
```

## Model Files on Render

Since model files are large, ensure they're available:

1. **If using Git LFS**: Render will automatically pull LFS files
2. **If models are in repo**: They'll be included in the build
3. **If models need download**: Add download step to `build.sh`

## Troubleshooting

### Models Not Found
Check Render logs for:
```
🔍 FaceStats: Checking 4 possible model locations...
❌ FaceStats: Model not found in any location!
```

**Solution**: Ensure model files are in the repository or add download step to `build.sh`

### Cold Start Timeout
Render free tier has cold starts (30-60 seconds). The code already handles this with:
- 90-second timeout
- Retry logic
- Health check endpoint

### Memory Issues
If you get out-of-memory errors:
- Upgrade to paid tier (more RAM)
- Or use smaller models
- Or load models lazily (already implemented)

## Expected Logs After Deployment

When you make a request, you should see:
```
🎯 ATTRACTIVENESS SCORING - Starting ensemble prediction
🔍 FaceStats: Starting model loading...
✅ FaceStats: Model found at ... (0.8 MB)
✅ FaceStats: Model loaded successfully
🔮 FaceStats: Running prediction...
✅ FaceStats: Final score = XX.X
...
🎯 Ensemble Result: XX.X (from 2 model(s))
```

## Next Steps

1. Deploy to Render
2. Test with your iOS app
3. Check logs for model loading status
4. Share logs if models don't load - we can debug further

Good luck! 🚀

