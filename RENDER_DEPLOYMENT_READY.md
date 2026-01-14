# ✅ Ready for Render Deployment

## Status: Code is Ready!

All code changes have been committed. The deployment configuration is updated.

## ⚠️ Important: Model Files

The beauty-classifier model (94 MB) is too large for GitHub. You have **two options**:

### Option 1: Use Git LFS (Recommended for Production)

```bash
cd backend
git lfs install
git lfs track "models/*.pt"
git add .gitattributes
git add models/attractiveness_classifier.pt
git commit -m "Add model files via Git LFS"
git push origin main
```

### Option 2: Download Models on Render (Quick Fix)

Add to `build.sh` to download models from cloud storage (S3, Google Drive, etc.) during build.

## Deployment Steps

### 1. Go to Render Dashboard
- Visit: https://dashboard.render.com
- Sign in with GitHub

### 2. Create New Web Service
- Click **"New +"** → **"Web Service"**
- Connect your GitHub repository: `vw7865/face`
- Select branch: `main`
- **Root Directory**: `backend` ⚠️ **IMPORTANT**

### 3. Configure Settings

**Basic Settings:**
- **Name**: `looksmax-backend` (or your choice)
- **Environment**: `Python 3`
- **Region**: Choose closest to you

**Build & Deploy:**
- **Build Command**: `./build.sh` (already set in render.yaml)
- **Start Command**: `gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120 --workers 1`
- **Auto-Deploy**: `Yes`

**Advanced:**
- **Health Check Path**: `/health`

### 4. Environment Variables
No special env vars needed for basic deployment.

### 5. Deploy!

Click **"Create Web Service"** and wait for deployment.

## Verify Deployment

Once deployed, test:

```bash
# Health check
curl https://your-app-name.onrender.com/health

# Expected response:
{
  "status": "healthy",
  "mediapipe_installed": true,
  "deepface_available": true,
  "attractiveness_scoring_available": true,
  "beauty_classifier_available": true
}
```

## Model Files Status

**Current Status:**
- ✅ FaceStats model: `models/attractiveness_regressor.pt` (0.8 MB) - Can be in repo
- ⚠️ Beauty-classifier: `models/attractiveness_classifier.pt` (94 MB) - Too large for GitHub

**Solutions:**
1. Use Git LFS (best for production)
2. Download during build (add to build.sh)
3. Upload manually to Render file system (not recommended)

## Expected Behavior

After deployment, when you make a request:

1. **First Request** (Cold Start):
   - May take 30-60 seconds (Render free tier)
   - Models will load on first use
   - Subsequent requests will be faster

2. **Logs Will Show:**
   ```
   🎯 ATTRACTIVENESS SCORING - Starting ensemble prediction
   🔍 FaceStats: Starting model loading...
   ✅ FaceStats: Model found at ... (0.8 MB)
   ✅ FaceStats: Model loaded successfully
   ...
   🎯 Ensemble Result: XX.X (from 2 model(s))
   ```

## Troubleshooting

### Models Not Found
Check Render logs. If you see:
```
❌ FaceStats: Model not found in any location!
```

**Solution**: Ensure model files are available (Git LFS or download during build)

### Cold Start Timeout
- Already handled with 90-second timeout and retry logic
- First request may take 30-60 seconds on free tier

### Memory Issues
- Free tier: 512 MB RAM
- If OOM errors, upgrade to paid tier or optimize model loading

## Next Steps

1. ✅ Code is ready and pushed
2. ⏳ Deploy on Render (follow steps above)
3. ⏳ Test with iOS app
4. ⏳ Check logs for model loading status

**You're ready to deploy!** 🚀

