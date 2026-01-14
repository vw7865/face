# Railway Migration Guide - LooksMax AI Backend

## Quick Start (5 minutes)

### Step 1: Sign Up for Railway
1. Go to [railway.app](https://railway.app)
2. Click "Start a New Project"
3. Sign up with GitHub (recommended) or email
4. **You get $5 free credits for 30 days** - no credit card needed!

### Step 2: Deploy from GitHub
1. Click "New Project" → "Deploy from GitHub repo"
2. Select your repository: `vw7865/face`
3. Railway will auto-detect it's a Python app
4. **Important**: Set **Root Directory** to: `backend`
   - Click on the service → Settings → Root Directory → Enter `backend`

### Step 3: Configure Resources (Hobby Plan)
1. Go to your service → Settings → Resources
2. Set:
   - **RAM**: 2GB (minimum for ML models, can go up to 8GB)
   - **CPU**: 2 vCPU (recommended for faster model loading)
3. Railway will auto-scale if needed

### Step 4: Deploy
1. Railway will automatically:
   - Detect Python
   - Install dependencies from `requirements.txt`
   - Run build script if needed
   - Start the app with gunicorn
2. Watch the build logs (should take 10-15 minutes)
3. Railway will give you a URL: `https://your-app-name.up.railway.app`

### Step 5: Test
```bash
# Test root endpoint
curl https://your-app-name.up.railway.app/

# Test health endpoint
curl https://your-app-name.up.railway.app/health
```

## Configuration Files

Railway will use:
- **railway.json** - Deployment configuration (already set up)
- **requirements.txt** - Python dependencies
- **runtime.txt** - Python version (3.11.9)
- **build.sh** - Build script (optional, Railway auto-detects)

## Environment Variables

Railway automatically sets:
- `PORT` - Don't set this manually, Railway handles it

No other environment variables needed for basic deployment.

## Resource Settings (Hobby Plan - $5/month)

**Recommended for your ML app:**
- **RAM**: 2-4GB (your models need ~2GB)
- **CPU**: 2-4 vCPU (faster model loading)
- **Cost**: ~$5-10/month (often just $5 with included credits)

**How to set:**
1. Service → Settings → Resources
2. Adjust RAM/CPU sliders
3. Railway charges per-second, so you only pay for what you use

## Advantages Over Render

✅ **No spin-down** - Service stays awake 24/7  
✅ **Faster builds** - 10-15 min vs 20-30 min  
✅ **Better resources** - More CPU/RAM available  
✅ **Pay-per-second** - Only pay for actual usage  
✅ **Cheaper** - $5/month vs $19/month (Render Professional)  
✅ **No cold starts** - Consistent 5-15s response times  

## Update iOS App

Once deployed, update `FaceAnalysisService.swift`:

```swift
private let baseURL = "https://your-app-name.up.railway.app"
```

Replace `your-app-name` with your actual Railway domain.

## Monitoring

1. **Logs**: Service → Deployments → Click deployment → View logs
2. **Metrics**: Service → Metrics tab (CPU, RAM, Network)
3. **Health**: Service → Settings → Health Check (set to `/`)

## Troubleshooting

### Build Fails
- Check build logs in Railway dashboard
- Ensure `backend/` is set as root directory
- Verify `requirements.txt` is correct

### Models Not Loading
- Check that `models/` directory exists in Git
- Verify model files are committed (FaceStats model should be there)
- Check logs for model loading errors

### Out of Memory
- Increase RAM in Settings → Resources (up to 8GB on Hobby plan)
- Models need ~2GB minimum

### Slow First Request
- Normal! First request loads models (30-60s)
- Subsequent requests are fast (5-15s)
- Models preload in background thread

## Cost Estimate

**Hobby Plan ($5/month minimum):**
- Includes $5 usage credits
- 2GB RAM + 2 vCPU, 24/7: ~$3-5/month usage
- **Total: ~$5-10/month** (often just $5 if usage stays under credits)

**First month:**
- Free trial: $5 credits (30 days)
- **Total: $0** for first month!

## Next Steps

1. ✅ Deploy to Railway
2. ✅ Test endpoints
3. ✅ Update iOS app with new URL
4. ✅ Monitor performance
5. ✅ Enjoy no cold starts! 🎉

## Support

- Railway Docs: https://docs.railway.app
- Railway Discord: https://discord.gg/railway
- Your build logs will show any errors

