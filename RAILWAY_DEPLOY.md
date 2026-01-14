# Railway Deployment Guide

## Quick Setup

1. **Sign up/Login to Railway**
   - Go to [railway.app](https://railway.app)
   - Sign up with GitHub

2. **Create New Project**
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Choose your `vw7865/face` repository

3. **Configure Service**
   - Railway will auto-detect Python
   - Root Directory: `backend` (if your backend code is in a subfolder)
   - Or leave empty if backend files are in root

4. **Environment Variables** (Optional)
   - `PORT` - Railway sets this automatically
   - No other vars needed for now

5. **Deploy**
   - Railway will automatically build and deploy
   - Watch the build logs to ensure MediaPipe installs

## Build Settings

Railway will use:
- **Build Command**: Automatically detected (installs from requirements.txt)
- **Start Command**: `gunicorn app:app --bind 0.0.0.0:$PORT`

## Verify Deployment

1. Railway will give you a URL like: `https://your-app-name.up.railway.app`
2. Test health endpoint:
   ```bash
   curl https://your-app-name.up.railway.app/health
   ```
3. Should see: `{"mediapipe_installed": true, ...}`

## Update iOS App

Once deployed, update `FaceAnalysisService.swift`:
```swift
private let baseURL = "https://your-app-name.up.railway.app"
```

## Advantages of Railway

- ✅ Better MediaPipe support
- ✅ Automatic HTTPS
- ✅ Free tier available
- ✅ Easy GitHub integration
- ✅ Better build logs
- ✅ No cold starts on paid plans

## Troubleshooting

If MediaPipe still doesn't install:
1. Check Railway build logs
2. Try updating requirements.txt to use latest MediaPipe
3. Railway's Nixpacks builder handles native deps better than Render


