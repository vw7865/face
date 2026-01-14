# Git Push Troubleshooting - HTTP 400 Error

## Issue
LFS objects upload successfully (100%), but final push fails with HTTP 400.

## Possible Causes

### 1. GitHub LFS Quota Exceeded
- Free tier: 1 GB storage + 1 GB bandwidth/month
- Your files: ~99 MB (within limits, but check cumulative usage)

**Check quota:**
- Go to: https://github.com/settings/billing
- Look for "Git LFS Data" section

### 2. Repository Size Limits
- GitHub has soft limits on repository size
- Large commits can trigger HTTP 400

### 3. Network/Timeout Issues
- Large file transfers can timeout
- GitHub API rate limiting

## Solutions

### Solution 1: Push LFS Files Separately
```bash
# Push LFS objects first
git lfs push origin main --all

# Then push commits
git push origin main
```

### Solution 2: Increase Git Buffer
```bash
git config http.postBuffer 524288000  # 500 MB
git config http.maxRequestBuffer 100M
git push origin main
```

### Solution 3: Push in Smaller Chunks
```bash
# Push without LFS first (if possible)
# Or push commits one at a time
git push origin main --force-with-lease
```

### Solution 4: Use SSH Instead of HTTPS
```bash
# Change remote to SSH
git remote set-url origin git@github.com:vw7865/face.git
git push origin main
```

### Solution 5: Check for Large Non-LFS Files
```bash
# Find large files in commit
git diff --stat HEAD~5 HEAD
git ls-tree -r -l HEAD | sort -k4 -n -r | head -10
```

### Solution 6: Alternative - Skip LFS for Now
If LFS continues to fail, you can:
1. Remove LFS tracking temporarily
2. Use cloud storage (S3, Google Drive) for models
3. Download models during Render build

## Recommended Next Steps

1. **Check GitHub LFS Quota**: https://github.com/settings/billing
2. **Try SSH push**: More reliable for large files
3. **Increase buffer**: `git config http.postBuffer 524288000`
4. **Retry push**: `git push origin main`

## For Render Deployment

**Good news**: Even if GitHub push fails, you can still deploy:

1. **Option A**: Models in cloud storage
   - Upload models to S3/Google Drive
   - Download in `build.sh` during Render build

2. **Option B**: Manual upload to Render
   - Deploy code first (without models)
   - Use Render's file system/SSH to upload models

3. **Option C**: Use smaller models
   - FaceStats model (0.8 MB) can be in regular Git
   - Only beauty-classifier (94 MB) needs LFS

The app will work with just FaceStats model if beauty-classifier isn't available!

