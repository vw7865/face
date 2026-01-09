# Setup Google Drive Download for Beauty-Classifier Model

## Quick Setup Guide

### Step 1: Upload Model to Google Drive

1. Go to: https://drive.google.com
2. Upload `models/attractiveness_classifier.pt` (94 MB)
3. Right-click the file → "Get link"
4. Set sharing to "Anyone with the link"
5. Copy the link

### Step 2: Get File ID

Your Google Drive link looks like:
```
https://drive.google.com/file/d/FILE_ID_HERE/view?usp=sharing
```

Extract the `FILE_ID_HERE` part (the long string between `/d/` and `/view`).

### Step 3: Update build.sh

1. Open `build.sh`
2. Find the line: `# GOOGLE_DRIVE_FILE_ID="YOUR_FILE_ID_HERE"`
3. Replace `YOUR_FILE_ID_HERE` with your actual file ID
4. Uncomment the Google Drive download section (remove `#`)

Example:
```bash
GOOGLE_DRIVE_FILE_ID="1ABC123xyz789..."  # Your actual file ID
if [ -n "$GOOGLE_DRIVE_FILE_ID" ]; then
    echo "Downloading from Google Drive..."
    wget --no-check-certificate "https://drive.google.com/uc?export=download&id=$GOOGLE_DRIVE_FILE_ID" -O models/attractiveness_classifier.pt || echo "Download failed"
fi
```

### Step 4: Commit and Push

```bash
git add build.sh
git commit -m "Add Google Drive download for beauty-classifier model"
git push origin main
```

### Step 5: Deploy

Render will automatically:
- Run `build.sh` during deployment
- Download the model from Google Drive
- Make it available in `models/` directory

## Alternative: Use Environment Variable

For better security, you can use Render's environment variables:

1. In Render dashboard → Your Service → Environment
2. Add variable: `GOOGLE_DRIVE_FILE_ID` = `your_file_id`
3. Update `build.sh` to use: `GOOGLE_DRIVE_FILE_ID="${GOOGLE_DRIVE_FILE_ID:-}"`

This way the file ID isn't in your code!

## Testing

After deployment, check logs for:
```
✅ Beauty-classifier model downloaded successfully
```

Or if download fails:
```
⚠️  Beauty-classifier model not available
   App will work with FaceStats only
```

## That's It!

The model will download automatically during Render builds. No GitHub LFS needed! 🚀

