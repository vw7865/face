# Free Storage Options for Beauty-Classifier Model (94 MB)

## ✅ Best Option: Use Free Cloud Storage + Download During Build

### Option 1: Google Drive (Recommended - Easiest)

**Steps:**

1. **Upload model to Google Drive:**
   - Upload `models/attractiveness_classifier.pt` to your Google Drive
   - Right-click → "Get link" → Set to "Anyone with the link"
   - Copy the shareable link

2. **Get Direct Download Link:**
   - Convert Google Drive link to direct download:
   - If link is: `https://drive.google.com/file/d/FILE_ID/view?usp=sharing`
   - Use: `https://drive.google.com/uc?export=download&id=FILE_ID`

3. **Update build.sh:**
   ```bash
   # Add to build.sh after model check section
   if [ ! -f "models/attractiveness_classifier.pt" ]; then
       echo "Downloading beauty-classifier model from Google Drive..."
       wget --no-check-certificate "https://drive.google.com/uc?export=download&id=YOUR_FILE_ID" -O models/attractiveness_classifier.pt
   fi
   ```

### Option 2: Dropbox (Also Free)

1. Upload model to Dropbox
2. Get shareable link
3. Convert to direct download: Change `www.dropbox.com` to `dl.dropboxusercontent.com` in the URL
4. Add to `build.sh`:
   ```bash
   if [ ! -f "models/attractiveness_classifier.pt" ]; then
       echo "Downloading beauty-classifier model from Dropbox..."
       wget "https://dl.dropboxusercontent.com/s/YOUR_FILE_ID/attractiveness_classifier.pt" -O models/attractiveness_classifier.pt
   fi
   ```

### Option 3: Use Render File System (After Deployment)

1. **Deploy code first** (without model)
2. **Use Render's file system:**
   - Render provides a persistent file system
   - You can upload files via:
     - Render Dashboard → Your Service → "Shell" tab
     - Or use `scp` if SSH is enabled
     - Or use Render's file upload feature

3. **Upload model manually:**
   ```bash
   # Via Render Shell (in dashboard)
   mkdir -p models
   # Then upload attractiveness_classifier.pt to models/
   ```

### Option 4: Use GitLab (Free Alternative)

GitLab offers **10 GB free LFS storage** (vs GitHub's 1 GB):

1. Create account at: https://gitlab.com
2. Create new repository
3. Push code there (GitLab has better free LFS limits)
4. Connect Render to GitLab instead of GitHub

### Option 5: Just Use FaceStats (Simplest!)

**The app works perfectly with just FaceStats!**

- ✅ FaceStats model is already in Git (0.8 MB)
- ✅ App will use FaceStats for attractiveness scoring
- ✅ Beauty-classifier is optional (for ensemble scoring)
- ✅ You can add it later if needed

**This is the easiest option - no setup needed!**

## Recommended: Google Drive + build.sh

I'll update your `build.sh` to download from Google Drive. You just need to:
1. Upload the model to Google Drive
2. Get the file ID
3. Update the FILE_ID in build.sh

Let me prepare the updated build.sh for you!

