# Deployment Guide

## Backend Deployment

### Option 1: Heroku (Recommended for Quick Start)

1. **Install Heroku CLI**:
   ```bash
   brew install heroku/brew/heroku  # macOS
   ```

2. **Login to Heroku**:
   ```bash
   heroku login
   ```

3. **Create Heroku App**:
   ```bash
   cd backend
   heroku create your-app-name
   ```

4. **Deploy**:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git push heroku main
   ```

5. **Get Your URL**:
   ```bash
   heroku info
   ```
   Your backend URL will be: `https://your-app-name.herokuapp.com`

### Option 2: Docker Deployment

1. **Build Docker Image**:
   ```bash
   cd backend
   docker build -t looksmax-backend .
   ```

2. **Run Locally**:
   ```bash
   docker run -p 5000:5000 looksmax-backend
   ```

3. **Deploy to Cloud**:
   - **AWS**: Use ECS or Elastic Beanstalk
   - **Google Cloud**: Use Cloud Run
   - **Azure**: Use Container Instances

### Option 3: Python Anywhere / Railway / Render

1. **Railway**:
   - Connect your GitHub repo
   - Railway auto-detects Python
   - Add environment variables if needed

2. **Render**:
   - Create new Web Service
   - Connect GitHub repo
   - Set build command: `pip install -r requirements.txt`
   - Set start command: `gunicorn app:app`

## iOS App Configuration

### Update Backend URL

1. Open `FaceAnalysisService.swift`
2. Update the `baseURL`:
   ```swift
   private let baseURL = "https://your-app-name.herokuapp.com"
   ```

### For Local Testing

If testing with local backend:

1. **Backend**: Run `python app.py` (runs on `http://localhost:5000`)
2. **iOS Simulator**: Use `http://localhost:5000`
3. **Physical Device**: 
   - Find your Mac's IP: `ifconfig | grep "inet "`
   - Use `http://YOUR_IP:5000`
   - Update `Info.plist` to allow HTTP (see below)

### Allow HTTP (Development Only)

Add to `Info.plist`:
```xml
<key>NSAppTransportSecurity</key>
<dict>
    <key>NSAllowsArbitraryLoads</key>
    <true/>
</dict>
```

**⚠️ Warning**: Only use this for development. Remove for production!

## Testing

### Test Backend Locally

```bash
cd backend
python app.py
```

Test with curl:
```bash
curl -X POST http://localhost:5000/api/analyze-face \
  -F "front_image=@path/to/front.jpg" \
  -F "side_image=@path/to/side.jpg" \
  -F "gender=Male"
```

### Test Health Endpoint

```bash
curl http://localhost:5000/health
```

## Environment Variables

### Backend

- `PORT`: Server port (default: 5000)
- `FLASK_ENV`: `production` or `development`

### iOS

No environment variables needed. Just update the `baseURL` in `FaceAnalysisService.swift`.

## Troubleshooting

### Backend Issues

1. **MediaPipe not installing**:
   ```bash
   pip install --upgrade pip
   pip install mediapipe
   ```

2. **Port already in use**:
   ```bash
   # Find process using port 5000
   lsof -i :5000
   # Kill it
   kill -9 <PID>
   ```

3. **CORS errors**:
   - Ensure `flask-cors` is installed
   - Check CORS is enabled in `app.py`

### iOS Issues

1. **Network errors**:
   - Check backend URL is correct
   - Verify backend is running
   - Check device/simulator can reach backend

2. **Image upload fails**:
   - Check image size (backend may have limits)
   - Verify multipart form data format

3. **Date parsing errors**:
   - Check date format from backend matches iOS expectations
   - Verify ISO8601 formatting

## Production Checklist

- [ ] Backend deployed and accessible
- [ ] Backend URL updated in iOS app
- [ ] HTTPS enabled (required for App Store)
- [ ] Error handling tested
- [ ] Loading states working
- [ ] Subscription/unlock flow tested
- [ ] Result history working
- [ ] Privacy policy updated
- [ ] Terms of service updated

## Next Steps

1. **Add Authentication**: Protect your API with API keys or user authentication
2. **Add Rate Limiting**: Prevent abuse
3. **Add Logging**: Monitor usage and errors
4. **Add Analytics**: Track usage patterns
5. **Optimize Performance**: Cache results, optimize MediaPipe processing







