# Service Down - Troubleshooting Guide

## Why UptimeRobot Can't Fix This

**UptimeRobot only prevents spin-downs, it can't fix:**
- ❌ Failed deployments
- ❌ Crashed services
- ❌ Build errors
- ❌ Application errors

**What UptimeRobot DOES:**
- ✅ Pings your service every 5 minutes
- ✅ Keeps service awake (prevents 15-min spin-down)
- ✅ Alerts you when service is down

## Current Issue: Service Completely Down

If `https://looksmax-backend.onrender.com/health` returns nothing, the service is likely:
1. **Deployment failed** - Check Render dashboard
2. **Service crashed** - Check Render logs
3. **Build error** - Check build logs

## How to Fix

### Step 1: Check Render Dashboard

1. Go to: https://dashboard.render.com
2. Click on `looksmax-backend` service
3. Check **"Events"** or **"Deployments"** tab
4. Look for latest deployment status:
   - ✅ **"Live"** = Service is running
   - ⚠️ **"Failed"** = Deployment failed (see logs)
   - 🔄 **"Building"** = Still deploying (wait)

### Step 2: Check Logs

1. In Render dashboard, click **"Logs"** tab
2. Look for errors:
   - Build errors (during installation)
   - Runtime errors (after deployment)
   - Port binding errors

### Step 3: Manual Redeploy

If deployment failed:

1. In Render dashboard, click **"Manual Deploy"**
2. Select **"Deploy latest commit"**
3. Click **"Deploy"**
4. Wait for build to complete (5-10 minutes)

### Step 4: Verify Service is Up

After deployment, test:
```bash
curl https://looksmax-backend.onrender.com/
curl https://looksmax-backend.onrender.com/health
```

Expected response:
```json
{"status": "ok", "message": "LooksMax AI Backend is running"}
```

## Common Issues

### Issue 1: Port Binding Timeout
**Symptom**: "Port scan timeout reached"
**Fix**: Already fixed with lazy imports - should work now

### Issue 2: Build Timeout
**Symptom**: Build takes too long
**Fix**: This is normal (5-10 min for large packages), just wait

### Issue 3: Service Crashes on Startup
**Symptom**: Service starts then immediately crashes
**Fix**: Check logs for Python errors, missing dependencies

### Issue 4: Service Spins Down
**Symptom**: Works but takes 30-60s on first request
**Fix**: Set up UptimeRobot (see UPTIME_SETUP.md)

## After Service is Fixed

Once service is back up, make sure UptimeRobot is configured:

1. **UptimeRobot Settings**:
   - URL: `https://looksmax-backend.onrender.com/health`
   - Interval: **5 minutes**
   - Timeout: **60 seconds** (important for cold starts)

2. **Why 60-second timeout?**
   - Render free tier can take 30-60s to wake up
   - Shorter timeout = false "Down" alerts
   - 60s timeout = accurate monitoring

## Quick Check Commands

```bash
# Check if service responds
curl https://looksmax-backend.onrender.com/

# Check health endpoint
curl https://looksmax-backend.onrender.com/health

# Check with timeout (60s for cold start)
curl --max-time 60 https://looksmax-backend.onrender.com/health
```

## Summary

**Right now**: Service is down - check Render dashboard for deployment status
**After fix**: Configure UptimeRobot to keep it awake
**Long-term**: Consider upgrading to Render paid tier for always-on service

