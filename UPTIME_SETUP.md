# Keep Render Service Awake - UptimeRobot Setup

## Why This Is Needed
Render's free tier spins down services after ~15 minutes of inactivity. This causes the first request to take 30-60 seconds to wake up the service.

## Solution: UptimeRobot (Free)

UptimeRobot will ping your `/health` endpoint every 5 minutes to keep the service awake.

### Step-by-Step Setup:

1. **Sign up for UptimeRobot** (free):
   - Go to: https://uptimerobot.com/
   - Click "Sign Up" (free account allows 50 monitors)

2. **Add a New Monitor**:
   - Click "Add New Monitor"
   - Monitor Type: Select **"HTTP(s)"**
   - Friendly Name: `LooksmaxAI Backend Keep-Alive`
   - URL: `https://looksmax-backend.onrender.com/health`
   - Monitoring Interval: **5 minutes** (minimum for free tier)
   - Alert Contacts: Add your email (optional, for notifications)

3. **Save the Monitor**:
   - Click "Create Monitor"
   - Done! It will start pinging your service every 5 minutes

## Alternative: cron-job.org (Also Free)

If you prefer cron-job.org:

1. Go to: https://cron-job.org/
2. Sign up (free)
3. Create a new cron job:
   - URL: `https://looksmax-backend.onrender.com/health`
   - Schedule: Every 10 minutes (`*/10 * * * *`)
   - Save

## Important Settings

**Timeout Setting**: Make sure to set the monitor timeout to **60 seconds** (or maximum allowed). Render's free tier can take 30-60 seconds to wake up, so a shorter timeout will cause false "Down" alerts.

**Why It Shows "Down" Sometimes**:
- If Render spins down, the first ping might timeout (takes 30-60s to wake)
- The monitor might mark it as "Down" if timeout is too short
- This is normal for free tier - the service will wake up on the next ping
- The important thing is it keeps pinging every 5 minutes to prevent long spin-downs

## Result

Your Render service will stay awake 24/7, eliminating the 30-60 second cold start delay!

Even if UptimeRobot shows occasional "Down" status (due to wake-up delays), the constant pinging prevents the service from staying spun down for long periods.

## Note

The free tier of UptimeRobot checks every 5 minutes, which is perfect - it's more frequent than Render's 15-minute spin-down threshold.

