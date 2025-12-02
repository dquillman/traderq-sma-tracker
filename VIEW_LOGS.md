# How to View Streamlit Cloud Logs

## Streamlit Cloud Logs

The logs in Streamlit Cloud **automatically refresh** when you:
1. Push new code to GitHub
2. The app redeploys
3. Streamlit Cloud restarts the app

## To See Fresh Logs:

1. **Wait for new deployment**
   - Go to: https://share.streamlit.io
   - Click your app: `traderq-sma-tracker`
   - Look for "🔄 Updated app!" messages in the logs
   - New logs will appear at the bottom

2. **Filter/Scroll**
   - Scroll down to see the most recent logs
   - The newest logs are at the bottom
   - Look for timestamps like `[HH:MM:SS]`

3. **What to Look For:**
   - `✅` = Success messages
   - `❗` or `❌` = Error messages
   - `🔄 Updated app!` = New deployment
   - Python tracebacks = Error details

## If Logs Look Old:

The logs are **append-only** - old messages stay but new ones are added. To see only recent activity:

1. Scroll to the bottom of the log panel
2. Look for the latest timestamp
3. Focus on entries after "🔄 Updated app!"

## Alternative: Check App Status

Instead of logs, you can also:
- Visit your app URL: https://traderq-sma-tracker.streamlit.app
- See if it loads or shows an error message
- The app itself will show errors if it gets past startup

