# Alerts Setup Guide

## Current Alert System (In-App Only)

The app currently shows alerts **only when you're viewing the ticker** in the Tracker tab. Alerts are checked in real-time as you view charts.

### How to Set Up Alerts

1. **Navigate to the "🔔 Alerts" tab**
2. **Fill in the alert form:**
   - **Ticker**: Enter the symbol (e.g., "SPY", "AAPL", "BTC-USD")
   - **Alert Type**: Choose from:
     - **Golden Cross**: Alert when SMA20 crosses above SMA200 (bullish signal)
     - **Death Cross**: Alert when SMA20 crosses below SMA200 (bearish signal)
     - **Price Above**: Alert when price exceeds your threshold
     - **Price Below**: Alert when price falls below your threshold
     - **RSI Overbought**: Alert when RSI reaches your threshold (default: ≥70)
     - **RSI Oversold**: Alert when RSI reaches your threshold (default: ≤30)
   - **Threshold**: For price/RSI alerts, set your desired value
3. **Click "Add Alert"**

### Viewing Triggered Alerts

- Alerts appear as **warning messages** in the right sidebar when viewing that ticker in the Tracker tab
- You must be actively viewing the app for alerts to show

## Future: Text/Email Alerts (Not Yet Implemented)

To receive alerts via **SMS or Email**, the following would need to be added:

### Option 1: Email Alerts (Recommended)
- Uses SMTP (Gmail, Outlook, etc.)
- Requires email credentials in config
- Can send alerts when app runs in background

### Option 2: SMS Alerts
- Uses services like Twilio, AWS SNS, or Pushover
- Requires API keys and may have costs
- More immediate than email

### Option 3: Push Notifications
- Browser push notifications
- Requires user permission
- Works when browser tab is open

### Option 4: Desktop Notifications
- System tray notifications
- Requires additional libraries
- Works when app is running

## Recommended Implementation

For a production system, I'd recommend:
1. **Email alerts** (easiest, most reliable)
2. **Background scheduler** (check alerts periodically even when not viewing)
3. **Alert history** (log when alerts were triggered)
4. **One-time vs. recurring** (some alerts should only fire once)

Would you like me to implement email/SMS alerts? I can add:
- Email configuration in settings
- Background alert checking
- Alert history logging
- Notification preferences per alert

