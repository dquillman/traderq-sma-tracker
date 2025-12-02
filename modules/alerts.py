"""
Alerts module for TraderQ
Handles alert checking, email notifications, and alert history management
"""

import pandas as pd
import json
from datetime import datetime, timedelta
from pathlib import Path


# Constants - will be updated when integrated with main app
SMA_SHORT = 20
SMA_LONG = 200
ALERT_HISTORY_FILE = Path(__file__).parent.parent / ".alert_history.json"


def send_email_alert(to_email: str, subject: str, body: str, config: dict) -> bool:
    """Send email alert using SMTP"""
    try:
        import smtplib
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart

        smtp_server = config.get("smtp_server", "smtp.gmail.com")
        smtp_port = config.get("smtp_port", 587)
        from_email = config.get("from_email", "")
        password = config.get("password", "")

        if not from_email or not password:
            return False

        msg = MIMEMultipart()
        msg["From"] = from_email
        msg["To"] = to_email
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(from_email, password)
        server.send_message(msg)
        server.quit()
        return True
    except Exception as e:
        print(f"Email send error: {e}")
        return False


def load_alert_history() -> list:
    """Load alert history from JSON file - will be replaced with Firestore"""
    if ALERT_HISTORY_FILE.exists():
        try:
            with open(ALERT_HISTORY_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_alert_history(history: list):
    """Save alert history to JSON file - will be replaced with Firestore"""
    try:
        with open(ALERT_HISTORY_FILE, "w") as f:
            json.dump(history, f, indent=2, default=str)
    except Exception:
        pass


def check_alerts(ticker: str, df: pd.DataFrame, alerts: list, send_email: bool = False,
                 load_email_config_func=None, _sma_func=None, _rsi_func=None) -> list:
    """
    Check if any alerts should be triggered. Returns list of triggered alerts.

    Args:
        ticker: Stock/crypto ticker symbol
        df: Price data DataFrame
        alerts: List of alert configurations
        send_email: Whether to send email notifications
        load_email_config_func: Function to load email config (passed to avoid circular import)
        _sma_func: SMA calculation function (from indicators module)
        _rsi_func: RSI calculation function (from indicators module)
    """
    if df.empty:
        return []

    triggered = []
    last = df.iloc[-1]
    price = float(last["close"])

    # Calculate indicators if needed
    rsi = None
    if any(a.get("type") in ["rsi_overbought", "rsi_oversold"] for a in alerts):
        if _rsi_func:
            rsi_series = _rsi_func(df["close"], window=14)
            rsi = float(rsi_series.iloc[-1]) if not pd.isna(rsi_series.iloc[-1]) else None

    # Load alert history to prevent duplicate notifications
    alert_history = load_alert_history()
    recent_alerts = {
        h.get("alert_id")
        for h in alert_history[-100:]
        if h.get("timestamp", "") > (datetime.now() - timedelta(hours=24)).isoformat()
    }

    for alert in alerts:
        if alert.get("ticker") != ticker or not alert.get("enabled", True):
            continue

        alert_type = alert.get("type")
        triggered_alert = None
        alert_id = f"{ticker}_{alert_type}_{alert.get('value', '')}"

        # Skip if already notified recently
        if alert_id in recent_alerts and not send_email:
            continue

        if alert_type == "golden_cross":
            if _sma_func:
                d = df.copy()
                d["SMA20"] = _sma_func(d["close"], SMA_SHORT)
                d["SMA200"] = _sma_func(d["close"], SMA_LONG)
                s_prev = d["SMA20"].shift(1)
                l_prev = d["SMA200"].shift(1)
                if len(d) > 1 and s_prev.iloc[-1] <= l_prev.iloc[-1] and d["SMA20"].iloc[-1] > d["SMA200"].iloc[-1]:
                    triggered_alert = {
                        "alert": alert,
                        "message": f"Golden Cross detected for {ticker} at ${price:.2f}"
                    }

        elif alert_type == "death_cross":
            if _sma_func:
                d = df.copy()
                d["SMA20"] = _sma_func(d["close"], SMA_SHORT)
                d["SMA200"] = _sma_func(d["close"], SMA_LONG)
                s_prev = d["SMA20"].shift(1)
                l_prev = d["SMA200"].shift(1)
                if len(d) > 1 and s_prev.iloc[-1] >= l_prev.iloc[-1] and d["SMA20"].iloc[-1] < d["SMA200"].iloc[-1]:
                    triggered_alert = {
                        "alert": alert,
                        "message": f"Death Cross detected for {ticker} at ${price:.2f}"
                    }

        elif alert_type == "price_above" and price >= alert.get("value", 0):
            triggered_alert = {
                "alert": alert,
                "message": f"{ticker} price ${price:.2f} is above ${alert.get('value', 0):.2f}"
            }

        elif alert_type == "price_below" and price <= alert.get("value", 0):
            triggered_alert = {
                "alert": alert,
                "message": f"{ticker} price ${price:.2f} is below ${alert.get('value', 0):.2f}"
            }

        elif alert_type == "rsi_oversold" and rsi is not None and rsi <= alert.get("value", 30):
            triggered_alert = {
                "alert": alert,
                "message": f"{ticker} RSI {rsi:.1f} is oversold (≤{alert.get('value', 30)})"
            }

        elif alert_type == "rsi_overbought" and rsi is not None and rsi >= alert.get("value", 70):
            triggered_alert = {
                "alert": alert,
                "message": f"{ticker} RSI {rsi:.1f} is overbought (≥{alert.get('value', 70)})"
            }

        if triggered_alert:
            triggered.append(triggered_alert)

            # Send email if configured
            if send_email and alert.get("email_enabled", False) and load_email_config_func:
                email_config = load_email_config_func()
                to_email = alert.get("email", email_config.get("default_email", ""))
                if to_email and email_config.get("from_email"):
                    subject = f"TraderQ Alert: {ticker}"
                    body = triggered_alert["message"]
                    if send_email_alert(to_email, subject, body, email_config):
                        # Log successful email
                        alert_history.append({
                            "alert_id": alert_id,
                            "ticker": ticker,
                            "type": alert_type,
                            "message": triggered_alert["message"],
                            "timestamp": datetime.now().isoformat(),
                            "email_sent": True
                        })
                        save_alert_history(alert_history)

            # Log alert to history
            alert_history.append({
                "alert_id": alert_id,
                "ticker": ticker,
                "type": alert_type,
                "message": triggered_alert["message"],
                "timestamp": datetime.now().isoformat(),
                "email_sent": send_email and alert.get("email_enabled", False)
            })
            save_alert_history(alert_history)

    return triggered
