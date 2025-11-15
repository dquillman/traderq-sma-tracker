# TraderQ — SMA 20/200 Tracker (Stocks + Crypto)
# v1.4.8
# Single-file Streamlit app with clean SMA logic, pretouch screener, cross markers,
# crypto fallback, and trend chips (Bullish/Bearish) without emoji.
from __future__ import annotations

import json
import math
import time
from datetime import date, datetime, timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def add_cross_markers(fig: go.Figure, df: pd.DataFrame,
                      price_col: str = "Close",
                      s20: str = "SMA20",
                      s200: str = "SMA200",
                      row: int = 1, col: int = 1) -> None:
    """
    Golden cross: 20 crosses above 200 (green triangle-up)
    Death  cross: 20 crosses below 200 (red triangle-down)
    Works if df has df[price_col], df[s20], df[s200].
    """
    if df is None or df.empty:
        return
    needed = (price_col, s20, s200)
    if any(c not in df.columns for c in needed):
        return
    s_prev = df[s20].shift(1)
    l_prev = df[s200].shift(1)
    cross_up = (s_prev <= l_prev) & (df[s20] > df[s200])
    cross_dn = (s_prev >= l_prev) & (df[s20] < df[s200])
    xu, yu = df.index[cross_up], df.loc[cross_up, price_col]
    xd, yd = df.index[cross_dn], df.loc[cross_dn, price_col]
    if len(xu):
        fig.add_trace(go.Scatter(
            x=xu, y=yu, mode="markers", name="Golden Cross",
            marker=dict(symbol="triangle-up", size=25, color="#17c964",
                        line=dict(width=1, color="#0b3820")),
            hovertemplate="Golden: %{x|%Y-%m-%d}<br>Price: %{y:.2f}<extra></extra>",
            showlegend=True
        ), row=row, col=col)
    if len(xd):
        fig.add_trace(go.Scatter(
            x=xd, y=yd, mode="markers", name="Death Cross",
            marker=dict(symbol="triangle-down", size=25, color="#f31260",
                        line=dict(width=1, color="#4a0b19")),
            hovertemplate="Death: %{x|%Y-%m-%d}<br>Price: %{y:.2f}<extra></extra>",
            showlegend=True
        ), row=row, col=col)


import streamlit as st
import ui_glow_patch
import yf_patch  # glow+session patch

APP_VERSION = "v2.0.0"
# v2.0.0 – email alerts, multi-timeframe, support/resistance, advanced screener, trade journal, patterns, news, correlation

# --- Settings / Defaults ---
DEFAULT_STOCKS = ["^GSPC", "^DJI", "^IXIC", "SPY", "QQQ"]
DEFAULT_CRYPTOS = ["BTC-USD", "ETH-USD", "SOL-USD"]
DEFAULT_PERIOD_DAYS = 365 * 2  # 2 years for stocks; crypto API may cap to 365 in fallback
SMA_SHORT = 20
SMA_LONG = 200

# --- File paths for persistence ---
CUSTOM_TICKERS_FILE = Path(__file__).parent / ".custom_tickers.json"
ALERTS_FILE = Path(__file__).parent / ".alerts.json"
PORTFOLIO_FILE = Path(__file__).parent / ".portfolio.json"
CROSS_HISTORY_FILE = Path(__file__).parent / ".cross_history.json"
WATCHLISTS_FILE = Path(__file__).parent / ".watchlists.json"
TRADE_JOURNAL_FILE = Path(__file__).parent / ".trade_journal.json"
ALERT_HISTORY_FILE = Path(__file__).parent / ".alert_history.json"
EMAIL_CONFIG_FILE = Path(__file__).parent / ".email_config.json"

def load_custom_tickers() -> dict:
    """Load custom tickers from JSON file."""
    if CUSTOM_TICKERS_FILE.exists():
        try:
            with open(CUSTOM_TICKERS_FILE, "r") as f:
                data = json.load(f)
                # Ensure both custom and selected keys exist
                if "custom" not in data:
                    data = {"custom": {"Stocks": [], "Crypto": []}, "selected": {"Stocks": [], "Crypto": []}}
                if "selected" not in data:
                    data["selected"] = {"Stocks": [], "Crypto": []}
                return data
        except Exception:
            return {"custom": {"Stocks": [], "Crypto": []}, "selected": {"Stocks": [], "Crypto": []}}
    return {"custom": {"Stocks": [], "Crypto": []}, "selected": {"Stocks": [], "Crypto": []}}

def save_custom_tickers(data: dict):
    """Save custom tickers to JSON file."""
    try:
        with open(CUSTOM_TICKERS_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass

# --- Alerts persistence ---
def load_alerts() -> list:
    """Load alerts from JSON file."""
    if ALERTS_FILE.exists():
        try:
            with open(ALERTS_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_alerts(alerts: list):
    """Save alerts to JSON file."""
    try:
        with open(ALERTS_FILE, "w") as f:
            json.dump(alerts, f, indent=2)
    except Exception:
        pass

# --- Portfolio persistence ---
def load_portfolio() -> dict:
    """Load portfolio from JSON file."""
    if PORTFOLIO_FILE.exists():
        try:
            with open(PORTFOLIO_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {"tickers": [], "weights": {}}
    return {"tickers": [], "weights": {}}

def save_portfolio(portfolio: dict):
    """Save portfolio to JSON file."""
    try:
        with open(PORTFOLIO_FILE, "w") as f:
            json.dump(portfolio, f, indent=2)
    except Exception:
        pass

# --- Cross history persistence ---
def load_cross_history() -> dict:
    """Load cross history from JSON file."""
    if CROSS_HISTORY_FILE.exists():
        try:
            with open(CROSS_HISTORY_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_cross_history(history: dict):
    """Save cross history to JSON file."""
    try:
        with open(CROSS_HISTORY_FILE, "w") as f:
            json.dump(history, f, indent=2, default=str)
    except Exception:
        pass

# --- Watchlists persistence ---
def load_watchlists() -> dict:
    """Load watchlists from JSON file."""
    if WATCHLISTS_FILE.exists():
        try:
            with open(WATCHLISTS_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_watchlists(watchlists: dict):
    """Save watchlists to JSON file."""
    try:
        with open(WATCHLISTS_FILE, "w") as f:
            json.dump(watchlists, f, indent=2)
    except Exception:
        pass

# --- Performance Metrics ---
def calculate_performance_metrics(df: pd.DataFrame) -> dict:
    """Calculate performance metrics: returns, Sharpe ratio, volatility."""
    if df.empty or "close" not in df.columns:
        return {}
    
    returns = df["close"].pct_change().dropna()
    if returns.empty:
        return {}
    
    # Total return
    total_return = (df["close"].iloc[-1] / df["close"].iloc[0] - 1) * 100
    
    # Annualized return (assuming daily data)
    days = len(df)
    annualized_return = ((df["close"].iloc[-1] / df["close"].iloc[0]) ** (252 / days) - 1) * 100 if days > 0 else 0
    
    # Volatility (annualized)
    volatility = returns.std() * np.sqrt(252) * 100
    
    # Sharpe ratio (assuming risk-free rate of 0 for simplicity)
    sharpe_ratio = (annualized_return / 100) / (volatility / 100) if volatility > 0 else 0
    
    # Maximum drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min() * 100
    
    # Win rate (if we consider up days as wins)
    win_rate = (returns > 0).sum() / len(returns) * 100 if len(returns) > 0 else 0
    
    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "volatility": volatility,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "days": days
    }

# --- Email Configuration & Alerts ---
def load_email_config() -> dict:
    """Load email configuration from JSON file."""
    if EMAIL_CONFIG_FILE.exists():
        try:
            with open(EMAIL_CONFIG_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_email_config(config: dict):
    """Save email configuration to JSON file."""
    try:
        with open(EMAIL_CONFIG_FILE, "w") as f:
            json.dump(config, f, indent=2)
    except Exception:
        pass

def send_email_alert(to_email: str, subject: str, body: str, config: dict) -> bool:
    """Send email alert using SMTP."""
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
    """Load alert history from JSON file."""
    if ALERT_HISTORY_FILE.exists():
        try:
            with open(ALERT_HISTORY_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_alert_history(history: list):
    """Save alert history to JSON file."""
    try:
        with open(ALERT_HISTORY_FILE, "w") as f:
            json.dump(history, f, indent=2, default=str)
    except Exception:
        pass

# --- Trade Journal ---
def load_trade_journal() -> list:
    """Load trade journal from JSON file."""
    if TRADE_JOURNAL_FILE.exists():
        try:
            with open(TRADE_JOURNAL_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_trade_journal(journal: list):
    """Save trade journal to JSON file."""
    try:
        with open(TRADE_JOURNAL_FILE, "w") as f:
            json.dump(journal, f, indent=2, default=str)
    except Exception:
        pass

# --- Support & Resistance Detection ---
def find_support_resistance(df: pd.DataFrame, window: int = 20, lookback: int = 5) -> dict:
    """Find support and resistance levels using pivot points."""
    if df.empty or "high" not in df.columns or "low" not in df.columns:
        return {"support": [], "resistance": []}
    
    highs = df["high"].values
    lows = df["low"].values
    
    # Find local maxima (resistance) and minima (support)
    from scipy.signal import argrelextrema
    
    resistance_levels = []
    support_levels = []
    
    # Find resistance (local highs)
    max_indices = argrelextrema(highs, np.greater, order=lookback)[0]
    for idx in max_indices:
        if idx < len(df):
            resistance_levels.append({
                "price": float(highs[idx]),
                "date": df.index[idx],
                "strength": 1  # Can be enhanced with volume/retest count
            })
    
    # Find support (local lows)
    min_indices = argrelextrema(lows, np.less, order=lookback)[0]
    for idx in min_indices:
        if idx < len(df):
            support_levels.append({
                "price": float(lows[idx]),
                "date": df.index[idx],
                "strength": 1
            })
    
    # Sort and get most recent/relevant levels
    resistance_levels.sort(key=lambda x: x["price"], reverse=True)
    support_levels.sort(key=lambda x: x["price"], reverse=True)
    
    # Filter to most significant levels (within 2% of current price)
    current_price = float(df["close"].iloc[-1])
    resistance_levels = [r for r in resistance_levels if abs(r["price"] - current_price) / current_price < 0.2]
    support_levels = [s for s in support_levels if abs(s["price"] - current_price) / current_price < 0.2]
    
    return {
        "support": support_levels[:5],  # Top 5 support levels
        "resistance": resistance_levels[:5]  # Top 5 resistance levels
    }

# --- Pattern Detection ---
def detect_patterns(df: pd.DataFrame) -> list:
    """Detect common chart patterns."""
    if df.empty or len(df) < 50:
        return []
    
    patterns = []
    prices = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    
    # Simple pattern detection (can be enhanced)
    # Double Top
    if len(highs) >= 20:
        recent_highs = highs[-20:]
        max_idx = np.argmax(recent_highs)
        if max_idx > 5 and max_idx < len(recent_highs) - 5:
            left_peak = recent_highs[max_idx - 5:max_idx].max()
            right_peak = recent_highs[max_idx:max_idx + 5].max()
            if abs(left_peak - right_peak) / left_peak < 0.02:  # Within 2%
                patterns.append({
                    "type": "Double Top",
                    "confidence": 0.6,
                    "description": "Bearish reversal pattern detected"
                })
    
    # Double Bottom
    if len(lows) >= 20:
        recent_lows = lows[-20:]
        min_idx = np.argmin(recent_lows)
        if min_idx > 5 and min_idx < len(recent_lows) - 5:
            left_trough = recent_lows[min_idx - 5:min_idx].min()
            right_trough = recent_lows[min_idx:min_idx + 5].min()
            if abs(left_trough - right_trough) / left_trough < 0.02:
                patterns.append({
                    "type": "Double Bottom",
                    "confidence": 0.6,
                    "description": "Bullish reversal pattern detected"
                })
    
    return patterns

# --- Correlation Analysis ---
def calculate_correlation(tickers: list[str], start: date, end: date, mode: str, interval: str = "1d") -> pd.DataFrame:
    """Calculate correlation matrix for multiple tickers."""
    if len(tickers) < 2:
        return pd.DataFrame()
    
    prices = {}
    for ticker in tickers:
        df = load_data(ticker, start, end, mode, interval=interval)
        if not df.empty and "close" in df.columns:
            prices[ticker] = df["close"]
    
    if len(prices) < 2:
        return pd.DataFrame()
    
    # Align all price series
    price_df = pd.DataFrame(prices)
    price_df = price_df.dropna()
    
    if price_df.empty:
        return pd.DataFrame()
    
    # Calculate returns
    returns = price_df.pct_change().dropna()
    
    # Calculate correlation
    correlation = returns.corr()
    return correlation

# --- News Fetching ---
def fetch_news(ticker: str, max_items: int = 5) -> list:
    """Fetch news for a ticker using Yahoo Finance."""
    try:
        yf = _yf()
        stock = yf.Ticker(ticker)
        news = stock.news
        if news:
            return news[:max_items]
    except Exception:
        pass
    return []

# --- Lazy imports for data providers ---
@lru_cache(maxsize=1)
def _yf():
    import yfinance as yf
    return yf

@lru_cache(maxsize=1)
def _cg():
    from pycoingecko import CoinGeckoAPI
    return CoinGeckoAPI()

# --- Utilities ---
def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=False)
    df = df.sort_index()
    return df

def _to_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize to ['open','high','low','close','volume'].
    Robust to yfinance MultiIndex columns like ('Open','SPY') or ('SPY','Open').
    """
    import numpy as _np
    import pandas as _pd

    if df is None or len(df) == 0:
        return _pd.DataFrame(columns=["open","high","low","close","volume"])

    # Flatten MultiIndex columns by picking the OHLC token where present
    if isinstance(df.columns, _pd.MultiIndex):
        flat_cols = []
        for col in df.columns:
            parts = [str(p) for p in col if p is not None]
            tokens = [p.lower() for p in parts]
            field = None
            for p in tokens:
                if p in {"open","high","low","close","adj close","adj_close","adjclose","volume"}:
                    field = p
                    break
            if field is None:
                field = tokens[-1] if tokens else "close"
            if field == "adj close":
                field = "adj_close"
            flat_cols.append(field)
        df.columns = flat_cols
    else:
        df.columns = [str(c) for c in df.columns]

    cols_lower = [c.lower() for c in df.columns]
    out = _pd.DataFrame(index=df.index)

    def pick(*cands):
        for name in cands:
            name_l = name.lower()
            if name_l in cols_lower:
                return df.iloc[:, cols_lower.index(name_l)]
        return _pd.Series(_np.nan, index=df.index, dtype="float64")

    out["open"]   = pick("open")
    out["high"]   = pick("high")
    out["low"]    = pick("low")
    close_series  = pick("close")
    if close_series.isna().all():
        close_series = pick("adj_close","adj close","adjclose")
    out["close"]  = close_series
    out["volume"] = pick("volume")

    return out

def _sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=1).mean()

def _ema(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(span=window, adjust=False).mean()

def _pct(a: float, b: float) -> float:
    if b == 0 or (b is None) or (a is None) or np.isnan(a) or np.isnan(b):
        return np.nan
    return (a / b - 1.0) * 100.0

# --- Technical Indicators ---
def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Calculate Relative Strength Index."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def _macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate MACD (Moving Average Convergence Divergence).
    Returns: (MACD line, Signal line, Histogram)"""
    ema_fast = _ema(series, fast)
    ema_slow = _ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def _bollinger_bands(series: pd.Series, window: int = 20, num_std: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate Bollinger Bands.
    Returns: (Upper band, Middle band (SMA), Lower band)"""
    middle = _sma(series, window)
    std = series.rolling(window=window).std()
    upper = middle + (std * num_std)
    lower = middle - (std * num_std)
    return upper, middle, lower

def _volume_sma(volume: pd.Series, window: int = 20) -> pd.Series:
    """Calculate volume moving average."""
    return _sma(volume, window)

def _vwap(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Calculate Volume Weighted Average Price."""
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    vwap = (typical_price * df["volume"]).rolling(window=window).sum() / df["volume"].rolling(window=window).sum()
    return vwap

# --- Cross Detection and History ---
def detect_crosses(df: pd.DataFrame, s20: str = "SMA20", s200: str = "SMA200") -> list:
    """Detect all Golden and Death crosses in the dataframe. Returns list of cross events."""
    if df is None or df.empty or s20 not in df.columns or s200 not in df.columns:
        return []
    
    crosses = []
    s_prev = df[s20].shift(1)
    l_prev = df[s200].shift(1)
    cross_up = (s_prev <= l_prev) & (df[s20] > df[s200])
    cross_dn = (s_prev >= l_prev) & (df[s20] < df[s200])
    
    for idx in df.index[cross_up]:
        crosses.append({
            "date": idx.strftime("%Y-%m-%d") if isinstance(idx, pd.Timestamp) else str(idx),
            "type": "Golden",
            "price": float(df.loc[idx, "close"]),
            "sma20": float(df.loc[idx, s20]),
            "sma200": float(df.loc[idx, s200])
        })
    
    for idx in df.index[cross_dn]:
        crosses.append({
            "date": idx.strftime("%Y-%m-%d") if isinstance(idx, pd.Timestamp) else str(idx),
            "type": "Death",
            "price": float(df.loc[idx, "close"]),
            "sma20": float(df.loc[idx, s20]),
            "sma200": float(df.loc[idx, s200])
        })
    
    return sorted(crosses, key=lambda x: x["date"])

def analyze_cross_performance(df: pd.DataFrame, crosses: list, days_after: int = 30) -> list:
    """Analyze performance after each cross. Returns crosses with performance metrics."""
    if df.empty or not crosses:
        return crosses
    
    result = []
    for cross in crosses:
        try:
            cross_date = pd.to_datetime(cross["date"])
            cross_idx = df.index.get_indexer([cross_date], method="nearest")[0]
            if cross_idx < 0 or cross_idx >= len(df):
                cross["performance"] = None
                result.append(cross)
                continue
            
            # Get price at cross
            price_at_cross = df.iloc[cross_idx]["close"]
            
            # Get price N days after (or last available)
            future_idx = min(cross_idx + days_after, len(df) - 1)
            price_after = df.iloc[future_idx]["close"]
            
            # Calculate performance
            pct_change = ((price_after - price_at_cross) / price_at_cross) * 100
            days_actual = future_idx - cross_idx
            
            cross["performance"] = {
                "price_after": float(price_after),
                "pct_change": float(pct_change),
                "days_actual": int(days_actual),
                "days_target": days_after
            }
        except Exception:
            cross["performance"] = None
        
        result.append(cross)
    
    return result

# --- Alert Checking ---
def check_alerts(ticker: str, df: pd.DataFrame, alerts: list, send_email: bool = False) -> list:
    """Check if any alerts should be triggered. Returns list of triggered alerts."""
    if df.empty:
        return []
    
    triggered = []
    last = df.iloc[-1]
    price = float(last["close"])
    
    # Calculate indicators if needed
    rsi = None
    if any(a.get("type") in ["rsi_overbought", "rsi_oversold"] for a in alerts):
        rsi_series = _rsi(df["close"], window=14)
        rsi = float(rsi_series.iloc[-1]) if not pd.isna(rsi_series.iloc[-1]) else None
    
    # Load alert history to prevent duplicate notifications
    alert_history = load_alert_history()
    recent_alerts = {h.get("alert_id") for h in alert_history[-100:] if h.get("timestamp", "") > (datetime.now() - timedelta(hours=24)).isoformat()}
    
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
            d = df.copy()
            d["SMA20"] = _sma(d["close"], SMA_SHORT)
            d["SMA200"] = _sma(d["close"], SMA_LONG)
            s_prev = d["SMA20"].shift(1)
            l_prev = d["SMA200"].shift(1)
            if len(d) > 1 and s_prev.iloc[-1] <= l_prev.iloc[-1] and d["SMA20"].iloc[-1] > d["SMA200"].iloc[-1]:
                triggered_alert = {"alert": alert, "message": f"Golden Cross detected for {ticker} at ${price:.2f}"}
        
        elif alert_type == "death_cross":
            d = df.copy()
            d["SMA20"] = _sma(d["close"], SMA_SHORT)
            d["SMA200"] = _sma(d["close"], SMA_LONG)
            s_prev = d["SMA20"].shift(1)
            l_prev = d["SMA200"].shift(1)
            if len(d) > 1 and s_prev.iloc[-1] >= l_prev.iloc[-1] and d["SMA20"].iloc[-1] < d["SMA200"].iloc[-1]:
                triggered_alert = {"alert": alert, "message": f"Death Cross detected for {ticker} at ${price:.2f}"}
        
        elif alert_type == "price_above" and price >= alert.get("value", 0):
            triggered_alert = {"alert": alert, "message": f"{ticker} price ${price:.2f} is above ${alert.get('value', 0):.2f}"}
        
        elif alert_type == "price_below" and price <= alert.get("value", 0):
            triggered_alert = {"alert": alert, "message": f"{ticker} price ${price:.2f} is below ${alert.get('value', 0):.2f}"}
        
        elif alert_type == "rsi_oversold" and rsi is not None and rsi <= alert.get("value", 30):
            triggered_alert = {"alert": alert, "message": f"{ticker} RSI {rsi:.1f} is oversold (≤{alert.get('value', 30)})"}
        
        elif alert_type == "rsi_overbought" and rsi is not None and rsi >= alert.get("value", 70):
            triggered_alert = {"alert": alert, "message": f"{ticker} RSI {rsi:.1f} is overbought (≥{alert.get('value', 70)})"}
        
        if triggered_alert:
            triggered.append(triggered_alert)
            
            # Send email if configured
            if send_email and alert.get("email_enabled", False):
                email_config = load_email_config()
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

# --- Portfolio Calculations ---
def calculate_portfolio_metrics(portfolio: dict, start: date, end: date, mode: str) -> dict:
    """Calculate portfolio-level metrics."""
    if not portfolio.get("tickers") or len(portfolio["tickers"]) == 0:
        return {"error": "No tickers in portfolio"}
    
    tickers = portfolio["tickers"]
    weights = portfolio.get("weights", {})
    
    # Normalize weights to sum to 1.0
    total_weight = sum(weights.get(t, 1.0/len(tickers)) for t in tickers)
    if total_weight == 0:
        total_weight = len(tickers)
    
    portfolio_value = 0.0
    portfolio_return = 0.0
    individual_returns = {}
    
    for ticker in tickers:
        weight = weights.get(ticker, 1.0/len(tickers)) / total_weight if total_weight > 0 else 1.0/len(tickers)
        df = load_data(ticker, start, end, mode)
        
        if df.empty:
            continue
        
        start_price = float(df.iloc[0]["close"])
        end_price = float(df.iloc[-1]["close"])
        ticker_return = ((end_price - start_price) / start_price) * 100
        
        portfolio_value += end_price * weight
        portfolio_return += ticker_return * weight
        individual_returns[ticker] = {
            "return": ticker_return,
            "weight": weight,
            "price": end_price
        }
    
    return {
        "portfolio_value": portfolio_value,
        "portfolio_return": portfolio_return,
        "individual_returns": individual_returns,
        "ticker_count": len(tickers)
    }

# --- Backtesting ---
def backtest_strategy(df: pd.DataFrame, strategy: str = "golden_death", initial_capital: float = 10000.0) -> dict:
    """Backtest a trading strategy. Returns performance metrics."""
    if df.empty or len(df) < SMA_LONG:
        return {"error": "Insufficient data for backtesting"}
    
    df = df.copy()
    df["SMA20"] = _sma(df["close"], SMA_SHORT)
    df["SMA200"] = _sma(df["close"], SMA_LONG)
    
    capital = initial_capital
    shares = 0.0
    trades = []
    position = None  # "long" or None
    
    s_prev = df["SMA20"].shift(1)
    l_prev = df["SMA200"].shift(1)
    golden_crosses = (s_prev <= l_prev) & (df["SMA20"] > df["SMA200"])
    death_crosses = (s_prev >= l_prev) & (df["SMA20"] < df["SMA200"])
    
    for i in range(SMA_LONG, len(df)):
        price = float(df.iloc[i]["close"])
        date = df.index[i]
        
        if strategy == "golden_death":
            # Buy on Golden Cross, Sell on Death Cross
            if golden_crosses.iloc[i] and position is None:
                # Buy
                shares = capital / price
                capital = 0
                position = "long"
                trades.append({"date": date, "type": "BUY", "price": price, "shares": shares})
            
            elif death_crosses.iloc[i] and position == "long":
                # Sell
                capital = shares * price
                shares = 0
                position = None
                trades.append({"date": date, "type": "SELL", "price": price, "capital": capital})
        
        elif strategy == "buy_hold":
            # Buy and hold from start
            if i == SMA_LONG and position is None:
                shares = capital / price
                capital = 0
                position = "long"
                trades.append({"date": date, "type": "BUY", "price": price, "shares": shares})
    
    # Final value
    if position == "long" and shares > 0:
        final_price = float(df.iloc[-1]["close"])
        final_capital = shares * final_price
    else:
        final_capital = capital
    
    # Calculate metrics
    total_return = ((final_capital - initial_capital) / initial_capital) * 100
    num_trades = len([t for t in trades if t["type"] == "BUY"])
    
    # Calculate drawdown by tracking position through time
    equity_curve = []
    current_shares = 0.0
    current_capital = initial_capital
    current_position = None
    
    for i in range(SMA_LONG, len(df)):
        price = float(df.iloc[i]["close"])
        
        # Update position based on strategy
        if strategy == "golden_death":
            if golden_crosses.iloc[i] and current_position is None:
                current_shares = current_capital / price
                current_capital = 0
                current_position = "long"
            elif death_crosses.iloc[i] and current_position == "long":
                current_capital = current_shares * price
                current_shares = 0
                current_position = None
        elif strategy == "buy_hold":
            if i == SMA_LONG and current_position is None:
                current_shares = current_capital / price
                current_capital = 0
                current_position = "long"
        
        # Calculate current equity
        if current_position == "long":
            equity = current_shares * price
        else:
            equity = current_capital
        equity_curve.append(equity)
    
    peak = equity_curve[0]
    max_drawdown = 0.0
    for equity in equity_curve:
        if equity > peak:
            peak = equity
        drawdown = ((peak - equity) / peak) * 100
        if drawdown > max_drawdown:
            max_drawdown = drawdown
    
    return {
        "initial_capital": initial_capital,
        "final_capital": final_capital,
        "total_return": total_return,
        "num_trades": num_trades,
        "max_drawdown": max_drawdown,
        "trades": trades,
        "strategy": strategy
    }

def _badge_color(trend: str) -> str:
    if trend == "Bullish":
        return "background-color:#12391a;color:#1fd16c;font-weight:600"
    if trend == "Bearish":
        return "background-color:#3a1919;color:#ff4d4f;font-weight:600"
    return ""

# --- Data loaders ---
@st.cache_data(show_spinner=False)
def load_stock(ticker: str, start: date, end: date, interval: str = "1d") -> pd.DataFrame:
    yf = _yf()
    
    # New yfinance (0.2.66+) uses curl_cffi and handles sessions automatically
    # Try using Ticker API first (more reliable)
    try:
        stock = yf.Ticker(ticker)  # Don't pass session - let yfinance handle it
        df = stock.history(start=start, end=(end + timedelta(days=1)), interval=interval, auto_adjust=True)
        if df is not None and len(df) > 0:
            df = _ensure_datetime_index(df)
            df = _to_ohlc(df)
            return df
    except Exception as e:
        import sys
        print(f"Ticker API failed for {ticker}: {e}", file=sys.stderr)
    
    # Fallback to download API
    max_retries = 2
    for attempt in range(max_retries):
        try:
            df = yf.download(
                tickers=ticker,
                start=start.isoformat(),
                end=(end + timedelta(days=1)).isoformat(),
                interval=interval,
                auto_adjust=True,
                progress=False,
                group_by="column",
                threads=False,
            )
            if df is not None and len(df) > 0:
                df = _ensure_datetime_index(df)
                df = _to_ohlc(df)
                return df
            if attempt < max_retries - 1:
                time.sleep(1)
        except Exception as e:
            if attempt < max_retries - 1:
                import sys
                print(f"Download attempt {attempt + 1} failed for {ticker}: {e}", file=sys.stderr)
                time.sleep(2)
                continue
    
    return pd.DataFrame()

def _cg_id_for(symbol_usd: str) -> str | None:
    # crude mapping for popular coins
    symbol = symbol_usd.upper().replace("-USD", "")
    mapping = {
        "BTC": "bitcoin",
        "ETH": "ethereum",
        "SOL": "solana",
        "ADA": "cardano",
        "XRP": "ripple",
        "DOGE": "dogecoin",
    }
    return mapping.get(symbol)

@st.cache_data(show_spinner=False)
def load_crypto(ticker: str, start: date, end: date) -> pd.DataFrame:
    """Try Yahoo Finance first for -USD pairs; fallback to CoinGecko (capped to 365d)."""
    # Try yfinance for e.g. 'BTC-USD'
    try:
        df_y = load_stock(ticker, start, end)
        if len(df_y) > 0 and not df_y["close"].isna().all():
            return df_y
    except Exception:
        pass

    # CoinGecko fallback (range limited to 365 days on public plan)
    try:
        cg = _cg()
        cg_id = _cg_id_for(ticker)
        if not cg_id:
            return pd.DataFrame()

        # Respect 365d cap
        max_days = 365
        s_cap = max(end - timedelta(days=max_days), start)
        from_ts = int(datetime.combine(s_cap, datetime.min.time()).timestamp())
        to_ts = int(datetime.combine(end, datetime.min.time()).timestamp())

        payload = cg.get_coin_market_chart_range_by_id(
            id=cg_id,
            vs_currency="usd",
            from_timestamp=from_ts,
            to_timestamp=to_ts,
        )
        if not payload or "prices" not in payload:
            return pd.DataFrame()

        # Build OHLC from prices (minute/daily granularity → we’ll downsample to 1D)
        prices = payload.get("prices", [])
        df = pd.DataFrame(prices, columns=["ts", "price"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms")
        # 1D resample for OHLC
        ohlc = df.set_index("ts")["price"].resample("1D").agg(["first", "max", "min", "last"])
        ohlc.columns = ["open", "high", "low", "close"]
        ohlc["volume"] = np.nan
        ohlc = ohlc.loc[(ohlc.index.date >= start) & (ohlc.index.date <= end)]
        return ohlc
    except Exception:
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_data(ticker: str, start: date, end: date, mode: str, interval: str = "1d") -> pd.DataFrame:
    if mode == "Stocks":
        return load_stock(ticker, start, end, interval=interval)
    # Crypto doesn't support intervals in the same way, but we can resample if needed
    df = load_crypto(ticker, start, end)
    if not df.empty and interval != "1d":
        # Resample crypto data to weekly/monthly
        if interval == "1wk":
            df = df.resample("W").agg({
                "open": "first", "high": "max", "low": "min", 
                "close": "last", "volume": "sum"
            }).dropna()
        elif interval == "1mo":
            df = df.resample("M").agg({
                "open": "first", "high": "max", "low": "min", 
                "close": "last", "volume": "sum"
            }).dropna()
    return df

# --- Chart builder ---
def make_chart(df: pd.DataFrame, title: str, theme: str, pretouch_pct: float | None,
               show_volume: bool = True, show_rsi: bool = True, show_macd: bool = True,
               show_bollinger: bool = True, show_sma20: bool = True, show_sma200: bool = True,
               show_support_resistance: bool = False) -> go.Figure:
    if df.empty:
        fig = go.Figure()
        fig.update_layout(template="plotly_dark" if theme == "Dark" else "plotly_white", title=title)
        return fig

    df = df.copy()
    df["SMA20"] = _sma(df["close"], SMA_SHORT)
    df["SMA200"] = _sma(df["close"], SMA_LONG)
    
    # Set neutral background (the fill between SMAs will provide the color)
    plot_bgcolor = "rgba(0, 0, 0, 0)" if theme == "Dark" else "rgba(255, 255, 255, 0)"
    paper_bgcolor = "rgba(0, 0, 0, 0)" if theme == "Dark" else "rgba(255, 255, 255, 0)"

    template = "plotly_dark" if theme == "Dark" else "plotly_white"
    
    # Determine number of subplots needed
    num_subplots = 1  # Main price chart
    if show_volume:
        num_subplots += 1
    if show_rsi:
        num_subplots += 1
    if show_macd:
        num_subplots += 1
    
    # Create subplots
    from plotly.subplots import make_subplots
    fig = make_subplots(
        rows=num_subplots, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.5] + [0.5 / (num_subplots - 1)] * (num_subplots - 1) if num_subplots > 1 else [1.0],
        subplot_titles=([title] + 
                       (["Volume"] if show_volume else []) +
                       (["RSI"] if show_rsi else []) +
                       (["MACD"] if show_macd else []))
    )
    
    row = 1
    
    # Add filled area between SMAs with dynamic color based on which is on top
    # This must be added BEFORE the candlestick so it appears behind
    if show_sma20 and show_sma200:
        # Create segments where SMA20 is above vs SMA200 is above
        sma20_above = df["SMA20"] > df["SMA200"]
        
        # Find transition points where the relationship changes
        transitions = (sma20_above != sma20_above.shift(1)).fillna(False)
        transition_indices = df.index[transitions].tolist()
        
        # Add start and end to transition list
        all_points = [df.index[0]] + transition_indices + [df.index[-1]]
        
        # Create segments
        for i in range(len(all_points) - 1):
            start_idx = all_points[i]
            end_idx = all_points[i + 1]
            segment_df = df.loc[start_idx:end_idx]
            
            if len(segment_df) < 2:
                continue
            
            # Determine which SMA is on top in this segment
            is_sma20_top = segment_df["SMA20"].iloc[0] > segment_df["SMA200"].iloc[0]
            
            # Create upper and lower bounds for fill
            upper = segment_df[["SMA20", "SMA200"]].max(axis=1)
            lower = segment_df[["SMA20", "SMA200"]].min(axis=1)
            
            # Set color based on which is on top
            if is_sma20_top:
                # SMA20 on top (green/blue) - green fill
                fillcolor = "rgba(23, 201, 100, 0.2)"
                line_color = "rgba(23, 201, 100, 0.3)"
            else:
                # SMA200 on top (red) - red fill
                fillcolor = "rgba(243, 18, 96, 0.2)"
                line_color = "rgba(243, 18, 96, 0.3)"
            
            # Add filled area
            fig.add_trace(go.Scatter(
                x=segment_df.index,
                y=upper,
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip"
            ), row=row, col=1)
            
            fig.add_trace(go.Scatter(
                x=segment_df.index,
                y=lower,
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor=fillcolor,
                showlegend=False,
                hoverinfo="skip"
            ), row=row, col=1)
    
    # Main price chart (add after fill so it appears on top)
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        name="Price"
    ), row=row, col=1)
    
    # Add SMA20 line if enabled
    if show_sma20:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["SMA20"], mode="lines", name=f"SMA {SMA_SHORT}", 
            line=dict(width=2, color="#4fa3ff")
        ), row=row, col=1)
    
    # Add SMA200 line if enabled
    if show_sma200:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["SMA200"], mode="lines", name=f"SMA {SMA_LONG}", 
            line=dict(width=2, color="#ff6b6b")
        ), row=row, col=1)

    # Bollinger Bands
    if show_bollinger:
        bb_upper, bb_middle, bb_lower = _bollinger_bands(df["close"], window=20, num_std=2.0)
        fig.add_trace(go.Scatter(
            x=df.index, y=bb_upper, mode="lines", name="BB Upper",
            line=dict(width=1, color="#888", dash="dash"), showlegend=False
        ), row=row, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=bb_lower, mode="lines", name="BB Lower",
            line=dict(width=1, color="#888", dash="dash"), fill="tonexty",
            fillcolor="rgba(128,128,128,0.1)", showlegend=False
        ), row=row, col=1)

    # Add support and resistance levels
    if show_support_resistance:
        sr_levels = find_support_resistance(df)
        for level in sr_levels.get("support", []):
            fig.add_hline(
                y=level["price"],
                line_dash="dash",
                line_color="green",
                opacity=0.5,
                annotation_text=f"Support ${level['price']:.2f}",
                row=row, col=1
            )
        for level in sr_levels.get("resistance", []):
            fig.add_hline(
                y=level["price"],
                line_dash="dash",
                line_color="red",
                opacity=0.5,
                annotation_text=f"Resistance ${level['price']:.2f}",
                row=row, col=1
            )
    
    # Add cross markers (golden/death crosses)
    add_cross_markers(fig, df, price_col="close", s20="SMA20", s200="SMA200", row=row, col=1)

    # Pretouch band (symmetric % around SMA200)
    if pretouch_pct and pretouch_pct > 0:
        band = df["SMA200"] * (pretouch_pct / 100.0)
        upper = df["SMA200"] + band
        lower = df["SMA200"] - band
        fig.add_trace(go.Scatter(
            x=df.index, y=upper, line=dict(width=0), showlegend=False, hoverinfo="skip"
        ), row=row, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=lower, fill="tonexty", name=f"Pretouch ±{pretouch_pct:.2f}%",
            hoverinfo="skip", opacity=0.15
        ), row=row, col=1)

    # Volume subplot
    if show_volume and not df["volume"].isna().all():
        row += 1
        colors = ["#17c964" if df["close"].iloc[i] >= df["open"].iloc[i] else "#f31260" 
                 for i in range(len(df))]
        fig.add_trace(go.Bar(
            x=df.index, y=df["volume"], name="Volume",
            marker_color=colors, opacity=0.6
        ), row=row, col=1)
        
        # Volume SMA
        vol_sma = _volume_sma(df["volume"], window=20)
        fig.add_trace(go.Scatter(
            x=df.index, y=vol_sma, mode="lines", name="Vol SMA 20",
            line=dict(width=1, color="#888")
        ), row=row, col=1)
        
        fig.update_yaxes(title_text="Volume", row=row, col=1)

    # RSI subplot
    if show_rsi:
        row += 1
        rsi = _rsi(df["close"], window=14)
        fig.add_trace(go.Scatter(
            x=df.index, y=rsi, mode="lines", name="RSI",
            line=dict(width=2, color="#9b59b6")
        ), row=row, col=1)
        
        # RSI levels
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=row, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=row, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.3, row=row, col=1)
        
        fig.update_yaxes(title_text="RSI", range=[0, 100], row=row, col=1)

    # MACD subplot
    if show_macd:
        row += 1
        macd_line, signal_line, histogram = _macd(df["close"])
        fig.add_trace(go.Scatter(
            x=df.index, y=macd_line, mode="lines", name="MACD",
            line=dict(width=2, color="#3498db")
        ), row=row, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=signal_line, mode="lines", name="Signal",
            line=dict(width=2, color="#e74c3c")
        ), row=row, col=1)
        
        # Histogram
        colors_hist = ["#17c964" if h >= 0 else "#f31260" for h in histogram]
        fig.add_trace(go.Bar(
            x=df.index, y=histogram, name="Histogram",
            marker_color=colors_hist, opacity=0.6
        ), row=row, col=1)
        
        fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.3, row=row, col=1)
        fig.update_yaxes(title_text="MACD", row=row, col=1)

    fig.update_layout(
        template=template,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=10, r=10, t=60, b=10),
        height=400 + (200 * (num_subplots - 1)),
        plot_bgcolor=plot_bgcolor,
        paper_bgcolor=paper_bgcolor
    )
    
    # Update the main chart's background color
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(128,128,128,0.2)", row=row, col=1)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(128,128,128,0.2)", row=row, col=1)
    
    return fig

# --- Screener helper ---
def build_screener(tickers: list[str], start: date, end: date, mode: str, pretouch_pct: float | None, interval: str = "1d") -> pd.DataFrame:
    rows = []
    for t in tickers:
        d = load_data(t, start, end, mode, interval=interval)
        if d.empty or d["close"].isna().all():
            continue
        d = d.copy()
        d["SMA20"] = _sma(d["close"], SMA_SHORT)
        d["SMA200"] = _sma(d["close"], SMA_LONG)
        last = d.iloc[-1]

        price = float(last["close"])
        sma20 = float(last["SMA20"])
        sma200 = float(last["SMA200"])

        trend = ("Bullish" if (not math.isnan(sma20) and not math.isnan(sma200) and sma20 > sma200)
                 else ("Bearish" if (not math.isnan(sma20) and not math.isnan(sma200)) else "-"))

        row = {
            "Ticker": t,
            "Trend": trend,
            "Last": price,
            "SMA20": sma20,
            "SMA200": sma200,
            "Dist to SMA200 (%)": _pct(price, sma200),
            "Dist to SMA20 (%)": _pct(price, sma20),
        }

        # Check if SMA20 is touching the SMA200 pretouch band
        touch = ""
        if pretouch_pct and pretouch_pct > 0 and not math.isnan(sma20) and not math.isnan(sma200):
            band = sma200 * (pretouch_pct / 100.0)
            upper = sma200 + band
            lower = sma200 - band
            if lower <= sma20 <= upper:
                touch = "🟡"  # Yellow circle when SMA20 is in the band
        row["Touch"] = touch

        # Add RSI
        rsi = _rsi(d["close"], window=14)
        rsi_last_val = rsi.iloc[-1]
        rsi_last = float(rsi_last_val) if not (pd.isna(rsi_last_val) or math.isnan(rsi_last_val)) else np.nan
        row["RSI"] = rsi_last

        # Add MACD
        macd_line, signal_line, histogram = _macd(d["close"])
        macd_last_val = macd_line.iloc[-1]
        signal_last_val = signal_line.iloc[-1]
        hist_last_val = histogram.iloc[-1]
        macd_last = float(macd_last_val) if not (pd.isna(macd_last_val) or math.isnan(macd_last_val)) else np.nan
        signal_last = float(signal_last_val) if not (pd.isna(signal_last_val) or math.isnan(signal_last_val)) else np.nan
        hist_last = float(hist_last_val) if not (pd.isna(hist_last_val) or math.isnan(hist_last_val)) else np.nan
        row["MACD"] = macd_last
        row["MACD Signal"] = signal_last
        row["MACD Hist"] = hist_last

        # Add Volume metrics
        if not d["volume"].isna().all():
            vol_last = float(last["volume"])
            vol_sma = _volume_sma(d["volume"], window=20)
            vol_sma_last_val = vol_sma.iloc[-1]
            vol_sma_last = float(vol_sma_last_val) if not (pd.isna(vol_sma_last_val) or math.isnan(vol_sma_last_val)) else np.nan
            row["Volume"] = vol_last
            row["Vol SMA20"] = vol_sma_last
            row["Vol Ratio"] = vol_last / vol_sma_last if vol_sma_last > 0 and not math.isnan(vol_sma_last) else np.nan
        else:
            row["Volume"] = np.nan
            row["Vol SMA20"] = np.nan
            row["Vol Ratio"] = np.nan

        # Add Bollinger Bands position
        bb_upper, bb_middle, bb_lower = _bollinger_bands(d["close"], window=20, num_std=2.0)
        bb_upper_last_val = bb_upper.iloc[-1]
        bb_lower_last_val = bb_lower.iloc[-1]
        bb_upper_last = float(bb_upper_last_val) if not (pd.isna(bb_upper_last_val) or math.isnan(bb_upper_last_val)) else np.nan
        bb_lower_last = float(bb_lower_last_val) if not (pd.isna(bb_lower_last_val) or math.isnan(bb_lower_last_val)) else np.nan
        if not math.isnan(bb_upper_last) and not math.isnan(bb_lower_last):
            bb_width = bb_upper_last - bb_lower_last
            bb_position = ((price - bb_lower_last) / bb_width * 100) if bb_width > 0 else np.nan
            row["BB Position %"] = bb_position
        else:
            row["BB Position %"] = np.nan

        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values(by=["Dist to SMA200 (%)"], key=lambda s: s.abs()).reset_index(drop=True)
    return df

# --- UI ---
st.set_page_config(page_title="TraderQ SMA 20/200", layout="wide")
ui_glow_patch.apply()  # apply glow after set_page_config
st.title(f"TraderQ SMA 20/200 Tracker — {APP_VERSION}")

# Main tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "📈 Tracker", "🔔 Alerts", "📊 Cross History", "💼 Portfolio", "🧪 Backtesting",
    "📰 News", "📐 Patterns", "🔗 Correlation", "📝 Journal"
])

# Sidebar controls
mode = st.sidebar.radio("Market", ["Stocks", "Crypto"], horizontal=True)
theme = st.sidebar.radio("Chart Theme", ["Dark", "Light"], index=0, horizontal=True)
timeframe = st.sidebar.selectbox("Timeframe", ["Daily", "Weekly", "Monthly"], index=0)
interval_map = {"Daily": "1d", "Weekly": "1wk", "Monthly": "1mo"}
interval = interval_map[timeframe]
pretouch = st.sidebar.slider("Pretouch band around SMA200 (%)", min_value=0.0, max_value=5.0, value=1.0, step=0.25)
period_days = st.sidebar.select_slider("Lookback (days)", options=[180, 365, 540, 730], value=365)

# Indicator toggles
st.sidebar.markdown("---")
st.sidebar.markdown("**Indicators**")
show_sma20 = st.sidebar.checkbox("Show SMA 20", value=True)
show_sma200 = st.sidebar.checkbox("Show SMA 200", value=True)
show_volume = st.sidebar.checkbox("Show Volume", value=True)
show_rsi = st.sidebar.checkbox("Show RSI", value=True)
show_macd = st.sidebar.checkbox("Show MACD", value=True)
show_bollinger = st.sidebar.checkbox("Show Bollinger Bands", value=True)
show_support_resistance = st.sidebar.checkbox("Show Support/Resistance", value=False)

# Date range
end_d = date.today()
start_d = end_d - timedelta(days=int(period_days))

# Ticker selection with persistent custom tickers
universe = DEFAULT_STOCKS if mode == "Stocks" else DEFAULT_CRYPTOS

# Initialize session state for custom tickers per mode (load from file on first run)
if "custom_tickers_loaded" not in st.session_state:
    st.session_state["custom_tickers_loaded"] = True
    saved_data = load_custom_tickers()
    st.session_state["custom_tickers_Stocks"] = saved_data.get("custom", {}).get("Stocks", [])
    st.session_state["custom_tickers_Crypto"] = saved_data.get("custom", {}).get("Crypto", [])
    st.session_state["selected_tickers_Stocks"] = saved_data.get("selected", {}).get("Stocks", [])
    st.session_state["selected_tickers_Crypto"] = saved_data.get("selected", {}).get("Crypto", [])

if f"custom_tickers_{mode}" not in st.session_state:
    st.session_state[f"custom_tickers_{mode}"] = []
if f"selected_tickers_{mode}" not in st.session_state:
    st.session_state[f"selected_tickers_{mode}"] = []

left, right = st.columns([1, 3])
with left:
    st.subheader("Symbols")
    # Combine universe with custom tickers for options
    all_options = list(dict.fromkeys(universe + st.session_state[f"custom_tickers_{mode}"]))
    # Use saved selected tickers if available, otherwise default to universe
    default_selected = st.session_state[f"selected_tickers_{mode}"] if st.session_state[f"selected_tickers_{mode}"] else universe
    selected = st.multiselect("Choose tickers", options=all_options, default=default_selected, key=f"choose_{mode}")

    # Save selected tickers whenever they change
    if selected != st.session_state.get(f"_prev_selected_{mode}", []):
        st.session_state[f"_prev_selected_{mode}"] = selected
        st.session_state[f"selected_tickers_{mode}"] = selected
        save_custom_tickers({
            "custom": {
                "Stocks": st.session_state.get("custom_tickers_Stocks", []),
                "Crypto": st.session_state.get("custom_tickers_Crypto", [])
            },
            "selected": {
                "Stocks": st.session_state.get("selected_tickers_Stocks", []),
                "Crypto": st.session_state.get("selected_tickers_Crypto", [])
            }
        })

    # Reorder controls
    if len(selected) > 1:
        st.markdown("**Reorder:**")
        reorder_cols = st.columns([3, 1, 1])
        with reorder_cols[0]:
            ticker_to_move = st.selectbox("Select ticker to move", selected, key=f"reorder_select_{mode}")
        with reorder_cols[1]:
            if st.button("↑", key=f"move_up_{mode}", help="Move up"):
                idx = selected.index(ticker_to_move)
                if idx > 0:
                    selected[idx], selected[idx-1] = selected[idx-1], selected[idx]
                    st.session_state[f"selected_tickers_{mode}"] = selected
                    save_custom_tickers({
                        "custom": {
                            "Stocks": st.session_state.get("custom_tickers_Stocks", []),
                            "Crypto": st.session_state.get("custom_tickers_Crypto", [])
                        },
                        "selected": {
                            "Stocks": st.session_state.get("selected_tickers_Stocks", []),
                            "Crypto": st.session_state.get("selected_tickers_Crypto", [])
                        }
                    })
                    st.rerun()
        with reorder_cols[2]:
            if st.button("↓", key=f"move_down_{mode}", help="Move down"):
                idx = selected.index(ticker_to_move)
                if idx < len(selected) - 1:
                    selected[idx], selected[idx+1] = selected[idx+1], selected[idx]
                    st.session_state[f"selected_tickers_{mode}"] = selected
                    save_custom_tickers({
                        "custom": {
                            "Stocks": st.session_state.get("custom_tickers_Stocks", []),
                            "Crypto": st.session_state.get("custom_tickers_Crypto", [])
                        },
                        "selected": {
                            "Stocks": st.session_state.get("selected_tickers_Stocks", []),
                            "Crypto": st.session_state.get("selected_tickers_Crypto", [])
                        }
                    })
                    st.rerun()

with right:
    st.subheader("Quick Add")
    custom = st.text_input("Add ticker (Yahoo symbol)", value="", placeholder="e.g., AAPL or BTC-USD", key=f"quick_add_{mode}")
    if st.button("Add", key=f"add_btn_{mode}"):
        if custom.strip():
            custom_upper = custom.strip().upper()
            if custom_upper not in st.session_state[f"custom_tickers_{mode}"]:
                st.session_state[f"custom_tickers_{mode}"].append(custom_upper)
                # Also add to selected tickers
                if custom_upper not in st.session_state[f"selected_tickers_{mode}"]:
                    st.session_state[f"selected_tickers_{mode}"].append(custom_upper)
                # Save to file
                save_custom_tickers({
                    "custom": {
                        "Stocks": st.session_state.get("custom_tickers_Stocks", []),
                        "Crypto": st.session_state.get("custom_tickers_Crypto", [])
                    },
                    "selected": {
                        "Stocks": st.session_state.get("selected_tickers_Stocks", []),
                        "Crypto": st.session_state.get("selected_tickers_Crypto", [])
                    }
                })
                st.rerun()
    
    # Watchlists
    st.subheader("📋 Watchlists")
    watchlists = load_watchlists()
    watchlist_names = list(watchlists.keys())
    
    # Load watchlist selector
    if watchlist_names:
        selected_watchlist = st.selectbox("Select Watchlist", ["-- New --"] + watchlist_names, key=f"watchlist_select_{mode}")
        if selected_watchlist != "-- New --":
            if st.button("Load Watchlist", key=f"load_watchlist_{mode}"):
                st.session_state[f"selected_tickers_{mode}"] = watchlists[selected_watchlist].get("tickers", [])
                st.rerun()
            if st.button("Delete Watchlist", key=f"delete_watchlist_{mode}"):
                del watchlists[selected_watchlist]
                save_watchlists(watchlists)
                st.rerun()
    
    # Create new watchlist
    new_watchlist_name = st.text_input("New Watchlist Name", key=f"new_watchlist_{mode}", placeholder="e.g., Tech Stocks")
    if st.button("Create Watchlist", key=f"create_watchlist_{mode}"):
        if new_watchlist_name.strip() and selected:
            watchlists[new_watchlist_name.strip()] = {
                "tickers": selected.copy(),
                "mode": mode,
                "created": date.today().isoformat()
            }
            save_watchlists(watchlists)
            st.success(f"Watchlist '{new_watchlist_name.strip()}' created!")
            st.rerun()
    
    # Save current selection as watchlist
    if selected:
        save_as_name = st.text_input("Save Current as Watchlist", key=f"save_watchlist_{mode}", placeholder="e.g., My Portfolio")
        if st.button("Save", key=f"save_watchlist_btn_{mode}"):
            if save_as_name.strip():
                watchlists[save_as_name.strip()] = {
                    "tickers": selected.copy(),
                    "mode": mode,
                    "created": date.today().isoformat()
                }
                save_watchlists(watchlists)
                st.success(f"Saved as '{save_as_name.strip()}'!")
                st.rerun()

# Safety: unique list
selected = list(dict.fromkeys(selected))

# ===== TAB 1: TRACKER =====
with tab1:
    # --- Per-ticker charts ---
    for t in selected:
        df = load_data(t, start_d, end_d, mode, interval=interval)
        cols = st.columns([3, 1], gap="large")
        with cols[0]:
            st.markdown(f"**{t}**")
            fig = make_chart(df, f"{t} — SMA {SMA_SHORT}/{SMA_LONG}", theme, pretouch,
                            show_volume=show_volume, show_rsi=show_rsi, 
                            show_macd=show_macd, show_bollinger=show_bollinger,
                            show_sma20=show_sma20, show_sma200=show_sma200,
                            show_support_resistance=show_support_resistance)
            st.plotly_chart(fig, use_container_width=True, key=f"chart_{t}_{mode}")
            
            # Export chart buttons
            export_cols = st.columns(3)
            with export_cols[0]:
                try:
                    img_bytes = fig.to_image(format="png", width=1200, height=800)
                    st.download_button(
                        label="📥 Download PNG",
                        data=img_bytes,
                        file_name=f"{t}_{timeframe}_{date.today().isoformat()}.png",
                        mime="image/png",
                        key=f"export_png_{t}"
                    )
                except Exception as e:
                    st.caption("PNG export requires kaleido. Install: pip install kaleido")
            with export_cols[1]:
                try:
                    html_str = fig.to_html(include_plotlyjs="cdn")
                    st.download_button(
                        label="📥 Download HTML",
                        data=html_str,
                        file_name=f"{t}_{timeframe}_{date.today().isoformat()}.html",
                        mime="text/html",
                        key=f"export_html_{t}"
                    )
                except Exception:
                    pass
        with cols[1]:
            if df.empty:
                st.warning(f"⚠️ No data available for {t}. This could be due to:\n- Yahoo Finance API issues\n- Network connectivity\n- Invalid ticker symbol\n\nTry refreshing the page or check your internet connection.")
            else:
                d = df.copy()
                d["SMA20"] = _sma(d["close"], SMA_SHORT)
                d["SMA200"] = _sma(d["close"], SMA_LONG)
                last = d.iloc[-1]
                price = float(last["close"])
                sma20 = float(last["SMA20"])
                sma200 = float(last["SMA200"])
                st.metric("Last Close", f"${price:,.2f}")
                st.metric(f"SMA {SMA_SHORT}", f"${sma20:,.2f}")
                st.metric(f"SMA {SMA_LONG}", f"${sma200:,.2f}")
                st.metric("20 vs 200", f"{_pct(sma20, sma200):+.2f}%")
                
                # RSI
                if show_rsi:
                    rsi = _rsi(d["close"], window=14)
                    rsi_last_val = rsi.iloc[-1]
                    rsi_last = float(rsi_last_val) if not (pd.isna(rsi_last_val) or math.isnan(rsi_last_val)) else np.nan
                    if not math.isnan(rsi_last):
                        rsi_color = "#f31260" if rsi_last > 70 else "#17c964" if rsi_last < 30 else "#888"
                        st.metric("RSI (14)", f"{rsi_last:.1f}", delta=None)
                
                # MACD
                if show_macd:
                    macd_line, signal_line, histogram = _macd(d["close"])
                    macd_last_val = macd_line.iloc[-1]
                    signal_last_val = signal_line.iloc[-1]
                    hist_last_val = histogram.iloc[-1]
                    macd_last = float(macd_last_val) if not (pd.isna(macd_last_val) or math.isnan(macd_last_val)) else np.nan
                    signal_last = float(signal_last_val) if not (pd.isna(signal_last_val) or math.isnan(signal_last_val)) else np.nan
                    hist_last = float(hist_last_val) if not (pd.isna(hist_last_val) or math.isnan(hist_last_val)) else np.nan
                    if not math.isnan(macd_last):
                        st.metric("MACD", f"{macd_last:.2f}")
                        st.metric("Signal", f"{signal_last:.2f}")
                        st.metric("Histogram", f"{hist_last:.2f}")
                
                # Volume
                if show_volume and not d["volume"].isna().all():
                    vol_last = float(last["volume"])
                    vol_sma = _volume_sma(d["volume"], window=20)
                    vol_sma_last_val = vol_sma.iloc[-1]
                    vol_sma_last = float(vol_sma_last_val) if not (pd.isna(vol_sma_last_val) or math.isnan(vol_sma_last_val)) else np.nan
                    if not math.isnan(vol_sma_last) and vol_sma_last > 0:
                        vol_ratio = vol_last / vol_sma_last
                        st.metric("Volume", f"{vol_last:,.0f}")
                        st.metric("Vol SMA20", f"{vol_sma_last:,.0f}")
                        st.metric("Vol Ratio", f"{vol_ratio:.2f}x")
                
                # Performance Metrics
                metrics = calculate_performance_metrics(df)
                if metrics:
                    st.markdown("---")
                    st.markdown("**📊 Performance Metrics:**")
                    st.metric("Total Return", f"{metrics['total_return']:.2f}%")
                    st.metric("Annualized Return", f"{metrics['annualized_return']:.2f}%")
                    st.metric("Volatility (Annual)", f"{metrics['volatility']:.2f}%")
                    st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
                    st.metric("Max Drawdown", f"{metrics['max_drawdown']:.2f}%")
                    st.metric("Win Rate", f"{metrics['win_rate']:.1f}%")
                    st.caption(f"Based on {metrics['days']} {timeframe.lower()} periods")
                
                # Check alerts for this ticker
                alerts = load_alerts()
                triggered = check_alerts(t, df, alerts)
                if triggered:
                    st.markdown("---")
                    st.markdown("**🔔 Alerts:**")
                    for trig in triggered:
                        st.warning(trig["message"])

    # --- Screener ---
    st.divider()
    st.subheader("Pretouch Screener (closest to SMA200 on top)")

    screener_df = build_screener(selected or universe, start_d, end_d, mode, pretouch, interval=interval)
    if screener_df.empty:
        st.info("No data to screen.")
    else:
        # Build column list based on available data
        base_cols = ["Ticker", "Trend", "Touch", "Last", "SMA20", "SMA200", "Dist to SMA200 (%)", "Dist to SMA20 (%)"]
        indicator_cols = []
        if "RSI" in screener_df.columns:
            indicator_cols.append("RSI")
        if "MACD" in screener_df.columns:
            indicator_cols.extend(["MACD", "MACD Signal", "MACD Hist"])
        if "Vol Ratio" in screener_df.columns:
            indicator_cols.extend(["Volume", "Vol Ratio"])
        if "BB Position %" in screener_df.columns:
            indicator_cols.append("BB Position %")
        
        show_cols = base_cols + indicator_cols
        # Filter to only columns that exist
        show_cols = [c for c in show_cols if c in screener_df.columns]
        
        format_dict = {
            "Last": "${:,.2f}",
            "SMA20": "${:,.2f}",
            "SMA200": "${:,.2f}",
            "Dist to SMA200 (%)": "{:+.2f}%",
            "Dist to SMA20 (%)": "{:+.2f}%",
            "RSI": "{:.1f}",
            "MACD": "{:.2f}",
            "MACD Signal": "{:.2f}",
            "MACD Hist": "{:.2f}",
            "Volume": "{:,.0f}",
            "Vol SMA20": "{:,.0f}",
            "Vol Ratio": "{:.2f}x",
            "BB Position %": "{:.1f}%"
        }
        
        s = screener_df[show_cols].style.apply(
            lambda col: [_badge_color(v) for v in col], subset=["Trend"]
        ).format({k: v for k, v in format_dict.items() if k in show_cols})
        st.dataframe(s, use_container_width=True, hide_index=True)
        st.download_button(
            label="Download Screener CSV",
            data=screener_df.to_csv(index=False).encode("utf-8"),
            file_name="pretouch_screener.csv",
            mime="text/csv",
            key="dl_screener_csv"
        )

    # --- Footnote / Diagnostics ---
    with st.expander("About / Diagnostics"):
        st.write("If crypto history appears short, CoinGecko free API caps range to ~365 days. "
                 "The app falls back between Yahoo and CoinGecko automatically.")
        st.write("Built with Streamlit, yfinance, CoinGecko, Plotly. No paid data sources.")

# ===== TAB 2: ALERTS =====
with tab2:
    st.header("🔔 Alerts & Notifications")
    st.markdown("Set up alerts for cross events, price levels, and RSI conditions.")
    
    alerts = load_alerts()
    
    # Display existing alerts
    if alerts:
        st.subheader("Active Alerts")
        for i, alert in enumerate(alerts):
            col1, col2, col3, col4 = st.columns([3, 2, 1, 1])
            with col1:
                alert_type_names = {
                    "golden_cross": "Golden Cross",
                    "death_cross": "Death Cross",
                    "price_above": f"Price Above ${alert.get('value', 0):.2f}",
                    "price_below": f"Price Below ${alert.get('value', 0):.2f}",
                    "rsi_overbought": f"RSI Overbought (≥{alert.get('value', 70)})",
                    "rsi_oversold": f"RSI Oversold (≤{alert.get('value', 30)})"
                }
                st.write(f"**{alert.get('ticker', 'N/A')}** - {alert_type_names.get(alert.get('type', ''), alert.get('type', ''))}")
            with col2:
                status = "✅ Enabled" if alert.get("enabled", True) else "❌ Disabled"
                st.write(status)
            with col3:
                if st.button("Toggle", key=f"toggle_{i}"):
                    alerts[i]["enabled"] = not alerts[i].get("enabled", True)
                    save_alerts(alerts)
                    st.rerun()
            with col4:
                if st.button("Delete", key=f"delete_{i}"):
                    alerts.pop(i)
                    save_alerts(alerts)
                    st.rerun()
        st.divider()
    
    # Add new alert
    st.subheader("Add New Alert")
    alert_col1, alert_col2 = st.columns(2)
    with alert_col1:
        alert_ticker = st.text_input("Ticker", placeholder="e.g., SPY", key="alert_ticker")
        alert_type = st.selectbox("Alert Type", [
            "golden_cross", "death_cross", "price_above", "price_below", 
            "rsi_overbought", "rsi_oversold"
        ], key="alert_type")
    
    with alert_col2:
        alert_value = None
        if alert_type in ["price_above", "price_below"]:
            alert_value = st.number_input("Price Threshold ($)", min_value=0.0, value=0.0, step=0.01, key="alert_price")
        elif alert_type in ["rsi_overbought", "rsi_oversold"]:
            alert_value = st.number_input("RSI Threshold", min_value=0.0, max_value=100.0, 
                                         value=70.0 if alert_type == "rsi_overbought" else 30.0, 
                                         step=1.0, key="alert_rsi")
    
    if st.button("Add Alert", key="add_alert"):
        if alert_ticker.strip():
            new_alert = {
                "ticker": alert_ticker.strip().upper(),
                "type": alert_type,
                "enabled": True,
                "created": datetime.now().isoformat()
            }
            if alert_value is not None:
                new_alert["value"] = float(alert_value)
            alerts.append(new_alert)
            save_alerts(alerts)
            st.success(f"Alert added for {new_alert['ticker']}")
            st.rerun()
    
    # Email Configuration
    st.divider()
    st.subheader("📧 Email Configuration")
    email_config = load_email_config()
    
    with st.expander("Configure Email Alerts"):
        email_from = st.text_input("From Email", value=email_config.get("from_email", ""), 
                                   placeholder="your.email@gmail.com", key="email_from")
        email_password = st.text_input("Password/App Password", type="password", 
                                      value=email_config.get("password", ""), key="email_password")
        email_to = st.text_input("Default To Email", value=email_config.get("default_email", ""), 
                                placeholder="recipient@example.com", key="email_to")
        smtp_server = st.text_input("SMTP Server", value=email_config.get("smtp_server", "smtp.gmail.com"), key="smtp_server")
        smtp_port = st.number_input("SMTP Port", value=email_config.get("smtp_port", 587), min_value=1, max_value=65535, key="smtp_port")
        
        if st.button("Save Email Config", key="save_email_config"):
            email_config = {
                "from_email": email_from,
                "password": email_password,
                "default_email": email_to,
                "smtp_server": smtp_server,
                "smtp_port": int(smtp_port)
            }
            save_email_config(email_config)
            st.success("Email configuration saved!")
            st.info("💡 For Gmail, use an App Password instead of your regular password. Enable 2FA and generate an app password.")
    
    # Alert History
    st.divider()
    st.subheader("📜 Alert History")
    alert_history = load_alert_history()
    if alert_history:
        recent_history = alert_history[-20:]  # Show last 20
        history_data = []
        for h in reversed(recent_history):
            history_data.append({
                "Time": h.get("timestamp", "")[:19] if len(h.get("timestamp", "")) > 19 else h.get("timestamp", ""),
                "Ticker": h.get("ticker", ""),
                "Type": h.get("type", ""),
                "Message": h.get("message", ""),
                "Email Sent": "✅" if h.get("email_sent") else "❌"
            })
        history_df = pd.DataFrame(history_data)
        st.dataframe(history_df, use_container_width=True, hide_index=True)
    else:
        st.info("No alert history yet.")

# ===== TAB 3: CROSS HISTORY =====
with tab3:
    st.header("📊 Historical Cross Analysis")
    st.markdown("Analyze past Golden and Death Cross events and their performance.")
    
    hist_ticker = st.selectbox("Select Ticker", selected or universe, key="hist_ticker")
    days_after = st.slider("Analyze performance after (days)", min_value=7, max_value=180, value=30, step=7, key="hist_days")
    
    if st.button("Analyze Crosses", key="analyze_crosses"):
        df = load_data(hist_ticker, start_d, end_d, mode, interval=interval)
        if not df.empty:
            df = df.copy()
            df["SMA20"] = _sma(df["close"], SMA_SHORT)
            df["SMA200"] = _sma(df["close"], SMA_LONG)
            
            crosses = detect_crosses(df)
            crosses_with_perf = analyze_cross_performance(df, crosses, days_after)
            
            # Save to history
            history = load_cross_history()
            history[hist_ticker] = crosses_with_perf
            save_cross_history(history)
            
            if crosses_with_perf:
                st.subheader(f"Cross Events for {hist_ticker}")
                
                # Summary statistics
                golden_crosses = [c for c in crosses_with_perf if c["type"] == "Golden"]
                death_crosses = [c for c in crosses_with_perf if c["type"] == "Death"]
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Crosses", len(crosses_with_perf))
                with col2:
                    st.metric("Golden Crosses", len(golden_crosses))
                with col3:
                    st.metric("Death Crosses", len(death_crosses))
                with col4:
                    if golden_crosses:
                        avg_perf = np.mean([c["performance"]["pct_change"] for c in golden_crosses if c.get("performance")])
                        st.metric("Avg Golden Return", f"{avg_perf:.2f}%")
                
                # Display table
                display_data = []
                for cross in crosses_with_perf:
                    row = {
                        "Date": cross["date"],
                        "Type": cross["type"],
                        "Price at Cross": f"${cross['price']:.2f}",
                        "SMA20": f"${cross['sma20']:.2f}",
                        "SMA200": f"${cross['sma200']:.2f}"
                    }
                    if cross.get("performance"):
                        perf = cross["performance"]
                        row["Price After"] = f"${perf['price_after']:.2f}"
                        row["Return %"] = f"{perf['pct_change']:+.2f}%"
                        row["Days"] = f"{perf['days_actual']}"
                    else:
                        row["Price After"] = "N/A"
                        row["Return %"] = "N/A"
                        row["Days"] = "N/A"
                    display_data.append(row)
                
                df_display = pd.DataFrame(display_data)
                st.dataframe(df_display, use_container_width=True, hide_index=True)
            else:
                st.info(f"No cross events found for {hist_ticker} in the selected period.")
        else:
            st.error(f"No data available for {hist_ticker}")

# ===== TAB 4: PORTFOLIO =====
with tab4:
    st.header("💼 Portfolio Tracking")
    st.markdown("Track multiple tickers as a portfolio with custom weights.")
    
    portfolio = load_portfolio()
    
    # Portfolio tickers
    st.subheader("Portfolio Tickers")
    portfolio_tickers = st.multiselect(
        "Select tickers for portfolio",
        options=selected or universe,
        default=portfolio.get("tickers", []),
        key="portfolio_tickers"
    )
    
    if portfolio_tickers:
        st.subheader("Portfolio Weights")
        st.caption("Weights will be normalized to sum to 100%. Leave empty for equal weighting.")
        
        weights = {}
        for ticker in portfolio_tickers:
            current_weight = portfolio.get("weights", {}).get(ticker, 0.0)
            weight = st.number_input(
                f"{ticker} Weight (%)",
                min_value=0.0,
                max_value=100.0,
                value=current_weight if current_weight > 0 else (100.0 / len(portfolio_tickers)),
                step=1.0,
                key=f"weight_{ticker}"
            )
            weights[ticker] = weight
        
        if st.button("Update Portfolio", key="update_portfolio"):
            portfolio = {"tickers": portfolio_tickers, "weights": weights}
            save_portfolio(portfolio)
            st.success("Portfolio updated!")
            st.rerun()
        
        # Calculate and display portfolio metrics
        if st.button("Calculate Portfolio Metrics", key="calc_portfolio"):
            metrics = calculate_portfolio_metrics(portfolio, start_d, end_d, mode)
            
            if "error" not in metrics:
                st.subheader("Portfolio Performance")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Portfolio Return", f"{metrics['portfolio_return']:.2f}%")
                with col2:
                    st.metric("Number of Tickers", metrics['ticker_count'])
                with col3:
                    st.metric("Portfolio Value", f"${metrics['portfolio_value']:,.2f}")
                
                st.subheader("Individual Performance")
                perf_data = []
                for ticker, data in metrics['individual_returns'].items():
                    perf_data.append({
                        "Ticker": ticker,
                        "Weight": f"{data['weight']*100:.1f}%",
                        "Return": f"{data['return']:.2f}%",
                        "Price": f"${data['price']:,.2f}"
                    })
                perf_df = pd.DataFrame(perf_data)
                st.dataframe(perf_df, use_container_width=True, hide_index=True)
            else:
                st.error(metrics["error"])

# ===== TAB 5: BACKTESTING =====
with tab5:
    st.header("🧪 Strategy Backtesting")
    st.markdown("Test trading strategies based on SMA crosses and compare performance.")
    
    backtest_ticker = st.selectbox("Select Ticker for Backtest", selected or universe, key="backtest_ticker")
    strategy = st.selectbox("Strategy", ["golden_death", "buy_hold"], key="backtest_strategy")
    initial_capital = st.number_input("Initial Capital ($)", min_value=1000.0, value=10000.0, step=1000.0, key="backtest_capital")
    
    strategy_names = {
        "golden_death": "Golden/Death Cross Strategy (Buy on Golden Cross, Sell on Death Cross)",
        "buy_hold": "Buy and Hold Strategy"
    }
    st.caption(f"**Strategy:** {strategy_names.get(strategy, strategy)}")
    
    if st.button("Run Backtest", key="run_backtest"):
        df = load_data(backtest_ticker, start_d, end_d, mode, interval=interval)
        if not df.empty:
            result = backtest_strategy(df, strategy, initial_capital)
            
            if "error" not in result:
                st.subheader("Backtest Results")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Initial Capital", f"${result['initial_capital']:,.2f}")
                with col2:
                    st.metric("Final Capital", f"${result['final_capital']:,.2f}")
                with col3:
                    st.metric("Total Return", f"{result['total_return']:.2f}%")
                with col4:
                    st.metric("Max Drawdown", f"{result['max_drawdown']:.2f}%")
                
                st.metric("Number of Trades", result['num_trades'])
                
                # Trade history
                if result['trades']:
                    st.subheader("Trade History")
                    trades_data = []
                    for trade in result['trades']:
                        trade_date = trade['date']
                        if isinstance(trade_date, pd.Timestamp):
                            trade_date = trade_date.strftime("%Y-%m-%d")
                        trades_data.append({
                            "Date": trade_date,
                            "Type": trade['type'],
                            "Price": f"${trade['price']:.2f}",
                            "Shares": f"{trade.get('shares', 0):.4f}" if 'shares' in trade else "N/A",
                            "Capital": f"${trade.get('capital', 0):,.2f}" if 'capital' in trade else "N/A"
                        })
                    trades_df = pd.DataFrame(trades_data)
                    st.dataframe(trades_df, use_container_width=True, hide_index=True)
            else:
                st.error(result["error"])
        else:
            st.error(f"No data available for {backtest_ticker}")

# ===== TAB 6: NEWS =====
with tab6:
    st.header("📰 News & Sentiment")
    news_ticker = st.selectbox("Select Ticker for News", selected or universe, key="news_ticker")
    
    if st.button("Fetch News", key="fetch_news"):
        news_items = fetch_news(news_ticker, max_items=10)
        if news_items:
            for item in news_items:
                with st.expander(f"**{item.get('title', 'No title')}** - {item.get('publisher', 'Unknown')}"):
                    st.write(f"**Published:** {item.get('providerPublishTime', 'Unknown')}")
                    if item.get('link'):
                        st.markdown(f"[Read Article]({item['link']})")
                    if item.get('summary'):
                        st.write(item['summary'])
        else:
            st.info(f"No news found for {news_ticker}")

# ===== TAB 7: PATTERNS =====
with tab7:
    st.header("📐 Chart Pattern Detection")
    pattern_ticker = st.selectbox("Select Ticker for Pattern Analysis", selected or universe, key="pattern_ticker")
    
    if st.button("Analyze Patterns", key="analyze_patterns"):
        df = load_data(pattern_ticker, start_d, end_d, mode, interval=interval)
        if not df.empty:
            patterns = detect_patterns(df)
            if patterns:
                st.subheader(f"Detected Patterns for {pattern_ticker}")
                for pattern in patterns:
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.write(f"**{pattern['type']}**")
                        st.caption(pattern.get('description', ''))
                    with col2:
                        confidence_pct = pattern.get('confidence', 0) * 100
                        st.metric("Confidence", f"{confidence_pct:.0f}%")
            else:
                st.info(f"No patterns detected for {pattern_ticker}")
        else:
            st.error(f"No data available for {pattern_ticker}")

# ===== TAB 8: CORRELATION =====
with tab8:
    st.header("🔗 Correlation Analysis")
    st.markdown("Analyze correlation between multiple tickers.")
    
    if len(selected) >= 2:
        correlation_df = calculate_correlation(selected, start_d, end_d, mode, interval=interval)
        if not correlation_df.empty:
            st.subheader("Correlation Matrix")
            # Create heatmap
            fig_corr = go.Figure(data=go.Heatmap(
                z=correlation_df.values,
                x=correlation_df.columns,
                y=correlation_df.index,
                colorscale='RdBu',
                zmid=0,
                text=correlation_df.values,
                texttemplate='%{text:.2f}',
                textfont={"size": 10},
                colorbar=dict(title="Correlation")
            ))
            fig_corr.update_layout(
                title="Correlation Matrix",
                width=800,
                height=600
            )
            st.plotly_chart(fig_corr, use_container_width=True)
            
            st.subheader("Correlation Data")
            # Format correlation values for display
            formatted_df = correlation_df.round(3)
            st.dataframe(formatted_df, use_container_width=True)
        else:
            st.warning("Could not calculate correlation. Ensure all tickers have data.")
    else:
        st.info("Select at least 2 tickers to analyze correlation.")

# ===== TAB 9: TRADE JOURNAL =====
with tab9:
    st.header("📝 Trade Journal")
    st.markdown("Log your trades and track performance.")
    
    journal = load_trade_journal()
    
    # Add new trade
    st.subheader("Add New Trade")
    with st.form("add_trade_form"):
        col1, col2 = st.columns(2)
        with col1:
            trade_ticker = st.text_input("Ticker", key="trade_ticker")
            trade_type = st.selectbox("Type", ["Buy", "Sell"], key="trade_type")
            trade_price = st.number_input("Price", min_value=0.0, step=0.01, key="trade_price")
        with col2:
            trade_shares = st.number_input("Shares", min_value=0.0, step=0.01, key="trade_shares")
            trade_date = st.date_input("Date", value=date.today(), key="trade_date")
            trade_notes = st.text_area("Notes", key="trade_notes")
        
        if st.form_submit_button("Add Trade"):
            if trade_ticker and trade_price > 0 and trade_shares > 0:
                new_trade = {
                    "ticker": trade_ticker.upper(),
                    "type": trade_type,
                    "price": float(trade_price),
                    "shares": float(trade_shares),
                    "date": trade_date.isoformat(),
                    "notes": trade_notes,
                    "timestamp": datetime.now().isoformat()
                }
                journal.append(new_trade)
                save_trade_journal(journal)
                st.success("Trade added!")
                st.rerun()
    
    # Display trades
    if journal:
        st.subheader("Trade History")
        journal_df = pd.DataFrame(journal)
        journal_df = journal_df.sort_values("date", ascending=False)
        
        # Calculate P&L if we have current prices
        if st.checkbox("Calculate P&L", key="calc_pnl"):
            for idx, trade in journal_df.iterrows():
                ticker = trade["ticker"]
                df_current = load_data(ticker, start_d, end_d, mode, interval=interval)
                if not df_current.empty:
                    current_price = float(df_current["close"].iloc[-1])
                    if trade["type"] == "Buy":
                        pnl = (current_price - trade["price"]) * trade["shares"]
                        pnl_pct = ((current_price - trade["price"]) / trade["price"]) * 100
                    else:
                        pnl = (trade["price"] - current_price) * trade["shares"]
                        pnl_pct = ((trade["price"] - current_price) / current_price) * 100
                    journal_df.at[idx, "Current Price"] = current_price
                    journal_df.at[idx, "P&L"] = pnl
                    journal_df.at[idx, "P&L %"] = pnl_pct
        
        st.dataframe(journal_df, use_container_width=True, hide_index=True)
        
        # Summary
        if len(journal) > 0:
            st.subheader("Summary")
            total_trades = len(journal)
            buy_trades = len([t for t in journal if t.get("type") == "Buy"])
            sell_trades = len([t for t in journal if t.get("type") == "Sell"])
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Trades", total_trades)
            with col2:
                st.metric("Buy Trades", buy_trades)
            with col3:
                st.metric("Sell Trades", sell_trades)
    else:
        st.info("No trades logged yet. Add your first trade above.")


