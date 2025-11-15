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

APP_VERSION = "v1.5.1"
# v1.5.1 – fixed yfinance API compatibility (upgraded to 0.2.66), improved data loading

# --- Settings / Defaults ---
DEFAULT_STOCKS = ["^GSPC", "^DJI", "^IXIC", "SPY", "QQQ"]
DEFAULT_CRYPTOS = ["BTC-USD", "ETH-USD", "SOL-USD"]
DEFAULT_PERIOD_DAYS = 365 * 2  # 2 years for stocks; crypto API may cap to 365 in fallback
SMA_SHORT = 20
SMA_LONG = 200

# --- Custom ticker persistence ---
CUSTOM_TICKERS_FILE = Path(__file__).parent / ".custom_tickers.json"

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

def _badge_color(trend: str) -> str:
    if trend == "Bullish":
        return "background-color:#12391a;color:#1fd16c;font-weight:600"
    if trend == "Bearish":
        return "background-color:#3a1919;color:#ff4d4f;font-weight:600"
    return ""

# --- Data loaders ---
@st.cache_data(show_spinner=False)
def load_stock(ticker: str, start: date, end: date) -> pd.DataFrame:
    yf = _yf()
    
    # New yfinance (0.2.66+) uses curl_cffi and handles sessions automatically
    # Try using Ticker API first (more reliable)
    try:
        stock = yf.Ticker(ticker)  # Don't pass session - let yfinance handle it
        df = stock.history(start=start, end=(end + timedelta(days=1)), auto_adjust=True)
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
def load_data(ticker: str, start: date, end: date, mode: str) -> pd.DataFrame:
    if mode == "Stocks":
        return load_stock(ticker, start, end)
    return load_crypto(ticker, start, end)

# --- Chart builder ---
def make_chart(df: pd.DataFrame, title: str, theme: str, pretouch_pct: float | None,
               show_volume: bool = True, show_rsi: bool = True, show_macd: bool = True,
               show_bollinger: bool = True) -> go.Figure:
    if df.empty:
        fig = go.Figure()
        fig.update_layout(template="plotly_dark" if theme == "Dark" else "plotly_white", title=title)
        return fig

    df = df.copy()
    df["SMA20"] = _sma(df["close"], SMA_SHORT)
    df["SMA200"] = _sma(df["close"], SMA_LONG)

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
    
    # Main price chart
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        name="Price"
    ), row=row, col=1)
    
    fig.add_trace(go.Scatter(
        x=df.index, y=df["SMA20"], mode="lines", name=f"SMA {SMA_SHORT}", 
        line=dict(width=2, color="#4fa3ff")
    ), row=row, col=1)
    
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
        height=400 + (200 * (num_subplots - 1))
    )
    
    return fig

# --- Screener helper ---
def build_screener(tickers: list[str], start: date, end: date, mode: str, pretouch_pct: float | None) -> pd.DataFrame:
    rows = []
    for t in tickers:
        d = load_data(t, start, end, mode)
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

# Sidebar controls
mode = st.sidebar.radio("Market", ["Stocks", "Crypto"], horizontal=True)
theme = st.sidebar.radio("Chart Theme", ["Dark", "Light"], index=0, horizontal=True)
pretouch = st.sidebar.slider("Pretouch band around SMA200 (%)", min_value=0.0, max_value=5.0, value=1.0, step=0.25)
period_days = st.sidebar.select_slider("Lookback (days)", options=[180, 365, 540, 730], value=365)

# Indicator toggles
st.sidebar.markdown("---")
st.sidebar.markdown("**Indicators**")
show_volume = st.sidebar.checkbox("Show Volume", value=True)
show_rsi = st.sidebar.checkbox("Show RSI", value=True)
show_macd = st.sidebar.checkbox("Show MACD", value=True)
show_bollinger = st.sidebar.checkbox("Show Bollinger Bands", value=True)

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

# Safety: unique list
selected = list(dict.fromkeys(selected))

# --- Per-ticker charts ---
for t in selected:
    df = load_data(t, start_d, end_d, mode)
    cols = st.columns([3, 1], gap="large")
    with cols[0]:
        st.markdown(f"**{t}**")
        fig = make_chart(df, f"{t} — SMA {SMA_SHORT}/{SMA_LONG}", theme, pretouch,
                        show_volume=show_volume, show_rsi=show_rsi, 
                        show_macd=show_macd, show_bollinger=show_bollinger)
        st.plotly_chart(fig, use_container_width=True, key=f"chart_{t}_{mode}")
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

# --- Screener ---
st.divider()
st.subheader("Pretouch Screener (closest to SMA200 on top)")

screener_df = build_screener(selected or universe, start_d, end_d, mode, pretouch)
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




