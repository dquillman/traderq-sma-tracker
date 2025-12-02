"""
Technical Indicators module for TraderQ
Contains all technical analysis calculations: SMA, EMA, RSI, MACD, Bollinger Bands, Supertrend, FVG, etc.
"""

import pandas as pd
import numpy as np


# --- Basic Moving Averages ---
def _sma(series: pd.Series, window: int) -> pd.Series:
    """Calculate Simple Moving Average"""
    return series.rolling(window=window, min_periods=1).mean()


def _ema(series: pd.Series, window: int) -> pd.Series:
    """Calculate Exponential Moving Average"""
    return series.ewm(span=window, adjust=False).mean()


def _pct(a: float, b: float) -> float:
    """Calculate percentage change between two values"""
    if b == 0 or (b is None) or (a is None) or np.isnan(a) or np.isnan(b):
        return np.nan
    return (a / b - 1.0) * 100.0


# --- Technical Indicators ---
def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Calculate Relative Strength Index"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate MACD (Moving Average Convergence Divergence)
    Returns: (MACD line, Signal line, Histogram)
    """
    ema_fast = _ema(series, fast)
    ema_slow = _ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _sideways_mask(price: pd.Series, window: int = 10, threshold: float = 0.08) -> pd.Series:
    """
    Detect sideways / range-bound conditions.
    We look at the rolling high-low range relative to the rolling mean price.
    If the range is below `threshold` (e.g. ~8%), we consider that window sideways.
    Returns a boolean Series aligned with `price`.
    """
    rolling_max = price.rolling(window=window, min_periods=window).max()
    rolling_min = price.rolling(window=window, min_periods=window).min()
    rolling_mean = price.rolling(window=window, min_periods=window).mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        price_range_pct = (rolling_max - rolling_min) / rolling_mean
    return price_range_pct < threshold


def _macd_extended(series: pd.Series,
                   fast: int = 12,
                   slow: int = 26,
                   signal: int = 9,
                   sideways_window: int = 10,
                   sideways_threshold: float = 0.08) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Extended MACD that flattens during sideways markets.
    - Computes standard MACD
    - Detects sideways/range-bound conditions using `_sideways_mask`
    - During sideways periods, MACD, signal, and histogram are forced toward 0,
      so the indicator appears flat when the market is chopping sideways
    """
    macd_line, signal_line, histogram = _macd(series, fast=fast, slow=slow, signal=signal)
    mask = _sideways_mask(series, window=sideways_window, threshold=sideways_threshold)

    # Make copies so we don't mutate the originals
    macd_ext = macd_line.copy()
    signal_ext = signal_line.copy()
    hist_ext = histogram.copy()

    macd_ext[mask] = 0.0
    signal_ext[mask] = 0.0
    hist_ext[mask] = 0.0

    return macd_ext, signal_ext, hist_ext


def _bollinger_bands(series: pd.Series, window: int = 20, num_std: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Bollinger Bands
    Returns: (Upper band, Middle band (SMA), Lower band)
    """
    middle = _sma(series, window)
    std = series.rolling(window=window).std()
    upper = middle + (std * num_std)
    lower = middle - (std * num_std)
    return upper, middle, lower


def _volume_sma(volume: pd.Series, window: int = 20) -> pd.Series:
    """Calculate volume moving average"""
    return _sma(volume, window)


def _vwap(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Calculate Volume Weighted Average Price"""
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    vwap = (typical_price * df["volume"]).rolling(window=window).sum() / df["volume"].rolling(window=window).sum()
    return vwap


def _atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Calculate Average True Range"""
    high_low = df["high"] - df["low"]
    high_close = np.abs(df["high"] - df["close"].shift())
    low_close = np.abs(df["low"] - df["close"].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    return true_range.rolling(window=window).mean()


def _supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> tuple[pd.Series, pd.Series]:
    """
    Calculate Supertrend indicator
    Returns: (supertrend_line, trend_direction) where trend_direction is 1 for uptrend, -1 for downtrend
    """
    atr = _atr(df, window=period)
    hl_avg = (df["high"] + df["low"]) / 2

    # Initialize bands
    upper_band = hl_avg + (multiplier * atr)
    lower_band = hl_avg - (multiplier * atr)

    # Convert to numpy arrays for iteration
    upper_values = upper_band.values
    lower_values = lower_band.values
    close_values = df["close"].values

    supertrend_values = np.full(len(df), np.nan)
    trend_values = np.full(len(df), np.nan)

    # Calculate Supertrend
    for i in range(len(df)):
        if i == 0:
            supertrend_values[i] = upper_values[i]
            trend_values[i] = 1
            continue

        # Adjust upper band
        if close_values[i] <= upper_values[i-1]:
            upper_values[i] = min(upper_values[i], upper_values[i-1])

        # Adjust lower band
        if close_values[i] >= lower_values[i-1]:
            lower_values[i] = max(lower_values[i], lower_values[i-1])

        # Determine trend
        if close_values[i] <= supertrend_values[i-1]:
            supertrend_values[i] = upper_values[i]
            trend_values[i] = -1
        else:
            supertrend_values[i] = lower_values[i]
            trend_values[i] = 1

    supertrend = pd.Series(supertrend_values, index=df.index)
    trend = pd.Series(trend_values, index=df.index)

    return supertrend, trend


# --- Fair Value Gap Detection ---
def find_fair_value_gaps(df: pd.DataFrame) -> list:
    """
    Detect Fair Value Gaps (FVG) in price action.
    FVG is a 3-candle pattern where:
    - Candle 1 and Candle 3 overlap
    - Candle 2 creates a gap (doesn't overlap with Candle 1 or 3)

    Returns list of FVG dictionaries with start/end price and date range.
    """
    if df.empty or len(df) < 3:
        return []

    if "high" not in df.columns or "low" not in df.columns:
        return []

    fvgs = []

    for i in range(len(df) - 2):
        c1_high = df["high"].iloc[i]
        c1_low = df["low"].iloc[i]
        c2_high = df["high"].iloc[i + 1]
        c2_low = df["low"].iloc[i + 1]
        c3_high = df["high"].iloc[i + 2]
        c3_low = df["low"].iloc[i + 2]

        # Bullish FVG: Gap up (candle 2 is above candle 1, and candle 3 overlaps with candle 1)
        if (c2_low > c1_high and c3_low <= c1_high and c3_high >= c1_low):
            fvgs.append({
                "type": "bullish",
                "start_price": c1_high,  # Gap starts here
                "end_price": c2_low,     # Gap ends here
                "start_date": df.index[i],
                "end_date": df.index[i + 2],
                "gap_high": c2_low,      # Top of the gap (candle 2 low)
                "gap_low": c1_high       # Bottom of the gap (candle 1 high)
            })

        # Bearish FVG: Gap down (candle 2 is below candle 1, and candle 3 overlaps with candle 1)
        elif (c2_high < c1_low and c3_high >= c1_low and c3_low <= c1_high):
            fvgs.append({
                "type": "bearish",
                "start_price": c1_low,   # Gap starts here
                "end_price": c2_high,    # Gap ends here
                "start_date": df.index[i],
                "end_date": df.index[i + 2],
                "gap_high": c1_low,      # Top of the gap (candle 1 low)
                "gap_low": c2_high       # Bottom of the gap (candle 2 high)
            })

    return fvgs


# --- Support & Resistance Detection ---
def find_support_resistance(df: pd.DataFrame, window: int = 20, lookback: int = 5) -> dict:
    """Find support and resistance levels using pivot points"""
    if df.empty or "high" not in df.columns or "low" not in df.columns:
        return {"support": [], "resistance": []}

    highs = df["high"].rolling(window=window, center=True).apply(
        lambda x: x[len(x)//2] if x[len(x)//2] == x.max() else np.nan,
        raw=True
    )
    lows = df["low"].rolling(window=window, center=True).apply(
        lambda x: x[len(x)//2] if x[len(x)//2] == x.min() else np.nan,
        raw=True
    )

    # Get resistance levels (peaks)
    resistance_levels = highs.dropna().values
    resistance_dates = df.index[highs.notna()].tolist()

    # Get support levels (troughs)
    support_levels = lows.dropna().values
    support_dates = df.index[lows.notna()].tolist()

    # Take most recent N levels
    resistance = [
        {"price": float(resistance_levels[i]), "date": resistance_dates[i]}
        for i in range(max(0, len(resistance_levels) - lookback), len(resistance_levels))
    ]
    support = [
        {"price": float(support_levels[i]), "date": support_dates[i]}
        for i in range(max(0, len(support_levels) - lookback), len(support_levels))
    ]

    return {"support": support, "resistance": resistance}


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
