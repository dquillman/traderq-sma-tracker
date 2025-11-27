"""
Data loading module for TraderQ
Handles fetching stock and crypto data from Yahoo Finance, CoinGecko, and Google Finance
"""

import pandas as pd
import numpy as np
import streamlit as st
from datetime import date, datetime, timedelta
from functools import lru_cache
import time


# --- Lazy imports for data providers ---
@lru_cache(maxsize=1)
def _yf():
    """Lazy load yfinance module"""
    import yfinance as yf
    return yf


@lru_cache(maxsize=1)
def _cg():
    """Lazy load CoinGecko API"""
    from pycoingecko import CoinGeckoAPI
    return CoinGeckoAPI()


@lru_cache(maxsize=1)
def _gf():
    """Initialize Google Finance scraper"""
    import requests
    return requests


# --- Utility Functions ---
def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure DataFrame has a timezone-naive datetime index"""
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=False)
    # Ensure timezone-naive index to avoid UTC offset issues
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
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


def _cg_id_for(symbol_usd: str) -> str | None:
    """Crude mapping for popular coins to CoinGecko IDs"""
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


# --- Main Data Loading Functions ---
@st.cache_data(show_spinner=False)
def load_stock(ticker: str, start: date, end: date, interval: str = "1d") -> pd.DataFrame:
    """Load stock data from Yahoo Finance"""
    yf = _yf()

    # For intraday intervals (5m, 15m, etc.), we need to adjust the period
    # yfinance intraday data is typically available for last 60 days max
    if interval in ["1m", "2m", "5m", "15m", "30m", "60m", "90m"]:
        # Limit to last 60 days for intraday data
        max_start = date.today() - timedelta(days=60)
        if start < max_start:
            start = max_start

    # New yfinance (0.2.66+) uses curl_cffi and handles sessions automatically
    # Try using Ticker API first (more reliable)
    try:
        stock = yf.Ticker(ticker)  # Don't pass session - let yfinance handle it
        df = stock.history(start=start, end=(end + timedelta(days=1)), interval=interval, auto_adjust=True)
        if df is not None and len(df) > 0:
            df = _ensure_datetime_index(df)
            df = _to_ohlc(df)
            # For 5-minute interval, limit to last 500 candles if we have more
            if interval == "5m" and len(df) > 500:
                df = df.tail(500)
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
                # For 5-minute interval, limit to last 500 candles if we have more
                if interval == "5m" and len(df) > 500:
                    df = df.tail(500)
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


@st.cache_data(show_spinner=False)
def load_crypto(ticker: str, start: date, end: date) -> pd.DataFrame:
    """Try Yahoo Finance first for -USD pairs; fallback to CoinGecko (capped to 365d)"""
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

        # Build OHLC from prices (minute/daily granularity → we'll downsample to 1D)
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


@st.cache_data(show_spinner=False, ttl=60)  # Cache for 60 seconds
def load_google_finance_stock(ticker: str, start: date, end: date, interval: str = "1d",
                              force_refresh: bool = False) -> pd.DataFrame:
    """Load stock data from Google Finance - uses real-time data fetching"""
    try:
        import requests
        from bs4 import BeautifulSoup
        import re

        # Google Finance URL format
        quote_url = f"https://www.google.com/finance/quote/{ticker}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }

        # Try to fetch real-time data from Google Finance
        response = requests.get(quote_url, headers=headers, timeout=15)

        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')

            # Try to extract current price (real-time data)
            price_elem = soup.find('div', class_=re.compile('YMlKec|fxKbKc')) or \
                        soup.find('div', jsname='vWLAgc') or \
                        soup.find('div', class_='AHmHk') or \
                        soup.find('div', {'data-last-price': True})

            current_price = None
            if price_elem:
                try:
                    # Try to get price from text
                    price_text = price_elem.text.strip()
                    # Remove currency symbols and commas
                    price_text = re.sub(r'[^\d.]', '', price_text)
                    if price_text:
                        current_price = float(price_text)
                except (ValueError, AttributeError):
                    # Try data attribute
                    if price_elem.get('data-last-price'):
                        current_price = float(price_elem.get('data-last-price'))

            # Try to get historical data from Google Finance chart data
            chart_data_pattern = re.search(r'var\s+chartData\s*=\s*(\[.*?\]);', response.text, re.DOTALL)
            if not chart_data_pattern:
                chart_data_pattern = re.search(r'chartData\s*:\s*(\[.*?\]),', response.text, re.DOTALL)

            if chart_data_pattern:
                try:
                    import json
                    chart_data = json.loads(chart_data_pattern.group(1))
                    if chart_data and len(chart_data) > 0:
                        # Process chart data into DataFrame
                        dates = []
                        prices = []
                        for item in chart_data:
                            if isinstance(item, list) and len(item) >= 2:
                                ts = pd.to_datetime(item[0], unit='ms')
                                if ts.tz is not None:
                                    ts = ts.tz_localize(None)
                                dates.append(ts)
                                if len(item) >= 5:
                                    prices.append({
                                        'open': item[1],
                                        'high': item[2],
                                        'low': item[3],
                                        'close': item[4],
                                        'volume': item[5] if len(item) > 5 else 0
                                    })

                        if dates and prices:
                            df = pd.DataFrame(prices, index=dates)
                            df = df.sort_index()
                            if df.index.tz is not None:
                                df.index = df.index.tz_localize(None)
                            df = df.loc[(df.index.date >= start) & (df.index.date <= end)]
                            if len(df) > 0:
                                df = _ensure_datetime_index(df)
                                df = _to_ohlc(df)
                                # Add current real-time price if available
                                if current_price and len(df) > 0:
                                    last_date = df.index[-1]
                                    today = date.today()
                                    if last_date.date() < today:
                                        new_row = pd.DataFrame({
                                            'open': [current_price],
                                            'high': [current_price],
                                            'low': [current_price],
                                            'close': [current_price],
                                            'volume': [0]
                                        }, index=[pd.Timestamp(today)])
                                        df = pd.concat([df, new_row])
                                return df
                except (json.JSONDecodeError, ValueError, KeyError) as e:
                    import sys
                    print(f"Failed to parse Google Finance chart data: {e}", file=sys.stderr)

        # Fallback: Use pandas_datareader if available
        try:
            import pandas_datareader.data as web
            df = web.DataReader(ticker, 'google', start, end)
            if df is not None and len(df) > 0:
                df = _ensure_datetime_index(df)
                df = _to_ohlc(df)
                return df
        except (ImportError, Exception):
            pass

        # Final fallback: Use yfinance
        yf = _yf()
        try:
            if interval in ["1m", "2m", "5m", "15m", "30m", "60m", "90m"]:
                max_start = date.today() - timedelta(days=60)
                if start < max_start:
                    start = max_start

            stock = yf.Ticker(ticker)
            df = stock.history(start=start, end=(end + timedelta(days=1)), interval=interval, auto_adjust=True)
            if df is not None and len(df) > 0:
                df = _ensure_datetime_index(df)
                df = _to_ohlc(df)
                if interval == "5m" and len(df) > 500:
                    df = df.tail(500)
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                # Try to update with real-time price
                if current_price and len(df) > 0:
                    last_date = df.index[-1]
                    today = date.today()
                    if last_date.date() < today:
                        new_row = pd.DataFrame({
                            'open': [current_price],
                            'high': [current_price],
                            'low': [current_price],
                            'close': [current_price],
                            'volume': [0]
                        }, index=[pd.Timestamp(today)])
                        df = pd.concat([df, new_row])
                return df
        except Exception:
            pass

        return pd.DataFrame()
    except Exception as e:
        import sys
        print(f"Google Finance load failed for {ticker}: {e}", file=sys.stderr)
        # Final fallback to yfinance
        try:
            yf = _yf()
            stock = yf.Ticker(ticker)
            df = stock.history(start=start, end=(end + timedelta(days=1)), interval=interval, auto_adjust=True)
            if df is not None and len(df) > 0:
                df = _ensure_datetime_index(df)
                df = _to_ohlc(df)
                return df
        except Exception:
            pass
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_data(ticker: str, start: date, end: date, mode: str, interval: str = "1d", data_source: str = "Yahoo Finance") -> pd.DataFrame:
    """Load data from specified data source"""
    if mode == "Stocks":
        if data_source == "Google Finance":
            return load_google_finance_stock(ticker, start, end, interval=interval)
        else:
            return load_stock(ticker, start, end, interval=interval)

    # Crypto doesn't support intervals in the same way, but we can resample if needed
    if data_source == "Google Finance":
        df = load_google_finance_stock(ticker, start, end, interval=interval)
    else:
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
