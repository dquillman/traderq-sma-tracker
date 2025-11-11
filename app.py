# app.py  — SMA 20/200 Tracker with Pretouch Bands
# Uses Yahoo Finance (yfinance). Works for stocks, indexes, ETFs, and crypto (e.g., BTC-USD).
# No paper trading. Just signals.

import datetime as dt
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="SMA 20/200 Tracker", page_icon="📈", layout="wide")

# -----------------------------
# Data
# -----------------------------
@st.cache_data(ttl=3600)
def load_data(ticker: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        return df
    df = df.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    df.index = pd.to_datetime(df.index)
    df["SMA20"] = df["close"].rolling(window=20, min_periods=20).mean()
    df["SMA200"] = df["close"].rolling(window=200, min_periods=200).mean()

    # Signals: above/below, then crossover detection
    sig = np.where(
        (df["SMA20"].notna()) & (df["SMA200"].notna()) & (df["SMA20"] > df["SMA200"]),
        1,
        np.where(
            (df["SMA20"].notna()) & (df["SMA200"].notna()) & (df["SMA20"] < df["SMA200"]),
            -1,
            0,
        ),
    )
    df["signal"] = sig
    df["xover"] = df["signal"].diff().fillna(0)
    return df

def fmt(x):
    try:
        return f"{x:,.2f}"
    except Exception:
        return "—"

# -----------------------------
# Chart
# -----------------------------
def make_chart(ticker: str, df: pd.DataFrame, pretouch_pct: float, watch_20: bool, watch_200: bool):
    fig = go.Figure()

    # Price candles
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name=f"{ticker}",
        )
    )

    # Core SMAs
    fig.add_trace(go.Scatter(x=df.index, y=df["SMA20"], mode="lines", name="SMA20"))
    fig.add_trace(go.Scatter(x=df.index, y=df["SMA200"], mode="lines", name="SMA200"))

    # Pretouch bands (±% around the chosen SMA lines)
    if pretouch_pct and pretouch_pct > 0:
        if watch_20 and df["SMA20"].notna().any():
            upper20 = df["SMA20"] * (1 + pretouch_pct / 100)
            lower20 = df["SMA20"] * (1 - pretouch_pct / 100)
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=lower20,
                    mode="lines",
                    line=dict(width=0.5),
                    name="SMA20 band (lower)",
                    showlegend=False,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=upper20,
                    mode="lines",
                    fill="tonexty",
                    line=dict(width=0.5),
                    name=f"±{pretouch_pct:.2f}% of SMA20",
                )
            )
        if watch_200 and df["SMA200"].notna().any():
            upper200 = df["SMA200"] * (1 + pretouch_pct / 100)
            lower200 = df["SMA200"] * (1 - pretouch_pct / 100)
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=lower200,
                    mode="lines",
                    line=dict(width=0.5),
                    name="SMA200 band (lower)",
                    showlegend=False,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=upper200,
                    mode="lines",
                    fill="tonexty",
                    line=dict(width=0.5),
                    name=f"±{pretouch_pct:.2f}% of SMA200",
                )
            )

    # Crossover markers
    xover_points = df.index[df["signal"].diff().fillna(0) != 0]
    golden_points = [i for i in xover_points if df.loc[i, "signal"] == 1]
    death_points = [i for i in xover_points if df.loc[i, "signal"] == -1]

    if golden_points:
        fig.add_trace(
            go.Scatter(
                x=golden_points,
                y=df.loc[golden_points, "close"],
                mode="markers",
                marker=dict(size=9, symbol="triangle-up"),
                name="Golden Cross",
            )
        )
    if death_points:
        fig.add_trace(
            go.Scatter(
                x=death_points,
                y=df.loc[death_points, "close"],
                mode="markers",
                marker=dict(size=9, symbol="triangle-down"),
                name="Death Cross",
            )
        )

    # Pretouch markers (recent window)
    recent = df.tail(60)
    pretouch_dates = recent.index[recent["pretouch"] == True]
    if len(pretouch_dates) > 0:
        fig.add_trace(
            go.Scatter(
                x=pretouch_dates,
                y=recent.loc[pretouch_dates, "close"],
                mode="markers",
                marker=dict(size=8),
                name="Pretouch",
            )
        )

    fig.update_xaxes(rangeslider_visible=False)
    fig.update_layout(
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified",
        xaxis=dict(
            rangeselector=dict(
                buttons=list(
                    [
                        dict(count=1, label="1M", step="month", stepmode="backward"),
                        dict(count=3, label="3M", step="month", stepmode="backward"),
                        dict(count=6, label="6M", step="month", stepmode="backward"),
                        dict(label="YTD", step="year", stepmode="todate"),
                        dict(count=1, label="1Y", step="year", stepmode="backward"),
                        dict(label="All", step="all"),
                    ]
                )
            )
        ),
    )
    return fig

# -----------------------------
# UI
# -----------------------------
st.title("📈 SMA 20/200 Tracker (Real Market Data)")
st.caption(
    "Data: Yahoo Finance via yfinance (prices may be delayed). Supports **stocks, indexes, ETFs, and crypto (e.g., BTC-USD, ETH-USD)**. Not financial advice."
)

with st.sidebar:
    st.header("Settings")

    data_set = st.radio(
        "Preset list",
        ["Stocks & Indexes", "Crypto"],
        help="Choose a starter list; you can still edit tickers below.",
    )

    if data_set == "Stocks & Indexes":
        default_tickers = ["^GSPC", "^DJI", "^IXIC", "SPY", "QQQ"]
        tickers_help = "Examples: SPY, QQQ, AAPL, MSFT, ^GSPC (S&P 500), ^DJI (Dow), ^IXIC (Nasdaq)"
    else:
        default_tickers = ["BTC-USD", "ETH-USD", "SOL-USD", "DOGE-USD"]
        tickers_help = "Crypto format: COIN-USD (e.g., BTC-USD, ETH-USD, SOL-USD)."

    tickers_text = st.text_input(
        "Tickers (comma-separated)",
        value=", ".join(default_tickers),
        help=tickers_help,
    )

    start = st.date_input("Start date", value=(dt.date.today() - dt.timedelta(days=365 * 3)))
    end = st.date_input("End date", value=dt.date.today())

    st.info("Tip: Crypto trades 24/7 (e.g., BTC-USD). Stocks trade in sessions (e.g., ^GSPC).")

    st.markdown("---")
    st.subheader("Pretouch")
    pretouch_pct = st.slider(
        "Distance from SMA (%)",
        min_value=0.1,
        max_value=5.0,
        value=1.0,
        step=0.1,
        help="Alert when price is within this percent of the SMA line.",
    )
    watch_choice = st.multiselect(
        "Watch which SMA lines?",
        ["SMA20", "SMA200"],
        default=["SMA200"],
        help="Choose one or both.",
    )
    watch_20 = "SMA20" in watch_choice
    watch_200 = "SMA200" in watch_choice

    st.markdown("---")
    st.subheader("Download")
    dl_choice = st.selectbox("Download which?", ["Current Ticker (below)", "All Tickers (zip)"])

# Parse tickers safely
raw_tickers = [t.strip() for t in tickers_text.split(",") if t.strip()]

# Collect alerts across all tickers
alerts = []

# Per-ticker panels
for tkr in raw_tickers:
    st.markdown("---")
    st.subheader(f"{tkr}")
    df = load_data(tkr, start, end)
    if df.empty:
        st.warning("No data returned. Check ticker symbol or date range.")
        continue

    # Pretouch math
    df["dist20_pct"] = np.where(
        df["SMA20"].notna(), (df["close"] - df["SMA20"]) / df["SMA20"] * 100, np.nan
    )
    df["dist200_pct"] = np.where(
        df["SMA200"].notna(), (df["close"] - df["SMA200"]) / df["SMA200"] * 100, np.nan
    )
    if watch_20 or watch_200:
        conds = []
        if watch_20:
            conds.append(df["dist20_pct"].abs() <= pretouch_pct)
        if watch_200:
            conds.append(df["dist200_pct"].abs() <= pretouch_pct)
        df["pretouch"] = np.column_stack(conds).any(axis=1)
    else:
        df["pretouch"] = False

    latest = df.dropna(subset=["close"]).iloc[-1]
    price = latest["close"]
    sma20 = latest["SMA20"] if not np.isnan(latest["SMA20"]) else None
    sma200 = latest["SMA200"] if not np.isnan(latest["SMA200"]) else None
    status = (
        "Above 200-day" if (sma200 is not None and price > sma200) else ("Below 200-day" if sma200 is not None else "—")
    )

    cols = st.columns(5)
    cols[0].metric("Last Close", fmt(price))
    cols[1].metric("SMA 20", fmt(sma20) if sma20 else "—")
    cols[2].metric("SMA 200", fmt(sma200) if sma200 else "—")
    cols[3].metric("Position vs 200D", status)

    d20 = df["dist20_pct"].iloc[-1] if not np.isnan(df["dist20_pct"].iloc[-1]) else None
    d200 = df["dist200_pct"].iloc[-1] if not np.isnan(df["dist200_pct"].iloc[-1]) else None
    badge = []
    if watch_20 and d20 is not None:
        badge.append(f"Δ20: {d20:.2f}%")
    if watch_200 and d200 is not None:
        badge.append(f"Δ200: {d200:.2f}%")
    cols[4].metric("Distance (%)", " | ".join(badge) if badge else "—")

    # Chart
    fig = make_chart(
        tkr, df, pretouch_pct=float(pretouch_pct), watch_20=watch_20, watch_200=watch_200
    )
    st.plotly_chart(fig, use_container_width=True)

    # Crossovers table
    sig_df = df[df["signal"].diff().fillna(0) != 0][["close", "SMA20", "SMA200", "signal"]].copy()
    if not sig_df.empty:
        sig_df["event"] = np.where(sig_df["signal"] == 1, "Golden Cross (20 > 200)", "Death Cross (20 < 200)")
        st.markdown("**Recent Crossovers**")
        st.dataframe(
            sig_df[["event", "close", "SMA20", "SMA200"]]
            .tail(10)
            .rename(columns={"close": "Close", "SMA20": "SMA 20", "SMA200": "SMA 200"})
        )
    else:
        st.info("No SMA 20/200 crossovers in the selected period.")

    # Alerts for this ticker (latest row)
    latest_msgs = []
    if watch_20 and pd.notna(latest.get("dist20_pct")) and abs(latest["dist20_pct"]) <= pretouch_pct:
        latest_msgs.append(f"within {pretouch_pct:.2f}% of SMA20 ({latest['dist20_pct']:.2f}%)")
    if watch_200 and pd.notna(latest.get("dist200_pct")) and abs(latest["dist200_pct"]) <= pretouch_pct:
        latest_msgs.append(f"within {pretouch_pct:.2f}% of SMA200 ({latest['dist200_pct']:.2f}%)")
    if latest_msgs:
        alerts.append((tkr, "; ".join(latest_msgs)))

    # CSV download for current ticker
    csv = df.to_csv().encode("utf-8")
    st.download_button(
        label=f"Download {tkr} CSV",
        data=csv,
        file_name=f"{tkr.replace('^','')}_{start}_{end}.csv",
        mime="text/csv",
        key=f"dl_{tkr}",
    )

# Global alerts
if alerts:
    st.success("**Pretouch Alerts** — The following tickers are within the selected threshold:")
    alert_df = pd.DataFrame(alerts, columns=["Ticker", "Condition"])
    st.dataframe(alert_df, use_container_width=True)

st.markdown("---")
with st.expander("Notes & Disclaimers"):
    st.write(
        """
        - **Data source**: Yahoo Finance via `yfinance`. Prices may be delayed and are for informational purposes only.
        - **SMAs**: Computed on adjusted close (auto-adjusted).
        - **No paper trading**: This app does not simulate trades or performance.
        - **Not financial advice**.
        """
    )
