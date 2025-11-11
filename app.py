import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
import time, json, os
import plotly.graph_objects as go
import pandas_datareader.data as pdr
from pycoingecko import CoinGeckoAPI

APP_VERSION = "v1.3.7"

# ============== Page / UX ==============
st.set_page_config(page_title="SMA Pro", layout="wide")
print(f"Launching TraderQ SMA Tracker {APP_VERSION}")

APP_DIR = os.path.dirname(__file__)
SETTINGS_PATH = os.path.join(APP_DIR, "settings.json")

st.title(f"TraderQ SMA 20/200 Tracker - {APP_VERSION}")
st.caption("Data: Stooq & CoinGecko (free tiers), Yahoo fallback for crypto")
st.markdown("---")

# ============== Settings / Defaults ==============
INDEX_TO_ETF = {"^GSPC": "SPY", "^DJI": "DIA", "^IXIC": "QQQ"}
CG_IDS = {"BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana", "DOGE": "dogecoin"}

DEFAULTS = {
    "theme": "Dark",
    "mode": "Stocks / Indexes",
    "tickers_stocks": ["^GSPC", "SPY", "^DJI", "DIA", "^IXIC", "QQQ"],
    "tickers_crypto": ["BTC-USD", "ETH-USD", "SOL-USD", "DOGE-USD"],
    "pretouch_pct": 2.0,
    "autorefresh": {"enabled": False, "seconds": 120},
}

def load_settings():
    path = SETTINGS_PATH
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return {**DEFAULTS, **json.load(f)}
        except Exception:
            return DEFAULTS.copy()
    return DEFAULTS.copy()

def save_settings(s):
    try:
        with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
            json.dump(s, f, indent=2)
    except Exception:
        pass

if "settings" not in st.session_state:
    st.session_state.settings = load_settings()
S = st.session_state.settings

# ============== Helpers ==============
def fmt(value, kind="usd"):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "-"
    if kind == "usd":
        return f"${value:,.2f}"
    elif kind == "pct":
        return f"{value:+.2f}%"
    else:
        return f"{value:,.2f}"

def _compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.rename(columns={"Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"})
    df.index = pd.to_datetime(df.index)
    df["SMA20"]  = df["close"].rolling(20,  min_periods=20).mean()
    df["SMA200"] = df["close"].rolling(200, min_periods=200).mean()
    df["signal"] = np.where(
        (df["SMA20"].notna())&(df["SMA200"].notna())&(df["SMA20"]>df["SMA200"]), 1,
        np.where((df["SMA20"].notna())&(df["SMA200"].notna())&(df["SMA20"]<df["SMA200"]), -1, 0)
    )
    df["xover"] = df["signal"].diff().fillna(0)  # +2 golden, -2 death
    return df

# ============== Data (no paid APIs) ==============
def _load_equity_stooq(ticker: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    t = INDEX_TO_ETF.get(ticker, ticker)
    try:
        df = pdr.DataReader(t, "stooq", start, end).sort_index()
    except Exception:
        df = pd.DataFrame()
    return _compute_indicators(df)

def _load_crypto_coingecko(ticker: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    # Expect BTC-USD, ETH-USD, ...
    if not ticker.upper().endswith("-USD"):
        return pd.DataFrame()

    symbol = ticker.split("-")[0].upper()
    cg_id = CG_IDS.get(symbol)

    total_days = max(1, (end - start).days)
    days = 365 if total_days > 365 else max(1, total_days)

    prices = []
    if cg_id:
        try:
            cg = CoinGeckoAPI()
            data = cg.get_coin_market_chart_by_id(id=cg_id, vs_currency="usd", days=days)
            prices = data.get("prices", []) or []
        except Exception:
            prices = []

    if prices:
        s = pd.Series({pd.to_datetime(p[0], unit="ms").normalize(): p[1] for p in prices})
        s = s.groupby(s.index).last()
        df = s.to_frame("close")
        df["open"] = df["close"]; df["high"] = df["close"]; df["low"] = df["close"]; df["volume"] = 0.0
        df.index = pd.to_datetime(df.index)
        df = df.loc[(df.index.date >= start) & (df.index.date <= end)]
        return _compute_indicators(df)

    # Fallback: Yahoo yfinance (if available)
    try:
        import yfinance as yf
        yf_tkr = f"{symbol}-USD"
        if total_days > 3640:
            ydf = yf.download(yf_tkr, period="10y", interval="1d", progress=False, auto_adjust=False)
        else:
            ydf = yf.download(yf_tkr, start=start, end=end + dt.timedelta(days=1), interval="1d",
                              progress=False, auto_adjust=False)
        if ydf is None or ydf.empty:
            return pd.DataFrame()
        ydf = ydf.rename(columns=str.title).sort_index()
        return _compute_indicators(ydf)
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_data(ticker: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    if ticker.upper().endswith("-USD"):
        return _load_crypto_coingecko(ticker, start, end)
    return _load_equity_stooq(ticker, start, end)

# ============== Analytics ==============
def signal_table(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    sigs = df[df["xover"].isin([2, -2])]
    for idx, row in sigs.iterrows():
        out.append({
            "Date": idx.date(),
            "Type": "Golden Cross" if row["xover"] == 2 else "Death Cross",
            "Close": round(float(row["close"]), 2) if not np.isnan(row["close"]) else None,
            "SMA20": round(float(row["SMA20"]), 2) if not np.isnan(row["SMA20"]) else None,
            "SMA200": round(float(row["SMA200"]), 2) if not np.isnan(row["SMA200"]) else None,
        })
    return pd.DataFrame(out)

def distance_to_sma(df: pd.DataFrame, which="SMA200"):
    if df.empty or which not in df: return np.nan
    c = df["close"].iloc[-1]
    s = df[which].iloc[-1]
    return (c/s - 1)*100 if s else np.nan

def backtest_sma(df: pd.DataFrame, fee_bp: float = 1.0):
    d = df.dropna(subset=["SMA20","SMA200","close"]).copy()
    if d.empty:
        return {"CAGR (Strategy)": np.nan, "CAGR (Buy&Hold)": np.nan, "Max Drawdown": np.nan, "Sharpe": np.nan, "Trades": 0}, pd.DataFrame()
    d["pos"] = (d["SMA20"] > d["SMA200"]).astype(int)
    d["ret"] = d["close"].pct_change().fillna(0)
    d["strat_ret"] = d["pos"].shift(1).fillna(0) * d["ret"]
    trades = (d["pos"].diff().abs() > 0).astype(int)
    d["strat_ret"] -= trades * (fee_bp/10000.0)
    d["eq"] = (1 + d["strat_ret"]).cumprod()
    d["bh"] = (1 + d["ret"]).cumprod()

    n_years = max(1e-9, (d.index[-1]-d.index[0]).days/365.25)
    cagr     = d["eq"].iloc[-1]**(1/n_years) - 1
    cagr_bh  = d["bh"].iloc[-1]**(1/n_years) - 1
    dd       = (d["eq"]/d["eq"].cummax()-1).min()
    vol      = d["strat_ret"].std()*np.sqrt(252)
    sharpe   = (d["strat_ret"].mean()*252) / (vol + 1e-12)

    return {
        "CAGR (Strategy)": cagr, "CAGR (Buy&Hold)": cagr_bh,
        "Max Drawdown": dd, "Sharpe": sharpe, "Trades": int(trades.sum())
    }, d[["eq","bh"]]

def portfolio_equity(data_map: dict):
    dfs = []
    for t, df in data_map.items():
        if df.empty: continue
        s = df["close"].pct_change().rename(t).fillna(0)
        dfs.append(s)
    if not dfs: 
        return pd.DataFrame()
    return pd.concat(dfs, axis=1).fillna(0)

# ============== Charting ==============
def add_verticals(fig, df):
    for idx, row in df[df["xover"].isin([2,-2])].iterrows():
        fig.add_vline(x=idx, line_width=1, line_dash="dash", line_color="rgba(255,255,255,0.3)")

def plot_chart(df: pd.DataFrame, ticker: str, pretouch_pct: float = 0, theme="Dark"):
    fig = go.Figure()
    if pretouch_pct and pretouch_pct > 0:
        upper_band = df["SMA200"] * (1 + pretouch_pct/100)
        lower_band = df["SMA200"] * (1 - pretouch_pct/100)
        fig.add_trace(go.Scatter(x=df.index, y=upper_band, line=dict(color="gray", width=1, dash="dot"), name=f"SMA200 +{pretouch_pct}%"))
        fig.add_trace(go.Scatter(x=df.index, y=lower_band, line=dict(color="gray", width=1, dash="dot"), name=f"SMA200 -{pretouch_pct}%"))

    fig.add_trace(go.Candlestick(x=df.index, open=df["open"], high=df["high"], low=df["low"], close=df["close"], name="Price"))
    fig.add_trace(go.Scatter(x=df.index, y=df["SMA20"],  name="SMA 20",  line=dict(color="orange", width=2)))
    fig.add_trace(go.Scatter(x=df.index, y=df["SMA200"], name="SMA 200", line=dict(color="blue",   width=2)))

    golden = df[df["xover"] == 2]; death  = df[df["xover"] == -2]
    fig.add_trace(go.Scatter(x=golden.index, y=golden["close"], mode="markers+text", name="Golden Cross",
                             text=["G"]*len(golden), textposition="top center",
                             marker=dict(size=16, color="lime", symbol="triangle-up", line=dict(width=1, color="black")),
                             hovertext=["Golden Cross"]*len(golden), hoverinfo="text+y"))
    fig.add_trace(go.Scatter(x=death.index, y=death["close"], mode="markers+text", name="Death Cross",
                             text=["D"]*len(death), textposition="bottom center",
                             marker=dict(size=16, color="red", symbol="triangle-down", line=dict(width=1, color="black")),
                             hovertext=["Death Cross"]*len(death), hoverinfo="text+y"))
    add_verticals(fig, df)

    if theme == "Dark":
        plot_bg="#0e1117"; paper_bg="#0e1117"; text_c="#e8e6e3"
    else:
        plot_bg="#ffffff"; paper_bg="#ffffff"; text_c="#111827"

    fig.update_layout(title=f"{ticker} â€“ SMA 20/200 with Pretouch & Crosses",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                      plot_bgcolor=plot_bg, paper_bgcolor=paper_bg, font=dict(color=text_c),
                      xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True, key=f"{ticker}_chart")

# ============== Theme + Sidebar ==============
theme = st.sidebar.radio("Theme", ["Dark", "Light"], index=(0 if S["theme"]=="Dark" else 1), horizontal=True)
S["theme"] = theme

if theme == "Dark":
    st.markdown("""
    <style>.stMetric{background:#121417;border-radius:12px;padding:8px}.stDataFrame{filter:brightness(0.97)}</style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>.stMetric{background:#f7f7f9;border-radius:12px;padding:8px}</style>
    """, unsafe_allow_html=True)

with st.sidebar.expander("Settings", expanded=False):
    st.write("Defaults persist to settings.json")
    tstocks = st.text_input("Default Stock/Index tickers", ", ".join(S["tickers_stocks"]))
    tcrypto = st.text_input("Default Crypto tickers (-USD)", ", ".join(S["tickers_crypto"]))
    pretouch_default = st.number_input("Default Pretouch %", 0.0, 10.0, value=float(S["pretouch_pct"]), step=0.5)
    if st.button("Save defaults"):
        S["tickers_stocks"] = [x.strip() for x in tstocks.split(",") if x.strip()]
        S["tickers_crypto"] = [x.strip() for x in tcrypto.split(",") if x.strip()]
        S["pretouch_pct"] = float(pretouch_default)
        save_settings(S)
        st.success("Saved.")

with st.sidebar.expander("Live Refresh", expanded=False):
    enabled = st.checkbox("Enable auto-refresh", value=S["autorefresh"]["enabled"])
    seconds = st.number_input("Refresh every (seconds)", min_value=10, max_value=3600, value=int(S["autorefresh"]["seconds"]), step=10)
    S["autorefresh"]["enabled"] = bool(enabled)
    S["autorefresh"]["seconds"] = int(seconds)
    if enabled:
        st.autorefresh(interval=seconds*1000, key="auto_refresh")

# ============== Mode & Tickers (robust toggle) ==============
if "current_mode" not in st.session_state:
    st.session_state.current_mode = S.get("mode", "Stocks / Indexes")

mode = st.sidebar.radio("Market Type", ["Stocks / Indexes", "Crypto (USD Pairs)"],
                        index=(0 if S.get("mode","Stocks / Indexes").startswith("Stocks") else 1),
                        key="mode_radio")
S["mode"] = mode

switched = (mode != st.session_state.current_mode)
default_tickers = S["tickers_stocks"] if mode.startswith("Stocks") else S["tickers_crypto"]
if switched:
    st.session_state.current_mode = mode
    st.session_state["tickers_input"] = ", ".join(default_tickers)

tickers_text = st.sidebar.text_input("Tickers (comma separated)",
                                     value=st.session_state.get("tickers_input", ", ".join(default_tickers)),
                                     key="tickers_input")
tickers = [t.strip() for t in tickers_text.split(",") if t.strip()]

start = st.sidebar.date_input("Start Date", dt.date.today() - dt.timedelta(days=365))
end   = st.sidebar.date_input("End Date",   dt.date.today())
pretouch_pct = st.sidebar.slider("Pretouch Band (%)", 0.0, 10.0, float(S["pretouch_pct"]), 0.5)
show_screener = st.sidebar.checkbox("Show Pretouch Screener", value=True)
portfolio_mode = st.sidebar.checkbox("Portfolio mode (weights)")
weights_df = None
if portfolio_mode:
    base = pd.DataFrame({"Ticker": tickers, "Weight": [round(1.0/len(tickers),3)]*len(tickers) if tickers else []})
    weights_df = st.sidebar.data_editor(base, num_rows="dynamic", use_container_width=True, key="weights_editor")

# ============== Screener (with Trend column) ==============
if show_screener and len(tickers) > 0:
    rows = []
    for t in tickers:
        d = load_data(t, start, end)
        if not d.empty and not np.isnan(d["SMA20"].iloc[-1]) and not np.isnan(d["SMA200"].iloc[-1]):
            trend = "â–²" if d["SMA20"].iloc[-1] > d["SMA200"].iloc[-1] else "â–¼"
        else:
            trend = "â€“"
        rows.append({
            "Ticker": t,
            "Trend": ("Bullish" if (not d.empty and not np.isnan(d["SMA20"].iloc[-1]) and not np.isnan(d["SMA200"].iloc[-1]) and d["SMA20"].iloc[-1] > d["SMA200"].iloc[-1]) else ("Bearish" if (not d.empty and not np.isnan(d["SMA20"].iloc[-1]) and not np.isnan(d["SMA200"].iloc[-1])) else "-")),
            "Dist to SMA200 (%)": round(distance_to_sma(d,"SMA200"), 2),
            "Dist to SMA20 (%)":  round(distance_to_sma(d,"SMA20"),  2)
        })
        if t.upper().endswith("-USD"):
            time.sleep(0.5)
    screener_df = pd.DataFrame(rows).sort_values("Dist to SMA200 (%)", key=lambda s: s.abs())
    st.subheader("Pretouch Screener (closest to SMA200 on top)")

styled = screener_df[["Ticker","Trend","Dist to SMA200 (%)","Dist to SMA20 (%)"]].style.apply(
    lambda s: [
        ("background-color:#12391a;color:#1fd16c;font-weight:600" if v == "Bullish"
         else ("background-color:#3a1919;color:#ff4d4f;font-weight:600" if v == "Bearish" else ""))
        for v in s
    ],
    subset=["Trend"]
)
st.dataframe(styled, use_container_width=True, hide_index=True)

st.download_button(
    "Download Screener CSV",
    screener_df.to_csv(index=False).encode("utf-8"),
    file_name="pretouch_screener.csv",
    mime="text/csv",
    key="screener_csv"
)")
    styled = screener_df[["Ticker","Trend","Dist to SMA200 (%)","Dist to SMA20 (%)"]].style.apply(
    lambda s: ["background-color:#12391a;color:#1fd16c;font-weight:600" if v=="â–²" else ("background-color:#3a1919;color:#ff4d4f;font-weight:600" if v=="â–¼" else "") for v in s],
    subset=["Trend"]
)
st.dataframe(styled, use_container_width=True, hide_index=True)
    st.download_button("Download Screener CSV",
                       screener_df.to_csv(index=False).encode("utf-8"),
                       file_name="pretouch_screener.csv", mime="text/csv", key="screener_csv")

# ============== Portfolio View ==============
def portfolio_equity(data_map: dict):
    dfs = []
    for t, df in data_map.items():
        if df.empty: continue
        s = df["close"].pct_change().rename(t).fillna(0)
        dfs.append(s)
    if not dfs: 
        return pd.DataFrame()
    return pd.concat(dfs, axis=1).fillna(0)

if portfolio_mode and tickers:
    st.markdown("### Portfolio")
    data_map = {t: load_data(t, start, end) for t in tickers}
    R = portfolio_equity(data_map)
    if not R.empty and weights_df is not None and len(weights_df) > 0:
        w = (weights_df.set_index("Ticker")["Weight"]).reindex(R.columns).fillna(0.0)
        if w.abs().sum() == 0:
            st.warning("All weights are zero. Set some weights to view portfolio equity.")
        else:
            w = w / w.abs().sum()
            port_ret = (R * w).sum(axis=1)
            eq = (1 + port_ret).cumprod()
            bh = (1 + R.mean(axis=1)).cumprod()

            if len(eq) > 1:
                years = max(1e-9, (eq.index[-1] - eq.index[0]).days/365.25)
                cagr = eq.iloc[-1]**(1/years) - 1
                dd = (eq/eq.cummax()-1).min()
                vol = port_ret.std()*np.sqrt(252)
                sharpe = (port_ret.mean()*252)/(vol + 1e-12)
            else:
                cagr=dd=vol=sharpe=np.nan

            c1,c2,c3,c4 = st.columns(4)
            c1.metric("CAGR (Portfolio)", fmt(cagr*100,"pct"))
            c2.metric("Max Drawdown", fmt(dd*100,"pct"))
            c3.metric("Sharpe", f"{sharpe:.2f}" if not np.isnan(sharpe) else "-")
            c4.metric("Assets", str((w!=0).sum()))

            pf = go.Figure()
            pf.add_trace(go.Scatter(x=eq.index, y=eq, name="Portfolio Equity"))
            pf.add_trace(go.Scatter(x=bh.index, y=bh, name="Equal-Weight B&H", line=dict(dash="dot")))
            if S["theme"] == "Dark":
                pf.update_layout(plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font=dict(color="#e8e6e3"))
            st.plotly_chart(pf, use_container_width=True, key="portfolio_curve")

# ============== Per-Ticker Tabs ==============
for tkr in tickers:
    st.markdown("---")
    st.subheader(tkr)

    df = load_data(tkr, start, end)
    if df.empty:
        st.warning(f"No data for {tkr}.")
        continue

    price = df["close"].iloc[-1]
    sma20 = df["SMA20"].iloc[-1]
    sma200 = df["SMA200"].iloc[-1]
    pct_diff = ((sma20 - sma200) / sma200 * 100) if (sma200 and not np.isnan(sma200)) else np.nan

    m1, m2, m3 = st.columns(3)
    m1.metric("Last Close", fmt(price, "usd"))
    m2.metric("SMA 20",     fmt(sma20, "usd"))
    m3.metric("20 vs 200",  fmt(pct_diff, "pct"))

    tab_chart, tab_signals, tab_stats = st.tabs(["Chart", "Signals", "Stats"])

    with tab_chart:
        plot_chart(df, tkr, pretouch_pct, theme=S["theme"])
        csv = df.to_csv().encode("utf-8")
        st.download_button("Download CSV", data=csv, file_name=f"{tkr}.csv", mime="text/csv", key=f"{tkr}_csv")

    with tab_signals:
        st.dataframe(signal_table(df), use_container_width=True, hide_index=True)

    with tab_stats:
        metrics, curve = backtest_sma(df, fee_bp=1.0)
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("CAGR (Strat)", fmt(metrics["CAGR (Strategy)"]*100, "pct"))
        c2.metric("CAGR (B&H)",   fmt(metrics["CAGR (Buy&Hold)"]*100, "pct"))
        c3.metric("Max Drawdown", fmt(metrics["Max Drawdown"]*100, "pct"))
        c4.metric("Sharpe",       f'{metrics["Sharpe"]:.2f}' if not np.isnan(metrics["Sharpe"]) else "-")
        c5.metric("Trades",       str(metrics["Trades"]))
        if not curve.empty:
            curve_fig = go.Figure()
            curve_fig.add_trace(go.Scatter(x=curve.index, y=curve["eq"], name="Strategy Equity"))
            curve_fig.add_trace(go.Scatter(x=curve.index, y=curve["bh"], name="Buy & Hold", line=dict(dash="dot")))
            if S["theme"] == "Dark":
                curve_fig.update_layout(plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font=dict(color="#e8e6e3"))
            st.plotly_chart(curve_fig, use_container_width=True, key=f"{tkr}_curve")

st.caption("Educational use only. Not financial advice.")



