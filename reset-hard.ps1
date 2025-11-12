# reset-hard.ps1 — clean reset to a stable, ASCII-only TraderQ app; yfinance+coingecko; pretouch+trend chips
param(
  [int]$Port = 8620
)

$ErrorActionPreference = 'Stop'
Set-Location G:\Users\daveq\traderq

function OK($m){ Write-Host "[OK] $m" -ForegroundColor Green }
function Info($m){ Write-Host "[i] $m" -ForegroundColor Cyan }
function Die($m){ Write-Error $m; exit 1 }

# Stop any Streamlit
Get-Process -Name streamlit -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue

# Backup old app.py
if (Test-Path .\app.py) {
  $bak = "app.py.bak_{0}" -f (Get-Date -Format yyyyMMdd_HHmmss)
  Copy-Item .\app.py $bak -Force
  Info "Backup -> $bak"
}

# Write clean ASCII config (optional, safe)
$newToml = @"
[theme]
base = "dark"
primaryColor = "#3aa7ff"
backgroundColor = "#0b1020"
secondaryBackgroundColor = "#11162a"
textColor = "#e8ecf3"
"@
$stDir = ".\.streamlit"
if (!(Test-Path $stDir)) { New-Item -ItemType Directory -Path $stDir | Out-Null }
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText((Join-Path $stDir "config.toml"), $newToml, $utf8NoBom)

# Known-good, ASCII-only app.py
$app = @'
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import yfinance as yf
from datetime import datetime, timedelta, timezone

try:
    from pycoingecko import CoinGeckoAPI
except Exception:
    CoinGeckoAPI = None

APP_VERSION = "v1.8.0"

# First Streamlit call must be set_page_config
st.set_page_config(page_title="TraderQ SMA 20/200", layout="wide")

# ---- Helpers ----

def normalize_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with columns Open, High, Low, Close, Volume if possible."""
    if df is None or len(df) == 0:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [" ".join([str(x) for x in col if str(x) != "nan"]).strip() for col in df.columns]
    # Map common names (yfinance usually has Title Case already)
    mapping = {}
    for c in df.columns:
        cl = str(c).lower()
        if cl == "open": mapping["Open"] = c
        elif cl == "high": mapping["High"] = c
        elif cl == "low": mapping["Low"] = c
        elif cl == "close": mapping["Close"] = c
        elif cl == "adj close" and "Close" not in mapping: mapping["Close"] = c
        elif cl == "volume": mapping["Volume"] = c
    out = {}
    for k in ["Open", "High", "Low", "Close", "Volume"]:
        if k in df.columns:
            out[k] = df[k]
        elif k in mapping:
            out[k] = df[mapping[k]]
    if "Close" not in out:
        # fallback: any column with "close" in name
        close_cols = [c for c in df.columns if "close" in str(c).lower()]
        if close_cols:
            out["Close"] = df[close_cols[0]]
    if not out:
        if isinstance(df, pd.Series):
            return pd.DataFrame({"Close": df})
        return pd.DataFrame()
    outdf = pd.DataFrame(out)
    outdf.index = pd.to_datetime(df.index)
    return outdf

@st.cache_data(show_spinner=False)
def load_stock(ticker: str, start: datetime, end: datetime) -> pd.DataFrame:
    try:
        df = yf.download(ticker, start=start, end=end, progress=False, threads=False, auto_adjust=False)
    except Exception as e:
        st.warning(f"yfinance failed for {ticker}: {e}")
        return pd.DataFrame()
    return normalize_ohlc(df)

CRYPTO_MAP = {"BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana"}

@st.cache_data(show_spinner=False)
def load_crypto(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    if CoinGeckoAPI is None:
        st.info("pycoingecko not available; install pycoingecko to enable crypto.")
        return pd.DataFrame()
    # clamp to 365 days for free API
    if (end - start).days > 365:
        start = end - timedelta(days=365)
    cg = CoinGeckoAPI()
    cid = CRYPTO_MAP.get(symbol.upper())
    if not cid:
        st.warning(f"Unknown crypto symbol: {symbol}")
        return pd.DataFrame()
    try:
        from_ts = int(start.replace(tzinfo=timezone.utc).timestamp())
        to_ts = int(end.replace(tzinfo=timezone.utc).timestamp())
        data = cg.get_coin_market_chart_range_by_id(id=cid, vs_currency="usd", from_timestamp=from_ts, to_timestamp=to_ts)
        prices = data.get("prices", [])
        if not prices:
            return pd.DataFrame()
        pts = [(datetime.utcfromtimestamp(p[0]/1000), p[1]) for p in prices]
        df = pd.DataFrame(pts, columns=["Date", "Close"]).set_index("Date")
        return df
    except Exception as e:
        st.warning(f"CoinGecko failed for {symbol}: {e}")
        return pd.DataFrame()

def make_chart(df: pd.DataFrame, ticker: str, show_bands: bool, band_pct: float):
    if df.empty:
        return None
    d = df.copy()
    d["SMA20"] = d["Close"].rolling(20).mean()
    d["SMA200"] = d["Close"].rolling(200).mean()

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=d.index,
        open=d.get("Open"), high=d.get("High"), low=d.get("Low"), close=d["Close"],
        name=ticker, showlegend=False
    ))
    fig.add_trace(go.Scatter(x=d.index, y=d["SMA20"], name="SMA20", line=dict(width=2)))
    fig.add_trace(go.Scatter(x=d.index, y=d["SMA200"], name="SMA200", line=dict(width=2)))

    if show_bands and band_pct and band_pct > 0:
        up = (1 + band_pct/100.0)
        dn = (1 - band_pct/100.0)
        fig.add_trace(go.Scatter(x=d.index, y=d["SMA200"]*up, name=f"Pretouch +{band_pct}%", line=dict(width=1, dash="dot")))
        fig.add_trace(go.Scatter(x=d.index, y=d["SMA200"]*dn, name=f"Pretouch -{band_pct}%", line=dict(width=1, dash="dot")))

    # golden/death cross markers
    cross = np.sign(d["SMA20"] - d["SMA200"]).diff()
    for idx, val in cross.dropna().items():
        if val > 0:
            fig.add_vline(x=idx, line_width=1, line_dash="dash", line_color="green")
        elif val < 0:
            fig.add_vline(x=idx, line_width=1, line_dash="dash", line_color="red")

    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=520, xaxis_title=None)
    return fig

def chip_html(state: str) -> str:
    s = (state or "").lower()
    if "bull" in s or "golden" in s or "up" in s:
        color = "#26d07c"; label = "Bullish"
    elif "bear" in s or "death" in s or "down" in s:
        color = "#ff6b6b"; label = "Bearish"
    else:
        color = "#a6b0c3"; label = "Sideways"
    return f'<span style="display:inline-flex;align-items:center;border:1px solid #334;border-radius:999px;padding:2px 8px;background:rgba(255,255,255,0.05);font-weight:600;"><span style="width:8px;height:8px;border-radius:50%;background:{color};display:inline-block;margin-right:6px;"></span>{label}</span>'

def build_screener(tickers, start, end, mode, band_pct):
    rows = []
    for t in tickers:
        df = load_crypto(t, start, end) if mode == "crypto" else load_stock(t, start, end)
        if df.empty or "Close" not in df:
            rows.append({"Ticker": t, "Last Close": None, "SMA20": None, "SMA200": None, "20v200%": None, "Dist%": None, "Trend": "n/a"})
            continue
        d = df.copy()
        d["SMA20"] = d["Close"].rolling(20).mean()
        d["SMA200"] = d["Close"].rolling(200).mean()
        nz = d.dropna()
        last = nz.iloc[-1] if not nz.empty else d.iloc[-1]
        price = float(last.get("Close", float("nan")))
        s20 = float(last.get("SMA20", float("nan")))
        s200 = float(last.get("SMA200", float("nan")))
        ratio = (s20/s200 - 1.0)*100.0 if (np.isfinite(s20) and np.isfinite(s200) and s200 != 0.0) else None
        dist = (price/s200 - 1.0)*100.0 if (np.isfinite(price) and np.isfinite(s200) and s200 != 0.0) else None
        state = "Bullish" if (np.isfinite(s20) and np.isfinite(s200) and s20 > s200) else ("Bearish" if (np.isfinite(s20) and np.isfinite(s200) and s20 < s200) else "Sideways")
        rows.append({"Ticker": t, "Last Close": price, "SMA20": s20, "SMA200": s200, "20v200%": ratio, "Dist%": dist, "Trend": state})
    df = pd.DataFrame(rows)
    if not df.empty and "Dist%" in df.columns:
        df = df.sort_values(by="Dist%", key=lambda s: s.abs(), ascending=True)
    df["TrendChip"] = [chip_html(x) for x in df.get("Trend", [])]
    return df

# ---- UI ----

with st.sidebar:
    st.write("TraderQ")
    dark_default = True
    dark = st.toggle("Dark mode", value=dark_default)
pio.templates.default = "plotly_dark" if dark else "plotly_white"

st.title("TraderQ SMA 20/200 Tracker")

colA, colB, colC = st.columns([1, 1, 1.4])
with colA:
    mode = st.radio("Mode", ["stocks", "crypto"], horizontal=True, index=0)
with colB:
    band_pct = st.slider("Pretouch distance (%)", 0.0, 10.0, 2.0, 0.5)
with colC:
    default = "^GSPC,SPY,QQQ" if mode == "stocks" else "BTC,ETH,SOL"
    tickers = st.text_input("Tickers (comma separated)", value=default)

end = datetime.utcnow()
start = end - timedelta(days=500)  # stocks can use longer; crypto loader clamps internally

tick_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
if not tick_list:
    st.stop()

# Chart the first symbol
sym = tick_list[0]
df0 = load_crypto(sym, start, end) if mode == "crypto" else load_stock(sym, start, end)
if df0.empty or "Close" not in df0:
    st.warning(f"No data for {sym}.")
else:
    fig = make_chart(df0, sym, True, band_pct)
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True, key=f"chart_{mode}_{sym}")

# Screener
st.subheader("Pretouch Screener (closest to SMA200 on top)")
scr = build_screener(tick_list, start, end, mode, band_pct)

# Format numeric columns for display
def fmt2(x):
    return "" if (x is None or (isinstance(x, float) and not np.isfinite(x))) else f"{x:,.2f}"
def fmtp(x):
    return "" if (x is None or (isinstance(x, float) and not np.isfinite(x))) else f"{x:,.2f}%"

if not scr.empty:
    show = scr[["Ticker", "Last Close", "SMA20", "SMA200", "20v200%", "Dist%", "Trend", "TrendChip"]].copy()
    show["Last Close"] = show["Last Close"].apply(fmt2)
    show["SMA20"] = show["SMA20"].apply(fmt2)
    show["SMA200"] = show["SMA200"].apply(fmt2)
    show["20v200%"] = show["20v200%"].apply(fmtp)
    show["Dist%"] = show["Dist%"].apply(fmtp)
    st.write(show.to_html(index=False, escape=False), unsafe_allow_html=True)
else:
    st.info("No rows to show yet.")

# CSV download (exclude HTML column)
csv = scr.drop(columns=["TrendChip"], errors="ignore").to_csv(index=False).encode("utf-8")
st.download_button("Download Screener CSV", data=csv, file_name="screener.csv", mime="text/csv")

st.caption(f"Version {APP_VERSION}")
'@

[System.IO.File]::WriteAllText(".\app.py", $app, $utf8NoBom)
OK "Wrote clean app.py"

# Venv/deps
if (!(Test-Path .\.venv\Scripts\python.exe)) {
  Info "Creating venv..."
  if (Test-Path "G:\Python311\python.exe") { & G:\Python311\python.exe -m venv .\.venv } else { & python -m venv .\.venv }
}
. .\.venv\Scripts\Activate.ps1
OK "Venv ready: $((.\.venv\Scripts\python.exe -V))"

Info "Installing deps..."
python -m pip install --upgrade pip >$null
python -m pip install --no-cache-dir streamlit==1.39.0 yfinance==0.2.40 pycoingecko plotly pandas numpy >$null
OK "Dependencies installed."

# Free the port then run
try {
  Get-NetTCPConnection -LocalPort $Port -ErrorAction Stop | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }
} catch {}
Info "Launching on http://localhost:$Port ..."
python -m streamlit run .\app.py --server.port $Port --server.address localhost
