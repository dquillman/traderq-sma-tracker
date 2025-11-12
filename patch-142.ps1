# patch-142.ps1 — Fix yfinance MultiIndex columns in _to_ohlc; bump v1.4.2
param(
  [switch]$Run,
  [int]$Port = 8602
)

$ErrorActionPreference = 'Stop'
Set-Location G:\Users\daveq\traderq
$src = "app.py"
if (!(Test-Path $src)) { Write-Error "app.py not found"; exit 1 }

# Backup
$bak = "$src.bak_$(Get-Date -Format yyyyMMdd_HHmmss)"
Copy-Item $src $bak -Force
Write-Host "[OK] Backup -> $bak"

# Load file
$txt = Get-Content $src -Raw -Encoding UTF8

# 1) Bump version
$txt = [regex]::Replace($txt,'APP_VERSION\s*=\s*"(?:[^"]*)"', 'APP_VERSION = "v1.4.2"', 1)

# 2) Replace _to_ohlc with robust MultiIndex-safe version
$find = 'def\s+_to_ohlc\s*\([^\)]*\)\s*:\s*(?:.|\r|\n)*?^\s*def\s+_sma\s*\('
$repl = @"
def _to_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize price DataFrame to columns: ['open','high','low','close','volume'].
    Robust to yfinance MultiIndex columns like ('Open','SPY') or ('SPY','Open').
    """
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["open","high","low","close","volume"])

    # If yfinance returned MultiIndex, flatten to a single level choosing the OHLC token.
    if isinstance(df.columns, pd.MultiIndex):
        flat_cols = []
        for col in df.columns:
            parts = [str(p) for p in col if p is not None]
            tokens = [p.lower() for p in parts]
            field = None
            # prefer explicit OHLCV tokens
            for p in tokens:
                if p in {"open","high","low","close","adj close","adj_close","adjclose","volume"}:
                    field = p
                    break
            if field is None:
                # if no obvious token, fall back to the last piece
                field = tokens[-1] if tokens else "close"
            if field == "adj close":
                field = "adj_close"
            flat_cols.append(field)
        df.columns = flat_cols
    else:
        # Single-level: normalize to str lower
        df.columns = [str(c) for c in df.columns]

    cols_lower = [c.lower() for c in df.columns]
    out = pd.DataFrame(index=df.index)

    def pick(*cands):
        for name in cands:
            name_l = name.lower()
            if name_l in cols_lower:
                return df.iloc[:, cols_lower.index(name_l)]
        return pd.Series(np.nan, index=df.index, dtype="float64")

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
"@

$txt2 = [regex]::Replace($txt, $find, $repl, 1, [System.Text.RegularExpressions.RegexOptions]::Multiline)
if ($txt2 -eq $txt) {
  Write-Host "[i] _to_ohlc() block not found (maybe already patched)."
} else {
  $txt = $txt2
  Write-Host "[OK] Patched _to_ohlc() for MultiIndex handling."
}

# Save file
$txt | Set-Content -Encoding UTF8 $src
Write-Host "[OK] Wrote app.py v1.4.2"

# Optional: run
if ($Run) {
  Get-Process -Name streamlit -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
  Get-Process -Name python    -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
  .\.venv\Scripts\Activate.ps1
  python -m pip install --no-cache-dir -U streamlit==1.39.0 yfinance pycoingecko plotly pandas numpy pandas-datareader
  streamlit cache clear | Out-Null
  python -m streamlit run .\app.py --server.port $Port --server.address localhost
}
