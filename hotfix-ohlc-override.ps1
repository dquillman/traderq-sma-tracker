# hotfix-ohlc-override.ps1  — Override _to_ohlc to handle yfinance MultiIndex; bump version; relaunch
param(
  [switch]$Run,
  [int]$Port = 8603
)

$ErrorActionPreference = 'Stop'
Set-Location G:\Users\daveq\traderq
$src = "app.py"
if (-not (Test-Path $src)) { Write-Error "app.py not found"; exit 1 }

# 1) Backup
$bak = "$src.bak_$(Get-Date -Format yyyyMMdd_HHmmss)"
Copy-Item $src $bak -Force
Write-Host "[OK] Backup -> $bak"

# 2) Bump version tag to v1.4.3
$text = Get-Content $src -Raw -Encoding UTF8
$text = [regex]::Replace($text, 'APP_VERSION\s*=\s*"(?:[^"]*)"', 'APP_VERSION = "v1.4.3"', 1)

# 3) Append a replacement _to_ohlc() definition at the end of app.py.
#    In Python, later defs override earlier ones. load_stock refers to _to_ohlc by name at call-time,
#    so it will use this safe version.
$append = @'
# ===== HOTFIX (_to_ohlc override) v1.4.3 =====
def _to_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize to columns: ['open','high','low','close','volume'].
    Robust to yfinance MultiIndex like ('Open','SPY') or ('SPY','Open').
    """
    import numpy as _np
    import pandas as _pd

    if df is None or len(df) == 0:
        return _pd.DataFrame(columns=["open","high","low","close","volume"])

    # Flatten MultiIndex columns by choosing the OHLC token if present.
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
# ===== END HOTFIX =====
'@

$text = $text + "`r`n" + $append
Set-Content $src $text -Encoding UTF8
Write-Host "[OK] Appended safe _to_ohlc override and bumped to v1.4.3"

# 4) Optional: relaunch clean
if ($Run) {
  Get-Process -Name streamlit -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
  Get-Process -Name python    -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
  .\.venv\Scripts\Activate.ps1
  python -m pip install --no-cache-dir -U streamlit==1.39.0 yfinance pycoingecko plotly pandas numpy pandas-datareader
  streamlit cache clear | Out-Null
  python -m streamlit run .\app.py --server.port $Port --server.address localhost
}
