# ui-upgrade-160.ps1 — High-tech UI polish for TraderQ; bump v1.6.0
param(
  [switch]$Run,
  [int]$Port = 8616
)

$ErrorActionPreference = 'Stop'
Set-Location G:\Users\daveq\traderq
$src = "app.py"
if (-not (Test-Path $src)) { Write-Error "app.py not found"; exit 1 }

# 0) Backup
$bak = "$src.bak_$(Get-Date -Format yyyyMMdd_HHmmss)"
Copy-Item $src $bak -Force
Write-Host "[OK] Backup -> $bak"

# 1) Load file
$txt = Get-Content $src -Raw -Encoding UTF8

# 2) Bump version
$txt = [regex]::Replace($txt, 'APP_VERSION\s*=\s*"(?:[^"]*)"', 'APP_VERSION = "v1.6.0"', 1)

# 3) Ensure page config near the top
if ($txt -notmatch 'st\.set_page_config') {
  $cfg = @"
# --- Page config (wide layout, clean icon) ---
st.set_page_config(page_title="TraderQ SMA 20/200", page_icon="📈", layout="wide")
"@
  # Insert after the first "import streamlit as st"
  $idx = $txt.IndexOf("import streamlit as st")
  if ($idx -ge 0) {
    $lineEnd = $txt.IndexOf("`n", $idx)
    if ($lineEnd -lt 0) { $lineEnd = $idx + 25 }
    $txt = $txt.Insert($lineEnd+1, $cfg + "`n")
  } else {
    $txt = $cfg + "`n" + $txt
  }
}

# 4) Add modern UI block once (after imports). Also replace experimental query params.
$uiBlock = @"
# === UI PRO BLOCK v1.6.0 START ===
import plotly.io as pio

# Replace deprecated experimental query params with new API
try:
    _qp = st.query_params
except Exception:
    # Fallback to dict-like if older Streamlit (shouldn't happen on 1.39)
    _qp = {}

def _get_qp_bool(name: str, default: bool = False) -> bool:
    v = st.query_params.get(name)
    if v is None: 
        return default
    if isinstance(v, (list, tuple)) and v:
        v = v[0]
    return str(v).lower() in {"1","true","yes","on"}

def _set_qp(**kwargs):
    # Persist selections to URL
    for k,v in kwargs.items():
        st.query_params[k] = str(v)

# --- Theme toggle (persists in URL) ---
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = _get_qp_bool("dark", False)

with st.sidebar:
    st.markdown("### Appearance")
    dark = st.toggle("Dark mode", value=st.session_state.dark_mode, key="dark_mode")
    _set_qp(dark=dark)

# Set default plotly template globally (propagates to all figures)
pio.templates.default = "plotly_dark" if st.session_state.dark_mode else "plotly_white"

# --- Global CSS for a high-tech, professional look ---
st.markdown(
    '''
    <style>
    :root {
      --bg: #0b1020;
      --card: #11162a;
      --text: #e8ecf3;
      --muted: #a6b0c3;
      --accent: #6ae3ff;
      --accent2: #7af59a;
      --warn: #ffb86b;
      --bad: #ff6b6b;
      --good: #26d07c;
      --border: #20263d;
    }
    .stApp {
      font-family: "Inter", "Segoe UI", system-ui, -apple-system, Roboto, "Helvetica Neue", Arial, sans-serif;
      letter-spacing: .01em;
    }
    /* Auto light/dark */
    body {
      background: %(bg)s;
      color: %(text)s;
    }
    /* Cards (metrics, containers) */
    .metric-card, .stContainer {
      border: 1px solid %(border)s !important;
      background: %(card)s !important;
      border-radius: 14px !important;
      padding: 12px 14px !important;
      box-shadow: 0 0 0 1px rgba(255,255,255,0.02), 0 8px 24px rgba(0,0,0,0.35);
    }
    /* Headers */
    h1,h2,h3 { letter-spacing: .02em; }
    h1 { font-weight: 700; }
    /* Buttons */
    .stButton>button {
      border-radius: 10px; border: 1px solid %(border)s;
      background: linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02));
      color: %(text)s;
    }
    .stButton>button:hover { border-color: %(accent)s; box-shadow: 0 0 0 1px %(accent)s inset; }
    /* Chips */
    .chip {
      display:inline-flex; align-items:center; gap:8px; padding:4px 10px; border-radius:999px;
      border:1px solid %(border)s; background:rgba(255,255,255,0.04); font-weight:600;
    }
    .chip .dot { width:8px; height:8px; border-radius:50%%; display:inline-block; }
    .chip.bull .dot { background:%(good)s; }
    .chip.bear .dot { background:%(bad)s; }
    .chip.flat .dot { background:%(muted)s; }
    /* Dataframe tweaks */
    .stDataFrame { border-radius: 12px; overflow:hidden; }
    .stDataFrame [data-testid="stTable"] { background: transparent; }
    </style>
    ''' % ({
        "bg": "#0b1020" if st.session_state.dark_mode else "white",
        "card": "#11162a" if st.session_state.dark_mode else "white",
        "text": "#e8ecf3" if st.session_state.dark_mode else "#111827",
        "muted": "#a6b0c3" if st.session_state.dark_mode else "#6b7280",
        "accent": "#6ae3ff",
        "accent2": "#7af59a",
        "warn": "#ffb86b",
        "bad": "#ff6b6b",
        "good": "#26d07c",
        "border": "#20263d" if st.session_state.dark_mode else "#e5e7eb",
    }),
    unsafe_allow_html=True,
)

def chip_trend(state: str) -> str:
    state = (state or "").lower()
    if "bull" in state or "up" in state or "golden" in state:
        klass = "bull"; label = "Bullish"
    elif "bear" in state or "down" in state or "death" in state:
        klass = "bear"; label = "Bearish"
    else:
        klass = "flat"; label = "Sideways"
    return f'<span class="chip {klass}"><span class="dot"></span>{label}</span>'

# === UI PRO BLOCK v1.6.0 END ===
"@

# Insert only once
if ($txt -notmatch 'UI PRO BLOCK v1\.6\.0 START') {
  # Place after first import section
  $pos = $txt.IndexOf("import streamlit as st")
  if ($pos -lt 0) { $pos = 0 }
  $insertAt = $txt.IndexOf("`n", $pos)
  if ($insertAt -lt 0) { $insertAt = $pos }
  $txt = $txt.Insert($insertAt+1, "`n" + $uiBlock + "`n")
}

# 5) Replace deprecated experimental query param calls (safely noop if none)
$txt = $txt -replace 'st\.experimental_get_query_params', 'st.query_params'
$txt = $txt -replace 'st\.experimental_set_query_params', 'st.query_params'

# 6) Save updated app.py
$txt | Set-Content -Encoding UTF8 $src
Write-Host "[OK] Wrote app.py v1.6.0 (UI upgraded)."

# 7) Optional run
if ($Run) {
  Get-Process -Name streamlit -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
  .\.venv\Scripts\Activate.ps1 | Out-Null
  python -m pip install --no-cache-dir -U plotly >$null
  python -m streamlit run .\app.py --server.port $Port --server.address localhost
}
