# pro-ui-pack.ps1 — TraderQ "High Tech" UI pack (theme + CSS + header), bump v1.7.0 and restart
param(
  [switch]$Run = $true,
  [int]$Port = 8617
)

$ErrorActionPreference = 'Stop'
Set-Location G:\Users\daveq\traderq

# --- Ensure Streamlit theme file ---
$stDir = ".\.streamlit"
if (!(Test-Path $stDir)) { New-Item -ItemType Directory -Path $stDir | Out-Null }
$config = @"
[theme]
base="dark"
primaryColor="#6ae3ff"
backgroundColor="#0b1020"
secondaryBackgroundColor="#11162a"
textColor="#e8ecf3"
"@
$configPath = Join-Path $stDir "config.toml"
$config | Set-Content -Encoding UTF8 $configPath

# --- Patch app.py: bump version, inject CSS block + header bar once ---
$src = "app.py"
if (!(Test-Path $src)) { throw "app.py not found" }
$txt = Get-Content $src -Raw -Encoding UTF8

# 1) bump version
$txt = [regex]::Replace($txt, 'APP_VERSION\s*=\s*"(?:[^"]*)"', 'APP_VERSION = "v1.7.0"', 1)

# 2) guarantee single set_page_config at top
$txt = [regex]::Replace($txt, '^\s*st\.set_page_config\([^\)]*\)\s*$', '', 'Multiline')
$needle = "import streamlit as st"
$pos = $txt.IndexOf($needle)
if ($pos -lt 0) { $txt = "$needle`r`n$txt"; $pos = 0 }
$lineEnd = $txt.IndexOf("`n", $pos); if ($lineEnd -lt 0){ $lineEnd = $pos + $needle.Length }
$cfg = 'st.set_page_config(page_title="TraderQ SMA 20/200", page_icon="📈", layout="wide")'
if ($txt -notmatch [regex]::Escape($cfg)) {
  $head = $txt.Substring(0, $lineEnd+1); $tail = $txt.Substring($lineEnd+1).TrimStart("`r","`n")
  $txt = $head + $cfg + "`r`n" + $tail
}

# 3) inject PRO CSS + header (only once)
if ($txt -notmatch "PRO_CSS_v170_START") {
  $ui = @"
# === PRO_CSS_v170_START ===
import plotly.io as pio

# Dark/Light toggle in sidebar
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True
with st.sidebar:
    st.markdown("### Appearance")
    st.session_state.dark_mode = st.toggle("Dark mode", value=st.session_state.dark_mode)

# Plotly theme
pio.templates.default = "plotly_dark" if st.session_state.dark_mode else "plotly_white"

# Neon / glass CSS
st.markdown(
    f'''
    <style>
    :root {{
      --bg: {'#0b1020' if st.session_state.dark_mode else '#ffffff'};
      --card: {'#11162a' if st.session_state.dark_mode else '#ffffff'};
      --text: {'#e8ecf3' if st.session_state.dark_mode else '#111827'};
      --muted: {'#a6b0c3' if st.session_state.dark_mode else '#6b7280'};
      --accent: #6ae3ff;
      --good: #26d07c;
      --bad: #ff6b6b;
      --border: {'#20263d' if st.session_state.dark_mode else '#e5e7eb'};
      --glow: 0 0 24px rgba(106,227,255,.35), 0 0 48px rgba(106,227,255,.15);
    }}
    .stApp {{ background: var(--bg); color: var(--text); font-family: Inter, Segoe UI, system-ui, -apple-system, Roboto, Arial, sans-serif; }}

    /* Top header bar */
    .traderq-nav {{
      position: sticky; top: 0; z-index: 999;
      display:flex; align-items:center; justify-content:space-between; gap:16px;
      padding: 14px 18px; margin: -1rem -1rem 1rem;
      background: linear-gradient(180deg, rgba(20,28,50,.95), rgba(20,28,50,.75));
      border-bottom: 1px solid var(--border);
      box-shadow: var(--glow);
      backdrop-filter: blur(8px);
    }}
    .traderq-brand {{ font-weight:800; letter-spacing:.02em; font-size: 1.1rem; }}
    .traderq-pill {{
      border:1px solid var(--border); border-radius:999px; padding:6px 10px;
      background: rgba(255,255,255,.05); font-weight:600;
    }}
    /* Cards */
    .pro-card {{
      border: 1px solid var(--border);
      background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.01));
      border-radius: 14px; padding: 14px 16px; box-shadow: var(--glow);
    }}
    .chip {{ display:inline-flex; align-items:center; gap:8px; padding:4px 10px; border-radius:999px; border:1px solid var(--border); background:rgba(255,255,255,0.05); font-weight:600; }}
    .chip .dot {{ width:8px; height:8px; border-radius:50%; display:inline-block; }}
    .chip.bull .dot {{ background: var(--good); }}
    .chip.bear .dot {{ background: var(--bad); }}
    .chip.flat .dot {{ background: var(--muted); }}

    /* tighten layout */
    section.main > div {{ padding-top: 0.5rem; }}
    </style>
    ''',
    unsafe_allow_html=True
)

# Header bar (shows version)
st.markdown(
    f'''
    <div class="traderq-nav">
      <div class="traderq-brand">⚡ TraderQ — SMA 20/200</div>
      <div class="traderq-pill">Version: <b>{APP_VERSION}</b></div>
    </div>
    ''',
    unsafe_allow_html=True
)

def render_trend_chip(state: str) -> str:
    s = (state or "").lower()
    if any(x in s for x in ["golden","bull","up"]):
        klass, label = "bull", "Bullish"
    elif any(x in s for x in ["death","bear","down"]):
        klass, label = "bear", "Bearish"
    else:
        klass, label = "flat", "Sideways"
    return f'<span class="chip {klass}"><span class="dot"></span>{label}</span>'
# === PRO_CSS_v170_END ===
"@
  # Insert right after set_page_config
  $insertPos = $txt.IndexOf($cfg)
  $lnEnd = $txt.IndexOf("`n", $insertPos); if ($lnEnd -lt 0){ $lnEnd = $insertPos + $cfg.Length }
  $txt = $txt.Insert($lnEnd+1, "`r`n$ui`r`n")
}

# 4) Save
$txt | Set-Content -Encoding UTF8 $src
Write-Host "[OK] UI pack applied. Version v1.7.0. Theme + CSS + header injected."

# 5) Restart
if ($Run) {
  Get-Process -Name streamlit -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
  .\.venv\Scripts\Activate.ps1 | Out-Null
  python -m pip install --no-cache-dir -U plotly >$null
  python -m streamlit run .\app.py --server.port $Port --server.address localhost
}
