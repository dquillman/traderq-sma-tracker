# ui_glow_patch.py - injects styles/glow.css after set_page_config (ASCII only)
from pathlib import Path
import streamlit as st
def apply():
    css_path = Path(__file__).resolve().parent.joinpath("styles", "glow.css")
    if css_path.exists():
        css = css_path.read_text(encoding="ascii", errors="ignore")
        st.markdown("<style>" + css + "</style>", unsafe_allow_html=True)
