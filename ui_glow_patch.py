# ui_glow_patch.py - injects styles/glow.css after set_page_config
from pathlib import Path
import streamlit as st

def apply():
    try:
        css_path = Path(__file__).resolve().parent.joinpath("styles", "glow.css")
        if css_path.exists():
            # Use utf-8 to correctly handle all characters
            css = css_path.read_text(encoding="utf-8", errors="replace")
            st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    except Exception as e:
        # Fail silently or log to console, but don't crash the app
        print(f"Error applying glow patch: {e}")
