"""
Minimal test version of app.py to diagnose startup issues
This will help us see if the problem is with imports or Firebase
"""
import sys
sys.stderr.write("=" * 70 + "\n")
sys.stderr.write("MINIMAL APP STARTING\n")
sys.stderr.write("=" * 70 + "\n")
sys.stderr.flush()

try:
    sys.stderr.write("Step 1: Importing streamlit...\n")
    sys.stderr.flush()
    import streamlit as st
    sys.stderr.write("✓ Streamlit imported\n")
    sys.stderr.flush()
    
    sys.stderr.write("Step 2: Setting page config...\n")
    sys.stderr.flush()
    st.set_page_config(
        page_title="TraderQ Test",
        page_icon="📈",
        layout="wide"
    )
    sys.stderr.write("✓ Page config set\n")
    sys.stderr.flush()
    
    sys.stderr.write("Step 3: Displaying test message...\n")
    sys.stderr.flush()
    st.title("✅ App Started Successfully!")
    st.info("If you see this, the basic app structure works.")
    
    sys.stderr.write("=" * 70 + "\n")
    sys.stderr.write("MINIMAL APP SUCCESS\n")
    sys.stderr.write("=" * 70 + "\n")
    sys.stderr.flush()
    
except Exception as e:
    sys.stderr.write("=" * 70 + "\n")
    sys.stderr.write(f"ERROR: {type(e).__name__}: {str(e)}\n")
    sys.stderr.write("=" * 70 + "\n")
    import traceback
    traceback.print_exc(file=sys.stderr)
    sys.stderr.flush()
    
    st.error(f"Error: {e}")
    import traceback
    st.code(traceback.format_exc())

