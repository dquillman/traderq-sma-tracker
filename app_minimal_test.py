"""
Minimal test app to debug startup issues
This will help us see where the crash is happening
"""
import sys

print("STEP 1: Starting minimal app", file=sys.stderr)
sys.stderr.flush()

try:
    print("STEP 2: Importing streamlit", file=sys.stderr)
    sys.stderr.flush()
    import streamlit as st
    print("STEP 3: Streamlit imported", file=sys.stderr)
    sys.stderr.flush()
    
    print("STEP 4: Setting page config", file=sys.stderr)
    sys.stderr.flush()
    st.set_page_config(page_title="Test", layout="wide")
    print("STEP 5: Page config set", file=sys.stderr)
    sys.stderr.flush()
    
    print("STEP 6: Writing to page", file=sys.stderr)
    sys.stderr.flush()
    st.write("✅ App started successfully!")
    print("STEP 7: Done", file=sys.stderr)
    sys.stderr.flush()
    
except Exception as e:
    error_type = type(e).__name__
    error_msg = str(e)
    print(f"ERROR: {error_type}: {error_msg}", file=sys.stderr)
    import traceback
    traceback.print_exc(file=sys.stderr)
    sys.stderr.flush()
    
    if 'st' in locals():
        st.error(f"Error: {error_type}: {error_msg}")
        import traceback
        with st.expander("Error Details"):
            st.code(traceback.format_exc())

