"""
Minimal test script to check if Firebase can initialize
This will help us see what's actually failing
"""
import sys
import traceback

print("=" * 70)
print("STARTING STARTUP TEST")
print("=" * 70)

try:
    print("Step 1: Importing streamlit...")
    import streamlit as st
    print("✓ Streamlit imported")
    
    print("Step 2: Checking st.secrets...")
    try:
        has_secrets = hasattr(st, 'secrets') and st.secrets is not None
        print(f"✓ st.secrets exists: {has_secrets}")
        
        if has_secrets:
            has_firebase = 'firebase' in st.secrets
            print(f"✓ firebase in secrets: {has_firebase}")
            
            if has_firebase:
                firebase_keys = list(st.secrets['firebase'].keys())
                print(f"✓ Firebase keys found: {firebase_keys}")
    except Exception as e:
        print(f"✗ Error checking secrets: {e}")
        traceback.print_exc()
    
    print("Step 3: Importing firebase_config...")
    from firebase_config import get_firebase_credentials
    print("✓ firebase_config imported")
    
    print("Step 4: Getting Firebase credentials...")
    creds_path = get_firebase_credentials()
    print(f"✓ Credentials path: {creds_path}")
    
    print("Step 5: Importing firebase_admin...")
    import firebase_admin
    from firebase_admin import credentials as firebase_creds
    print("✓ firebase_admin imported")
    
    print("Step 6: Initializing Firebase Admin SDK...")
    if not firebase_admin._apps:
        cred = firebase_creds.Certificate(creds_path)
        firebase_admin.initialize_app(cred)
        print("✓ Firebase Admin SDK initialized")
    else:
        print("✓ Firebase Admin SDK already initialized")
    
    print("=" * 70)
    print("SUCCESS: All startup steps completed!")
    print("=" * 70)
    
except Exception as e:
    print("=" * 70)
    print("ERROR OCCURRED:")
    print("=" * 70)
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {str(e)}")
    print()
    print("Full traceback:")
    traceback.print_exc()
    print("=" * 70)
    sys.exit(1)

