#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verify that TraderQ is ready for deployment
Checks for required files, dependencies, and configuration
"""

import os
import sys
from pathlib import Path

# Fix Windows console encoding for emojis
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

def check_file_exists(filepath, description):
    """Check if a file exists"""
    exists = Path(filepath).exists()
    status = "✅" if exists else "❌"
    print(f"{status} {description}: {filepath}")
    return exists

def check_import(module_name, description):
    """Check if a Python module can be imported"""
    try:
        __import__(module_name)
        print(f"✅ {description}: {module_name}")
        return True
    except ImportError:
        print(f"❌ {description}: {module_name} (not installed)")
        return False

def main():
    print("=" * 70)
    print("TraderQ Deployment Verification")
    print("=" * 70)
    print()
    
    all_checks_passed = True
    
    # Check required files
    print("📁 Checking Required Files...")
    print("-" * 70)
    
    required_files = [
        ("app.py", "Main application file"),
        ("firebase_auth.py", "Firebase authentication module"),
        ("firebase_config.py", "Firebase configuration module"),
        ("firebase_db.py", "Firebase database module"),
        ("requirements.txt", "Python dependencies"),
        ("Procfile", "Deployment configuration"),
        ("firestore.rules", "Firestore security rules"),
        (".gitignore", "Git ignore file"),
    ]
    
    for filepath, description in required_files:
        if not check_file_exists(filepath, description):
            all_checks_passed = False
    
    print()
    
    # Check critical Python modules
    print("🐍 Checking Python Dependencies...")
    print("-" * 70)
    
    critical_modules = [
        ("streamlit", "Streamlit framework"),
        ("firebase_admin", "Firebase Admin SDK"),
        ("pandas", "Data processing"),
        ("numpy", "Numerical computing"),
        ("plotly", "Charting library"),
    ]
    
    for module, description in critical_modules:
        if not check_import(module, description):
            all_checks_passed = False
    
    print()
    
    # Check Firebase configuration
    print("🔥 Checking Firebase Configuration...")
    print("-" * 70)
    
    # Check if serviceAccountKey.json exists (for local development)
    has_local_key = check_file_exists("serviceAccountKey.json", "Firebase service account key (local)")
    
    if not has_local_key:
        print("⚠️  serviceAccountKey.json not found (OK for cloud deployment)")
        print("   For local development, download from Firebase Console")
        print("   For cloud deployment, use Streamlit secrets instead")
    
    # Check firebase_config.py handles both local and cloud
    try:
        with open("firebase_config.py", "r") as f:
            content = f.read()
            if "st.secrets" in content and "serviceAccountKey.json" in content:
                print("✅ firebase_config.py handles both local and cloud environments")
            else:
                print("⚠️  firebase_config.py may not handle cloud environment")
                all_checks_passed = False
    except Exception as e:
        print(f"❌ Could not verify firebase_config.py: {e}")
        all_checks_passed = False
    
    print()
    
    # Check .gitignore
    print("🔒 Checking Security...")
    print("-" * 70)
    
    sensitive_files = [
        "serviceAccountKey.json",
        ".env",
        ".streamlit/secrets.toml",
    ]
    
    try:
        with open(".gitignore", "r") as f:
            gitignore_content = f.read()
        
        for sensitive_file in sensitive_files:
            if sensitive_file in gitignore_content:
                print(f"✅ {sensitive_file} is in .gitignore")
            else:
                print(f"⚠️  {sensitive_file} should be in .gitignore")
    except Exception as e:
        print(f"⚠️  Could not verify .gitignore: {e}")
    
    print()
    
    # Check requirements.txt
    print("📦 Checking Requirements...")
    print("-" * 70)
    
    required_packages = [
        "streamlit",
        "firebase-admin",
        "pandas",
        "numpy",
        "plotly",
    ]
    
    try:
        with open("requirements.txt", "r") as f:
            requirements_content = f.read().lower()
        
        for package in required_packages:
            if package.lower() in requirements_content:
                print(f"✅ {package} in requirements.txt")
            else:
                print(f"⚠️  {package} not found in requirements.txt")
    except Exception as e:
        print(f"❌ Could not verify requirements.txt: {e}")
        all_checks_passed = False
    
    print()
    
    # Final summary
    print("=" * 70)
    if all_checks_passed:
        print("✅ All critical checks passed!")
        print()
        print("Your app is ready for deployment!")
        print()
        print("Next steps:")
        print("1. Run: python convert_key_to_toml.py (to generate Streamlit secrets)")
        print("2. Push code to GitHub")
        print("3. Deploy to Streamlit Cloud")
        print("4. See QUICK_DEPLOY.md for detailed instructions")
    else:
        print("⚠️  Some checks failed. Please review the issues above.")
        print()
        print("Common fixes:")
        print("- Install missing packages: pip install -r requirements.txt")
        print("- Download serviceAccountKey.json from Firebase Console (for local dev)")
        print("- Review error messages above")
    print("=" * 70)
    
    return 0 if all_checks_passed else 1

if __name__ == "__main__":
    sys.exit(main())

