#!/usr/bin/env python3
"""
TraderQ Setup Verification Script
Checks that all prerequisites are properly configured
"""

import sys
import os
from pathlib import Path

# Color codes for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def print_success(msg):
    print(f"{GREEN}✅ {msg}{RESET}")

def print_error(msg):
    print(f"{RED}❌ {msg}{RESET}")

def print_warning(msg):
    print(f"{YELLOW}⚠️  {msg}{RESET}")

def print_info(msg):
    print(f"{BLUE}ℹ️  {msg}{RESET}")

def check_python_version():
    """Check Python version is 3.8+"""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print_success(f"Python version: {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print_error(f"Python version {version.major}.{version.minor} is too old. Need 3.8+")
        return False

def check_dependencies():
    """Check if all required packages are installed"""
    required_packages = [
        'streamlit',
        'firebase_admin',
        'google.cloud.firestore',
        'yfinance',
        'pandas',
        'plotly',
        'numpy'
    ]

    all_installed = True
    for package in required_packages:
        try:
            if package == 'google.cloud.firestore':
                __import__('google.cloud.firestore')
            else:
                __import__(package)
            print_success(f"Package installed: {package}")
        except ImportError:
            print_error(f"Package missing: {package}")
            all_installed = False

    return all_installed

def check_firebase_files():
    """Check if Firebase configuration files exist"""
    base_dir = Path(__file__).parent

    checks = {
        'firebase.json': base_dir / 'firebase.json',
        'firestore.rules': base_dir / 'firestore.rules',
        'firestore.indexes.json': base_dir / 'firestore.indexes.json',
        'firebase_auth.py': base_dir / 'firebase_auth.py',
        'firebase_db.py': base_dir / 'firebase_db.py',
        'firebase_config.py': base_dir / 'firebase_config.py',
    }

    all_exist = True
    for name, path in checks.items():
        if path.exists():
            print_success(f"File exists: {name}")
        else:
            print_error(f"File missing: {name}")
            all_exist = False

    return all_exist

def check_service_account_key():
    """Check if service account key exists"""
    base_dir = Path(__file__).parent
    key_path = base_dir / 'serviceAccountKey.json'

    if key_path.exists():
        print_success("Service account key found: serviceAccountKey.json")
        return True
    else:
        print_warning("Service account key NOT found: serviceAccountKey.json")
        print_info("Download from Firebase Console → Project Settings → Service Accounts")
        return False

def check_gitignore():
    """Check if sensitive files are in .gitignore"""
    base_dir = Path(__file__).parent
    gitignore_path = base_dir / '.gitignore'

    if not gitignore_path.exists():
        print_error(".gitignore file missing!")
        return False

    with open(gitignore_path, 'r') as f:
        content = f.read()

    sensitive_files = [
        'serviceAccountKey.json',
        '.env',
        '.streamlit/secrets.toml'
    ]

    all_ignored = True
    for file in sensitive_files:
        if file in content:
            print_success(f"Protected in .gitignore: {file}")
        else:
            print_error(f"NOT in .gitignore: {file}")
            all_ignored = False

    return all_ignored

def check_firebase_cli():
    """Check if Firebase CLI is installed"""
    import subprocess
    try:
        result = subprocess.run(['firebase', '--version'],
                              capture_output=True,
                              text=True,
                              timeout=5)
        if result.returncode == 0:
            version = result.stdout.strip()
            print_success(f"Firebase CLI installed: {version}")
            return True
        else:
            print_warning("Firebase CLI not working properly")
            return False
    except FileNotFoundError:
        print_warning("Firebase CLI not installed")
        print_info("Install: npm install -g firebase-tools")
        return False
    except Exception as e:
        print_warning(f"Could not check Firebase CLI: {e}")
        return False

def check_firebase_initialization():
    """Check if Firebase project is initialized"""
    base_dir = Path(__file__).parent
    firebaserc_path = base_dir / '.firebaserc'

    if firebaserc_path.exists():
        print_success("Firebase project initialized (.firebaserc found)")
        try:
            import json
            with open(firebaserc_path, 'r') as f:
                config = json.load(f)
                project_id = config.get('projects', {}).get('default', 'unknown')
                print_info(f"Project ID: {project_id}")
        except:
            pass
        return True
    else:
        print_warning("Firebase project NOT initialized (.firebaserc missing)")
        print_info("Run: firebase init")
        return False

def check_modules():
    """Check if extracted modules exist"""
    base_dir = Path(__file__).parent
    modules_dir = base_dir / 'modules'

    if not modules_dir.exists():
        print_error("modules/ directory missing!")
        return False

    required_modules = [
        'data_loader.py',
        'indicators.py',
        'charts.py',
        'alerts.py',
        'portfolio.py'
    ]

    all_exist = True
    for module in required_modules:
        module_path = modules_dir / module
        if module_path.exists():
            print_success(f"Module exists: modules/{module}")
        else:
            print_error(f"Module missing: modules/{module}")
            all_exist = False

    return all_exist

def main():
    print("=" * 60)
    print("TraderQ Firebase Setup Verification")
    print("=" * 60)
    print()

    checks = {
        "Python Version": check_python_version(),
        "Required Packages": check_dependencies(),
        "Firebase Config Files": check_firebase_files(),
        "Service Account Key": check_service_account_key(),
        "Git Protection": check_gitignore(),
        "Firebase CLI": check_firebase_cli(),
        "Firebase Initialization": check_firebase_initialization(),
        "Extracted Modules": check_modules()
    }

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = sum(1 for v in checks.values() if v)
    total = len(checks)

    for name, status in checks.items():
        status_str = f"{GREEN}PASS{RESET}" if status else f"{RED}FAIL{RESET}"
        print(f"{name:.<40} {status_str}")

    print()
    print(f"Overall: {passed}/{total} checks passed")

    if passed == total:
        print()
        print_success("All checks passed! You're ready to run the app.")
        print_info("Next step: streamlit run app.py")
        return 0
    elif passed >= total - 2:
        print()
        print_warning("Almost ready! Fix the remaining issues.")
        print_info("Most critical: Service Account Key and Firebase CLI")
        return 1
    else:
        print()
        print_error("Several issues found. Please review and fix.")
        print_info("See FIREBASE_SETUP.md and DEPLOYMENT.md for instructions")
        return 2

if __name__ == "__main__":
    sys.exit(main())
