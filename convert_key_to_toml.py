#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert Firebase serviceAccountKey.json to Streamlit secrets TOML format
Run this script to generate the TOML configuration for Streamlit Cloud
"""

import json
import sys
from pathlib import Path

# Fix Windows console encoding for emojis
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


def convert_to_toml(json_path: str = "serviceAccountKey.json"):
    """Convert JSON service account key to TOML format"""

    # Check if file exists
    if not Path(json_path).exists():
        print(f"❌ Error: {json_path} not found!")
        print(f"Please download your service account key from Firebase Console")
        print(f"and save it as '{json_path}' in this directory.")
        return

    # Load JSON file
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            creds = json.load(f)
    except Exception as e:
        print(f"❌ Error reading {json_path}: {e}")
        return

    # Generate TOML format
    toml_output = f"""# Streamlit Secrets Configuration
# Copy this entire content to Streamlit Cloud → App Settings → Secrets

[firebase]
type = "{creds.get('type', 'service_account')}"
project_id = "{creds.get('project_id', '')}"
private_key_id = "{creds.get('private_key_id', '')}"
private_key = "{creds.get('private_key', '').replace(chr(10), '\\n')}"
client_email = "{creds.get('client_email', '')}"
client_id = "{creds.get('client_id', '')}"
auth_uri = "{creds.get('auth_uri', 'https://accounts.google.com/o/oauth2/auth')}"
token_uri = "{creds.get('token_uri', 'https://oauth2.googleapis.com/token')}"
auth_provider_x509_cert_url = "{creds.get('auth_provider_x509_cert_url', 'https://www.googleapis.com/oauth2/v1/certs')}"
client_x509_cert_url = "{creds.get('client_x509_cert_url', '')}"
"""

    # Save to file
    output_file = ".streamlit_secrets_toml.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(toml_output)

    print("✅ Conversion successful!")
    print()
    print(f"📝 TOML configuration saved to: {output_file}")
    print()
    print("=" * 70)
    print("NEXT STEPS:")
    print("=" * 70)
    print()
    print("1. Open the file:", output_file)
    print("2. Copy ALL the contents")
    print("3. Go to Streamlit Cloud → Your App → Settings → Secrets")
    print("4. Paste the contents into the secrets text box")
    print("5. Click 'Save'")
    print()
    print("=" * 70)
    print()
    print("Preview:")
    print("=" * 70)
    print(toml_output[:500])
    if len(toml_output) > 500:
        print("...")
        print(f"(Full content in {output_file})")
    print("=" * 70)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        convert_to_toml(sys.argv[1])
    else:
        convert_to_toml()
