"""
Firebase Configuration Module
Handles Firebase credentials for both local development and Streamlit Cloud deployment
"""

import streamlit as st
import os
import json
import tempfile
from typing import Optional


def get_firebase_credentials() -> str:
    """
    Get Firebase credentials from Streamlit secrets or local file

    This function automatically detects the environment:
    - On Streamlit Cloud: Reads from st.secrets and creates a temporary file
    - On local development: Uses serviceAccountKey.json file

    Returns:
        str: Path to Firebase credentials file

    Raises:
        FileNotFoundError: If credentials are not found in either location
    """
    # Check if running on Streamlit Cloud (secrets are available)
    if hasattr(st, 'secrets') and st.secrets and 'firebase' in st.secrets:
        # Running on Streamlit Cloud - use secrets
        try:
            firebase_secrets = st.secrets['firebase']
            creds_dict = dict(firebase_secrets)
            
            # Validate required fields
            required_fields = ['type', 'project_id', 'private_key', 'client_email']
            missing_fields = [f for f in required_fields if f not in creds_dict or not creds_dict[f]]
            if missing_fields:
                raise ValueError(f"Missing required Firebase credential fields: {', '.join(missing_fields)}")

            # Create a temporary file with the credentials
            # tempfile.NamedTemporaryFile with delete=False keeps the file after closing
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json', encoding='utf-8') as f:
                json.dump(creds_dict, f, ensure_ascii=False, indent=2)
                temp_path = f.name

            return temp_path

        except KeyError as e:
            raise FileNotFoundError(
                f"Firebase secrets not found in Streamlit secrets. Missing key: {e}\n"
                "Please configure Firebase secrets at: "
                "https://share.streamlit.io → Your App → Settings → Secrets\n"
                "Expected format:\n"
                "[firebase]\n"
                "type = \"service_account\"\n"
                "project_id = \"your-project-id\"\n"
                "..."
            )
        except Exception as e:
            raise FileNotFoundError(
                f"Failed to read Firebase credentials from Streamlit secrets: {e}\n"
                "Please check your secrets configuration at: "
                "https://share.streamlit.io → Your App → Settings → Secrets"
            )
    else:
        # Running locally - use serviceAccountKey.json file
        service_account_path = os.getenv(
            'GOOGLE_APPLICATION_CREDENTIALS',
            './serviceAccountKey.json'
        )

        if not os.path.exists(service_account_path):
            raise FileNotFoundError(
                f"Service account key not found at {service_account_path}.\n"
                "Please follow FIREBASE_SETUP.md to download your service account key.\n"
                "Alternatively, set the GOOGLE_APPLICATION_CREDENTIALS environment variable "
                "to point to your service account key file."
            )

        return service_account_path


def get_project_id() -> Optional[str]:
    """
    Get Firebase project ID from credentials

    Returns:
        str: Firebase project ID, or None if not found
    """
    try:
        creds_path = get_firebase_credentials()
        with open(creds_path, 'r') as f:
            creds = json.load(f)
            return creds.get('project_id')
    except Exception:
        return None
