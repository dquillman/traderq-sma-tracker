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
    if hasattr(st, 'secrets') and 'firebase' in st.secrets:
        # Running on Streamlit Cloud - use secrets
        try:
            creds_dict = dict(st.secrets['firebase'])

            # Create a temporary file with the credentials
            # tempfile.NamedTemporaryFile with delete=False keeps the file after closing
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
                json.dump(creds_dict, f)
                temp_path = f.name

            return temp_path

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
