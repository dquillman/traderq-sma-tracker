"""
Firebase Authentication module for TraderQ
Handles user authentication, session management, and user creation
"""

# Lazy import streamlit to avoid import-time issues
import firebase_admin
from firebase_admin import auth, credentials, firestore
from datetime import datetime, timedelta
import os
from typing import Optional, Dict

def _get_streamlit():
    """Lazy import streamlit to avoid import-time issues"""
    import streamlit as st
    return st


class FirebaseAuth:
    """
    Firebase Authentication wrapper for Streamlit applications
    Manages user authentication state using Streamlit's session state
    """

    def __init__(self):
        """Initialize Firebase Admin SDK if not already initialized"""
        if not firebase_admin._apps:
            try:
                # Import here to avoid circular imports
                from firebase_config import get_firebase_credentials

                # Get credentials path (works for both local and Streamlit Cloud)
                # Add timeout protection
                import sys
                sys.stderr.write("Getting Firebase credentials...\n")
                sys.stderr.flush()
                
                creds_path = get_firebase_credentials()
                
                sys.stderr.write(f"Credentials path: {creds_path}\n")
                sys.stderr.flush()
                
                sys.stderr.write("Creating certificate...\n")
                sys.stderr.flush()
                cred = credentials.Certificate(creds_path)
                
                sys.stderr.write("Initializing Firebase Admin SDK...\n")
                sys.stderr.flush()
                firebase_admin.initialize_app(cred)
                
                sys.stderr.write("✓ Firebase Admin SDK initialized\n")
                sys.stderr.flush()
            except FileNotFoundError:
                # Re-raise FileNotFoundError as-is (will be handled in app.py)
                raise
            except Exception as e:
                # Re-raise other exceptions to be handled upstream
                import sys
                import traceback
                sys.stderr.write(f"✗ Firebase initialization error: {e}\n")
                traceback.print_exc(file=sys.stderr)
                sys.stderr.flush()
                raise RuntimeError(f"Firebase initialization failed: {e}") from e

        self.db = firestore.client()

        # Lazy import streamlit and store it for use in methods
        self.st = _get_streamlit()
        
        # Initialize session state if not exists
        if 'authenticated' not in self.st.session_state:
            self.st.session_state['authenticated'] = False
        if 'user_id' not in self.st.session_state:
            self.st.session_state['user_id'] = None
        if 'user_email' not in self.st.session_state:
            self.st.session_state['user_email'] = None
        if 'display_name' not in self.st.session_state:
            self.st.session_state['display_name'] = None
        if 'auth_timestamp' not in self.st.session_state:
            self.st.session_state['auth_timestamp'] = None

    def signup(self, email: str, password: str, display_name: str) -> Dict:
        """
        Create a new user account

        Args:
            email: User's email address
            password: User's password (min 6 characters)
            display_name: User's display name

        Returns:
            Dict with 'success' boolean and 'message' or 'error'
        """
        try:
            # Validate inputs
            if not email or '@' not in email:
                return {"success": False, "error": "Invalid email address"}

            if not password or len(password) < 6:
                return {"success": False, "error": "Password must be at least 6 characters"}

            if not display_name:
                return {"success": False, "error": "Display name is required"}

            # Create user with Firebase Auth
            user = auth.create_user(
                email=email,
                password=password,
                display_name=display_name
            )

            # Create user document in Firestore
            user_ref = self.db.collection('users').document(user.uid)
            user_ref.set({
                'email': email,
                'displayName': display_name,
                'createdAt': firestore.SERVER_TIMESTAMP,
                'lastLogin': firestore.SERVER_TIMESTAMP,
                'preferences': {
                    'defaultMode': 'Stocks',
                    'defaultDataSource': 'Yahoo Finance',
                    'refreshInterval': 5
                }
            })

            return {
                "success": True,
                "message": f"Account created successfully for {email}",
                "user_id": user.uid
            }

        except auth.EmailAlreadyExistsError:
            return {"success": False, "error": "Email already exists"}
        except Exception as e:
            return {"success": False, "error": f"Signup failed: {str(e)}"}

    def login(self, email: str, password: str) -> Dict:
        """
        Authenticate user and create session

        Note: Firebase Admin SDK doesn't verify passwords directly.
        For production, you should use Firebase Auth REST API or client SDK.
        This is a simplified version for the migration.

        Args:
            email: User's email
            password: User's password

        Returns:
            Dict with 'success' boolean and user data or 'error'
        """
        try:
            # Get user by email
            user = auth.get_user_by_email(email)

            # Note: Firebase Admin SDK cannot verify passwords
            # In production, use Firebase Auth REST API or client SDK
            # For now, we'll create the session if user exists
            # TODO: Implement proper password verification

            # Update last login timestamp
            user_ref = self.db.collection('users').document(user.uid)
            user_doc = user_ref.get()

            if not user_doc.exists:
                # Create user document if it doesn't exist
                user_ref.set({
                    'email': email,
                    'displayName': user.display_name or email.split('@')[0],
                    'createdAt': firestore.SERVER_TIMESTAMP,
                    'lastLogin': firestore.SERVER_TIMESTAMP
                })
            else:
                user_ref.update({
                    'lastLogin': firestore.SERVER_TIMESTAMP
                })

            # Set session state
            self.st.session_state['authenticated'] = True
            self.st.session_state['user_id'] = user.uid
            self.st.session_state['user_email'] = user.email
            self.st.session_state['display_name'] = user.display_name or email.split('@')[0]
            self.st.session_state['auth_timestamp'] = datetime.now()

            return {
                "success": True,
                "message": "Login successful",
                "user_id": user.uid,
                "email": user.email,
                "display_name": user.display_name
            }

        except auth.UserNotFoundError:
            return {"success": False, "error": "User not found. Please sign up first."}
        except Exception as e:
            return {"success": False, "error": f"Login failed: {str(e)}"}

    def logout(self):
        """Clear authentication session"""
        self.st.session_state['authenticated'] = False
        self.st.session_state['user_id'] = None
        self.st.session_state['user_email'] = None
        self.st.session_state['display_name'] = None
        self.st.session_state['auth_timestamp'] = None

    def is_authenticated(self) -> bool:
        """
        Check if user is currently authenticated

        Returns:
            True if authenticated, False otherwise
        """
        st = self.st
        # Check if session exists and hasn't expired (24 hour timeout)
        if not st.session_state.get('authenticated', False):
            return False

        # Check session timeout (24 hours)
        auth_time = st.session_state.get('auth_timestamp')
        if auth_time:
            if datetime.now() - auth_time > timedelta(hours=24):
                self.logout()
                return False

        return True

    def get_user_id(self) -> Optional[str]:
        """
        Get current user's ID

        Returns:
            User ID if authenticated, None otherwise
        """
        if self.is_authenticated():
            return self.st.session_state.get('user_id')
        return None

    def get_user_email(self) -> Optional[str]:
        """
        Get current user's email

        Returns:
            Email if authenticated, None otherwise
        """
        if self.is_authenticated():
            return self.st.session_state.get('user_email')
        return None

    def get_display_name(self) -> Optional[str]:
        """
        Get current user's display name

        Returns:
            Display name if authenticated, None otherwise
        """
        if self.is_authenticated():
            return self.st.session_state.get('display_name')
        return None

    def require_auth(self):
        """
        Decorator/guard to require authentication
        Call this at the start of your app to ensure user is logged in

        Returns:
            True if authenticated, False if showing login UI
        """
        if not self.is_authenticated():
            self._show_login_ui()
            self.st.stop()  # Stop execution until authenticated
            return False
        return True

    def _show_login_ui(self):
        """Display login/signup UI (internal method)"""
        st = self.st
        st.title("🔐 TraderQ Login")

        tab1, tab2 = st.tabs(["Login", "Sign Up"])

        with tab1:
            st.subheader("Login to Your Account")
            email = st.text_input("Email", key="login_email")
            password = st.text_input("Password", type="password", key="login_password")

            if st.button("Login", key="login_button", type="primary"):
                if email and password:
                    with st.spinner("Logging in..."):
                        result = self.login(email, password)
                        if result["success"]:
                            st.success(result["message"])
                            st.rerun()
                        else:
                            st.error(result["error"])
                else:
                    st.warning("Please enter email and password")

        with tab2:
            st.subheader("Create New Account")
            signup_email = st.text_input("Email", key="signup_email")
            signup_password = st.text_input("Password (min 6 characters)", type="password", key="signup_password")
            signup_password_confirm = st.text_input("Confirm Password", type="password", key="signup_password_confirm")
            signup_name = st.text_input("Display Name", key="signup_name")

            if st.button("Sign Up", key="signup_button", type="primary"):
                if signup_email and signup_password and signup_name:
                    if signup_password != signup_password_confirm:
                        st.error("Passwords do not match")
                    elif len(signup_password) < 6:
                        st.error("Password must be at least 6 characters")
                    else:
                        with st.spinner("Creating account..."):
                            result = self.signup(signup_email, signup_password, signup_name)
                            if result["success"]:
                                st.success(result["message"])
                                st.info("Please login with your new account")
                            else:
                                st.error(result["error"])
                else:
                    st.warning("Please fill in all fields")

    def delete_user(self, user_id: str = None) -> Dict:
        """
        Delete a user account (admin function)

        Args:
            user_id: User ID to delete (defaults to current user)

        Returns:
            Dict with 'success' boolean and 'message' or 'error'
        """
        try:
            if not user_id:
                user_id = self.get_user_id()

            if not user_id:
                return {"success": False, "error": "No user ID provided"}

            # Delete from Firebase Auth
            auth.delete_user(user_id)

            # Delete user document from Firestore
            self.db.collection('users').document(user_id).delete()

            # Logout if deleting current user
            if user_id == self.get_user_id():
                self.logout()

            return {"success": True, "message": "User account deleted successfully"}

        except Exception as e:
            return {"success": False, "error": f"Failed to delete user: {str(e)}"}

    def update_display_name(self, new_name: str) -> Dict:
        """
        Update current user's display name

        Args:
            new_name: New display name

        Returns:
            Dict with 'success' boolean and 'message' or 'error'
        """
        try:
            user_id = self.get_user_id()
            if not user_id:
                return {"success": False, "error": "Not authenticated"}

            # Update in Firebase Auth
            auth.update_user(user_id, display_name=new_name)

            # Update in Firestore
            self.db.collection('users').document(user_id).update({
                'displayName': new_name
            })

            # Update session state
            self.st.session_state['display_name'] = new_name

            return {"success": True, "message": "Display name updated successfully"}

        except Exception as e:
            return {"success": False, "error": f"Failed to update name: {str(e)}"}


# Convenience function for quick integration
def require_authentication() -> FirebaseAuth:
    """
    Convenience function to require authentication in your Streamlit app

    Usage:
        auth = require_authentication()
        # Your app code here - only runs if user is authenticated

    Returns:
        FirebaseAuth instance
    """
    auth = FirebaseAuth()
    auth.require_auth()
    return auth
