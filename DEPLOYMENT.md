# TraderQ Deployment Guide

This guide covers deploying TraderQ to **Streamlit Community Cloud** - a free, serverless platform for Streamlit apps.

## Why Streamlit Community Cloud?

- ✅ **100% Serverless** - No Docker, no containers, no infrastructure management
- ✅ **Free tier** - Perfect for personal projects
- ✅ **Native Streamlit support** - Built specifically for Streamlit apps
- ✅ **Auto-scaling** - Handles traffic automatically
- ✅ **GitHub integration** - Deploy directly from your repository
- ✅ **Works with Firebase** - Full support for Firebase Python SDK

---

## Prerequisites

- GitHub account
- Firebase project with Firestore and Authentication enabled (see FIREBASE_SETUP.md)
- Service account key downloaded (serviceAccountKey.json)

---

## Deployment Steps

### Step 1: Push Code to GitHub

1. Create a new GitHub repository (public or private)

2. Push your TraderQ code to GitHub:

```bash
git init
git add .
git commit -m "Initial commit - TraderQ with Firebase"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/traderq.git
git push -u origin main
```

**Important**: Make sure `.gitignore` is properly configured (it already is) to exclude:
- `serviceAccountKey.json`
- `.env`
- Local JSON files
- `__pycache__` and other temporary files

---

### Step 2: Set Up Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)

2. Click "New app"

3. Connect your GitHub account (if not already connected)

4. Select your repository, branch, and main file:
   - **Repository**: `YOUR_USERNAME/traderq`
   - **Branch**: `main`
   - **Main file path**: `app.py`

5. Click "Advanced settings" before deploying

---

### Step 3: Configure Secrets

Streamlit Cloud uses a TOML-based secrets system instead of environment files.

1. In the "Advanced settings" section, find the "Secrets" text area

2. Add your Firebase service account credentials in TOML format:

```toml
# Firebase Configuration
[firebase]
type = "service_account"
project_id = "your-project-id"
private_key_id = "your-private-key-id"
private_key = "-----BEGIN PRIVATE KEY-----\nYOUR_PRIVATE_KEY_HERE\n-----END PRIVATE KEY-----\n"
client_email = "firebase-adminsdk-xxxxx@your-project-id.iam.gserviceaccount.com"
client_id = "your-client-id"
auth_uri = "https://accounts.google.com/o/oauth2/auth"
token_uri = "https://oauth2.googleapis.com/token"
auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
client_x509_cert_url = "your-cert-url"
```

**How to get these values:**
Open your `serviceAccountKey.json` file and copy each field into the TOML format above.

**Important notes about the private_key:**
- Keep the quotes around the key
- Keep the `\n` characters (they represent line breaks)
- Make sure the entire key is on ONE line in the TOML format
- Example: `private_key = "-----BEGIN PRIVATE KEY-----\nMIIE...\n-----END PRIVATE KEY-----\n"`

---

### Step 4: Update Code for Streamlit Cloud

Create a new file called `firebase_config.py` to handle credentials:

```python
# firebase_config.py
import streamlit as st
import os
import json
import tempfile


def get_firebase_credentials():
    """
    Get Firebase credentials from Streamlit secrets or local file
    Returns path to credentials file
    """
    # Check if running on Streamlit Cloud
    if hasattr(st, 'secrets') and 'firebase' in st.secrets:
        # Create a temporary file with the credentials
        creds_dict = dict(st.secrets['firebase'])

        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            json.dump(creds_dict, f)
            return f.name
    else:
        # Local development - use serviceAccountKey.json
        service_account_path = os.getenv(
            'GOOGLE_APPLICATION_CREDENTIALS',
            './serviceAccountKey.json'
        )

        if not os.path.exists(service_account_path):
            raise FileNotFoundError(
                f"Service account key not found at {service_account_path}. "
                "Please follow FIREBASE_SETUP.md to download your service account key."
            )

        return service_account_path
```

Then update `firebase_auth.py` and `data_migration.py` to use this new function:

**In firebase_auth.py** (line 20-36), replace the `__init__` method:

```python
def __init__(self):
    """Initialize Firebase Admin SDK if not already initialized"""
    if not firebase_admin._apps:
        from firebase_config import get_firebase_credentials

        try:
            cred_path = get_firebase_credentials()
            cred = credentials.Certificate(cred_path)
            firebase_admin.initialize_app(cred)
        except Exception as e:
            st.error(f"Failed to initialize Firebase: {e}")
            st.stop()

    self.db = firestore.client()
    # ... rest of the code stays the same
```

**In data_migration.py** (line 38-58), update the `initialize_firebase` function:

```python
def initialize_firebase():
    """Initialize Firebase Admin SDK"""
    try:
        if not firebase_admin._apps:
            # For migration script, always use local serviceAccountKey.json
            service_account_path = os.getenv(
                'GOOGLE_APPLICATION_CREDENTIALS',
                './serviceAccountKey.json'
            )

            if not os.path.exists(service_account_path):
                print(f"❌ Error: Service account key not found at {service_account_path}")
                print("Please follow FIREBASE_SETUP.md to download your service account key.")
                return False

            cred = credentials.Certificate(service_account_path)
            firebase_admin.initialize_app(cred)
            print("✅ Firebase initialized successfully")
            return True
    except Exception as e:
        print(f"❌ Error initializing Firebase: {e}")
        return False
```

---

### Step 5: Deploy!

1. Click "Deploy" in Streamlit Cloud

2. Wait for the app to build and deploy (usually 2-3 minutes)

3. Your app will be available at: `https://YOUR_USERNAME-traderq-main-app-xxxxx.streamlit.app`

---

## Post-Deployment

### Test Your Deployment

1. Visit your Streamlit Cloud URL
2. Test sign up functionality
3. Test login functionality
4. Verify Firestore data is being saved

### Update Your App

Streamlit Cloud automatically redeploys when you push to GitHub:

```bash
# Make changes to your code
git add .
git commit -m "Update feature"
git push
```

Your app will automatically rebuild and redeploy within a few minutes.

### View Logs

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click on your app
3. Click "Manage app" → "Logs"

This shows real-time logs and errors from your application.

---

## Migrate Your Local Data

After deployment, you can migrate your local JSON data to Firestore:

1. Run the migration script **locally** (not on Streamlit Cloud):

```bash
python data_migration.py
```

2. Enter the email address you used to sign up in the deployed app

3. Confirm the migration

4. Your data will be uploaded to Firestore and accessible in the deployed app

---

## Custom Domain (Optional)

Streamlit Community Cloud supports custom domains on paid plans. Alternatively, you can use Firebase Hosting as a proxy:

### Option 1: Firebase Hosting Proxy

1. Update `firebase.json`:

```json
{
  "hosting": {
    "public": "public",
    "rewrites": [
      {
        "source": "**",
        "destination": "https://YOUR_USERNAME-traderq-main-app-xxxxx.streamlit.app"
      }
    ]
  }
}
```

2. Deploy Firebase Hosting:

```bash
firebase deploy --only hosting
```

3. Your app will be available at `https://YOUR_PROJECT_ID.web.app`

---

## Cost Comparison

### Streamlit Community Cloud
- **Free tier**: Unlimited public apps, 1 private app
- **Pro plan**: $20/month for unlimited private apps and custom resources
- **Teams**: $40/user/month for team collaboration

### Firebase (Firestore + Auth)
- **Spark Plan (Free)**:
  - Firestore: 1 GB storage, 50K reads/day, 20K writes/day
  - Authentication: Unlimited users
  - Enough for personal use and testing
- **Blaze Plan (Pay as you go)**:
  - Firestore: $0.18/GB storage, $0.06 per 100K reads, $0.18 per 100K writes
  - Free tier included
  - Typical cost for moderate use: $5-15/month

**Total cost for personal use: $0-20/month** (depending on usage and privacy needs)

---

## Troubleshooting

### Issue: "Firebase initialization failed"

**Solution**: Check your secrets configuration:
1. Go to Streamlit Cloud → Manage app → Secrets
2. Verify all fields are present and correctly formatted
3. Make sure `private_key` has `\n` characters preserved

### Issue: "Module not found" errors

**Solution**: Ensure `requirements.txt` includes all dependencies:
```bash
git add requirements.txt
git commit -m "Update dependencies"
git push
```

### Issue: "Permission denied" in Firestore

**Solution**:
1. Check `firestore.rules` - users should only access their own data
2. Verify the user is authenticated before making Firestore calls
3. Check Firebase Console → Firestore → Rules

### Issue: App runs locally but fails on Streamlit Cloud

**Solution**:
1. Check Streamlit Cloud logs for specific errors
2. Verify `firebase_config.py` is included in your repository
3. Make sure secrets are configured correctly
4. Test with the same Python version locally (check `runtime.txt`)

---

## Security Best Practices

1. **Never commit serviceAccountKey.json** - Already in .gitignore
2. **Use Firestore security rules** - Already configured in firestore.rules
3. **Enable App Check** (optional) - Prevent abuse of Firebase services
4. **Monitor usage** - Set up billing alerts in Firebase Console
5. **Rotate service account keys** - Every 90 days for production apps

---

## Monitoring and Analytics

### Firebase Console

Monitor your app usage:
1. Go to [Firebase Console](https://console.firebase.google.com/)
2. Select your project
3. View:
   - **Authentication** → User activity and sign-ups
   - **Firestore** → Database usage and performance
   - **Usage and billing** → API calls and storage

### Streamlit Cloud Analytics

View app metrics:
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click your app → Analytics
3. See: viewers, runtime, resource usage

---

## Next Steps

After successful deployment:

1. ✅ Test all features thoroughly
2. ✅ Migrate your local data using `data_migration.py`
3. ✅ Share your app URL with others
4. ✅ Monitor Firebase usage to ensure you stay within free tier
5. ⏳ (Optional) Set up custom domain via Firebase Hosting
6. ⏳ (Optional) Integrate with Cloud Functions for advanced features
7. ⏳ (Optional) Move secrets to Google Secret Manager (Phase 2)

---

## Additional Resources

- [Streamlit Community Cloud Docs](https://docs.streamlit.io/streamlit-community-cloud)
- [Streamlit Secrets Management](https://docs.streamlit.io/streamlit-community-cloud/get-started/deploy-an-app/connect-to-data-sources/secrets-management)
- [Firebase Documentation](https://firebase.google.com/docs)
- [Firestore Security Rules](https://firebase.google.com/docs/firestore/security/get-started)

---

## Support

If you encounter issues:

1. Check Streamlit Cloud logs
2. Check Firebase Console for errors
3. Review firestore.rules for permission issues
4. Verify secrets are correctly configured
5. Test authentication flow step by step
