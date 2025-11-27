# TraderQ Firebase Migration - Quick Start Guide

This guide will get you up and running with Firebase in 15-20 minutes.

## Prerequisites

- Google account
- Node.js installed (for Firebase CLI)
- Python 3.8+ with pip

---

## 🚀 Quick Setup (Copy-Paste Commands)

### Step 1: Install Dependencies

```bash
# Install Python packages
pip install -r requirements.txt

# Install Firebase CLI (if not already installed)
npm install -g firebase-tools
```

### Step 2: Verify Installation

```bash
# Run the verification script
python verify_setup.py
```

This will show you what's missing and what needs to be done.

---

## 🔥 Firebase Setup

### Step 1: Create Firebase Project

1. Go to [Firebase Console](https://console.firebase.google.com/)
2. Click "Add project" or "Create a project"
3. Enter project name: `traderq` (or your preferred name)
4. **Disable** Google Analytics (not needed for this app)
5. Click "Create project"
6. Wait for project to be created (~30 seconds)

### Step 2: Enable Firestore Database

1. In Firebase Console, click "Firestore Database" in left menu
2. Click "Create database"
3. Select **"Start in production mode"** (we have custom security rules)
4. Choose location: `us-central` or closest to you
5. Click "Enable"
6. Wait for database to be created (~1 minute)

### Step 3: Enable Authentication

1. In Firebase Console, click "Authentication" in left menu
2. Click "Get started"
3. Click "Email/Password" under "Sign-in method"
4. Toggle **Enable** to ON
5. Leave "Email link (passwordless sign-in)" OFF
6. Click "Save"

### Step 4: Download Service Account Key

1. In Firebase Console, click ⚙️ (Settings) → "Project settings"
2. Click "Service accounts" tab
3. Click "Generate new private key"
4. Click "Generate key" in the confirmation dialog
5. Save the downloaded JSON file as `serviceAccountKey.json` in your `traderq` directory

**IMPORTANT:** Never commit this file to Git! (Already in .gitignore)

### Step 5: Initialize Firebase in Your Project

```bash
# Login to Firebase
firebase login

# Initialize Firebase project
firebase init

# When prompted, select:
# - Firestore: Configure security rules and indexes files
# - Hosting: Configure files for Firebase Hosting

# When asked "What do you want to use as your public directory?"
# Press Enter (accept default: public)

# When asked "Configure as a single-page app?"
# Type: n (No)

# When asked "Set up automatic builds and deploys with GitHub?"
# Type: n (No) - we'll do this manually later

# When asked about overwriting files, choose:
# - firestore.rules: N (No - keep existing)
# - firestore.indexes.json: N (No - keep existing)
# - firebase.json: N (No - keep existing)
```

### Step 6: Deploy Firestore Rules

```bash
# Deploy security rules to Firebase
firebase deploy --only firestore:rules,firestore:indexes
```

You should see:
```
✔  Deploy complete!
```

---

## ✅ Verify Setup

Run the verification script again:

```bash
python verify_setup.py
```

All checks should now pass!

---

## 🧪 Test the Application Locally

### Step 1: Run the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### Step 2: Create an Account

1. You'll see the login screen
2. Click the "Sign Up" tab
3. Enter:
   - Email: your email address
   - Password: at least 6 characters
   - Display Name: your name
4. Click "Sign Up"
5. Switch to "Login" tab and log in

### Step 3: Test Features

Try these to ensure everything works:

1. **Custom Tickers:**
   - Go to the Tracker tab
   - Scroll down to "Manage Custom Tickers"
   - Add a ticker (e.g., "AAPL")
   - Refresh the page - it should still be there

2. **Alerts:**
   - Go to the Alerts tab
   - Create a price alert
   - Refresh the page - alert should persist

3. **Portfolio:**
   - Go to the Portfolio tab
   - Add some tickers
   - Refresh the page - portfolio should persist

4. **Logout/Login:**
   - Click "Logout" in sidebar
   - Log in again
   - All your data should still be there

**If all of this works, your Firebase integration is successful!** 🎉

---

## 📦 Migrate Existing Data (Optional)

If you have existing data in JSON files, migrate it:

```bash
python data_migration.py
```

Follow the prompts:
1. Enter your email (the one you signed up with)
2. Type `yes` to confirm migration
3. Wait for migration to complete

Your existing data will be:
- Backed up to `json_backup/` directory
- Uploaded to Firestore
- Accessible when you log in

---

## 🌐 Deploy to Streamlit Community Cloud

### Step 1: Push to GitHub

```bash
# Initialize git (if not already)
git init

# Add all files
git add .

# Create commit
git commit -m "Add Firebase integration"

# Create GitHub repo and push
# (Follow GitHub's instructions to create a new repository)
git remote add origin https://github.com/YOUR_USERNAME/traderq.git
git branch -M main
git push -u origin main
```

### Step 2: Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click "New app"
3. Connect GitHub account (if not already)
4. Select:
   - Repository: `YOUR_USERNAME/traderq`
   - Branch: `main`
   - Main file: `app.py`
5. Click "Advanced settings"

### Step 3: Configure Secrets

In the "Secrets" section, paste your Firebase credentials in TOML format:

1. Open your `serviceAccountKey.json` file
2. Convert it to TOML format:

```toml
[firebase]
type = "service_account"
project_id = "YOUR_PROJECT_ID"
private_key_id = "YOUR_PRIVATE_KEY_ID"
private_key = "-----BEGIN PRIVATE KEY-----\nYOUR_PRIVATE_KEY_HERE\n-----END PRIVATE KEY-----\n"
client_email = "firebase-adminsdk-xxxxx@YOUR_PROJECT_ID.iam.gserviceaccount.com"
client_id = "YOUR_CLIENT_ID"
auth_uri = "https://accounts.google.com/o/oauth2/auth"
token_uri = "https://oauth2.googleapis.com/token"
auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
client_x509_cert_url = "YOUR_CERT_URL"
```

**Important:**
- Keep the `\n` characters in the private_key
- Put the entire private key on ONE line
- All values must be in quotes

### Step 4: Deploy

1. Click "Deploy"
2. Wait 2-3 minutes for deployment
3. Your app will be live at: `https://YOUR_USERNAME-traderq-xxxxx.streamlit.app`

### Step 5: Test Production

1. Visit your Streamlit Cloud URL
2. Sign up with a new account (or use existing)
3. Test all features
4. Verify data persists

---

## 🎯 Summary Checklist

- [ ] Install Python dependencies (`pip install -r requirements.txt`)
- [ ] Install Firebase CLI (`npm install -g firebase-tools`)
- [ ] Create Firebase project
- [ ] Enable Firestore Database
- [ ] Enable Email/Password Authentication
- [ ] Download service account key → `serviceAccountKey.json`
- [ ] Initialize Firebase (`firebase init`)
- [ ] Deploy Firestore rules (`firebase deploy --only firestore`)
- [ ] Run verification script (`python verify_setup.py`)
- [ ] Test locally (`streamlit run app.py`)
- [ ] Create account and test features
- [ ] Migrate existing data (optional: `python data_migration.py`)
- [ ] Push to GitHub
- [ ] Deploy to Streamlit Cloud
- [ ] Configure secrets in Streamlit Cloud
- [ ] Test production deployment

---

## ❓ Troubleshooting

### "Module not found" errors

```bash
pip install -r requirements.txt
```

### "Service account key not found"

Make sure `serviceAccountKey.json` is in the project root directory (same folder as `app.py`).

### "Permission denied" in Firestore

Run:
```bash
firebase deploy --only firestore:rules
```

This ensures security rules are deployed.

### App crashes on startup

1. Check if all dependencies are installed
2. Run `python verify_setup.py`
3. Check that `serviceAccountKey.json` exists and is valid JSON

### Can't log in / Sign up not working

1. Verify Email/Password authentication is enabled in Firebase Console
2. Check Firebase Console → Authentication → Users - any error messages?
3. Check browser console (F12) for JavaScript errors

### Data not persisting

1. Log out and log back in
2. Check Firebase Console → Firestore Database - is data being written?
3. Check security rules - users should only access their own data

---

## 📚 Additional Resources

- [Firebase Console](https://console.firebase.google.com/)
- [Streamlit Cloud](https://share.streamlit.io)
- [FIREBASE_SETUP.md](./FIREBASE_SETUP.md) - Detailed Firebase setup
- [DEPLOYMENT.md](./DEPLOYMENT.md) - Detailed deployment guide

---

## 🆘 Need Help?

1. Run `python verify_setup.py` to diagnose issues
2. Check the logs in Firebase Console → Firestore Database → Usage
3. Check Streamlit Cloud logs (if deployed)
4. Review security rules in Firebase Console → Firestore Database → Rules

---

**🎉 Congratulations!** Once everything is working, you have a fully cloud-native, multi-user trading analytics platform!

**Free tier limits:**
- Firebase: 50K reads/day, 20K writes/day, 1GB storage
- Streamlit: Unlimited public apps

This should be plenty for personal use and even small teams!
