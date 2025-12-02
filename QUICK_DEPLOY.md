# 🚀 Quick Deployment Guide - TraderQ

Get your TraderQ app live on the web in 15 minutes!

## Prerequisites Checklist

- [ ] GitHub account
- [ ] Firebase project created (see `FIREBASE_SETUP.md`)
- [ ] `serviceAccountKey.json` downloaded from Firebase
- [ ] Firestore Database enabled
- [ ] Firebase Authentication enabled (Email/Password)

## Step-by-Step Deployment

### 1. Prepare Your Code (5 minutes)

```bash
# Make sure you're in the traderq directory
cd G:\Users\daveq\traderq

# Initialize git (if not already done)
git init

# Check what will be committed (should NOT include serviceAccountKey.json)
git status

# Add all files
git add .

# Commit
git commit -m "TraderQ with Firebase - Ready for deployment"
```

### 2. Push to GitHub (3 minutes)

1. **Create a new repository on GitHub:**
   - Go to https://github.com/new
   - Repository name: `traderq` (or your preferred name)
   - Choose Public or Private
   - **DO NOT** initialize with README, .gitignore, or license
   - Click "Create repository"

2. **Push your code:**
   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/traderq.git
   git branch -M main
   git push -u origin main
   ```

### 3. Convert Firebase Credentials to TOML (2 minutes)

**Option A: Use the helper script**
```bash
python convert_key_to_toml.py
```
This will read your `serviceAccountKey.json` and output the TOML format.

**Option B: Manual conversion**
1. Open `serviceAccountKey.json` in a text editor
2. Copy each field value
3. Use the template in `DEPLOYMENT.md` (Step 3)

### 4. Deploy to Streamlit Cloud (5 minutes)

1. **Go to Streamlit Cloud:**
   - Visit: https://share.streamlit.io
   - Sign in with your GitHub account

2. **Create New App:**
   - Click "New app" button
   - Repository: Select `YOUR_USERNAME/traderq`
   - Branch: `main`
   - Main file path: `app.py`
   - App URL: (auto-generated or choose custom)

3. **Configure Secrets (CRITICAL):**
   - Click "Advanced settings" before deploying
   - In the "Secrets" text area, paste your TOML credentials
   - Format should look like:
   ```toml
   [firebase]
   type = "service_account"
   project_id = "your-project-id"
   private_key_id = "your-private-key-id"
   private_key = "-----BEGIN PRIVATE KEY-----\nYOUR_KEY\n-----END PRIVATE KEY-----\n"
   client_email = "firebase-adminsdk-xxxxx@your-project.iam.gserviceaccount.com"
   client_id = "your-client-id"
   auth_uri = "https://accounts.google.com/o/oauth2/auth"
   token_uri = "https://oauth2.googleapis.com/token"
   auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
   client_x509_cert_url = "your-cert-url"
   ```

4. **Deploy:**
   - Click "Deploy" button
   - Wait 2-3 minutes for build
   - Your app will be live! 🎉

### 5. Deploy Firestore Security Rules (2 minutes)

```bash
# Make sure you're logged into Firebase CLI
firebase login

# Deploy the security rules
firebase deploy --only firestore:rules
```

This ensures users can only access their own data.

### 6. Test Your Deployment

Visit your app URL (e.g., `https://yourusername-traderq-main-app-xxxxx.streamlit.app`)

Test these features:
- [ ] Sign up with a new account
- [ ] Log in
- [ ] Add custom tickers
- [ ] Create an alert
- [ ] Add to portfolio
- [ ] Log out and log back in (verify data persists)

## 🎯 Your App is Now Live!

**App URL:** `https://YOUR_USERNAME-traderq-main-app-xxxxx.streamlit.app`

Share this URL with anyone - they can access your app from anywhere in the world!

## 🔄 Updating Your App

Whenever you make changes:

```bash
git add .
git commit -m "Your update message"
git push
```

Streamlit Cloud will automatically redeploy your app within a few minutes.

## ❓ Troubleshooting

### "Firebase initialization failed"
- Check your secrets in Streamlit Cloud are correctly formatted
- Verify all fields from `serviceAccountKey.json` are present
- Ensure `private_key` has `\n` characters preserved

### "Permission denied"
- Deploy Firestore rules: `firebase deploy --only firestore:rules`
- Check Firebase Console → Firestore → Rules

### "Module not found"
- Verify `requirements.txt` includes all packages
- Check Streamlit Cloud logs for specific errors

## 📚 Additional Resources

- **Detailed Deployment Guide:** See `DEPLOYMENT.md`
- **Deployment Checklist:** See `DEPLOYMENT_CHECKLIST.md`
- **Firebase Setup:** See `FIREBASE_SETUP.md`
- **Streamlit Cloud Docs:** https://docs.streamlit.io/streamlit-community-cloud

---

**Need Help?** Check the logs:
- Streamlit Cloud: share.streamlit.io → Your App → Manage app → Logs
- Firebase Console: console.firebase.google.com → Your Project → Firestore → Usage

