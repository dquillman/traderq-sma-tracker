# 🚀 TraderQ Deployment Checklist

Use this checklist to ensure your app is ready for cloud deployment.

## ✅ Pre-Deployment Verification

### 1. Code Readiness
- [x] All data persistence functions use Firebase Firestore
- [x] No local file storage dependencies
- [x] Firebase authentication integrated
- [x] Environment detection works (local vs cloud)
- [x] All imports are correct
- [x] Requirements.txt includes all dependencies

### 2. Required Files
- [x] `app.py` - Main application file
- [x] `firebase_auth.py` - Authentication module
- [x] `firebase_config.py` - Configuration module (handles local/cloud)
- [x] `firebase_db.py` - Database operations
- [x] `requirements.txt` - All Python dependencies
- [x] `Procfile` - Deployment configuration (optional for Streamlit Cloud)
- [x] `.gitignore` - Excludes sensitive files

### 3. Firebase Setup
- [ ] Firebase project created
- [ ] Firestore Database enabled
- [ ] Firebase Authentication enabled (Email/Password)
- [ ] Service account key downloaded (`serviceAccountKey.json`)
- [ ] Firestore security rules deployed
- [ ] Firestore indexes created (if needed)

### 4. Security Check
- [x] `serviceAccountKey.json` is in `.gitignore`
- [x] `.env` files are in `.gitignore`
- [x] No hardcoded credentials in code
- [x] All sensitive data uses Firebase secrets

## 📋 Deployment Steps

### Step 1: Prepare GitHub Repository

```bash
# Initialize git (if not already done)
git init

# Add all files
git add .

# Commit
git commit -m "TraderQ with Firebase - Ready for deployment"

# Create GitHub repository first, then:
git remote add origin https://github.com/YOUR_USERNAME/traderq.git
git branch -M main
git push -u origin main
```

**Important**: Verify these files are NOT committed:
- `serviceAccountKey.json` ✅ (in .gitignore)
- `.env` ✅ (in .gitignore)
- `.streamlit/secrets.toml` ✅ (in .gitignore)
- Any `.json` config files ✅ (in .gitignore)

### Step 2: Deploy to Streamlit Community Cloud

1. **Go to Streamlit Cloud**
   - Visit: https://share.streamlit.io
   - Sign in with GitHub

2. **Create New App**
   - Click "New app"
   - Select your repository: `YOUR_USERNAME/traderq`
   - Branch: `main`
   - Main file path: `app.py`
   - App URL: (auto-generated or custom)

3. **Configure Secrets** (CRITICAL)
   - Click "Advanced settings"
   - In "Secrets" text area, paste your Firebase credentials in TOML format:

```toml
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
1. Open your `serviceAccountKey.json` file
2. Copy each field value into the TOML format above
3. For `private_key`: Keep it on ONE line with `\n` characters preserved

4. **Deploy**
   - Click "Deploy"
   - Wait 2-3 minutes for build
   - Your app will be live at: `https://YOUR_USERNAME-traderq-main-app-xxxxx.streamlit.app`

### Step 3: Post-Deployment Testing

- [ ] Visit your deployed app URL
- [ ] Test user signup
- [ ] Test user login
- [ ] Test adding custom tickers
- [ ] Test creating alerts
- [ ] Test portfolio features
- [ ] Verify data persists (refresh page)
- [ ] Test logout/login flow

### Step 4: Firestore Security Rules

Ensure your Firestore rules are deployed:

```bash
firebase deploy --only firestore:rules
```

Your `firestore.rules` should ensure users can only access their own data.

## 🔧 Troubleshooting

### Issue: "Firebase initialization failed"
**Solution:**
- Check Streamlit Cloud secrets are correctly formatted
- Verify all fields from `serviceAccountKey.json` are present
- Ensure `private_key` has `\n` characters preserved

### Issue: "Module not found"
**Solution:**
- Verify `requirements.txt` includes all dependencies
- Check Streamlit Cloud logs for missing packages
- Ensure Firebase packages are in requirements.txt

### Issue: "Permission denied" in Firestore
**Solution:**
- Deploy Firestore security rules: `firebase deploy --only firestore:rules`
- Verify rules allow authenticated users to access their own data
- Check Firebase Console → Firestore → Rules

### Issue: App works locally but fails on cloud
**Solution:**
- Check Streamlit Cloud logs (Manage app → Logs)
- Verify `firebase_config.py` is in repository
- Ensure secrets are configured correctly
- Test with same Python version locally

## 📊 Monitoring

### Firebase Console
- Monitor usage: https://console.firebase.google.com
- Check Authentication → Users
- Check Firestore → Data
- Monitor Usage & Billing

### Streamlit Cloud
- View analytics: share.streamlit.io → Your App → Analytics
- Check logs: Manage app → Logs
- Monitor resource usage

## 🎯 Next Steps After Deployment

1. ✅ Share your app URL with users
2. ✅ Test all features thoroughly
3. ✅ Monitor Firebase usage (stay within free tier)
4. ✅ Set up billing alerts in Firebase Console
5. ⏳ (Optional) Migrate local data using `data_migration.py`
6. ⏳ (Optional) Set up custom domain
7. ⏳ (Optional) Configure email notifications

## 📝 Quick Reference

- **Streamlit Cloud**: https://share.streamlit.io
- **Firebase Console**: https://console.firebase.google.com
- **GitHub**: https://github.com
- **Deployment Guide**: See `DEPLOYMENT.md` for detailed instructions

---

**Status**: ✅ Ready for deployment!
**Last Updated**: After Firebase conversion completion

