# TraderQ Firebase Setup Checklist

Use this checklist to track your progress through the setup process.

## ⚙️ Prerequisites

- [ ] Python 3.8 or higher installed
- [ ] Node.js installed (for Firebase CLI)
- [ ] Google account created
- [ ] Git installed

## 📦 Installation

- [ ] Clone/download TraderQ repository
- [ ] Navigate to project directory: `cd traderq`
- [ ] Install Python dependencies: `pip install -r requirements.txt`
- [ ] Install Firebase CLI: `npm install -g firebase-tools`
- [ ] Run verification: `python verify_setup.py`

## 🔥 Firebase Console Setup

### Create Project
- [ ] Go to https://console.firebase.google.com/
- [ ] Click "Add project" or "Create a project"
- [ ] Enter project name (e.g., "traderq")
- [ ] Disable Google Analytics
- [ ] Click "Create project"
- [ ] Wait for project creation

### Enable Firestore
- [ ] Click "Firestore Database" in left menu
- [ ] Click "Create database"
- [ ] Select "Start in production mode"
- [ ] Choose location (e.g., us-central)
- [ ] Click "Enable"
- [ ] Wait for database creation

### Enable Authentication
- [ ] Click "Authentication" in left menu
- [ ] Click "Get started"
- [ ] Click "Email/Password" provider
- [ ] Toggle "Enable" to ON
- [ ] Click "Save"

### Download Service Account Key
- [ ] Click ⚙️ → "Project settings"
- [ ] Click "Service accounts" tab
- [ ] Click "Generate new private key"
- [ ] Confirm and download JSON file
- [ ] Save as `serviceAccountKey.json` in project root
- [ ] Verify file exists: `ls serviceAccountKey.json`

## 🔧 Local Setup

### Firebase CLI
- [ ] Login to Firebase: `firebase login`
- [ ] Initialize project: `firebase init`
  - [ ] Select: Firestore and Hosting
  - [ ] Use existing project
  - [ ] Select your project from list
  - [ ] Keep default `firestore.rules` (press N when asked to overwrite)
  - [ ] Keep default `firestore.indexes.json` (press N)
  - [ ] Keep default `firebase.json` (press N)
  - [ ] Public directory: press Enter (keep "public")
  - [ ] Single-page app: N (No)
  - [ ] GitHub autodeploy: N (No)

### Deploy Rules
- [ ] Deploy Firestore rules: `firebase deploy --only firestore:rules,firestore:indexes`
- [ ] Verify success message: "Deploy complete!"

### Verification
- [ ] Run: `python verify_setup.py`
- [ ] All checks should pass (or at least 7/8)

## 🧪 Local Testing

### First Run
- [ ] Start app: `streamlit run app.py`
- [ ] App opens in browser (http://localhost:8501)
- [ ] Login screen appears

### Create Account
- [ ] Click "Sign Up" tab
- [ ] Enter email address
- [ ] Enter password (min 6 characters)
- [ ] Enter display name
- [ ] Click "Sign Up"
- [ ] See success message

### Login
- [ ] Switch to "Login" tab
- [ ] Enter email
- [ ] Enter password
- [ ] Click "Login"
- [ ] App loads successfully
- [ ] See your name in sidebar

### Test Features
- [ ] **Custom Tickers**: Add a ticker, refresh page, verify it persists
- [ ] **Alerts**: Create an alert, refresh page, verify it persists
- [ ] **Portfolio**: Add holdings, refresh page, verify they persist
- [ ] **Watchlist**: Create a watchlist, refresh page, verify it persists
- [ ] **Logout**: Click logout button, verify you're logged out
- [ ] **Re-login**: Log back in, verify all data is still there

### Verify Firebase Console
- [ ] Go to Firebase Console → Firestore Database
- [ ] See "users" collection
- [ ] See your user ID
- [ ] See subcollections (customTickers, alerts, portfolio, etc.)

## 📤 Data Migration (If you have existing data)

- [ ] Stop the app (Ctrl+C)
- [ ] Run: `python data_migration.py`
- [ ] Enter your email
- [ ] Confirm migration: `yes`
- [ ] Wait for completion
- [ ] Check backup location: `ls json_backup/`
- [ ] Restart app and verify migrated data appears

## 🌐 Deployment to Streamlit Cloud

### GitHub Setup
- [ ] Create new GitHub repository
- [ ] Initialize git: `git init`
- [ ] Add files: `git add .`
- [ ] Commit: `git commit -m "Add Firebase integration"`
- [ ] Add remote: `git remote add origin <YOUR_REPO_URL>`
- [ ] Push: `git push -u origin main`

### Streamlit Cloud Setup
- [ ] Go to https://share.streamlit.io
- [ ] Click "New app"
- [ ] Connect GitHub account
- [ ] Select repository: `YOUR_USERNAME/traderq`
- [ ] Select branch: `main`
- [ ] Main file: `app.py`
- [ ] Click "Advanced settings"

### Configure Secrets
- [ ] Run: `python convert_key_to_toml.py`
- [ ] Open: `.streamlit_secrets_toml.txt`
- [ ] Copy all contents
- [ ] Paste into Streamlit Cloud "Secrets" field
- [ ] Click "Deploy"

### Verify Deployment
- [ ] Wait for deployment (2-3 minutes)
- [ ] App URL opens automatically
- [ ] Login screen appears
- [ ] Create account or login
- [ ] Test all features
- [ ] Verify data persists

## 🎯 Final Checks

- [ ] Local app works: `streamlit run app.py`
- [ ] All features work locally
- [ ] Deployed app works on Streamlit Cloud
- [ ] All features work in production
- [ ] Data persists between sessions
- [ ] Can logout and login without issues
- [ ] Firebase Console shows data being written
- [ ] No errors in Streamlit Cloud logs

## 🔒 Security Verification

- [ ] `serviceAccountKey.json` is in `.gitignore`
- [ ] `serviceAccountKey.json` is NOT committed to Git
- [ ] `.env` files are in `.gitignore`
- [ ] `.streamlit/secrets.toml` is in `.gitignore`
- [ ] Firestore security rules are deployed
- [ ] Each user can only see their own data

## 📊 Monitor Usage

- [ ] Bookmark Firebase Console: https://console.firebase.google.com/
- [ ] Check: Firestore Database → Usage tab
- [ ] Monitor reads/writes per day
- [ ] Set up billing alerts (if desired)

---

## ✅ Success Criteria

You're done when:
1. ✅ App runs locally without errors
2. ✅ Can create account and login
3. ✅ All features work (tickers, alerts, portfolio, etc.)
4. ✅ Data persists after logout/login
5. ✅ App is deployed to Streamlit Cloud
6. ✅ Production app works the same as local
7. ✅ No sensitive files in Git

---

## 🎉 Congratulations!

If all items are checked, you have successfully migrated TraderQ to Firebase!

**Next steps:**
- Use the app and monitor Firebase usage
- Share your Streamlit Cloud URL with others (if desired)
- Set up custom domain (optional)
- Explore additional Firebase features (Cloud Functions, etc.)

**Costs:**
- Firebase Spark Plan: **FREE** (50K reads/day, 20K writes/day)
- Streamlit Community Cloud: **FREE** (public apps)
- **Total: $0/month**

Enjoy your cloud-native trading analytics platform! 🚀
