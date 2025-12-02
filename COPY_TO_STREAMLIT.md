# 🔐 Copy Firebase Secrets to Streamlit Cloud

## Quick Setup Guide

### Your Firebase Secrets (Already Generated!)

The TOML configuration is ready in: `.streamlit_secrets_toml.txt`

---

## 📋 Step-by-Step Instructions

### Step 1: Copy the Secrets
1. **Open** `.streamlit_secrets_toml.txt` (double-click it or use Notepad)
2. **Select ALL** (Ctrl+A)
3. **Copy** (Ctrl+C)

### Step 2: Go to Streamlit Cloud
1. Open: **https://share.streamlit.io**
2. Sign in with your GitHub account
3. Find your app: **traderq-sma-tracker**
4. Click on it

### Step 3: Configure Secrets
1. Click **"⚙️ Settings"** (gear icon in top right)
2. Click **"Secrets"** tab
3. **Paste** the entire contents you copied (Ctrl+V)
4. Click **"Save"**

### Step 4: Wait for Redeployment
- Streamlit Cloud will automatically redeploy (2-3 minutes)
- Watch the logs to see it update
- Your app should now work! 🎉

---

## 🚀 Or Run the Helper Script

Just run:
```powershell
.\setup_streamlit_secrets.ps1
```

This will:
- ✅ Open the TOML file for you
- ✅ Open Streamlit Cloud in your browser
- ✅ Show you step-by-step instructions

---

## ✅ Verify It Worked

After saving secrets and waiting 2-3 minutes:
1. Check your app URL: `https://traderq-sma-tracker.streamlit.app`
2. You should see either:
   - ✅ The login page (Firebase working!)
   - ❌ A helpful error message (if something is wrong)

---

## 🔍 Troubleshooting

**If the app still shows an error:**
- Check the Streamlit Cloud logs for specific error messages
- Verify the secrets were saved correctly
- Make sure you copied the ENTIRE TOML file (including the `[firebase]` header)

