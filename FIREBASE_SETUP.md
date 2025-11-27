# Firebase Setup Instructions

Follow these steps to set up Firebase for TraderQ migration.

## Step 1: Create Firebase Project

1. Go to [Firebase Console](https://console.firebase.google.com/)
2. Click "Add project" or "Create a project"
3. Enter project name: `traderq` (or your preferred name)
4. Disable Google Analytics (optional, not needed for this project)
5. Click "Create project"

## Step 2: Enable Firestore Database

1. In Firebase Console, click "Firestore Database" in left sidebar
2. Click "Create database"
3. Select "Start in **production mode**" (we have security rules ready)
4. Choose location: `us-central` (or closest to your users)
5. Click "Enable"

## Step 3: Enable Firebase Authentication

1. Click "Authentication" in left sidebar
2. Click "Get started"
3. Click "Email/Password" provider
4. Enable "Email/Password" toggle
5. Click "Save"

## Step 4: Create Service Account Key

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Select your Firebase project from dropdown
3. Go to "IAM & Admin" → "Service Accounts"
4. Find the "Firebase Admin SDK" service account
5. Click the three dots (⋮) → "Manage keys"
6. Click "Add Key" → "Create new key"
7. Choose "JSON" format
8. Click "Create" - this downloads `serviceAccountKey.json`
9. **Move this file to your traderq project root directory**
10. **IMPORTANT**: Add to `.gitignore` (already done)

## Step 5: Install Firebase CLI

### Windows (using npm):
```bash
npm install -g firebase-tools
```

### Or using standalone installer:
Download from: https://firebase.google.com/docs/cli#windows-standalone-binary

### Verify installation:
```bash
firebase --version
```

## Step 6: Initialize Firebase in Project

1. Open terminal in your `traderq` directory
2. Login to Firebase:
```bash
firebase login
```

3. Initialize Firebase:
```bash
firebase init
```

4. Select these features (use spacebar to select):
   - [x] Firestore
   - [x] Hosting

5. Use existing project and select your `traderq` project

6. For Firestore:
   - Rules file: `firestore.rules` (already exists)
   - Indexes file: `firestore.indexes.json` (already exists)

7. For Hosting:
   - Public directory: `public` (create empty folder if needed)
   - Configure as single-page app: No
   - Set up automatic builds: No

## Step 7: Deploy Firestore Rules

```bash
firebase deploy --only firestore:rules
firebase deploy --only firestore:indexes
```

## Step 8: Configure Environment Variables

1. Copy `.env.example` to `.env`:
```bash
copy .env.example .env
```

2. Edit `.env` and update:
```bash
FIREBASE_PROJECT_ID=your-actual-project-id
```

3. Verify `serviceAccountKey.json` is in the root directory

## Step 9: Verify Setup

Run this test script to verify Firebase connection:

```python
# test_firebase.py
import firebase_admin
from firebase_admin import credentials, firestore
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize Firebase
cred = credentials.Certificate('serviceAccountKey.json')
firebase_admin.initialize_app(cred)

# Test Firestore connection
db = firestore.client()
print("✅ Firebase connection successful!")

# Test write
test_ref = db.collection('test').document('connection_test')
test_ref.set({'status': 'working', 'timestamp': firestore.SERVER_TIMESTAMP})
print("✅ Firestore write successful!")

# Test read
doc = test_ref.get()
if doc.exists:
    print(f"✅ Firestore read successful! Data: {doc.to_dict()}")

# Cleanup
test_ref.delete()
print("✅ Test complete!")
```

Run it:
```bash
python test_firebase.py
```

## Step 10: Enable Cloud Run API (for deployment)

1. Go to [Cloud Run Console](https://console.cloud.google.com/run)
2. Click "Enable API" if prompted
3. This may take a few minutes

## Troubleshooting

### Error: "Permission denied"
- Make sure you've deployed security rules: `firebase deploy --only firestore:rules`
- Verify service account has proper permissions

### Error: "Module not found"
- Install Firebase dependencies: `pip install -r requirements.txt`

### Error: "Project not found"
- Double-check `FIREBASE_PROJECT_ID` in `.env` matches your Firebase Console project ID

## Next Steps

After completing this setup:
1. Proceed to Week 1, Day 3-5: Modular Refactoring
2. The Firebase infrastructure is now ready for integration!

## Important Security Notes

- **NEVER commit `serviceAccountKey.json` to Git**
- **NEVER commit `.env` to Git**
- Both are already in `.gitignore`
- Store backups of these files securely (password manager, encrypted drive)
