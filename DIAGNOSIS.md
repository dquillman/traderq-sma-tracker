# Troubleshooting: App Still Crashing After Secrets Configuration

If you've copied Firebase secrets 3+ times and the app is still failing the health check, here's what to check:

## 1. Verify Secrets Were Actually Saved

Go to: https://share.streamlit.io → Your App → Settings → Secrets

**Check:**
- Do you see a `[firebase]` section?
- Are ALL fields present? (type, project_id, private_key, client_email, etc.)
- Did you click "Save" and see a confirmation?

## 2. Check Streamlit Cloud Logs

The logs will show the ACTUAL error. To see them:

1. Go to: https://share.streamlit.io
2. Click your app: `traderq-sma-tracker`
3. Look at the **Logs** section on the right
4. Scroll to find error messages (usually in red or with ❗)

**Look for:**
- `FileNotFoundError` = Secrets not being read
- `ValueError` = Missing required fields
- `RuntimeError` = Firebase Admin SDK initialization failed
- Any Python traceback

## 3. Common Issues

### Issue A: Secrets Format Wrong
- The `private_key` must have `\n` (not actual newlines)
- All strings must be in quotes
- The `[firebase]` header must be present

### Issue B: Firebase Project Not Set Up
- Check Firebase Console: https://console.firebase.google.com/
- Is your project active?
- Is Firestore enabled?
- Are the service account permissions correct?

### Issue C: Network/Timeout
- Firebase Admin SDK might be timing out
- Check if your Firebase project allows external connections

## 4. Next Steps

**After checking the logs, share the error message** and I can help fix it.

The latest code changes should now show error messages in the app itself (if it gets past the health check), but if the health check is failing, we need to see the logs.

