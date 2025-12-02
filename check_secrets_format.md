# Firebase Secrets Format Check

When you paste secrets into Streamlit Cloud, make sure:

1. **The format is EXACTLY:**
```toml
[firebase]
type = "service_account"
project_id = "trader-q"
private_key_id = "881e7b06c6411876f93cd3dbd10b6973bb027336"
private_key = "-----BEGIN PRIVATE KEY-----\nMIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQC+50Ru3NaUTapI\no4QOduvoH0NzXNDMOSDS8uOU6ex8y/+D1R/TpW1sqa5EPs5Q3JRXcU9ULG3D0xDy\n9mkGq4MDEnmkPeb0hNi9loIv2QrxpcPkP+0oVo8YfPWF3JviAhaYAuuu/harERVM\n78VZLpyY0dcmnVutYA+X9W7lp9qsBfVz1YrLzC0gSxF9p+/c7cRzCQazo3dpxLUD\nERirXL2TwudY0AUJeIXbZ0pch3h3aqoxZbEuBLyfaVR7MMOzyRzedO0FFhpP/RDO\n2VZOUDfobmfliJPiwHDONn58qm6NBs9BkXQq6vDlCYWppz4nOTez7LkskrbO/gZl\nrLOl4WBfAgMBAAECggEAH/q43YN0FLrf7DQsIoosYaGfGHuzZIqrY3sZwa/gFFXO\nE2UH5FoBdyMDlN4ajLQuy2kpW1Xn+1Znr/O0S1A+7axSIT/XaR25+yRz4ZUPvsRA\niQNmdIvvg7AnJwn6OQWViFhw58sbvANsGIvP+O0UgYshahARM/10D0YnkiQovwkk\nAxPzLf0P++1sbIQgPCQKhIwzg9Bgu4WqfuAgj14+x0Ub55eBQglSqMab9avxSFSv\nrv4x8FyWSwmPSGenE/oEkCaPzA3D4OHmCDsvoHyYZx967SehMKqEGGakWcGSwJYo\n5MqLa7AV8+wZ/QkQXEYVia8LFGxaFgi3SDcQXTunTQKBgQD2I8FUNnQ3a76TeddS\nYn2yMTJ1p4wc+24igeaIawRlLDvOTOtINj0uti2bQqzvFWQ/tCQt0rDTiwZgUyDT\nlWLrzFMh258XnHa6Gqb96kdSOK0bfHjvj1whgY03eSe9AbT2iTgTY70LA4lHx49o\nDluEO0UBi3s5Z0YT8rM4UxgX2wKBgQDGjQupOrAjerW78CUOePlUsF+bV4dHr3fz\nt7R1MMjgRKVrUgaBiZFAjDRQtq7lw38NidDOQPo8f3qyKqD3cAvGJwixUpXTyAnV\nJPV55SeZLJGrcOUN/Fn5QcEgS8mRvvjIYwRpsR655ASfL/Cm/3lDCS4bMALgUvVN\nQ4x3gQCyzQKBgQDscEhEBtL4cc1tiPrnrqijxVJ9ZmbXaEbRawryPCrKrQT+FTFl\n+oQnHOUOYawRNfFIqFigk+U8MC391ZyQ4s1VSL8KpRdb7Pa3quaCvsuoFb1jy9u3\n83RC01m90en0S3Fz6Tgul/5+V/VFOFNvV2tdyDlvVHcYGzZb7yVtk5RvRQKBgQCB\nRFsPNrJupvmi/lph7ckGpj3YuUfOGCOUfUnz8msV+Btqn+C0fYgf6ig1VHrSBFG3\n0r4rSoqg0K5lSPO7pStFOyyhpg797wLXzlQzpEn/o9DDOaEnVeCOM401JaJ6TUdT\nz4OT/Ejw5c9MhL29PB8K0fM+qCd3PQuP3iaZt1dpEQKBgBq81ygfy2B3/SHNqJ64\n+DTVRORmPnaEoTNDcvG0DfvusQS0z0YGX5b6WiWmKAX1xn+ANieFujTPcXq8FocK\n0KjBFVbpuSAVrOp6En8rlcyzc/rrbAQkNIRSp60YkXwYUYe0culs4AW5RiOA2o/N\noJJqOEg2A5drpZAQKxF8xYAi\n-----END PRIVATE KEY-----\n"
client_email = "firebase-adminsdk-fbsvc@trader-q.iam.gserviceaccount.com"
client_id = "103427870400898512129"
auth_uri = "https://accounts.google.com/o/oauth2/auth"
token_uri = "https://oauth2.googleapis.com/token"
auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
client_x509_cert_url = "https://www.googleapis.com/robot/v1/metadata/x509/firebase-adminsdk-fbsvc%40trader-q.iam.gserviceaccount.com"
```

2. **Common mistakes:**
   - ❌ Missing `[firebase]` header
   - ❌ Extra quotes around values that already have quotes
   - ❌ Missing newlines (the `\n` in private_key)
   - ❌ Copying only part of the file

3. **How to verify:**
   - After pasting, the Secrets box should show the `[firebase]` section
   - All fields should be visible (scroll down to see all)
   - Click "Save" and wait for confirmation

4. **After saving:**
   - Wait 2-3 minutes for redeployment
   - Check the Streamlit Cloud logs for any errors
   - The app should now show either the login page OR a clear error message

