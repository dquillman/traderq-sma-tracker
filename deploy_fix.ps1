# Deploy Fix Script
Write-Host "Deploying TraderQ Streamlit App to Cloud Run..." -ForegroundColor Cyan

# Deploy to Cloud Run using source (builds on GCP)
# Service name matches firebase.json: "traderq-streamlit"
gcloud run deploy traderq-streamlit `
    --source . `
    --region us-central1 `
    --allow-unauthenticated `
    --project trader-q `
    --quiet

if ($LASTEXITCODE -eq 0) {
    Write-Host "Cloud Run Deployment Successful!" -ForegroundColor Green
    
    Write-Host "Deploying Firebase Hosting..." -ForegroundColor Cyan
    firebase deploy --only hosting --project trader-q
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Firebase Hosting Deployment Successful!" -ForegroundColor Green
        Write-Host "App should be live at: https://trader-q.web.app" -ForegroundColor Green
    } else {
        Write-Host "Firebase Hosting Deployment Failed!" -ForegroundColor Red
    }
} else {
    Write-Host "Cloud Run Deployment Failed!" -ForegroundColor Red
}
