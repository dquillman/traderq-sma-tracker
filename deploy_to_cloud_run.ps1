# Deploy to Cloud Run using Source-based deployment (Buildpacks)
Write-Host "Deploying to Cloud Run..."
gcloud run deploy traderq-app --source . --region us-central1 --allow-unauthenticated --project trader-q

# Deploy Firebase Hosting
Write-Host "Deploying Firebase Hosting..."
firebase deploy --only hosting --project trader-q

Write-Host "Deployment Complete!"
Write-Host "App URL: https://trader-q.web.app"
