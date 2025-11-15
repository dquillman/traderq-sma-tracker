# Save this as: G:\Users\daveq\MSS\Deploy-MSS.ps1

#Requires -Version 5.1

<#
.SYNOPSIS
    MSS Deployment Automation Script
.DESCRIPTION
    Automates deployment tasks for the MSS application to Google Cloud Run
#>

[CmdletBinding()]
param(
    [switch]$All,
    [switch]$Validate,
    [switch]$Build,
    [switch]$Deploy,
    [switch]$Test
)

$ErrorActionPreference = "Stop"
$PROJECT_ROOT = "G:\Users\daveq\MSS"

# ============================================
# FUNCTION DEFINITIONS
# ============================================

function Write-Success { 
    param($Message) 
    Write-Host "[SUCCESS] $Message" -ForegroundColor Green 
}

function Write-Info { 
    param($Message) 
    Write-Host "[INFO] $Message" -ForegroundColor Cyan 
}

function Write-Warn { 
    param($Message) 
    Write-Host "[WARNING] $Message" -ForegroundColor Yellow 
}

function Write-Err { 
    param($Message) 
    Write-Host "[ERROR] $Message" -ForegroundColor Red 
}

# Validation Function
function Test-Environment {
    Write-Info "Validating environment..."
    
    # Check Docker
    try {
        $dockerVersion = docker --version
        Write-Success "Docker found: $dockerVersion"
    } catch {
        Write-Err "Docker not found. Please install Docker Desktop."
        return $false
    }
    
    # Check gcloud
    try {
        $gcloudVersion = gcloud --version | Select-Object -First 1
        Write-Success "gcloud found: $gcloudVersion"
    } catch {
        Write-Warn "gcloud CLI not found. Install from: https://cloud.google.com/sdk/docs/install"
    }
    
    # Check .env file
    if (Test-Path "$PROJECT_ROOT\.env") {
        Write-Success ".env file found"
    } else {
        Write-Warn ".env file not found. Copy from .env.example"
    }
    
    # Check required files (with correct paths)
    $requiredFiles = @(
        "Dockerfile.app",
        "entrypoint-app.sh",
        "web\api_server.py",
        "requirements.txt"
    )
    
    foreach ($file in $requiredFiles) {
        if (Test-Path "$PROJECT_ROOT\$file") {
            Write-Success "$file exists"
        } else {
            Write-Err "$file missing!"
            return $false
        }
    }
    
    return $true
}

# Build Function
function Build-DockerImage {
    Write-Info "Building Docker image..."
    
    Set-Location $PROJECT_ROOT
    
    $imageName = "mss-app"
    $imageTag = "latest"
    
    try {
        docker build -f Dockerfile.app -t "${imageName}:${imageTag}" .
        Write-Success "Docker image built successfully: ${imageName}:${imageTag}"
        return $true
    } catch {
        Write-Err "Docker build failed: $_"
        return $false
    }
}

# Test Function
function Test-DockerImage {
    Write-Info "Testing Docker image locally..."
    
    $containerName = "mss-test"
    
    # Stop existing container
    docker stop $containerName 2>$null
    docker rm $containerName 2>$null
    
    try {
        # Run container
        docker run -d `
            --name $containerName `
            -p 8080:8080 `
            --env-file .env `
            mss-app:latest
        
        Write-Success "Container started: $containerName"
        Write-Info "Waiting for app to start..."
        Start-Sleep -Seconds 5
        
        # Test health endpoint
        try {
            $response = Invoke-WebRequest -Uri "http://localhost:8080/health" -UseBasicParsing
            if ($response.StatusCode -eq 200) {
                Write-Success "Health check passed!"
                Write-Host $response.Content
            }
        } catch {
            Write-Warn "Health check failed. Check logs with: docker logs $containerName"
        }
        
        Write-Info "`nContainer is running. Test at: http://localhost:8080"
        Write-Info "View logs: docker logs -f $containerName"
        Write-Info "Stop container: docker stop $containerName"
        
        return $true
    } catch {
        Write-Err "Container test failed: $_"
        return $false
    }
}

# Deploy Function
function Deploy-ToCloudRun {
    Write-Info "Deploying to Google Cloud Run..."
    
    # Check if gcloud is authenticated
    try {
        $account = gcloud auth list --filter=status:ACTIVE --format="value(account)" 2>$null
        if (-not $account) {
            Write-Warn "Not authenticated with gcloud. Running: gcloud auth login"
            gcloud auth login
        } else {
            Write-Success "Authenticated as: $account"
        }
    } catch {
        Write-Err "gcloud authentication failed"
        return $false
    }
    
    # Load project ID from .env
    if (Test-Path "$PROJECT_ROOT\.env") {
        $envContent = Get-Content "$PROJECT_ROOT\.env"
        $projectId = ($envContent | Where-Object { $_ -match "^GCP_PROJECT_ID=" }) -replace "GCP_PROJECT_ID=", ""
        
        if ($projectId) {
            Write-Success "Project ID: $projectId"
            gcloud config set project $projectId
        } else {
            Write-Err "GCP_PROJECT_ID not found in .env file"
            return $false
        }
    }
    
    Write-Info "Use GitHub Actions for deployment or run manual gcloud commands"
    Write-Info "Commit and push changes to trigger GitHub Actions workflow"
    
    return $true
}

# ============================================
# MAIN EXECUTION
# ============================================

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "   MSS Deployment Automation" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

try {
    Set-Location $PROJECT_ROOT
    
    if ($All) {
        $Validate = $true
        $Build = $true
        $Test = $true
        $Deploy = $true
    }
    
    if (-not ($Validate -or $Build -or $Test -or $Deploy)) {
        Write-Host "Usage: .\Deploy-MSS.ps1 [-All] [-Validate] [-Build] [-Test] [-Deploy]"
        Write-Host "`nExamples:"
        Write-Host "  .\Deploy-MSS.ps1 -All          # Run all steps"
        Write-Host "  .\Deploy-MSS.ps1 -Validate     # Only validate environment"
        Write-Host "  .\Deploy-MSS.ps1 -Build -Test  # Build and test locally"
        exit 0
    }
    
    if ($Validate) {
        if (-not (Test-Environment)) {
            Write-Err "Environment validation failed!"
            exit 1
        }
    }
    
    if ($Build) {
        if (-not (Build-DockerImage)) {
            Write-Err "Docker build failed!"
            exit 1
        }
    }
    
    if ($Test) {
        if (-not (Test-DockerImage)) {
            Write-Err "Docker test failed!"
            exit 1
        }
    }
    
    if ($Deploy) {
        if (-not (Deploy-ToCloudRun)) {
            Write-Err "Deployment preparation failed!"
            exit 1
        }
    }
    
    Write-Host "`n========================================" -ForegroundColor Green
    Write-Host "   Deployment Script Completed!" -ForegroundColor Green
    Write-Host "========================================`n" -ForegroundColor Green
    
} catch {
    Write-Err "Script failed: $_"
    exit 1
}