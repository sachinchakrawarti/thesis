# scripts/run_backend.ps1
# Starts the FastAPI backend on http://127.0.0.1:8000

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path "$PSScriptRoot\..").Path
Set-Location $ProjectRoot

Write-Host "Installing backend dependencies (first run only)..." -ForegroundColor Cyan
pip install -r backend\requirements.txt

Write-Host ""
Write-Host "Starting FastAPI on http://127.0.0.1:8000 ..." -ForegroundColor Green
uvicorn backend.api:app --reload --port 8000
