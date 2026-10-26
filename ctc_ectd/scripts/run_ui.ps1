# run_ui.ps1 - install deps (once) + start FastAPI backend + static UI.

$ErrorActionPreference = "Stop"
$Root = "D:\Projects\done\thesis\ctc_ectd"

Write-Host ""
Write-Host "CTI Threat Analysis - UI launcher"
Write-Host "Project root: $Root"
Write-Host ""

Set-Location $Root

Write-Host "Installing backend dependencies..."
pip install -r backend\requirements.txt | Out-Null

if (-not (Test-Path "artifacts\cti_text_classifier.joblib")) {
    Write-Host ""
    Write-Host "WARNING: model not found at artifacts\cti_text_classifier.joblib" -ForegroundColor Yellow
    Write-Host "Run notebooks 01 -> 03 first, then restart this launcher." -ForegroundColor Yellow
    Write-Host ""
}

Write-Host "Starting FastAPI on http://127.0.0.1:8000"
Write-Host "Open your browser at: http://127.0.0.1:8000"
Write-Host "Press Ctrl+C to stop."
Write-Host ""

uvicorn backend.api:app --reload --port 8000
