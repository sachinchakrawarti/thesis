# scripts/run_ui.ps1
# Serves the static UI on http://127.0.0.1:5500

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path "$PSScriptRoot\..").Path
Set-Location $ProjectRoot

$Port = 5500
Write-Host "Serving ui/ on http://127.0.0.1:$Port ..." -ForegroundColor Green
python -m http.server $Port --directory ui
