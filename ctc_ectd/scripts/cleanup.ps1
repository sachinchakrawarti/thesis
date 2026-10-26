# cleanup.ps1
# Removes unwanted files/folders from ctc_ectd, after backing them up.
#
# Run:
#     cd D:\Projects\done\thesis\ctc_ectd
#     powershell -ExecutionPolicy Bypass -File scripts\cleanup.ps1

$ErrorActionPreference = "Stop"

$Root     = "D:\Projects\done\thesis\ctc_ectd"
$Backup   = Join-Path $Root ("_backup_" + (Get-Date -Format "yyyyMMdd_HHmmss"))

# ---------------------------------------------------------------------------
# Items to REMOVE (relative to $Root)
# ---------------------------------------------------------------------------
$ToRemove = @(
    # --- whole folders ---
    "backend",
    "docs",
    "understand",
    "scripts\__pycache__",
    "scripts\archive",

    # --- notebooks ---
    "notebooks\cti_threat_analysis.ipynb",
    "datasetanaysis.ipynb",
    "test.ipynb",

    # --- artifacts ---
    "artifacts\final",
    "artifacts\README_Hindi.md",
    "artifacts\README.md",

    # --- scripts ---
    "scripts\analyze_uploads.py",
    "scripts\phase_1.py",
    "scripts\phase_2.py",
    "scripts\phase_3.py",
    "scripts\phase_4.py",
    "scripts\phase_5.py"
)

# ---------------------------------------------------------------------------
# Items to KEEP (for the safety check at the end)
# ---------------------------------------------------------------------------
$MustKeep = @(
    "notebooks\01_data_extraction_and_cleaning.ipynb",
    "notebooks\02_exploratory_data_analysis.ipynb",
    "notebooks\03_model_training_and_evaluation.ipynb",
    "notebooks\04_document_classification_and_prediction.ipynb",
    "notebooks\05_temporal_topic_modeling_and_emerging_threats.ipynb",
    "scripts\1notebook.py",
    "scripts\document_extractor.py",
    "scripts\generate_sample_pdfs.py",
    "data\processed\cleaned_reports.pkl",
    "data\processed\uploaded_docs.pkl",
    "dataset\synthetic_cti_reports.csv",
    "artifacts\classified_documents.csv",
    "artifacts\emerging_threat_alerts.csv",
    "artifacts\model_results.csv",
    "artifacts\monthly_topic_shares.csv",
    "artifacts\cti_text_classifier.joblib",
    "ui\index.html",
    "ui\app.js",
    "ui\config.js",
    "ui\styles.css",
    "README.md",
    "requirements.txt"
)

# ---------------------------------------------------------------------------
# Execute
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host "Project root : $Root"
Write-Host "Backup dir   : $Backup"
Write-Host ""

if (-not (Test-Path $Root)) {
    throw "Project root not found: $Root"
}

New-Item -ItemType Directory -Path $Backup | Out-Null
$moved = 0

foreach ($rel in $ToRemove) {
    $src = Join-Path $Root $rel
    if (-not (Test-Path $src)) {
        Write-Host "SKIP (not found): $rel"
        continue
    }

    $dst = Join-Path $Backup $rel
    $dstParent = Split-Path $dst -Parent
    if (-not (Test-Path $dstParent)) {
        New-Item -ItemType Directory -Path $dstParent -Force | Out-Null
    }

    try {
        Move-Item -LiteralPath $src -Destination $dst -Force
        Write-Host "MOVED: $rel"
        $moved++
    } catch {
        Write-Host "ERROR moving ${rel}: $($_.Exception.Message)"
    }
}

# Remove now-empty folders
foreach ($empty in @("backend", "docs", "understand", "scripts\archive", "artifacts\final")) {
    $p = Join-Path $Root $empty
    if ((Test-Path $p) -and -not (Get-ChildItem $p -Force)) {
        Remove-Item -LiteralPath $p -Force
        Write-Host "REMOVED EMPTY DIR: $empty"
    }
}

# ---------------------------------------------------------------------------
# Safety check — confirm required files still exist
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host "Verifying required files are still present..."
$missing = @()
foreach ($keep in $MustKeep) {
    $p = Join-Path $Root $keep
    if (-not (Test-Path $p)) {
        $missing += $keep
    }
}

if ($missing.Count -gt 0) {
    Write-Host ""
    Write-Host "MISSING REQUIRED FILES:" -ForegroundColor Red
    $missing | ForEach-Object { Write-Host "  - $_" -ForegroundColor Red }
    Write-Host ""
    Write-Host "Restore from backup if needed: $Backup"
} else {
    Write-Host "All required files are present." -ForegroundColor Green
}

Write-Host ""
Write-Host "Done. Moved $moved item(s) to $Backup"
Write-Host "If everything looks good, delete the backup manually:"
Write-Host "    Remove-Item -Recurse -Force '$Backup'"