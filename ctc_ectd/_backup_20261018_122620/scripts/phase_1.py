# scripts/phase_1.py
"""
PHASE 1 — Cleanup & Consolidation
=================================

Does everything Phase 1 requires:

  1. Deletes dead weight (backups, __pycache__, empty folders)
  2. Archives scaffolding scripts into scripts/archive/
  3. Prunes redundant generated PDFs (keeps a clean cluster corpus)
  4. Creates/updates .gitignore
  5. Writes a project README.md
  6. Reports final structure

Idempotent: safe to run multiple times.

Run:
    python scripts/phase_1.py

Dry-run (shows what would happen, changes nothing):
    python scripts/phase_1.py --dry-run
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ------------------------------------------------------------------
# What to delete (folder names anywhere in the tree)
# ------------------------------------------------------------------
DELETE_DIRS = [
    "brackup",
    "v",
    "uploads_backup",
    "__pycache__",
    ".ipynb_checkpoints",
    "backup",
    "bkp",
]

# ------------------------------------------------------------------
# What to move into scripts/archive/
# ------------------------------------------------------------------
ARCHIVE_SCRIPTS = [
    "fix_notebook.py",
    "fix_alerts.py",
    "setup_ui.py",
    "setup_ui.ps1",
    "generate_wave_reports.py",
    "generate_wave_reports_v2.py",
    "generate_realistic_mix.py",
    "run_backend.ps1",
    "run_ui.ps1",
]

# ------------------------------------------------------------------
# Scripts to keep in scripts/
# ------------------------------------------------------------------
KEEP_SCRIPTS = {
    "document_extractor.py",
    "analyze_uploads.py",
    "generate_sample_pdfs.py",
    "phase_1.py",
    "README.md",
}

# ------------------------------------------------------------------
# PDF retention rules: keep only these filenames
# ------------------------------------------------------------------
KEEP_PDF_PATTERNS = [
    "2024-01_phishing_report_01.pdf",
    "2024-03_ransomware_report_02.pdf",
    "2024-05_credential_theft_report_03.pdf",
    "2024-07_ddos_report_04.pdf",
    "2024-09_web_attack_report_05.pdf",
    "2024-11_data_breach_report_06.pdf",
    "2025-01_insider_threat_report_07.pdf",
    "2025-03_botnet_report_08.pdf",
    "2025-05_vulnerability_report_09.pdf",
    "2025-07_credential_theft_report_10.pdf",
]


# ==================================================================
# Helpers
# ==================================================================
class Reporter:
    def __init__(self, dry: bool = False):
        self.dry = dry
        self.deleted = 0
        self.archived = 0
        self.created = 0

    def banner(self, text: str) -> None:
        print()
        print("=" * 62)
        print(text)
        print("=" * 62)

    def ok(self, msg: str) -> None:
        prefix = "[DRY] " if self.dry else "  +   "
        print(f"{prefix}{msg}")

    def warn(self, msg: str) -> None:
        print(f"  !  {msg}")

    def skip(self, msg: str) -> None:
        print(f"  .  {msg}")


# ==================================================================
# Step 1 — Delete junk folders
# ==================================================================
def delete_junk(rep: Reporter) -> None:
    rep.banner("[1/6] Deleting junk folders")

    found = []
    for name in DELETE_DIRS:
        for path in PROJECT_ROOT.rglob(name):
            if not path.is_dir():
                continue
            if ".git" in path.parts:
                continue
            found.append(path)

    if not found:
        rep.skip("nothing to delete")
        return

    for path in found:
        rel = path.relative_to(PROJECT_ROOT)
        if rep.dry:
            rep.ok(f"would delete {rel}/")
        else:
            try:
                shutil.rmtree(path)
                rep.ok(f"deleted {rel}/")
                rep.deleted += 1
            except Exception as exc:
                rep.warn(f"could not delete {rel}/ -- {exc}")


# ==================================================================
# Step 2 — Archive scaffolding scripts
# ==================================================================
def archive_scaffolding(rep: Reporter) -> None:
    rep.banner("[2/6] Archiving scaffolding scripts")

    scripts_dir = PROJECT_ROOT / "scripts"
    archive_dir = scripts_dir / "archive"

    if not scripts_dir.exists():
        rep.warn("scripts/ folder missing")
        return

    if not rep.dry:
        archive_dir.mkdir(parents=True, exist_ok=True)

    moved_any = False

    # 2a — explicit archive list
    for name in ARCHIVE_SCRIPTS:
        src = scripts_dir / name
        dst = archive_dir / name
        if not src.exists():
            continue
        if dst.exists():
            rep.skip(f"{name} already archived")
            continue
        if rep.dry:
            rep.ok(f"would move scripts/{name} -> scripts/archive/")
        else:
            shutil.move(str(src), str(dst))
            rep.ok(f"moved scripts/{name} -> scripts/archive/")
            rep.archived += 1
        moved_any = True

    # 2b — anything else not in KEEP_SCRIPTS
    for item in scripts_dir.iterdir():
        if item.is_dir():
            continue
        if item.name in KEEP_SCRIPTS:
            continue
        if item.name.startswith("."):
            continue
        dst = archive_dir / item.name
        if dst.exists():
            continue
        if rep.dry:
            rep.ok(f"would archive scripts/{item.name}")
        else:
            shutil.move(str(item), str(dst))
            rep.ok(f"archived scripts/{item.name}")
            rep.archived += 1
        moved_any = True

    if not moved_any:
        rep.skip("no scripts to archive")


# ==================================================================
# Step 3 — Prune redundant PDFs
# ==================================================================
def prune_pdfs(rep: Reporter) -> None:
    rep.banner("[3/6] Pruning redundant PDFs")

    uploads = PROJECT_ROOT / "uploads"
    if not uploads.exists():
        rep.skip("uploads/ folder does not exist")
        return

    removed = 0
    for pdf in sorted(uploads.glob("*.pdf")):
        if pdf.name in KEEP_PDF_PATTERNS:
            continue
        if rep.dry:
            rep.ok(f"would delete uploads/{pdf.name}")
        else:
            try:
                pdf.unlink()
                removed += 1
            except Exception as exc:
                rep.warn(f"could not delete {pdf.name} -- {exc}")

    if removed:
        rep.ok(f"removed {removed} redundant PDF(s)")
    else:
        rep.skip("uploads/ already clean")


# ==================================================================
# Step 4 — .gitignore (line list, not triple-quoted block)
# ==================================================================
GITIGNORE_LINES = [
    "# Python",
    "__pycache__/",
    "*.py[cod]",
    "*.egg-info/",
    ".Python",
    "build/",
    "dist/",
    ".venv/",
    "venv/",
    "env/",
    "",
    "# Jupyter",
    ".ipynb_checkpoints/",
    "*.ipynb_checkpoints",
    "",
    "# OS",
    ".DS_Store",
    "Thumbs.db",
    "desktop.ini",
    "",
    "# Editors",
    ".vscode/",
    ".idea/",
    "*.swp",
    "*.swo",
    "*~",
    "",
    "# Project artifacts (large, regenerable)",
    "artifacts/*.joblib",
    "artifacts/*.pkl",
    "artifacts/final/*.png",
    "artifacts/final/*.pdf",
    "",
    "# Data (large)",
    "uploads/*.pdf",
    "!uploads/.gitkeep",
    "",
    "# Backups",
    "*.bak",
    "*.backup",
    "*_backup/",
    "backup/",
    "brackup/",
    "",
    "# Logs",
    "*.log",
    "logs/",
    "",
    "# Secrets",
    ".env",
    ".env.local",
    "secrets.*",
]


def build_gitignore() -> str:
    return "\n".join(GITIGNORE_LINES) + "\n"


def write_gitignore(rep: Reporter) -> None:
    rep.banner("[4/6] Writing .gitignore")

    content = build_gitignore()
    path = PROJECT_ROOT / ".gitignore"

    if path.exists():
        if path.read_text(encoding="utf-8") == content:
            rep.skip(".gitignore already up to date")
            return
        if rep.dry:
            rep.ok("would update .gitignore")
            return
        path.write_text(content, encoding="utf-8")
        rep.ok("updated .gitignore")
        rep.created += 1
    else:
        if rep.dry:
            rep.ok("would create .gitignore")
            return
        path.write_text(content, encoding="utf-8")
        rep.ok("created .gitignore")
        rep.created += 1


# ==================================================================
# Step 5 — README (line list, safe)
# ==================================================================
README_LINES = [
    "# CTI Threat Analysis",
    "",
    "End-to-end prototype for Cyber-Threat-Intelligence report triage:",
    "",
    "- **Text classification** -- TF-IDF + Logistic Regression / Naive Bayes / Linear SVM",
    "- **Temporal topic modeling** -- NMF over monthly report text",
    "- **Emerging-threat detection** -- heuristic threshold on topic-share drift",
    "- **Document ingestion** -- PDF / DOCX / TXT / HTML / MD extraction",
    "- **Optional web UI** -- FastAPI backend + static HTML/JS frontend (demo only)",
    "",
    "## Project structure",
    "",
    "```",
    "ctc_ectd/",
    "|-- artifacts/              # trained models, exported CSVs, final plots",
    "|-- backend/                # FastAPI backend (demo only)",
    "|-- dataset/",
    "|   `-- synthetic_cti_reports.csv",
    "|-- notebooks/",
    "|   `-- cti_threat_analysis.ipynb",
    "|-- scripts/",
    "|   |-- document_extractor.py",
    "|   |-- analyze_uploads.py",
    "|   |-- generate_sample_pdfs.py",
    "|   |-- phase_1.py",
    "|   `-- archive/            # scaffolding, one-off fixers",
    "|-- ui/                     # static frontend (demo only)",
    "|-- uploads/                # drop CTI documents here",
    "|-- README.md",
    "`-- requirements.txt",
    "```",
    "",
    "## Install",
    "",
    "```powershell",
    "pip install -r requirements.txt",
    "pip install pypdf python-docx beautifulsoup4 lxml reportlab",
    "```",
    "",
    "## Run the pipeline (CLI)",
    "",
    "```powershell",
    "python scripts/analyze_uploads.py",
    "```",
    "",
    "Reads every supported file in `uploads/`, extracts text, classifies each",
    "document, runs temporal topic modeling on dated documents, and writes:",
    "",
    "- `artifacts/uploaded_classifications.csv`",
    "- `artifacts/emerging_threat_alerts.csv`",
    "",
    "## Regenerate demo PDFs",
    "",
    "```powershell",
    "python scripts/generate_sample_pdfs.py",
    "```",
    "",
    "## Notebook",
    "",
    "```powershell",
    "jupyter notebook notebooks/cti_threat_analysis.ipynb",
    "```",
    "",
    "Run all cells top-to-bottom on a fresh kernel.",
    "",
    "## Optional web UI (demo)",
    "",
    "```powershell",
    "# Terminal 1",
    "python -m uvicorn backend.api:app --reload --port 8000",
    "",
    "# Terminal 2",
    "python -m http.server 5500 --directory ui",
    "",
    "# Browser",
    "# http://127.0.0.1:5500",
    "```",
    "",
    "The UI is a demonstration layer only and is not part of the thesis",
    "evaluation.",
    "",
    "## Reproducibility",
    "",
    "- Fixed random seed: `42`",
    "- Dataset: `dataset/synthetic_cti_reports.csv` (10,000 rows, 17 columns)",
    "- All plots and CSVs referenced in the thesis live in `artifacts/final/`",
    "",
    "## Limitations",
    "",
    "- The dataset is **synthetic**. Category words may leak into report text.",
    "- High classification scores reflect dataset construction, not real-world capability.",
    "- Topic models require human interpretation.",
    "- Emerging-threat alerts are heuristic and need analyst review.",
    "- Real-world generalization requires evaluation on independently collected CTI data.",
    "",
    "## License",
    "",
    "Academic use only -- MSc thesis project.",
]


def build_readme() -> str:
    return "\n".join(README_LINES) + "\n"


def write_readme(rep: Reporter) -> None:
    rep.banner("[5/6] Writing README.md")

    content = build_readme()
    path = PROJECT_ROOT / "README.md"

    if path.exists():
        if path.read_text(encoding="utf-8") == content:
            rep.skip("README.md already up to date")
            return
        if rep.dry:
            rep.ok("would update README.md")
            return
        path.write_text(content, encoding="utf-8")
        rep.ok("updated README.md")
        rep.created += 1
    else:
        if rep.dry:
            rep.ok("would create README.md")
            return
        path.write_text(content, encoding="utf-8")
        rep.ok("created README.md")
        rep.created += 1


# ==================================================================
# Step 6 — Report final structure
# ==================================================================
def report_structure(rep: Reporter) -> None:
    rep.banner("[6/6] Final structure")

    for root, dirs, files in os.walk(PROJECT_ROOT):
        dirs[:] = [d for d in dirs
                   if d not in {".git", "__pycache__", ".ipynb_checkpoints"}
                   and not d.startswith(".")]

        root_path = Path(root)
        rel = root_path.relative_to(PROJECT_ROOT)
        depth = len(rel.parts)
        indent = "    " * depth

        name = root_path.name or str(root_path)
        print(f"{indent}{name}/")

        for f in sorted(files):
            if f.startswith("."):
                continue
            print(f"{indent}    {f}")


# ==================================================================
# Main
# ==================================================================
def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 1 cleanup")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would happen, change nothing")
    args = parser.parse_args()

    rep = Reporter(dry=args.dry_run)

    print("=" * 62)
    print("PHASE 1 -- Cleanup & Consolidation")
    print("=" * 62)
    print(f"Project root: {PROJECT_ROOT}")
    if args.dry_run:
        print("MODE: DRY-RUN (no changes will be made)")

    delete_junk(rep)
    archive_scaffolding(rep)
    prune_pdfs(rep)
    write_gitignore(rep)
    write_readme(rep)
    report_structure(rep)

    print()
    print("=" * 62)
    print("PHASE 1 SUMMARY")
    print("=" * 62)
    print(f"  Deleted folders      : {rep.deleted}")
    print(f"  Archived files       : {rep.archived}")
    print(f"  Created/updated files: {rep.created}")
    print()

    if args.dry_run:
        print("Re-run without --dry-run to apply changes:")
        print("    python scripts\\phase_1.py")
    else:
        print("Phase 1 complete.")
        print()
        print("Next steps:")
        print("  1. Verify structure above looks clean")
        print("  2. git add -A && git commit -m \"Phase 1: cleanup\"")
        print("  3. Move on to Phase 2 (leakage audit)")

    return 0


if __name__ == "__main__":
    sys.exit(main())