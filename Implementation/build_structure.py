"""Build minimal project structure for CTI thesis."""
from pathlib import Path
import json

ROOT = Path(r"D:\Projects\done\thesis\Implementation")

# ============================================================
# FOLDERS
# ============================================================
FOLDERS = [
    "dataset", "notebooks", "src", "scripts",
    "output", "output/models", "output/figures",
    "output/reports", "output/topics",
]

for f in FOLDERS:
    (ROOT / f).mkdir(parents=True, exist_ok=True)

# ============================================================
# src/ — EMPTY PLACEHOLDER FILES
# ============================================================
SRC_FILES = [
    "__init__.py", "config.py", "preprocess.py", "features.py",
    "classify.py", "topic_model.py", "temporal.py",
    "emerging.py", "visualize.py",
]
for name in SRC_FILES:
    (ROOT / "src" / name).touch(exist_ok=True)

# ============================================================
# scripts/ — EMPTY PLACEHOLDER FILES
# ============================================================
SCRIPTS = [
    "04_run_preprocessing.ps1", "05_run_classification.ps1",
    "06_run_topic_modeling.ps1", "07_run_emerging_threats.ps1",
]
for name in SCRIPTS:
    (ROOT / "scripts" / name).touch(exist_ok=True)

# ============================================================
# notebooks/ — EMPTY NOTEBOOKS
# ============================================================
NOTEBOOKS = [
    "01_eda.ipynb", "02_preprocessing.ipynb",
    "03_classification.ipynb", "04_topic_modeling.ipynb",
    "05_emerging_threats.ipynb",
]

EMPTY_NB = {
    "cells": [],
    "metadata": {
        "kernelspec": {"display_name": "Python 3",
                       "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

for name in NOTEBOOKS:
    (ROOT / "notebooks" / name).write_text(
        json.dumps(EMPTY_NB, indent=1), encoding="utf-8"
    )

# ============================================================
# ROOT FILES
# ============================================================
(ROOT / "requirements.txt").touch(exist_ok=True)
(ROOT / "99_run_all.ps1").touch(exist_ok=True)

# ============================================================
# PRINT TREE
# ============================================================
print(f"\nStructure created at: {ROOT}\n")
for p in sorted(ROOT.rglob("*")):
    if ".venv" in str(p):
        continue
    rel = p.relative_to(ROOT)
    indent = "  " * (len(rel.parts) - 1)
    icon = "[D]" if p.is_dir() else "[F]"
    print(f"  {indent}{icon} {rel.name}")