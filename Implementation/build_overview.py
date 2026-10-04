"""Create notebooks/00_overview.ipynb with rich markdown + CSS styling."""
from pathlib import Path
import json

ROOT = Path(r"D:\Projects\done\thesis\Implementation")
NB_PATH = ROOT / "notebooks" / "00_overview.ipynb"

NB_PATH.parent.mkdir(parents=True, exist_ok=True)


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": [text]}


def code(lines):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines,
    }


# ============================================================
# CSS STYLE CELL
# ============================================================
CSS_CELL = """
<style>
:root {
    --primary: #0d47a1;
    --accent: #1976d2;
    --bg-soft: #f4f8fc;
    --border: #cfd8dc;
    --text: #212121;
    --warn: #ff6f00;
}

h1, h2, h3 {
    color: var(--primary);
    font-family: 'Segoe UI', Roboto, sans-serif;
    letter-spacing: 0.3px;
}

h1 { border-bottom: 3px solid var(--accent); padding-bottom: 8px; }
h2 { border-left: 5px solid var(--accent); padding-left: 12px; margin-top: 28px; }

blockquote {
    border-left: 4px solid var(--accent);
    background: var(--bg-soft);
    padding: 10px 16px;
    color: #37474f;
    font-style: normal;
    border-radius: 4px;
}

table {
    border-collapse: collapse;
    width: 100%;
    margin: 12px 0;
    font-size: 0.95em;
}
table th {
    background: var(--primary);
    color: white;
    padding: 8px 12px;
    text-align: left;
}
table td {
    border: 1px solid var(--border);
    padding: 8px 12px;
}
table tr:nth-child(even) { background: var(--bg-soft); }

code {
    background: #eceff1;
    color: #c62828;
    padding: 2px 6px;
    border-radius: 3px;
    font-family: 'Consolas', monospace;
}

pre {
    background: #1e1e1e !important;
    color: #d4d4d4 !important;
    padding: 14px;
    border-radius: 6px;
    overflow-x: auto;
}

.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 0.8em;
    font-weight: 600;
    color: white;
    margin-right: 6px;
}
.badge-blue   { background: #1976d2; }
.badge-green  { background: #2e7d32; }
.badge-orange { background: #ef6c00; }
.badge-red    { background: #c62828; }

.callout {
    background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
    border-left: 5px solid #1976d2;
    padding: 14px 18px;
    border-radius: 6px;
    margin: 14px 0;
}

.card-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 14px;
    margin: 16px 0;
}
.card {
    background: white;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 14px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.06);
}
.card h4 {
    margin: 0 0 8px 0;
    color: var(--primary);
    font-size: 1.05em;
}
.card p {
    margin: 0;
    color: #546e7a;
    font-size: 0.9em;
}

.flow {
    background: #263238;
    color: #b2ff59;
    padding: 16px;
    border-radius: 6px;
    font-family: 'Consolas', monospace;
    font-size: 0.9em;
    line-height: 1.5;
    overflow-x: auto;
    white-space: pre;
}
</style>
"""


# ============================================================
# NOTEBOOK CELLS
# ============================================================
cells = [

    # --- CSS ---
    code([CSS_CELL]),

    # --- Title ---
    md(
        "# Real-Time Cyber Threat Classification and Emerging Threat Detection\n"
        "### Using Machine Learning and Temporal Topic Modeling"
    ),

    md(
        "> **Objective** — Build an end-to-end framework that automatically "
        "classifies cyber threat reports into predefined categories and detects "
        "emerging threat patterns through temporal topic modeling."
    ),

    md(
        '<span class="badge badge-blue">ML</span>'
        '<span class="badge badge-green">NLP</span>'
        '<span class="badge badge-orange">LDA</span>'
        '<span class="badge badge-red">NMF</span>'
        '<span class="badge badge-blue">Temporal Analysis</span>'
    ),

    # --- Table of contents ---
    md(
        "## Table of Contents\n"
        "\n"
        "1. [Project Structure](#1-project-structure)\n"
        "2. [Pipeline Overview](#2-pipeline-overview)\n"
        "3. [Module Reference](#3-module-reference)\n"
        "4. [Notebook Sequence](#4-notebook-sequence)\n"
        "5. [Outputs → Thesis Chapters](#5-outputs--thesis-chapters)\n"
        "6. [Environment Check](#6-environment-check)\n"
        "7. [Quick Start](#7-quick-start)\n"
    ),

    # --- 1. Structure ---
    md("## 1. Project Structure"),
    md("Directory tree of the project (excluding `.venv` and caches):"),
    code([
        "from pathlib import Path",
        "ROOT = Path.cwd().parent",
        "for p in sorted(ROOT.rglob('*')):",
        "    if any(x in str(p) for x in ['.venv', '__pycache__', '.ipynb_checkpoints']):",
        "        continue",
        "    rel = p.relative_to(ROOT)",
        "    indent = '  ' * (len(rel.parts) - 1)",
        "    icon = '[D]' if p.is_dir() else '[F]'",
        "    print(f'{indent}{icon} {rel.name}')",
    ]),

    # --- 2. Pipeline ---
    md("## 2. Pipeline Overview"),
    md(
        '<div class="flow">'
        "Raw CTI Reports\n"
        "      |\n"
        "      v\n"
        "Preprocessing  -->  clean text + TF-IDF\n"
        "      |\n"
        "      +----------------+----------------+\n"
        "      |                                 |\n"
        "      v                                 v\n"
        "Classification (NB/LR/SVM/RF)     Topic Modeling (LDA/NMF)\n"
        "      |                                 |\n"
        "      v                                 v\n"
        "Threat Labels                     Topic Distributions\n"
        "                                        |\n"
        "                                        v\n"
        "                                Temporal Analysis\n"
        "                                        |\n"
        "                                        v\n"
        "                                Emerging Threat Score\n"
        "      |                                 |\n"
        "      +----------------+----------------+\n"
        "                       v\n"
        "              Evaluation & Visualization"
        "</div>"
    ),

    # --- 3. Modules ---
    md("## 3. Module Reference"),
    md(
        "| Module | Purpose |\n"
        "|--------|---------|\n"
        "| `src/config.py` | Paths, hyperparameters, category list, ETS weights |\n"
        "| `src/preprocess.py` | Lowercase, HTML strip, regex clean, lemmatize |\n"
        "| `src/features.py` | TF-IDF vectorizer builder |\n"
        "| `src/classify.py` | NB, LR, SVM, RF training + metrics |\n"
        "| `src/topic_model.py` | LDA & NMF training, top-word extraction |\n"
        "| `src/temporal.py` | Monthly topic prevalence aggregation |\n"
        "| `src/emerging.py` | Novelty/Growth/Persistence/Activity/TermChange → ETS |\n"
        "| `src/visualize.py` | Confusion matrices, plots |"
    ),

    # --- 4. Notebooks ---
    md("## 4. Notebook Sequence"),
    md(
        "| # | Notebook | Focus |\n"
        "|---|----------|-------|\n"
        "| 00 | `00_overview.ipynb` | Project map (this notebook) |\n"
        "| 01 | `01_eda.ipynb` | Class distribution, temporal spread, word clouds |\n"
        "| 02 | `02_preprocessing.ipynb` | Cleaning + TF-IDF shape |\n"
        "| 03 | `03_classification.ipynb` | Train 4 classifiers, confusion matrices |\n"
        "| 04 | `04_topic_modeling.ipynb` | LDA & NMF topic terms |\n"
        "| 05 | `05_emerging_threats.ipynb` | Temporal prevalence + ETS ranking |"
    ),

    # --- 5. Outputs → Chapters ---
    md("## 5. Outputs → Thesis Chapters"),
    md(
        '<div class="card-grid">'
        '<div class="card"><h4>Chapter 3</h4><p>'
        'Data preparation — <code>cleaned_cti.csv</code>, TF-IDF vectorizer'
        '</p></div>'
        '<div class="card"><h4>Chapter 4</h4><p>'
        'Methodology — <code>classify.py</code>, <code>topic_model.py</code>'
        '</p></div>'
        '<div class="card"><h4>Chapter 5</h4><p>'
        'Results — <code>classification_results.csv</code>, <code>cm_*.png</code>'
        '</p></div>'
        '<div class="card"><h4>Chapter 6</h4><p>'
        'Emerging — <code>emerging_threat_scores.csv</code>, trend plots'
        '</p></div>'
        '</div>'
    ),

    md(
        "| Artifact | Maps To |\n"
        "|----------|---------|\n"
        "| `output/cleaned_cti.csv` | Ch. 3 dataset |\n"
        "| `output/models/tfidf_vectorizer.pkl` | Ch. 3 features |\n"
        "| `output/reports/classification_results.csv` | Ch. 5 Table 5.7.5 |\n"
        "| `output/figures/cm_*.png` | Ch. 5 Figure 5.9 |\n"
        "| `output/topics/lda_topics.csv` | Ch. 5 Table 5.11.1 |\n"
        "| `output/topics/nmf_topics.csv` | Ch. 5 Table 5.11.1 |\n"
        "| `output/topics/temporal_prevalence.csv` | Ch. 5 Table 5.13.1 |\n"
        "| `output/reports/emerging_threat_scores.csv` | Ch. 5 Table 5.14.1 |"
    ),

    # --- 6. Environment check ---
    md("## 6. Environment Check"),
    md("Verify Python version and required packages are available."),
    code([
        "import sys",
        "print('Python:', sys.version.split()[0])",
        "",
        "packages = ['pandas', 'numpy', 'sklearn', 'nltk', 'matplotlib', 'seaborn', 'joblib']",
        "for pkg in packages:",
        "    try:",
        "        mod = __import__(pkg)",
        "        version = getattr(mod, '__version__', 'unknown')",
        "        print(f'  [OK] {pkg:12s} {version}')",
        "    except ImportError:",
        "        print(f'  [MISSING] {pkg}')",
    ]),

    md(
        '<div class="callout">'
        '<b>Note:</b> If any package shows <code>[MISSING]</code>, run '
        '<code>.\\00_setup.ps1</code> from the project root to install dependencies.'
        '</div>'
    ),

    # --- 7. Quick start ---
    md("## 7. Quick Start"),
    md(
        "**Step 1 — Setup environment (one time)**\n"
        "```powershell\n"
        "cd D:\\Projects\\done\\thesis\\Implementation\n"
        ".\\00_setup.ps1\n"
        "```"
    ),
    md(
        "**Step 2 — Run the full pipeline**\n"
        "```powershell\n"
        ".\\99_run_all.ps1\n"
        "```"
    ),
    md(
        "**Step 3 — Explore notebooks**\n"
        "```powershell\n"
        "jupyter notebook notebooks\\\n"
        "```"
    ),

    md(
        '<div class="callout">'
        '<b>Recommended order:</b> Run notebooks sequentially '
        '<code>01 → 02 → 03 → 04 → 05</code> so each step uses the artifacts '
        'produced by the previous one.'
        '</div>'
    ),

    # --- Footer ---
    md(
        "---\n"
        "*End of overview notebook.*"
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NB_PATH.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print(f"Created: {NB_PATH}")
print(f"Cells:   {len(cells)}")