"""Create notebooks/01_eda.ipynb — Exploratory Data Analysis."""
from pathlib import Path
import json

ROOT = Path(r"D:\Projects\done\thesis\Implementation")
NB_PATH = ROOT / "notebooks" / "01_eda.ipynb"

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
    --accent:  #1976d2;
    --bg-soft: #f4f8fc;
    --border:  #cfd8dc;
    --warn:    #ff6f00;
    --ok:      #2e7d32;
}

h1, h2, h3 {
    color: var(--primary);
    font-family: 'Segoe UI', Roboto, sans-serif;
}
h1 { border-bottom: 3px solid var(--accent); padding-bottom: 8px; }
h2 { border-left: 5px solid var(--accent); padding-left: 12px; margin-top: 26px; }

table { border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 0.95em; }
table th { background: var(--primary); color: white; padding: 8px 12px; text-align: left; }
table td { border: 1px solid var(--border); padding: 8px 12px; }
table tr:nth-child(even) { background: var(--bg-soft); }

code {
    background: #eceff1; color: #c62828; padding: 2px 6px;
    border-radius: 3px; font-family: 'Consolas', monospace;
}
pre {
    background: #1e1e1e !important; color: #d4d4d4 !important;
    padding: 14px; border-radius: 6px; overflow-x: auto;
}

.callout {
    background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
    border-left: 5px solid #1976d2;
    padding: 14px 18px; border-radius: 6px; margin: 14px 0;
}

.badge {
    display: inline-block; padding: 3px 10px; border-radius: 12px;
    font-size: 0.8em; font-weight: 600; color: white; margin-right: 6px;
}
.badge-blue   { background: #1976d2; }
.badge-green  { background: #2e7d32; }
.badge-orange { background: #ef6c00; }

.stat-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 12px; margin: 14px 0;
}
.stat {
    background: white; border: 1px solid var(--border);
    border-radius: 8px; padding: 12px; text-align: center;
    box-shadow: 0 2px 6px rgba(0,0,0,0.05);
}
.stat .value { font-size: 1.6em; font-weight: 700; color: var(--primary); }
.stat .label { font-size: 0.85em; color: #546e7a; margin-top: 4px; }
</style>
"""


# ============================================================
# NOTEBOOK CELLS
# ============================================================
cells = [

    # CSS
    code([CSS_CELL]),

    # Title
    md("# 01 — Exploratory Data Analysis"),
    md(
        "> **Purpose** — Understand the CTI dataset: size, class balance, "
        "temporal spread, text characteristics, and potential data quality issues "
        "before building models."
    ),

    md(
        '<span class="badge badge-blue">EDA</span>'
        '<span class="badge badge-green">Data Quality</span>'
        '<span class="badge badge-orange">Visualization</span>'
    ),

    # TOC
    md(
        "## Table of Contents\n"
        "\n"
        "1. [Setup & Imports](#1-setup--imports)\n"
        "2. [Load Dataset](#2-load-dataset)\n"
        "3. [Basic Info & Shape](#3-basic-info--shape)\n"
        "4. [Missing Values & Duplicates](#4-missing-values--duplicates)\n"
        "5. [Class Distribution](#5-class-distribution)\n"
        "6. [Temporal Coverage](#6-temporal-coverage)\n"
        "7. [Text Length Analysis](#7-text-length-analysis)\n"
        "8. [Top Terms per Class](#8-top-terms-per-class)\n"
        "9. [Word Clouds](#9-word-clouds)\n"
        "10. [Key Observations](#10-key-observations)\n"
    ),

    # ==================================================
    # 1. Setup
    # ==================================================
    md("## 1. Setup & Imports"),
    code([
        "import sys, pathlib",
        "sys.path.append(str(pathlib.Path.cwd().parent))",
        "",
        "import pandas as pd",
        "import numpy as np",
        "import matplotlib.pyplot as plt",
        "import seaborn as sns",
        "from collections import Counter",
        "",
        "from src.config import DATASET_FILE, FIGURES_DIR",
        "",
        "sns.set_style('whitegrid')",
        "plt.rcParams['figure.dpi'] = 100",
        "plt.rcParams['savefig.dpi'] = 150",
        "",
        "print('Dataset path:', DATASET_FILE)",
        "print('Figures dir :', FIGURES_DIR)",
    ]),

    # ==================================================
    # 2. Load
    # ==================================================
    md("## 2. Load Dataset"),
    code([
        "df = pd.read_csv(DATASET_FILE)",
        "df['date'] = pd.to_datetime(df['date'])",
        "df['month'] = df['date'].dt.to_period('M').astype(str)",
        "",
        "print('Loaded:', df.shape)",
        "df.head()",
    ]),

    code([
        "df.info()",
    ]),

    # ==================================================
    # 3. Basic Info
    # ==================================================
    md("## 3. Basic Info & Shape"),
    code([
        "n_rows, n_cols = df.shape",
        "n_categories  = df['category'].nunique()",
        "date_min      = df['date'].min().date()",
        "date_max      = df['date'].max().date()",
        "n_months      = df['month'].nunique()",
        "n_sources     = df['source'].nunique() if 'source' in df.columns else 0",
        "",
        "print(f'Rows           : {n_rows}')",
        "print(f'Columns        : {n_cols}')",
        "print(f'Categories     : {n_categories}')",
        "print(f'Date range     : {date_min} -> {date_max}')",
        "print(f'Months covered : {n_months}')",
        "print(f'Sources        : {n_sources}')",
        "print(f'Columns        : {list(df.columns)}')",
    ]),

    # ==================================================
    # 4. Missing / Duplicates
    # ==================================================
    md("## 4. Missing Values & Duplicates"),
    code([
        "missing = df.isnull().sum()",
        "missing = missing[missing > 0]",
        "print('Missing values per column:')",
        "print(missing if len(missing) else '  (none)')",
        "",
        "dup_text = df.duplicated(subset=['text']).sum()",
        "dup_full = df.duplicated().sum()",
        "print(f'\\nDuplicate text rows : {dup_text}')",
        "print(f'Duplicate full rows : {dup_full}')",
    ]),

    md(
        '<div class="callout">'
        '<b>Data quality note:</b> If duplicates exist, they should be removed '
        'in the preprocessing step (<code>02_preprocessing.ipynb</code>) to '
        'prevent leakage and bias in the classifiers.'
        '</div>'
    ),

    # ==================================================
    # 5. Class Distribution
    # ==================================================
    md("## 5. Class Distribution"),
    code([
        "class_counts = df['category'].value_counts()",
        "class_pct    = (class_counts / len(df) * 100).round(2)",
        "",
        "summary = pd.DataFrame({",
        "    'Count': class_counts,",
        "    'Percent': class_pct,",
        "})",
        "summary",
    ]),

    code([
        "fig, ax = plt.subplots(figsize=(9, 4))",
        "colors = sns.color_palette('viridis', len(class_counts))",
        "bars = ax.bar(class_counts.index, class_counts.values, color=colors)",
        "ax.set_title('Class Distribution', fontsize=13, fontweight='bold')",
        "ax.set_xlabel('Threat Category')",
        "ax.set_ylabel('Number of Documents')",
        "",
        "for bar, v in zip(bars, class_counts.values):",
        "    ax.text(bar.get_x() + bar.get_width()/2, v + 1, str(v),",
        "            ha='center', fontsize=9)",
        "",
        "plt.tight_layout()",
        "plt.savefig(FIGURES_DIR / '01_class_distribution.png')",
        "plt.show()",
    ]),

    code([
        "plt.figure(figsize=(7, 7))",
        "plt.pie(class_counts.values, labels=class_counts.index,",
        "        autopct='%1.1f%%', startangle=140,",
        "        colors=sns.color_palette('Set3', len(class_counts)))",
        "plt.title('Class Distribution (Percent)', fontweight='bold')",
        "plt.tight_layout()",
        "plt.savefig(FIGURES_DIR / '01_class_pie.png')",
        "plt.show()",
    ]),

    md("### Class Imbalance Check"),
    code([
        "maj = class_counts.max()",
        "mnr = class_counts.min()",
        "ratio = round(maj / mnr, 2)",
        "print(f'Majority class : {class_counts.idxmax()} ({maj})')",
        "print(f'Minority class : {class_counts.idxmin()} ({mnr})')",
        "print(f'Imbalance ratio: {ratio}:1')",
        "if ratio < 2:",
        "    print('-> Balanced dataset.')",
        "else:",
        "    print('-> Imbalanced. Use macro-F1 in evaluation.')",
    ]),

    # ==================================================
    # 6. Temporal Coverage
    # ==================================================
    md("## 6. Temporal Coverage"),
    code([
        "monthly = df.groupby(['month', 'category']).size().unstack(fill_value=0)",
        "print('Monthly counts per category:')",
        "monthly.head()",
    ]),

    code([
        "fig, ax = plt.subplots(figsize=(12, 5))",
        "monthly.plot(ax=ax, marker='o', linewidth=1.5)",
        "ax.set_title('Temporal Distribution by Category', fontsize=13, fontweight='bold')",
        "ax.set_xlabel('Month')",
        "ax.set_ylabel('Documents')",
        "plt.xticks(rotation=45)",
        "plt.legend(title='Category', bbox_to_anchor=(1.02, 1), loc='upper left')",
        "plt.tight_layout()",
        "plt.savefig(FIGURES_DIR / '01_temporal_distribution.png')",
        "plt.show()",
    ]),

    code([
        "monthly_total = df.groupby('month').size()",
        "print('Total documents per month:')",
        "print(monthly_total.to_string())",
    ]),

    # ==================================================
    # 7. Text Length
    # ==================================================
    md("## 7. Text Length Analysis"),
    code([
        "df['text_len']  = df['text'].astype(str).str.len()",
        "df['word_cnt']  = df['text'].astype(str).str.split().str.len()",
        "",
        "print('Character length stats:')",
        "print(df['text_len'].describe().round(1))",
        "print('\\nWord count stats:')",
        "print(df['word_cnt'].describe().round(1))",
    ]),

    code([
        "fig, axes = plt.subplots(1, 2, figsize=(12, 4))",
        "",
        "axes[0].hist(df['word_cnt'], bins=30, color='steelblue', edgecolor='white')",
        "axes[0].set_title('Word Count Distribution')",
        "axes[0].set_xlabel('Words per document')",
        "axes[0].set_ylabel('Frequency')",
        "",
        "df.boxplot(column='word_cnt', by='category', ax=axes[1], grid=False)",
        "axes[1].set_title('Word Count by Category')",
        "axes[1].set_xlabel('Category')",
        "axes[1].set_ylabel('Words')",
        "plt.suptitle('')",
        "",
        "plt.tight_layout()",
        "plt.savefig(FIGURES_DIR / '01_text_length.png')",
        "plt.show()",
    ]),

    # ==================================================
    # 8. Top Terms per Class
    # ==================================================
    md("## 8. Top Terms per Class"),
    code([
        "def top_terms(texts, n=15):",
        "    words = ' '.join(texts).lower().split()",
        "    return Counter(words).most_common(n)",
        "",
        "for cat in df['category'].unique():",
        "    texts = df.loc[df['category'] == cat, 'text'].astype(str).tolist()",
        "    terms = top_terms(texts, n=12)",
        "    print(f'\\n[{cat}]')",
        "    print('  ', ', '.join([t for t, _ in terms]))",
    ]),

    # ==================================================
    # 9. Word Clouds
    # ==================================================
    md("## 9. Word Clouds per Category"),
    code([
        "try:",
        "    from wordcloud import WordCloud",
        "except ImportError:",
        "    import subprocess",
        "    subprocess.run(['pip', 'install', 'wordcloud', '-q'])",
        "    from wordcloud import WordCloud",
        "",
        "categories = df['category'].unique()",
        "fig, axes = plt.subplots(2, 3, figsize=(15, 8))",
        "axes = axes.flatten()",
        "",
        "for i, cat in enumerate(categories):",
        "    text = ' '.join(df.loc[df['category'] == cat, 'text'].astype(str))",
        "    wc = WordCloud(width=600, height=350,",
        "                   background_color='white',",
        "                   colormap='viridis',",
        "                   max_words=100).generate(text)",
        "    axes[i].imshow(wc, interpolation='bilinear')",
        "    axes[i].set_title(cat, fontweight='bold')",
        "    axes[i].axis('off')",
        "",
        "for j in range(len(categories), len(axes)):",
        "    axes[j].axis('off')",
        "",
        "plt.suptitle('Word Clouds by Threat Category', fontsize=14, fontweight='bold')",
        "plt.tight_layout()",
        "plt.savefig(FIGURES_DIR / '01_wordclouds.png')",
        "plt.show()",
    ]),

    # ==================================================
    # 10. Observations
    # ==================================================
    md("## 10. Key Observations"),
    md(
        "| Aspect | Finding |\n"
        "|--------|---------|\n"
        "| Dataset size | See Section 3 |\n"
        "| Class balance | See Section 5 imbalance check |\n"
        "| Temporal spread | See Section 6 monthly counts |\n"
        "| Text length | See Section 7 statistics |\n"
        "| Data quality | See Section 4 missing/duplicates |"
    ),

    md(
        '<div class="callout">'
        '<b>Next step:</b> Proceed to <code>02_preprocessing.ipynb</code> to '
        'clean the text and build TF-IDF features.'
        '</div>'
    ),

    code([
        "print('EDA complete.')",
        "print(f'Rows: {len(df)} | Categories: {df[\"category\"].nunique()} '",
        "      f'| Months: {df[\"month\"].nunique()}')",
    ]),

    md("---\n*End of EDA notebook.*"),
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