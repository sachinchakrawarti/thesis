"""Create notebooks/02_preprocessing.ipynb — Text Preprocessing."""
from pathlib import Path
import json

ROOT = Path(r"D:\Projects\done\thesis\Implementation")
NB_PATH = ROOT / "notebooks" / "02_preprocessing.ipynb"

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


CSS_CELL = """
<style>
:root { --primary:#0d47a1; --accent:#1976d2; --bg-soft:#f4f8fc; --border:#cfd8dc; }
h1,h2,h3 { color: var(--primary); font-family:'Segoe UI',Roboto,sans-serif; }
h1 { border-bottom:3px solid var(--accent); padding-bottom:8px; }
h2 { border-left:5px solid var(--accent); padding-left:12px; margin-top:26px; }
table { border-collapse:collapse; width:100%; margin:10px 0; font-size:0.95em; }
table th { background:var(--primary); color:#fff; padding:8px 12px; text-align:left; }
table td { border:1px solid var(--border); padding:8px 12px; }
table tr:nth-child(even) { background:var(--bg-soft); }
code { background:#eceff1; color:#c62828; padding:2px 6px; border-radius:3px;
       font-family:'Consolas',monospace; }
pre { background:#1e1e1e !important; color:#d4d4d4 !important;
      padding:14px; border-radius:6px; overflow-x:auto; }
.callout { background:linear-gradient(135deg,#e3f2fd 0%,#bbdefb 100%);
           border-left:5px solid #1976d2; padding:14px 18px;
           border-radius:6px; margin:14px 0; }
.badge { display:inline-block; padding:3px 10px; border-radius:12px;
         font-size:0.8em; font-weight:600; color:#fff; margin-right:6px; }
.badge-blue { background:#1976d2; }
.badge-green { background:#2e7d32; }
.badge-orange { background:#ef6c00; }
.stat-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr));
             gap:12px; margin:14px 0; }
.stat { background:#fff; border:1px solid var(--border); border-radius:8px;
        padding:12px; text-align:center; box-shadow:0 2px 6px rgba(0,0,0,0.05); }
.stat .value { font-size:1.6em; font-weight:700; color:var(--primary); }
.stat .label { font-size:0.85em; color:#546e7a; margin-top:4px; }
</style>
"""


cells = [

    code([CSS_CELL]),

    md("# 02 — Text Preprocessing & Feature Extraction"),
    md(
        "> **Purpose** — Transform raw CTI text into a clean, numeric "
        "representation suitable for machine learning and topic modeling. "
        "This produces the artifacts consumed by all downstream notebooks."
    ),

    md(
        '<span class="badge badge-blue">NLP</span>'
        '<span class="badge badge-green">Cleaning</span>'
        '<span class="badge badge-orange">TF-IDF</span>'
    ),

    md(
        "## Table of Contents\n"
        "\n"
        "1. [Setup & Imports](#1-setup--imports)\n"
        "2. [Load Raw Dataset](#2-load-raw-dataset)\n"
        "3. [Inspect Raw Text Samples](#3-inspect-raw-text-samples)\n"
        "4. [Text Cleaning Function](#4-text-cleaning-function)\n"
        "5. [Apply Preprocessing](#5-apply-preprocessing)\n"
        "6. [Before / After Comparison](#6-before--after-comparison)\n"
        "7. [Duplicate Removal](#7-duplicate-removal)\n"
        "8. [Token & Vocabulary Statistics](#8-token--vocabulary-statistics)\n"
        "9. [Temporal Parsing](#9-temporal-parsing)\n"
        "10. [TF-IDF Feature Extraction](#10-tf-idf-feature-extraction)\n"
        "11. [Save Cleaned Dataset & Vectorizer](#11-save-cleaned-dataset--vectorizer)\n"
        "12. [Summary](#12-summary)\n"
    ),

    # ==================================================
    md("## 1. Setup & Imports"),
    code([
        "import sys, pathlib, re",
        "sys.path.append(str(pathlib.Path.cwd().parent))",
        "",
        "import pandas as pd",
        "import numpy as np",
        "import matplotlib.pyplot as plt",
        "import seaborn as sns",
        "import joblib",
        "",
        "from collections import Counter",
        "",
        "from src.config import DATASET_FILE, OUTPUT_DIR, MODELS_DIR, FIGURES_DIR",
        "from src.preprocess import clean_text",
        "from src.features import build_tfidf",
        "",
        "sns.set_style('whitegrid')",
        "plt.rcParams['figure.dpi'] = 100",
        "plt.rcParams['savefig.dpi'] = 150",
        "",
        "print('Raw dataset :', DATASET_FILE)",
        "print('Output dir  :', OUTPUT_DIR)",
        "print('Models dir  :', MODELS_DIR)",
    ]),

    # ==================================================
    md("## 2. Load Raw Dataset"),
    code([
        "df = pd.read_csv(DATASET_FILE)",
        "print('Loaded rows:', len(df))",
        "print('Columns     :', list(df.columns))",
        "df.head(3)",
    ]),

    # ==================================================
    md("## 3. Inspect Raw Text Samples"),
    md("Sample raw documents per category to understand vocabulary and noise."),
    code([
        "for cat in df['category'].unique():",
        "    sample = df[df['category'] == cat]['text'].iloc[0]",
        "    print(f'\\n--- {cat} ---')",
        "    print(sample[:250])",
    ]),

    md(
        '<div class="callout">'
        '<b>Observation:</b> Raw text contains mixed casing, punctuation, '
        'URLs, IP addresses, hashes, and CTI-specific tokens that must be '
        'handled carefully.'
        '</div>'
    ),

    # ==================================================
    md("## 4. Text Cleaning Function"),
    md(
        "The `clean_text()` function applies:\n"
        "- Lowercase normalization\n"
        "- HTML tag stripping\n"
        "- Non-alphanumeric removal (preserves `.`, `-`, `/`, `:`, `_` for IoCs)\n"
        "- Whitespace normalization\n"
        "- Stop-word removal\n"
        "- Lemmatization"
    ),
    code([
        "sample = df['text'].iloc[0]",
        "print('RAW:')",
        "print(sample[:300])",
        "print()",
        "cleaned = clean_text(sample)",
        "print('CLEANED:')",
        "print(cleaned[:300])",
    ]),

    # ==================================================
    md("## 5. Apply Preprocessing"),
    code([
        "df['clean_text'] = df['text'].astype(str).apply(clean_text)",
        "df['orig_len']   = df['text'].astype(str).str.len()",
        "df['clean_len']  = df['clean_text'].str.len()",
        "df['orig_words'] = df['text'].astype(str).str.split().str.len()",
        "df['clean_words']= df['clean_text'].str.split().str.len()",
        "",
        "print('Preprocessing complete.')",
        "print(df[['orig_words','clean_words']].describe().round(1))",
    ]),

    code([
        "empty = (df['clean_text'].str.strip() == '').sum()",
        "print(f'Empty after cleaning: {empty}')",
        "if empty > 0:",
        "    print('Removing empty rows...')",
        "    df = df[df['clean_text'].str.strip() != ''].reset_index(drop=True)",
        "    print(f'Rows remaining: {len(df)}')",
    ]),

    # ==================================================
    md("## 6. Before / After Comparison"),
    code([
        "compare = pd.DataFrame({",
        "    'original': df['text'].head(3).str.slice(0, 120),",
        "    'cleaned' : df['clean_text'].head(3).str.slice(0, 120),",
        "})",
        "pd.set_option('display.max_colwidth', None)",
        "compare",
    ]),

    code([
        "fig, ax = plt.subplots(figsize=(10, 4))",
        "ax.hist(df['orig_words'],  bins=40, alpha=0.6, label='Original', color='steelblue')",
        "ax.hist(df['clean_words'], bins=40, alpha=0.6, label='Cleaned',  color='crimson')",
        "ax.set_title('Word Count Before vs After Cleaning', fontweight='bold')",
        "ax.set_xlabel('Words')",
        "ax.set_ylabel('Documents')",
        "ax.legend()",
        "plt.tight_layout()",
        "plt.savefig(FIGURES_DIR / '02_wordcount_comparison.png')",
        "plt.show()",
    ]),

    # ==================================================
    md("## 7. Duplicate Removal"),
    code([
        "before = len(df)",
        "df = df.drop_duplicates(subset=['clean_text']).reset_index(drop=True)",
        "removed = before - len(df)",
        "print(f'Rows before  : {before}')",
        "print(f'Duplicates   : {removed}')",
        "print(f'Rows after   : {len(df)}')",
    ]),

    # ==================================================
    md("## 8. Token & Vocabulary Statistics"),
    code([
        "all_tokens = ' '.join(df['clean_text']).split()",
        "vocab      = Counter(all_tokens)",
        "",
        "n_tokens = len(all_tokens)",
        "n_unique = len(vocab)",
        "",
        "print(f'Total tokens      : {n_tokens:,}')",
        "print(f'Unique vocabulary : {n_unique:,}')",
        "print(f'Avg tokens/doc    : {n_tokens / len(df):.1f}')",
        "print()",
        "print('Top 20 terms:')",
        "for term, cnt in vocab.most_common(20):",
        "    print(f'  {term:20s} {cnt}')",
    ]),

    code([
        "top = vocab.most_common(25)",
        "terms, counts = zip(*top)",
        "",
        "plt.figure(figsize=(11, 5))",
        "plt.barh(range(len(terms)), counts, color='teal')",
        "plt.yticks(range(len(terms)), terms)",
        "plt.gca().invert_yaxis()",
        "plt.title('Top 25 Terms (after cleaning)', fontweight='bold')",
        "plt.xlabel('Frequency')",
        "plt.tight_layout()",
        "plt.savefig(FIGURES_DIR / '02_top_terms.png')",
        "plt.show()",
    ]),

    # ==================================================
    md("## 9. Temporal Parsing"),
    md(
        "Parse dates and derive a monthly `period` column. "
        "The `period` field is used later for temporal topic analysis "
        "and emerging-threat scoring."
    ),
    code([
        "df['date'] = pd.to_datetime(df['date'], errors='coerce')",
        "invalid_dates = df['date'].isna().sum()",
        "print(f'Invalid/missing dates: {invalid_dates}')",
        "",
        "df = df.dropna(subset=['date']).reset_index(drop=True)",
        "df['period'] = df['date'].dt.to_period('M').astype(str)",
        "",
        "print('Date range :', df['date'].min().date(), '->', df['date'].max().date())",
        "print('Periods    :', df['period'].nunique())",
        "df[['date','period']].head()",
    ]),

    code([
        "period_counts = df.groupby('period').size()",
        "",
        "plt.figure(figsize=(11, 4))",
        "period_counts.plot(kind='bar', color='darkorange', edgecolor='white')",
        "plt.title('Documents per Month', fontweight='bold')",
        "plt.xlabel('Period')",
        "plt.ylabel('Count')",
        "plt.xticks(rotation=45)",
        "plt.tight_layout()",
        "plt.savefig(FIGURES_DIR / '02_period_counts.png')",
        "plt.show()",
    ]),

    # ==================================================
    md("## 10. TF-IDF Feature Extraction"),
    md(
        "Build the TF-IDF matrix using unigrams + bigrams. "
        "Configuration is read from `src/config.py`."
    ),
    code([
        "X, vectorizer = build_tfidf(df['clean_text'])",
        "",
        "print('TF-IDF matrix shape :', X.shape)",
        "print('Vocabulary size     :', len(vectorizer.vocabulary_))",
        "print('N-gram range        :', vectorizer.ngram_range)",
        "print('Max features        :', vectorizer.max_features)",
    ]),

    code([
        "# Sparsity check",
        "density = X.nnz / (X.shape[0] * X.shape[1])",
        "print(f'Non-zero entries : {X.nnz:,}')",
        "print(f'Density          : {density:.4%}')",
        "print(f'Sparsity         : {1 - density:.4%}')",
    ]),

    code([
        "# Sample of most informative features (highest IDF-weighted terms)",
        "feature_names = vectorizer.get_feature_names_out()",
        "sample_feats = feature_names[::max(1, len(feature_names) // 30)]",
        "print('Sample features (every Nth):')",
        "print(', '.join(sample_feats[:30]))",
    ]),

    md(
        '<div class="callout">'
        '<b>Interpretation:</b> High sparsity (>99%) is expected for text. '
        'The SVM and Logistic Regression classifiers are well-suited to '
        'sparse high-dimensional feature spaces.'
        '</div>'
    ),

    # ==================================================
    md("## 11. Save Cleaned Dataset & Vectorizer"),
    code([
        "cleaned_path = OUTPUT_DIR / 'cleaned_cti.csv'",
        "df.to_csv(cleaned_path, index=False)",
        "print(f'Saved cleaned dataset  : {cleaned_path}')",
        "print(f'  Rows: {len(df)}  Columns: {list(df.columns)}')",
    ]),

    code([
        "vec_path = MODELS_DIR / 'tfidf_vectorizer.pkl'",
        "joblib.dump(vectorizer, vec_path)",
        "print(f'Saved TF-IDF vectorizer: {vec_path}')",
    ]),

    code([
        "# Sanity check - reload and verify",
        "df_check = pd.read_csv(cleaned_path)",
        "vec_check = joblib.load(vec_path)",
        "print('Reload OK:')",
        "print(f'  Rows        : {len(df_check)}')",
        "print(f'  Columns     : {len(df_check.columns)}')",
        "print(f'  Vocab size  : {len(vec_check.vocabulary_)}')",
    ]),

    # ==================================================
    md("## 12. Summary"),
    md(
        '<div class="stat-grid">'
        '<div class="stat"><div class="value">' + 'N</div>'
        '<div class="label">Documents</div></div>'
        '<div class="stat"><div class="value">V</div>'
        '<div class="label">Vocabulary</div></div>'
        '<div class="stat"><div class="value">M</div>'
        '<div class="label">Months</div></div>'
        '<div class="stat"><div class="value">C</div>'
        '<div class="label">Categories</div></div>'
        '</div>'
    ),

    code([
        "print('=== PREPROCESSING SUMMARY ===')",
        "print(f'Documents         : {len(df):,}')",
        "print(f'Vocabulary        : {len(vectorizer.vocabulary_):,}')",
        "print(f'TF-IDF shape      : {X.shape}')",
        "print(f'Months covered    : {df[\"period\"].nunique()}')",
        "print(f'Categories        : {df[\"category\"].nunique()}')",
        "print(f'Avg tokens/doc    : {df[\"clean_words\"].mean():.1f}')",
        "print()",
        "print('Artifacts saved:')",
        "print(f'  - {OUTPUT_DIR / \"cleaned_cti.csv\"}')",
        "print(f'  - {MODELS_DIR / \"tfidf_vectorizer.pkl\"}')",
    ]),

    md(
        '<div class="callout">'
        '<b>Next step:</b> Proceed to <code>03_classification.ipynb</code> to '
        'train classifiers on the cleaned dataset and TF-IDF features.'
        '</div>'
    ),

    md("---\n*End of preprocessing notebook.*"),
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