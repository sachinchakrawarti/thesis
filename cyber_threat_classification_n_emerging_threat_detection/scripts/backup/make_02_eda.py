# scripts/make_02_eda_only.py
"""Generate notebooks/02_eda_and_data_visualization.ipynb — EDA ONLY (no classifiers)."""
from pathlib import Path
import nbformat as nbf


def md(t): return nbf.v4.new_markdown_cell(t)
def code(t): return nbf.v4.new_code_cell(t)


CELLS = [

    md(
        "# Notebook 2: EDA and Data Visualization\n"
        "\n"
        "**Purpose:** Descriptive statistics and visualizations for the thesis results chapter.\n"
        "\n"
        "**Input:** `dataset/cti_dataset_clean.csv` (from Notebook 01)\n"
        "**Output:** 11 PNG charts in `results/02_eda_and_data_visualization/`\n"
        "\n"
        "**Note:** This notebook does **not** train classifiers. Classification happens in Notebooks 03–04."
    ),

    md("## Step 0 — Setup"),
    code(
        "import warnings\n"
        "from pathlib import Path\n"
        "warnings.filterwarnings('ignore')\n"
        "\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n"
        "\n"
        "PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n"
        "DATA_PATH    = PROJECT_ROOT / 'dataset' / 'cti_dataset_clean.csv'\n"
        "RESULTS_DIR  = PROJECT_ROOT / 'results'\n"
        "EDA_DIR      = RESULTS_DIR / '02_eda_and_data_visualization'\n"
        "EDA_DIR.mkdir(parents=True, exist_ok=True)\n"
        "\n"
        "sns.set_theme(style='whitegrid')\n"
        "print('Charts will be saved to:', EDA_DIR)"
    ),

    md("## Step 1 — Load Clean Data"),
    code(
        "if not DATA_PATH.exists():\n"
        "    raise FileNotFoundError(f'{DATA_PATH} not found. Run Notebook 01 first.')\n"
        "\n"
        "df = pd.read_csv(DATA_PATH)\n"
        "df['label'] = (df['relevant'].str.lower() == 'yes').astype(int)\n"
        "df['length'] = df['sentence_text'].astype(str).str.split().str.len()\n"
        "df['date_parsed'] = pd.to_datetime(df['date'], errors='coerce')\n"
        "df['year'] = df['date_parsed'].dt.year\n"
        "\n"
        "print(f'Rows      : {len(df):,}')\n"
        "print(f'Documents : {df[\"doc_id\"].nunique():,}')\n"
        "print(f'Positive  : {df[\"label\"].sum():,} ({df[\"label\"].mean()*100:.1f}%)')"
    ),

    md("## Step 2 — Class Balance"),
    code(
        "counts = df['relevant'].value_counts()\n"
        "plt.figure(figsize=(6, 4))\n"
        "plt.bar(counts.index.astype(str), counts.values, color=['#d62728', '#2ca02c'])\n"
        "for i, v in enumerate(counts.values):\n"
        "    plt.text(i, v, f'{v:,}', ha='center', va='bottom')\n"
        "plt.title('Relevance Class Balance')\n"
        "plt.ylabel('Sentences')\n"
        "plt.tight_layout()\n"
        "plt.savefig(EDA_DIR / '01_class_balance.png', dpi=120)\n"
        "plt.show()"
    ),

    md("## Step 3 — Sentence Length Distribution"),
    code(
        "plt.figure(figsize=(8, 4))\n"
        "sns.histplot(df['length'], bins=40, kde=True, color='#1f77b4')\n"
        "plt.title('Sentence Length Distribution')\n"
        "plt.xlabel('words')\n"
        "plt.xlim(0, 100)\n"
        "plt.tight_layout()\n"
        "plt.savefig(EDA_DIR / '02_length_hist.png', dpi=120)\n"
        "plt.show()"
    ),

    md("## Step 4 — Length by Relevance"),
    code(
        "plt.figure(figsize=(8, 4))\n"
        "sns.boxplot(x='relevant', y='length', data=df, palette=['#2ca02c', '#d62728'])\n"
        "plt.title('Sentence Length by Relevance')\n"
        "plt.tight_layout()\n"
        "plt.savefig(EDA_DIR / '03_length_by_class.png', dpi=120)\n"
        "plt.show()"
    ),

    md("## Step 5 — Technique Distribution"),
    code(
        "techs = df.loc[df['label'] == 1, 'techniques'].dropna().astype(str)\n"
        "techs = techs[techs.str.strip() != '']\n"
        "first_tech = techs.str.split(',').str[0].str.strip().str.split('.').str[0]\n"
        "top_techs = first_tech.value_counts().head(15)\n"
        "print(f'Distinct parent techniques: {first_tech.nunique()}')\n"
        "plt.figure(figsize=(10, 5))\n"
        "top_techs.plot(kind='barh', color='#9467bd')\n"
        "plt.title('Top 15 Parent MITRE Techniques')\n"
        "plt.xlabel('sentences')\n"
        "plt.gca().invert_yaxis()\n"
        "plt.tight_layout()\n"
        "plt.savefig(EDA_DIR / '04_techniques.png', dpi=120)\n"
        "plt.show()"
    ),

    md("## Step 6 — Severity Distribution"),
    code(
        "plt.figure(figsize=(6, 4))\n"
        "df['severity'].value_counts().plot(\n"
        "    kind='pie', autopct='%1.1f%%',\n"
        "    colors=['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4'])\n"
        "plt.title('Threat Severity Distribution')\n"
        "plt.ylabel('')\n"
        "plt.tight_layout()\n"
        "plt.savefig(EDA_DIR / '05_severity.png', dpi=120)\n"
        "plt.show()"
    ),

    md("## Step 7 — Zero-Day Flag"),
    code(
        "plt.figure(figsize=(6, 4))\n"
        "df['zero_day'].value_counts().plot(kind='bar', color=['#1f77b4', '#d62728'])\n"
        "plt.title('Zero-Day Flag Distribution')\n"
        "plt.ylabel('sentences')\n"
        "plt.xticks(rotation=0)\n"
        "plt.tight_layout()\n"
        "plt.savefig(EDA_DIR / '06_zero_day.png', dpi=120)\n"
        "plt.show()"
    ),

    md("## Step 8 — Temporal Volume"),
    code(
        "valid_dts = df['date_parsed'].dropna()\n"
        "print(f'Date range: {valid_dts.min().date()} to {valid_dts.max().date()}')\n"
        "plt.figure(figsize=(12, 4))\n"
        "valid_dts.dt.to_period('M').value_counts().sort_index().plot(\n"
        "    kind='line', marker='o', color='#17becf')\n"
        "plt.title('Sentences per Month')\n"
        "plt.ylabel('count')\n"
        "plt.xticks(rotation=45)\n"
        "plt.tight_layout()\n"
        "plt.savefig(EDA_DIR / '07_temporal_volume.png', dpi=120)\n"
        "plt.show()"
    ),

    md("## Step 9 — Source Distribution"),
    code(
        "plt.figure(figsize=(8, 4))\n"
        "df['source'].value_counts().plot(kind='bar', color='#8c564b')\n"
        "plt.title('Data Sources')\n"
        "plt.ylabel('sentences')\n"
        "plt.xticks(rotation=45)\n"
        "plt.tight_layout()\n"
        "plt.savefig(EDA_DIR / '08_sources.png', dpi=120)\n"
        "plt.show()"
    ),

    md("## Step 10 — Severity by Year"),
    code(
        "if df['year'].notna().any():\n"
        "    plt.figure(figsize=(10, 4))\n"
        "    sns.countplot(data=df, x='year', hue='severity', palette='viridis')\n"
        "    plt.title('Severity Distribution by Year')\n"
        "    plt.tight_layout()\n"
        "    plt.savefig(EDA_DIR / '09_severity_by_year.png', dpi=120)\n"
        "    plt.show()"
    ),

    md("## Step 11 — Top Entities"),
    code(
        "ents = df['entities'].dropna().astype(str)\n"
        "ents = ents[ents.str.strip() != '']\n"
        "if len(ents):\n"
        "    exploded = ents.str.split(';').explode().str.strip()\n"
        "    exploded = exploded[exploded != '']\n"
        "    top_ents = exploded.value_counts().head(10)\n"
        "    plt.figure(figsize=(10, 4))\n"
        "    top_ents.plot(kind='barh', color='#bcbd22')\n"
        "    plt.title('Top 10 Named Entities')\n"
        "    plt.xlabel('mentions')\n"
        "    plt.gca().invert_yaxis()\n"
        "    plt.tight_layout()\n"
        "    plt.savefig(EDA_DIR / '10_top_entities.png', dpi=120)\n"
        "    plt.show()\n"
        "else:\n"
        "    print('No entity data')"
    ),

    md("## Step 12 — Word Frequency"),
    code(
        "from sklearn.feature_extraction.text import CountVectorizer\n"
        "cv = CountVectorizer(stop_words='english', max_features=20)\n"
        "rel = df[df['label'] == 1]['sentence_text'].astype(str)\n"
        "irr = df[df['label'] == 0]['sentence_text'].astype(str)\n"
        "cv.fit(pd.concat([rel, irr]))\n"
        "rel_words = pd.Series(\n"
        "    cv.transform(rel).sum(axis=0).A1, index=cv.get_feature_names_out()\n"
        ").sort_values(ascending=False).head(10)\n"
        "irr_words = pd.Series(\n"
        "    cv.transform(irr).sum(axis=0).A1, index=cv.get_feature_names_out()\n"
        ").sort_values(ascending=False).head(10)\n"
        "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n"
        "rel_words.plot(kind='barh', ax=axes[0], color='#2ca02c')\n"
        "axes[0].set_title('Top Words in Relevant')\n"
        "irr_words.plot(kind='barh', ax=axes[1], color='#d62728')\n"
        "axes[1].set_title('Top Words in Irrelevant')\n"
        "plt.tight_layout()\n"
        "plt.savefig(EDA_DIR / '11_word_frequency.png', dpi=120)\n"
        "plt.show()"
    ),

    md("## Step 13 — EDA Summary"),
    code(
        "print('=' * 55)\n"
        "print('EDA SUMMARY')\n"
        "print('=' * 55)\n"
        "print(f'Rows                  : {len(df):,}')\n"
        "print(f'Documents             : {df[\"doc_id\"].nunique():,}')\n"
        "print(f'Positive rate         : {df[\"label\"].mean()*100:.1f}%')\n"
        "print(f'Distinct techniques   : {first_tech.nunique()}')\n"
        "print(f'Mean sentence length  : {df[\"length\"].mean():.1f} words')\n"
        "print(f'Date range            : {valid_dts.min().date()} to {valid_dts.max().date()}')\n"
        "print('=' * 55)\n"
        "print(f'\\nCharts saved to: {EDA_DIR}')\n"
        "for f in sorted(EDA_DIR.glob('*.png')):\n"
        "    print(f'  {f.name}')"
    ),
]


def main():
    nb = nbf.v4.new_notebook()
    nb["cells"] = CELLS
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
    }
    here = Path(__file__).resolve().parent
    root = here.parent if here.name == "scripts" else here
    out = root / "notebooks" / "02_eda_and_data_visualization.ipynb"
    out.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()