# scripts/make_02_eda.py
"""Generate notebooks/02_eda_and_data_visualization.ipynb — EDA for unified dataset."""
from pathlib import Path
import nbformat as nbf

def md(t): return nbf.v4.new_markdown_cell(t)
def code(t): return nbf.v4.new_code_cell(t)

CELLS = [
    md("# Notebook 2: EDA and Data Visualization\n"
       "\n"
       "**Purpose:** Explore the cleaned unified dataset. Produce charts for the thesis.\n"
       "\n"
       "**Input:** `data/processed/cti_unified_clean.csv`\n"
       "**Output:** `results/02_eda/` (charts)"),

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
        "from collections import Counter\n"
        "\n"
        "PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n"
        "DATA_PATH    = PROJECT_ROOT / 'data' / 'processed' / 'cti_unified_clean.csv'\n"
        "EDA_DIR      = PROJECT_ROOT / 'results' / '02_eda'\n"
        "EDA_DIR.mkdir(parents=True, exist_ok=True)\n"
        "\n"
        "sns.set_theme(style='whitegrid')\n"
        "print('Charts will be saved to:', EDA_DIR)"
    ),

    md("## Step 1 — Load Data"),
    code(
        "df = pd.read_csv(DATA_PATH)\n"
        "print(f'Shape: {df.shape}')\n"
        "print(f'Columns: {list(df.columns)}')\n"
        "df.head(3)"
    ),

    md("## Step 2 — Basic Statistics"),
    code(
        "print('=== DATASET OVERVIEW ===')\n"
        "print(f'Rows            : {len(df):,}')\n"
        "print(f'Unique CVEs     : {df[\"cve_id\"].nunique():,}')\n"
        "print(f'Sources         : {df[\"source\"].nunique()}')\n"
        "print()\n"
        "print('Sources:')\n"
        "print(df['source'].value_counts().to_string())"
    ),

    md("## Step 3 — Severity Distribution"),
    code(
        "sev_order = ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']\n"
        "sev_counts = df['severity'].value_counts().reindex(sev_order).fillna(0).astype(int)\n"
        "\n"
        "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n"
        "\n"
        "colors = ['#2ca02c', '#ff7f0e', '#d62728', '#8b0000']\n"
        "axes[0].bar(sev_order, sev_counts.values, color=colors, edgecolor='black')\n"
        "axes[0].set_title('Severity Distribution (Absolute)')\n"
        "axes[0].set_ylabel('Count')\n"
        "for i, v in enumerate(sev_counts.values):\n"
        "    axes[0].text(i, v, f'{v:,}', ha='center', va='bottom')\n"
        "\n"
        "pcts = sev_counts / sev_counts.sum() * 100\n"
        "axes[1].bar(sev_order, pcts.values, color=colors, edgecolor='black')\n"
        "axes[1].set_title('Severity Distribution (Percentage)')\n"
        "axes[1].set_ylabel('%')\n"
        "for i, v in enumerate(pcts.values):\n"
        "    axes[1].text(i, v, f'{v:.1f}%', ha='center', va='bottom')\n"
        "\n"
        "plt.tight_layout()\n"
        "plt.savefig(EDA_DIR / '01_severity_distribution.png', dpi=120)\n"
        "plt.show()\n"
        "print(sev_counts.to_string())"
    ),

    md("## Step 4 — Relevance Distribution"),
    code(
        "rel_counts = df['relevant'].value_counts(dropna=False)\n"
        "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n"
        "\n"
        "colors = ['#d62728' if x == 'no' else '#2ca02c' if x == 'yes' else '#888888' for x in rel_counts.index]\n"
        "axes[0].bar([str(x) for x in rel_counts.index], rel_counts.values, color=colors, edgecolor='black')\n"
        "axes[0].set_title('Relevance Distribution')\n"
        "axes[0].set_ylabel('Count')\n"
        "for i, v in enumerate(rel_counts.values):\n"
        "    axes[0].text(i, v, f'{v:,}', ha='center', va='bottom')\n"
        "\n"
        "# Pie chart\n"
        "axes[1].pie(rel_counts.values, labels=[str(x) for x in rel_counts.index],\n"
        "            autopct='%1.1f%%', colors=colors, startangle=90)\n"
        "axes[1].set_title('Relevance Percentage')\n"
        "\n"
        "plt.tight_layout()\n"
        "plt.savefig(EDA_DIR / '02_relevance_distribution.png', dpi=120)\n"
        "plt.show()\n"
        "\n"
        "print('Note: 111:1 imbalance — requires rebalancing for binary classifier')"
    ),

    md("## Step 5 — CVSS Score Distribution"),
    code(
        "cvss = pd.to_numeric(df['cvss_score'], errors='coerce').dropna()\n"
        "print(f'CVSS valid: {len(cvss):,} / {len(df):,}')\n"
        "print(f'Mean: {cvss.mean():.2f}  Median: {cvss.median():.2f}')\n"
        "\n"
        "fig, axes = plt.subplots(1, 2, figsize=(14, 4))\n"
        "\n"
        "axes[0].hist(cvss, bins=50, color='#1f77b4', edgecolor='black', alpha=0.7)\n"
        "axes[0].axvline(4.0, color='#2ca02c', linestyle='--', label='LOW/MED boundary (4)')\n"
        "axes[0].axvline(7.0, color='#ff7f0e', linestyle='--', label='MED/HIGH boundary (7)')\n"
        "axes[0].axvline(9.0, color='#d62728', linestyle='--', label='HIGH/CRIT boundary (9)')\n"
        "axes[0].set_title('CVSS Score Distribution')\n"
        "axes[0].set_xlabel('CVSS Score')\n"
        "axes[0].legend()\n"
        "\n"
        "sns.boxplot(x=df['severity'], y=pd.to_numeric(df['cvss_score'], errors='coerce'),\n"
        "            order=sev_order, palette=['#2ca02c', '#ff7f0e', '#d62728', '#8b0000'], ax=axes[1])\n"
        "axes[1].set_title('CVSS Score by Severity')\n"
        "axes[1].set_xlabel('Severity')\n"
        "axes[1].set_ylabel('CVSS Score')\n"
        "\n"
        "plt.tight_layout()\n"
        "plt.savefig(EDA_DIR / '03_cvss_distribution.png', dpi=120)\n"
        "plt.show()"
    ),

    md("## Step 6 — Sentence Length Distribution"),
    code(
        "df['length'] = df['sentence_text'].astype(str).str.split().str.len()\n"
        "print(df['length'].describe().round(1).to_string())\n"
        "\n"
        "plt.figure(figsize=(10, 5))\n"
        "sns.histplot(df['length'], bins=60, kde=True, color='#1f77b4')\n"
        "plt.axvline(df['length'].median(), color='#d62728', linestyle='--',\n"
        "            label=f'median={df[\"length\"].median():.0f}')\n"
        "plt.title('Sentence Length Distribution (words)')\n"
        "plt.xlabel('words')\n"
        "plt.legend()\n"
        "plt.savefig(EDA_DIR / '04_length_distribution.png', dpi=120)\n"
        "plt.show()"
    ),

    md("## Step 7 — ATT&CK Technique Distribution"),
    code(
        "att_mask = df['techniques'].astype(str).str.contains(r'T\\d', na=False)\n"
        "att_df = df[att_mask].copy()\n"
        "att_df['parent'] = att_df['techniques'].astype(str).str.split(';').str[0].str.split('.').str[0].str.strip()\n"
        "\n"
        "print(f'Rows with ATT&CK: {len(att_df):,}')\n"
        "print(f'Distinct parent techniques: {att_df[\"parent\"].nunique()}')\n"
        "\n"
        "TOP8 = ['T1059', 'T1190', 'T1005', 'T1068', 'T1203', 'T1189', 'T1499', 'T1078']\n"
        "top8_counts = att_df['parent'].value_counts().reindex(TOP8).fillna(0).astype(int)\n"
        "\n"
        "ATTACK_NAMES = {\n"
        "    'T1059': 'Command/Scripting',\n"
        "    'T1190': 'Public App Exploit',\n"
        "    'T1005': 'Data from Local',\n"
        "    'T1068': 'Priv Escalation',\n"
        "    'T1203': 'Client Exploit',\n"
        "    'T1189': 'Drive-by Compromise',\n"
        "    'T1499': 'Endpoint DoS',\n"
        "    'T1078': 'Valid Accounts',\n"
        "}\n"
        "\n"
        "plt.figure(figsize=(12, 6))\n"
        "labels = [f'{t}\\n({ATTACK_NAMES[t]})' for t in TOP8]\n"
        "plt.bar(labels, top8_counts.values, color='#1f77b4', edgecolor='black')\n"
        "plt.title('Top-8 ATT&CK Techniques')\n"
        "plt.ylabel('Count')\n"
        "plt.xticks(rotation=45, ha='right')\n"
        "for i, v in enumerate(top8_counts.values):\n"
        "    plt.text(i, v, str(v), ha='center', va='bottom')\n"
        "plt.tight_layout()\n"
        "plt.savefig(EDA_DIR / '05_attack_distribution.png', dpi=120)\n"
        "plt.show()\n"
        "print(top8_counts.to_string())"
    ),

    md("## Step 8 — Temporal Distribution"),
    code(
        "df['date_parsed'] = pd.to_datetime(df['published'], errors='coerce')\n"
        "with_date = df[df['date_parsed'].notna()]\n"
        "\n"
        "if len(with_date) > 0:\n"
        "    print(f'Rows with valid date: {len(with_date):,}')\n"
        "    print(f'Date range: {with_date[\"date_parsed\"].min().date()} to {with_date[\"date_parsed\"].max().date()}')\n"
        "\n"
        "    with_date['month'] = with_date['date_parsed'].dt.to_period('M').astype(str)\n"
        "    month_counts = with_date['month'].value_counts().sort_index()\n"
        "\n"
        "    plt.figure(figsize=(14, 5))\n"
        "    plt.bar(range(len(month_counts)), month_counts.values, color='#1f77b4', edgecolor='black')\n"
        "    plt.xticks(range(len(month_counts)), month_counts.index, rotation=45, ha='right')\n"
        "    plt.title('CVE Publications per Month')\n"
        "    plt.xlabel('Month')\n"
        "    plt.ylabel('Count')\n"
        "    plt.tight_layout()\n"
        "    plt.savefig(EDA_DIR / '06_temporal_distribution.png', dpi=120)\n"
        "    plt.show()\n"
        "else:\n"
        "    print('No valid dates available')"
    ),

    md("## Step 9 — CWE Analysis"),
    code(
        "cwe_flat = df['cwe_ids'].dropna().astype(str).str.strip()\n"
        "cwe_flat = cwe_flat[cwe_flat.str.len() > 0]\n"
        "cwe_items = cwe_flat.str.split(';').explode().str.strip()\n"
        "cwe_items = cwe_items[cwe_items.str.startswith('CWE-')]\n"
        "\n"
        "print(f'Total CWE mentions: {len(cwe_items):,}')\n"
        "top_cwe = cwe_items.value_counts().head(15)\n"
        "\n"
        "plt.figure(figsize=(12, 6))\n"
        "plt.barh(range(len(top_cwe)), top_cwe.values, color='#2ca02c', edgecolor='black')\n"
        "plt.yticks(range(len(top_cwe)), top_cwe.index)\n"
        "plt.gca().invert_yaxis()\n"
        "plt.title('Top-15 CWEs')\n"
        "plt.xlabel('Count')\n"
        "for i, v in enumerate(top_cwe.values):\n"
        "    plt.text(v, i, f' {v}', va='center')\n"
        "plt.tight_layout()\n"
        "plt.savefig(EDA_DIR / '07_top_cwes.png', dpi=120)\n"
        "plt.show()"
    ),

    md("## Step 10 — EPSS Distribution"),
    code(
        "epss = pd.to_numeric(df['epss'], errors='coerce').dropna()\n"
        "print(f'EPSS valid: {len(epss):,}')\n"
        "print(f'Mean: {epss.mean():.4f}  Median: {epss.median():.4f}')\n"
        "\n"
        "plt.figure(figsize=(10, 5))\n"
        "sns.histplot(epss, bins=60, kde=True, color='#9467bd')\n"
        "plt.axvline(0.1, color='#ff7f0e', linestyle='--', label='0.1 threshold')\n"
        "plt.axvline(0.5, color='#d62728', linestyle='--', label='0.5 threshold')\n"
        "plt.title('EPSS Score Distribution (Exploit Probability)')\n"
        "plt.xlabel('EPSS Score')\n"
        "plt.legend()\n"
        "plt.savefig(EDA_DIR / '08_epss_distribution.png', dpi=120)\n"
        "plt.show()"
    ),

    md("## Step 11 — Cross-Analysis: Severity × Source"),
    code(
        "ct = pd.crosstab(df['source'], df['severity']).reindex(columns=sev_order, fill_value=0)\n"
        "\n"
        "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n"
        "\n"
        "ct.plot(kind='bar', ax=axes[0], color=['#2ca02c', '#ff7f0e', '#d62728', '#8b0000'], edgecolor='black')\n"
        "axes[0].set_title('Severity × Source (Absolute)')\n"
        "axes[0].set_xlabel('Source')\n"
        "axes[0].set_ylabel('Count')\n"
        "axes[0].legend(title='Severity')\n"
        "axes[0].tick_params(axis='x', rotation=0)\n"
        "\n"
        "ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100\n"
        "ct_pct.plot(kind='bar', stacked=True, ax=axes[1],\n"
        "            color=['#2ca02c', '#ff7f0e', '#d62728', '#8b0000'], edgecolor='black')\n"
        "axes[1].set_title('Severity × Source (Percentage)')\n"
        "axes[1].set_xlabel('Source')\n"
        "axes[1].set_ylabel('%')\n"
        "axes[1].legend(title='Severity', loc='center left', bbox_to_anchor=(1, 0.5))\n"
        "axes[1].tick_params(axis='x', rotation=0)\n"
        "\n"
        "plt.tight_layout()\n"
        "plt.savefig(EDA_DIR / '09_severity_by_source.png', dpi=120)\n"
        "plt.show()"
    ),

    md("## Step 12 — Heatmap: Correlation Matrix"),
    code(
        "num_df = df[['cvss_score', 'epss']].copy()\n"
        "num_df['length'] = df['sentence_text'].astype(str).str.split().str.len()\n"
        "num_df['relevant_num'] = (df['relevant'] == 'yes').astype(int)\n"
        "num_df['severity_num'] = df['severity'].map({'LOW': 0, 'MEDIUM': 1, 'HIGH': 2, 'CRITICAL': 3})\n"
        "\n"
        "corr = num_df.corr()\n"
        "plt.figure(figsize=(8, 6))\n"
        "sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0, square=True)\n"
        "plt.title('Correlation Matrix')\n"
        "plt.tight_layout()\n"
        "plt.savefig(EDA_DIR / '10_correlation_matrix.png', dpi=120)\n"
        "plt.show()"
    ),

    md("## Step 13 — Top Affected Products"),
    code(
        "products = df['affected_products'] if 'affected_products' in df.columns else df['cwe_ids']\n"
        "if 'affected_products' not in df.columns:\n"
        "    print('Note: affected_products column not found. Using source instead.')\n"
        "    top_products = df['source'].value_counts().head(15)\n"
        "else:\n"
        "    prod_flat = df['affected_products'].dropna().astype(str)\n"
        "    prod_flat = prod_flat[prod_flat.str.len() > 0]\n"
        "    prod_items = prod_flat.str.split(';').explode().str.strip()\n"
        "    top_products = prod_items.value_counts().head(15)\n"
        "\n"
        "plt.figure(figsize=(12, 6))\n"
        "plt.barh(range(len(top_products)), top_products.values, color='#8c564b', edgecolor='black')\n"
        "plt.yticks(range(len(top_products)), top_products.index)\n"
        "plt.gca().invert_yaxis()\n"
        "plt.title('Top-15 Affected Products')\n"
        "plt.xlabel('Count')\n"
        "for i, v in enumerate(top_products.values):\n"
        "    plt.text(v, i, f' {v}', va='center')\n"
        "plt.tight_layout()\n"
        "plt.savefig(EDA_DIR / '11_top_products.png', dpi=120)\n"
        "plt.show()"
    ),

    md("## Step 14 — EDA Summary Report"),
    code(
        "print('=' * 60)\n"
        "print('EDA SUMMARY REPORT')\n"
        "print('=' * 60)\n"
        "print(f'Total rows            : {len(df):,}')\n"
        "print(f'Unique CVEs           : {df[\"cve_id\"].nunique():,}')\n"
        "print(f'Sources               : {df[\"source\"].nunique()}')\n"
        "print(f'\\nSeverity distribution:')\n"
        "for s in sev_order:\n"
        "    n = (df['severity'] == s).sum()\n"
        "    pct = n / len(df) * 100\n"
        "    print(f'  {s:10s}: {n:>7,}  ({pct:>5.1f}%)')\n"
        "print(f'\\nBinary (relevant):')\n"
        "for r in ['yes', 'no']:\n"
        "    n = (df['relevant'] == r).sum()\n"
        "    pct = n / len(df) * 100\n"
        "    print(f'  {r:10s}: {n:>7,}  ({pct:>5.1f}%)')\n"
        "print(f'\\nATT&CK:')\n"
        "print(f'  Rows with ATT&CK     : {att_mask.sum():,}')\n"
        "print(f'  Top-8 techniques     : {att_df[\"parent\"].isin(TOP8).sum():,}')\n"
        "print(f'\\nTemporal:')\n"
        "print(f'  Valid dates          : {with_date.shape[0]:,}')\n"
        "if with_date.shape[0] > 0:\n"
        "    print(f'  Range                : {with_date[\"date_parsed\"].min().date()} to {with_date[\"date_parsed\"].max().date()}')\n"
        "print(f'\\nNumeric:')\n"
        "print(f'  CVSS mean            : {cvss.mean():.2f}')\n"
        "print(f'  EPSS mean            : {epss.mean():.4f}')\n"
        "print('=' * 60)"
    ),

    md("## Step 15 — Thesis-Ready EDA Paragraph"),
    code(
        "summary = (\n"
        "    f'The cleaned corpus contains {len(df):,} CVE records. '\n"
        "    f'Severity is distributed as LOW {sev_counts[\"LOW\"]:,} ({sev_counts[\"LOW\"]/len(df)*100:.1f}%), '\n"
        "    f'MEDIUM {sev_counts[\"MEDIUM\"]:,} ({sev_counts[\"MEDIUM\"]/len(df)*100:.1f}%), '\n"
        "    f'HIGH {sev_counts[\"HIGH\"]:,} ({sev_counts[\"HIGH\"]/len(df)*100:.1f}%), and '\n"
        "    f'CRITICAL {sev_counts[\"CRITICAL\"]:,} ({sev_counts[\"CRITICAL\"]/len(df)*100:.1f}%). '\n"
        "    f'Only {(df[\"relevant\"] == \"yes\").sum():,} records ({(df[\"relevant\"] == \"yes\").mean()*100:.1f}%) '\n"
        "    f'are labeled relevant, indicating a severe class imbalance (111:1). '\n"
        "    f'The dataset contains {att_mask.sum():,} ATT&CK-labeled records across '\n"
        "    f'{att_df[\"parent\"].nunique()} distinct techniques, with the top-8 accounting for '\n"
        "    f'{att_df[\"parent\"].isin(TOP8).sum():,} records. '\n"
        "    f'CVSS scores average {cvss.mean():.2f} (range 0–10). '\n"
        "    f'Temporal coverage spans {with_date[\"date_parsed\"].min().date()} to '\n"
        "    f'{with_date[\"date_parsed\"].max().date()} ({with_date[\"date_parsed\"].dt.to_period(\"M\").nunique()} months).'\n"
        ")\n"
        "print(summary)\n"
        "with open(EDA_DIR / 'eda_summary.txt', 'w') as f:\n"
        "    f.write(summary)\n"
        "print(f'\\n✅ Saved to {EDA_DIR / \"eda_summary.txt\"}')"
    ),
]


def main():
    nb = nbf.v4.new_notebook()
    nb["cells"] = CELLS
    nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}
    here = Path(__file__).resolve().parent
    root = here.parent if here.name == "scripts" else here
    out = root / "notebooks" / "02_eda_and_data_visualization.ipynb"
    out.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, out)
    print(f"✅ Wrote: {out}")


if __name__ == "__main__":
    main()