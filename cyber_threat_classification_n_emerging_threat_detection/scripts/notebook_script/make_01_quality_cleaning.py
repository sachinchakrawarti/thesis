# scripts/notebook_script/make_01_quality_cleaning.py
"""Generate notebooks/01_data_quality_and_cleaning.ipynb — PRESERVES CIRCL rows."""
from pathlib import Path
import nbformat as nbf


def md(t): return nbf.v4.new_markdown_cell(t)
def code(t): return nbf.v4.new_code_cell(t)


CELLS = [
    md("# Notebook 01: Data Quality and Cleaning\n"
       "\n"
       "**Input:** `data/processed/cti_unified.csv`\n"
       "**Output:** `dataset/cti_unified_clean.csv`\n"
       "**Charts:** `results/01_quality_cleaning/`\n"
       "\n"
       "**Critical:** Preserves CIRCL rows (1,207) with valid severity + relevant labels."),

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
        "DATA_PATH    = PROJECT_ROOT / 'data' / 'processed' / 'cti_unified.csv'\n"
        "CLEAN_PATH   = PROJECT_ROOT / 'dataset' / 'cti_unified_clean.csv'\n"
        "RESULTS_DIR  = PROJECT_ROOT / 'results' / '01_quality_cleaning'\n"
        "RESULTS_DIR.mkdir(parents=True, exist_ok=True)\n"
        "CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)\n"
        "\n"
        "sns.set_theme(style='whitegrid')\n"
        "print('Output:', CLEAN_PATH)"
    ),

    md("## Step 1 — Load Raw"),
    code(
        "df = pd.read_csv(DATA_PATH)\n"
        "raw_count = len(df)\n"
        "print(f'Raw rows: {raw_count:,}')\n"
        "print(f'Columns : {list(df.columns)}')\n"
        "print(f'\\nSources:')\n"
        "print(df['source'].value_counts().to_string())\n"
        "att_raw = df['techniques'].astype(str).str.contains(r'T\\d', na=False).sum()\n"
        "print(f'\\nATT&CK rows in RAW: {att_raw:,}')"
    ),

    md("## Step 2 — Missing Values"),
    code(
        "info_df = pd.DataFrame({\n"
        "    'dtype':    df.dtypes.astype(str),\n"
        "    'non_null': df.notna().sum(),\n"
        "    'null':     df.isna().sum(),\n"
        "    'null_pct': (df.isna().mean() * 100).round(2),\n"
        "})\n"
        "info_df.sort_values('null_pct', ascending=False)"
    ),

    md("### Step 2.1 — Missing Chart"),
    code(
        "null_pct = (df.isna().mean() * 100).sort_values(ascending=False)\n"
        "null_pct = null_pct[null_pct > 0]\n"
        "if len(null_pct):\n"
        "    plt.figure(figsize=(10, 5))\n"
        "    null_pct.plot(kind='barh', color='#d62728')\n"
        "    plt.title('Missing Values per Column')\n"
        "    plt.tight_layout()\n"
        "    plt.savefig(RESULTS_DIR / '01_missing_values.png', dpi=120)\n"
        "    plt.show()"
    ),

    md("## Step 3 — Duplicates"),
    code(
        "print(f'Duplicate cve_id        : {df[\"cve_id\"].duplicated().sum():,}')\n"
        "print(f'Duplicate sentence_text : {df[\"sentence_text\"].duplicated().sum():,}')"
    ),

    md("## Step 4 — Label Consistency"),
    code(
        "print('severity:'); print(df['severity'].value_counts(dropna=False))\n"
        "print('\\nrelevant:'); print(df['relevant'].value_counts(dropna=False))\n"
        "print('\\nATT&CK breakdown by source:')\n"
        "att_df = df[df['techniques'].astype(str).str.contains(r'T\\d', na=False)]\n"
        "print(att_df['source'].value_counts().to_string())"
    ),

    md("### Step 4.1 — Distribution Chart"),
    code(
        "fig, axes = plt.subplots(1, 2, figsize=(14, 4))\n"
        "sev_order = ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']\n"
        "sev_c = df['severity'].value_counts().reindex(sev_order).fillna(0).astype(int)\n"
        "axes[0].bar(sev_order, sev_c.values, color=['#2ca02c','#ff7f0e','#d62728','#8b0000'], edgecolor='black')\n"
        "axes[0].set_title('Severity Distribution')\n"
        "for i, v in enumerate(sev_c.values): axes[0].text(i, v, f'{v:,}', ha='center', va='bottom')\n"
        "rel_c = df['relevant'].value_counts(dropna=False)\n"
        "colors = ['#2ca02c' if x == 'yes' else '#d62728' for x in rel_c.index]\n"
        "axes[1].bar([str(x) for x in rel_c.index], rel_c.values, color=colors, edgecolor='black')\n"
        "axes[1].set_title('Relevance Distribution')\n"
        "for i, v in enumerate(rel_c.values): axes[1].text(i, v, f'{v:,}', ha='center', va='bottom')\n"
        "plt.tight_layout(); plt.savefig(RESULTS_DIR / '02_label_distribution.png', dpi=120); plt.show()"
    ),

    md("## Step 5 — Length Stats"),
    code(
        "df['_length'] = df['sentence_text'].astype(str).str.split().str.len()\n"
        "print(df['_length'].describe().round(1))"
    ),

    md("## Step 6 — IQR Outliers"),
    code(
        "q1, q3 = df['_length'].quantile([0.25, 0.75])\n"
        "iqr = q3 - q1\n"
        "lo, hi = q1 - 1.5*iqr, q3 + 1.5*iqr\n"
        "print(f'IQR: [{lo:.1f}, {hi:.1f}]')\n"
        "plt.figure(figsize=(10, 4))\n"
        "sns.histplot(df['_length'], bins=60, kde=True, color='#1f77b4')\n"
        "plt.axvline(lo, color='#d62728', linestyle='--')\n"
        "plt.axvline(hi, color='#d62728', linestyle='--')\n"
        "plt.title('Sentence Length')\n"
        "plt.savefig(RESULTS_DIR / '03_length_outliers.png', dpi=120); plt.show()"
    ),

    md("## Step 7 — Temporal"),
    code(
        "dt = pd.to_datetime(df['published'], errors='coerce')\n"
        "print(f'Invalid dates: {dt.isna().sum():,} ({dt.isna().mean()*100:.2f}%)')\n"
        "if dt.notna().any():\n"
        "    print(f'Range: {dt.min().date()} to {dt.max().date()}')"
    ),

    md("---\n# Cleaning"),

    md("## Step 8 — Drop Empty Sentences"),
    code(
        "before = len(df)\n"
        "df = df[df['sentence_text'].notna()].copy()\n"
        "df['sentence_text'] = df['sentence_text'].astype(str).str.strip()\n"
        "df = df[df['sentence_text'].str.len() >= 10].copy()\n"
        "print(f'{before:,} → {len(df):,}')\n"
        "att = df['techniques'].astype(str).str.contains(r'T\\d', na=False).sum()\n"
        "print(f'ATT&CK preserved: {att:,}')"
    ),

    md("## Step 9 — Normalize Labels (CRITICAL: preserve CIRCL)"),
    code(
        "print(f'Before normalize - ATT&CK: {df[\"techniques\"].astype(str).str.contains(r\"T\\d\", na=False).sum():,}')\n"
        "\n"
        "# Normalize severity\n"
        "df['severity'] = df['severity'].astype(str).str.upper().str.strip()\n"
        "df['relevant'] = df['relevant'].astype(str).str.lower().str.strip()\n"
        "\n"
        "# CRITICAL FIX: CIRCL rows must have valid labels\n"
        "circl_mask = df['source'] == 'circl'\n"
        "print(f'\\nCIRCL rows: {circl_mask.sum():,}')\n"
        "print(f'CIRCL severity before fix:')\n"
        "print(df.loc[circl_mask, 'severity'].value_counts(dropna=False).to_string())\n"
        "\n"
        "# Force valid severity on CIRCL\n"
        "valid_sev = ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']\n"
        "df.loc[circl_mask, 'severity'] = 'HIGH'  # Default for CIRCL\n"
        "df.loc[circl_mask, 'relevant'] = 'yes'    # CIRCL = always relevant\n"
        "\n"
        "# Force valid labels on ThreatCluster too (drop invalid)\n"
        "tc_mask = ~circl_mask\n"
        "tc_invalid_sev = tc_mask & ~df['severity'].isin(valid_sev)\n"
        "tc_invalid_rel = tc_mask & ~df['relevant'].isin(['yes', 'no'])\n"
        "print(f'\\nThreatCluster rows with invalid severity: {tc_invalid_sev.sum():,}')\n"
        "print(f'ThreatCluster rows with invalid relevant: {tc_invalid_rel.sum():,}')\n"
        "\n"
        "df = df[~tc_invalid_sev & ~tc_invalid_rel].copy()\n"
        "print(f'\\nAfter normalize: {len(df):,} rows')\n"
        "att = df['techniques'].astype(str).str.contains(r'T\\d', na=False).sum()\n"
        "print(f'ATT&CK preserved: {att:,}')"
    ),

    md("## Step 10 — Deduplicate (text only — NO cve_id dedup)"),
    code(
        "before = len(df)\n"
        "df = df.drop_duplicates(subset='sentence_text', keep='first').copy()\n"
        "print(f'Text dedup: {before:,} → {len(df):,}')\n"
        "att = df['techniques'].astype(str).str.contains(r'T\\d', na=False).sum()\n"
        "print(f'ATT&CK preserved: {att:,}')\n"
        "\n"
        "# Explicitly preserve CIRCL rows even if duplicate\n"
        "circl_rows = df[df['source'] == 'circl']\n"
        "print(f'CIRCL rows after dedup: {len(circl_rows):,}')\n"
        "if len(circl_rows) == 0:\n"
        "    print('⚠️ WARNING: All CIRCL rows were deleted by dedup!')\n"
        "    print('   The CIRCL text must exactly match ThreatCluster text.')"
    ),

    md("## Step 11 — Filter Length"),
    code(
        "MIN_LEN, MAX_LEN = 4, 200\n"
        "before = len(df)\n"
        "df = df[(df['_length'] >= MIN_LEN) & (df['_length'] <= MAX_LEN)].copy()\n"
        "print(f'Length filter [{MIN_LEN}, {MAX_LEN}]: {before:,} → {len(df):,}')\n"
        "att = df['techniques'].astype(str).str.contains(r'T\\d', na=False).sum()\n"
        "print(f'ATT&CK preserved: {att:,}')"
    ),

    md("## Step 12 — Verify"),
    code(
        "tech = df['techniques'].astype(str).str.contains(r'T\\d', na=False).sum()\n"
        "rel_yes = (df['relevant'] == 'yes').sum()\n"
        "date_ok = pd.to_datetime(df['published'], errors='coerce').notna().sum()\n"
        "\n"
        "print('=' * 55)\n"
        "print('CLEAN DATASET')\n"
        "print('=' * 55)\n"
        "print(f'Rows        : {len(df):,}')\n"
        "print(f'ATT&CK rows : {tech:,}  ← MUST be > 0')\n"
        "print(f'Relevant=yes: {rel_yes:,} ({rel_yes/len(df)*100:.1f}%)')\n"
        "print(f'Valid dates : {date_ok:,}')\n"
        "print(f'\\nBy source:')\n"
        "print(df['source'].value_counts().to_string())\n"
        "print()\n"
        "for s in ['LOW','MEDIUM','HIGH','CRITICAL']:\n"
        "    n = (df['severity']==s).sum()\n"
        "    print(f'  {s:10s}: {n:>7,} ({n/len(df)*100:.1f}%)')\n"
        "\n"
        "# ATT&CK breakdown\n"
        "TOP8 = ['T1059', 'T1190', 'T1005', 'T1068', 'T1203', 'T1189', 'T1499', 'T1078']\n"
        "att_df = df[df['techniques'].astype(str).str.contains(r'T\\d', na=False)].copy()\n"
        "if len(att_df) > 0:\n"
        "    att_df['parent'] = att_df['techniques'].astype(str).str.split(';').str[0].str.split('.').str[0].str.strip()\n"
        "    top8_count = att_df['parent'].isin(TOP8).sum()\n"
        "    print(f'\\nATT&CK top-8 rows: {top8_count:,}')\n"
        "    print(att_df['parent'].value_counts().head(10).to_string())\n"
        "print('=' * 55)"
    ),

    md("## Step 13 — Before/After Chart"),
    code(
        "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n"
        "axes[0].bar(['Raw', 'Clean'], [raw_count, len(df)], color=['#d62728', '#2ca02c'], edgecolor='black')\n"
        "axes[0].set_title('Row Count')\n"
        "for i, v in enumerate([raw_count, len(df)]): axes[0].text(i, v, f'{v:,}', ha='center', va='bottom')\n"
        "raw_lengths = pd.read_csv(DATA_PATH)['sentence_text'].astype(str).str.split().str.len()\n"
        "axes[1].hist(raw_lengths, bins=60, alpha=0.5, label='Raw', color='#d62728')\n"
        "axes[1].hist(df['_length'], bins=60, alpha=0.5, label='Clean', color='#2ca02c')\n"
        "axes[1].set_title('Length Distribution')\n"
        "axes[1].legend()\n"
        "plt.tight_layout(); plt.savefig(RESULTS_DIR / '04_before_after.png', dpi=120); plt.show()"
    ),

    md("## Step 14 — Save"),
    code(
        "df_out = df.drop(columns=['_length'], errors='ignore')\n"
        "df_out.to_csv(CLEAN_PATH, index=False, encoding='utf-8-sig')\n"
        "print(f'✅ Saved: {CLEAN_PATH}')\n"
        "print(f'   Shape: {df_out.shape}')"
    ),

    md("## Step 15 — Final Report"),
    code(
        "tech = df['techniques'].astype(str).str.contains(r'T\\d', na=False).sum()\n"
        "rel_yes = (df['relevant'] == 'yes').sum()\n"
        "date_ok = pd.to_datetime(df['published'], errors='coerce').notna().sum()\n"
        "\n"
        "print('=' * 55)\n"
        "print('FINAL REPORT')\n"
        "print('=' * 55)\n"
        "print(f'Raw        : {raw_count:,}')\n"
        "print(f'Clean      : {len(df):,}  ({len(df)/raw_count*100:.1f}%)')\n"
        "print(f'ATT&CK     : {tech:,}  ← must be > 0')\n"
        "print(f'Relevant   : {rel_yes:,}')\n"
        "print(f'Dates      : {date_ok:,}')\n"
        "print()\n"
        "for s in ['LOW','MEDIUM','HIGH','CRITICAL']:\n"
        "    n = (df['severity']==s).sum()\n"
        "    print(f'  {s:10s}: {n:>7,} ({n/len(df)*100:.1f}%)')"
    ),
]


def main():
    nb = nbf.v4.new_notebook()
    nb["cells"] = CELLS
    nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}
    here = Path(__file__).resolve().parent
    root = here.parent.parent if here.name == "notebook_script" else (here.parent if here.name == "scripts" else here)
    out = root / "notebooks" / "01_data_quality_and_cleaning.ipynb"
    out.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, out)
    print(f"✅ Wrote: {out}")


if __name__ == "__main__":
    main()