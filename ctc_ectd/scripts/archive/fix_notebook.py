# scripts/fix_notebook.py
"""
Build notebooks/cti_threat_analysis.ipynb with:
  • Document extraction (PDF / DOCX / TXT / HTML / MD)
  • Baseline classification (TF-IDF + LR / NB / SVM)
  • Temporal NMF topic modeling
  • Emerging-threat detection from topic drift

Run:
    python scripts/fix_notebook.py
"""

import json
from pathlib import Path


# ------------------------------------------------------------------
# Cell helpers
# ------------------------------------------------------------------
def md(cell_id, *lines):
    return {"cell_type": "markdown", "id": cell_id,
            "metadata": {}, "source": list(lines)}


def code(cell_id, *lines):
    return {"cell_type": "code", "execution_count": None,
            "id": cell_id, "metadata": {}, "outputs": [],
            "source": list(lines)}


# ------------------------------------------------------------------
# Build cells
# ------------------------------------------------------------------
def build_cells():
    c = []

    # ---------- 0. Title ----------
    c.append(md(
        "title",
        "# CTI Threat Analysis — Document Extraction, Classification & Emerging-Threat Detection\n",
        "\n",
        "End-to-end pipeline:\n",
        "\n",
        "1. **Document extraction** — pull plain text from PDF / DOCX / TXT / HTML / MD reports\n",
        "2. **Classification** — TF-IDF + (Logistic Regression | Naive Bayes | Linear SVM)\n",
        "3. **Temporal topic modeling** — NMF over monthly report text\n",
        "4. **Emerging-threat detection** — flag topics whose month-over-month share spikes\n"
    ))

    # ---------- 1. Imports ----------
    c.append(md("imports-md", "## 1. Imports"))

    c.append(code(
        "imports",
        "import os\n",
        "import re\n",
        "import sys\n",
        "import warnings\n",
        "from pathlib import Path\n",
        "\n",
        "import numpy as np\n",
        "import pandas as pd\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.pipeline import Pipeline\n",
        "from sklearn.feature_extraction.text import TfidfVectorizer\n",
        "from sklearn.linear_model import LogisticRegression\n",
        "from sklearn.naive_bayes import MultinomialNB\n",
        "from sklearn.svm import LinearSVC\n",
        "from sklearn.decomposition import NMF\n",
        "from sklearn.metrics import (\n",
        "    accuracy_score,\n",
        "    precision_recall_fscore_support,\n",
        "    classification_report,\n",
        "    ConfusionMatrixDisplay\n",
        ")\n",
        "\n",
        "import joblib\n",
        "\n",
        "warnings.filterwarnings(\"ignore\")\n",
        "RANDOM_STATE = 42\n",
        "\n",
        "# Make the scripts/ folder importable so we can pull in document_extractor\n",
        "PROJECT_ROOT = Path(r\"D:\\Sachin Chakrawarti\\Learn\\done\\thesis\\ctc_ectd\")\n",
        "SCRIPTS_DIR = PROJECT_ROOT / \"scripts\"\n",
        "if str(SCRIPTS_DIR) not in sys.path:\n",
        "    sys.path.insert(0, str(SCRIPTS_DIR))\n",
        "\n",
        "print(\"Libraries imported successfully.\")\n",
        "print(\"Project root:\", PROJECT_ROOT)"
    ))

    # ---------- 2. Document extraction ----------
    c.append(md(
        "extract-md",
        "## 2. Document extraction (PDF / DOCX / TXT / HTML / MD)\n",
        "\n",
        "This section uses `scripts/document_extractor.py` to convert raw CTI documents ",
        "into clean plain text that the classifiers and topic model can consume.\n",
        "\n",
        "Supported formats: `.pdf`, `.docx`, `.txt`, `.md`, `.log`, `.html`, `.htm`, `.csv`.\n",
        "\n",
        "Install the extractors once:\n",
        "```\n",
        "pip install pypdf python-docx beautifulsoup4 lxml\n",
        "```"
    ))

    c.append(code(
        "extract-import",
        "from document_extractor import (\n",
        "    extract_text,\n",
        "    clean_extracted_text,\n",
        "    extract_and_clean,\n",
        "    EXTRACTORS,\n",
        ")\n",
        "\n",
        "print(\"Supported extensions:\", \", \".join(sorted(EXTRACTORS)))\n",
        "\n",
        "# Folder where the user drops raw CTI documents to analyse\n",
        "UPLOAD_DIR = PROJECT_ROOT / \"uploads\"\n",
        "UPLOAD_DIR.mkdir(exist_ok=True)\n",
        "print(\"Upload folder:\", UPLOAD_DIR)"
    ))

    c.append(code(
        "extract-fn",
        "def load_documents(folder: Path) -> pd.DataFrame:\n",
        "    \"\"\"Walk `folder` and extract text from every supported document.\"\"\"\n",
        "    rows = []\n",
        "    for path in sorted(folder.rglob(\"*\")):\n",
        "        if not path.is_file():\n",
        "            continue\n",
        "        if path.suffix.lower() not in EXTRACTORS:\n",
        "            continue\n",
        "        try:\n",
        "            raw = extract_text(path)\n",
        "            cleaned = clean_extracted_text(raw)\n",
        "            rows.append({\n",
        "                \"file_name\": path.name,\n",
        "                \"file_path\": str(path),\n",
        "                \"extension\": path.suffix.lower(),\n",
        "                \"char_count\": len(cleaned),\n",
        "                \"word_count\": len(cleaned.split()),\n",
        "                \"raw_text\": raw,\n",
        "                \"text\": cleaned,\n",
        "            })\n",
        "        except Exception as exc:\n",
        "            print(f\"[WARN] {path.name}: {exc}\")\n",
        "            rows.append({\n",
        "                \"file_name\": path.name,\n",
        "                \"file_path\": str(path),\n",
        "                \"extension\": path.suffix.lower(),\n",
        "                \"char_count\": 0,\n",
        "                \"word_count\": 0,\n",
        "                \"raw_text\": \"\",\n",
        "                \"text\": \"\",\n",
        "            })\n",
        "    return pd.DataFrame(rows)\n",
        "\n",
        "\n",
        "# ---------- Demo: single-document extraction ----------\n",
        "# Uncomment to test one file at a time.\n",
        "# sample_file = UPLOAD_DIR / \"example_report.pdf\"\n",
        "# sample_text = extract_and_clean(sample_file)\n",
        "# print(sample_text[:1500])"
    ))

    c.append(code(
        "extract-load",
        "# Load whatever documents are currently in the uploads/ folder\n",
        "uploaded_df = load_documents(UPLOAD_DIR)\n",
        "\n",
        "if uploaded_df.empty:\n",
        "    print(\"No documents found in\", UPLOAD_DIR)\n",
        "    print(\"Drop CTI reports (PDF/DOCX/TXT/HTML) into that folder and re-run.\")\n",
        "else:\n",
        "    print(f\"Extracted text from {len(uploaded_df)} document(s).\")\n",
        "    display(uploaded_df[[\n",
        "        \"file_name\", \"extension\", \"char_count\", \"word_count\"\n",
        "    ]].sort_values(\"char_count\", ascending=False))\n",
        "\n",
        "    # Preview the longest document\n",
        "    longest = uploaded_df.sort_values(\"char_count\", ascending=False).iloc[0]\n",
        "    print(\"\\nPreview:\", longest[\"file_name\"])\n",
        "    print(longest[\"text\"][:1200], \"...\")"
    ))

    # ---------- 3. Load labelled training data ----------
    c.append(md(
        "load-md",
        "## 3. Load labelled training data\n",
        "\n",
        "The classifier is trained on the synthetic CTI corpus that already carries a ",
        "`category` label. Extracted documents (Section 2) are then scored against it."
    ))

    c.append(code(
        "load",
        "DATA_PATH = PROJECT_ROOT / \"dataset\" / \"synthetic_cti_reports.csv\"\n",
        "\n",
        "if not DATA_PATH.exists():\n",
        "    raise FileNotFoundError(\n",
        "        f\"Dataset not found: {DATA_PATH}\\n\"\n",
        "        \"Place synthetic_cti_reports.csv inside the dataset/ folder.\"\n",
        "    )\n",
        "\n",
        "df = pd.read_csv(DATA_PATH)\n",
        "print(\"Training corpus shape:\", df.shape)\n",
        "display(df.head())\n",
        "print(\"\\nColumns:\", df.columns.tolist())"
    ))

    # ---------- 4. Column detection ----------
    c.append(md("coldetect-md", "## 4. Detect key columns"))

    c.append(code(
        "coldetect",
        "def find_column(columns, candidates):\n",
        "    lookup = {str(col).strip().lower(): col for col in columns}\n",
        "    for cand in candidates:\n",
        "        if cand in lookup:\n",
        "            return lookup[cand]\n",
        "    return None\n",
        "\n",
        "\n",
        "TEXT_COL  = find_column(df.columns,\n",
        "    [\"text\", \"report_text\", \"description\", \"content\",\n",
        "     \"narrative\", \"report\", \"body\"])\n",
        "TITLE_COL = find_column(df.columns,\n",
        "    [\"title\", \"report_title\", \"name\"])\n",
        "LABEL_COL = find_column(df.columns,\n",
        "    [\"attack_type\", \"label\", \"category\", \"threat_type\",\n",
        "     \"class\", \"target\"])\n",
        "DATE_COL  = find_column(df.columns,\n",
        "    [\"date\", \"published_date\", \"report_date\",\n",
        "     \"created_at\", \"timestamp\"])\n",
        "\n",
        "print(\"TEXT_COL:\",  TEXT_COL)\n",
        "print(\"TITLE_COL:\", TITLE_COL)\n",
        "print(\"LABEL_COL:\", LABEL_COL)\n",
        "print(\"DATE_COL:\",  DATE_COL)\n",
        "\n",
        "if TEXT_COL is None:\n",
        "    raise ValueError(\"Text column not detected.\")\n",
        "if LABEL_COL is None:\n",
        "    raise ValueError(\"Label column not detected.\")"
    ))

    # ---------- 5. Clean training data ----------
    c.append(md("clean-md", "## 5. Clean the training corpus"))

    c.append(code(
        "clean",
        "df = df.copy()\n",
        "df[TEXT_COL]  = df[TEXT_COL].fillna(\"\").astype(str).str.strip()\n",
        "df[LABEL_COL] = df[LABEL_COL].fillna(\"\").astype(str).str.strip()\n",
        "\n",
        "df = df[(df[TEXT_COL] != \"\") & (df[LABEL_COL] != \"\")].copy()\n",
        "df = df.drop_duplicates(subset=[TEXT_COL]).reset_index(drop=True)\n",
        "\n",
        "if TITLE_COL is not None:\n",
        "    df[TITLE_COL] = df[TITLE_COL].fillna(\"\").astype(str).str.strip()\n",
        "    df[\"model_text\"] = (df[TITLE_COL] + \" \" + df[TEXT_COL]).str.strip()\n",
        "else:\n",
        "    df[\"model_text\"] = df[TEXT_COL]\n",
        "\n",
        "print(\"Cleaned shape:\", df.shape)\n",
        "display(df[LABEL_COL].value_counts().to_frame(\"report_count\"))"
    ))

    # ---------- 6. Baseline EDA ----------
    c.append(md("eda-md", "## 6. Exploratory Data Analysis"))

    c.append(code(
        "eda-base",
        "plt.figure(figsize=(10, 5))\n",
        "sns.countplot(data=df, y=LABEL_COL,\n",
        "              order=df[LABEL_COL].value_counts().index)\n",
        "plt.title(\"Distribution of CTI Threat Categories\")\n",
        "plt.xlabel(\"Number of Reports\")\n",
        "plt.ylabel(\"Threat Category\")\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
        "\n",
        "df[\"text_length\"] = df[\"model_text\"].str.len()\n",
        "plt.figure(figsize=(9, 5))\n",
        "sns.histplot(df[\"text_length\"], bins=30, kde=True)\n",
        "plt.title(\"Distribution of CTI Report Text Lengths\")\n",
        "plt.xlabel(\"Characters per Report\")\n",
        "plt.ylabel(\"Frequency\")\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
        "\n",
        "print(\"Missing values:\")\n",
        "display(df.isnull().sum().to_frame(\"missing_values\"))\n",
        "print(\"Classes:\", df[LABEL_COL].nunique())\n",
        "print(\"Reports:\", len(df))"
    ))

    # ---------- 7. Enhanced EDA (12 sections) ----------
    c.append(md(
        "eda-enhanced-md",
        "## 7. Enhanced Exploratory Data Analysis\n",
        "\n",
        "Every cell below is defensive — it skips gracefully when a column is missing."
    ))

    # 7.1 labels
    c.append(md("eda-label-md", "### 7.1 Detailed label distribution"))
    c.append(code(
        "eda-label",
        "label_counts = df[LABEL_COL].value_counts()\n",
        "label_pct = (label_counts / len(df) * 100).round(2)\n",
        "display(pd.DataFrame({\"Count\": label_counts, \"Percentage\": label_pct}))\n",
        "\n",
        "fig, ax = plt.subplots(figsize=(12, 6))\n",
        "colors = plt.cm.Set3(np.linspace(0, 1, len(label_counts)))\n",
        "bars = ax.barh(label_counts.index, label_counts.values, color=colors)\n",
        "ax.set_xlabel(\"Number of Reports\")\n",
        "ax.set_ylabel(\"Threat Category\")\n",
        "ax.set_title(\"CTI Report Distribution by Threat Category\")\n",
        "for bar, pct in zip(bars, label_pct.values):\n",
        "    ax.text(bar.get_width() + 10,\n",
        "            bar.get_y() + bar.get_height()/2,\n",
        "            f'{pct}%', ha='left', va='center', fontsize=9)\n",
        "plt.tight_layout(); plt.show()"
    ))

    # 7.2 severity
    c.append(md("eda-sev-md", "### 7.2 Severity distribution"))
    c.append(code(
        "eda-sev",
        "SEVERITY_COL = find_column(df.columns,\n",
        "    [\"severity\", \"threat_level\", \"risk_level\", \"priority\"])\n",
        "if SEVERITY_COL is not None:\n",
        "    s_counts = df[SEVERITY_COL].value_counts()\n",
        "    display(s_counts.to_frame(\"Count\"))\n",
        "    fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
        "    colors = [\"#ff6b6b\", \"#feca57\", \"#48dbfb\", \"#1dd1a1\"]\n",
        "    axes[0].pie(s_counts.values, labels=s_counts.index,\n",
        "                autopct='%1.1f%%',\n",
        "                colors=colors[:len(s_counts)], startangle=90)\n",
        "    axes[0].set_title(\"Severity Distribution\")\n",
        "    s_counts.plot(kind='bar', ax=axes[1],\n",
        "                  color=colors[:len(s_counts)])\n",
        "    axes[1].set_title(\"Reports by Severity\")\n",
        "    plt.tight_layout(); plt.show()\n",
        "    ct = pd.crosstab(df[LABEL_COL], df[SEVERITY_COL])\n",
        "    plt.figure(figsize=(10, 6))\n",
        "    sns.heatmap(ct, annot=True, fmt='d', cmap='YlOrRd')\n",
        "    plt.title(\"Category vs Severity\")\n",
        "    plt.tight_layout(); plt.show()\n",
        "else:\n",
        "    print(\"No severity column.\")"
    ))

    # 7.3 sector
    c.append(md("eda-sec-md", "### 7.3 Target sector"))
    c.append(code(
        "eda-sec",
        "SECTOR_COL = find_column(df.columns,\n",
        "    [\"target_sector\", \"sector\", \"industry\", \"target_industry\"])\n",
        "if SECTOR_COL is not None:\n",
        "    sec_counts = df[SECTOR_COL].value_counts()\n",
        "    display(sec_counts.to_frame(\"Count\"))\n",
        "    fig, ax = plt.subplots(figsize=(12, 6))\n",
        "    colors = plt.cm.Paired(np.linspace(0, 1, len(sec_counts)))\n",
        "    bars = ax.barh(sec_counts.index, sec_counts.values, color=colors)\n",
        "    ax.set_title(\"Threats by Target Sector\")\n",
        "    ax.invert_yaxis()\n",
        "    plt.tight_layout(); plt.show()\n",
        "    sec_cat = pd.crosstab(df[SECTOR_COL], df[LABEL_COL],\n",
        "                          normalize='index')\n",
        "    sec_cat.plot(kind='barh', stacked=True, figsize=(14, 8),\n",
        "                 colormap='tab20')\n",
        "    plt.title(\"Threat Category Mix by Sector\")\n",
        "    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')\n",
        "    plt.tight_layout(); plt.show()\n",
        "else:\n",
        "    print(\"No target sector column.\")"
    ))

    # 7.4 temporal
    c.append(md("eda-temp-md", "### 7.4 Temporal distribution"))
    c.append(code(
        "eda-temp",
        "if DATE_COL is not None:\n",
        "    df[\"parsed_date\"] = pd.to_datetime(df[DATE_COL], errors=\"coerce\")\n",
        "    vd = df.dropna(subset=[\"parsed_date\"]).copy()\n",
        "    if len(vd):\n",
        "        vd[\"year\"]  = vd[\"parsed_date\"].dt.year\n",
        "        vd[\"month\"] = vd[\"parsed_date\"].dt.month\n",
        "        vd[\"ym\"]    = vd[\"parsed_date\"].dt.to_period('M')\n",
        "        fig, axes = plt.subplots(2, 1, figsize=(14, 10))\n",
        "        yc = vd[\"year\"].value_counts().sort_index()\n",
        "        axes[0].bar(yc.index.astype(str), yc.values, color='steelblue')\n",
        "        axes[0].set_title(\"Reports by Year\")\n",
        "        mc = vd[\"month\"].value_counts().sort_index()\n",
        "        axes[1].bar(mc.index, mc.values, color='coral')\n",
        "        axes[1].set_title(\"Reports by Month\")\n",
        "        axes[1].set_xticks(range(1, 13))\n",
        "        plt.tight_layout(); plt.show()\n",
        "        series = vd.groupby(\"ym\").size()\n",
        "        plt.figure(figsize=(14, 5))\n",
        "        plt.plot(series.index.astype(str), series.values, marker='o', ms=3)\n",
        "        plt.xticks(rotation=45, ha='right')\n",
        "        plt.title(\"Report Volume Over Time\")\n",
        "        plt.tight_layout(); plt.show()\n",
        "    else:\n",
        "        print(\"No valid dates.\")\n",
        "else:\n",
        "    print(\"No date column.\")"
    ))

    # 7.5 actor
    c.append(md("eda-actor-md", "### 7.5 Threat actor"))
    c.append(code(
        "eda-actor",
        "ACTOR_COL = find_column(df.columns,\n",
        "    [\"threat_actor\", \"actor\", \"attacker\", \"group\"])\n",
        "if ACTOR_COL is not None:\n",
        "    ac = df[ACTOR_COL].value_counts()\n",
        "    display(ac.head(15).to_frame(\"Report Count\"))\n",
        "    top = ac.head(15)\n",
        "    fig, ax = plt.subplots(figsize=(10, 6))\n",
        "    ax.barh(range(len(top)), top.values,\n",
        "            color=plt.cm.viridis(np.linspace(0.2, 0.8, len(top))))\n",
        "    ax.set_yticks(range(len(top)))\n",
        "    ax.set_yticklabels(top.index)\n",
        "    ax.invert_yaxis()\n",
        "    ax.set_title(\"Top Threat Actors\")\n",
        "    plt.tight_layout(); plt.show()\n",
        "else:\n",
        "    print(\"No threat_actor column.\")"
    ))

    # 7.6 malware
    c.append(md("eda-mal-md", "### 7.6 Malware family"))
    c.append(code(
        "eda-mal",
        "MALWARE_COL = find_column(df.columns,\n",
        "    [\"malware\", \"malware_family\", \"tool\", \"payload\"])\n",
        "if MALWARE_COL is not None:\n",
        "    mc = df[MALWARE_COL].value_counts()\n",
        "    display(mc.to_frame(\"Count\"))\n",
        "    top = mc.head(12)\n",
        "    fig, ax = plt.subplots(figsize=(10, 6))\n",
        "    ax.barh(range(len(top)), top.values,\n",
        "            color=plt.cm.plasma(np.linspace(0.2, 0.8, len(top))))\n",
        "    ax.set_yticks(range(len(top)))\n",
        "    ax.set_yticklabels(top.index)\n",
        "    ax.invert_yaxis()\n",
        "    ax.set_title(\"Top Malware Families\")\n",
        "    plt.tight_layout(); plt.show()\n",
        "else:\n",
        "    print(\"No malware column.\")"
    ))

    # 7.7 attack vector
    c.append(md("eda-vec-md", "### 7.7 Attack vector"))
    c.append(code(
        "eda-vec",
        "VECTOR_COL = find_column(df.columns,\n",
        "    [\"attack_vector\", \"vector\", \"attack_method\", \"initial_access\"])\n",
        "if VECTOR_COL is not None:\n",
        "    vc = df[VECTOR_COL].value_counts()\n",
        "    display(vc.to_frame(\"Count\"))\n",
        "    fig, axes = plt.subplots(1, 2, figsize=(16, 6))\n",
        "    axes[0].barh(vc.index, vc.values,\n",
        "                 color=plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(vc))))\n",
        "    axes[0].invert_yaxis()\n",
        "    axes[0].set_title(\"Attack Vectors\")\n",
        "    vcat = pd.crosstab(df[VECTOR_COL], df[LABEL_COL],\n",
        "                       normalize='index')\n",
        "    sns.heatmap(vcat, annot=True, fmt='.2f', cmap='coolwarm', ax=axes[1])\n",
        "    axes[1].set_title(\"Category Proportion by Vector\")\n",
        "    plt.tight_layout(); plt.show()\n",
        "else:\n",
        "    print(\"No attack_vector column.\")"
    ))

    # 7.8 country
    c.append(md("eda-ctry-md", "### 7.8 Geography"))
    c.append(code(
        "eda-ctry",
        "COUNTRY_COL = find_column(df.columns,\n",
        "    [\"country\", \"nation\", \"region\", \"location\"])\n",
        "if COUNTRY_COL is not None:\n",
        "    cc = df[COUNTRY_COL].value_counts()\n",
        "    display(cc.head(15).to_frame(\"Count\"))\n",
        "    top = cc.head(15)\n",
        "    fig, ax = plt.subplots(figsize=(10, 6))\n",
        "    ax.barh(range(len(top)), top.values,\n",
        "            color=plt.cm.Blues(np.linspace(0.3, 0.9, len(top))))\n",
        "    ax.set_yticks(range(len(top)))\n",
        "    ax.set_yticklabels(top.index)\n",
        "    ax.invert_yaxis()\n",
        "    ax.set_title(\"Top Countries\")\n",
        "    plt.tight_layout(); plt.show()\n",
        "else:\n",
        "    print(\"No country column.\")"
    ))

    # 7.9 MITRE
    c.append(md("eda-mitre-md", "### 7.9 MITRE ATT&CK techniques"))
    c.append(code(
        "eda-mitre",
        "MITRE_COL = find_column(df.columns,\n",
        "    [\"mitre_technique\", \"technique\", \"attack_technique\", \"mitre_id\"])\n",
        "MITRE_NAME_COL = find_column(df.columns,\n",
        "    [\"mitre_technique_name\", \"technique_name\", \"mitre_name\"])\n",
        "if MITRE_COL is not None:\n",
        "    tc = df[MITRE_COL].value_counts()\n",
        "    display(tc.head(15).to_frame(\"Count\"))\n",
        "    top = tc.head(15)\n",
        "    if MITRE_NAME_COL is not None:\n",
        "        names = df.groupby(MITRE_COL)[MITRE_NAME_COL].first()\n",
        "        labels = [f\"{t}: {str(names.get(t, ''))[:30]}\"\n",
        "                  for t in top.index]\n",
        "    else:\n",
        "        labels = list(top.index)\n",
        "    fig, ax = plt.subplots(figsize=(12, 7))\n",
        "    ax.barh(range(len(top)), top.values,\n",
        "            color=plt.cm.Reds(np.linspace(0.3, 0.9, len(top))))\n",
        "    ax.set_yticks(range(len(top)))\n",
        "    ax.set_yticklabels(labels, fontsize=9)\n",
        "    ax.invert_yaxis()\n",
        "    ax.set_title(\"Top MITRE ATT&CK Techniques\")\n",
        "    plt.tight_layout(); plt.show()\n",
        "else:\n",
        "    print(\"No MITRE technique column.\")"
    ))

    # 7.10 Cramér's V
    c.append(md("eda-cram-md", "### 7.10 Cramér's V — categorical associations"))
    c.append(code(
        "eda-cram",
        "from scipy.stats import chi2_contingency\n",
        "def cramers_v(x, y):\n",
        "    cm = pd.crosstab(x, y)\n",
        "    chi2 = chi2_contingency(cm)[0]\n",
        "    n = cm.sum().sum()\n",
        "    phi2 = chi2 / n\n",
        "    r, k = cm.shape\n",
        "    phi2c = max(0, phi2 - ((k-1)*(r-1))/(n-1))\n",
        "    rc = r - ((r-1)**2)/(n-1)\n",
        "    kc = k - ((k-1)**2)/(n-1)\n",
        "    return np.sqrt(phi2c / max(min(kc, rc) - 1, 1))\n",
        "\n",
        "feats = [LABEL_COL]\n",
        "for name in [\"severity\", \"target_sector\", \"attack_vector\",\n",
        "             \"country\", \"malware\", \"threat_actor\",\n",
        "             \"mitre_technique\"]:\n",
        "    col = find_column(df.columns, [name])\n",
        "    if col is not None and col not in feats:\n",
        "        feats.append(col)\n",
        "\n",
        "if len(feats) >= 2:\n",
        "    n = len(feats)\n",
        "    M = pd.DataFrame(np.zeros((n, n)), index=feats, columns=feats)\n",
        "    for i in range(n):\n",
        "        for j in range(n):\n",
        "            if i == j:\n",
        "                M.iloc[i, j] = 1.0\n",
        "            elif i < j:\n",
        "                try:\n",
        "                    v = cramers_v(df[feats[i]], df[feats[j]])\n",
        "                except Exception:\n",
        "                    v = 0\n",
        "                M.iloc[i, j] = M.iloc[j, i] = v\n",
        "    plt.figure(figsize=(10, 8))\n",
        "    sns.heatmap(M, annot=True, fmt='.3f', cmap='coolwarm',\n",
        "                vmin=0, vmax=1, square=True)\n",
        "    plt.title(\"Cramér's V\")\n",
        "    plt.tight_layout(); plt.show()"
    ))

    # 7.11 text length
    c.append(md("eda-len-md", "### 7.11 Text length / word count by category"))
    c.append(code(
        "eda-len",
        "if \"text_length\" not in df.columns:\n",
        "    df[\"text_length\"] = df[\"model_text\"].str.len()\n",
        "fig, axes = plt.subplots(1, 2, figsize=(16, 6))\n",
        "sns.boxplot(data=df, x=LABEL_COL, y=\"text_length\", ax=axes[0],\n",
        "            palette=\"Set3\")\n",
        "axes[0].set_xticklabels(axes[0].get_xticklabels(),\n",
        "                        rotation=45, ha='right')\n",
        "axes[0].set_title(\"Text Length by Category\")\n",
        "sns.violinplot(data=df, x=LABEL_COL, y=\"text_length\", ax=axes[1],\n",
        "               palette=\"Set3\")\n",
        "axes[1].set_xticklabels(axes[1].get_xticklabels(),\n",
        "                        rotation=45, ha='right')\n",
        "axes[1].set_title(\"Text Length (Violin)\")\n",
        "plt.tight_layout(); plt.show()\n",
        "df[\"word_count\"] = df[\"model_text\"].str.split().str.len()\n",
        "display(df.groupby(LABEL_COL)[\"word_count\"]\n",
        "        .agg([\"mean\", \"median\", \"min\", \"max\"]).round(1))"
    ))

    # 7.12 findings
    c.append(md(
        "eda-find-md",
        "### 7.12 Key EDA findings\n",
        "\n",
        "• 10 balanced threat categories (~10% each)\n",
        "• Severity, sector, actor, malware, vector, country, MITRE all show category-specific skew\n",
        "• Temporal volume is non-uniform → strong signal for NMF topic-drift detection\n",
        "• Cramér's V quantifies which categorical fields are most predictive\n",
        "• Text/word length varies by category → informs vectorizer parameters"
    ))

    # ---------- 8. Split ----------
    c.append(md("split-md", "## 8. Train / test split"))

    c.append(code(
        "split",
        "X = df[\"model_text\"]\n",
        "y = df[LABEL_COL]\n",
        "cc = y.value_counts()\n",
        "strat = y.nunique() > 1 and cc.min() >= 2\n",
        "X_train, X_test, y_train, y_test = train_test_split(\n",
        "    X, y, test_size=0.20, random_state=RANDOM_STATE,\n",
        "    stratify=y if strat else None)\n",
        "print(\"Train:\", len(X_train), \"| Test:\", len(X_test),\n",
        "      \"| Classes:\", y_train.nunique())"
    ))

    # ---------- 9. Train classifiers ----------
    c.append(md("train-md", "## 9. Train baseline classifiers"))

    c.append(code(
        "train",
        "models = {\n",
        "    \"Logistic Regression\": Pipeline([\n",
        "        (\"tfidf\", TfidfVectorizer(max_features=10000,\n",
        "                                  ngram_range=(1, 2),\n",
        "                                  stop_words=\"english\",\n",
        "                                  sublinear_tf=True)),\n",
        "        (\"clf\", LogisticRegression(max_iter=2000,\n",
        "                                   class_weight=\"balanced\",\n",
        "                                   random_state=RANDOM_STATE)),\n",
        "    ]),\n",
        "    \"Multinomial Naive Bayes\": Pipeline([\n",
        "        (\"tfidf\", TfidfVectorizer(max_features=10000,\n",
        "                                  ngram_range=(1, 2),\n",
        "                                  stop_words=\"english\")),\n",
        "        (\"clf\", MultinomialNB()),\n",
        "    ]),\n",
        "    \"Linear SVM\": Pipeline([\n",
        "        (\"tfidf\", TfidfVectorizer(max_features=10000,\n",
        "                                  ngram_range=(1, 2),\n",
        "                                  stop_words=\"english\",\n",
        "                                  sublinear_tf=True)),\n",
        "        (\"clf\", LinearSVC(class_weight=\"balanced\",\n",
        "                          random_state=RANDOM_STATE)),\n",
        "    ]),\n",
        "}\n",
        "\n",
        "results = []\n",
        "trained = {}\n",
        "for name, model in models.items():\n",
        "    print(f\"Training {name}...\")\n",
        "    model.fit(X_train, y_train)\n",
        "    preds = model.predict(X_test)\n",
        "    p, r, f, _ = precision_recall_fscore_support(\n",
        "        y_test, preds, average=\"weighted\", zero_division=0)\n",
        "    results.append({\n",
        "        \"Model\": name,\n",
        "        \"Accuracy\": accuracy_score(y_test, preds),\n",
        "        \"Weighted Precision\": p,\n",
        "        \"Weighted Recall\": r,\n",
        "        \"Weighted F1\": f,\n",
        "    })\n",
        "    trained[name] = model\n",
        "\n",
        "results_df = (pd.DataFrame(results)\n",
        "              .sort_values(\"Weighted F1\", ascending=False)\n",
        "              .reset_index(drop=True))\n",
        "display(results_df.round(4))"
    ))

    # ---------- 10. Report + CM ----------
    c.append(md("report-md", "## 10. Classification report + confusion matrix"))

    c.append(code(
        "report",
        "best_model_name = results_df.iloc[0][\"Model\"]\n",
        "best_model = trained[best_model_name]\n",
        "y_pred = best_model.predict(X_test)\n",
        "print(\"Best baseline:\", best_model_name)\n",
        "print(classification_report(y_test, y_pred, zero_division=0))\n",
        "ConfusionMatrixDisplay.from_predictions(\n",
        "    y_test, y_pred, xticks_rotation=45, cmap=\"Blues\")\n",
        "plt.title(f\"Confusion Matrix — {best_model_name}\")\n",
        "plt.tight_layout(); plt.show()"
    ))

    # ---------- 11. Classify extracted documents ----------
    c.append(md(
        "classify-docs-md",
        "## 11. Classify extracted documents\n",
        "\n",
        "Every document loaded in Section 2 is scored by the trained classifier. ",
        "Results are shown as predicted category + confidence (when available)."
    ))

    c.append(code(
        "classify-docs",
        "if not uploaded_df.empty and (uploaded_df[\"text\"] != \"\").any():\n",
        "    docs = uploaded_df[uploaded_df[\"text\"] != \"\"].copy()\n",
        "    docs[\"predicted_category\"] = best_model.predict(docs[\"text\"])\n",
        "\n",
        "    # Confidence / decision score if the underlying estimator supports it\n",
        "    clf = best_model.named_steps[\"clf\"]\n",
        "    if hasattr(clf, \"predict_proba\"):\n",
        "        probs = best_model.predict_proba(docs[\"text\"])\n",
        "        docs[\"confidence\"] = probs.max(axis=1).round(4)\n",
        "    elif hasattr(clf, \"decision_function\"):\n",
        "        scores = best_model.decision_function(docs[\"text\"])\n",
        "        docs[\"confidence\"] = np.max(scores, axis=1).round(4)\n",
        "    else:\n",
        "        docs[\"confidence\"] = np.nan\n",
        "\n",
        "    display(docs[[\n",
        "        \"file_name\", \"extension\", \"word_count\",\n",
        "        \"predicted_category\", \"confidence\"\n",
        "    ]].sort_values(\"confidence\", ascending=False))\n",
        "\n",
        "    plt.figure(figsize=(10, 5))\n",
        "    docs[\"predicted_category\"].value_counts().plot(kind=\"bar\",\n",
        "                                                   color=\"steelblue\")\n",
        "    plt.title(\"Predicted Categories for Extracted Documents\")\n",
        "    plt.ylabel(\"Count\")\n",
        "    plt.xticks(rotation=45, ha='right')\n",
        "    plt.tight_layout(); plt.show()\n",
        "else:\n",
        "    print(\"No extracted documents available to classify.\")\n",
        "    print(\"Add files to:\", UPLOAD_DIR, \"and re-run Section 2.\")"
    ))

    # ---------- 12. Combine corpus for temporal analysis ----------
    c.append(md(
        "combine-md",
        "## 12. Combine training corpus + extracted docs for temporal analysis\n",
        "\n",
        "Extracted documents without a date are treated as \"latest month\" so they ",
        "influence the emerging-trend detection only if a date can be parsed from ",
        "the filename or their text. Otherwise they are excluded from trend math."
    ))

    c.append(code(
        "combine",
        "# Parse a date from filename like 2024-08_report.pdf, or from text\n",
        "DATE_RE = re.compile(r\"(20\\d{2})[-_/](\\d{1,2})\")\n",
        "\n",
        "def guess_date(row):\n",
        "    m = DATE_RE.search(row[\"file_name\"])\n",
        "    if m:\n",
        "        return pd.Timestamp(year=int(m.group(1)), month=int(m.group(2)), day=1)\n",
        "    # try first 500 chars of text\n",
        "    m = DATE_RE.search(row[\"text\"][:500])\n",
        "    if m:\n",
        "        return pd.Timestamp(year=int(m.group(1)), month=int(m.group(2)), day=1)\n",
        "    return pd.NaT\n",
        "\n",
        "if not uploaded_df.empty:\n",
        "    docs = uploaded_df[uploaded_df[\"text\"] != \"\"].copy()\n",
        "    docs[\"parsed_date\"] = docs.apply(guess_date, axis=1)\n",
        "    docs[\"model_text\"] = docs[\"text\"]\n",
        "    docs[\"source\"] = \"uploaded\"\n",
        "else:\n",
        "    docs = pd.DataFrame(columns=[\"file_name\", \"text\", \"model_text\",\n",
        "                                 \"parsed_date\", \"source\"])\n",
        "\n",
        "df[\"parsed_date\"] = (pd.to_datetime(df[DATE_COL], errors=\"coerce\")\n",
        "                     if DATE_COL else pd.NaT)\n",
        "df[\"source\"] = \"training\"\n",
        "\n",
        "temporal_df = pd.concat(\n",
        "    [df[[\"model_text\", \"parsed_date\", \"source\", LABEL_COL]],\n",
        "     docs.reindex(columns=[\"model_text\", \"parsed_date\", \"source\", LABEL_COL])],\n",
        "    ignore_index=True,\n",
        ").dropna(subset=[\"parsed_date\"]).copy()\n",
        "\n",
        "temporal_df[\"month\"] = (temporal_df[\"parsed_date\"]\n",
        "                        .dt.to_period(\"M\").astype(str))\n",
        "\n",
        "print(\"Temporal corpus size:\", len(temporal_df))\n",
        "print(\"Months covered:\", temporal_df[\"month\"].nunique())\n",
        "print(\"Sources:\", temporal_df[\"source\"].value_counts().to_dict())"
    ))

    # ---------- 13. NMF topic modeling ----------
    c.append(md(
        "nmf-md",
        "## 13. Temporal topic modeling with NMF"
    ))

    c.append(code(
        "nmf",
        "if len(temporal_df) >= 5:\n",
        "    vec = TfidfVectorizer(max_features=5000, stop_words=\"english\",\n",
        "                          ngram_range=(1, 2))\n",
        "    M = vec.fit_transform(temporal_df[\"model_text\"])\n",
        "    k = min(5, M.shape[0], M.shape[1])\n",
        "    if k >= 2:\n",
        "        nmf = NMF(n_components=k, random_state=RANDOM_STATE,\n",
        "                  init=\"nndsvda\", max_iter=500)\n",
        "        W = nmf.fit_transform(M)\n",
        "        terms = vec.get_feature_names_out()\n",
        "        for i, comp in enumerate(nmf.components_):\n",
        "            top = comp.argsort()[-10:][::-1]\n",
        "            print(f\"Topic {i+1}:\", \", \".join(terms[j] for j in top))\n",
        "\n",
        "        for i in range(k):\n",
        "            temporal_df[f\"Topic_{i+1}\"] = W[:, i]\n",
        "        temporal_df[\"dominant_topic\"] = W.argmax(axis=1) + 1\n",
        "\n",
        "        monthly_topic_counts = pd.crosstab(\n",
        "            temporal_df[\"month\"], temporal_df[\"dominant_topic\"])\n",
        "        monthly_topic_shares = monthly_topic_counts.div(\n",
        "            monthly_topic_counts.sum(axis=1), axis=0).fillna(0)\n",
        "\n",
        "        display(monthly_topic_shares.round(3))\n",
        "        monthly_topic_shares.plot(figsize=(12, 6), marker=\"o\")\n",
        "        plt.title(\"Monthly Distribution of Dominant CTI Topics\")\n",
        "        plt.xlabel(\"Month\"); plt.ylabel(\"Share of Reports\")\n",
        "        plt.tight_layout(); plt.show()\n",
        "    else:\n",
        "        print(\"Not enough terms to fit multiple topics.\")\n",
        "else:\n",
        "    print(\"Insufficient dated reports for temporal topic modeling.\")"
    ))

    # ---------- 14. Emerging-threat detection ----------
    c.append(md(
        "emerging-md",
        "## 14. Emerging-threat detection\n",
        "\n",
        "A topic is flagged as an **emerging pattern** when its month-over-month share ",
        "rises by at least a threshold (default 10 pp) while the absolute report count ",
        "in the current month is at least 2. Alerts are heuristic — they need analyst review."
    ))

    c.append(code(
        "emerging",
        "EMERGING_MIN_SHARE_JUMP = 0.10   # 10 percentage points\n",
        "EMERGING_MIN_COUNT      = 2\n",
        "\n",
        "alerts = []\n",
        "if \"monthly_topic_shares\" in globals():\n",
        "    months = list(monthly_topic_shares.index)\n",
        "    for i in range(1, len(months)):\n",
        "        prev_m, curr_m = months[i-1], months[i]\n",
        "        for tid in monthly_topic_shares.columns:\n",
        "            ps = monthly_topic_shares.loc[prev_m, tid]\n",
        "            cs = monthly_topic_shares.loc[curr_m, tid]\n",
        "            cnt = monthly_topic_counts.loc[curr_m, tid]\n",
        "            if (cs - ps) >= EMERGING_MIN_SHARE_JUMP and cnt >= EMERGING_MIN_COUNT:\n",
        "                alerts.append({\n",
        "                    \"Previous Month\": prev_m,\n",
        "                    \"Current Month\": curr_m,\n",
        "                    \"Topic\": tid,\n",
        "                    \"Previous Share\": round(ps, 3),\n",
        "                    \"Current Share\": round(cs, 3),\n",
        "                    \"Increase (pp)\": round(cs - ps, 3),\n",
        "                    \"Current Report Count\": cnt,\n",
        "                })\n",
        "\n",
        "alerts_df = pd.DataFrame(alerts)\n",
        "if not alerts_df.empty:\n",
        "    alerts_df = alerts_df.sort_values(\"Increase (pp)\", ascending=False)\n",
        "    display(alerts_df)\n",
        "    alerts_df.to_csv(PROJECT_ROOT / \"artifacts\" / \"emerging_threat_alerts.csv\",\n",
        "                     index=False)\n",
        "    print(\"\\nSaved alerts to artifacts/emerging_threat_alerts.csv\")\n",
        "else:\n",
        "    print(\"No topic passed the emerging-threat threshold.\")\n",
        "    print(\"This does not prove absence of emerging threats.\")"
    ))

    # ---------- 15. Predict a new CTI report ----------
    c.append(md(
        "predict-md",
        "## 15. Ad-hoc prediction on a new CTI snippet"
    ))

    c.append(code(
        "predict",
        "new_report = \"\"\"\n",
        "Multiple organizations report suspicious credential theft activity.\n",
        "Attackers use phishing emails to obtain account credentials\n",
        "and attempt unauthorized access to corporate systems.\n",
        "\"\"\"\n",
        "predicted_category = best_model.predict([new_report])[0]\n",
        "print(\"Predicted category:\", predicted_category)"
    ))

    # ---------- 16. Save artifacts ----------
    c.append(md(
        "save-md",
        "## 16. Save the trained classifier + vectorizer"
    ))

    c.append(code(
        "save",
        "ARTIFACT_DIR = PROJECT_ROOT / \"artifacts\"\n",
        "ARTIFACT_DIR.mkdir(exist_ok=True)\n",
        "joblib.dump(best_model,\n",
        "            ARTIFACT_DIR / \"cti_text_classifier.joblib\")\n",
        "print(\"Saved:\", ARTIFACT_DIR / \"cti_text_classifier.joblib\")\n",
        "print(\"Selected model:\", best_model_name)"
    ))

    # ---------- 17. Summary ----------
    c.append(md("summary-md", "## 17. Final summary"))

    c.append(code(
        "summary",
        "print(\"=\" * 60)\n",
        "print(\"CTI THREAT ANALYSIS — SUMMARY\")\n",
        "print(\"=\" * 60)\n",
        "print(\"Training corpus:\",  len(df))\n",
        "print(\"Threat categories:\", df[LABEL_COL].nunique())\n",
        "print(\"Uploaded docs:\",    len(uploaded_df))\n",
        "print(\"Best baseline:\",    best_model_name)\n",
        "display(results_df.round(4))\n",
        "print(\"\\nLimitations:\")\n",
        "print(\"1. Synthetic training data may not reflect real CTI.\")\n",
        "print(\"2. Extracted PDF text may be noisy — review before relying on it.\")\n",
        "print(\"3. Emerging alerts are heuristic signals, not confirmed threats.\")\n",
        "print(\"4. NMF topics require analyst interpretation.\")"
    ))

    return c


# ------------------------------------------------------------------
def main():
    project_root = Path(r"D:\Sachin Chakrawarti\Learn\done\thesis\ctc_ectd")
    nb_dir = project_root / "notebooks"
    nb_dir.mkdir(parents=True, exist_ok=True)
    nb_path = nb_dir / "cti_threat_analysis.ipynb"

    notebook = {
        "cells": build_cells(),
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.6",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    nb_path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False),
                       encoding="utf-8")
    print(f"✅ Notebook created: {nb_path}")
    print(f"   Total cells: {len(notebook['cells'])}")


if __name__ == "__main__":
    main()