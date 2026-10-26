# D:\Projects\done\thesis\cyber_threat_classification_n_emerging_threat_detection\scripts\makenotebook.py
"""Regenerate notebooks/cti.ipynb for the real cti_dataset.csv corpus."""
from pathlib import Path
import nbformat as nbf


def md(t): return nbf.v4.new_markdown_cell(t)
def code(t): return nbf.v4.new_code_cell(t)


CELLS = [

    md(
        "# Cyber Threat Classification and Emerging Threat Detection\n"
        "\n"
        "**Data:** `dataset/cti_dataset.csv` (real corpus: 1000 docs, ~12k sentences)\n"
        "**Tasks:** EDA -> Relevance Classifier -> Technique Classifier -> BERTopic -> PDF Analysis"
    ),

    md("## 0. Setup"),
    code(
        "import json, time, re, warnings, logging\n"
        "from pathlib import Path\n"
        "from collections import Counter\n"
        "\n"
        "warnings.filterwarnings('ignore')\n"
        "logging.getLogger('tensorflow').setLevel(logging.ERROR)\n"
        "\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n"
        "\n"
        "from sklearn.model_selection import GroupShuffleSplit\n"
        "from sklearn.feature_extraction.text import TfidfVectorizer\n"
        "from sklearn.svm import LinearSVC\n"
        "from sklearn.metrics import accuracy_score, precision_recall_fscore_support\n"
        "\n"
        "PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n"
        "DATA_PATH    = PROJECT_ROOT / 'dataset' / 'cti_dataset.csv'\n"
        "PDF_DIR      = PROJECT_ROOT / 'sample_documents'\n"
        "CLS_DIR      = PROJECT_ROOT / 'models' / 'classification'\n"
        "THREAT_DIR   = PROJECT_ROOT / 'models' / 'emerging_threat'\n"
        "RESULTS_DIR  = PROJECT_ROOT / 'results'\n"
        "for d in (CLS_DIR, THREAT_DIR, RESULTS_DIR): d.mkdir(parents=True, exist_ok=True)\n"
        "\n"
        "RANDOM_SEED = 42\n"
        "N_SPLITS = 5\n"
        "sns.set_theme(style='whitegrid')\n"
        "print('setup complete')"
    ),

    md("## 1. Load Data"),
    code(
        "if not DATA_PATH.exists(): raise FileNotFoundError(DATA_PATH)\n"
        "df = pd.read_csv(DATA_PATH)\n"
        "print(f'Loaded {len(df)} rows, {df[\"doc_id\"].nunique()} documents.')"
    ),

    # ============ EDA ============
    md("## 2. Exploratory Data Analysis"),

    md("### 2.1 Relevance Balance"),
    code(
        "plt.figure(figsize=(6, 4))\n"
        "df['relevant'].value_counts().plot(kind='bar', color=['#2ca02c', '#d62728'])\n"
        "plt.title('Relevance Class Balance')\n"
        "plt.xticks(rotation=0)\n"
        "plt.tight_layout()\n"
        "plt.savefig(RESULTS_DIR / 'eda_01_class_balance.png')\n"
        "plt.show()"
    ),

    md("### 2.2 Sentence Length"),
    code(
        "df['length'] = df['sentence_text'].astype(str).str.split().str.len()\n"
        "plt.figure(figsize=(8, 4))\n"
        "sns.histplot(df['length'], bins=40, kde=True, color='#1f77b4')\n"
        "plt.title('Sentence Length Distribution')\n"
        "plt.xlim(0, 50)\n"
        "plt.savefig(RESULTS_DIR / 'eda_02_length.png')\n"
        "plt.show()"
    ),

    md("### 2.3 Length by Relevance"),
    code(
        "plt.figure(figsize=(8, 4))\n"
        "sns.boxplot(x='relevant', y='length', data=df, palette=['#2ca02c', '#d62728'])\n"
        "plt.title('Sentence Length by Relevance')\n"
        "plt.savefig(RESULTS_DIR / 'eda_03_length_rel.png')\n"
        "plt.show()"
    ),

    md("### 2.4 Technique Distribution"),
    code(
        "plt.figure(figsize=(10, 5))\n"
        "df['techniques'].value_counts().head(15).plot(kind='barh', color='#9467bd')\n"
        "plt.title('Top 15 MITRE ATT&CK Techniques')\n"
        "plt.gca().invert_yaxis()\n"
        "plt.savefig(RESULTS_DIR / 'eda_04_techniques.png')\n"
        "plt.show()"
    ),

    md("### 2.5 Tactic Distribution"),
    code(
        "if df['tactic'].notna().any():\n"
        "    plt.figure(figsize=(10, 5))\n"
        "    df['tactic'].value_counts().plot(kind='barh', color='#ff7f0e')\n"
        "    plt.title('Tactic Distribution')\n"
        "    plt.gca().invert_yaxis()\n"
        "    plt.savefig(RESULTS_DIR / 'eda_05_tactics.png')\n"
        "    plt.show()\n"
        "else:\n"
        "    print('No tactic data')"
    ),

    md("### 2.6 Severity Distribution"),
    code(
        "plt.figure(figsize=(6, 4))\n"
        "df['severity'].value_counts().plot(kind='pie', autopct='%1.1f%%')\n"
        "plt.title('Threat Severity Distribution')\n"
        "plt.ylabel('')\n"
        "plt.savefig(RESULTS_DIR / 'eda_06_severity.png')\n"
        "plt.show()"
    ),

    md("### 2.7 Zero-Day Distribution"),
    code(
        "plt.figure(figsize=(6, 4))\n"
        "df['zero_day'].value_counts().plot(kind='bar', color=['#1f77b4', '#d62728'])\n"
        "plt.title('Zero-Day Flag Distribution')\n"
        "plt.xticks(rotation=0)\n"
        "plt.savefig(RESULTS_DIR / 'eda_07_zero_day.png')\n"
        "plt.show()"
    ),

    md("### 2.8 Temporal Volume"),
    code(
        "if 'date' in df.columns:\n"
        "    plt.figure(figsize=(12, 4))\n"
        "    pd.to_datetime(df['date']).dt.to_period('M').value_counts().sort_index().plot(\n"
        "        kind='line', marker='o', color='#17becf')\n"
        "    plt.title('Sentences Over Time (Monthly)')\n"
        "    plt.xticks(rotation=45)\n"
        "    plt.tight_layout()\n"
        "    plt.savefig(RESULTS_DIR / 'eda_08_temporal.png')\n"
        "    plt.show()\n"
        "else:\n"
        "    print('No date column')"
    ),

    md("### 2.9 Source Distribution"),
    code(
        "plt.figure(figsize=(8, 4))\n"
        "df['source'].value_counts().plot(kind='bar', color='#8c564b')\n"
        "plt.title('Data Sources')\n"
        "plt.xticks(rotation=45)\n"
        "plt.savefig(RESULTS_DIR / 'eda_09_sources.png')\n"
        "plt.show()"
    ),

    md("### 2.10 Documents per Source"),
    code(
        "plt.figure(figsize=(8, 4))\n"
        "df.groupby('source')['doc_id'].nunique().plot(kind='bar', color='#e377c2')\n"
        "plt.title('Unique Documents per Source')\n"
        "plt.xticks(rotation=45)\n"
        "plt.savefig(RESULTS_DIR / 'eda_10_docs_per_source.png')\n"
        "plt.show()"
    ),

    md("### 2.11 Top Entities"),
    code(
        "plt.figure(figsize=(10, 4))\n"
        "df['entities'].dropna().astype(str).str.split(';').explode().str.strip() \\\n"
        "    .value_counts().head(10).plot(kind='barh', color='#bcbd22')\n"
        "plt.title('Top Entities (Actors/Malware)')\n"
        "plt.gca().invert_yaxis()\n"
        "plt.savefig(RESULTS_DIR / 'eda_11_entities.png')\n"
        "plt.show()"
    ),

    md("### 2.12 Length by Source"),
    code(
        "plt.figure(figsize=(10, 4))\n"
        "sns.boxplot(x='source', y='length', data=df, palette='Set2')\n"
        "plt.xticks(rotation=45)\n"
        "plt.title('Sentence Length by Source')\n"
        "plt.savefig(RESULTS_DIR / 'eda_12_length_source.png')\n"
        "plt.show()"
    ),

    md("### 2.13 Zero-Day Mentions Over Time"),
    code(
        "if 'date' in df.columns:\n"
        "    zd = df[df['zero_day'] == 'yes'].copy()\n"
        "    if len(zd):\n"
        "        plt.figure(figsize=(12, 4))\n"
        "        pd.to_datetime(zd['date']).dt.to_period('M').value_counts().sort_index().plot(\n"
        "            kind='bar', color='#d62728')\n"
        "        plt.title('Zero-Day Mentions per Month')\n"
        "        plt.xticks(rotation=45)\n"
        "        plt.tight_layout()\n"
        "        plt.savefig(RESULTS_DIR / 'eda_13_zeroday_time.png')\n"
        "        plt.show()\n"
        "    else:\n"
        "        print('No zero-day data')"
    ),

    md("### 2.14 Severity by Year"),
    code(
        "if 'date' in df.columns:\n"
        "    df['year'] = pd.to_datetime(df['date']).dt.year\n"
        "    plt.figure(figsize=(10, 4))\n"
        "    sns.countplot(data=df, x='year', hue='severity', palette='viridis')\n"
        "    plt.title('Severity Distribution by Year')\n"
        "    plt.savefig(RESULTS_DIR / 'eda_14_severity_year.png')\n"
        "    plt.show()"
    ),

    md("### 2.15 Relevance by Tactic"),
    code(
        "if df['tactic'].notna().any():\n"
        "    plt.figure(figsize=(10, 4))\n"
        "    df.groupby('tactic')['relevant'].apply(lambda x: (x == 'yes').mean()).sort_values().plot(\n"
        "        kind='barh', color='#2ca02c')\n"
        "    plt.title('Relevance Ratio by Tactic')\n"
        "    plt.savefig(RESULTS_DIR / 'eda_15_tactic_rel.png')\n"
        "    plt.show()"
    ),

    md("### 2.16 Top Words Relevant vs Irrelevant"),
    code(
        "from sklearn.feature_extraction.text import CountVectorizer\n"
        "cv = CountVectorizer(stop_words='english', max_features=20)\n"
        "rel = df[df['relevant'] == 'yes']['sentence_text'].astype(str)\n"
        "irr = df[df['relevant'] == 'no']['sentence_text'].astype(str)\n"
        "cv.fit(pd.concat([rel, irr]))\n"
        "rel_words = pd.Series(cv.transform(rel).sum(axis=0).A1, index=cv.get_feature_names_out()).sort_values(ascending=False).head(10)\n"
        "irr_words = pd.Series(cv.transform(irr).sum(axis=0).A1, index=cv.get_feature_names_out()).sort_values(ascending=False).head(10)\n"
        "\n"
        "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n"
        "rel_words.plot(kind='barh', ax=axes[0], color='#2ca02c'); axes[0].set_title('Relevant')\n"
        "irr_words.plot(kind='barh', ax=axes[1], color='#d62728'); axes[1].set_title('Irrelevant')\n"
        "plt.tight_layout()\n"
        "plt.savefig(RESULTS_DIR / 'eda_16_words.png')\n"
        "plt.show()"
    ),

    md("### 2.17 Topic Cluster Distribution"),
    code(
        "plt.figure(figsize=(10, 4))\n"
        "df['topic_cluster'].value_counts().head(10).plot(kind='barh', color='#1f77b4')\n"
        "plt.title('Top 10 Topic Clusters')\n"
        "plt.savefig(RESULTS_DIR / 'eda_17_topics.png')\n"
        "plt.show()"
    ),

    md("### 2.18 Sentences per Document"),
    code(
        "plt.figure(figsize=(8, 4))\n"
        "df.groupby('doc_id').size().plot(kind='hist', bins=30, color='#ff7f0e')\n"
        "plt.title('Sentences per Document')\n"
        "plt.savefig(RESULTS_DIR / 'eda_18_docs.png')\n"
        "plt.show()"
    ),

    md("### 2.19 Tactic vs Technique Heatmap"),
    code(
        "if df['tactic'].notna().any() and df['techniques'].notna().any():\n"
        "    pivot = df.groupby(['tactic', 'techniques']).size().unstack(fill_value=0)\n"
        "    plt.figure(figsize=(12, 6))\n"
        "    sns.heatmap(pivot.iloc[:10, :10], cmap='Blues')\n"
        "    plt.title('Tactic vs Technique Co-occurrence')\n"
        "    plt.savefig(RESULTS_DIR / 'eda_19_heatmap.png')\n"
        "    plt.show()\n"
        "else:\n"
        "    print('Insufficient tactic/technique data')"
    ),

    md("### 2.20 Correlation"),
    code(
        "num_cols = [c for c in ['length', 'page_number', 'sentence_id'] if c in df]\n"
        "plt.figure(figsize=(6, 4))\n"
        "sns.heatmap(df[num_cols].corr(), annot=True, cmap='coolwarm')\n"
        "plt.title('Numeric Feature Correlation')\n"
        "plt.savefig(RESULTS_DIR / 'eda_20_correlation.png')\n"
        "plt.show()"
    ),

    # ============ CLASSIFICATION ============
    md("## 3. Relevance Classifier"),
    code(
        "df = df.dropna(subset=['sentence_text', 'relevant']).copy()\n"
        "df['label'] = (df['relevant'].str.lower() == 'yes').astype(int)\n"
        "X_text = df['sentence_text'].astype(str).tolist()\n"
        "y = df['label'].values\n"
        "groups = df['doc_id'].values\n"
        "\n"
        "gss = GroupShuffleSplit(n_splits=N_SPLITS, test_size=0.2, random_state=RANDOM_SEED)\n"
        "splits = list(gss.split(X_text, y, groups))\n"
        "\n"
        "from sentence_transformers import SentenceTransformer\n"
        "sbert = SentenceTransformer('all-MiniLM-L6-v2')\n"
        "X_sbert = sbert.encode(X_text, batch_size=64, show_progress_bar=True, normalize_embeddings=True)\n"
        "\n"
        "def eval_splits(name, fit_predict_fn):\n"
        "    rows = []\n"
        "    for tr, te in splits:\n"
        "        pred = fit_predict_fn(tr, te)\n"
        "        p, r, f, _ = precision_recall_fscore_support(y[te], pred, average='macro', zero_division=0)\n"
        "        rows.append(dict(acc=accuracy_score(y[te], pred), f1=f))\n"
        "    res = pd.DataFrame(rows)\n"
        "    print(f'{name:12s} acc={res.acc.mean()*100:.1f}%  f1={res.f1.mean()*100:.1f}%')\n"
        "    return res\n"
        "\n"
        "def fit_base(tr, te):\n"
        "    v = TfidfVectorizer(ngram_range=(1,2)).fit([X_text[i] for i in tr])\n"
        "    c = LinearSVC(class_weight='balanced', random_state=42).fit(v.transform([X_text[i] for i in tr]), y[tr])\n"
        "    return c.predict(v.transform([X_text[i] for i in te]))\n"
        "\n"
        "def fit_sbert(tr, te):\n"
        "    c = LinearSVC(class_weight='balanced', C=2.0, random_state=42).fit(X_sbert[tr], y[tr])\n"
        "    return c.predict(X_sbert[te])\n"
        "\n"
        "res_base = eval_splits('Baseline', fit_base)\n"
        "res_sbert = eval_splits('SBERT', fit_sbert)\n"
        "\n"
        "import joblib\n"
        "clf_final = LinearSVC(class_weight='balanced', C=2.0, random_state=42).fit(X_sbert, y)\n"
        "joblib.dump(clf_final, CLS_DIR / 'sbert_relevance.joblib')\n"
        "print('saved relevance model')"
    ),

    md("## 4. Technique Classifier"),
    code(
        "mask = df['relevant'].eq('yes') & df['techniques'].fillna('').str.strip().ne('')\n"
        "df_tech = df[mask].copy()\n"
        "df_tech['technique'] = df_tech['techniques'].str.split(',').str[0].str.strip()\n"
        "X_tech = sbert.encode(df_tech['sentence_text'].tolist(), normalize_embeddings=True)\n"
        "y_tech = df_tech['technique'].values\n"
        "\n"
        "clf_tech = LinearSVC(class_weight='balanced', C=2.0, random_state=42).fit(X_tech, y_tech)\n"
        "joblib.dump(clf_tech, CLS_DIR / 'sbert_technique.joblib')\n"
        "print(f'Saved technique classifier ({len(set(y_tech))} classes)')"
    ),

    # ============ BERTopic ============
    md("## 5. Emerging Threat Model (BERTopic)"),
    code(
        "from bertopic import BERTopic\n"
        "from umap import UMAP\n"
        "from hdbscan import HDBSCAN\n"
        "\n"
        "pos_df = df[df['label'] == 1].copy()\n"
        "docs = pos_df['sentence_text'].tolist()\n"
        "\n"
        "hdbscan_model = HDBSCAN(min_cluster_size=10, metric='euclidean', prediction_data=True)\n"
        "topic_model = BERTopic(\n"
        "    embedding_model=sbert,\n"
        "    umap_model=UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42),\n"
        "    hdbscan_model=hdbscan_model,\n"
        "    min_topic_size=10, calculate_probabilities=False, verbose=False\n"
        ")\n"
        "topics, _ = topic_model.fit_transform(docs)\n"
        "pos_df['topic'] = topics\n"
        "topic_model.save(str(THREAT_DIR / 'bertopic_model'), serialization='pickle')\n"
        "print('Saved BERTopic model')"
    ),

    md("## 6. Temporal Trends & Emerging Rule"),
    code(
        "if 'pos_df' not in globals() or 'topic_model' not in globals():\n"
        "    print('Skipped: run BERTopic training cell first')\n"
        "elif 'date' in pos_df.columns:\n"
        "    pos_df['_date'] = pd.to_datetime(pos_df['date'], errors='coerce')\n"
        "    valid = pos_df.dropna(subset=['_date'])\n"
        "    if len(valid) == 0:\n"
        "        print('Skipped: no valid dates')\n"
        "    else:\n"
        "        tot = topic_model.topics_over_time(\n"
        "            docs=valid['sentence_text'].tolist(),\n"
        "            timestamps=valid['_date'].tolist(),\n"
        "            nr_bins=12,\n"
        "        )\n"
        "        tot.to_csv(RESULTS_DIR / 'topic_trends.csv', index=False)\n"
        "        print(f'wrote {RESULTS_DIR / \"topic_trends.csv\"} ({len(tot)} rows)')\n"
        "\n"
        "        flags = []\n"
        "        for tid, g in tot.groupby('Topic'):\n"
        "            g = g.sort_values('Timestamp')\n"
        "            shares = g['Frequency'].values\n"
        "            run = best = 0\n"
        "            for i in range(1, len(shares)):\n"
        "                run = run + 1 if shares[i] > shares[i - 1] else 0\n"
        "                best = max(best, run)\n"
        "            docs_n = pos_df.loc[pos_df['topic'] == tid, 'doc_id'].nunique()\n"
        "            if best >= 3 and docs_n >= 3:\n"
        "                flags.append({'topic': int(tid), 'max_rise_run': int(best), 'distinct_docs': int(docs_n)})\n"
        "\n"
        "        if flags:\n"
        "            pd.DataFrame(flags).to_csv(RESULTS_DIR / 'emerging_topics.csv', index=False)\n"
        "            print('Emerging threats detected:')\n"
        "            print(pd.DataFrame(flags))\n"
        "        else:\n"
        "            print('No topics met the emerging rule')"
    ),

    # ============ PDF ANALYSIS ============
    md("## 7. PDF Utilities & Analysis"),
    code(
        "import pymupdf\n"
        "def extract_text_from_pdf(path):\n"
        "    if not path.exists() or path.stat().st_size == 0:\n"
        "        return '', 'Invalid'\n"
        "    try:\n"
        "        doc = pymupdf.open(str(path))\n"
        "        text = '\\n'.join(p.get_text() for p in doc)\n"
        "        doc.close()\n"
        "        return text, None\n"
        "    except Exception as e:\n"
        "        return '', str(e)\n"
        "\n"
        "def split_sentences(text):\n"
        "    text = re.sub(r'\\s+', ' ', text).strip()\n"
        "    return [p.strip() for p in re.split(r'(?<=[.!?])\\s+', text) if len(p.split()) >= 4]\n"
        "\n"
        "valid_pdfs = [p for p in PDF_DIR.glob('*.pdf') if p.stat().st_size > 0]\n"
        "rows = []\n"
        "single_payload = None\n"
        "\n"
        "for p in valid_pdfs[:10]:\n"
        "    text, err = extract_text_from_pdf(p)\n"
        "    if err: continue\n"
        "    sents = split_sentences(text)\n"
        "    if not sents: continue\n"
        "    emb = sbert.encode(sents, normalize_embeddings=True)\n"
        "    rel_pred = clf_final.predict(emb)\n"
        "    rel_sents = [s for s, r in zip(sents, rel_pred) if r == 1]\n"
        "    tech_counts = {}\n"
        "    top_tech = None\n"
        "    if rel_sents:\n"
        "        rel_emb = sbert.encode(rel_sents, normalize_embeddings=True)\n"
        "        tech_preds = clf_tech.predict(rel_emb)\n"
        "        c = Counter(tech_preds)\n"
        "        tech_counts = {str(k): int(v) for k, v in c.items()}\n"
        "        top_tech = c.most_common(1)[0][0]\n"
        "    rows.append(dict(file=p.name, n_sentences=len(sents), n_relevant=len(rel_sents), top_tech=top_tech))\n"
        "    if not single_payload:\n"
        "        single_payload = dict(file=p.name, n_sentences=len(sents), n_relevant=len(rel_sents),\n"
        "                              top_tech=top_tech, tech_counts=tech_counts, sents=rel_sents[:5])\n"
        "\n"
        "pd.DataFrame(rows).to_csv(RESULTS_DIR / 'batch_predictions.csv', index=False)\n"
        "with open(RESULTS_DIR / 'demo_single_report.json', 'w') as f:\n"
        "    json.dump(single_payload, f, indent=2)\n"
        "print('Saved PDF batch predictions and JSON demo')"
    ),

    # ============ SUMMARY ============
    md("## 8. Plain-English Summary"),
    code(
        "print('=' * 60)\n"
        "print('FINAL SUMMARY')\n"
        "print('=' * 60)\n"
        "print(f'1) DATA: {len(df)} sentences from {df[\"doc_id\"].nunique()} documents')\n"
        "print(f'   Relevant: {df[\"label\"].sum()} | Irrelevant: {len(df) - df[\"label\"].sum()}')\n"
        "print()\n"
        "print('2) RELEVANCE CLASSIFIER')\n"
        "print(f'   TF-IDF Baseline: {res_base.acc.mean()*100:.1f}%')\n"
        "print(f'   SBERT Proposed : {res_sbert.acc.mean()*100:.1f}%')\n"
        "print()\n"
        "print('3) TECHNIQUE CLASSIFIER')\n"
        "print(f'   Classes: {len(set(y_tech))}')\n"
        "print(f'   Examples: {list(set(y_tech))[:3]}')\n"
        "print()\n"
        "print('4) EMERGING THREAT MODEL')\n"
        "print(f'   Topics discovered: {len(topic_model.get_topic_info()) - 1}')\n"
        "print()\n"
        "print('5) PDF ANALYSIS')\n"
        "print(f'   Processed {len(rows)} PDFs')\n"
        "for r in rows[:3]:\n"
        "    print(f'   - {r[\"file\"]}: {r[\"n_sentences\"]} sents, {r[\"n_relevant\"]} relevant, Top: {r[\"top_tech\"]}')"
    ),
]


def main():
    nb = nbf.v4.new_notebook()
    nb["cells"] = CELLS
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    }
    here = Path(__file__).resolve().parent
    root = here.parent if here.name == "scripts" else here
    out = root / "notebooks" / "cti.ipynb"
    out.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()