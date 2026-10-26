# scripts/make_03_technique_xgboost.py
"""Generate notebooks/03_technique_classifier.ipynb — XGBoost hybrid + UI simulation."""
from pathlib import Path
import nbformat as nbf


def md(t): return nbf.v4.new_markdown_cell(t)
def code(t): return nbf.v4.new_code_cell(t)


CELLS = [

    md(
        "# Notebook 3: Technique Classifier (XGBoost + Hybrid Features)\n"
        "\n"
        "**Goal:** Reach 85–90% accuracy on MITRE ATT&CK technique classification.\n"
        "\n"
        "**Strategy:**\n"
        "1. Map 18 parent techniques → 6 broad categories\n"
        "2. Augment with keyword-labeled ThreatCluster CVEs\n"
        "3. Hybrid features: TF-IDF bigrams + Sentence-BERT embeddings\n"
        "4. XGBoost classifier\n"
        "5. Live inference demo (mimics the web UI)\n"
        "\n"
        "**Input:** `dataset/cti_dataset_clean.csv`\n"
        "**Outputs:**\n"
        "- `models/classification/sbert_technique.joblib`\n"
        "- `models/classification/technique_config.json`\n"
        "- `results/03_technique_classifier/` charts and tables"
    ),

    md("## Step 0 — Setup"),
    code(
        "import warnings, time, json, joblib\n"
        "from pathlib import Path\n"
        "warnings.filterwarnings('ignore')\n"
        "\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n"
        "\n"
        "from scipy.sparse import hstack, csr_matrix\n"
        "from sklearn.feature_extraction.text import TfidfVectorizer\n"
        "from sklearn.preprocessing import StandardScaler\n"
        "from sklearn.model_selection import StratifiedShuffleSplit\n"
        "from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix\n"
        "\n"
        "PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n"
        "DATA_PATH    = PROJECT_ROOT / 'dataset' / 'cti_dataset_clean.csv'\n"
        "CLS_DIR      = PROJECT_ROOT / 'models' / 'classification'\n"
        "RESULTS_DIR  = PROJECT_ROOT / 'results'\n"
        "TECH_DIR     = RESULTS_DIR / '03_technique_classifier'\n"
        "for d in (CLS_DIR, RESULTS_DIR, TECH_DIR):\n"
        "    d.mkdir(parents=True, exist_ok=True)\n"
        "RANDOM_SEED = 42\n"
        "sns.set_theme(style='whitegrid')\n"
        "print('Setup complete.')"
    ),

    md("## Step 1 — Load Clean Data"),
    code(
        "if not DATA_PATH.exists():\n"
        "    raise FileNotFoundError(f'{DATA_PATH} not found. Run Notebook 01 first.')\n"
        "df = pd.read_csv(DATA_PATH)\n"
        "print(f'Loaded {len(df):,} rows from {df[\"doc_id\"].nunique():,} documents')"
    ),

    md("## Step 2 — Map 18 Techniques → 6 Categories"),
    code(
        "TECH_TO_CATEGORY = {\n"
        "    # Initial Access\n"
        "    'T1190': 'Initial Access', 'T1189': 'Initial Access', 'T1078': 'Initial Access',\n"
        "    'T1133': 'Initial Access', 'T1566': 'Initial Access',\n"
        "    # Execution\n"
        "    'T1059': 'Execution', 'T1203': 'Execution', 'T1204': 'Execution', 'T1047': 'Execution',\n"
        "    # Privilege Escalation\n"
        "    'T1068': 'Privilege Escalation', 'T1548': 'Privilege Escalation',\n"
        "    # Defense Evasion\n"
        "    'T1055': 'Defense Evasion', 'T1027': 'Defense Evasion', 'T1036': 'Defense Evasion',\n"
        "    'T1574': 'Defense Evasion',\n"
        "    # Credential Access\n"
        "    'T1003': 'Credential Access', 'T1110': 'Credential Access', 'T1552': 'Credential Access',\n"
        "    # Collection / Impact\n"
        "    'T1005': 'Impact', 'T1499': 'Impact', 'T1486': 'Impact', 'T1485': 'Impact',\n"
        "    'T1490': 'Impact', 'T1041': 'Impact', 'T1565': 'Impact',\n"
        "    # Lateral Movement\n"
        "    'T1021': 'Lateral Movement', 'T1091': 'Lateral Movement',\n"
        "    # Resource Development\n"
        "    'T1608': 'Resource Development', 'T1105': 'Resource Development',\n"
        "    'T1211': 'Resource Development',\n"
        "    # Discovery\n"
        "    'T1083': 'Discovery', 'T1040': 'Discovery',\n"
        "}\n"
        "\n"
        "mask = (df['relevant'].str.lower() == 'yes') & \\\n"
        "       (df['techniques'].fillna('').astype(str).str.strip() != '')\n"
        "df_tech = df[mask].copy()\n"
        "df_tech['parent'] = (\n"
        "    df_tech['techniques']\n"
        "    .astype(str).str.split(',').str[0].str.strip().str.split('.').str[0]\n"
        ")\n"
        "df_tech['category'] = df_tech['parent'].map(TECH_TO_CATEGORY)\n"
        "df_tech = df_tech[df_tech['category'].notna()].copy()\n"
        "\n"
        "print(f'Technique-labeled rows : {len(df_tech):,}')\n"
        "print(f'Distinct parents       : {df_tech[\"parent\"].nunique()}')\n"
        "print(f'Distinct categories    : {df_tech[\"category\"].nunique()}')\n"
        "print()\n"
        "print('Category distribution:')\n"
        "print(df_tech['category'].value_counts().to_string())"
    ),

    md("## Step 3 — Augment with Keyword Labels from ThreatCluster"),
    code(
        "KEYWORD_TO_CATEGORY = {\n"
        "    # Initial Access\n"
        "    'phishing': 'Initial Access', 'spear-phishing': 'Initial Access',\n"
        "    'exploit public': 'Initial Access', 'drive-by': 'Initial Access',\n"
        "    'valid account': 'Initial Access', 'stolen credential': 'Initial Access',\n"
        "    'external remote': 'Initial Access',\n"
        "    # Execution\n"
        "    'powershell': 'Execution', 'command line': 'Execution',\n"
        "    'script execution': 'Execution', 'wmi ': 'Execution',\n"
        "    'cmd.exe': 'Execution', 'batch script': 'Execution',\n"
        "    # Privilege Escalation\n"
        "    'privilege escalation': 'Privilege Escalation',\n"
        "    'elevation of privilege': 'Privilege Escalation',\n"
        "    'gain root': 'Privilege Escalation', 'privilege management': 'Privilege Escalation',\n"
        "    # Defense Evasion\n"
        "    'obfuscat': 'Defense Evasion', 'disable security': 'Defense Evasion',\n"
        "    'evasion': 'Defense Evasion', 'code signing': 'Defense Evasion',\n"
        "    'process injection': 'Defense Evasion',\n"
        "    # Credential Access\n"
        "    'credential dump': 'Credential Access', 'lsass': 'Credential Access',\n"
        "    'brute force': 'Credential Access', 'password hash': 'Credential Access',\n"
        "    'kerberoast': 'Credential Access',\n"
        "    # Impact\n"
        "    'ransomware': 'Impact', 'encrypted for impact': 'Impact',\n"
        "    'data destruction': 'Impact', 'denial of service': 'Impact',\n"
        "    'data encrypted': 'Impact',\n"
        "    # Lateral Movement\n"
        "    'lateral movement': 'Lateral Movement', 'remote service': 'Lateral Movement',\n"
        "    'smb ': 'Lateral Movement', 'rdp ': 'Lateral Movement',\n"
        "}\n"
        "\n"
        "def category_from_text(text):\n"
        "    t = str(text).lower()\n"
        "    for kw, cat in KEYWORD_TO_CATEGORY.items():\n"
        "        if kw in t:\n"
        "            return cat\n"
        "    return None\n"
        "\n"
        "# Take all positive rows WITHOUT technique labels and try to auto-label them\n"
        "df_pos = df[(df['relevant'].str.lower() == 'yes') &\n"
        "            (df['techniques'].fillna('').astype(str).str.strip() == '')].copy()\n"
        "df_pos['auto_category'] = df_pos['sentence_text'].apply(category_from_text)\n"
        "df_aug = df_pos[df_pos['auto_category'].notna()].copy()\n"
        "\n"
        "print(f'Unlabeled positives          : {len(df_pos):,}')\n"
        "print(f'Auto-labeled via keywords    : {len(df_aug):,}')\n"
        "print(f'CIRCL-labeled (original)     : {len(df_tech):,}')\n"
        "print(f'Total training rows          : {len(df_tech) + len(df_aug):,}')"
    ),

    md("## Step 4 — Combine Labeled + Augmented Rows"),
    code(
        "df_circl = df_tech[['sentence_text', 'category', 'parent', 'doc_id']].copy()\n"
        "df_circl['label_source'] = 'circl_expert'\n"
        "\n"
        "df_augmented = df_aug[['sentence_text', 'auto_category', 'doc_id']].rename(\n"
        "    columns={'auto_category': 'category'}\n"
        ")\n"
        "df_augmented['parent'] = None\n"
        "df_augmented['label_source'] = 'keyword_auto'\n"
        "\n"
        "df_train = pd.concat([df_circl, df_augmented], ignore_index=True)\n"
        "df_train = df_train.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)\n"
        "\n"
        "print(f'Total training rows: {len(df_train):,}')\n"
        "print()\n"
        "print('Category distribution:')\n"
        "print(df_train['category'].value_counts().to_string())\n"
        "print()\n"
        "print('By label source:')\n"
        "print(df_train.groupby(['category', 'label_source']).size().unstack(fill_value=0))"
    ),

    md("## Step 5 — Filter Rare Categories"),
    code(
        "MIN_CAT_ROWS = 50\n"
        "counts = df_train['category'].value_counts()\n"
        "keep = counts[counts >= MIN_CAT_ROWS].index\n"
        "df_train = df_train[df_train['category'].isin(keep)].copy()\n"
        "print(f'Categories kept: {df_train[\"category\"].nunique()} of {len(counts)}')\n"
        "print(f'Rows remaining: {len(df_train):,}')\n"
        "print()\n"
        "print(df_train['category'].value_counts().to_string())"
    ),

    md("## Step 6 — Hybrid Features (TF-IDF + SBERT)"),
    code(
        "from sentence_transformers import SentenceTransformer\n"
        "sbert = SentenceTransformer('all-MiniLM-L6-v2')\n"
        "\n"
        "texts = df_train['sentence_text'].tolist()\n"
        "y = df_train['category'].values\n"
        "\n"
        "# SBERT embeddings\n"
        "t0 = time.perf_counter()\n"
        "X_sbert = sbert.encode(texts, batch_size=64, show_progress_bar=True,\n"
        "                       normalize_embeddings=True)\n"
        "print(f'SBERT encoded {len(X_sbert)} sentences in {time.perf_counter()-t0:.1f}s')\n"
        "\n"
        "# TF-IDF bigrams\n"
        "tfidf = TfidfVectorizer(\n"
        "    ngram_range=(1, 2), max_features=5000, min_df=2, sublinear_tf=True\n"
        ")\n"
        "X_tfidf = tfidf.fit_transform(texts)\n"
        "print(f'TF-IDF: {X_tfidf.shape}')\n"
        "\n"
        "# Hybrid: TF-IDF sparse + scaled SBERT dense\n"
        "scaler = StandardScaler(with_mean=False).fit(X_sbert)\n"
        "X_hybrid = hstack([X_tfidf, csr_matrix(scaler.transform(X_sbert))]).tocsr()\n"
        "print(f'Hybrid: {X_hybrid.shape}')"
    ),

    md("## Step 7 — Train/Test Split"),
    code(
        "sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_SEED)\n"
        "tr, te = next(sss.split(X_hybrid, y))\n"
        "print(f'Train: {len(tr):,} | Test: {len(te):,}')\n"
        "print(f'Train classes: {sorted(set(y[tr]))}')\n"
        "print(f'Test classes : {sorted(set(y[te]))}')"
    ),

    md("## Step 8 — Train XGBoost"),
    code(
        "from xgboost import XGBClassifier\n"
        "\n"
        "t0 = time.perf_counter()\n"
        "clf = XGBClassifier(\n"
        "    n_estimators=300,\n"
        "    max_depth=6,\n"
        "    learning_rate=0.1,\n"
        "    subsample=0.8,\n"
        "    colsample_bytree=0.8,\n"
        "    use_label_encoder=False,\n"
        "    eval_metric='mlogloss',\n"
        "    random_state=RANDOM_SEED,\n"
        "    n_jobs=-1,\n"
        ")\n"
        "clf.fit(X_hybrid[tr], y[tr])\n"
        "train_t = time.perf_counter() - t0\n"
        "\n"
        "t0 = time.perf_counter()\n"
        "pred = clf.predict(X_hybrid[te])\n"
        "infer_t = time.perf_counter() - t0\n"
        "\n"
        "acc = accuracy_score(y[te], pred)\n"
        "macro_f1 = f1_score(y[te], pred, average='macro', zero_division=0)\n"
        "weighted_f1 = f1_score(y[te], pred, average='weighted', zero_division=0)\n"
        "\n"
        "print(f'Training time : {train_t:.2f}s')\n"
        "print(f'Inference time: {infer_t:.3f}s')\n"
        "print()\n"
        "print(f'Accuracy    : {acc:.4f}  ({acc*100:.1f}%)')\n"
        "print(f'Macro-F1    : {macro_f1:.4f}')\n"
        "print(f'Weighted-F1 : {weighted_f1:.4f}')"
    ),

    md("## Step 9 — Per-Class Report"),
    code(
        "print(classification_report(y[te], pred, zero_division=0))"
    ),

    md("## Step 10 — Confusion Matrix"),
    code(
        "labels = sorted(set(y))\n"
        "cm = confusion_matrix(y[te], pred, labels=labels)\n"
        "\n"
        "plt.figure(figsize=(9, 7))\n"
        "sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',\n"
        "            xticklabels=labels, yticklabels=labels, cbar=False)\n"
        "plt.title(f'XGBoost Hybrid — Accuracy {acc*100:.1f}%')\n"
        "plt.ylabel('True')\n"
        "plt.xlabel('Predicted')\n"
        "plt.xticks(rotation=45, ha='right')\n"
        "plt.yticks(rotation=0)\n"
        "plt.tight_layout()\n"
        "plt.savefig(TECH_DIR / 'confusion_matrix.png', dpi=120)\n"
        "plt.show()"
    ),

    md("## Step 11 — Compare to Base Paper"),
    code(
        "paper_results = {\n"
        "    'Paper (SVM, synthetic)':  96.35,\n"
        "    'Paper (XGBoost, synthetic)': 91.13,\n"
        "    'Our XGBoost (real, 6-cat)':  round(acc * 100, 2),\n"
        "}\n"
        "for k, v in paper_results.items():\n"
        "    print(f'{k:32s} : {v:.2f}%')\n"
        "\n"
        "plt.figure(figsize=(8, 4))\n"
        "names = list(paper_results.keys())\n"
        "vals = list(paper_results.values())\n"
        "colors = ['#888888', '#888888', '#2ca02c']\n"
        "plt.barh(names, vals, color=colors)\n"
        "for i, v in enumerate(vals):\n"
        "    plt.text(v + 1, i, f'{v:.1f}%', va='center')\n"
        "plt.xlim(0, 105)\n"
        "plt.title('Base Paper vs Our Reproduction (6 Categories)')\n"
        "plt.tight_layout()\n"
        "plt.savefig(TECH_DIR / 'paper_comparison.png', dpi=120)\n"
        "plt.show()"
    ),

    md("## Step 12 — Save Results + Model + Config"),
    code(
        "# Save per-class report\n"
        "report = classification_report(y[te], pred, output_dict=True, zero_division=0)\n"
        "rows = [{'class': k, **v} for k, v in report.items() if isinstance(v, dict)]\n"
        "pd.DataFrame(rows).to_csv(TECH_DIR / 'technique_results.csv', index=False)\n"
        "\n"
        "# Save model bundle (XGBoost + vectorizer + scaler)\n"
        "bundle = {\n"
        "    'classifier': clf,\n"
        "    'tfidf': tfidf,\n"
        "    'scaler': scaler,\n"
        "    'labels': labels,\n"
        "}\n"
        "joblib.dump(bundle, CLS_DIR / 'technique_bundle.joblib')\n"
        "\n"
        "# Save config\n"
        "config = {\n"
        "    'embedding_model': 'all-MiniLM-L6-v2',\n"
        "    'embedding_dim': 384,\n"
        "    'classifier': 'XGBClassifier',\n"
        "    'n_estimators': 300,\n"
        "    'max_depth': 6,\n"
        "    'learning_rate': 0.1,\n"
        "    'feature_type': 'hybrid_tfidf_sbert',\n"
        "    'tfidf_max_features': 5000,\n"
        "    'trained_on_rows': int(len(X_hybrid)),\n"
        "    'n_classes': int(len(labels)),\n"
        "    'labels': list(labels),\n"
        "    'accuracy': round(float(acc), 4),\n"
        "    'macro_f1': round(float(macro_f1), 4),\n"
        "    'weighted_f1': round(float(weighted_f1), 4),\n"
        "}\n"
        "with open(CLS_DIR / 'technique_config.json', 'w') as f:\n"
        "    json.dump(config, f, indent=2)\n"
        "\n"
        "print('Saved:')\n"
        "for f in ['technique_bundle.joblib', 'technique_config.json']:\n"
        "    print(f'  {CLS_DIR / f}')\n"
        "print(f'  {TECH_DIR / \"technique_results.csv\"}')\n"
        "print(f'  {TECH_DIR / \"confusion_matrix.png\"}')\n"
        "print(f'  {TECH_DIR / \"paper_comparison.png\"}')"
    ),

    # ============================================================
    # PART B — UI SIMULATION
    # ============================================================
    md(
        "---\n"
        "# Part B: Live Inference Demo (What the UI Does Behind the Scenes)\n"
        "\n"
        "When a user uploads a PDF to the web app:\n"
        "1. **Backend** extracts text from PDF\n"
        "2. **Sentence splitter** divides into sentences\n"
        "3. **Relevance classifier** marks relevant vs irrelevant\n"
        "4. **Technique classifier** (this notebook's model) predicts category\n"
        "5. **Topic model** (Notebook 04) discovers themes\n"
        "6. **Frontend** renders the results\n"
        "\n"
        "The cells below simulate this exact pipeline on a sample text — "
        "showing what the API endpoint `/analyse` does internally."
    ),

    md("## Step 13 — Load All Three Models"),
    code(
        "# Relevance classifier (from Notebook 03 alt / or separate training)\n"
        "relevance_path = CLS_DIR / 'sbert_relevance.joblib'\n"
        "relevance_model = joblib.load(relevance_path) if relevance_path.exists() else None\n"
        "\n"
        "# Technique classifier (this notebook)\n"
        "technique_bundle = joblib.load(CLS_DIR / 'technique_bundle.joblib')\n"
        "\n"
        "print(f'Relevance model: {\"loaded\" if relevance_model else \"NOT FOUND — will skip relevance filtering\"}')\n"
        "print(f'Technique model: loaded ({len(technique_bundle[\"labels\"])} categories)')\n"
        "print()\n"
        "print(f'Categories: {technique_bundle[\"labels\"]}')"
    ),

    md("## Step 14 — Sample Input (Simulates a PDF Upload)"),
    code(
        "# This is what the UI sends to /analyse after extracting PDF text\n"
        "sample_pdf_text = \"\"\"\n"
        "Critical Advisory: New ransomware campaign exploiting CVE-2023-12345. The threat actor\n"
        "used a spear-phishing email to deliver a malicious attachment. Once executed, the malware\n"
        "performed credential dumping from LSASS memory and moved laterally via SMB. The actor\n"
        "then deployed ransomware that encrypted production servers, demanding payment in Bitcoin.\n"
        "Indicators of compromise include malicious-domain.example and IP 192.0.2.55.\n"
        "\"\"\"\n"
        "print('Sample input:')\n"
        "print(sample_pdf_text)"
    ),

    md("## Step 15 — Sentence Splitting (Backend Preprocessing)"),
    code(
        "import re\n"
        "\n"
        "def split_sentences(text):\n"
        "    text = re.sub(r'\\s+', ' ', text).strip()\n"
        "    parts = re.split(r'(?<=[.!?])\\s+', text)\n"
        "    return [p.strip() for p in parts if len(p.split()) >= 4]\n"
        "\n"
        "sentences = split_sentences(sample_pdf_text)\n"
        "print(f'Split into {len(sentences)} sentences:')\n"
        "for i, s in enumerate(sentences, 1):\n"
        "    print(f'  [{i}] {s[:100]}...')"
    ),

    md("## Step 16 — Relevance Filter"),
    code(
        "if relevance_model is not None:\n"
        "    emb = sbert.encode(sentences, normalize_embeddings=True)\n"
        "    rel_pred = relevance_model.predict(emb)\n"
        "    relevant_sents = [s for s, r in zip(sentences, rel_pred) if r == 1]\n"
        "    print(f'Relevant: {len(relevant_sents)} / {len(sentences)}')\n"
        "else:\n"
        "    # No relevance filter — treat all as relevant for demo\n"
        "    relevant_sents = sentences\n"
        "    print(f'Relevance model not loaded — treating all {len(sentences)} as relevant')\n"
        "\n"
        "print()\n"
        "for i, s in enumerate(relevant_sents, 1):\n"
        "    print(f'  [REL {i}] {s[:100]}...')"
    ),

    md("## Step 17 — Technique Classification (Core Inference)"),
    code(
        "def predict_category(sentences, bundle):\n"
        "    tfidf_obj = bundle['tfidf']\n"
        "    scaler_obj = bundle['scaler']\n"
        "    clf_obj = bundle['classifier']\n"
        "    sbert_vecs = sbert.encode(sentences, normalize_embeddings=True)\n"
        "    tfidf_vecs = tfidf_obj.transform(sentences)\n"
        "    hybrid_vecs = hstack([tfidf_vecs, csr_matrix(scaler_obj.transform(sbert_vecs))]).tocsr()\n"
        "    return clf_obj.predict(hybrid_vecs), sbert_vecs\n"
        "\n"
        "tech_preds, sbert_vecs = predict_category(relevant_sents, technique_bundle)\n"
        "\n"
        "print('Technique predictions per sentence:')\n"
        "for s, t in zip(relevant_sents, tech_preds):\n"
        "    print(f'  → {t:25s} | {s[:80]}...')"
    ),

    md("## Step 18 — Aggregate Results (What the UI Displays)"),
    code(
        "from collections import Counter\n"
        "\n"
        "tech_counts = Counter(tech_preds)\n"
        "top_tech = tech_counts.most_common(1)[0][0]\n"
        "\n"
        "print('=' * 55)\n"
        "print('UI DASHBOARD OUTPUT')\n"
        "print('=' * 55)\n"
        "print(f'File              : sample_cti.pdf')\n"
        "print(f'Sentences total   : {len(sentences)}')\n"
        "print(f'Relevant CTI      : {len(relevant_sents)}')\n"
        "print(f'Irrelevant        : {len(sentences) - len(relevant_sents)}')\n"
        "print(f'Top category      : {top_tech}')\n"
        "print()\n"
        "print('Category distribution:')\n"
        "for cat, count in tech_counts.most_common():\n"
        "    bar = '█' * count\n"
        "    print(f'  {cat:25s} {bar} ({count})')\n"
        "print('=' * 55)"
    ),

    md("## Step 19 — Category Distribution Chart (UI Bar Chart)"),
    code(
        "fig, ax = plt.subplots(figsize=(10, 4))\n"
        "cats = [c for c, _ in tech_counts.most_common()]\n"
        "vals = [n for _, n in tech_counts.most_common()]\n"
        "ax.barh(cats, vals, color='#3b82f6')\n"
        "ax.set_xlabel('Sentences')\n"
        "ax.set_title('Technique Category Distribution (from PDF)')\n"
        "ax.invert_yaxis()\n"
        "plt.tight_layout()\n"
        "plt.savefig(TECH_DIR / 'ui_demo_chart.png', dpi=120)\n"
        "plt.show()"
    ),

    md(
        "## Step 20 — What the Frontend Receives (JSON Preview)\n"
        "\n"
        "This is the **exact JSON** the FastAPI endpoint `/analyse` returns to the frontend:"
    ),
    code(
        "api_response = {\n"
        "    'file': 'sample_cti.pdf',\n"
        "    'n_sentences': len(sentences),\n"
        "    'n_relevant': len(relevant_sents),\n"
        "    'n_irrelevant': len(sentences) - len(relevant_sents),\n"
        "    'top_technique': top_tech,\n"
        "    'technique_counts': dict(tech_counts),\n"
        "    'top_relevant_sentences': relevant_sents[:5],\n"
        "}\n"
        "print(json.dumps(api_response, indent=2))"
    ),

    md(
        "## Step 21 — Summary\n"
        "\n"
        "**Part A** trained the technique classifier (XGBoost on hybrid features) and saved:\n"
        "- `models/classification/technique_bundle.joblib`\n"
        "- `models/classification/technique_config.json`\n"
        "\n"
        "**Part B** simulated what happens when a user uploads a PDF to the web app:\n"
        "1. Text extracted → split into sentences\n"
        "2. Relevance model filters noise\n"
        "3. Technique model predicts category\n"
        "4. Results aggregated → JSON returned to frontend\n"
        "5. Frontend renders bar chart + top technique + sample sentences\n"
        "\n"
        "**Your API (`server/app.py`) uses the same three models**:\n"
        "- `sbert_relevance.joblib` (from relevance training)\n"
        "- `technique_bundle.joblib` (from this notebook)\n"
        "- `bertopic_model/` (from Notebook 04)\n"
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
    out = root / "notebooks" / "03_technique_classifier.ipynb"
    out.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()