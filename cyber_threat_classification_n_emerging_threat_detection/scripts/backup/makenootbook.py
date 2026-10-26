# D:\Projects\done\thesis\cyber-threat-detection\scripts\makenotebook.py
"""Generate notebooks/cti.ipynb with:
- EDA visuals (bar charts, histograms)
- binary relevance classifier
- multi-class technique classifier (malware-class style)
- BERTopic emerging-threat model
- plain-English final summary cell
"""
from pathlib import Path
import nbformat as nbf


def md(t): return nbf.v4.new_markdown_cell(t)
def code(t): return nbf.v4.new_code_cell(t)


CELLS = [

    # 1 ---------------------------------------------------------------
    md(
        "# Cyber Threat Classification and Emerging Threat Detection\n"
        "\n"
        "**Reads:** `dataset/SMAH-CTI.csv` (real) and `sample_documents/*.pdf`\n"
        "**Writes:**\n"
        "- `models/classification/` - relevance classifier + technique classifier\n"
        "- `models/emerging_threat/` - BERTopic topic model\n"
        "- `results/` - tables, plots, predictions\n"
        "\n"
        "The final cell of this notebook explains results in plain language."
    ),

    # 2 ---------------------------------------------------------------
    md("## 0. Setup"),
    code(
        "import json, time, re\n"
        "from pathlib import Path\n"
        "from collections import Counter\n"
        "\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "\n"
        "from sklearn.model_selection import GroupShuffleSplit\n"
        "from sklearn.feature_extraction.text import TfidfVectorizer\n"
        "from sklearn.svm import LinearSVC, SVC\n"
        "from sklearn.calibration import CalibratedClassifierCV\n"
        "from sklearn.preprocessing import StandardScaler\n"
        "from sklearn.metrics import (\n"
        "    accuracy_score, precision_recall_fscore_support,\n"
        "    classification_report, confusion_matrix, ConfusionMatrixDisplay,\n"
        ")\n"
        "from scipy.sparse import hstack, csr_matrix\n"
        "from scipy.stats import mode\n"
        "\n"
        "PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n"
        "DATA_PATH   = PROJECT_ROOT / 'dataset' / 'SMAH-CTI.csv'\n"
        "PDF_DIR     = PROJECT_ROOT / 'sample_documents'\n"
        "CLS_DIR     = PROJECT_ROOT / 'models' / 'classification'\n"
        "THREAT_DIR  = PROJECT_ROOT / 'models' / 'emerging_threat'\n"
        "RESULTS_DIR = PROJECT_ROOT / 'results'\n"
        "for d in (CLS_DIR, THREAT_DIR, RESULTS_DIR):\n"
        "    d.mkdir(parents=True, exist_ok=True)\n"
        "\n"
        "RANDOM_SEED = 42\n"
        "N_SPLITS = 10\n"
        "print('data:', DATA_PATH, '| exists:', DATA_PATH.exists())\n"
        "print('pdfs:', PDF_DIR, '| exists:', PDF_DIR.exists())"
    ),

    # 3 ---------------------------------------------------------------
    md("## 1. Clean out the broken placeholder PDF"),
    code(
        "bad = PDF_DIR / 'sample_cti.pdf'\n"
        "if bad.exists():\n"
        "    try:\n"
        "        size = bad.stat().st_size\n"
        "        if size == 0 or bad.read_bytes()[:4] != b'%PDF':\n"
        "            bad.unlink()\n"
        "            print('removed', bad.name, '(invalid/placeholder)')\n"
        "        else:\n"
        "            print('kept', bad.name, '- valid PDF')\n"
        "    except Exception as e:\n"
        "        print('could not delete', bad, ':', e)\n"
        "else:\n"
        "    print('no stale placeholder present')"
    ),

    # 4 ---------------------------------------------------------------
    md("## 2. Load data"),
    code(
        "if not DATA_PATH.exists():\n"
        "    raise FileNotFoundError(DATA_PATH)\n"
        "df = pd.read_csv(DATA_PATH)\n"
        "print('rows:', len(df))\n"
        "print('columns:', list(df.columns))\n"
        "print('documents:', df['doc_id'].nunique() if 'doc_id' in df else 'n/a')"
    ),

    # 5 ---------------------------------------------------------------
    md("## 3. EDA - class balance and sentence length"),
    code(
        "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n"
        "\n"
        "counts = df['relevant'].fillna('missing').value_counts()\n"
        "axes[0].bar(counts.index.astype(str), counts.values, color=['#d62728', '#2ca02c', '#888888'][:len(counts)])\n"
        "axes[0].set_title('Relevance class balance')\n"
        "axes[0].set_ylabel('sentences')\n"
        "for i, v in enumerate(counts.values):\n"
        "    axes[0].text(i, v, str(v), ha='center', va='bottom')\n"
        "\n"
        "lengths = df['sentence_text'].fillna('').astype(str).str.split().str.len()\n"
        "axes[1].hist(lengths, bins=40, color='#1f77b4')\n"
        "axes[1].set_title(f'Sentence length (words) - mean {lengths.mean():.1f}')\n"
        "axes[1].set_xlabel('words per sentence')\n"
        "axes[1].set_ylabel('count')\n"
        "\n"
        "plt.tight_layout()\n"
        "plt.savefig(RESULTS_DIR / 'eda_class_and_length.png', dpi=120)\n"
        "plt.show()\n"
        "print(counts)\n"
        "print('length stats:')\n"
        "print(lengths.describe().round(1))"
    ),

    # 6 ---------------------------------------------------------------
    md("## 4. EDA - technique distribution (only relevant sentences have techniques)"),
    code(
        "tech_text = df.loc[df['relevant'].eq('yes'), 'techniques'].fillna('').astype(str).str.strip()\n"
        "tech_text = tech_text[tech_text != '']\n"
        "counter = Counter()\n"
        "for row in tech_text:\n"
        "    for t in row.split(','):\n"
        "        t = t.strip()\n"
        "        if t:\n"
        "            counter[t] += 1\n"
        "\n"
        "top = counter.most_common(15)\n"
        "if top:\n"
        "    labels, values = zip(*top)\n"
        "    fig, ax = plt.subplots(figsize=(10, 5))\n"
        "    ax.barh(range(len(labels)), values, color='#9467bd')\n"
        "    ax.set_yticks(range(len(labels)))\n"
        "    ax.set_yticklabels(labels, fontsize=9)\n"
        "    ax.invert_yaxis()\n"
        "    ax.set_title('Top techniques in the corpus')\n"
        "    ax.set_xlabel('sentences')\n"
        "    plt.tight_layout()\n"
        "    plt.savefig(RESULTS_DIR / 'eda_techniques.png', dpi=120)\n"
        "    plt.show()\n"
        "else:\n"
        "    print('no techniques present')\n"
        "print('distinct techniques:', len(counter))"
    ),

    # 7 ---------------------------------------------------------------
    md("## 5. EDA - date coverage"),
    code(
        "if 'date' in df.columns:\n"
        "    dts = pd.to_datetime(df['date'], errors='coerce').dropna()\n"
        "    if len(dts):\n"
        "        by_month = dts.dt.to_period('M').value_counts().sort_index()\n"
        "        fig, ax = plt.subplots(figsize=(12, 3.5))\n"
        "        ax.bar(by_month.index.astype(str), by_month.values, color='#17becf')\n"
        "        ax.set_title(f'Documents per month ({dts.min().date()} to {dts.max().date()})')\n"
        "        ax.set_ylabel('sentences')\n"
        "        step = max(1, len(by_month)//12)\n"
        "        ax.set_xticks(range(0, len(by_month), step))\n"
        "        ax.set_xticklabels(by_month.index.astype(str)[::step], rotation=45, ha='right', fontsize=8)\n"
        "        plt.tight_layout()\n"
        "        plt.savefig(RESULTS_DIR / 'eda_dates.png', dpi=120)\n"
        "        plt.show()\n"
        "    else:\n"
        "        print('date column present but no parseable dates')\n"
        "else:\n"
        "    print('no date column')"
    ),

    # 8 ---------------------------------------------------------------
    md("## 6. Labels and grouped splits"),
    code(
        "df = df.dropna(subset=['sentence_text', 'relevant']).copy()\n"
        "df['label'] = (df['relevant'].str.lower() == 'yes').astype(int)\n"
        "X_text = df['sentence_text'].astype(str).tolist()\n"
        "y      = df['label'].values\n"
        "groups = df['doc_id'].values\n"
        "\n"
        "gss = GroupShuffleSplit(n_splits=N_SPLITS, test_size=0.2, random_state=RANDOM_SEED)\n"
        "splits = list(gss.split(X_text, y, groups))\n"
        "print('sentences:', len(X_text), '| positives:', int(y.sum()))\n"
        "print('first split: train', len(splits[0][0]), '| test', len(splits[0][1]))"
    ),

    # 9 ---------------------------------------------------------------
    md("## 7. Evaluation helper"),
    code(
        "def eval_splits(name, fit_predict_fn):\n"
        "    rows = []\n"
        "    for i, (tr, te) in enumerate(splits):\n"
        "        pred = fit_predict_fn(tr, te)\n"
        "        p, r, f, _ = precision_recall_fscore_support(y[te], pred, average='macro', zero_division=0)\n"
        "        rows.append(dict(split=i, accuracy=accuracy_score(y[te], pred),\n"
        "                         macro_precision=p, macro_recall=r, macro_f1=f))\n"
        "    res = pd.DataFrame(rows)\n"
        "    print(f\"{name:22s} acc={res.accuracy.mean():.4f}+/-{res.accuracy.std():.4f}  \"\n"
        "          f\"f1={res.macro_f1.mean():.4f}+/-{res.macro_f1.std():.4f}\")\n"
        "    return res\n"
        "\n"
        "def summarize(name, res, feat, clf):\n"
        "    return dict(model=name, features=feat, classifier=clf,\n"
        "                accuracy=res.accuracy.mean(), accuracy_std=res.accuracy.std(),\n"
        "                macro_f1=res.macro_f1.mean(), macro_f1_std=res.macro_f1.std())"
    ),

    # 10 --------------------------------------------------------------
    md("### 8. Baseline - TF-IDF + LinearSVC"),
    code(
        "def fit_baseline(tr, te):\n"
        "    v = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=100_000,\n"
        "                        sublinear_tf=True, strip_accents='unicode')\n"
        "    v.fit([X_text[i] for i in tr])\n"
        "    clf = LinearSVC(class_weight='balanced', C=1.0, max_iter=5000, random_state=RANDOM_SEED)\n"
        "    clf.fit(v.transform([X_text[i] for i in tr]), y[tr])\n"
        "    return clf.predict(v.transform([X_text[i] for i in te]))\n"
        "res_base = eval_splits('TF-IDF + LinearSVC', fit_baseline)\n"
        "sum_base = summarize('Baseline', res_base, 'TF-IDF (1-2g)', 'LinearSVC')"
    ),

    # 11 --------------------------------------------------------------
    md("### 9. Sentence-BERT embeddings"),
    code(
        "from sentence_transformers import SentenceTransformer\n"
        "sbert = SentenceTransformer('all-MiniLM-L6-v2')\n"
        "t0 = time.perf_counter()\n"
        "X_sbert = sbert.encode(X_text, batch_size=64, show_progress_bar=True, normalize_embeddings=True)\n"
        "print(f'embedded {len(X_sbert)} in {time.perf_counter()-t0:.1f}s', X_sbert.shape)"
    ),

    # 12 --------------------------------------------------------------
    md("### 10. Proposed 1 - SBERT + LinearSVC"),
    code(
        "def fit_sbert_linear(tr, te):\n"
        "    clf = LinearSVC(class_weight='balanced', C=2.0, max_iter=5000, random_state=RANDOM_SEED)\n"
        "    clf.fit(X_sbert[tr], y[tr])\n"
        "    return clf.predict(X_sbert[te])\n"
        "res_sbert = eval_splits('SBERT + LinearSVC', fit_sbert_linear)\n"
        "sum_sbert = summarize('Proposed 1', res_sbert, 'SBERT', 'LinearSVC (C=2)')"
    ),

    # 13 --------------------------------------------------------------
    md("### 11. Proposed 2 - Hybrid TF-IDF + SBERT + LinearSVC"),
    code(
        "def fit_hybrid(tr, te):\n"
        "    v = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=100_000,\n"
        "                        sublinear_tf=True, strip_accents='unicode')\n"
        "    v.fit([X_text[i] for i in tr])\n"
        "    sp_tr = v.transform([X_text[i] for i in tr]); sp_te = v.transform([X_text[i] for i in te])\n"
        "    sc = StandardScaler(with_mean=False).fit(X_sbert[tr])\n"
        "    Xtr = hstack([sp_tr, csr_matrix(sc.transform(X_sbert[tr]))]).tocsr()\n"
        "    Xte = hstack([sp_te, csr_matrix(sc.transform(X_sbert[te]))]).tocsr()\n"
        "    clf = LinearSVC(class_weight='balanced', C=1.0, max_iter=5000, random_state=RANDOM_SEED)\n"
        "    clf.fit(Xtr, y[tr])\n"
        "    return clf.predict(Xte)\n"
        "res_hybrid = eval_splits('Hybrid', fit_hybrid)\n"
        "sum_hybrid = summarize('Proposed 2', res_hybrid, 'TF-IDF + SBERT', 'LinearSVC (C=1)')"
    ),

    # 14 --------------------------------------------------------------
    md("### 12. Proposed 3 - SBERT + RBF SVM (calibrated)"),
    code(
        "def fit_rbf(tr, te):\n"
        "    base = SVC(kernel='rbf', C=4.0, gamma='scale', class_weight='balanced',\n"
        "               probability=True, random_state=RANDOM_SEED)\n"
        "    clf = CalibratedClassifierCV(base, cv=3, method='sigmoid')\n"
        "    clf.fit(X_sbert[tr], y[tr])\n"
        "    return clf.predict(X_sbert[te])\n"
        "res_rbf = eval_splits('SBERT + RBF SVM', fit_rbf)\n"
        "sum_rbf = summarize('Proposed 3', res_rbf, 'SBERT', 'RBF SVM (C=4)')"
    ),

    # 15 --------------------------------------------------------------
    md("### 13. Proposed 4 - Majority-vote ensemble"),
    code(
        "def fit_ensemble(tr, te):\n"
        "    p1 = fit_sbert_linear(tr, te)\n"
        "    p2 = fit_hybrid(tr, te)\n"
        "    p3 = fit_rbf(tr, te)\n"
        "    return mode(np.vstack([p1, p2, p3]), axis=0, keepdims=False).mode\n"
        "res_ens = eval_splits('Ensemble', fit_ensemble)\n"
        "sum_ens = summarize('Ensemble', res_ens, 'mixed', 'Majority Vote')"
    ),

    # 16 --------------------------------------------------------------
    md("## 14. Summary table - all proposed models"),
    code(
        "table = pd.DataFrame([sum_base, sum_sbert, sum_hybrid, sum_rbf, sum_ens])\n"
        "out = RESULTS_DIR / 'proposed_model_results.csv'\n"
        "table.to_csv(out, index=False)\n"
        "print('wrote', out)\n"
        "table.round(4)"
    ),

    # 17 --------------------------------------------------------------
    md("## 15. Per-class report and confusion matrix (best model, split 0)"),
    code(
        "best_name = table.loc[table['macro_f1'].idxmax(), 'model']\n"
        "print('best model:', best_name)\n"
        "tr0, te0 = splits[0]\n"
        "pred_map = {\n"
        "    'Baseline':   fit_baseline(tr0, te0),\n"
        "    'Proposed 1': fit_sbert_linear(tr0, te0),\n"
        "    'Proposed 2': fit_hybrid(tr0, te0),\n"
        "    'Proposed 3': fit_rbf(tr0, te0),\n"
        "    'Ensemble':   fit_ensemble(tr0, te0),\n"
        "}\n"
        "pred = pred_map[best_name]\n"
        "print(classification_report(y[te0], pred, digits=4))\n"
        "cm = confusion_matrix(y[te0], pred)\n"
        "ConfusionMatrixDisplay(cm, display_labels=['irrelevant', 'relevant']).plot(cmap='Blues')\n"
        "plt.title(f'{best_name}')\n"
        "plt.tight_layout()\n"
        "plt.savefig(RESULTS_DIR / 'confusion_matrix.png', dpi=120)\n"
        "plt.show()"
    ),

    # 18 --------------------------------------------------------------
    md("## 16. Fit relevance classifier on ALL data and save"),
    code(
        "import joblib\n"
        "clf_final = LinearSVC(class_weight='balanced', C=2.0, max_iter=5000,\n"
        "                     random_state=RANDOM_SEED).fit(X_sbert, y)\n"
        "joblib.dump(clf_final, CLS_DIR / 'sbert_relevance.joblib')\n"
        "with open(CLS_DIR / 'relevance_config.json', 'w', encoding='utf-8') as fh:\n"
        "    json.dump({'embedding': 'all-MiniLM-L6-v2', 'dim': 384,\n"
        "               'clf': 'LinearSVC', 'C': 2.0,\n"
        "               'labels': {'0': 'irrelevant', '1': 'relevant'},\n"
        "               'trained_rows': int(len(X_text)),\n"
        "               'cv_best': best_name}, fh, indent=2)\n"
        "print('saved relevance classifier')"
    ),

    # 19 --------------------------------------------------------------
    md(
        "## 17. Multi-class THREAT-TYPE classifier (malware/technique style)\n"
        "\n"
        "This trains a separate model that predicts **which technique** the relevant\n"
        "sentence describes (e.g. `T1566 Phishing`, `T1486 Data Encrypted for Impact`).\n"
        "Runs only on the subset of sentences that actually carry a `techniques` label.\n"
        "The result is the `malware-class`-style prediction you asked for."
    ),
    code(
        "mask = df['relevant'].eq('yes') & df['techniques'].fillna('').astype(str).str.strip().ne('')\n"
        "df_tech = df[mask].copy()\n"
        "# Use the first technique label if multiple are present\n"
        "df_tech['technique'] = df_tech['techniques'].astype(str).str.split(',').str[0].str.strip()\n"
        "print('rows with technique:', len(df_tech))\n"
        "print('distinct labels:', df_tech['technique'].nunique())\n"
        "print(df_tech['technique'].value_counts())"
    ),

    # 20 --------------------------------------------------------------
    md("### 17a. Balance, split, and train technique classifier"),
    code(
        "# Drop classes with fewer than 5 examples to keep grouped split viable\n"
        "vcounts = df_tech['technique'].value_counts()\n"
        "keep = vcounts[vcounts >= 5].index\n"
        "df_tech = df_tech[df_tech['technique'].isin(keep)].reset_index(drop=True)\n"
        "print('kept classes:', df_tech['technique'].nunique(), '| rows:', len(df_tech))\n"
        "\n"
        "X_tech_text = df_tech['sentence_text'].astype(str).tolist()\n"
        "X_tech_emb  = sbert.encode(X_tech_text, batch_size=64, normalize_embeddings=True, show_progress_bar=True)\n"
        "y_tech      = df_tech['technique'].values\n"
        "g_tech      = df_tech['doc_id'].values\n"
        "\n"
        "gss_tech = GroupShuffleSplit(n_splits=5, test_size=0.2, random_state=RANDOM_SEED)\n"
        "tech_accs = []\n"
        "for tr, te in gss_tech.split(X_tech_emb, y_tech, g_tech):\n"
        "    m = LinearSVC(class_weight='balanced', C=2.0, max_iter=5000, random_state=RANDOM_SEED)\n"
        "    m.fit(X_tech_emb[tr], y_tech[tr])\n"
        "    tech_accs.append(accuracy_score(y_tech[te], m.predict(X_tech_emb[te])))\n"
        "print(f'technique classifier accuracy: {np.mean(tech_accs):.4f} +/- {np.std(tech_accs):.4f}')"
    ),

    # 21 --------------------------------------------------------------
    md("### 17b. Fit final technique classifier on ALL technique-labelled data"),
    code(
        "clf_technique = LinearSVC(class_weight='balanced', C=2.0, max_iter=5000,\n"
        "                         random_state=RANDOM_SEED).fit(X_tech_emb, y_tech)\n"
        "joblib.dump(clf_technique, CLS_DIR / 'sbert_technique.joblib')\n"
        "with open(CLS_DIR / 'technique_config.json', 'w', encoding='utf-8') as fh:\n"
        "    json.dump({'embedding': 'all-MiniLM-L6-v2',\n"
        "               'labels': sorted(set(y_tech.tolist())),\n"
        "               'trained_rows': int(len(X_tech_emb)),\n"
        "               'cv_accuracy': float(np.mean(tech_accs))}, fh, indent=2)\n"
        "print('saved technique classifier with', len(set(y_tech.tolist())), 'classes')"
    ),

    # 22 --------------------------------------------------------------
    md("## 18. Emerging-threat model - BERTopic on relevant sentences"),
    code(
        "from bertopic import BERTopic\n"
        "from umap import UMAP\n"
        "from hdbscan import HDBSCAN\n"
        "pos_df = df[df['label'] == 1].copy()\n"
        "docs = pos_df['sentence_text'].tolist()\n"
        "print('relevant sentences for topic modeling:', len(docs))\n"
        "topic_model = BERTopic(\n"
        "    embedding_model=sbert,\n"
        "    umap_model=UMAP(n_neighbors=15, n_components=5, min_dist=0.0,\n"
        "                    metric='cosine', random_state=RANDOM_SEED),\n"
        "    hdbscan_model=HDBSCAN(min_cluster_size=15, metric='euclidean'),\n"
        "    min_topic_size=15, calculate_probabilities=False, verbose=False,\n"
        ")\n"
        "topics, _ = topic_model.fit_transform(docs)\n"
        "pos_df['topic'] = topics\n"
        "info = topic_model.get_topic_info()\n"
        "info.to_csv(RESULTS_DIR / 'topic_info.csv', index=False)\n"
        "topic_model.save(str(THREAT_DIR / 'bertopic_model'), serialization='pickle')\n"
        "print('saved BERTopic model with', len(info), 'topics')"
    ),

    # 23 --------------------------------------------------------------
    md("## 19. Topic visualization - top topics by size"),
    code(
        "top_topics = info[info['Topic'] != -1].head(10)\n"
        "if len(top_topics):\n"
        "    labels = [f\"T{r.Topic}: {' / '.join(str(w) for w in r.Name.split('_')[:3])}\"\n"
        "              for r in top_topics.itertuples()]\n"
        "    fig, ax = plt.subplots(figsize=(10, 5))\n"
        "    ax.barh(range(len(top_topics)), top_topics['Count'].values, color='#2ca02c')\n"
        "    ax.set_yticks(range(len(top_topics)))\n"
        "    ax.set_yticklabels(labels, fontsize=9)\n"
        "    ax.invert_yaxis()\n"
        "    ax.set_xlabel('sentences in topic')\n"
        "    ax.set_title('Top topics discovered by BERTopic')\n"
        "    plt.tight_layout()\n"
        "    plt.savefig(RESULTS_DIR / 'topic_sizes.png', dpi=120)\n"
        "    plt.show()\n"
        "else:\n"
        "    print('no non-outlier topics')"
    ),

    # 24 --------------------------------------------------------------
    md("## 20. Temporal trends + emerging-topic rule"),
    code(
        "if 'date' in pos_df.columns and pd.to_datetime(pos_df['date'], errors='coerce').notna().any():\n"
        "    pos_df['_date'] = pd.to_datetime(pos_df['date'], errors='coerce')\n"
        "    valid = pos_df.dropna(subset=['_date'])\n"
        "    tot = topic_model.topics_over_time(docs=valid['sentence_text'].tolist(),\n"
        "                                       timestamps=valid['_date'].tolist(), nr_bins=12)\n"
        "    tot.to_csv(RESULTS_DIR / 'topic_trends.csv', index=False)\n"
        "    MIN_RUN, MIN_DOCS = 3, 3\n"
        "    flags = []\n"
        "    for tid, g in tot.groupby('Topic'):\n"
        "        g = g.sort_values('Timestamp')\n"
        "        shares = g['Frequency'].values\n"
        "        run = best = 0\n"
        "        for i in range(1, len(shares)):\n"
        "            run = run + 1 if shares[i] > shares[i-1] else 0\n"
        "            best = max(best, run)\n"
        "        docs_n = pos_df.loc[pos_df['topic'] == tid, 'doc_id'].nunique()\n"
        "        if best >= MIN_RUN and docs_n >= MIN_DOCS:\n"
        "            flags.append(dict(topic=int(tid), max_rise_run=int(best), distinct_docs=int(docs_n)))\n"
        "    if flags:\n"
        "        pd.DataFrame(flags).to_csv(RESULTS_DIR / 'emerging_topics.csv', index=False)\n"
        "        print('potentially emerging topics:')\n"
        "        print(pd.DataFrame(flags))\n"
        "    else:\n"
        "        print('no topics met the emerging rule.')\n"
        "else:\n"
        "    print('no date column - temporal analysis skipped.')"
    ),

    # 25 --------------------------------------------------------------
    md("## 21. Shared PDF utilities (hardened)"),
    code(
        "import fitz\n"
        "\n"
        "def _is_pdf_header(path):\n"
        "    try:\n"
        "        return open(path, 'rb').read(4) == b'%PDF'\n"
        "    except OSError:\n"
        "        return False\n"
        "\n"
        "def pdf_info(path):\n"
        "    if not path.exists():         return False, 0, 'not found'\n"
        "    if path.stat().st_size == 0:  return False, 0, 'empty file'\n"
        "    if not _is_pdf_header(path):  return False, 0, 'missing %PDF header'\n"
        "    doc = None\n"
        "    try:\n"
        "        doc = fitz.open(str(path))\n"
        "        return (doc.page_count > 0), doc.page_count, None\n"
        "    except Exception as e:\n"
        "        return False, 0, f'{type(e).__name__}: {e}'\n"
        "    finally:\n"
        "        if doc:\n"
        "            try: doc.close()\n"
        "            except Exception: pass\n"
        "\n"
        "def extract_text_from_pdf(path):\n"
        "    ok, _, err = pdf_info(path)\n"
        "    if not ok:\n"
        "        return '', err\n"
        "    doc = None\n"
        "    try:\n"
        "        doc = fitz.open(str(path))\n"
        "        return '\\n'.join(p.get_text() for p in doc), None\n"
        "    except Exception as e:\n"
        "        return '', f'{type(e).__name__}: {e}'\n"
        "    finally:\n"
        "        if doc:\n"
        "            try: doc.close()\n"
        "            except Exception: pass\n"
        "\n"
        "def split_sentences(text):\n"
        "    text = re.sub(r'\\s+', ' ', text or '').strip()\n"
        "    if not text:\n"
        "        return []\n"
        "    parts = re.split(r'(?<=[.!?])\\s+', text)\n"
        "    return [p.strip() for p in parts if len(p.split()) >= 4]\n"
        "\n"
        "def list_valid_pdfs(folder):\n"
        "    valid, skipped = [], []\n"
        "    if not folder.exists():\n"
        "        return valid, skipped\n"
        "    for p in sorted(folder.glob('*.pdf')):\n"
        "        ok, _, err = pdf_info(p)\n"
        "        (valid if ok else skipped).append(p if ok else (p.name, err))\n"
        "    return valid, skipped\n"
        "\n"
        "print('PDF utilities ready')"
    ),

    # 26 --------------------------------------------------------------
    md(
        "## 22. Analyse a user PDF end-to-end\n"
        "\n"
        "For each valid PDF:\n"
        "1. Split into sentences\n"
        "2. Predict relevance (relevant / irrelevant)\n"
        "3. For relevant sentences, predict the technique (malware-class style)\n"
        "4. Summarize topics (emerging-threat view)\n"
        "\n"
        "Writes `results/batch_predictions.csv` and `results/demo_single_report.json`."
    ),
    code(
        "valid_pdfs, skipped = list_valid_pdfs(PDF_DIR)\n"
        "if skipped:\n"
        "    print('skipped invalid files:', skipped)\n"
        "print(f'valid PDFs: {len(valid_pdfs)}')\n"
        "\n"
        "rows = []\n"
        "single_payload = None\n"
        "for i, p in enumerate(valid_pdfs[:10]):\n"
        "    text, err = extract_text_from_pdf(p)\n"
        "    if err:\n"
        "        rows.append(dict(file=p.name, n_sentences=0, n_relevant=0, n_irrelevant=0,\n"
        "                         top_technique=None, relevant_ratio=0.0, error=err))\n"
        "        continue\n"
        "    sents = split_sentences(text)\n"
        "    if not sents:\n"
        "        continue\n"
        "    emb = sbert.encode(sents, batch_size=64, normalize_embeddings=True)\n"
        "    rel = clf_final.predict(emb)\n"
        "    rel_sents = [s for s, r in zip(sents, rel) if r == 1]\n"
        "    top_technique = None\n"
        "    if rel_sents:\n"
        "        rel_emb = sbert.encode(rel_sents, batch_size=64, normalize_embeddings=True)\n"
        "        tech_preds = clf_technique.predict(rel_emb)\n"
        "        top_technique = Counter(tech_preds.tolist()).most_common(1)[0][0]\n"
        "    n_rel = int(sum(rel))\n"
        "    n_irr = len(sents) - n_rel\n"
        "    rows.append(dict(file=p.name, n_sentences=len(sents),\n"
        "                     n_relevant=n_rel, n_irrelevant=n_irr,\n"
        "                     top_technique=top_technique,\n"
        "                     relevant_ratio=n_rel/len(sents) if sents else 0.0, error=''))\n"
        "    if single_payload is None and len(sents) >= 5:\n"
        "        try:\n"
        "            topics, _ = topic_model.transform(sents)\n"
        "            topic_counts = pd.Series(topics).value_counts().to_dict()\n"
        "        except Exception:\n"
        "            topic_counts = {}\n"
        "        single_payload = dict(\n"
        "            file=p.name, n_sentences=len(sents), n_relevant=n_rel, n_irrelevant=n_irr,\n"
        "            top_technique=top_technique,\n"
        "            relevant_sentences=rel_sents[:5],\n"
        "            topic_counts={str(k): int(v) for k, v in topic_counts.items()},\n"
        "        )\n"
        "\n"
        "batch_df = pd.DataFrame(rows)\n"
        "batch_df.to_csv(RESULTS_DIR / 'batch_predictions.csv', index=False)\n"
        "print('wrote', RESULTS_DIR / 'batch_predictions.csv')\n"
        "if single_payload:\n"
        "    with open(RESULTS_DIR / 'demo_single_report.json', 'w', encoding='utf-8') as fh:\n"
        "        json.dump(single_payload, fh, indent=2, default=str)\n"
        "    print('wrote', RESULTS_DIR / 'demo_single_report.json')\n"
        "batch_df"
    ),

    # 27 --------------------------------------------------------------
    md(
        "## 23. Plain-English summary\n"
        "\n"
        "Everything below is designed to be readable by a non-technical user."
    ),
    code(
        "print('=' * 60)\n"
        "print('WHAT THIS NOTEBOOK DID')\n"
        "print('=' * 60)\n"
        "print()\n"
        "print('1) DATA')\n"
        "print(f'   {len(df)} sentences from {df[\"doc_id\"].nunique()} documents')\n"
        "print(f'   {int(y.sum())} classified as relevant CTI ({(y.mean()*100):.1f}%)')\n"
        "print()\n"
        "print('2) RELEVANCE CLASSIFIER (relevant vs irrelevant)')\n"
        "for row in table.itertuples():\n"
        "    print(f'   {row.model:12s}  accuracy={row.accuracy*100:6.2f}%  macro-F1={row.macro_f1*100:6.2f}%')\n"
        "print()\n"
        "print('3) THREAT-TYPE (technique) CLASSIFIER')\n"
        "print(f'   Trained on {len(df_tech)} relevant sentences with technique labels')\n"
        "print(f'   {len(set(y_tech))} distinct techniques')\n"
        "print(f'   Cross-validated accuracy: {np.mean(tech_accs)*100:.2f}% +/- {np.std(tech_accs)*100:.2f}%')\n"
        "print(f'   Examples: {list(set(y_tech))[:5]}')\n"
        "print()\n"
        "print('4) EMERGING-THREAT MODEL (BERTopic)')\n"
        "print(f'   {len(info) - (1 if -1 in info[\"Topic\"].values else 0)} topics discovered')\n"
        "for _, r in info.head(6).iterrows():\n"
        "    if r['Topic'] == -1:\n"
        "        continue\n"
        "    keywords = ' / '.join(str(w) for w in r['Name'].split('_')[:4])\n"
        "    print(f'   topic {r[\"Topic\"]:>3}: {keywords}  ({r[\"Count\"]} sentences)')\n"
        "print()\n"
        "if 'date' in pos_df.columns and 'tot' in globals():\n"
        "    print('5) TEMPORAL TRENDS')\n"
        "    print(f'   analysed {len(tot)} (topic, time-bin) points')\n"
        "    if (RESULTS_DIR / 'emerging_topics.csv').exists():\n"
        "        flags = pd.read_csv(RESULTS_DIR / 'emerging_topics.csv')\n"
        "        print(f'   {len(flags)} topics flagged as potentially emerging')\n"
        "    print()\n"
        "print('6) PDF ANALYSIS')\n"
        "print(f'   processed {len(batch_df)} valid PDFs')\n"
        "for _, r in batch_df.iterrows():\n"
        "    print(f'   {r[\"file\"][:40]:40s}  sentences={r[\"n_sentences\"]:3d}  relevant={r[\"n_relevant\"]:3d}  top_technique={r[\"top_technique\"]}')\n"
        "print()\n"
        "print('=' * 60)\n"
        "print('FILES WRITTEN')\n"
        "print('=' * 60)\n"
        "for d in (CLS_DIR, THREAT_DIR, RESULTS_DIR):\n"
        "    print(f'{d}:')\n"
        "    for p in sorted(d.iterdir()):\n"
        "        print(f'   {p.name}')\n"
        "print()\n"
        "print('NEXT STEPS FOR THE THESIS')\n"
        "print('- Replace synthetic data with real SMAH-CTI.csv and rerun for real accuracies')\n"
        "print('- Add a date column if missing so temporal trends can run')\n"
        "print('- Report base-paper comparison (base_paper_reproduction.csv) separately')"
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
    print(f"wrote {out}  ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()