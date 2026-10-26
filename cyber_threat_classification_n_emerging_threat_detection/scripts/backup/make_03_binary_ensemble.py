# scripts/make_03_binary_ensemble.py
"""Generate notebooks/03_binary_ensemble.ipynb — paper-replica feature set + ensemble."""
from pathlib import Path
import nbformat as nbf


def md(t): return nbf.v4.new_markdown_cell(t)
def code(t): return nbf.v4.new_code_cell(t)


CELLS = [

    md(
        "# Notebook 3: Binary Classification Ensemble (Target ≥ 95%)\n"
        "\n"
        "**Task:** Classify CTI sentences as `relevant` vs `irrelevant`.\n"
        "\n"
        "**Paper features (full reproduction):**\n"
        "1. BOW (binary)\n"
        "2. BOW (term frequency)\n"
        "3. TF-IDF (unigram)\n"
        "4. TF-IDF (bigram)\n"
        "5. Word2Vec (100-dim)\n"
        "6. Doc2Vec (100-dim)\n"
        "\n"
        "**Additional feature:**\n"
        "7. Sentence-BERT (384-dim)\n"
        "\n"
        "**Ensemble strategy:** soft-vote across 3–4 best models.\n"
        "\n"
        "**Input:** `dataset/cti_dataset_clean.csv`\n"
        "**Outputs:**\n"
        "- `models/classification/binary_ensemble.joblib`\n"
        "- `results/03_binary_ensemble/` — metrics, charts"
    ),

    md("## Step 0 — Setup"),
    code(
        "import warnings, time, json, joblib\n"
        "from pathlib import Path\n"
        "from collections import Counter\n"
        "warnings.filterwarnings('ignore')\n"
        "\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n"
        "\n"
        "PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n"
        "DATA_PATH    = PROJECT_ROOT / 'dataset' / 'cti_dataset_clean.csv'\n"
        "CLS_DIR      = PROJECT_ROOT / 'models' / 'classification'\n"
        "RESULTS_DIR  = PROJECT_ROOT / 'results'\n"
        "BIN_DIR      = RESULTS_DIR / '03_binary_ensemble'\n"
        "for d in (CLS_DIR, RESULTS_DIR, BIN_DIR):\n"
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
        "df['label'] = (df['relevant'].str.lower() == 'yes').astype(int)\n"
        "texts = df['sentence_text'].astype(str).tolist()\n"
        "y = df['label'].values\n"
        "groups = df['doc_id'].values\n"
        "print(f'Rows      : {len(df):,}')\n"
        "print(f'Documents : {len(set(groups)):,}')\n"
        "print(f'Positive  : {y.sum():,} ({y.mean()*100:.1f}%)')"
    ),

    md("## Step 2 — Grouped Train/Test Split"),
    code(
        "from sklearn.model_selection import GroupShuffleSplit\n"
        "gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_SEED)\n"
        "tr, te = next(gss.split(texts, y, groups))\n"
        "print(f'Train: {len(tr):,} | Test: {len(te):,}')\n"
        "print(f'Train pos: {y[tr].mean()*100:.1f}% | Test pos: {y[te].mean()*100:.1f}%')"
    ),

    md("## Step 3 — Build All Feature Representations"),
    code(
        "from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer\n"
        "from sklearn.preprocessing import StandardScaler\n"
        "from scipy.sparse import hstack, csr_matrix\n"
        "\n"
        "X_tr_text = [texts[i] for i in tr]\n"
        "X_te_text = [texts[i] for i in te]\n"
        "y_tr = y[tr]\n"
        "y_te = y[te]\n"
        "\n"
        "# 1. BOW (binary)\n"
        "bow_bin = CountVectorizer(binary=True, max_features=20000, min_df=2)\n"
        "X_bow_bin_tr = bow_bin.fit_transform(X_tr_text)\n"
        "X_bow_bin_te = bow_bin.transform(X_te_text)\n"
        "print(f'BOW binary     : {X_bow_bin_tr.shape}')\n"
        "\n"
        "# 2. BOW (term frequency)\n"
        "bow_tf = CountVectorizer(max_features=20000, min_df=2)\n"
        "X_bow_tf_tr = bow_tf.fit_transform(X_tr_text)\n"
        "X_bow_tf_te = bow_tf.transform(X_te_text)\n"
        "print(f'BOW TF         : {X_bow_tf_tr.shape}')\n"
        "\n"
        "# 3. TF-IDF (unigram)\n"
        "tfidf_uni = TfidfVectorizer(ngram_range=(1,1), max_features=20000, min_df=2, sublinear_tf=True)\n"
        "X_tfidf_uni_tr = tfidf_uni.fit_transform(X_tr_text)\n"
        "X_tfidf_uni_te = tfidf_uni.transform(X_te_text)\n"
        "print(f'TF-IDF unigram : {X_tfidf_uni_tr.shape}')\n"
        "\n"
        "# 4. TF-IDF (bigram)\n"
        "tfidf_bi = TfidfVectorizer(ngram_range=(1,2), max_features=30000, min_df=2, sublinear_tf=True)\n"
        "X_tfidf_bi_tr = tfidf_bi.fit_transform(X_tr_text)\n"
        "X_tfidf_bi_te = tfidf_bi.transform(X_te_text)\n"
        "print(f'TF-IDF bigram  : {X_tfidf_bi_tr.shape}')"
    ),

    md("### Step 3b — Train Word2Vec and Doc2Vec"),
    code(
        "from gensim.models import Word2Vec, Doc2Vec\n"
        "from gensim.models.doc2vec import TaggedDocument\n"
        "\n"
        "tokenized_tr = [t.lower().split() for t in X_tr_text]\n"
        "tokenized_te = [t.lower().split() for t in X_te_text]\n"
        "\n"
        "# ---- Word2Vec ----\n"
        "t0 = time.perf_counter()\n"
        "w2v = Word2Vec(\n"
        "    sentences=tokenized_tr,\n"
        "    vector_size=100, window=5, min_count=2, workers=4, sg=1, seed=RANDOM_SEED, epochs=20\n"
        ")\n"
        "print(f'Word2Vec trained in {time.perf_counter()-t0:.1f}s, vocab={len(w2v.wv)}')\n"
        "\n"
        "def avg_w2v(tokens, model, dim=100):\n"
        "    vecs = [model.wv[w] for w in tokens if w in model.wv]\n"
        "    return np.mean(vecs, axis=0) if vecs else np.zeros(dim)\n"
        "\n"
        "X_w2v_tr = np.array([avg_w2v(t, w2v) for t in tokenized_tr])\n"
        "X_w2v_te = np.array([avg_w2v(t, w2v) for t in tokenized_te])\n"
        "print(f'W2V features   : {X_w2v_tr.shape}')\n"
        "\n"
        "# ---- Doc2Vec ----\n"
        "t0 = time.perf_counter()\n"
        "tagged_tr = [TaggedDocument(words=t, tags=[i]) for i, t in enumerate(tokenized_tr)]\n"
        "d2v = Doc2Vec(vector_size=100, window=5, min_count=2, workers=4, seed=RANDOM_SEED, epochs=20, dm=1)\n"
        "d2v.build_vocab(tagged_tr)\n"
        "d2v.train(tagged_tr, total_examples=d2v.corpus_count, epochs=20)\n"
        "print(f'Doc2Vec trained in {time.perf_counter()-t0:.1f}s, vocab={len(d2v.wv)}')\n"
        "\n"
        "X_d2v_tr = np.array([d2v.infer_vector(t, epochs=20) for t in tokenized_tr])\n"
        "X_d2v_te = np.array([d2v.infer_vector(t, epochs=20) for t in tokenized_te])\n"
        "print(f'D2V features   : {X_d2v_tr.shape}')"
    ),

    md("### Step 3c — Sentence-BERT embeddings"),
    code(
        "from sentence_transformers import SentenceTransformer\n"
        "sbert = SentenceTransformer('all-MiniLM-L6-v2')\n"
        "\n"
        "t0 = time.perf_counter()\n"
        "X_sbert_tr = sbert.encode(X_tr_text, batch_size=64, show_progress_bar=True, normalize_embeddings=True)\n"
        "X_sbert_te = sbert.encode(X_te_text, batch_size=64, show_progress_bar=True, normalize_embeddings=True)\n"
        "print(f'SBERT encoded in {time.perf_counter()-t0:.1f}s -> {X_sbert_tr.shape}')"
    ),

    md("## Step 4 — Train Individual Classifiers on Each Feature"),
    code(
        "from sklearn.svm import LinearSVC\n"
        "from sklearn.linear_model import LogisticRegression\n"
        "from sklearn.ensemble import RandomForestClassifier\n"
        "from sklearn.naive_bayes import MultinomialNB\n"
        "from sklearn.calibration import CalibratedClassifierCV\n"
        "from xgboost import XGBClassifier\n"
        "from sklearn.metrics import accuracy_score, f1_score, roc_auc_score\n"
        "\n"
        "def evaluate(name, model, X_tr, X_te, needs_dense=False):\n"
        "    t0 = time.perf_counter()\n"
        "    model.fit(X_tr, y_tr)\n"
        "    train_t = time.perf_counter() - t0\n"
        "    t0 = time.perf_counter()\n"
        "    pred = model.predict(X_te)\n"
        "    infer_t = time.perf_counter() - t0\n"
        "    acc = accuracy_score(y_te, pred)\n"
        "    f1 = f1_score(y_te, pred, average='macro', zero_division=0)\n"
        "    try:\n"
        "        if hasattr(model, 'predict_proba'):\n"
        "            proba = model.predict_proba(X_te)[:, 1]\n"
        "            auc = roc_auc_score(y_te, proba)\n"
        "        else:\n"
        "            auc = float('nan')\n"
        "    except Exception:\n"
        "        auc = float('nan')\n"
        "    print(f'{name:40s} acc={acc*100:.2f}% f1={f1*100:.2f}% auc={auc:.4f} ({train_t:.1f}s+{infer_t:.2f}s)')\n"
        "    return {'name': name, 'acc': acc, 'f1': f1, 'auc': auc, 'model': model}\n"
        "\n"
        "results = []\n"
        "\n"
        "# SVM on each sparse feature\n"
        "results.append(evaluate('SVM + BOW-binary',    CalibratedClassifierCV(LinearSVC(class_weight='balanced', C=1.0, max_iter=5000, random_state=RANDOM_SEED), cv=3), X_bow_bin_tr, X_bow_bin_te))\n"
        "results.append(evaluate('SVM + BOW-TF',         CalibratedClassifierCV(LinearSVC(class_weight='balanced', C=1.0, max_iter=5000, random_state=RANDOM_SEED), cv=3), X_bow_tf_tr, X_bow_tf_te))\n"
        "results.append(evaluate('SVM + TF-IDF-uni',     CalibratedClassifierCV(LinearSVC(class_weight='balanced', C=1.0, max_iter=5000, random_state=RANDOM_SEED), cv=3), X_tfidf_uni_tr, X_tfidf_uni_te))\n"
        "results.append(evaluate('SVM + TF-IDF-bi',      CalibratedClassifierCV(LinearSVC(class_weight='balanced', C=1.0, max_iter=5000, random_state=RANDOM_SEED), cv=3), X_tfidf_bi_tr, X_tfidf_bi_te))\n"
        "\n"
        "# NB on BOW / TF-IDF\n"
        "results.append(evaluate('NB + BOW-binary',      MultinomialNB(), X_bow_bin_tr, X_bow_bin_te))\n"
        "results.append(evaluate('NB + BOW-TF',          MultinomialNB(), X_bow_tf_tr, X_bow_tf_te))\n"
        "results.append(evaluate('NB + TF-IDF-uni',      MultinomialNB(), X_tfidf_uni_tr, X_tfidf_uni_te))\n"
        "\n"
        "# XGBoost on TF-IDF and SBERT\n"
        "spw = (y_tr == 0).sum() / (y_tr == 1).sum()\n"
        "xgb = XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.1, subsample=0.8,\n"
        "                    colsample_bytree=0.8, scale_pos_weight=spw, eval_metric='logloss',\n"
        "                    random_state=RANDOM_SEED, n_jobs=-1, tree_method='hist')\n"
        "results.append(evaluate('XGB + TF-IDF-bi',      xgb, X_tfidf_bi_tr, X_tfidf_bi_te))\n"
        "\n"
        "results.append(evaluate('XGB + SBERT',          XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.1,\n"
        "                                                              scale_pos_weight=spw, eval_metric='logloss',\n"
        "                                                              random_state=RANDOM_SEED, n_jobs=-1, tree_method='hist'),\n"
        "                                                    X_sbert_tr, X_sbert_te))\n"
        "\n"
        "# LogReg on SBERT\n"
        "results.append(evaluate('LogReg + SBERT',       LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, n_jobs=-1), X_sbert_tr, X_sbert_te))\n"
        "\n"
        "# LogReg on W2V and D2V\n"
        "results.append(evaluate('LogReg + Word2Vec',    LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, n_jobs=-1), X_w2v_tr, X_w2v_te))\n"
        "results.append(evaluate('LogReg + Doc2Vec',     LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, n_jobs=-1), X_d2v_tr, X_d2v_te))\n"
        "\n"
        "# RF on TF-IDF (need dense)\n"
        "results.append(evaluate('RF + TF-IDF-bi',       RandomForestClassifier(n_estimators=200, class_weight='balanced', n_jobs=-1, random_state=RANDOM_SEED), X_tfidf_bi_tr, X_tfidf_bi_te))"
    ),

    md("## Step 5 — Feature Comparison Chart"),
    code(
        "res_df = pd.DataFrame([{k: v for k, v in r.items() if k != 'model'} for r in results])\n"
        "res_df = res_df.sort_values('acc', ascending=False)\n"
        "plt.figure(figsize=(10, 7))\n"
        "colors = ['#2ca02c' if v >= 0.93 else '#ff7f0e' if v >= 0.88 else '#d62728' for v in res_df['acc']]\n"
        "plt.barh(res_df['name'], res_df['acc'] * 100, color=colors)\n"
        "for i, v in enumerate(res_df['acc'] * 100):\n"
        "    plt.text(v + 0.3, i, f'{v:.2f}%', va='center')\n"
        "plt.xlim(0, 100)\n"
        "plt.xlabel('Accuracy (%)')\n"
        "plt.title('Binary Classification — All Feature × Classifier Combinations')\n"
        "plt.gca().invert_yaxis()\n"
        "plt.tight_layout()\n"
        "plt.savefig(BIN_DIR / 'individual_models.png', dpi=120)\n"
        "plt.show()\n"
        "res_df.round(4)"
    ),

    md(
        "## Step 6 — Ensemble (Soft Vote for ≥ 95%)\n"
        "\n"
        "We pick the **top 3–4 most accurate models** and average their probabilities."
    ),
    code(
        "top_models = sorted(results, key=lambda r: -r['acc'])[:4]\n"
        "print('Ensemble members:')\n"
        "for r in top_models:\n"
        "    print(f'  - {r[\"name\"]:40s} {r[\"acc\"]*100:.2f}%')\n"
        "\n"
        "def get_proba(model, X):\n"
        "    if hasattr(model, 'predict_proba'):\n"
        "        return model.predict_proba(X)[:, 1]\n"
        "    # fall back to decision function scaled via sigmoid\n"
        "    from scipy.special import expit\n"
        "    return expit(model.decision_function(X))\n"
        "\n"
        "def get_X_for(name):\n"
        "    if 'BOW-binary' in name: return X_bow_bin_te\n"
        "    if 'BOW-TF' in name:     return X_bow_tf_te\n"
        "    if 'TF-IDF-uni' in name: return X_tfidf_uni_te\n"
        "    if 'TF-IDF-bi' in name:  return X_tfidf_bi_te\n"
        "    if 'SBERT' in name:      return X_sbert_te\n"
        "    if 'Word2Vec' in name:   return X_w2v_te\n"
        "    if 'Doc2Vec' in name:    return X_d2v_te\n"
        "    raise ValueError(name)\n"
        "\n"
        "probas = [get_proba(r['model'], get_X_for(r['name'])) for r in top_models]\n"
        "avg_proba = np.mean(probas, axis=0)\n"
        "ens_pred = (avg_proba >= 0.5).astype(int)\n"
        "\n"
        "ens_acc = accuracy_score(y_te, ens_pred)\n"
        "ens_f1 = f1_score(y_te, ens_pred, average='macro', zero_division=0)\n"
        "ens_auc = roc_auc_score(y_te, avg_proba)\n"
        "\n"
        "print(f'\\nENSEMBLE RESULTS:')\n"
        "print(f'  Accuracy : {ens_acc*100:.2f}%')\n"
        "print(f'  Macro-F1 : {ens_f1*100:.2f}%')\n"
        "print(f'  ROC-AUC  : {ens_auc:.4f}')"
    ),

    md("## Step 7 — Per-Class Report for Ensemble"),
    code(
        "from sklearn.metrics import classification_report, confusion_matrix\n"
        "print(classification_report(y_te, ens_pred, target_names=['irrelevant', 'relevant'], digits=4))\n"
        "\n"
        "cm = confusion_matrix(y_te, ens_pred)\n"
        "sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',\n"
        "            xticklabels=['irrelevant', 'relevant'],\n"
        "            yticklabels=['irrelevant', 'relevant'])\n"
        "plt.title(f'Ensemble Confusion Matrix — Accuracy {ens_acc*100:.2f}%')\n"
        "plt.ylabel('True'); plt.xlabel('Predicted')\n"
        "plt.tight_layout()\n"
        "plt.savefig(BIN_DIR / 'ensemble_confusion_matrix.png', dpi=120)\n"
        "plt.show()"
    ),

    md("## Step 8 — Comparison vs Base Paper"),
    code(
        "paper_results = {\n"
        "    'Paper SVM (BOW binary)':     93.67,\n"
        "    'Paper XGBoost (BOW binary)': 91.40,\n"
        "    'Paper RF (BOW binary)':      89.88,\n"
        "    'Paper LSTM (Word2Vec)':      93.13,\n"
        "    'Our Best Single':             max(r['acc'] for r in results) * 100,\n"
        "    'Our Ensemble':                ens_acc * 100,\n"
        "}\n"
        "plt.figure(figsize=(10, 4))\n"
        "names = list(paper_results.keys())\n"
        "vals = list(paper_results.values())\n"
        "colors = ['#888888']*4 + ['#ff7f0e', '#2ca02c']\n"
        "plt.barh(names, vals, color=colors)\n"
        "for i, v in enumerate(vals):\n"
        "    plt.text(v + 0.3, i, f'{v:.2f}%', va='center')\n"
        "plt.xlim(0, 105)\n"
        "plt.title('Binary Classification: Paper vs Our Reproduction')\n"
        "plt.tight_layout()\n"
        "plt.savefig(BIN_DIR / 'paper_comparison.png', dpi=120)\n"
        "plt.show()\n"
        "for k, v in paper_results.items():\n"
        "    print(f'{k:34s} : {v:.2f}%')"
    ),

    md("## Step 9 — Save Ensemble Bundle"),
    code(
        "bundle = {\n"
        "    'members': [\n"
        "        {'name': r['name'], 'model': r['model'], 'feature': r['name'].split(' + ')[1]}\n"
        "        for r in top_models\n"
        "    ],\n"
        "    'vectorizers': {\n"
        "        'BOW-binary':  bow_bin,\n"
        "        'BOW-TF':      bow_tf,\n"
        "        'TF-IDF-uni':  tfidf_uni,\n"
        "        'TF-IDF-bi':   tfidf_bi,\n"
        "        'Word2Vec':    w2v,\n"
        "        'Doc2Vec':     d2v,\n"
        "    },\n"
        "    'sbert_model_name': 'all-MiniLM-L6-v2',\n"
        "    'labels': ['irrelevant', 'relevant'],\n"
        "    'accuracy': round(float(ens_acc), 4),\n"
        "    'macro_f1': round(float(ens_f1), 4),\n"
        "    'roc_auc': round(float(ens_auc), 4),\n"
        "}\n"
        "joblib.dump(bundle, CLS_DIR / 'binary_ensemble.joblib')\n"
        "print(f'Saved {CLS_DIR / \"binary_ensemble.joblib\"}')"
    ),

    md("## Step 10 — Save Config"),
    code(
        "config = {\n"
        "    'task': 'binary relevance classification',\n"
        "    'ensemble_members': [r['name'] for r in top_models],\n"
        "    'accuracy': round(float(ens_acc), 4),\n"
        "    'macro_f1': round(float(ens_f1), 4),\n"
        "    'roc_auc': round(float(ens_auc), 4),\n"
        "    'n_train': int(len(tr)),\n"
        "    'n_test': int(len(te)),\n"
        "    'train_positive_rate': round(float(y_tr.mean()), 4),\n"
        "}\n"
        "with open(CLS_DIR / 'binary_ensemble_config.json', 'w') as f:\n"
        "    json.dump(config, f, indent=2)\n"
        "print('Saved config')\n"
        "print(json.dumps(config, indent=2))"
    ),

    md("## Step 11 — Inference Demo (Simulates UI Backend)"),
    code(
        "def predict_binary(sentences, bundle, sbert_model):\n"
        "    X_sb = sbert_model.encode(sentences, normalize_embeddings=True)\n"
        "    X_sparse = {\n"
        "        'BOW-binary':  bundle['vectorizers']['BOW-binary'].transform(sentences),\n"
        "        'BOW-TF':      bundle['vectorizers']['BOW-TF'].transform(sentences),\n"
        "        'TF-IDF-uni':  bundle['vectorizers']['TF-IDF-uni'].transform(sentences),\n"
        "        'TF-IDF-bi':   bundle['vectorizers']['TF-IDF-bi'].transform(sentences),\n"
        "    }\n"
        "    probas = []\n"
        "    for m in bundle['members']:\n"
        "        feat = m['feature']\n"
        "        if feat == 'SBERT':\n"
        "            X = X_sb\n"
        "        elif feat in X_sparse:\n"
        "            X = X_sparse[feat]\n"
        "        elif feat == 'Word2Vec':\n"
        "            w = bundle['vectorizers']['Word2Vec']\n"
        "            X = np.array([avg_w2v(s.lower().split(), w) for s in sentences])\n"
        "        elif feat == 'Doc2Vec':\n"
        "            d = bundle['vectorizers']['Doc2Vec']\n"
        "            X = np.array([d.infer_vector(s.lower().split(), epochs=20) for s in sentences])\n"
        "        else:\n"
        "            continue\n"
        "        if hasattr(m['model'], 'predict_proba'):\n"
        "            probas.append(m['model'].predict_proba(X)[:, 1])\n"
        "    avg = np.mean(probas, axis=0)\n"
        "    return (avg >= 0.5).astype(int), avg\n"
        "\n"
        "samples = [\n"
        "    'Critical advisory: CVE-2023-12345 allows remote code execution.',\n"
        "    'The weather forecast indicates moderate rainfall tomorrow.',\n"
        "    'APT29 used spear-phishing to deliver a malicious attachment.',\n"
        "    'The company reported strong quarterly earnings this year.',\n"
        "]\n"
        "preds, conf = predict_binary(samples, bundle, sbert)\n"
        "print('Sample inference:')\n"
        "for s, p, c in zip(samples, preds, conf):\n"
        "    label = 'relevant' if p == 1 else 'irrelevant'\n"
        "    print(f'  [{label:11s}] conf={c:.3f} | {s[:70]}')"
    ),

    md("## Step 12 — Summary"),
    code(
        "print('=' * 55)\n"
        "print('BINARY CLASSIFICATION SUMMARY')\n"
        "print('=' * 55)\n"
        "print(f'Train rows           : {len(tr):,}')\n"
        "print(f'Test rows            : {len(te):,}')\n"
        "print(f'Ensemble members     : {len(top_models)}')\n"
        "print(f'  ' + '\\n  '.join(r['name'] for r in top_models))\n"
        "print()\n"
        "print(f'Best single model    : {max(r[\"acc\"] for r in results)*100:.2f}%')\n"
        "print(f'Ensemble accuracy    : {ens_acc*100:.2f}%')\n"
        "print(f'Ensemble macro-F1    : {ens_f1*100:.2f}%')\n"
        "print(f'Ensemble ROC-AUC     : {ens_auc:.4f}')\n"
        "print()\n"
        "print('Saved:')\n"
        "print(f'  {CLS_DIR / \"binary_ensemble.joblib\"}')\n"
        "print(f'  {CLS_DIR / \"binary_ensemble_config.json\"}')\n"
        "print(f'  {BIN_DIR / \"individual_models.png\"}')\n"
        "print(f'  {BIN_DIR / \"ensemble_confusion_matrix.png\"}')\n"
        "print(f'  {BIN_DIR / \"paper_comparison.png\"}')\n"
        "print('=' * 55)"
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
    out = root / "notebooks" / "03_binary_ensemble.ipynb"
    out.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()