# scripts/make_accuracy_notebook.py
"""Generate notebooks/accuracy.ipynb — focused on classifier benchmarking."""
from pathlib import Path
import nbformat as nbf

def md(t): return nbf.v4.new_markdown_cell(t)
def code(t): return nbf.v4.new_code_cell(t)

CELLS = [
    md(
        "# Accuracy Benchmark Notebook\n"
        "\n"
        "**Goal:** Compare 6 classifiers on the real `cti_dataset.csv`.\n"
        "**Splits:** Document-aware with automatic sentence-level fallback.\n"
        "**Outputs:** `proposed_model_results.csv`, `confusion_matrix.png`"
    ),

    md("## 0. Setup"),
    code(
        "import warnings, logging, time\n"
        "from pathlib import Path\n"
        "warnings.filterwarnings('ignore')\n"
        "logging.getLogger('tensorflow').setLevel(logging.ERROR)\n"
        "\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n"
        "\n"
        "from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold\n"
        "from sklearn.feature_extraction.text import TfidfVectorizer\n"
        "from sklearn.svm import LinearSVC, SVC\n"
        "from sklearn.linear_model import LogisticRegression\n"
        "from sklearn.ensemble import RandomForestClassifier\n"
        "from sklearn.calibration import CalibratedClassifierCV\n"
        "from sklearn.preprocessing import StandardScaler\n"
        "from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay\n"
        "from scipy.sparse import hstack, csr_matrix\n"
        "from scipy.stats import mode\n"
        "\n"
        "PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n"
        "DATA_PATH = PROJECT_ROOT / 'dataset' / 'cti_dataset.csv'\n"
        "RESULTS_DIR = PROJECT_ROOT / 'results'\n"
        "RESULTS_DIR.mkdir(parents=True, exist_ok=True)\n"
        "\n"
        "RANDOM_SEED = 42\n"
        "N_SPLITS = 5\n"
        "sns.set_theme(style='whitegrid')\n"
        "print('Setup complete.')"
    ),

    md("## 1. Load & Prepare Data"),
    code(
        "df = pd.read_csv(DATA_PATH).dropna(subset=['sentence_text', 'relevant'])\n"
        "df['label'] = (df['relevant'].str.lower() == 'yes').astype(int)\n"
        "X_text = df['sentence_text'].astype(str).tolist()\n"
        "y = df['label'].values\n"
        "groups = df['doc_id'].values\n"
        "print(f'{len(X_text)} sentences, {len(set(groups))} documents')\n"
        "print(f'Positives: {y.sum()} ({y.mean()*100:.1f}%)')"
    ),

    md("## 2. Robust Splits"),
    code(
        "per_doc = df.groupby('doc_id')['label'].agg(['mean', 'count'])\n"
        "mixed = ((per_doc['mean'] > 0) & (per_doc['mean'] < 1)).sum()\n"
        "print(f'Documents with both classes: {mixed} / {len(per_doc)}')\n"
        "\n"
        "if mixed > 0:\n"
        "    doc_ratio = per_doc['mean'].sort_values()\n"
        "    doc_list = doc_ratio.index.tolist()\n"
        "    n_docs = len(doc_list)\n"
        "    n_test = max(1, int(n_docs * 0.2))\n"
        "    splits = []\n"
        "    for fold in range(N_SPLITS):\n"
        "        test_docs = set(doc_list[fold*n_test:(fold+1)*n_test])\n"
        "        te = np.where(df['doc_id'].isin(test_docs).values)[0]\n"
        "        tr = np.where(~df['doc_id'].isin(test_docs).values)[0]\n"
        "        if len(set(y[te])) >= 2 and len(set(y[tr])) >= 2:\n"
        "            splits.append((tr, te))\n"
        "    if not splits:\n"
        "        splits = list(StratifiedShuffleSplit(n_splits=N_SPLITS, test_size=0.2, random_state=RANDOM_SEED).split(X_text, y))\n"
        "else:\n"
        "    splits = list(StratifiedShuffleSplit(n_splits=N_SPLITS, test_size=0.2, random_state=RANDOM_SEED).split(X_text, y))\n"
        "\n"
        "for i, (tr, te) in enumerate(splits):\n"
        "    print(f'Split {i}: train_pos={y[tr].mean():.2f}, test_pos={y[te].mean():.2f}')"
    ),

    md("## 3. Encode with SBERT"),
    code(
        "from sentence_transformers import SentenceTransformer\n"
        "sbert = SentenceTransformer('all-MiniLM-L6-v2')\n"
        "X_sbert = sbert.encode(X_text, batch_size=64, show_progress_bar=True, normalize_embeddings=True)\n"
        "print('SBERT shape:', X_sbert.shape)"
    ),

    md("## 4. Define All Classifiers"),
    code(
        "def eval_model(name, fit_predict_fn):\n"
        "    rows = []\n"
        "    for tr, te in splits:\n"
        "        t0 = time.perf_counter()\n"
        "        pred = fit_predict_fn(tr, te)\n"
        "        t1 = time.perf_counter() - t0\n"
        "        p, r, f, _ = precision_recall_fscore_support(y[te], pred, average='macro', zero_division=0)\n"
        "        rows.append(dict(model=name, acc=accuracy_score(y[te], pred), f1=f, sec=t1))\n"
        "    res = pd.DataFrame(rows)\n"
        "    print(f'{name:22s} acc={res[\"acc\"].mean()*100:.2f}%  f1={res[\"f1\"].mean()*100:.2f}%  t={res[\"sec\"].mean():.2f}s')\n"
        "    return res"
    ),

    md("### 4.1 TF-IDF Baseline (LinearSVC)"),
    code(
        "def fit_tfidf_linear(tr, te):\n"
        "    v = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)\n"
        "    X_tr = v.fit_transform([X_text[i] for i in tr])\n"
        "    X_te = v.transform([X_text[i] for i in te])\n"
        "    c = LinearSVC(class_weight='balanced', random_state=RANDOM_SEED, max_iter=5000)\n"
        "    c.fit(X_tr, y[tr])\n"
        "    return c.predict(X_te)\n"
        "res_tfidf = eval_model('TF-IDF + LinearSVC', fit_tfidf_linear)"
    ),

    md("### 4.2 TF-IDF + Logistic Regression"),
    code(
        "def fit_tfidf_lr(tr, te):\n"
        "    v = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)\n"
        "    X_tr = v.fit_transform([X_text[i] for i in tr])\n"
        "    X_te = v.transform([X_text[i] for i in te])\n"
        "    c = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=RANDOM_SEED)\n"
        "    c.fit(X_tr, y[tr])\n"
        "    return c.predict(X_te)\n"
        "res_lr = eval_model('TF-IDF + LogReg', fit_tfidf_lr)"
    ),

    md("### 4.3 TF-IDF + Random Forest"),
    code(
        "def fit_tfidf_rf(tr, te):\n"
        "    v = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True, max_features=2000)\n"
        "    X_tr = v.fit_transform([X_text[i] for i in tr])\n"
        "    X_te = v.transform([X_text[i] for i in te])\n"
        "    c = RandomForestClassifier(n_estimators=100, class_weight='balanced', n_jobs=-1, random_state=RANDOM_SEED)\n"
        "    c.fit(X_tr, y[tr])\n"
        "    return c.predict(X_te)\n"
        "res_rf = eval_model('TF-IDF + RF', fit_tfidf_rf)"
    ),

    md("### 4.4 SBERT + LinearSVC"),
    code(
        "def fit_sbert_linear(tr, te):\n"
        "    c = LinearSVC(class_weight='balanced', C=2.0, random_state=RANDOM_SEED, max_iter=5000)\n"
        "    c.fit(X_sbert[tr], y[tr])\n"
        "    return c.predict(X_sbert[te])\n"
        "res_sbert = eval_model('SBERT + LinearSVC', fit_sbert_linear)"
    ),

    md("### 4.5 SBERT + RBF SVM (calibrated)"),
    code(
        "def fit_sbert_rbf(tr, te):\n"
        "    base = SVC(kernel='rbf', C=4.0, gamma='scale', class_weight='balanced', probability=True, random_state=RANDOM_SEED)\n"
        "    c = CalibratedClassifierCV(base, cv=2, method='sigmoid')\n"
        "    c.fit(X_sbert[tr], y[tr])\n"
        "    return c.predict(X_sbert[te])\n"
        "res_rbf = eval_model('SBERT + RBF SVM', fit_sbert_rbf)"
    ),

    md("### 4.6 Hybrid (TF-IDF + SBERT) + LinearSVC"),
    code(
        "def fit_hybrid(tr, te):\n"
        "    v = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)\n"
        "    sp_tr = v.fit_transform([X_text[i] for i in tr])\n"
        "    sp_te = v.transform([X_text[i] for i in te])\n"
        "    sc = StandardScaler(with_mean=False).fit(X_sbert[tr])\n"
        "    X_tr = hstack([sp_tr, csr_matrix(sc.transform(X_sbert[tr]))]).tocsr()\n"
        "    X_te = hstack([sp_te, csr_matrix(sc.transform(X_sbert[te]))]).tocsr()\n"
        "    c = LinearSVC(class_weight='balanced', C=1.0, random_state=RANDOM_SEED, max_iter=5000)\n"
        "    c.fit(X_tr, y[tr])\n"
        "    return c.predict(X_te)\n"
        "res_hybrid = eval_model('Hybrid + LinearSVC', fit_hybrid)"
    ),

    md("### 4.7 Ensemble (Majority Vote)"),
    code(
        "def fit_ensemble(tr, te):\n"
        "    p1 = fit_sbert_linear(tr, te)\n"
        "    p2 = fit_hybrid(tr, te)\n"
        "    p3 = fit_sbert_rbf(tr, te)\n"
        "    return mode(np.vstack([p1, p2, p3]), axis=0, keepdims=False).mode\n"
        "res_ens = eval_model('Ensemble (3)', fit_ensemble)"
    ),

    md("## 5. Accuracy Leaderboard"),
    code(
        "all_results = pd.concat([res_tfidf, res_lr, res_rf, res_sbert, res_rbf, res_hybrid, res_ens])\n"
        "leaderboard = all_results.groupby('model').agg(\n"
        "    acc_mean=('acc', 'mean'),\n"
        "    acc_std=('acc', 'std'),\n"
        "    f1_mean=('f1', 'mean'),\n"
        "    f1_std=('f1', 'std'),\n"
        "    time_mean=('sec', 'mean')\n"
        ").sort_values('acc_mean', ascending=False).reset_index()\n"
        "leaderboard['acc_pct'] = (leaderboard['acc_mean'] * 100).round(2)\n"
        "leaderboard['f1_pct'] = (leaderboard['f1_mean'] * 100).round(2)\n"
        "leaderboard.to_csv(RESULTS_DIR / 'proposed_model_results.csv', index=False)\n"
        "print(leaderboard[['model', 'acc_pct', 'f1_pct', 'time_mean']].to_string(index=False))"
    ),

    md("## 6. Visualize Accuracy"),
    code(
        "plt.figure(figsize=(10, 5))\n"
        "sns.barplot(data=leaderboard, x='acc_pct', y='model', palette='viridis')\n"
        "plt.title('Classifier Accuracy on Real CTI Dataset')\n"
        "plt.xlabel('Accuracy (%)')\n"
        "plt.xlim(0, 100)\n"
        "for i, v in enumerate(leaderboard['acc_pct']):\n"
        "    plt.text(v + 1, i, f'{v:.1f}%', va='center')\n"
        "plt.tight_layout()\n"
        "plt.savefig(RESULTS_DIR / 'accuracy_leaderboard.png', dpi=120)\n"
        "plt.show()"
    ),

    md("## 7. Confusion Matrix (Best Model)"),
    code(
        "best_model = leaderboard.iloc[0]['model']\n"
        "print(f'Best model: {best_model}')\n"
        "\n"
        "fit_map = {\n"
        "    'TF-IDF + LinearSVC': fit_tfidf_linear,\n"
        "    'TF-IDF + LogReg': fit_tfidf_lr,\n"
        "    'TF-IDF + RF': fit_tfidf_rf,\n"
        "    'SBERT + LinearSVC': fit_sbert_linear,\n"
        "    'SBERT + RBF SVM': fit_sbert_rbf,\n"
        "    'Hybrid + LinearSVC': fit_hybrid,\n"
        "    'Ensemble (3)': fit_ensemble,\n"
        "}\n"
        "\n"
        "tr, te = splits[0]\n"
        "pred = fit_map[best_model](tr, te)\n"
        "cm = confusion_matrix(y[te], pred)\n"
        "ConfusionMatrixDisplay(cm, display_labels=['irrelevant', 'relevant']).plot(cmap='Blues')\n"
        "plt.title(best_model)\n"
        "plt.tight_layout()\n"
        "plt.savefig(RESULTS_DIR / 'confusion_matrix.png', dpi=120)\n"
        "plt.show()"
    ),

    md("## 8. Final Summary"),
    code(
        "print('=' * 60)\n"
        "print('ACCURACY RESULTS SUMMARY')\n"
        "print('=' * 60)\n"
        "print(f'Dataset: {len(X_text)} sentences, {len(set(groups))} documents')\n"
        "print(f'Positives: {y.sum()} ({y.mean()*100:.1f}%)')\n"
        "print()\n"
        "print('Model Leaderboard:')\n"
        "for _, row in leaderboard.iterrows():\n"
        "    print(f'  {row[\"model\"]:22s} | acc={row[\"acc_pct\"]:5.2f}% | f1={row[\"f1_pct\"]:5.2f}%')\n"
        "print()\n"
        "print('Files saved to:', RESULTS_DIR)\n"
        "print('  - proposed_model_results.csv')\n"
        "print('  - accuracy_leaderboard.png')\n"
        "print('  - confusion_matrix.png')"
    )
]

nb = nbf.v4.new_notebook()
nb["cells"] = CELLS
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}

here = Path(__file__).resolve().parent
root = here.parent if here.name == "scripts" else here
out = root / "notebooks" / "accuracy.ipynb"
out.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, out)
print(f"wrote {out}")