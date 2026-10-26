# scripts/notebook_script/make_03_binary.py
"""Generate notebooks/03_binary_classification.ipynb."""
from pathlib import Path
import nbformat as nbf


def md(t): return nbf.v4.new_markdown_cell(t)
def code(t): return nbf.v4.new_code_cell(t)


CELLS = [
    md("# Notebook 03: Binary Relevance Classification\n"
       "\n"
       "**Task:** Predict `relevant` (yes/no) from CVE description.\n"
       "**Input:** `dataset/cti_unified_clean.csv`\n"
       "**Strategy:** Oversample yes × 20 to fix 30:1 imbalance.\n"
       "**Output:** `models/classification/binary_relevance.joblib`"),

    md("## Step 0 — Setup"),
    code(
        "import warnings, json, joblib, time\n"
        "from pathlib import Path\n"
        "warnings.filterwarnings('ignore')\n"
        "\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n"
        "\n"
        "PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n"
        "DATA_PATH    = PROJECT_ROOT / 'dataset' / 'cti_unified_clean.csv'\n"
        "CLS_DIR      = PROJECT_ROOT / 'models' / 'classification'\n"
        "RES_DIR      = PROJECT_ROOT / 'results' / '03_binary'\n"
        "CLS_DIR.mkdir(parents=True, exist_ok=True)\n"
        "RES_DIR.mkdir(parents=True, exist_ok=True)\n"
        "RANDOM_SEED = 42\n"
        "sns.set_theme(style='whitegrid')\n"
        "print('Output:', CLS_DIR)"
    ),

    md("## Step 1 — Load and Rebalance (1,670 × 20 = 33,400 yes + 33,400 no)"),
    code(
        "df = pd.read_csv(DATA_PATH)\n"
        "print(f'Total: {len(df):,}')\n"
        "\n"
        "bin_df = df[df['relevant'].isin(['yes', 'no'])].copy()\n"
        "yes_df = bin_df[bin_df['relevant'] == 'yes']\n"
        "no_df  = bin_df[bin_df['relevant'] == 'no']\n"
        "print(f'yes: {len(yes_df):,}  no: {len(no_df):,}  ratio: {len(no_df)/len(yes_df):.1f}:1')\n"
        "\n"
        "OVERSAMPLE = 20\n"
        "n_target = len(yes_df) * OVERSAMPLE\n"
        "print(f'\\nOversample yes × {OVERSAMPLE} = {n_target:,}')\n"
        "\n"
        "yes_bal = yes_df.sample(n=n_target, replace=True, random_state=RANDOM_SEED)\n"
        "no_bal  = no_df.sample(n=min(n_target, len(no_df)), replace=False, random_state=RANDOM_SEED)\n"
        "\n"
        "balanced = pd.concat([yes_bal, no_bal]).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)\n"
        "print(f'\\nBalanced: {len(balanced):,}')\n"
        "print(balanced['relevant'].value_counts().to_string())"
    ),

    md("## Step 2 — Split"),
    code(
        "from sklearn.model_selection import train_test_split\n"
        "\n"
        "X = balanced['sentence_text'].astype(str).tolist()\n"
        "y = (balanced['relevant'] == 'yes').astype(int).values\n"
        "\n"
        "X_train, X_test, y_train, y_test = train_test_split(\n"
        "    X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED)\n"
        "print(f'Train: {len(X_train):,}  Test: {len(X_test):,}')\n"
        "print(f'Pos rate train: {y_train.mean()*100:.1f}%')"
    ),

    md("## Step 3 — Features"),
    code(
        "from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer\n"
        "\n"
        "bow_bin = CountVectorizer(binary=True, max_features=15000, min_df=2)\n"
        "X_bb_train = bow_bin.fit_transform(X_train)\n"
        "X_bb_test  = bow_bin.transform(X_test)\n"
        "print(f'BOW: {X_bb_train.shape}')\n"
        "\n"
        "tfidf = TfidfVectorizer(max_features=20000, min_df=2, ngram_range=(1,2), sublinear_tf=True)\n"
        "X_tfidf_train = tfidf.fit_transform(X_train)\n"
        "X_tfidf_test  = tfidf.transform(X_test)\n"
        "print(f'TF-IDF: {X_tfidf_train.shape}')"
    ),

    code(
        "from sentence_transformers import SentenceTransformer\n"
        "sbert = SentenceTransformer('all-MiniLM-L6-v2')\n"
        "t0 = time.time()\n"
        "X_sbert_train = sbert.encode(X_train, batch_size=64, normalize_embeddings=True, show_progress_bar=True)\n"
        "X_sbert_test  = sbert.encode(X_test,  batch_size=64, normalize_embeddings=True, show_progress_bar=True)\n"
        "print(f'SBERT: {X_sbert_train.shape}  ({time.time()-t0:.1f}s)')"
    ),

    md("## Step 4 — Train Base Models"),
    code(
        "from sklearn.svm import LinearSVC\n"
        "from sklearn.linear_model import LogisticRegression\n"
        "from sklearn.calibration import CalibratedClassifierCV\n"
        "from xgboost import XGBClassifier\n"
        "from sklearn.naive_bayes import MultinomialNB\n"
        "from sklearn.metrics import accuracy_score, f1_score, roc_auc_score\n"
        "from scipy.special import expit\n"
        "\n"
        "results = []\n"
        "\n"
        "def fit_eval(name, model, X_tr, X_te, feat):\n"
        "    t0 = time.time()\n"
        "    model.fit(X_tr, y_train)\n"
        "    elapsed = time.time() - t0\n"
        "    if hasattr(model, 'predict_proba'):\n"
        "        proba = model.predict_proba(X_te)[:, 1]\n"
        "    else:\n"
        "        proba = expit(model.decision_function(X_te))\n"
        "    pred = (proba >= 0.5).astype(int)\n"
        "    acc = accuracy_score(y_test, pred)\n"
        "    f1 = f1_score(y_test, pred, average='macro')\n"
        "    auc = roc_auc_score(y_test, proba)\n"
        "    print(f'{name:28s} acc={acc*100:.2f}% f1={f1*100:.2f}% auc={auc:.4f} ({elapsed:.1f}s)')\n"
        "    results.append({'name': name, 'model': model, 'feature': feat,\n"
        "                    'proba': proba, 'acc': acc, 'f1': f1, 'auc': auc})\n"
        "\n"
        "def svm_cal(): return CalibratedClassifierCV(\n"
        "    LinearSVC(class_weight='balanced', C=1.0, max_iter=5000, random_state=RANDOM_SEED), cv=3)\n"
        "\n"
        "fit_eval('SVM+BOW-binary', svm_cal(), X_bb_train, X_bb_test, 'BOW-binary')\n"
        "fit_eval('SVM+TF-IDF',     svm_cal(), X_tfidf_train, X_tfidf_test, 'TF-IDF')\n"
        "fit_eval('NB+BOW-binary',  MultinomialNB(), X_bb_train, X_bb_test, 'BOW-binary')\n"
        "fit_eval('NB+TF-IDF',      MultinomialNB(), X_tfidf_train, X_tfidf_test, 'TF-IDF')\n"
        "fit_eval('XGB+TF-IDF',     XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,\n"
        "                                         eval_metric='logloss', random_state=RANDOM_SEED,\n"
        "                                         n_jobs=-1, tree_method='hist'),\n"
        "          X_tfidf_train, X_tfidf_test, 'TF-IDF')\n"
        "fit_eval('LogReg+SBERT',   LogisticRegression(C=1.0, max_iter=1000, n_jobs=-1, random_state=RANDOM_SEED),\n"
        "          X_sbert_train, X_sbert_test, 'SBERT')\n"
        "fit_eval('XGB+SBERT',      XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,\n"
        "                                         eval_metric='logloss', random_state=RANDOM_SEED,\n"
        "                                         n_jobs=-1, tree_method='hist'),\n"
        "          X_sbert_train, X_sbert_test, 'SBERT')"
    ),

    md("## Step 5 — Model Comparison"),
    code(
        "res_df = pd.DataFrame([{k: v for k, v in r.items() if k not in ('model', 'proba')} for r in results])\n"
        "res_df = res_df.sort_values('f1', ascending=False)\n"
        "fig, ax = plt.subplots(figsize=(10, 5))\n"
        "x = np.arange(len(res_df))\n"
        "ax.barh(x - 0.2, res_df['f1']*100, 0.4, label='Macro-F1', color='#2ca02c')\n"
        "ax.barh(x + 0.2, res_df['acc']*100, 0.4, label='Accuracy', color='#1f77b4')\n"
        "ax.set_yticks(x); ax.set_yticklabels(res_df['name']); ax.invert_yaxis()\n"
        "ax.set_xlabel('Score (%)'); ax.legend()\n"
        "ax.set_title('Binary — Individual Models')\n"
        "plt.tight_layout(); plt.savefig(RES_DIR / '01_individual_models.png', dpi=120); plt.show()\n"
        "print(res_df.round(4).to_string())"
    ),

    md("## Step 6 — Diversity Ensemble"),
    code(
        "from scipy.stats import pearsonr\n"
        "\n"
        "sem = [r for r in results if r['feature'] == 'SBERT']\n"
        "lex = [r for r in results if r['feature'] != 'SBERT']\n"
        "best_sem = max(sem, key=lambda r: r['f1'])\n"
        "best_lex = max(lex, key=lambda r: r['f1'])\n"
        "chosen = [best_lex, best_sem]\n"
        "seen = {best_lex['feature'], best_sem['feature']}\n"
        "print(f'Start: {best_lex[\"name\"]} + {best_sem[\"name\"]}')\n"
        "\n"
        "def div_ok(c, chosen, rho=0.98):\n"
        "    return all(abs(pearsonr(c['proba'], m['proba'])[0]) <= rho for m in chosen)\n"
        "\n"
        "best_f1 = f1_score(y_test, (np.mean([m['proba'] for m in chosen], axis=0) >= 0.5).astype(int), average='macro')\n"
        "for cand in sorted([r for r in results if r not in chosen], key=lambda r: -r['f1']):\n"
        "    if cand['feature'] in seen or not div_ok(cand, chosen): continue\n"
        "    trial = chosen + [cand]\n"
        "    tproba = np.mean([m['proba'] for m in trial], axis=0)\n"
        "    tf1 = f1_score(y_test, (tproba >= 0.5).astype(int), average='macro')\n"
        "    if tf1 > best_f1:\n"
        "        chosen = trial; seen.add(cand['feature']); best_f1 = tf1\n"
        "        print(f'+ {cand[\"name\"]} → F1={tf1*100:.2f}%')\n"
        "\n"
        "print('\\nFinal members:')\n"
        "for r in chosen: print(f'  {r[\"name\"]:28s} feat={r[\"feature\"]:12s} f1={r[\"f1\"]*100:.2f}%')\n"
        "\n"
        "ens_proba = np.mean([m['proba'] for m in chosen], axis=0)\n"
        "ens_pred  = (ens_proba >= 0.5).astype(int)\n"
        "ens_acc = accuracy_score(y_test, ens_pred)\n"
        "ens_f1  = f1_score(y_test, ens_pred, average='macro')\n"
        "ens_auc = roc_auc_score(y_test, ens_proba)\n"
        "print(f'\\nENSEMBLE: acc={ens_acc*100:.2f}% f1={ens_f1*100:.2f}% auc={ens_auc:.4f}')"
    ),

    md("## Step 7 — Confusion Matrix"),
    code(
        "from sklearn.metrics import confusion_matrix, classification_report\n"
        "print(classification_report(y_test, ens_pred, target_names=['irrelevant', 'relevant'], digits=4))\n"
        "cm = confusion_matrix(y_test, ens_pred)\n"
        "plt.figure(figsize=(7, 5))\n"
        "sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',\n"
        "            xticklabels=['irrelevant', 'relevant'], yticklabels=['irrelevant', 'relevant'])\n"
        "plt.title(f'Binary Ensemble — Acc {ens_acc*100:.2f}% | F1 {ens_f1*100:.2f}%')\n"
        "plt.ylabel('True'); plt.xlabel('Predicted')\n"
        "plt.tight_layout(); plt.savefig(RES_DIR / '02_confusion_matrix.png', dpi=120); plt.show()"
    ),

    md("## Step 8 — 5-Fold CV"),
    code(
        "from sklearn.model_selection import StratifiedKFold\n"
        "from sklearn.base import clone\n"
        "\n"
        "skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)\n"
        "cv_acc, cv_f1 = [], []\n"
        "for fold, (tr, va) in enumerate(skf.split(X, y), 1):\n"
        "    X_tr = [X[i] for i in tr]; X_va = [X[i] for i in va]\n"
        "    y_tr = np.array([y[i] for i in tr]); y_va = np.array([y[i] for i in va])\n"
        "    probas = []\n"
        "    for m in chosen:\n"
        "        mdl = clone(m['model'])\n"
        "        if m['feature'] == 'BOW-binary':\n"
        "            Xt = bow_bin.transform(X_tr); Xv = bow_bin.transform(X_va)\n"
        "        elif m['feature'] == 'TF-IDF':\n"
        "            Xt = tfidf.transform(X_tr); Xv = tfidf.transform(X_va)\n"
        "        elif m['feature'] == 'SBERT':\n"
        "            Xt = sbert.encode(X_tr, normalize_embeddings=True, show_progress_bar=False)\n"
        "            Xv = sbert.encode(X_va, normalize_embeddings=True, show_progress_bar=False)\n"
        "        else: continue\n"
        "        mdl.fit(Xt, y_tr)\n"
        "        probas.append(mdl.predict_proba(Xv)[:, 1])\n"
        "    pred = (np.mean(probas, axis=0) >= 0.5).astype(int)\n"
        "    cv_acc.append(accuracy_score(y_va, pred))\n"
        "    cv_f1.append(f1_score(y_va, pred, average='macro'))\n"
        "    print(f'Fold {fold}: acc={cv_acc[-1]*100:.2f}% f1={cv_f1[-1]*100:.2f}%')\n"
        "print(f'\\n5-fold CV: acc={np.mean(cv_acc)*100:.2f}% ± {np.std(cv_acc)*100:.2f}%')\n"
        "print(f'           f1={np.mean(cv_f1)*100:.2f}% ± {np.std(cv_f1)*100:.2f}%')"
    ),

    md("## Step 9 — Save Bundle"),
    code(
        "bundle = {\n"
        "    'task': 'binary_relevance',\n"
        "    'members': [{'name': r['name'], 'model': r['model'], 'feature': r['feature']} for r in chosen],\n"
        "    'vectorizers': {'BOW-binary': bow_bin, 'TF-IDF': tfidf},\n"
        "    'sbert_model_name': 'all-MiniLM-L6-v2',\n"
        "    'labels': ['irrelevant', 'relevant'],\n"
        "    'accuracy': round(float(ens_acc), 4),\n"
        "    'macro_f1': round(float(ens_f1), 4),\n"
        "    'roc_auc':  round(float(ens_auc), 4),\n"
        "    'cv_5fold': {'acc_mean': round(float(np.mean(cv_acc)),4),\n"
        "                 'acc_std':  round(float(np.std(cv_acc)),4),\n"
        "                 'f1_mean':  round(float(np.mean(cv_f1)),4),\n"
        "                 'f1_std':   round(float(np.std(cv_f1)),4)},\n"
        "}\n"
        "joblib.dump(bundle, CLS_DIR / 'binary_relevance.joblib')\n"
        "print(f'✅ {CLS_DIR / \"binary_relevance.joblib\"}')\n"
        "with open(CLS_DIR / 'binary_relevance_config.json', 'w') as f:\n"
        "    json.dump({k: v for k, v in bundle.items() if k not in ('members','vectorizers')}, f, indent=2)\n"
        "print('✅ config saved')"
    ),

    md("## Step 10 — Inference Demo"),
    code(
        "samples = [\n"
        "    'Remote code execution in Chrome allows attacker to execute arbitrary code.',\n"
        "    'Heap buffer overflow in OpenSSL allows denial of service.',\n"
        "    'SQL injection in admin panel allows authentication bypass.',\n"
        "    'The weather today is sunny with a high of 75 degrees.',\n"
        "]\n"
        "def predict_binary(sentences, bundle, sbert_model):\n"
        "    probas = []\n"
        "    for m in bundle['members']:\n"
        "        feat = m['feature']\n"
        "        if feat == 'SBERT':\n"
        "            X = sbert_model.encode(sentences, normalize_embeddings=True)\n"
        "        else:\n"
        "            X = bundle['vectorizers'][feat].transform(sentences)\n"
        "        probas.append(m['model'].predict_proba(X)[:, 1])\n"
        "    return np.mean(probas, axis=0)\n"
        "\n"
        "probs = predict_binary(samples, bundle, sbert)\n"
        "for s, p in zip(samples, probs):\n"
        "    label = 'RELEVANT' if p >= 0.5 else 'irrelevant'\n"
        "    print(f'[{label:10s}] conf={p:.3f} | {s[:60]}')"
    ),

    md("## Step 11 — Summary"),
    code(
        "print('=' * 55)\n"
        "print('BINARY CLASSIFICATION SUMMARY')\n"
        "print('=' * 55)\n"
        "print(f'Rebalanced rows : {len(balanced):,}')\n"
        "print(f'Train / Test    : {len(X_train):,} / {len(X_test):,}')\n"
        "print(f'Members         : {len(chosen)}')\n"
        "for m in chosen: print(f'  - {m[\"name\"]}')\n"
        "print(f'\\nTest accuracy   : {ens_acc*100:.2f}%')\n"
        "print(f'Test macro-F1   : {ens_f1*100:.2f}%')\n"
        "print(f'Test ROC-AUC    : {ens_auc:.4f}')\n"
        "print(f'CV acc          : {np.mean(cv_acc)*100:.2f}% ± {np.std(cv_acc)*100:.2f}%')\n"
        "print(f'CV F1           : {np.mean(cv_f1)*100:.2f}% ± {np.std(cv_f1)*100:.2f}%')"
    ),
]


def main():
    nb = nbf.v4.new_notebook()
    nb["cells"] = CELLS
    nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}
    here = Path(__file__).resolve().parent
    root = here.parent.parent if here.name == "notebook_script" else (here.parent if here.name == "scripts" else here)
    out = root / "notebooks" / "03_binary_classification.ipynb"
    out.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, out)
    print(f"✅ Wrote: {out}")


if __name__ == "__main__":
    main()