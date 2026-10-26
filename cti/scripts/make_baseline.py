"""
Generates notebooks/03_baseline_tfidf_svm.ipynb — CORRECTED VERSION.

Fixes test-set leakage in hyperparameter selection:
  - TF-IDF fit on train only.
  - Model selection via 5-fold stratified CV on train.
  - Test set touched exactly once (final evaluation).

Inputs : artifacts/train.csv, artifacts/test.csv
Outputs: artifacts/metrics_tfidf.json
         artifacts/grid_tfidf_cv.csv          ← NEW: full CV grid for thesis appendix
         artifacts/figs/fig_cm_tfidf.png
         artifacts/figs/fig_tfidf_features.png

Run from project root:  python scripts/make_baseline.py
"""
import json, os

os.makedirs("notebooks", exist_ok=True)
os.makedirs("artifacts/figs", exist_ok=True)

def md(src):  return {"cell_type": "markdown", "metadata": {}, "source": src}
def code(src):
    return {"cell_type": "code", "execution_count": None,
            "metadata": {}, "outputs": [], "source": src}

cells = []

# ─────────────────────────────────────────────────────────
# Title / methodology note
# ─────────────────────────────────────────────────────────
cells.append(md(
"# 03 — Baseline: TF-IDF + Linear Classifier (CV-corrected)\n"
"\n"
"**Inputs:** `artifacts/train.csv`, `artifacts/test.csv`  \n"
"**Outputs:** `artifacts/metrics_tfidf.json`, `artifacts/grid_tfidf_cv.csv`, "
"`artifacts/figs/fig_cm_tfidf.png`, `artifacts/figs/fig_tfidf_features.png`  \n"
"**Feeds thesis:** Chapter 4.2 — Baseline Classification Results\n"
"\n"
"### Methodology corrections\n"
"\n"
"This notebook fixes a methodological flaw present in earlier versions:\n"
"hyperparameters are now selected using **5-fold stratified cross-validation on the\n"
"training set**, and the **test set is evaluated exactly once** at the end.\n"
"The earlier version selected hyperparameters by looking at the test set, which\n"
"leaked information into model selection and produced optimistically biased results.\n"
"\n"
"**Protocol (standard ML workflow):**\n"
"1. Fit TF-IDF vectorizer on **train only**.\n"
"2. Grid-search classifiers with **5-fold StratifiedKFold** on the training set.\n"
"3. Retrain the winning configuration on the full training set.\n"
"4. Evaluate once on the test set → final unbiased metrics.\n"
"\n"
"Reported test metrics reflect **generalisation performance**, not selection bias."
))

# ─────────────────────────────────────────────────────────
# Cell 1 — setup
# ─────────────────────────────────────────────────────────
cells.append(md("## 1. Setup"))
cells.append(code(
"import pandas as pd, numpy as np, json, time, os\n"
"import matplotlib.pyplot as plt\n"
"\n"
"from sklearn.feature_extraction.text import TfidfVectorizer\n"
"from sklearn.svm import LinearSVC\n"
"from sklearn.linear_model import LogisticRegression\n"
"from sklearn.pipeline import Pipeline\n"
"from sklearn.model_selection import GridSearchCV, StratifiedKFold\n"
"from sklearn.metrics import (\n"
"    accuracy_score, precision_score, recall_score, f1_score,\n"
"    classification_report, confusion_matrix, ConfusionMatrixDisplay)\n"
"\n"
"os.makedirs('../artifacts/figs', exist_ok=True)\n"
"train = pd.read_csv('../artifacts/train.csv')\n"
"test  = pd.read_csv('../artifacts/test.csv')\n"
"print('train:', train.shape, '| test:', test.shape)\n"
"print('train pos rate:', round(train.relevant.mean(), 4))\n"
"print('test  pos rate:', round(test.relevant.mean(), 4))"
))

# ─────────────────────────────────────────────────────────
# Cell 2 — TF-IDF on train only
# ─────────────────────────────────────────────────────────
cells.append(md(
"## 2. TF-IDF vectorisation\n"
"\n"
"**The vectorizer is fit on training data only.** The test matrix is created with\n"
"`transform()`, which uses the train vocabulary — this is standard and prevents\n"
"vocabulary leakage."
))
cells.append(code(
"vec = TfidfVectorizer(\n"
"    max_features=20000,\n"
"    ngram_range=(1, 2),\n"
"    min_df=2,\n"
"    sublinear_tf=True,\n"
")\n"
"X_train = vec.fit_transform(train['sentence_text'])\n"
"X_test  = vec.transform(test['sentence_text'])\n"
"print('train matrix:', X_train.shape)\n"
"print('test  matrix:', X_test.shape)\n"
"print('vocabulary size:', len(vec.vocabulary_))"
))

# ─────────────────────────────────────────────────────────
# Cell 3 — GridSearchCV on TRAIN only
# ─────────────────────────────────────────────────────────
cells.append(md(
"## 3. Model selection via 5-fold CV on training set\n"
"\n"
"Two candidate families (LinearSVC, LogReg) are evaluated with 5-fold stratified\n"
"cross-validation on `X_train`. **The test set is not touched here.**\n"
"\n"
"Scoring = `f1_macro` (handles the 1.86:1 class imbalance correctly)."
))
cells.append(code(
"cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)\n"
"\n"
"# --- Candidate family 1: LinearSVC ---\n"
"pipe_svc = Pipeline([\n"
"    ('clf', LinearSVC(random_state=42, max_iter=10000)),\n"
"])\n"
"grid_svc = {\n"
"    'clf__C':            [0.1, 0.3, 0.5, 1.0, 2.0],\n"
"    'clf__class_weight': [None, 'balanced'],\n"
"}\n"
"search_svc = GridSearchCV(\n"
"    pipe_svc, grid_svc, cv=cv, scoring='f1_macro',\n"
"    n_jobs=-1, refit=True, verbose=0\n"
")\n"
"\n"
"# --- Candidate family 2: Logistic Regression ---\n"
"pipe_log = Pipeline([\n"
"    ('clf', LogisticRegression(max_iter=2000, solver='liblinear', random_state=42)),\n"
"])\n"
"grid_log = {\n"
"    'clf__C':            [0.5, 1.0, 2.0],\n"
"    'clf__class_weight': ['balanced'],\n"
"}\n"
"search_log = GridSearchCV(\n"
"    pipe_log, grid_log, cv=cv, scoring='f1_macro',\n"
"    n_jobs=-1, refit=True, verbose=0\n"
")\n"
"\n"
"t0 = time.time()\n"
"search_svc.fit(X_train, train['relevant'])\n"
"search_log.fit(X_train, train['relevant'])\n"
"search_time = time.time() - t0\n"
"\n"
"print(f'CV search time: {search_time:.2f} s (both families)')\n"
"print()\n"
"print(f\"LinearSVC — best CV macro-F1: {search_svc.best_score_:.4f}\")\n"
"print(f\"  best params: {search_svc.best_params_}\")\n"
"print()\n"
"print(f\"LogReg    — best CV macro-F1: {search_log.best_score_:.4f}\")\n"
"print(f\"  best params: {search_log.best_params_}\")"
))

# ─────────────────────────────────────────────────────────
# Cell 4 — build full CV grid for appendix
# ─────────────────────────────────────────────────────────
cells.append(md(
"## 4. Full CV grid (for thesis appendix)\n"
"\n"
"Each row shows mean and standard deviation of macro-F1 across the 5 CV folds."
))
cells.append(code(
"def cv_grid_to_df(search, family_name):\n"
"    df = pd.DataFrame(search.cv_results_)\n"
"    df['family'] = family_name\n"
"    keep = ['family', 'params', 'mean_test_score', 'std_test_score', 'rank_test_score']\n"
"    return df[keep]\n"
"\n"
"grid_all = pd.concat([\n"
"    cv_grid_to_df(search_svc, 'LinearSVC'),\n"
"    cv_grid_to_df(search_log, 'LogReg'),\n"
"], ignore_index=True).sort_values('rank_test_score').reset_index(drop=True)\n"
"\n"
"grid_all['mean_test_score'] = grid_all['mean_test_score'].round(4)\n"
"grid_all['std_test_score']  = grid_all['std_test_score'].round(4)\n"
"\n"
"print(grid_all.to_string(index=False))\n"
"grid_all.to_csv('../artifacts/grid_tfidf_cv.csv', index=False)\n"
"print('\\nsaved -> artifacts/grid_tfidf_cv.csv')"
))

# ─────────────────────────────────────────────────────────
# Cell 5 — pick winner
# ─────────────────────────────────────────────────────────
cells.append(md(
"## 5. Select winning configuration by CV score\n"
"\n"
"The model with the highest **mean CV macro-F1** is selected. `best_estimator_` is\n"
"already refit on the full training set, so no extra fitting is needed."
))
cells.append(code(
"if search_svc.best_score_ >= search_log.best_score_:\n"
"    best_search = search_svc\n"
"    best_family = 'LinearSVC'\n"
"else:\n"
"    best_search = search_log\n"
"    best_family = 'LogReg'\n"
"\n"
"clf = best_search.best_estimator_\n"
"\n"
"print(f\"Selected model: {best_family}\")\n"
"print(f\"  CV macro-F1 (train only): {best_search.best_score_:.4f}\")\n"
"print(f\"  Best hyperparameters:     {best_search.best_params_}\")"
))

# ─────────────────────────────────────────────────────────
# Cell 6 — ONE-TIME evaluation on test
# ─────────────────────────────────────────────────────────
cells.append(md(
"## 6. Final evaluation on test set — **one time only**\n"
"\n"
"This is the **single** place where the test set is used. The number reported here\n"
"is the unbiased generalisation estimate of the selected model."
))
cells.append(code(
"t0 = time.time()\n"
"pred = clf.predict(X_test)\n"
"infer_time = time.time() - t0\n"
"\n"
"m = {\n"
"    'model': f\"{best_family} (C={best_search.best_params_['clf__C']}, \"\n"
"             f\"class_weight={best_search.best_params_['clf__class_weight']})\",\n"
"    'selection_strategy': '5-fold StratifiedKFold on training set',\n"
"    'cv_macro_f1':    float(best_search.best_score_),\n"
"    'accuracy':       float(accuracy_score(test['relevant'], pred)),\n"
"    'precision':      float(precision_score(test['relevant'], pred, zero_division=0)),\n"
"    'recall':         float(recall_score(test['relevant'], pred, zero_division=0)),\n"
"    'macro_f1':       float(f1_score(test['relevant'], pred, average='macro', zero_division=0)),\n"
"    'weighted_f1':    float(f1_score(test['relevant'], pred, average='weighted', zero_division=0)),\n"
"    'infer_time_s':   round(infer_time, 4),\n"
"    'hyperparams':    best_search.best_params_,\n"
"}\n"
"\n"
"for k, v in m.items():\n"
"    print(f'{k:>20s}: {v}')\n"
"print()\n"
"print('Classification report (test — evaluated once):\\n')\n"
"print(classification_report(test['relevant'], pred, digits=4))"
))

# ─────────────────────────────────────────────────────────
# Cell 7 — save metrics
# ─────────────────────────────────────────────────────────
cells.append(md("## 7. Save metrics"))
cells.append(code(
"json.dump(m, open('../artifacts/metrics_tfidf.json','w'), indent=2)\n"
"print('saved -> artifacts/metrics_tfidf.json')"
))

# ─────────────────────────────────────────────────────────
# Cell 8 — confusion matrix
# ─────────────────────────────────────────────────────────
cells.append(md("## 8. Confusion matrix (test set)"))
cells.append(code(
"cm = confusion_matrix(test['relevant'], pred, labels=[0, 1])\n"
"disp = ConfusionMatrixDisplay(cm, display_labels=['non-relevant','relevant'])\n"
"disp.plot(cmap='Blues', values_format='d')\n"
"plt.title(f\"Baseline — {best_family} — confusion matrix (test)\")\n"
"plt.tight_layout()\n"
"plt.savefig('../artifacts/figs/fig_cm_tfidf.png')\n"
"plt.show()\n"
"print('saved -> artifacts/figs/fig_cm_tfidf.png')"
))

# ─────────────────────────────────────────────────────────
# Cell 9 — feature importance
# ─────────────────────────────────────────────────────────
cells.append(md(
"## 9. Most informative features\n"
"\n"
"Linear models expose one coefficient per TF-IDF feature. The top-|coef| features\n"
"reveal what the model learned — useful for Chapter 5 interpretation."
))
cells.append(code(
"clf_inner = clf.named_steps['clf']\n"
"coefs = clf_inner.coef_[0]\n"
"\n"
"features = np.array(vec.get_feature_names_out())\n"
"top_pos_idx = np.argsort(coefs)[-20:][::-1]\n"
"top_neg_idx = np.argsort(coefs)[:20]\n"
"\n"
"print('Top 20 RELEVANT-indicative features:')\n"
"for i in top_pos_idx:\n"
"    print(f'  {features[i]:>30s}  coef = {coefs[i]:+.4f}')\n"
"print('\\nTop 20 NON-RELEVANT-indicative features:')\n"
"for i in top_neg_idx:\n"
"    print(f'  {features[i]:>30s}  coef = {coefs[i]:+.4f}')"
))

# ─────────────────────────────────────────────────────────
# Cell 10 — feature importance figure
# ─────────────────────────────────────────────────────────
cells.append(md("## 10. Feature importance figure"))
cells.append(code(
"fig, axes = plt.subplots(1, 2, figsize=(14, 6))\n"
"for ax, idx, title, color in [\n"
"    (axes[0], top_pos_idx, 'Top RELEVANT features',     '#4c72b0'),\n"
"    (axes[1], top_neg_idx, 'Top NON-RELEVANT features', '#dd8452')]:\n"
"    ax.barh(range(len(idx))[::-1], coefs[idx], color=color)\n"
"    ax.set_yticks(range(len(idx))[::-1])\n"
"    ax.set_yticklabels(features[idx])\n"
"    ax.axvline(0, color='black', linewidth=0.5)\n"
"    ax.set_title(title); ax.set_xlabel('Linear coefficient')\n"
"plt.tight_layout()\n"
"plt.savefig('../artifacts/figs/fig_tfidf_features.png')\n"
"plt.show()\n"
"print('saved -> artifacts/figs/fig_tfidf_features.png')"
))

# ─────────────────────────────────────────────────────────
# Cell 11 — summary
# ─────────────────────────────────────────────────────────
cells.append(md(
"## 11. Baseline summary\n"
"\n"
"**Note the gap** between `cv_macro_f1` (training-set CV) and `macro_f1` (test).\n"
"A small gap (~0.01–0.03) means the model generalises well. A large gap (>0.05)\n"
"would indicate overfitting."
))
cells.append(code(
"gap = m['cv_macro_f1'] - m['macro_f1']\n"
"\n"
"print('=' * 64)\n"
"print(f\"{'BASELINE (CV-corrected)':^64s}\")\n"
"print('=' * 64)\n"
"print(f\"  Model                : {m['model']}\")\n"
"print(f\"  Selection strategy   : {m['selection_strategy']}\")\n"
"print(f\"  CV macro-F1 (train)  : {m['cv_macro_f1']:.4f}\")\n"
"print(f\"  Test macro-F1        : {m['macro_f1']:.4f}\")\n"
"print(f\"  Generalisation gap   : {gap:+.4f}  "
"({'good' if abs(gap) < 0.03 else 'check for overfitting'})\")\n"
"print()\n"
"print(f\"  Test accuracy        : {m['accuracy']:.4f}\")\n"
"print(f\"  Test precision       : {m['precision']:.4f}\")\n"
"print(f\"  Test recall          : {m['recall']:.4f}\")\n"
"print(f\"  Test weighted-F1     : {m['weighted_f1']:.4f}\")\n"
"print('=' * 64)"
))

# ─────────────────────────────────────────────────────────
# Cell 12 — methodology note
# ─────────────────────────────────────────────────────────
cells.append(md(
"## 12. Methodology note (for thesis)\n"
"\n"
"Use this text (with the numbers filled in) in Chapter 4.2:\n"
"\n"
"> Hyperparameters were selected via 5-fold stratified cross-validation on the\n"
"> training set to prevent test-set leakage. Two model families (LinearSVC and\n"
"> Logistic Regression) were evaluated with grid search over regularisation strength\n"
"> and class weighting. The configuration with the highest mean CV macro-F1 was\n"
"> retrained on the full training set and evaluated **once** on the held-out test\n"
"> set. Reported test macro-F1 reflects generalisation performance, not selection\n"
"> bias. The generalisation gap between CV and test macro-F1 was +0.0XX, indicating\n"
"> no substantial overfitting."
))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"}
    },
    "nbformat": 4, "nbformat_minor": 5
}

out = os.path.join("notebooks", "03_baseline_tfidf_svm.ipynb")
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Wrote:", os.path.abspath(out))
print("Cells:", len(cells))