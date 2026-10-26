"""
Generates notebooks/04_sbert_classification.ipynb — CORRECTED VERSION.

Fixes test-set leakage in SBERT + hybrid model selection:
  - All hyperparameters selected via 5-fold StratifiedKFold on TRAIN.
  - Test set touched exactly once (final evaluation).
  - CV grids saved for all three approaches.
  - Baseline (from Notebook 3) is loaded and reused — no re-selection.

Inputs : artifacts/train.csv, artifacts/test.csv, artifacts/metrics_tfidf.json
Outputs: artifacts/train_emb.npy, artifacts/test_emb.npy
         artifacts/metrics_sbert.json, artifacts/metrics_hybrid.json
         artifacts/grid_sbert_cv.csv, artifacts/grid_hybrid_cv.csv
         artifacts/figs/fig_cm_sbert.png
         artifacts/figs/fig_three_way_comparison.png

Run from project root:  python scripts/make_sbert.py
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
"# 04 — SBERT and Hybrid Classifiers (CV-corrected)\n"
"\n"
"**Inputs:** `artifacts/train.csv`, `artifacts/test.csv`, `artifacts/metrics_tfidf.json`  \n"
"**Outputs:**\n"
"- `artifacts/train_emb.npy`, `artifacts/test_emb.npy`\n"
"- `artifacts/metrics_sbert.json`, `artifacts/metrics_hybrid.json`\n"
"- `artifacts/grid_sbert_cv.csv`, `artifacts/grid_hybrid_cv.csv`\n"
"- `artifacts/figs/fig_cm_sbert.png`, `artifacts/figs/fig_three_way_comparison.png`\n"
"\n"
"**Feeds thesis:** Chapter 4.2 — Proposed Classification Results\n"
"\n"
"### Methodology corrections\n"
"\n"
"Same protocol as Notebook 3 — **all hyperparameters selected via 5-fold\n"
"stratified cross-validation on the training set**, test set touched once.\n"
"The earlier version selected hyperparameters by looking at the test set,\n"
"which inflated the reported scores.\n"
"\n"
"**Protocol:**\n"
"1. SBERT embeddings computed and cached.\n"
"2. **GridSearchCV (5-fold CV)** for pure SBERT — select best by CV macro-F1.\n"
"3. **GridSearchCV (5-fold CV)** for hybrid — select best by CV macro-F1.\n"
"4. Baseline metrics loaded from Notebook 3 (already CV-corrected).\n"
"5. Test set evaluated **once** for each model.\n"
"\n"
"Reported test metrics reflect generalisation, not selection bias."
))

# ─────────────────────────────────────────────────────────
# Cell 1 — setup
# ─────────────────────────────────────────────────────────
cells.append(md("## 1. Setup"))
cells.append(code(
"import pandas as pd, numpy as np, json, time, os\n"
"import matplotlib.pyplot as plt\n"
"\n"
"from sentence_transformers import SentenceTransformer\n"
"from sklearn.svm import LinearSVC\n"
"from sklearn.linear_model import LogisticRegression\n"
"from sklearn.feature_extraction.text import TfidfVectorizer\n"
"from sklearn.preprocessing import normalize\n"
"from sklearn.pipeline import Pipeline\n"
"from sklearn.model_selection import GridSearchCV, StratifiedKFold\n"
"from scipy.sparse import hstack\n"
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
# Cell 2 — SBERT embeddings (cached)
# ─────────────────────────────────────────────────────────
cells.append(md(
"## 2. Generate / load Sentence-BERT embeddings\n"
"\n"
"Embeddings are cached to disk; a warm run skips re-encoding (~8 min saved)."
))
cells.append(code(
"EMB_TRAIN = '../artifacts/train_emb.npy'\n"
"EMB_TEST  = '../artifacts/test_emb.npy'\n"
"\n"
"if os.path.exists(EMB_TRAIN) and os.path.exists(EMB_TEST):\n"
"    X_train_emb = np.load(EMB_TRAIN)\n"
"    X_test_emb  = np.load(EMB_TEST)\n"
"    embed_time = 0.0\n"
"    print('Loaded cached embeddings.')\n"
"else:\n"
"    sbert = SentenceTransformer('all-MiniLM-L6-v2')\n"
"    t0 = time.time()\n"
"    X_train_emb = sbert.encode(train['sentence_text'].tolist(), batch_size=64,\n"
"                               show_progress_bar=True, convert_to_numpy=True)\n"
"    X_test_emb  = sbert.encode(test['sentence_text'].tolist(),  batch_size=64,\n"
"                               show_progress_bar=True, convert_to_numpy=True)\n"
"    embed_time = time.time() - t0\n"
"    np.save(EMB_TRAIN, X_train_emb); np.save(EMB_TEST, X_test_emb)\n"
"    print('Saved embeddings.')\n"
"\n"
"print('embedding time:', round(embed_time, 2), 's')\n"
"print('train shape:', X_train_emb.shape, '| test shape:', X_test_emb.shape)"
))

# ─────────────────────────────────────────────────────────
# Cell 3 — Pure SBERT: GridSearchCV on TRAIN only
# ─────────────────────────────────────────────────────────
cells.append(md(
"## 3. Pure SBERT — GridSearchCV on training set\n"
"\n"
"5-fold stratified CV on `X_train_emb`. **Test set not touched.**"
))
cells.append(code(
"cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)\n"
"\n"
"# --- LinearSVC ---\n"
"pipe_svc = Pipeline([('clf', LinearSVC(random_state=42, max_iter=10000))])\n"
"grid_svc = {\n"
"    'clf__C':            [0.1, 0.3, 0.5, 1.0, 2.0],\n"
"    'clf__class_weight': [None, 'balanced'],\n"
"}\n"
"search_svc = GridSearchCV(\n"
"    pipe_svc, grid_svc, cv=cv, scoring='f1_macro',\n"
"    n_jobs=-1, refit=True, verbose=0\n"
")\n"
"\n"
"# --- LogReg ---\n"
"pipe_log = Pipeline([('clf', LogisticRegression(max_iter=2000, solver='liblinear', random_state=42))])\n"
"grid_log = {\n"
"    'clf__C':            [0.3, 1.0, 2.0],\n"
"    'clf__class_weight': ['balanced'],\n"
"}\n"
"search_log = GridSearchCV(\n"
"    pipe_log, grid_log, cv=cv, scoring='f1_macro',\n"
"    n_jobs=-1, refit=True, verbose=0\n"
")\n"
"\n"
"t0 = time.time()\n"
"search_svc.fit(X_train_emb, train['relevant'])\n"
"search_log.fit(X_train_emb, train['relevant'])\n"
"search_time = time.time() - t0\n"
"\n"
"print(f'CV search time: {search_time:.1f} s')\n"
"print(f\"LinearSVC — best CV macro-F1: {search_svc.best_score_:.4f}  params: {search_svc.best_params_}\")\n"
"print(f\"LogReg    — best CV macro-F1: {search_log.best_score_:.4f}  params: {search_log.best_params_}\")"
))

# ─────────────────────────────────────────────────────────
# Cell 4 — SBERT CV grid
# ─────────────────────────────────────────────────────────
cells.append(md("## 4. SBERT CV grid (for appendix)"))
cells.append(code(
"def cv_grid_df(search, family):\n"
"    df = pd.DataFrame(search.cv_results_)\n"
"    df['family'] = family\n"
"    return df[['family','params','mean_test_score','std_test_score','rank_test_score']]\n"
"\n"
"grid_sbert_all = pd.concat([\n"
"    cv_grid_df(search_svc, 'LinearSVC'),\n"
"    cv_grid_df(search_log, 'LogReg'),\n"
"], ignore_index=True).sort_values('rank_test_score').reset_index(drop=True)\n"
"grid_sbert_all['mean_test_score'] = grid_sbert_all['mean_test_score'].round(4)\n"
"grid_sbert_all['std_test_score']  = grid_sbert_all['std_test_score'].round(4)\n"
"print(grid_sbert_all.to_string(index=False))\n"
"grid_sbert_all.to_csv('../artifacts/grid_sbert_cv.csv', index=False)\n"
"print('saved -> artifacts/grid_sbert_cv.csv')"
))

# ─────────────────────────────────────────────────────────
# Cell 5 — Select best SBERT model, evaluate on test once
# ─────────────────────────────────────────────────────────
cells.append(md(
"## 5. Select best pure-SBERT model, evaluate on test once"
))
cells.append(code(
"if search_svc.best_score_ >= search_log.best_score_:\n"
"    best_sbert_search = search_svc\n"
"    best_sbert_family = 'LinearSVC'\n"
"else:\n"
"    best_sbert_search = search_log\n"
"    best_sbert_family = 'LogReg'\n"
"\n"
"clf_sbert = best_sbert_search.best_estimator_\n"
"\n"
"t0 = time.time()\n"
"pred_sbert = clf_sbert.predict(X_test_emb)\n"
"infer_time = time.time() - t0\n"
"\n"
"m_sbert = {\n"
"    'model': f\"SBERT + {best_sbert_family}\",\n"
"    'embedding_model': 'all-MiniLM-L6-v2',\n"
"    'selection_strategy': '5-fold StratifiedKFold on training set',\n"
"    'cv_macro_f1':  float(best_sbert_search.best_score_),\n"
"    'accuracy':     float(accuracy_score(test['relevant'], pred_sbert)),\n"
"    'precision':    float(precision_score(test['relevant'], pred_sbert, zero_division=0)),\n"
"    'recall':       float(recall_score(test['relevant'], pred_sbert, zero_division=0)),\n"
"    'macro_f1':     float(f1_score(test['relevant'], pred_sbert, average='macro', zero_division=0)),\n"
"    'weighted_f1':  float(f1_score(test['relevant'], pred_sbert, average='weighted', zero_division=0)),\n"
"    'embed_time_s': round(embed_time, 2),\n"
"    'infer_time_s': round(infer_time, 4),\n"
"    'hyperparams':  best_sbert_search.best_params_,\n"
"}\n"
"\n"
"print(json.dumps(m_sbert, indent=2))\n"
"print('\\nClassification report (test):\\n')\n"
"print(classification_report(test['relevant'], pred_sbert, digits=4))\n"
"json.dump(m_sbert, open('../artifacts/metrics_sbert.json','w'), indent=2)"
))

# ─────────────────────────────────────────────────────────
# Cell 6 — Build hybrid matrix (TF-IDF fit on train only)
# ─────────────────────────────────────────────────────────
cells.append(md(
"## 6. Build hybrid features\n"
"\n"
"TF-IDF fit on **train only**; SBERT embeddings L2-normalized; concatenated via `hstack`."
))
cells.append(code(
"vec_h = TfidfVectorizer(max_features=10000, ngram_range=(1,2), sublinear_tf=True)\n"
"X_train_tfidf = vec_h.fit_transform(train['sentence_text'])\n"
"X_test_tfidf  = vec_h.transform(test['sentence_text'])\n"
"\n"
"X_train_emb_n = normalize(X_train_emb)\n"
"X_test_emb_n  = normalize(X_test_emb)\n"
"\n"
"X_train_hybrid = hstack([X_train_tfidf, X_train_emb_n]).tocsr()\n"
"X_test_hybrid  = hstack([X_test_tfidf,  X_test_emb_n]).tocsr()\n"
"print('hybrid train:', X_train_hybrid.shape)\n"
"print('hybrid test :', X_test_hybrid.shape)"
))

# ─────────────────────────────────────────────────────────
# Cell 7 — Hybrid: GridSearchCV on TRAIN only
# ─────────────────────────────────────────────────────────
cells.append(md(
"## 7. Hybrid — GridSearchCV on training set\n"
"\n"
"5-fold stratified CV on `X_train_hybrid`. **Test set not touched.**"
))
cells.append(code(
"pipe_h_log = Pipeline([\n"
"    ('clf', LogisticRegression(max_iter=2000, solver='liblinear', random_state=42))\n"
"])\n"
"grid_h_log = {\n"
"    'clf__C':            [0.5, 1.0, 2.0],\n"
"    'clf__class_weight': ['balanced'],\n"
"}\n"
"search_h_log = GridSearchCV(\n"
"    pipe_h_log, grid_h_log, cv=cv, scoring='f1_macro',\n"
"    n_jobs=-1, refit=True, verbose=0\n"
")\n"
"\n"
"pipe_h_svc = Pipeline([\n"
"    ('clf', LinearSVC(random_state=42, max_iter=10000))\n"
"])\n"
"grid_h_svc = {\n"
"    'clf__C':            [0.3, 0.5, 1.0, 2.0],\n"
"    'clf__class_weight': [None, 'balanced'],\n"
"}\n"
"search_h_svc = GridSearchCV(\n"
"    pipe_h_svc, grid_h_svc, cv=cv, scoring='f1_macro',\n"
"    n_jobs=-1, refit=True, verbose=0\n"
")\n"
"\n"
"t0 = time.time()\n"
"search_h_log.fit(X_train_hybrid, train['relevant'])\n"
"search_h_svc.fit(X_train_hybrid, train['relevant'])\n"
"hybrid_search_time = time.time() - t0\n"
"\n"
"print(f'CV search time: {hybrid_search_time:.1f} s')\n"
"print(f\"LogReg    — best CV macro-F1: {search_h_log.best_score_:.4f}  params: {search_h_log.best_params_}\")\n"
"print(f\"LinearSVC — best CV macro-F1: {search_h_svc.best_score_:.4f}  params: {search_h_svc.best_params_}\")"
))

# ─────────────────────────────────────────────────────────
# Cell 8 — Hybrid CV grid
# ─────────────────────────────────────────────────────────
cells.append(md("## 8. Hybrid CV grid (for appendix)"))
cells.append(code(
"grid_hybrid_all = pd.concat([\n"
"    cv_grid_df(search_h_log, 'LogReg'),\n"
"    cv_grid_df(search_h_svc, 'LinearSVC'),\n"
"], ignore_index=True).sort_values('rank_test_score').reset_index(drop=True)\n"
"grid_hybrid_all['mean_test_score'] = grid_hybrid_all['mean_test_score'].round(4)\n"
"grid_hybrid_all['std_test_score']  = grid_hybrid_all['std_test_score'].round(4)\n"
"print(grid_hybrid_all.to_string(index=False))\n"
"grid_hybrid_all.to_csv('../artifacts/grid_hybrid_cv.csv', index=False)\n"
"print('saved -> artifacts/grid_hybrid_cv.csv')"
))

# ─────────────────────────────────────────────────────────
# Cell 9 — Select best hybrid, evaluate on test once
# ─────────────────────────────────────────────────────────
cells.append(md(
"## 9. Select best hybrid model, evaluate on test once"
))
cells.append(code(
"if search_h_log.best_score_ >= search_h_svc.best_score_:\n"
"    best_hybrid_search = search_h_log\n"
"    best_hybrid_family = 'LogReg'\n"
"else:\n"
"    best_hybrid_search = search_h_svc\n"
"    best_hybrid_family = 'LinearSVC'\n"
"\n"
"clf_hybrid = best_hybrid_search.best_estimator_\n"
"pred_hybrid = clf_hybrid.predict(X_test_hybrid)\n"
"\n"
"m_hybrid = {\n"
"    'model': f\"Hybrid (TF-IDF + SBERT) + {best_hybrid_family}\",\n"
"    'selection_strategy': '5-fold StratifiedKFold on training set',\n"
"    'cv_macro_f1':  float(best_hybrid_search.best_score_),\n"
"    'accuracy':     float(accuracy_score(test['relevant'], pred_hybrid)),\n"
"    'precision':    float(precision_score(test['relevant'], pred_hybrid, zero_division=0)),\n"
"    'recall':       float(recall_score(test['relevant'], pred_hybrid, zero_division=0)),\n"
"    'macro_f1':     float(f1_score(test['relevant'], pred_hybrid, average='macro', zero_division=0)),\n"
"    'weighted_f1':  float(f1_score(test['relevant'], pred_hybrid, average='weighted', zero_division=0)),\n"
"    'hyperparams':  best_hybrid_search.best_params_,\n"
"}\n"
"\n"
"print(json.dumps(m_hybrid, indent=2))\n"
"print('\\nClassification report (test):\\n')\n"
"print(classification_report(test['relevant'], pred_hybrid, digits=4))\n"
"json.dump(m_hybrid, open('../artifacts/metrics_hybrid.json','w'), indent=2)"
))

# ─────────────────────────────────────────────────────────
# Cell 10 — Three-way comparison (with CV gap check)
# ─────────────────────────────────────────────────────────
cells.append(md(
"## 10. Three-way comparison\n"
"\n"
"Baseline metrics loaded from Notebook 3 (already CV-corrected).\n"
"Generalisation gap = `cv_macro_f1 − macro_f1` per model."
))
cells.append(code(
"with open('../artifacts/metrics_tfidf.json') as f:\n"
"    m_base = json.load(f)\n"
"\n"
"comp = pd.DataFrame([m_base, m_sbert, m_hybrid])[[\n"
"    'model','cv_macro_f1','macro_f1','accuracy','precision','recall','weighted_f1'\n"
"]].round(4)\n"
"comp['gap'] = (comp['cv_macro_f1'] - comp['macro_f1']).round(4)\n"
"print(comp.to_string(index=False))\n"
"\n"
"print('\\nMacro-F1 deltas vs baseline (test):')\n"
"print(f\"  SBERT  : {m_sbert['macro_f1'] - m_base['macro_f1']:+.4f}\")\n"
"print(f\"  Hybrid : {m_hybrid['macro_f1'] - m_base['macro_f1']:+.4f}\")"
))

# ─────────────────────────────────────────────────────────
# Cell 11 — Comparison figure
# ─────────────────────────────────────────────────────────
cells.append(md("## 11. Three-way comparison figure"))
cells.append(code(
"metrics_cols = ['accuracy','precision','recall','macro_f1']\n"
"x = np.arange(len(metrics_cols))\n"
"w = 0.25\n"
"\n"
"fig, ax = plt.subplots(figsize=(10, 5))\n"
"for i, (label, mrow, color) in enumerate([\n"
"    ('Baseline (TF-IDF)', m_base, '#8888ff'),\n"
"    ('SBERT',             m_sbert, '#ff5555'),\n"
"    ('Hybrid',            m_hybrid, '#55aa55')]):\n"
"    vals = [mrow[k] for k in metrics_cols]\n"
"    ax.bar(x + (i-1)*w, vals, w, label=label, color=color)\n"
"    for j, v in enumerate(vals):\n"
"        ax.text(j + (i-1)*w, v + 0.01, f'{v:.2f}', ha='center', fontsize=7)\n"
"\n"
"ax.set_xticks(x); ax.set_xticklabels(metrics_cols)\n"
"ax.set_ylim(0, 1); ax.set_ylabel('Score')\n"
"ax.set_title('Baseline vs SBERT vs Hybrid (CV-corrected selection)')\n"
"ax.legend()\n"
"plt.tight_layout()\n"
"plt.savefig('../artifacts/figs/fig_three_way_comparison.png')\n"
"plt.show()\n"
"print('saved -> artifacts/figs/fig_three_way_comparison.png')"
))

# ─────────────────────────────────────────────────────────
# Cell 12 — Confusion matrices
# ─────────────────────────────────────────────────────────
cells.append(md("## 12. Confusion matrices (test set)"))
cells.append(code(
"fig, axes = plt.subplots(1, 2, figsize=(12, 5))\n"
"\n"
"for ax, pred, title, cmap in [\n"
"    (axes[0], pred_sbert,  'SBERT',  'Reds'),\n"
"    (axes[1], pred_hybrid, 'Hybrid', 'Greens')]:\n"
"    cm = confusion_matrix(test['relevant'], pred, labels=[0, 1])\n"
"    ConfusionMatrixDisplay(cm, display_labels=['non-relevant','relevant']).plot(ax=ax, cmap=cmap, values_format='d', colorbar=False)\n"
"    ax.set_title(title)\n"
"\n"
"plt.tight_layout()\n"
"plt.savefig('../artifacts/figs/fig_cm_sbert.png')\n"
"plt.savefig('../artifacts/figs/fig_cm_hybrid.png')\n"
"plt.show()\n"
"print('saved -> artifacts/figs/fig_cm_sbert.png, fig_cm_hybrid.png')"
))

# ─────────────────────────────────────────────────────────
# Cell 13 — Summary
# ─────────────────────────────────────────────────────────
cells.append(md(
"## 13. Classification summary\n"
"\n"
"**Note the CV-vs-test gap** column — if |gap| < 0.03, generalisation is good."
))
cells.append(code(
"print('=' * 88)\n"
"print(f\"{'CLASSIFICATION — CV-CORRECTED THREE-WAY SUMMARY':^88s}\")\n"
"print('=' * 88)\n"
"print(f\"{'Model':<22s}{'CV F1':>8s}{'Test F1':>10s}{'Gap':>8s}{'Acc':>8s}{'Prec':>8s}{'Rec':>8s}\")\n"
"print('-' * 88)\n"
"for label, mrow in [('Baseline (TF-IDF)', m_base),\n"
"                    ('SBERT',             m_sbert),\n"
"                    ('Hybrid',            m_hybrid)]:\n"
"    gap = mrow.get('cv_macro_f1', 0) - mrow['macro_f1']\n"
"    print(f\"{label:<22s}\"\n"
"          f\"{mrow.get('cv_macro_f1', 0):>8.4f}\"\n"
"          f\"{mrow['macro_f1']:>10.4f}\"\n"
"          f\"{gap:>+8.4f}\"\n"
"          f\"{mrow['accuracy']:>8.4f}\"\n"
"          f\"{mrow['precision']:>8.4f}\"\n"
"          f\"{mrow['recall']:>8.4f}\")\n"
"print('=' * 88)\n"
"print('\\nMethodology note for Chapter 4.2:')\n"
"print('Hyperparameters selected via 5-fold stratified CV on the training set.')\n"
"print('Test set evaluated exactly once per model. Reported test metrics')\n"
"print('reflect generalisation performance, not selection bias.')\n"
"print('CV grids saved to artifacts/grid_sbert_cv.csv and grid_hybrid_cv.csv.')"
))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"}
    },
    "nbformat": 4, "nbformat_minor": 5
}

out = os.path.join("notebooks", "04_sbert_classification.ipynb")
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Wrote:", os.path.abspath(out))
print("Cells:", len(cells))