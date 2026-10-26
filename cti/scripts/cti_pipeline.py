# %% [markdown]
# # CTI Analysis Pipeline — Consolidated Notebook
# 
# **Cyber Threat Classification and Emerging Threat Detection using Sentence-BERT and BERTopic**
# 
# This single notebook consolidates the full pipeline:
# 1. Data Exploration (EDA)
# 2. Preprocessing & Document-Level Split
# 3. Baseline TF-IDF + LinearSVC (CV-corrected)
# 4. SBERT and Hybrid Classifiers (CV-corrected)
# 5. BERTopic Topic Discovery
# 6. Results Comparison & Discussion
#
# **Inputs:** `dataset/SMAH-CTI.csv`
# **Outputs:** All artifacts in `artifacts/` directory
#
# **Methodology corrections applied throughout:**
# - Group-aware cross-validation to prevent document leakage
# - TF-IDF fitted inside each CV fold (no vocabulary leakage)
# - Test set evaluated exactly once per model
# - Hyperparameters selected by CV macro-F1, not test performance
# - Document-level split with zero overlap assertions

# %% [markdown]
# ## 0. Setup — Imports & Project Root

# %%
from pathlib import Path
import os, json, re, time, hashlib, warnings, shutil
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    GridSearchCV, StratifiedKFold, StratifiedGroupKFold, GroupShuffleSplit
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.preprocessing import normalize
from scipy.sparse import hstack

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 120

# ------------------------------------------------------------------
# Locate project root
# ------------------------------------------------------------------
cwd = Path.cwd()
PROJECT_ROOT = next(
    (p for p in [cwd, *cwd.parents]
     if (p / "dataset" / "SMAH-CTI.csv").exists()),
    None,
)
if PROJECT_ROOT is None:
    raise FileNotFoundError(
        f"Could not find dataset/SMAH-CTI.csv from {cwd} or any parent."
    )

CSV_PATH  = PROJECT_ROOT / "dataset" / "SMAH-CTI.csv"
ARTIFACTS = PROJECT_ROOT / "artifacts"
FIG_DIR   = ARTIFACTS / "figs"
ARTIFACTS.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

print("Project root:", PROJECT_ROOT)
print("Dataset     :", CSV_PATH)
print("Artifacts   :", ARTIFACTS)

# %% [markdown]
# ## 1. Data Exploration (EDA)
#
# Load raw data, validate labels, inspect class balance, sentence lengths,
# document distribution, source distribution, annotation coverage,
# vocabulary statistics. Produces `artifacts/eda_summary.json` and figures.

# %%
print("\n" + "=" * 70)
print("STEP 1: EXPLORATORY DATA ANALYSIS")
print("=" * 70)

df = pd.read_csv(CSV_PATH, low_memory=False)
raw_rows = len(df)
raw_columns = df.columns.tolist()

print("Original shape   :", df.shape)
print("Original row count:", raw_rows)
print("Columns          :", raw_columns)

# --- Rename doc_id -> document_id ---
df = df.rename(columns={"doc_id": "document_id"})

# --- Validate labels BEFORE mapping ---
if "relevant" not in df.columns:
    raise ValueError("Required column 'relevant' is missing.")

label_values = df["relevant"].astype("string").str.strip().str.lower()
print("\nOriginal label value counts:")
print(label_values.value_counts(dropna=False))

allowed = {"yes", "no"}
unexpected = sorted(v for v in label_values.dropna().unique() if v not in allowed)
if unexpected:
    raise ValueError(f"Unexpected relevance labels: {unexpected}. Expected only {sorted(allowed)}.")

df["relevant"] = label_values.map({"yes": 1, "no": 0})

# --- Drop rows missing required fields ---
before_drop = len(df)
df = df.dropna(subset=["document_id", "sentence_text", "relevant"]).copy()
df["document_id"] = df["document_id"].astype(str).str.strip()
df = df[df["document_id"].ne("")].copy()
df["relevant"] = df["relevant"].astype(int)
df = df.reset_index(drop=True)
print(f"\nRows removed for missing text/labels: {before_drop - len(df)}")
print("Rows after normalization:", len(df))

# --- Class distribution ---
counts = df["relevant"].value_counts().sort_index()
print("\nClass counts:")
print(counts)
print("Unique documents:", df["document_id"].nunique())

# --- Sentence length stats ---
df["n_words"] = df["sentence_text"].astype(str).str.split().str.len()
df["n_chars"] = df["sentence_text"].astype(str).str.len()
print("\nSentence length stats:")
print(df[["n_words", "n_chars"]].describe().round(2))

# --- Vocabulary stats ---
all_tokens = re.findall(r"\b[a-z]{3,}\b",
                        " ".join(df["sentence_text"].astype(str).str.lower()))
freq = Counter(all_tokens)
hapax = sum(1 for w, c in freq.items() if c == 1)
vocab = len(set(all_tokens))
total_tokens = len(all_tokens)

# --- Source distribution ---
src = df["source"].value_counts() if "source" in df.columns else pd.Series(dtype=int)
print("\nSources:")
print(src)

# --- EDA summary ---
if counts.min() > 0 and len(counts) == 2:
    imbalance_ratio = round(float(counts.max() / counts.min()), 3)
else:
    imbalance_ratio = None

eda = {
    "raw_rows": int(raw_rows),
    "rows_after_cleaning": int(len(df)),
    "rows_removed": int(raw_rows - len(df)),
    "relevant": int((df["relevant"] == 1).sum()),
    "non_relevant": int((df["relevant"] == 0).sum()),
    "imbalance_ratio": imbalance_ratio,
    "num_documents": int(df["document_id"].nunique()),
    "avg_words": round(float(df["n_words"].mean()), 2),
    "median_words": float(df["n_words"].median()),
    "max_words": int(df["n_words"].max()),
    "vocab_size": int(vocab),
    "type_token_ratio": round(vocab / total_tokens, 4) if total_tokens else None,
    "hapax_pct": round(hapax / vocab * 100, 2) if vocab else None,
    "num_sources": int(df["source"].nunique()) if "source" in df.columns else None,
}
with open(ARTIFACTS / "eda_summary.json", "w", encoding="utf-8") as f:
    json.dump(eda, f, indent=2)
print("\nEDA summary saved:", ARTIFACTS / "eda_summary.json")

# --- EDA figures ---
# Class distribution
plt.figure(figsize=(5, 4))
vals = [counts.get(0, 0), counts.get(1, 0)]
plt.bar(["non-relevant (0)", "relevant (1)"], vals, color=["#dd8452", "#4c72b0"])
for i, v in enumerate(vals):
    plt.text(i, v, str(v), ha="center", va="bottom")
plt.ylabel("Count"); plt.title("Class distribution")
plt.tight_layout(); plt.savefig(FIG_DIR / "fig_class_distribution.png"); plt.show()

# Sentence length distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(df["n_words"], bins=40, color="#4c72b0")
axes[0].set_title("Words per sentence"); axes[0].set_xlabel("words"); axes[0].set_ylabel("count")
axes[1].hist(df["n_chars"], bins=40, color="#dd8452")
axes[1].set_title("Characters per sentence"); axes[1].set_xlabel("chars")
plt.tight_layout(); plt.savefig(FIG_DIR / "fig_sentence_length.png"); plt.show()

# Source distribution
if len(src) > 0:
    plt.figure(figsize=(7, 4))
    src.plot(kind="bar", color="#c44e52")
    plt.title("Sentences per source"); plt.ylabel("count")
    plt.tight_layout(); plt.savefig(FIG_DIR / "fig_sources.png"); plt.show()

print("EDA figures saved.")

# %% [markdown]
# ## 2. Preprocessing & Document-Level Split
#
# - Clean HTML tags (bounded regex)
# - Collapse whitespace
# - Check for conflicting labels
# - 80/20 document-level split via `GroupShuffleSplit`
# - Assert zero document overlap and zero sentence-text overlap
# - Save `train.csv`, `test.csv`, `split_info.json`

# %%
print("\n" + "=" * 70)
print("STEP 2: PREPROCESSING & DOCUMENT-LEVEL SPLIT")
print("=" * 70)

# --- Clean sentence_text ---
TAG_RE = re.compile(r"<[^>]{0,80}>")

def clean(text):
    if not isinstance(text, str):
        return ""
    text = TAG_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

before_clean = len(df)
df["sentence_text"] = df["sentence_text"].apply(clean)
df = df[df["sentence_text"].str.len() > 0].reset_index(drop=True)
print(f"Rows removed by cleaning: {before_clean - len(df)}")
print("Rows after cleaning:", len(df))

# --- Detect conflicting labels for identical sentence text ---
label_counts = df.groupby("sentence_text")["relevant"].nunique()
conflicting = label_counts[label_counts > 1].index
print(f"Conflicting sentence texts: {len(conflicting)}")
if len(conflicting) > 0:
    print("  Example conflicts:")
    sample = df[df["sentence_text"].isin(conflicting)].head(5)
    print(sample[["sentence_text", "relevant", "document_id"]].to_string(index=False))

# --- Document-level 80/20 split ---
if df["document_id"].nunique() < 2:
    raise ValueError("At least two distinct documents are required to split.")

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(df, y=df["relevant"], groups=df["document_id"]))

train = df.iloc[train_idx].reset_index(drop=True)
test  = df.iloc[test_idx].reset_index(drop=True)

if train.empty or test.empty:
    raise ValueError("Split produced empty train or test set.")
for name, part in [("train", train), ("test", test)]:
    if part["relevant"].nunique() < 2:
        raise ValueError(f"{name} set contains only one class after split.")

print(f"\nTrain: {len(train):,} rows ({len(train)/len(df):.1%}), "
      f"{train['document_id'].nunique():,} documents")
print(f"Test : {len(test):,} rows ({len(test)/len(df):.1%}), "
      f"{test['document_id'].nunique():,} documents")
print("Train pos rate:", round(train["relevant"].mean(), 4))
print("Test  pos rate:", round(test["relevant"].mean(), 4))

# --- Leakage check ---
document_overlap = set(train["document_id"]) & set(test["document_id"])
text_overlap     = set(train["sentence_text"]) & set(test["sentence_text"])
print("Overlapping documents:", len(document_overlap))
print("Overlapping sentence texts:", len(text_overlap))
assert not document_overlap, "Document leakage detected!"
assert not text_overlap, "Sentence-text overlap detected!"
print("Both leakage checks passed.")

# --- Save split artifacts ---
train.to_csv(ARTIFACTS / "train.csv", index=False)
test.to_csv(ARTIFACTS / "test.csv", index=False)

split_info = {
    "original_rows": int(raw_rows),
    "rows_after_preprocessing": int(len(df)),
    "train_size": int(len(train)),
    "test_size": int(len(test)),
    "train_docs": int(train["document_id"].nunique()),
    "test_docs": int(test["document_id"].nunique()),
    "overlap_docs": int(len(document_overlap)),
    "overlap_sentence_texts": int(len(text_overlap)),
    "train_pos_rate": round(float(train["relevant"].mean()), 4),
    "test_pos_rate": round(float(test["relevant"].mean()), 4),
    "random_state": 42,
    "split_method": "GroupShuffleSplit",
}
with open(ARTIFACTS / "split_info.json", "w", encoding="utf-8") as f:
    json.dump(split_info, f, indent=2)
print("Split artifacts saved.")

# %% [markdown]
# ## 3. Group-Aware CV Setup
#
# Build connected components from documents sharing exact sentence text.
# All rows in a component stay in the same CV fold — this prevents leakage
# that `StratifiedKFold` alone would allow.

# %%
print("\n" + "=" * 70)
print("STEP 3: GROUP-AWARE CROSS-VALIDATION SETUP")
print("=" * 70)

parent = {}
def find(x):
    parent.setdefault(x, x)
    if parent[x] != x:
        parent[x] = find(parent[x])
    return parent[x]
def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb:
        parent[rb] = ra

for doc in train["document_id"].astype(str).unique():
    find(doc)
for _, rows in train.groupby("sentence_text", sort=False):
    docs = rows["document_id"].astype(str).unique().tolist()
    for doc in docs[1:]:
        union(docs[0], doc)

train_docs = train["document_id"].astype(str)
groups = train_docs.map(find).to_numpy()
n_components = pd.Series(groups).nunique()
print("CV connected components:", n_components)
print("Largest component rows:", pd.Series(groups).value_counts().max())

N_SPLITS = 5
if n_components < N_SPLITS:
    raise ValueError(f"Only {n_components} connected components; cannot use {N_SPLITS}-fold grouped CV.")

cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
for fold, (tr_i, va_i) in enumerate(cv.split(train, train["relevant"], groups), start=1):
    overlap = set(groups[tr_i]) & set(groups[va_i])
    if overlap:
        raise AssertionError(f"Group leakage in fold {fold}")
    print(f"Fold {fold}: train={len(tr_i)}, val={len(va_i)}, "
          f"val pos rate={train['relevant'].iloc[va_i].mean():.3f}")

np.save(ARTIFACTS / "train_cv_groups.npy", groups)
cv_info = {
    "cv_method": "StratifiedGroupKFold",
    "n_splits": N_SPLITS,
    "random_state": 42,
    "grouping": "connected components of documents linked by exact sentence text within train split",
    "train_rows": int(len(train)),
    "train_components": int(n_components),
    "tfidf_fitted_inside_cv": True,
    "selection_metric": "macro_f1",
}
with open(ARTIFACTS / "group_cv_info.json", "w", encoding="utf-8") as f:
    json.dump(cv_info, f, indent=2)
print("CV groups saved:", ARTIFACTS / "train_cv_groups.npy")

# %% [markdown]
# ## 4. Baseline — TF-IDF + LinearSVC (CV-corrected)
#
# TF-IDF fitted **inside each CV fold** to prevent vocabulary leakage.
# Hyperparameters selected by 5-fold `StratifiedGroupKFold` macro-F1.
# Test set evaluated exactly once.

# %%
print("\n" + "=" * 70)
print("STEP 4: BASELINE TF-IDF + LinearSVC (grouped CV)")
print("=" * 70)

tfidf_pipe = Pipeline([
    ("tfidf", TfidfVectorizer(sublinear_tf=True, strip_accents="unicode", lowercase=True)),
    ("clf", LinearSVC(random_state=42, max_iter=20000)),
])
param_grid = {
    "tfidf__ngram_range": [(1, 1), (1, 2), (1, 3)],
    "tfidf__min_df": [1, 2, 3],
    "tfidf__max_features": [20000, 50000, None],
    "clf__C": [0.1, 0.3, 1.0, 3.0, 10.0],
    "clf__class_weight": [None, "balanced"],
}
search_tfidf = GridSearchCV(
    tfidf_pipe, param_grid, scoring="f1_macro", cv=cv,
    n_jobs=-1, refit=True, verbose=0, return_train_score=True,
)
t0 = time.time()
search_tfidf.fit(train["sentence_text"], train["relevant"], groups=groups)
print(f"Search time: {(time.time()-t0)/60:.1f} min")
print("Best grouped-CV macro-F1:", round(search_tfidf.best_score_, 4))
print("Best parameters:", search_tfidf.best_params_)

cv_results = pd.DataFrame(search_tfidf.cv_results_).sort_values("rank_test_score")
cv_results[["params", "mean_test_score", "std_test_score",
            "mean_train_score", "rank_test_score"]].head(15).to_csv(
    ARTIFACTS / "grid_tfidf_group_cv.csv", index=False)

# --- Evaluate once on test ---
best_tfidf = search_tfidf.best_estimator_
pred_tfidf = best_tfidf.predict(test["sentence_text"])
y_test = test["relevant"].to_numpy()

def metric_dict(name, y_true, y_pred, cv_score=None):
    result = {
        "model": name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    if cv_score is not None:
        result["cv_macro_f1"] = float(cv_score)
    return result

m_tfidf = metric_dict("TF-IDF + LinearSVC (grouped CV)",
                      y_test, pred_tfidf, search_tfidf.best_score_)
m_tfidf["selection_strategy"] = (
    "StratifiedGroupKFold over document-sentence connected components; "
    "TF-IDF fitted inside each fold"
)
m_tfidf["hyperparams"] = search_tfidf.best_params_
with open(ARTIFACTS / "metrics_tfidf.json", "w", encoding="utf-8") as f:
    json.dump(m_tfidf, f, indent=2)
print("\nTF-IDF test metrics:")
print(json.dumps({k: v for k, v in m_tfidf.items()
                  if k in ("accuracy", "precision", "recall", "macro_f1")}, indent=2))

# --- Confusion matrix ---
cm = confusion_matrix(y_test, pred_tfidf, labels=[0, 1])
ConfusionMatrixDisplay(cm, display_labels=["non-relevant", "relevant"]).plot(values_format="d")
plt.title("TF-IDF + LinearSVC — held-out test")
plt.tight_layout(); plt.savefig(FIG_DIR / "fig_cm_tfidf.png"); plt.show()

# --- Top features ---
clf_inner = best_tfidf.named_steps["clf"]
vec_inner = best_tfidf.named_steps["tfidf"]
coefs = clf_inner.coef_[0]
features = np.array(vec_inner.get_feature_names_out())
top_pos_idx = np.argsort(coefs)[-20:][::-1]
top_neg_idx = np.argsort(coefs)[:20]

print("\nTop 10 RELEVANT-indicative features:")
for i in top_pos_idx[:10]:
    print(f"  {features[i]:>30s}  coef = {coefs[i]:+.4f}")
print("\nTop 10 NON-RELEVANT-indicative features:")
for i in top_neg_idx[:10]:
    print(f"  {features[i]:>30s}  coef = {coefs[i]:+.4f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, idx, title, color in [
    (axes[0], top_pos_idx, "Top RELEVANT features", "#4c72b0"),
    (axes[1], top_neg_idx, "Top NON-RELEVANT features", "#dd8452")]:
    ax.barh(range(len(idx))[::-1], coefs[idx], color=color)
    ax.set_yticks(range(len(idx))[::-1]); ax.set_yticklabels(features[idx])
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_title(title); ax.set_xlabel("Linear coefficient")
plt.tight_layout(); plt.savefig(FIG_DIR / "fig_tfidf_features.png"); plt.show()

# %% [markdown]
# ## 5. SBERT and Hybrid Classifiers (CV-corrected)
#
# Same grouped CV protocol. SBERT embeddings cached to disk.
# Hybrid = TF-IDF (fit inside CV) + SBERT embeddings (L2-normalized).

# %%
print("\n" + "=" * 70)
print("STEP 5: SBERT AND HYBRID CLASSIFIERS (grouped CV)")
print("=" * 70)

try:
    from sentence_transformers import SentenceTransformer
    HAS_SBERT = True
except ImportError:
    HAS_SBERT = False
    print("sentence-transformers not installed. Skipping SBERT/hybrid.")
    print("Install with: pip install sentence-transformers")

if HAS_SBERT:
    EMB_TRAIN = ARTIFACTS / "train_emb.npy"
    EMB_TEST  = ARTIFACTS / "test_emb.npy"

    if EMB_TRAIN.exists() and EMB_TEST.exists():
        X_train_emb = np.load(EMB_TRAIN)
        X_test_emb  = np.load(EMB_TEST)
        embed_time = 0.0
        print("Loaded cached embeddings.")
    else:
        sbert = SentenceTransformer("all-MiniLM-L6-v2")
        t0 = time.time()
        X_train_emb = sbert.encode(train["sentence_text"].tolist(),
                                   batch_size=64, show_progress_bar=True,
                                   convert_to_numpy=True)
        X_test_emb  = sbert.encode(test["sentence_text"].tolist(),
                                   batch_size=64, show_progress_bar=True,
                                   convert_to_numpy=True)
        embed_time = time.time() - t0
        np.save(EMB_TRAIN, X_train_emb); np.save(EMB_TEST, X_test_emb)
        print(f"Embeddings computed in {embed_time:.1f}s and saved.")

    print("train emb shape:", X_train_emb.shape)

    # --- Pure SBERT ---
    pipe_svc = Pipeline([("clf", LinearSVC(random_state=42, max_iter=10000))])
    grid_svc = {"clf__C": [0.1, 0.3, 0.5, 1.0, 2.0],
                "clf__class_weight": [None, "balanced"]}
    search_svc = GridSearchCV(pipe_svc, grid_svc, cv=cv, scoring="f1_macro",
                              n_jobs=-1, refit=True, verbose=0)

    pipe_log = Pipeline([("clf", LogisticRegression(max_iter=2000, solver="liblinear",
                                                    random_state=42))])
    grid_log = {"clf__C": [0.3, 1.0, 2.0],
                "clf__class_weight": ["balanced"]}
    search_log = GridSearchCV(pipe_log, grid_log, cv=cv, scoring="f1_macro",
                              n_jobs=-1, refit=True, verbose=0)

    # NOTE: For SBERT, groups must be passed to fit for grouped CV.
    # GridSearchCV supports this via cv.split(X, y, groups) — but the
    # sklearn GridSearchCV API passes groups through `fit(X, y, groups=...)`.
    # For pure embedding data, we can pass groups directly.
    search_svc.fit(X_train_emb, train["relevant"], groups=groups)
    search_log.fit(X_train_emb, train["relevant"], groups=groups)

    if search_svc.best_score_ >= search_log.best_score_:
        best_sbert_search, best_sbert_family = search_svc, "LinearSVC"
    else:
        best_sbert_search, best_sbert_family = search_log, "LogReg"

    clf_sbert = best_sbert_search.best_estimator_
    t0 = time.time()
    pred_sbert = clf_sbert.predict(X_test_emb)
    infer_time = time.time() - t0

    m_sbert = metric_dict(f"SBERT + {best_sbert_family}",
                          y_test, pred_sbert, best_sbert_search.best_score_)
    m_sbert["embedding_model"] = "all-MiniLM-L6-v2"
    m_sbert["selection_strategy"] = "StratifiedGroupKFold on training set"
    m_sbert["embed_time_s"] = round(embed_time, 2)
    m_sbert["infer_time_s"] = round(infer_time, 4)
    m_sbert["hyperparams"] = best_sbert_search.best_params_
    with open(ARTIFACTS / "metrics_sbert.json", "w", encoding="utf-8") as f:
        json.dump(m_sbert, f, indent=2)
    print(f"\nSBERT best CV macro-F1: {best_sbert_search.best_score_:.4f}")
    print(f"SBERT test macro-F1   : {m_sbert['macro_f1']:.4f}")

    # --- Hybrid ---
    class TfidfTransform(BaseEstimator):
        """Fits TF-IDF inside each CV fold to prevent vocabulary leakage."""
        def __init__(self, max_features=10000, ngram_range=(1, 2), sublinear_tf=True):
            self.max_features = max_features
            self.ngram_range = ngram_range
            self.sublinear_tf = sublinear_tf
        def fit(self, X, y=None):
            self.vec_ = TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
                sublinear_tf=self.sublinear_tf,
            )
            self.vec_.fit(X)
            return self
        def transform(self, X):
            return self.vec_.transform(X)

    class HybridTransformer(BaseEstimator):
        """Combines TF-IDF (fit inside fold) with precomputed SBERT embeddings."""
        def __init__(self, emb):
            self.emb = emb
        def fit(self, X, y=None, groups=None):
            self.tfidf_ = TfidfVectorizer(max_features=10000, ngram_range=(1, 2),
                                          sublinear_tf=True)
            self.tfidf_.fit(X)
            return self
        def transform(self, X):
            tfidf_part = self.tfidf_.transform(X)
            emb_part = normalize(self.emb)
            return hstack([tfidf_part, emb_part]).tocsr()

    # Manual grouped CV loop for hybrid (TF-IDF refit per fold)
    from sklearn.model_selection import StratifiedGroupKFold as SGKF
    skf = SGKF(n_splits=5, shuffle=True, random_state=42)

    hybrid_scores_svc, hybrid_scores_log = [], []
    for tr_i, va_i in skf.split(train, train["relevant"], groups):
        X_tr_text = train["sentence_text"].iloc[tr_i]
        X_va_text = train["sentence_text"].iloc[va_i]
        emb_tr = X_train_emb[tr_i]
        emb_va = X_train_emb[va_i]

        # Fit TF-IDF on train fold only
        v = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), sublinear_tf=True)
        tf_tr = v.fit_transform(X_tr_text)
        tf_va = v.transform(X_va_text)
        X_tr = hstack([tf_tr, normalize(emb_tr)]).tocsr()
        X_va = hstack([tf_va, normalize(emb_va)]).tocsr()

        for C in [0.3, 1.0]:
            for cw in [None, "balanced"]:
                c = LinearSVC(C=C, class_weight=cw, random_state=42, max_iter=10000)
                c.fit(X_tr, train["relevant"].iloc[tr_i])
                p = c.predict(X_va)
                hybrid_scores_svc.append((C, cw, f1_score(train["relevant"].iloc[va_i], p,
                                                          average="macro", zero_division=0)))
        for C in [0.5, 1.0, 2.0]:
            c = LogisticRegression(C=C, class_weight="balanced", max_iter=2000,
                                   solver="liblinear", random_state=42)
            c.fit(X_tr, train["relevant"].iloc[tr_i])
            p = c.predict(X_va)
            hybrid_scores_log.append((C, "balanced", f1_score(train["relevant"].iloc[va_i], p,
                                                              average="macro", zero_division=0)))

    # Pick best hybrid config by mean CV score
    df_svc = pd.DataFrame(hybrid_scores_svc, columns=["C", "cw", "f1"])
    df_log = pd.DataFrame(hybrid_scores_log, columns=["C", "cw", "f1"])
    best_svc = df_svc.groupby(["C", "cw"])["f1"].mean().idxmax()
    best_log = df_log.groupby(["C", "cw"])["f1"].mean().idxmax()
    svc_score = df_svc.groupby(["C", "cw"])["f1"].mean().loc[best_svc]
    log_score = df_log.groupby(["C", "cw"])["f1"].mean().loc[best_log]

    print(f"\nHybrid LinearSVC best CV macro-F1: {svc_score:.4f} (C={best_svc[0]}, cw={best_svc[1]})")
    print(f"Hybrid LogReg    best CV macro-F1: {log_score:.4f} (C={best_log[0]}, cw={best_log[1]})")

    # Refit chosen hybrid on full train, evaluate on test
    v_full = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), sublinear_tf=True)
    tf_tr_full = v_full.fit_transform(train["sentence_text"])
    tf_te_full = v_full.transform(test["sentence_text"])
    X_tr_full = hstack([tf_tr_full, normalize(X_train_emb)]).tocsr()
    X_te_full = hstack([tf_te_full, normalize(X_test_emb)]).tocsr()

    if svc_score >= log_score:
        clf_hybrid = LinearSVC(C=best_svc[0], class_weight=best_svc[1],
                               random_state=42, max_iter=10000)
        clf_hybrid.fit(X_tr_full, train["relevant"])
        pred_hybrid = clf_hybrid.predict(X_te_full)
        m_hybrid = metric_dict("Hybrid (TF-IDF + SBERT) + LinearSVC",
                               y_test, pred_hybrid, svc_score)
        m_hybrid["hyperparams"] = {"C": best_svc[0], "class_weight": best_svc[1]}
    else:
        clf_hybrid = LogisticRegression(C=best_log[0], class_weight=best_log[1],
                                        max_iter=2000, solver="liblinear", random_state=42)
        clf_hybrid.fit(X_tr_full, train["relevant"])
        pred_hybrid = clf_hybrid.predict(X_te_full)
        m_hybrid = metric_dict("Hybrid (TF-IDF + SBERT) + LogReg",
                               y_test, pred_hybrid, log_score)
        m_hybrid["hyperparams"] = {"C": best_log[0], "class_weight": best_log[1]}

    m_hybrid["selection_strategy"] = (
        "StratifiedGroupKFold; TF-IDF fit inside each fold; "
        "SBERT embeddings precomputed"
    )
    with open(ARTIFACTS / "metrics_hybrid.json", "w", encoding="utf-8") as f:
        json.dump(m_hybrid, f, indent=2)
    print(f"\nHybrid test macro-F1: {m_hybrid['macro_f1']:.4f}")

    # --- Three-way comparison figure ---
    metrics_cols = ["accuracy", "precision", "recall", "macro_f1"]
    x = np.arange(len(metrics_cols)); w = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (label, mrow, color) in enumerate([
        ("Baseline (TF-IDF)", m_tfidf, "#8888ff"),
        ("SBERT", m_sbert, "#ff5555"),
        ("Hybrid", m_hybrid, "#55aa55")]):
        vals = [mrow[k] for k in metrics_cols]
        ax.bar(x + (i - 1) * w, vals, w, label=label, color=color)
        for j, v in enumerate(vals):
            ax.text(j + (i - 1) * w, v + 0.01, f"{v:.2f}", ha="center", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(metrics_cols)
    ax.set_ylim(0, 1); ax.set_ylabel("Score")
    ax.set_title("Baseline vs SBERT vs Hybrid (grouped CV)")
    ax.legend()
    plt.tight_layout(); plt.savefig(FIG_DIR / "fig_three_way_comparison.png"); plt.show()

# %% [markdown]
# ## 6. BERTopic Topic Discovery
#
# Applied to relevant sentences only. Reuses cached SBERT embeddings.
# Evaluates c_v coherence and topic diversity.

# %%
print("\n" + "=" * 70)
print("STEP 6: BERTOPIC TOPIC DISCOVERY")
print("=" * 70)

try:
    from bertopic import BERTopic
    from umap import UMAP
    from hdbscan import HDBSCAN
    from sentence_transformers import SentenceTransformer
    from gensim.corpora import Dictionary
    from gensim.models.coherencemodel import CoherenceModel
    HAS_BERTOPIC = True
except ImportError:
    HAS_BERTOPIC = False
    print("BERTopic stack not installed. Skipping topic discovery.")
    print("Install with: pip install bertopic umap-learn hdbscan gensim")

if HAS_BERTOPIC:
    rel = train[train["relevant"] == 1].reset_index(drop=True)
    print("relevant sentences:", len(rel))

    EMB_TRAIN = ARTIFACTS / "train_emb.npy"
    if EMB_TRAIN.exists():
        all_emb = np.load(EMB_TRAIN)
        emb = all_emb[train["relevant"].values == 1]
    else:
        sbert = SentenceTransformer("all-MiniLM-L6-v2")
        emb = sbert.encode(rel["sentence_text"].tolist(), batch_size=64,
                           show_progress_bar=True, convert_to_numpy=True)
    assert len(emb) == len(rel)

    umap_model = UMAP(n_neighbors=15, n_components=5, metric="cosine", random_state=42)
    hdbscan_model = HDBSCAN(min_cluster_size=20, metric="euclidean", prediction_data=True)
    topic_model = BERTopic(umap_model=umap_model, hdbscan_model=hdbscan_model,
                           calculate_probabilities=False, verbose=False)

    topics, _ = topic_model.fit_transform(rel["sentence_text"].tolist(), emb)
    info = topic_model.get_topic_info()
    print(info.head(10).to_string(index=False))

    # Save model + assignments
    rel_out = rel.copy(); rel_out["topic"] = topics
    rel_out.to_csv(ARTIFACTS / "rel_with_topics.csv", index=False)
    info.to_csv(ARTIFACTS / "topic_info.csv", index=False)

    # Coherence
    def topic_coherence(topic_model, docs, top_n=10):
        topics_words = []
        for tid in topic_model.get_topics():
            if tid == -1: continue
            topics_words.append([w for w, _ in topic_model.get_topic(tid)][:top_n])
        tokenized = [d.split() for d in docs]
        dic = Dictionary(tokenized)
        cm = CoherenceModel(topics=topics_words, texts=tokenized,
                            dictionary=dic, coherence="c_v")
        return cm.get_coherence()

    coh = topic_coherence(topic_model, rel["sentence_text"].tolist())

    # Diversity
    def topic_diversity(topic_model, top_n=10):
        words = []
        for tid in topic_model.get_topics():
            if tid == -1: continue
            words += [w for w, _ in topic_model.get_topic(tid)][:top_n]
        return len(set(words)) / max(len(words), 1)

    div = topic_diversity(topic_model)

    metrics_bt = {
        "n_topics": int(len(info) - 1),
        "n_outliers": int((np.array(topics) == -1).sum()),
        "coherence_cv": round(float(coh), 4),
        "diversity": round(float(div), 4),
        "min_cluster_size": 20,
    }
    with open(ARTIFACTS / "metrics_bertopic.json", "w", encoding="utf-8") as f:
        json.dump(metrics_bt, f, indent=2)
    print("\nBERTopic metrics:", json.dumps(metrics_bt, indent=2))

    # Topic size figure
    sizes = info[info["Topic"] != -1][["Topic", "Count"]].sort_values("Count", ascending=False).head(20)
    plt.figure(figsize=(10, 4))
    plt.bar(sizes["Topic"].astype(str), sizes["Count"], color="#55a868")
    plt.xlabel("topic id"); plt.ylabel("sentence count")
    plt.title("Top-20 topics by size")
    plt.tight_layout(); plt.savefig(FIG_DIR / "fig_topic_sizes.png"); plt.show()

# %% [markdown]
# ## 7. Results Comparison & Discussion
#
# Consolidate all metrics into final tables and a draft results paragraph.

# %%
print("\n" + "=" * 70)
print("STEP 7: RESULTS COMPARISON")
print("=" * 70)

def load_json(path):
    return json.load(open(path)) if os.path.exists(path) else None

eda_final   = load_json(ARTIFACTS / "eda_summary.json")
split_final = load_json(ARTIFACTS / "split_info.json")
m_base_f    = load_json(ARTIFACTS / "metrics_tfidf.json")
m_sbert_f   = load_json(ARTIFACTS / "metrics_sbert.json")
m_hybrid_f  = load_json(ARTIFACTS / "metrics_hybrid.json")
m_topic_f   = load_json(ARTIFACTS / "metrics_bertopic.json")

# --- Table 1: Dataset summary ---
rows = [
    ("Total sentences (raw)", eda_final["raw_rows"]),
    ("Relevant sentences", eda_final["relevant"]),
    ("Non-relevant sentences", eda_final["non_relevant"]),
    ("Imbalance ratio", f"{eda_final['imbalance_ratio']}:1"),
    ("Documents", eda_final["num_documents"]),
    ("Sources", eda_final["num_sources"]),
    ("Mean sentence length (words)", eda_final["avg_words"]),
    ("Median sentence length (words)", eda_final["median_words"]),
    ("Vocabulary size", eda_final["vocab_size"]),
    ("Train sentences (80%)", split_final["train_size"]),
    ("Test sentences (20%)", split_final["test_size"]),
    ("Train documents", split_final["train_docs"]),
    ("Test documents", split_final["test_docs"]),
    ("Document overlap", split_final["overlap_docs"]),
]
table1 = pd.DataFrame(rows, columns=["Attribute", "Value"])
table1.to_csv(ARTIFACTS / "table_dataset_summary.csv", index=False)
print("\nTable 1 — Dataset Summary:")
print(table1.to_string(index=False))

# --- Table 2: Classification comparison ---
cls_rows = [m for m in [m_base_f, m_sbert_f, m_hybrid_f] if m]
table2 = pd.DataFrame(cls_rows)[["model", "accuracy", "precision", "recall", "macro_f1"]].round(4)
table2.to_csv(ARTIFACTS / "table_classification.csv", index=False)
print("\nTable 2 — Classification Comparison:")
print(table2.to_string(index=False))

if len(table2) > 0:
    best_idx = table2["macro_f1"].idxmax()
    print(f"\nBest model: {table2.iloc[best_idx]['model']} "
          f"(macro-F1 = {table2.iloc[best_idx]['macro_f1']:.4f})")

# --- Table 3: Topic discovery ---
if m_topic_f:
    table3 = pd.DataFrame([
        ("Topics discovered", m_topic_f["n_topics"]),
        ("Coherence (c_v)", m_topic_f["coherence_cv"]),
        ("Diversity", m_topic_f["diversity"]),
        ("Outlier sentences", m_topic_f["n_outliers"]),
        ("Min cluster size", m_topic_f["min_cluster_size"]),
    ], columns=["Metric", "Value"])
    table3.to_csv(ARTIFACTS / "table_topic_discovery.csv", index=False)
    print("\nTable 3 — Topic Discovery:")
    print(table3.to_string(index=False))

# --- Draft results paragraph ---
print("\n" + "=" * 70)
print("DRAFT RESULTS PARAGRAPH (edit into fluent prose)")
print("=" * 70)
if m_base_f and m_sbert_f and m_hybrid_f:
    base_f1 = m_base_f["macro_f1"]
    sbert_f1 = m_sbert_f["macro_f1"]
    hybrid_f1 = m_hybrid_f["macro_f1"]
    best_name = table2.iloc[best_idx]["model"]
    print(f"\nThe SMAH-CTI dataset contains {eda_final['raw_rows']:,} sentences across "
          f"{eda_final['num_documents']} documents, with {eda_final['relevant']:,} "
          f"({eda_final['relevant']/eda_final['raw_rows']:.1%}) relevant and "
          f"{eda_final['non_relevant']:,} ({eda_final['non_relevant']/eda_final['raw_rows']:.1%}) "
          f"non-relevant sentences, giving an imbalance ratio of {eda_final['imbalance_ratio']}:1. "
          f"A document-level 80/20 split produced {split_final['train_size']:,} training "
          f"and {split_final['test_size']:,} test sentences with zero document overlap.")
    print(f"\nClassification performance (macro-F1 on held-out test): "
          f"TF-IDF baseline = {base_f1:.4f}, "
          f"SBERT = {sbert_f1:.4f}, "
          f"Hybrid = {hybrid_f1:.4f}. "
          f"The best-performing model was {best_name}.")
    if m_topic_f:
        print(f"\nBERTopic discovered {m_topic_f['n_topics']} topics with "
              f"c_v coherence = {m_topic_f['coherence_cv']:.4f} and "
              f"topic diversity = {m_topic_f['diversity']:.4f}, "
              f"leaving {m_topic_f['n_outliers']:,} sentences as outliers.")

# %% [markdown]
# ## 8. Final Summary

# %%
print("\n" + "=" * 70)
print("PIPELINE COMPLETE")
print("=" * 70)
print(f"\nAll artifacts saved to: {ARTIFACTS}")
print("\nKey outputs:")
for f in sorted(ARTIFACTS.glob("*.json")) + sorted(ARTIFACTS.glob("*.csv")):
    print(f"  {f.name}")
print(f"\nFigures saved to: {FIG_DIR}")
for f in sorted(FIG_DIR.glob("*.png")):
    print(f"  {f.name}")

print("\n" + "=" * 70)
print("METHODOLOGY NOTES FOR THESIS")
print("=" * 70)
print("""
1. All models use StratifiedGroupKFold over connected components of
   documents sharing exact sentence text, preventing document leakage.

2. TF-IDF is fitted inside each CV fold (both baseline and hybrid),
   preventing vocabulary/IDF leakage from validation folds.

3. Hyperparameters are selected by mean grouped-CV macro-F1, not by
   test performance. The test set is evaluated exactly once per model.

4. The document-level split asserts zero document overlap and zero
   sentence-text overlap between train and test.

5. Reported test metrics reflect generalisation performance, not
   selection bias. Generalisation gap = CV macro-F1 − test macro-F1.
""")