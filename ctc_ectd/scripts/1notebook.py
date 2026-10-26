"""
1notebook.py
Generates 5 stage-wise Jupyter notebooks for the CTI Threat Analysis pipeline.

Run:
    cd D:\Projects\done\thesis\ctc_ectd
    python scripts\1notebook.py

Produces:
    notebooks/01_data_extraction_and_cleaning.ipynb
    notebooks/02_exploratory_data_analysis.ipynb
    notebooks/03_model_training_and_evaluation.ipynb
    notebooks/04_document_classification_and_prediction.ipynb
    notebooks/05_temporal_topic_modeling_and_emerging_threats.ipynb
"""

from pathlib import Path
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(r"D:\Projects\done\thesis\ctc_ectd")
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Cell helpers
# ---------------------------------------------------------------------------
def md(text: str):
    return new_markdown_cell(text)


def code(text: str):
    cell = new_code_cell(text)
    cell["execution_count"] = None
    cell["outputs"] = []
    return cell


def write_notebook(name: str, cells):
    nb = new_notebook(cells=cells)
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.10.11",
        },
    }
    path = NOTEBOOKS_DIR / f"{name}.ipynb"
    with path.open("w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    print(f"Wrote: {path}")


# ---------------------------------------------------------------------------
# Shared bootstrap cell (identical in all notebooks)
# ---------------------------------------------------------------------------
BOOTSTRAP = r'''
# ---------------------------------------------------------------------------
# Bootstrap — paths, imports, global config
# ---------------------------------------------------------------------------
import os
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(r"D:\Projects\done\thesis\ctc_ectd")
SCRIPTS_DIR  = PROJECT_ROOT / "scripts"
UPLOAD_DIR   = PROJECT_ROOT / "uploads"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

for d in (SCRIPTS_DIR, UPLOAD_DIR, ARTIFACT_DIR, PROCESSED_DIR):
    d.mkdir(parents=True, exist_ok=True)

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

RANDOM_STATE = 42
plt.rcParams["figure.figsize"] = (10, 5)
sns.set_theme(style="whitegrid")

print("PROJECT_ROOT :", PROJECT_ROOT)
print("PROCESSED_DIR:", PROCESSED_DIR)
print("ARTIFACT_DIR :", ARTIFACT_DIR)
print("UPLOAD_DIR   :", UPLOAD_DIR)
'''


# ===========================================================================
# Notebook 01 — Data extraction & cleaning
# ===========================================================================
cells_01 = [
    md("""# 01 — Data Extraction & Cleaning

Pipeline stage 1 of 5.

Goals:
1. Load CTI documents (PDF / DOCX / TXT / HTML / MD) from `uploads/`
2. Load the labelled training corpus (`dataset/synthetic_cti_reports.csv`)
3. Detect key columns (text, title, label, date)
4. Clean, dedupe, and build a unified `model_text` field

Saves:
- `data/processed/cleaned_reports.pkl`
- `data/processed/uploaded_docs.pkl`
"""),
    code(BOOTSTRAP),
    md("## 1. Document extraction (uploads/)"),
    code(r'''
from document_extractor import (
    extract_text,
    clean_extracted_text,
    extract_and_clean,
    EXTRACTORS,
)

print("Supported extensions:", ", ".join(sorted(EXTRACTORS)))


def load_documents(folder: Path) -> pd.DataFrame:
    """Walk `folder` and extract text from every supported document."""
    rows = []
    for path in sorted(folder.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in EXTRACTORS:
            continue
        try:
            raw = extract_text(path)
            cleaned = clean_extracted_text(raw)
            rows.append({
                "file_name": path.name,
                "file_path": str(path),
                "extension": path.suffix.lower(),
                "char_count": len(cleaned),
                "word_count": len(cleaned.split()),
                "raw_text": raw,
                "text": cleaned,
            })
        except Exception as exc:
            print(f"[WARN] {path.name}: {exc}")
            rows.append({
                "file_name": path.name,
                "file_path": str(path),
                "extension": path.suffix.lower(),
                "char_count": 0,
                "word_count": 0,
                "raw_text": "",
                "text": "",
            })
    return pd.DataFrame(rows)
'''),
    code(r'''
uploaded_df = load_documents(UPLOAD_DIR)

if uploaded_df.empty:
    print("No documents found in", UPLOAD_DIR)
    print("Writing empty uploaded_docs.pkl so downstream notebooks do not break.")
    uploaded_df = pd.DataFrame(columns=[
        "file_name", "file_path", "extension",
        "char_count", "word_count", "raw_text", "text",
    ])
else:
    print(f"Extracted text from {len(uploaded_df)} document(s).")
    display(uploaded_df[["file_name", "extension", "char_count", "word_count"]]
            .sort_values("char_count", ascending=False))

uploaded_df.to_pickle(PROCESSED_DIR / "uploaded_docs.pkl")
print("Saved:", PROCESSED_DIR / "uploaded_docs.pkl")
'''),
    md("## 2. Load labelled training corpus"),
    code(r'''
DATA_PATH = PROJECT_ROOT / "dataset" / "synthetic_cti_reports.csv"
if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}\n"
        "Place synthetic_cti_reports.csv inside the dataset/ folder."
    )

df = pd.read_csv(DATA_PATH)
print("Training corpus shape:", df.shape)
display(df.head())
print("\nColumns:", df.columns.tolist())
'''),
    md("## 3. Detect key columns"),
    code(r'''
def find_column(columns, candidates):
    lookup = {str(col).strip().lower(): col for col in columns}
    for cand in candidates:
        if cand in lookup:
            return lookup[cand]
    return None


TEXT_COL  = find_column(df.columns,
    ["text", "report_text", "description", "content",
     "narrative", "report", "body"])
TITLE_COL = find_column(df.columns, ["title", "report_title", "name"])
LABEL_COL = find_column(df.columns,
    ["attack_type", "label", "category", "threat_type", "class", "target"])
DATE_COL  = find_column(df.columns,
    ["date", "published_date", "report_date", "created_at", "timestamp"])

print("TEXT_COL :", TEXT_COL)
print("TITLE_COL:", TITLE_COL)
print("LABEL_COL:", LABEL_COL)
print("DATE_COL :", DATE_COL)

if TEXT_COL is None:
    raise ValueError("Text column not detected.")
if LABEL_COL is None:
    raise ValueError("Label column not detected.")
'''),
    md("## 4. Clean & dedupe"),
    code(r'''
df = df.copy()
df[TEXT_COL]  = df[TEXT_COL].fillna("").astype(str).str.strip()
df[LABEL_COL] = df[LABEL_COL].fillna("").astype(str).str.strip()

df = df[(df[TEXT_COL] != "") & (df[LABEL_COL] != "")].copy()
df = df.drop_duplicates(subset=[TEXT_COL]).reset_index(drop=True)

if TITLE_COL is not None:
    df[TITLE_COL] = df[TITLE_COL].fillna("").astype(str).str.strip()
    df["model_text"] = (df[TITLE_COL] + " " + df[TEXT_COL]).str.strip()
else:
    df["model_text"] = df[TEXT_COL]

print("Cleaned shape:", df.shape)
display(df[LABEL_COL].value_counts().to_frame("report_count"))
'''),
    md("## 5. Save for downstream notebooks"),
    code(r'''
payload = {
    "df": df,
    "TEXT_COL": TEXT_COL,
    "TITLE_COL": TITLE_COL,
    "LABEL_COL": LABEL_COL,
    "DATE_COL": DATE_COL,
}
pd.to_pickle(payload, PROCESSED_DIR / "cleaned_reports.pkl")
print("Saved:", PROCESSED_DIR / "cleaned_reports.pkl")
'''),
]


# ===========================================================================
# Notebook 02 — EDA
# ===========================================================================
cells_02 = [
    md("""# 02 — Exploratory Data Analysis

Pipeline stage 2 of 5.

Loads `cleaned_reports.pkl` and produces every plot from the original monolith:
- Label distribution
- Severity
- Target sector
- Temporal distribution
- Threat actor
- Malware family
- Attack vector
- Country
- MITRE ATT&CK techniques
- Cramér's V heatmap
- Text-length / word-count by category
"""),
    code(BOOTSTRAP),
    code(r'''
payload = pd.read_pickle(PROCESSED_DIR / "cleaned_reports.pkl")
df = payload["df"]
TEXT_COL  = payload["TEXT_COL"]
TITLE_COL = payload["TITLE_COL"]
LABEL_COL = payload["LABEL_COL"]
DATE_COL  = payload["DATE_COL"]

print("Loaded:", df.shape)
print("LABEL_COL:", LABEL_COL, "| DATE_COL:", DATE_COL)


def find_column(columns, candidates):
    lookup = {str(col).strip().lower(): col for col in columns}
    for cand in candidates:
        if cand in lookup:
            return lookup[cand]
    return None
'''),
    md("## 1. Distribution of threat categories"),
    code(r'''
plt.figure(figsize=(10, 5))
sns.countplot(data=df, y=LABEL_COL,
              order=df[LABEL_COL].value_counts().index)
plt.title("Distribution of CTI Threat Categories")
plt.xlabel("Number of Reports"); plt.ylabel("Threat Category")
plt.tight_layout(); plt.show()

df["text_length"] = df["model_text"].str.len()
plt.figure(figsize=(9, 5))
sns.histplot(df["text_length"], bins=30, kde=True)
plt.title("Distribution of CTI Report Text Lengths")
plt.xlabel("Characters per Report"); plt.ylabel("Frequency")
plt.tight_layout(); plt.show()
'''),
    md("## 2. Detailed label distribution"),
    code(r'''
label_counts = df[LABEL_COL].value_counts()
label_pct = (label_counts / len(df) * 100).round(2)
display(pd.DataFrame({"Count": label_counts, "Percentage": label_pct}))

fig, ax = plt.subplots(figsize=(12, 6))
colors = plt.cm.Set3(np.linspace(0, 1, len(label_counts)))
bars = ax.barh(label_counts.index, label_counts.values, color=colors)
ax.set_xlabel("Number of Reports"); ax.set_ylabel("Threat Category")
ax.set_title("CTI Report Distribution by Threat Category")
for bar, pct in zip(bars, label_pct.values):
    ax.text(bar.get_width() + 10,
            bar.get_y() + bar.get_height() / 2,
            f"{pct}%", ha="left", va="center", fontsize=9)
plt.tight_layout(); plt.show()
'''),
    md("## 3. Severity distribution"),
    code(r'''
SEVERITY_COL = find_column(df.columns,
    ["severity", "threat_level", "risk_level", "priority"])
if SEVERITY_COL is not None:
    s_counts = df[SEVERITY_COL].value_counts()
    display(s_counts.to_frame("Count"))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ["#ff6b6b", "#feca57", "#48dbfb", "#1dd1a1"]
    axes[0].pie(s_counts.values, labels=s_counts.index,
                autopct="%1.1f%%", colors=colors[:len(s_counts)],
                startangle=90)
    axes[0].set_title("Severity Distribution")
    s_counts.plot(kind="bar", ax=axes[1], color=colors[:len(s_counts)])
    axes[1].set_title("Reports by Severity")
    plt.tight_layout(); plt.show()

    ct = pd.crosstab(df[LABEL_COL], df[SEVERITY_COL])
    plt.figure(figsize=(10, 6))
    sns.heatmap(ct, annot=True, fmt="d", cmap="YlOrRd")
    plt.title("Category vs Severity")
    plt.tight_layout(); plt.show()
else:
    print("No severity column.")
'''),
    md("## 4. Target sector"),
    code(r'''
SECTOR_COL = find_column(df.columns,
    ["target_sector", "sector", "industry", "target_industry"])
if SECTOR_COL is not None:
    sec_counts = df[SECTOR_COL].value_counts()
    display(sec_counts.to_frame("Count"))
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.Paired(np.linspace(0, 1, len(sec_counts)))
    ax.barh(sec_counts.index, sec_counts.values, color=colors)
    ax.set_title("Threats by Target Sector"); ax.invert_yaxis()
    plt.tight_layout(); plt.show()

    sec_cat = pd.crosstab(df[SECTOR_COL], df[LABEL_COL], normalize="index")
    sec_cat.plot(kind="barh", stacked=True, figsize=(14, 8), colormap="tab20")
    plt.title("Threat Category Mix by Sector")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout(); plt.show()
else:
    print("No target sector column.")
'''),
    md("## 5. Temporal distribution"),
    code(r'''
if DATE_COL is not None:
    df["parsed_date"] = pd.to_datetime(df[DATE_COL], errors="coerce")
    vd = df.dropna(subset=["parsed_date"]).copy()
    if len(vd):
        vd["year"]  = vd["parsed_date"].dt.year
        vd["month"] = vd["parsed_date"].dt.month
        vd["ym"]    = vd["parsed_date"].dt.to_period("M")
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        yc = vd["year"].value_counts().sort_index()
        axes[0].bar(yc.index.astype(str), yc.values, color="steelblue")
        axes[0].set_title("Reports by Year")
        mc = vd["month"].value_counts().sort_index()
        axes[1].bar(mc.index, mc.values, color="coral")
        axes[1].set_title("Reports by Month")
        axes[1].set_xticks(range(1, 13))
        plt.tight_layout(); plt.show()

        series = vd.groupby("ym").size()
        plt.figure(figsize=(14, 5))
        plt.plot(series.index.astype(str), series.values, marker="o", ms=3)
        plt.xticks(rotation=45, ha="right")
        plt.title("Report Volume Over Time")
        plt.tight_layout(); plt.show()
    else:
        print("No valid dates.")
else:
    print("No date column.")
'''),
    md("## 6. Threat actor"),
    code(r'''
ACTOR_COL = find_column(df.columns,
    ["threat_actor", "actor", "attacker", "group"])
if ACTOR_COL is not None:
    ac = df[ACTOR_COL].value_counts()
    display(ac.head(15).to_frame("Report Count"))
    top = ac.head(15)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(top)), top.values,
            color=plt.cm.viridis(np.linspace(0.2, 0.8, len(top))))
    ax.set_yticks(range(len(top))); ax.set_yticklabels(top.index)
    ax.invert_yaxis(); ax.set_title("Top Threat Actors")
    plt.tight_layout(); plt.show()
else:
    print("No threat_actor column.")
'''),
    md("## 7. Malware family"),
    code(r'''
MALWARE_COL = find_column(df.columns,
    ["malware", "malware_family", "tool", "payload"])
if MALWARE_COL is not None:
    mc = df[MALWARE_COL].value_counts()
    display(mc.to_frame("Count"))
    top = mc.head(12)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(top)), top.values,
            color=plt.cm.plasma(np.linspace(0.2, 0.8, len(top))))
    ax.set_yticks(range(len(top))); ax.set_yticklabels(top.index)
    ax.invert_yaxis(); ax.set_title("Top Malware Families")
    plt.tight_layout(); plt.show()
else:
    print("No malware column.")
'''),
    md("## 8. Attack vector"),
    code(r'''
VECTOR_COL = find_column(df.columns,
    ["attack_vector", "vector", "attack_method", "initial_access"])
if VECTOR_COL is not None:
    vc = df[VECTOR_COL].value_counts()
    display(vc.to_frame("Count"))
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].barh(vc.index, vc.values,
                 color=plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(vc))))
    axes[0].invert_yaxis(); axes[0].set_title("Attack Vectors")
    vcat = pd.crosstab(df[VECTOR_COL], df[LABEL_COL], normalize="index")
    sns.heatmap(vcat, annot=True, fmt=".2f", cmap="coolwarm", ax=axes[1])
    axes[1].set_title("Category Proportion by Vector")
    plt.tight_layout(); plt.show()
else:
    print("No attack_vector column.")
'''),
    md("## 9. Geography"),
    code(r'''
COUNTRY_COL = find_column(df.columns,
    ["country", "nation", "region", "location"])
if COUNTRY_COL is not None:
    cc = df[COUNTRY_COL].value_counts()
    display(cc.head(15).to_frame("Count"))
    top = cc.head(15)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(top)), top.values,
            color=plt.cm.Blues(np.linspace(0.3, 0.9, len(top))))
    ax.set_yticks(range(len(top))); ax.set_yticklabels(top.index)
    ax.invert_yaxis(); ax.set_title("Top Countries")
    plt.tight_layout(); plt.show()
else:
    print("No country column.")
'''),
    md("## 10. MITRE ATT&CK techniques"),
    code(r'''
MITRE_COL = find_column(df.columns,
    ["mitre_technique", "technique", "attack_technique", "mitre_id"])
MITRE_NAME_COL = find_column(df.columns,
    ["mitre_technique_name", "technique_name", "mitre_name"])
if MITRE_COL is not None:
    tc = df[MITRE_COL].value_counts()
    display(tc.head(15).to_frame("Count"))
    top = tc.head(15)
    if MITRE_NAME_COL is not None:
        names = df.groupby(MITRE_COL)[MITRE_NAME_COL].first()
        labels = [f"{t}: {str(names.get(t, ''))[:30]}" for t in top.index]
    else:
        labels = list(top.index)
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh(range(len(top)), top.values,
            color=plt.cm.Reds(np.linspace(0.3, 0.9, len(top))))
    ax.set_yticks(range(len(top))); ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis(); ax.set_title("Top MITRE ATT&CK Techniques")
    plt.tight_layout(); plt.show()
else:
    print("No MITRE technique column.")
'''),
    md("## 11. Cramér's V — categorical associations"),
    code(r'''
from scipy.stats import chi2_contingency


def cramers_v(x, y):
    cm = pd.crosstab(x, y)
    chi2 = chi2_contingency(cm)[0]
    n = cm.sum().sum()
    phi2 = chi2 / n
    r, k = cm.shape
    phi2c = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rc = r - ((r - 1) ** 2) / (n - 1)
    kc = k - ((k - 1) ** 2) / (n - 1)
    return np.sqrt(phi2c / max(min(kc, rc) - 1, 1))


feats = [LABEL_COL]
for name in ["severity", "target_sector", "attack_vector",
             "country", "malware", "threat_actor", "mitre_technique"]:
    col = find_column(df.columns, [name])
    if col is not None and col not in feats:
        feats.append(col)

if len(feats) >= 2:
    n = len(feats)
    M = pd.DataFrame(np.zeros((n, n)), index=feats, columns=feats)
    for i in range(n):
        for j in range(n):
            if i == j:
                M.iloc[i, j] = 1.0
            elif i < j:
                try:
                    v = cramers_v(df[feats[i]], df[feats[j]])
                except Exception:
                    v = 0
                M.iloc[i, j] = M.iloc[j, i] = v
    plt.figure(figsize=(10, 8))
    sns.heatmap(M, annot=True, fmt=".3f", cmap="coolwarm",
                vmin=0, vmax=1, square=True)
    plt.title("Cramér's V"); plt.tight_layout(); plt.show()
'''),
    md("## 12. Text / word length by category"),
    code(r'''
if "text_length" not in df.columns:
    df["text_length"] = df["model_text"].str.len()

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sns.boxplot(data=df, x=LABEL_COL, y="text_length", ax=axes[0], palette="Set3")
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha="right")
axes[0].set_title("Text Length by Category")
sns.violinplot(data=df, x=LABEL_COL, y="text_length", ax=axes[1], palette="Set3")
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha="right")
axes[1].set_title("Text Length (Violin)")
plt.tight_layout(); plt.show()

df["word_count"] = df["model_text"].str.split().str.len()
display(df.groupby(LABEL_COL)["word_count"]
        .agg(["mean", "median", "min", "max"]).round(1))
'''),
    md("""## Key EDA findings

- 10 balanced threat categories (~10% each)
- Severity, sector, actor, malware, vector, country, MITRE all show category-specific skew
- Temporal volume is non-uniform → strong signal for NMF topic-drift detection
- Cramér's V quantifies which categorical fields are most predictive
- Text/word length varies by category → informs vectorizer parameters
"""),
]


# ===========================================================================
# Notebook 03 — Model training & evaluation
# ===========================================================================
cells_03 = [
    md("""# 03 — Model Training & Evaluation

Pipeline stage 3 of 5.

Loads `cleaned_reports.pkl`, splits stratified 80/20, trains 3 baselines
(Logistic Regression, Multinomial Naive Bayes, Linear SVM), compares them,
and saves the best model.

Saves:
- `artifacts/cti_text_classifier.joblib`
- `artifacts/model_results.csv`
"""),
    code(BOOTSTRAP),
    code(r'''
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    ConfusionMatrixDisplay,
)
import joblib

payload = pd.read_pickle(PROCESSED_DIR / "cleaned_reports.pkl")
df = payload["df"]
LABEL_COL = payload["LABEL_COL"]
print("Loaded:", df.shape, "| LABEL_COL:", LABEL_COL)
'''),
    md("## 1. Stratified split"),
    code(r'''
X = df["model_text"]
y = df[LABEL_COL]
cc = y.value_counts()
strat = y.nunique() > 1 and cc.min() >= 2

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE,
    stratify=y if strat else None,
)
print("Train:", len(X_train), "| Test:", len(X_test),
      "| Classes:", y_train.nunique())
'''),
    md("## 2. Define baselines"),
    code(r'''
models = {
    "Logistic Regression": Pipeline([
        ("tfidf", TfidfVectorizer(max_features=10000,
                                  ngram_range=(1, 2),
                                  stop_words="english",
                                  sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=2000,
                                   class_weight="balanced",
                                   random_state=RANDOM_STATE)),
    ]),
    "Multinomial Naive Bayes": Pipeline([
        ("tfidf", TfidfVectorizer(max_features=10000,
                                  ngram_range=(1, 2),
                                  stop_words="english")),
        ("clf", MultinomialNB()),
    ]),
    "Linear SVM": Pipeline([
        ("tfidf", TfidfVectorizer(max_features=10000,
                                  ngram_range=(1, 2),
                                  stop_words="english",
                                  sublinear_tf=True)),
        ("clf", LinearSVC(class_weight="balanced",
                          random_state=RANDOM_STATE)),
    ]),
}
'''),
    md("## 3. Train + evaluate"),
    code(r'''
results = []
trained = {}
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    p, r, f, _ = precision_recall_fscore_support(
        y_test, preds, average="weighted", zero_division=0)
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, preds),
        "Weighted Precision": p,
        "Weighted Recall": r,
        "Weighted F1": f,
    })
    trained[name] = model

results_df = (pd.DataFrame(results)
              .sort_values("Weighted F1", ascending=False)
              .reset_index(drop=True))
display(results_df.round(4))
'''),
    md("## 4. Detailed report + confusion matrix for the best model"),
    code(r'''
best_model_name = results_df.iloc[0]["Model"]
best_model = trained[best_model_name]
y_pred = best_model.predict(X_test)

print("Best baseline:", best_model_name)
print(classification_report(y_test, y_pred, zero_division=0))

ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred, xticks_rotation=45, cmap="Blues")
plt.title(f"Confusion Matrix — {best_model_name}")
plt.tight_layout(); plt.show()
'''),
    md("## 5. Save model + results"),
    code(r'''
joblib.dump(best_model, ARTIFACT_DIR / "cti_text_classifier.joblib")
results_df.to_csv(ARTIFACT_DIR / "model_results.csv", index=False)

print("Saved:", ARTIFACT_DIR / "cti_text_classifier.joblib")
print("Saved:", ARTIFACT_DIR / "model_results.csv")
print("Selected model:", best_model_name)
'''),
]


# ===========================================================================
# Notebook 04 — Document classification & prediction
# ===========================================================================
cells_04 = [
    md("""# 04 — Document Classification & Ad-hoc Prediction

Pipeline stage 4 of 5.

Loads:
- `artifacts/cti_text_classifier.joblib`
- `data/processed/uploaded_docs.pkl`

Classifies every uploaded document and demonstrates ad-hoc snippet prediction.

Saves:
- `artifacts/classified_documents.csv`
"""),
    code(BOOTSTRAP),
    code(r'''
import joblib

best_model = joblib.load(ARTIFACT_DIR / "cti_text_classifier.joblib")
print("Model loaded:", type(best_model).__name__)
print("Steps:", list(best_model.named_steps))

uploaded_df = pd.read_pickle(PROCESSED_DIR / "uploaded_docs.pkl")
print("Uploaded docs:", len(uploaded_df))
'''),
    md("## 1. Classify uploaded documents"),
    code(r'''
if not uploaded_df.empty and (uploaded_df["text"] != "").any():
    docs = uploaded_df[uploaded_df["text"] != ""].copy()
    docs["predicted_category"] = best_model.predict(docs["text"])

    clf = best_model.named_steps["clf"]
    if hasattr(clf, "predict_proba"):
        probs = best_model.predict_proba(docs["text"])
        docs["confidence"] = probs.max(axis=1).round(4)
    elif hasattr(clf, "decision_function"):
        scores = best_model.decision_function(docs["text"])
        docs["confidence"] = np.max(scores, axis=1).round(4)
    else:
        docs["confidence"] = np.nan

    display(docs[["file_name", "extension", "word_count",
                  "predicted_category", "confidence"]]
            .sort_values("confidence", ascending=False))

    plt.figure(figsize=(10, 5))
    docs["predicted_category"].value_counts().plot(kind="bar",
                                                   color="steelblue")
    plt.title("Predicted Categories for Extracted Documents")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout(); plt.show()

    docs.to_csv(ARTIFACT_DIR / "classified_documents.csv", index=False)
    print("Saved:", ARTIFACT_DIR / "classified_documents.csv")
else:
    print("No extracted documents available to classify.")
    print("Add files to:", UPLOAD_DIR, "and re-run notebook 01.")
    pd.DataFrame(columns=["file_name", "predicted_category", "confidence"]) \
        .to_csv(ARTIFACT_DIR / "classified_documents.csv", index=False)
'''),
    md("## 2. Ad-hoc prediction on a new CTI snippet"),
    code(r'''
new_report = """
Multiple organizations report suspicious credential theft activity.
Attackers use phishing emails to obtain account credentials
and attempt unauthorized access to corporate systems.
"""

predicted_category = best_model.predict([new_report])[0]
print("Predicted category:", predicted_category)

clf = best_model.named_steps["clf"]
if hasattr(clf, "predict_proba"):
    proba = best_model.predict_proba([new_report])[0]
    classes = best_model.named_steps["clf"].classes_
    for cls, p in sorted(zip(classes, proba), key=lambda x: -x[1])[:5]:
        print(f"  {cls:<20} {p:.4f}")
'''),
]


# ===========================================================================
# Notebook 05 — Temporal topic modeling + emerging threats
# ===========================================================================
cells_05 = [
    md("""# 05 — Temporal Topic Modeling & Emerging-Threat Detection

Pipeline stage 5 of 5.

Loads `cleaned_reports.pkl` + `uploaded_docs.pkl`, builds a temporal corpus,
fits a 5-topic NMF model, tracks monthly topic shares, and flags emerging
patterns.

Saves:
- `artifacts/monthly_topic_shares.csv`
- `artifacts/emerging_threat_alerts.csv`
"""),
    code(BOOTSTRAP),
    code(r'''
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

payload = pd.read_pickle(PROCESSED_DIR / "cleaned_reports.pkl")
df = payload["df"]
LABEL_COL = payload["LABEL_COL"]
DATE_COL  = payload["DATE_COL"]
uploaded_df = pd.read_pickle(PROCESSED_DIR / "uploaded_docs.pkl")

print("Training rows:", len(df), "| Uploaded docs:", len(uploaded_df))
print("DATE_COL:", DATE_COL)
'''),
    md("## 1. Build temporal corpus"),
    code(r'''
DATE_RE = re.compile(r"(20\d{2})[-_/](\d{1,2})")


def guess_date(row):
    m = DATE_RE.search(row.get("file_name", ""))
    if m:
        return pd.Timestamp(year=int(m.group(1)), month=int(m.group(2)), day=1)
    m = DATE_RE.search(str(row.get("text", ""))[:500])
    if m:
        return pd.Timestamp(year=int(m.group(1)), month=int(m.group(2)), day=1)
    return pd.NaT


if not uploaded_df.empty:
    docs = uploaded_df[uploaded_df["text"] != ""].copy()
    docs["parsed_date"] = docs.apply(guess_date, axis=1)
    docs["model_text"] = docs["text"]
    docs["source"] = "uploaded"
else:
    docs = pd.DataFrame(columns=["file_name", "text", "model_text",
                                 "parsed_date", "source"])

if DATE_COL is not None:
    df["parsed_date"] = pd.to_datetime(df[DATE_COL], errors="coerce")
else:
    df["parsed_date"] = pd.NaT
df["source"] = "training"

temporal_df = pd.concat(
    [df[["model_text", "parsed_date", "source", LABEL_COL]],
     docs.reindex(columns=["model_text", "parsed_date", "source", LABEL_COL])],
    ignore_index=True,
).dropna(subset=["parsed_date"]).copy()

temporal_df["month"] = (temporal_df["parsed_date"]
                        .dt.to_period("M").astype(str))

print("Temporal corpus size:", len(temporal_df))
print("Months covered:", temporal_df["month"].nunique())
print("Sources:", temporal_df["source"].value_counts().to_dict())
'''),
    md("## 2. Fit NMF topics"),
    code(r'''
if len(temporal_df) >= 5:
    vec = TfidfVectorizer(max_features=5000, stop_words="english",
                          ngram_range=(1, 2))
    M = vec.fit_transform(temporal_df["model_text"])
    k = min(5, M.shape[0], M.shape[1])

    if k >= 2:
        nmf = NMF(n_components=k, random_state=RANDOM_STATE,
                  init="nndsvda", max_iter=500)
        W = nmf.fit_transform(M)
        terms = vec.get_feature_names_out()

        for i, comp in enumerate(nmf.components_):
            top = comp.argsort()[-10:][::-1]
            print(f"Topic {i+1}:", ", ".join(terms[j] for j in top))

        for i in range(k):
            temporal_df[f"Topic_{i+1}"] = W[:, i]
        temporal_df["dominant_topic"] = W.argmax(axis=1) + 1

        monthly_topic_counts = pd.crosstab(
            temporal_df["month"], temporal_df["dominant_topic"])
        monthly_topic_shares = monthly_topic_counts.div(
            monthly_topic_counts.sum(axis=1), axis=0).fillna(0)

        display(monthly_topic_shares.round(3))
        monthly_topic_shares.plot(figsize=(12, 6), marker="o")
        plt.title("Monthly Distribution of Dominant CTI Topics")
        plt.xlabel("Month"); plt.ylabel("Share of Reports")
        plt.tight_layout(); plt.show()

        monthly_topic_shares.to_csv(ARTIFACT_DIR / "monthly_topic_shares.csv")
        print("Saved:", ARTIFACT_DIR / "monthly_topic_shares.csv")
    else:
        print("Not enough terms to fit multiple topics.")
else:
    print("Insufficient dated reports for temporal topic modeling.")
'''),
    md("## 3. Emerging-threat detection"),
    code(r'''
EMERGING_MIN_SHARE_JUMP = 0.10   # 10 percentage points
EMERGING_MIN_COUNT      = 2

alerts = []
if "monthly_topic_shares" in dir():
    months = list(monthly_topic_shares.index)
    for i in range(1, len(months)):
        prev_m, curr_m = months[i - 1], months[i]
        for tid in monthly_topic_shares.columns:
            ps = monthly_topic_shares.loc[prev_m, tid]
            cs = monthly_topic_shares.loc[curr_m, tid]
            cnt = monthly_topic_counts.loc[curr_m, tid]
            if (cs - ps) >= EMERGING_MIN_SHARE_JUMP and cnt >= EMERGING_MIN_COUNT:
                alerts.append({
                    "Previous Month": prev_m,
                    "Current Month": curr_m,
                    "Topic": tid,
                    "Previous Share": round(ps, 3),
                    "Current Share": round(cs, 3),
                    "Increase (pp)": round(cs - ps, 3),
                    "Current Report Count": cnt,
                })

alerts_df = pd.DataFrame(alerts)
if not alerts_df.empty:
    alerts_df = alerts_df.sort_values("Increase (pp)", ascending=False)
    display(alerts_df)
    alerts_df.to_csv(ARTIFACT_DIR / "emerging_threat_alerts.csv", index=False)
    print("\nSaved alerts to:", ARTIFACT_DIR / "emerging_threat_alerts.csv")
else:
    print("No topic passed the emerging-threat threshold.")
    print("This does not prove absence of emerging threats.")
    pd.DataFrame(columns=["Previous Month", "Current Month", "Topic",
                          "Previous Share", "Current Share",
                          "Increase (pp)", "Current Report Count"]) \
        .to_csv(ARTIFACT_DIR / "emerging_threat_alerts.csv", index=False)
'''),
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    write_notebook("01_data_extraction_and_cleaning", cells_01)
    write_notebook("02_exploratory_data_analysis", cells_02)
    write_notebook("03_model_training_and_evaluation", cells_03)
    write_notebook("04_document_classification_and_prediction", cells_04)
    write_notebook("05_temporal_topic_modeling_and_emerging_threats", cells_05)
    print("\nAll 5 notebooks generated in:", NOTEBOOKS_DIR)


if __name__ == "__main__":
    main()