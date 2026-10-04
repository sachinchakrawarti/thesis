<#
.SYNOPSIS
    Cleans up the current project layout and moves files into the correct structure.
.DESCRIPTION
    - Removes duplicate setup_project.ps1 and utilits/ folder
    - Moves cti.ipynb into notebooks\
    - Moves generate_cti_dataset.py into scripts\ as _generate_dataset.py
    - Creates missing src\ modules
#>

$ErrorActionPreference = "Stop"
$Root = "D:\Projects\done\thesis\Implementation"
Set-Location $Root

Write-Host "`n=== Fixing Project Layout ===" -ForegroundColor Cyan

# ---------- 1. Remove typos / duplicates ----------
Write-Host "`n[1/6] Removing duplicates and typos..." -ForegroundColor Yellow

$toRemove = @(
    (Join-Path $Root "setup_project.ps1"),
    (Join-Path $Root "utilits")
)
foreach ($item in $toRemove) {
    if (Test-Path $item) {
        Remove-Item -Path $item -Recurse -Force
        Write-Host "  [-] Removed: $item" -ForegroundColor Green
    } else {
        Write-Host "  [=] Not found: $item" -ForegroundColor Gray
    }
}

# ---------- 2. Ensure folders exist ----------
Write-Host "`n[2/6] Ensuring folders exist..." -ForegroundColor Yellow
$folders = @("dataset","notebooks","src","scripts","output",
             "output\models","output\figures","output\reports","output\topics")
foreach ($f in $folders) {
    $p = Join-Path $Root $f
    if (-not (Test-Path $p)) {
        New-Item -ItemType Directory -Path $p -Force | Out-Null
        Write-Host "  [+] $f" -ForegroundColor Green
    } else {
        Write-Host "  [=] $f" -ForegroundColor Gray
    }
}

# ---------- 3. Move root notebook into notebooks\ ----------
Write-Host "`n[3/6] Moving loose notebook into notebooks\..." -ForegroundColor Yellow
$rootNotebook = Join-Path $Root "cti.ipynb"
if (Test-Path $rootNotebook) {
    $dest = Join-Path $Root "notebooks\00_overview.ipynb"
    Move-Item -Path $rootNotebook -Destination $dest -Force
    Write-Host "  [>] cti.ipynb -> notebooks\00_overview.ipynb" -ForegroundColor Green
} else {
    Write-Host "  [=] No root-level cti.ipynb to move" -ForegroundColor Gray
}

# ---------- 4. Write Python source modules ----------
Write-Host "`n[4/6] Writing src\ Python modules..." -ForegroundColor Yellow
$src = Join-Path $Root "src"

# __init__.py
Set-Content -Path (Join-Path $src "__init__.py") -Value "# CTI Thesis source package" -Encoding UTF8
Write-Host "  [+] src\__init__.py" -ForegroundColor Green

# config.py
$config = @'
"""Central configuration for the CTI thesis project."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR  = PROJECT_ROOT / "dataset"
OUTPUT_DIR   = PROJECT_ROOT / "output"
MODELS_DIR   = OUTPUT_DIR / "models"
FIGURES_DIR  = OUTPUT_DIR / "figures"
REPORTS_DIR  = OUTPUT_DIR / "reports"
TOPICS_DIR   = OUTPUT_DIR / "topics"

for d in [MODELS_DIR, FIGURES_DIR, REPORTS_DIR, TOPICS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

DATASET_FILE = DATASET_DIR / "synthetic_cti_reports.csv"

MAX_FEATURES = 10000
NGRAM_RANGE  = (1, 2)
MIN_DF       = 2
MAX_DF       = 0.95
RANDOM_STATE = 42
TEST_SIZE    = 0.20

N_TOPICS     = 10
TOP_N_WORDS  = 10

THREAT_CATEGORIES = ["Malware", "Phishing", "Ransomware",
                     "DDoS", "Botnets", "Spam"]

ETS_WEIGHTS = {
    "novelty":     0.20,
    "growth":      0.30,
    "persistence": 0.20,
    "activity":    0.15,
    "term_change": 0.15,
}
'@
Set-Content -Path (Join-Path $src "config.py") -Value $config -Encoding UTF8
Write-Host "  [+] src\config.py" -ForegroundColor Green

# preprocess.py
$pre = @'
"""Text cleaning and normalization for CTI documents."""
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

for pkg in ["stopwords", "wordnet", "omw-1.4", "punkt"]:
    try:
        nltk.data.find(f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

_STOP = set(stopwords.words("english"))
_LEM  = WordNetLemmatizer()

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z0-9\s\.\-:/_]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = []
    for tok in text.split():
        if tok in _STOP and len(tok) < 4:
            continue
        if re.fullmatch(r"\d+", tok):
            continue
        tokens.append(_LEM.lemmatize(tok))
    return " ".join(tokens)
'@
Set-Content -Path (Join-Path $src "preprocess.py") -Value $pre -Encoding UTF8
Write-Host "  [+] src\preprocess.py" -ForegroundColor Green

# features.py
$feat = @'
"""Feature extraction utilities."""
from sklearn.feature_extraction.text import TfidfVectorizer
from .config import MAX_FEATURES, NGRAM_RANGE, MIN_DF, MAX_DF

def build_tfidf(texts, **kwargs):
    params = dict(
        max_features=MAX_FEATURES,
        ngram_range=NGRAM_RANGE,
        min_df=MIN_DF,
        max_df=MAX_DF,
        sublinear_tf=True,
    )
    params.update(kwargs)
    vec = TfidfVectorizer(**params)
    X = vec.fit_transform(texts)
    return X, vec
'@
Set-Content -Path (Join-Path $src "features.py") -Value $feat -Encoding UTF8
Write-Host "  [+] src\features.py" -ForegroundColor Green

# classify.py
$cls = @'
"""Train and evaluate multiple classifiers."""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix)
from .config import TEST_SIZE, RANDOM_STATE

def _models():
    return {
        "NaiveBayes":         MultinomialNB(),
        "LogisticRegression": LogisticRegression(max_iter=1000, n_jobs=-1),
        "SVM":                LinearSVC(),
        "RandomForest":       RandomForestClassifier(n_estimators=200,
                                                     n_jobs=-1,
                                                     random_state=RANDOM_STATE),
    }

def train_all_models(X, y):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    results = []
    fitted  = {}
    for name, clf in _models().items():
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        results.append({
            "Model":     name,
            "Accuracy":  round(accuracy_score(y_te, y_pred), 4),
            "Precision": round(precision_score(y_te, y_pred, average="macro",
                                               zero_division=0), 4),
            "Recall":    round(recall_score(y_te, y_pred, average="macro",
                                            zero_division=0), 4),
            "F1":        round(f1_score(y_te, y_pred, average="macro",
                                        zero_division=0), 4),
            "CM":        confusion_matrix(y_te, y_pred, labels=np.unique(y)),
        })
        fitted[name] = (clf, y_te, y_pred)
    return results, fitted
'@
Set-Content -Path (Join-Path $src "classify.py") -Value $cls -Encoding UTF8
Write-Host "  [+] src\classify.py" -ForegroundColor Green

# topic_model.py
$tm = @'
"""LDA and NMF topic modeling."""
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from .config import N_TOPICS, TOP_N_WORDS, RANDOM_STATE

def _top_words(components, feature_names, n=TOP_N_WORDS):
    out = []
    for row in components:
        idx = np.argsort(row)[::-1][:n]
        out.append([feature_names[i] for i in idx])
    return out

def run_lda(docs, n_topics=N_TOPICS):
    vec = CountVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)
    X   = vec.fit_transform(docs)
    lda = LatentDirichletAllocation(
        n_components=n_topics, random_state=RANDOM_STATE,
        learning_method="batch", max_iter=20
    )
    lda.fit(X)
    return lda, _top_words(lda.components_, vec.get_feature_names_out())

def run_nmf(docs, n_topics=N_TOPICS):
    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)
    X   = vec.fit_transform(docs)
    nmf = NMF(n_components=n_topics, random_state=RANDOM_STATE,
              init="nndsvd", max_iter=400)
    nmf.fit(X)
    return nmf, _top_words(nmf.components_, vec.get_feature_names_out())
'@
Set-Content -Path (Join-Path $src "topic_model.py") -Value $tm -Encoding UTF8
Write-Host "  [+] src\topic_model.py" -ForegroundColor Green

# temporal.py
$tmp = @'
"""Temporal aggregation of topic prevalence."""
import pandas as pd

def topic_prevalence_by_period(doc_topic, dates, freq="M"):
    df = pd.DataFrame(
        doc_topic,
        columns=[f"topic_{i+1}" for i in range(doc_topic.shape[1])]
    )
    df["date"] = pd.to_datetime(dates.values)
    df["period"] = df["date"].dt.to_period(freq)
    cols = [c for c in df.columns if c.startswith("topic_")]
    return df.groupby("period")[cols].mean()
'@
Set-Content -Path (Join-Path $src "temporal.py") -Value $tmp -Encoding UTF8
Write-Host "  [+] src\temporal.py" -ForegroundColor Green

# emerging.py
$emg = @'
"""Emerging Threat Score (ETS) computation."""
import numpy as np
import pandas as pd
from .config import ETS_WEIGHTS

def _normalize(series):
    s = series.astype(float).fillna(0.0)
    lo, hi = s.min(), s.max()
    if hi - lo < 1e-9:
        return s * 0.0
    return (s - lo) / (hi - lo)

def compute_emerging_scores(prevalence_df):
    rows = []
    topics = [c for c in prevalence_df.columns if c.startswith("topic_")]
    for topic in topics:
        series = prevalence_df[topic]
        novelty     = series.iloc[0] == 0
        growth_vals = series.pct_change().replace([np.inf, -np.inf], 0).fillna(0)
        growth      = growth_vals.max()
        persistence = (series > series.mean()).sum() / len(series)
        activity    = series.mean()
        term_change = series.std()
        rows.append({
            "Topic":       topic,
            "Novelty":     int(novelty),
            "Growth":      growth,
            "Persistence": persistence,
            "Activity":    activity,
            "TermChange":  term_change,
        })
    df = pd.DataFrame(rows)
    for c in ["Novelty", "Growth", "Persistence", "Activity", "TermChange"]:
        df[c] = _normalize(df[c])
    df["ETS"] = (
        ETS_WEIGHTS["novelty"]     * df["Novelty"]     +
        ETS_WEIGHTS["growth"]      * df["Growth"]      +
        ETS_WEIGHTS["persistence"] * df["Persistence"] +
        ETS_WEIGHTS["activity"]    * df["Activity"]    +
        ETS_WEIGHTS["term_change"] * df["TermChange"]
    )
    return df.sort_values("ETS", ascending=False).reset_index(drop=True)
'@
Set-Content -Path (Join-Path $src "emerging.py") -Value $emg -Encoding UTF8
Write-Host "  [+] src\emerging.py" -ForegroundColor Green

# visualize.py
$vis = @'
"""Visualization helpers."""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
from .config import FIGURES_DIR

sns.set_style("whitegrid")

def plot_confusion_matrices(fitted_models, X, y):
    for name, (clf, y_te, y_pred) in fitted_models.items():
        labels = np.unique(y_te)
        cm = confusion_matrix(y_te, y_pred, labels=labels)
        plt.figure(figsize=(7, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=labels, yticklabels=labels)
        plt.title(f"Confusion Matrix - {name}")
        plt.xlabel("Predicted"); plt.ylabel("Actual")
        plt.tight_layout()
        out = FIGURES_DIR / f"cm_{name}.png"
        plt.savefig(out, dpi=150); plt.close()
        print(f"Saved: {out}")
'@
Set-Content -Path (Join-Path $src "visualize.py") -Value $vis -Encoding UTF8
Write-Host "  [+] src\visualize.py" -ForegroundColor Green

# ---------- 5. Move dataset generator into scripts\ ----------
Write-Host "`n[5/6] Organizing scripts..." -ForegroundColor Yellow
$oldGen = Join-Path $Root "utilits\generate_cti_dataset.py"
if (Test-Path $oldGen) {
    $newGen = Join-Path $Root "scripts\_generate_dataset.py"
    Move-Item -Path $oldGen -Destination $newGen -Force
    Write-Host "  [>] generate_cti_dataset.py -> scripts\_generate_dataset.py" -ForegroundColor Green
}

# ---------- 6. Verify ----------
Write-Host "`n[6/6] Final verification..." -ForegroundColor Yellow
Get-ChildItem -Path $Root -Recurse -Depth 2 |
    Where-Object { $_.FullName -notmatch "\\.venv" } |
    Select-Object @{n="Path";e={$_.FullName.Replace($Root, ".")}} |
    Format-Table -AutoSize

Write-Host "`n=== Layout fixed. ===" -ForegroundColor Green
Write-Host "Next: run .\00_setup.ps1 (or re-run if Python env already active)" -ForegroundColor Yellow