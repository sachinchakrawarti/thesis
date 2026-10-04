<#
.SYNOPSIS
    Master setup script for the M.Tech thesis project:
    "Real-Time Cyber Threat Classification and Emerging Threat Detection
     Using Machine Learning and Temporal Topic Modeling"

.DESCRIPTION
    This script:
      1. Checks Python installation
      2. Creates the complete folder structure
      3. Creates a virtual environment
      4. Installs all required dependencies
      5. Writes Python source modules (src\*.py)
      6. Generates the synthetic CTI dataset
      7. Creates all Jupyter notebooks
      8. Verifies the final project layout

.USAGE
    cd D:\Projects\done\thesis\Implementation
    .\00_setup.ps1

.NOTES
    Author : Sachin Chakrawarti (0228IT23MT13)
    Guide  : Prof. Sapna Choudhary
    Dept   : P.G. Computer Sc. & Engg., Shri Ram Group of Institution, Jabalpur
#>

# ============================================================
#  CONFIG
# ============================================================
$ErrorActionPreference = "Stop"
$Root       = "D:\Projects\done\thesis\Implementation"
$VenvPath   = Join-Path $Root ".venv"
$DatasetDir = Join-Path $Root "dataset"
$Dataset    = Join-Path $DatasetDir "synthetic_cti_reports.csv"
$SrcDir     = Join-Path $Root "src"
$NotebookDir= Join-Path $Root "notebooks"
$OutputDir  = Join-Path $Root "output"
$ScriptsDir = Join-Path $Root "scripts"

function Write-Step($n, $msg) {
    Write-Host "`n[$n] $msg" -ForegroundColor Yellow
}
function Write-OK($msg) { Write-Host "    [+] $msg" -ForegroundColor Green }
function Write-Skip($msg) { Write-Host "    [=] $msg" -ForegroundColor Gray }

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  CTI Thesis Project - Master Setup" -ForegroundColor Cyan
Write-Host "  $Root" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# ============================================================
#  1. CHECK PYTHON
# ============================================================
Write-Step "1/8" "Checking Python installation"
try {
    $pyVersion = (python --version 2>&1).ToString()
    Write-OK $pyVersion
} catch {
    Write-Host "    [!] Python not found on PATH." -ForegroundColor Red
    Write-Host "        Install Python 3.10+ from https://www.python.org/downloads/" -ForegroundColor Red
    exit 1
}

# ============================================================
#  2. CREATE FOLDER STRUCTURE
# ============================================================
Write-Step "2/8" "Creating folder structure"
$folders = @(
    $Root,
    $DatasetDir,
    $NotebookDir,
    $SrcDir,
    $ScriptsDir,
    $OutputDir,
    (Join-Path $OutputDir "models"),
    (Join-Path $OutputDir "figures"),
    (Join-Path $OutputDir "reports"),
    (Join-Path $OutputDir "topics")
)
foreach ($f in $folders) {
    if (-not (Test-Path $f)) {
        New-Item -ItemType Directory -Path $f -Force | Out-Null
        Write-OK $f
    } else {
        Write-Skip $f
    }
}

# ============================================================
#  3. CREATE VIRTUAL ENVIRONMENT
# ============================================================
Write-Step "3/8" "Creating Python virtual environment"
if (-not (Test-Path $VenvPath)) {
    python -m venv $VenvPath
    Write-OK "Created: $VenvPath"
} else {
    Write-Skip "Already exists: $VenvPath"
}

$activate = Join-Path $VenvPath "Scripts\Activate.ps1"
if (Test-Path $activate) {
    & $activate
    Write-OK "Activated venv"
} else {
    Write-Host "    [!] Activation script missing." -ForegroundColor Red
    exit 1
}

# ============================================================
#  4. INSTALL DEPENDENCIES
# ============================================================
Write-Step "4/8" "Installing Python dependencies"
python -m pip install --upgrade pip setuptools wheel | Out-Null

$requirements = @"
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
scipy>=1.10.0
nltk>=3.8.1
gensim>=4.3.0
beautifulsoup4>=4.12.0
pyLDAvis>=3.4.0
matplotlib>=3.7.0
seaborn>=0.12.0
wordcloud>=1.9.2
jupyter>=1.0.0
notebook>=7.0.0
ipykernel>=6.25.0
tqdm>=4.65.0
joblib>=1.3.0
"@
Set-Content -Path (Join-Path $Root "requirements.txt") -Value $requirements -Encoding UTF8
Write-OK "requirements.txt"

python -m pip install -r (Join-Path $Root "requirements.txt")
Write-OK "Dependencies installed"

# ============================================================
#  5. WRITE PYTHON SOURCE MODULES
# ============================================================
Write-Step "5/8" "Writing Python source modules"

# ---------- src\__init__.py ----------
Set-Content -Path (Join-Path $SrcDir "__init__.py") `
    -Value "# CTI Thesis source package" -Encoding UTF8
Write-OK "src\__init__.py"

# ---------- src\config.py ----------
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
Set-Content -Path (Join-Path $SrcDir "config.py") -Value $config -Encoding UTF8
Write-OK "src\config.py"

# ---------- src\preprocess.py ----------
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
Set-Content -Path (Join-Path $SrcDir "preprocess.py") -Value $pre -Encoding UTF8
Write-OK "src\preprocess.py"

# ---------- src\features.py ----------
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
Set-Content -Path (Join-Path $SrcDir "features.py") -Value $feat -Encoding UTF8
Write-OK "src\features.py"

# ---------- src\classify.py ----------
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
Set-Content -Path (Join-Path $SrcDir "classify.py") -Value $cls -Encoding UTF8
Write-OK "src\classify.py"

# ---------- src\topic_model.py ----------
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
Set-Content -Path (Join-Path $SrcDir "topic_model.py") -Value $tm -Encoding UTF8
Write-OK "src\topic_model.py"

# ---------- src\temporal.py ----------
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
Set-Content -Path (Join-Path $SrcDir "temporal.py") -Value $tmp -Encoding UTF8
Write-OK "src\temporal.py"

# ---------- src\emerging.py ----------
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
Set-Content -Path (Join-Path $SrcDir "emerging.py") -Value $emg -Encoding UTF8
Write-OK "src\emerging.py"

# ---------- src\visualize.py ----------
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
Set-Content -Path (Join-Path $SrcDir "visualize.py") -Value $vis -Encoding UTF8
Write-OK "src\visualize.py"

# ============================================================
#  6. GENERATE SYNTHETIC DATASET
# ============================================================
Write-Step "6/8" "Generating synthetic CTI dataset"

if (Test-Path $Dataset) {
    Write-Skip "Dataset already exists: $Dataset"
} else {
    $genScript = Join-Path $ScriptsDir "_generate_dataset.py"
    $pyCode = @'
"""Generate a synthetic CTI dataset for the thesis experiments."""
import random
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd

random.seed(42)
OUT = Path(r"D:\Projects\done\thesis\Implementation\dataset\synthetic_cti_reports.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)

CATEGORIES = {
    "Malware":    ["A new malware sample {name} spreads via {vector} using {tech}.",
                   "Malware {name} contacts C2 {ip} and domain {domain}."],
    "Phishing":   ["Phishing campaign targeting {sector} uses spoofed emails from {domain}.",
                   "Credential harvesting at {domain} mimics login portal ({theme})."],
    "Ransomware": ["Ransomware {name} encrypts files and demands Bitcoin. Hash: {hash}.",
                   "Ransomware {name} exploits {cve} against {sector}."],
    "DDoS":       ["DDoS attack peaking {gbps} Gbps targeted {sector}. Sources include {ip}.",
                   "Volumetric DDoS on port {port} lasted {hours} hours."],
    "Botnets":    ["Botnet {name} recruits IoT via default credentials, C2 at {ip}.",
                   "Mirai-variant {name} scans port {port} and uses {tech}."],
    "Spam":       ["Spam campaign distributing {name} from IP range {ip}.",
                   "Bulk spam impersonating {sector} redirects to {domain}."],
}
NAMES   = ["Emotet","TrickBot","Qakbot","LockBit","BlackCat","Conti","Mirai","Gafgyt","AgentTesla","RedLine"]
VECTORS = ["email attachments","malicious URLs","USB drives","exploit kits","RDP brute force"]
TECHS   = ["polymorphic packing","process hollowing","reflective DLL injection","obfuscated PowerShell"]
SECTORS = ["healthcare","finance","government","education","energy","retail","manufacturing"]
THEMES  = ["invoice","payroll","COVID-19 update","tax refund","package delivery","password reset"]
DOMAINS = ["secure-login-update.com","cdn-delivery.net","mail-verify.org","account-support.co"]
CVES    = ["CVE-2024-21412","CVE-2023-4966","CVE-2024-3400","CVE-2023-38831"]

def rip():  return f"{random.randint(1,223)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"
def rhash():return "".join(random.choices("abcdef0123456789", k=64))
def rport():return random.choice([22,23,25,80,443,445,3389,8080])

def make_row(cat, doc_id, date):
    tpl = random.choice(CATEGORIES[cat])
    text = tpl.format(name=random.choice(NAMES), vector=random.choice(VECTORS),
                      tech=random.choice(TECHS), sector=random.choice(SECTORS),
                      theme=random.choice(THEMES), domain=random.choice(DOMAINS),
                      ip=rip(), hash=rhash(), cve=random.choice(CVES),
                      gbps=random.randint(50,800), port=rport(),
                      hours=random.randint(1,72))
    return {"id": f"CTI-{doc_id:05d}", "title": f"{cat} Report {doc_id}",
            "text": text, "category": cat,
            "date": date.strftime("%Y-%m-%d"),
            "source": random.choice(["security-blog","advisory","incident-report",
                                     "vendor-report","forum"]),
            "url": f"https://example-cti.org/report/{doc_id}"}

rows, doc_id = [], 1
start = datetime(2024, 1, 1)
for cat in CATEGORIES:
    for _ in range(random.randint(130, 170)):
        rows.append(make_row(cat, doc_id, start + timedelta(days=random.randint(0, 365))))
        doc_id += 1

df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv(OUT, index=False)
print(f"Wrote {len(df)} rows -> {OUT}")
print(df["category"].value_counts())
'@
    Set-Content -Path $genScript -Value $pyCode -Encoding UTF8
    python $genScript
    Write-OK "Dataset created: $Dataset"
}

# ============================================================
#  7. CREATE JUPYTER NOTEBOOKS
# ============================================================
Write-Step "7/8" "Creating Jupyter notebooks"

function New-NB {
    param([string]$Path, [array]$Cells)
    $nb = [ordered]@{
        cells = @($Cells)
        metadata = [ordered]@{
            kernelspec = [ordered]@{
                display_name = "Python 3"
                language     = "python"
                name         = "python3"
            }
            language_info = [ordered]@{
                name = "python"; version = "3.10"
            }
        }
        nbformat = 4; nbformat_minor = 5
    }
    $nb | ConvertTo-Json -Depth 20 | Set-Content -Path $Path -Encoding UTF8
}
function MD([string]$t) {
    [ordered]@{ cell_type="markdown"; metadata=@{}; source=@($t) }
}
function CODE([string[]]$lines) {
    [ordered]@{ cell_type="code"; execution_count=$null; metadata=@{};
                outputs=@(); source=$lines }
}

# --- 01_eda.ipynb ---
New-NB -Path (Join-Path $NotebookDir "01_eda.ipynb") -Cells @(
  (MD "# 01 - Exploratory Data Analysis`n`nCTI dataset overview."),
  (CODE @(
    "import sys, pathlib",
    "sys.path.append(str(pathlib.Path.cwd().parent))",
    "import pandas as pd, numpy as np",
    "import matplotlib.pyplot as plt, seaborn as sns",
    "from src.config import DATASET_FILE, FIGURES_DIR",
    "sns.set_style('whitegrid')",
    "df = pd.read_csv(DATASET_FILE)",
    "print(df.shape); df.head()"
  )),
  (CODE @(
    "print(df.isnull().sum())",
    "df['date'] = pd.to_datetime(df['date'])",
    "df['month'] = df['date'].dt.to_period('M')",
    "df['category'].value_counts().plot(kind='bar'); plt.tight_layout(); plt.show()"
  ))
)
Write-OK "01_eda.ipynb"

# --- 02_preprocessing.ipynb ---
New-NB -Path (Join-Path $NotebookDir "02_preprocessing.ipynb") -Cells @(
  (MD "# 02 - Preprocessing & Feature Extraction"),
  (CODE @(
    "import sys, pathlib",
    "sys.path.append(str(pathlib.Path.cwd().parent))",
    "import pandas as pd",
    "from src.preprocess import clean_text",
    "from src.features import build_tfidf",
    "from src.config import DATASET_FILE",
    "df = pd.read_csv(DATASET_FILE)",
    "df['clean_text'] = df['text'].apply(clean_text)",
    "X, vec = build_tfidf(df['clean_text'])",
    "print('TF-IDF shape:', X.shape)"
  ))
)
Write-OK "02_preprocessing.ipynb"

# --- 03_classification.ipynb ---
New-NB -Path (Join-Path $NotebookDir "03_classification.ipynb") -Cells @(
  (MD "# 03 - Threat Classification"),
  (CODE @(
    "import sys, pathlib",
    "sys.path.append(str(pathlib.Path.cwd().parent))",
    "import pandas as pd",
    "from src.preprocess import clean_text",
    "from src.features import build_tfidf",
    "from src.classify import train_all_models",
    "from src.visualize import plot_confusion_matrices",
    "from src.config import DATASET_FILE",
    "df = pd.read_csv(DATASET_FILE)",
    "df['clean_text'] = df['text'].apply(clean_text)",
    "X, vec = build_tfidf(df['clean_text'])",
    "y = df['category'].values",
    "results, models = train_all_models(X, y)",
    "import pandas as pd; pd.DataFrame(results)[['Model','Accuracy','Precision','Recall','F1']]"
  )),
  (CODE @("plot_confusion_matrices(models, X, y)"))
)
Write-OK "03_classification.ipynb"

# --- 04_topic_modeling.ipynb ---
New-NB -Path (Join-Path $NotebookDir "04_topic_modeling.ipynb") -Cells @(
  (MD "# 04 - Topic Modeling (LDA & NMF)"),
  (CODE @(
    "import sys, pathlib",
    "sys.path.append(str(pathlib.Path.cwd().parent))",
    "import pandas as pd",
    "from src.preprocess import clean_text",
    "from src.topic_model import run_lda, run_nmf",
    "from src.config import DATASET_FILE, N_TOPICS",
    "df = pd.read_csv(DATASET_FILE)",
    "df['clean_text'] = df['text'].apply(clean_text)",
    "docs = df['clean_text'].tolist()",
    "lda_model, lda_topics = run_lda(docs, n_topics=N_TOPICS)",
    "for i,t in enumerate(lda_topics): print(f'LDA Topic {i}:', t)",
    "nmf_model, nmf_topics = run_nmf(docs, n_topics=N_TOPICS)",
    "for i,t in enumerate(nmf_topics): print(f'NMF Topic {i}:', t)"
  ))
)
Write-OK "04_topic_modeling.ipynb"

# --- 05_emerging_threats.ipynb ---
New-NB -Path (Join-Path $NotebookDir "05_emerging_threats.ipynb") -Cells @(
  (MD "# 05 - Temporal Topic Analysis & Emerging Threat Scoring"),
  (CODE @(
    "import sys, pathlib",
    "sys.path.append(str(pathlib.Path.cwd().parent))",
    "import pandas as pd",
    "from src.preprocess import clean_text",
    "from src.topic_model import run_nmf",
    "from src.temporal import topic_prevalence_by_period",
    "from src.emerging import compute_emerging_scores",
    "from src.config import DATASET_FILE, N_TOPICS",
    "df = pd.read_csv(DATASET_FILE)",
    "df['clean_text'] = df['text'].apply(clean_text)",
    "df['date'] = pd.to_datetime(df['date'])",
    "docs = df['clean_text'].tolist()",
    "nmf_model, topics = run_nmf(docs, n_topics=N_TOPICS)",
    "doc_topic = nmf_model.transform(__import__('sklearn.feature_extraction.text', fromlist=['TfidfVectorizer']).TfidfVectorizer(max_features=5000, ngram_range=(1,2)).fit_transform(docs))",
    "prev = topic_prevalence_by_period(doc_topic, df['date'])",
    "prev",
    "ets = compute_emerging_scores(prev)",
    "ets"
  ))
)
Write-OK "05_emerging_threats.ipynb"

# ============================================================
#  8. FINAL VERIFICATION
# ============================================================
Write-Step "8/8" "Verifying project layout"
Get-ChildItem -Path $Root -Recurse -Depth 2 |
    Where-Object { $_.FullName -notmatch "\\.venv" } |
    Select-Object FullName |
    Format-Table -AutoSize

Write-Host "`n==========================================================" -ForegroundColor Cyan
Write-Host "  SETUP COMPLETE" -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  Project root : $Root"
Write-Host "  Dataset      : $Dataset"
Write-Host "  Notebooks    : $NotebookDir"
Write-Host "  Source code  : $SrcDir"
Write-Host "  Output       : $OutputDir"
Write-Host ""
Write-Host "  NEXT STEPS:" -ForegroundColor Yellow
Write-Host "   1. Activate venv:  & '$activate'"
Write-Host "   2. Launch Jupyter: jupyter notebook"
Write-Host "   3. Open notebooks\01_eda.ipynb and run cells top-to-bottom."
Write-Host ""
Write-Host "  Or run the full pipeline from PowerShell:" -ForegroundColor Yellow
Write-Host "   python -c `"from src.classify import train_all_models; ...`""
Write-Host "==========================================================" -ForegroundColor Cyan