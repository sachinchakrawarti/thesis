# scripts/analyze_uploads.py
"""
CLI version of the notebook pipeline:
  1. Extract text from every file in uploads/
  2. Classify each document with a trained TF-IDF + Logistic Regression model
  3. Build a monthly topic model (NMF) from parsed dates
  4. Flag emerging threats

Run:
    python scripts/analyze_uploads.py
"""

import re
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import NMF
from sklearn.pipeline import Pipeline

# Make scripts/ importable
sys.path.insert(0, str(Path(__file__).resolve().parent))
from document_extractor import extract_and_clean, EXTRACTORS


PROJECT_ROOT = Path(r"D:\Sachin Chakrawarti\Learn\done\thesis\ctc_ectd")
UPLOAD_DIR = PROJECT_ROOT / "uploads"
DATASET = PROJECT_ROOT / "dataset" / "synthetic_cti_reports.csv"
ARTIFACTS = PROJECT_ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)

RANDOM_STATE = 42
EMERGING_MIN_SHARE_JUMP = 0.10
EMERGING_MIN_COUNT = 2


# ------------------------------------------------------------------
# 1. Load extracted documents
# ------------------------------------------------------------------
def load_uploads() -> pd.DataFrame:
    rows = []
    for path in sorted(UPLOAD_DIR.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in EXTRACTORS:
            continue
        try:
            txt = extract_and_clean(path)
        except Exception as exc:
            print(f"  [WARN] {path.name}: {exc}")
            continue
        rows.append({
            "file_name": path.name,
            "extension": path.suffix.lower(),
            "char_count": len(txt),
            "word_count": len(txt.split()),
            "text": txt,
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# 2. Train classifier on the labelled dataset
# ------------------------------------------------------------------
def train_classifier():
    if not DATASET.exists():
        raise FileNotFoundError(f"Training data not found: {DATASET}")

    df = pd.read_csv(DATASET)
    text_col = "text" if "text" in df.columns else df.columns[1]
    label_col = ("category" if "category" in df.columns
                 else "label" if "label" in df.columns
                 else "attack_type" if "attack_type" in df.columns
                 else None)
    if label_col is None:
        raise ValueError("No label column found in dataset.")

    title_col = "title" if "title" in df.columns else None
    df = df.dropna(subset=[text_col, label_col]).copy()
    df[text_col] = df[text_col].astype(str).str.strip()

    if title_col:
        df["model_text"] = (df[title_col].fillna("") + " " + df[text_col]).str.strip()
    else:
        df["model_text"] = df[text_col]

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=10000, ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
            stop_words="english", sublinear_tf=True)),
        ("clf", LogisticRegression(
            max_iter=2000, class_weight="balanced",
            random_state=RANDOM_STATE)),
    ])
    model.fit(df["model_text"], df[label_col])
    return model, df


# ------------------------------------------------------------------
# 3. Classify uploaded docs
# ------------------------------------------------------------------
def classify_docs(model, docs: pd.DataFrame) -> pd.DataFrame:
    if docs.empty:
        return docs
    docs = docs.copy()
    docs["predicted_category"] = model.predict(docs["text"])

    clf = model.named_steps["clf"]
    if hasattr(clf, "predict_proba"):
        probs = model.predict_proba(docs["text"])
        docs["confidence"] = probs.max(axis=1).round(4)
    elif hasattr(clf, "decision_function"):
        scores = model.decision_function(docs["text"])
        docs["confidence"] = np.max(scores, axis=1).round(4)
    else:
        docs["confidence"] = np.nan

    return docs


# ------------------------------------------------------------------
# 4. Emergin-threat detection from monthly topic shares
# ------------------------------------------------------------------
def detect_emerging_trends(docs: pd.DataFrame) -> pd.DataFrame:
    # Parse date from filename like 2024-08_...
    DATE_RE = re.compile(r"(20\d{2})[-_/](\d{1,2})")

    def parse_date(fname):
        m = DATE_RE.search(fname)
        if m:
            return pd.Timestamp(year=int(m.group(1)), month=int(m.group(2)), day=1)
        return pd.NaT

    docs = docs.copy()
    docs["parsed_date"] = docs["file_name"].apply(parse_date)
    dated = docs.dropna(subset=["parsed_date"]).copy()
    if len(dated) < 3:
        print("\nNot enough dated docs for topic modeling (need >= 3).")
        return pd.DataFrame()

    dated["month"] = dated["parsed_date"].dt.to_period("M").astype(str)

    vec = TfidfVectorizer(max_features=5000, stop_words="english",
                          ngram_range=(1, 2))
    M = vec.fit_transform(dated["text"])
    k = min(5, M.shape[0], M.shape[1])
    if k < 2:
        print("\nNot enough terms to fit topics.")
        return pd.DataFrame()

    nmf = NMF(n_components=k, random_state=RANDOM_STATE,
              init="nndsvda", max_iter=500)
    W = nmf.fit_transform(M)
    terms = vec.get_feature_names_out()

    print("\nTopic keywords:")
    for i, comp in enumerate(nmf.components_):
        top = comp.argsort()[-8:][::-1]
        print(f"  Topic {i+1}:", ", ".join(terms[j] for j in top))

    for i in range(k):
        dated[f"Topic_{i+1}"] = W[:, i]
    dated["dominant_topic"] = W.argmax(axis=1) + 1

    counts = pd.crosstab(dated["month"], dated["dominant_topic"])
    shares = counts.div(counts.sum(axis=1), axis=0).fillna(0)

    print("\nMonthly dominant-topic shares:")
    print(shares.round(3).to_string())

    alerts = []
    months = list(shares.index)
    for i in range(1, len(months)):
        prev_m, curr_m = months[i-1], months[i]
        for tid in shares.columns:
            ps, cs = shares.loc[prev_m, tid], shares.loc[curr_m, tid]
            cnt = counts.loc[curr_m, tid]
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

    return pd.DataFrame(alerts)


# ------------------------------------------------------------------
# 5. Main
# ------------------------------------------------------------------
def main():
    print("=" * 62)
    print("CTI UPLOAD ANALYSIS — Extraction → Classification → Trends")
    print("=" * 62)

    print("\n[1/4] Extracting documents from uploads/ ...")
    docs = load_uploads()
    if docs.empty:
        print("  No supported files in uploads/. Add PDF/DOCX/TXT and retry.")
        return
    print(f"  Extracted {len(docs)} document(s).")
    print(docs[["file_name", "char_count", "word_count"]].to_string(index=False))

    print("\n[2/4] Training classifier on synthetic CTI corpus ...")
    model, train_df = train_classifier()
    print(f"  Trained on {len(train_df)} rows, "
          f"{train_df[train_df.columns[-2] if False else 'category' if 'category' in train_df.columns else 'label' if 'label' in train_df.columns else train_df.columns[1]].nunique() if False else 'N/A'} classes.")

    print("\n[3/4] Classifying uploaded documents ...")
    docs = classify_docs(model, docs)
    print(docs[["file_name", "predicted_category",
                "confidence"]].to_string(index=False))

    out_csv = ARTIFACTS / "uploaded_classifications.csv"
    docs[["file_name", "word_count", "predicted_category", "confidence"]] \
        .to_csv(out_csv, index=False)
    print(f"\n  Saved classifications → {out_csv}")

    print("\n[4/4] Emerging-threat detection (NMF topic drift) ...")
    alerts = detect_emerging_trends(docs)
    if not alerts.empty:
        print("\nEmerging-threat alerts:")
        print(alerts.to_string(index=False))
        alerts.to_csv(ARTIFACTS / "emerging_threat_alerts.csv", index=False)
        print(f"\n  Saved alerts → {ARTIFACTS / 'emerging_threat_alerts.csv'}")
    else:
        print("  No topic passed the emerging-threat threshold.")

    print("\nDone.")


if __name__ == "__main__":
    main()