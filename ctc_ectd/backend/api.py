"""
FastAPI backend for the CTI Threat Analysis UI.
Wraps scripts/document_extractor.py + a TF-IDF + LogisticRegression classifier.

Run:
    uvicorn backend.api:app --reload --port 8000
"""

from __future__ import annotations

import re
import sys
import tempfile
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR  = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from document_extractor import extract_and_clean, EXTRACTORS  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer   # noqa: E402
from sklearn.linear_model import LogisticRegression           # noqa: E402
from sklearn.decomposition import NMF                         # noqa: E402
from sklearn.pipeline import Pipeline                         # noqa: E402


RANDOM_STATE = 42
EMERGING_MIN_SHARE_JUMP = 0.10
EMERGING_MIN_COUNT = 2

DATASET   = PROJECT_ROOT / "dataset" / "synthetic_cti_reports.csv"
ARTIFACTS = PROJECT_ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)

_model: Optional[Pipeline] = None

app = FastAPI(title="CTI Threat Analysis API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def train_classifier() -> Pipeline:
    global _model
    if _model is not None:
        return _model

    if not DATASET.exists():
        raise FileNotFoundError(f"Training data not found: {DATASET}")

    df = pd.read_csv(DATASET)
    text_col = "text" if "text" in df.columns else df.columns[1]
    label_col = next((c for c in ("category", "label", "attack_type")
                      if c in df.columns), None)
    if label_col is None:
        raise ValueError("No label column found in dataset.")

    title_col = "title" if "title" in df.columns else None
    df = df.dropna(subset=[text_col, label_col]).copy()
    df[text_col] = df[text_col].astype(str).str.strip()

    if title_col:
        df["model_text"] = (df[title_col].fillna("") + " " + df[text_col]).str.strip()
    else:
        df["model_text"] = df[text_col]

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=10000, ngram_range=(1, 2),
            stop_words="english", sublinear_tf=True)),
        ("clf", LogisticRegression(
            max_iter=2000, class_weight="balanced",
            random_state=RANDOM_STATE)),
    ])
    pipe.fit(df["model_text"], df[label_col])
    _model = pipe
    return pipe


DATE_TOKEN_RE = re.compile(r"\b(19|20)\d{2}[-_/]?\d{0,2}\b")
DATE_RE       = re.compile(r"(20\d{2})[-_/](\d{1,2})")


def strip_meta_tokens(text: str) -> str:
    text = DATE_TOKEN_RE.sub(" ", text)
    text = re.sub(r"\b\d{1,4}\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_date_from_name(name: str):
    m = DATE_RE.search(name)
    if not m:
        return None
    try:
        return pd.Timestamp(year=int(m.group(1)),
                            month=int(m.group(2)), day=1)
    except Exception:
        return None


DEMO_DIR = PROJECT_ROOT / "uploads"


def _run_pipeline(docs):
    """Shared classification + NMF topic-modeling pipeline."""
    if not docs:
        return {"documents": [], "alerts": [], "topics": None}

    model = train_classifier()
    df = pd.DataFrame(docs)
    df["predicted_category"] = model.predict(df["text"])

    clf = model.named_steps["clf"]
    if hasattr(clf, "predict_proba"):
        probs = model.predict_proba(df["text"])
        df["confidence"] = probs.max(axis=1)
    elif hasattr(clf, "decision_function"):
        scores = model.decision_function(df["text"])
        df["confidence"] = np.max(scores, axis=1)
    else:
        df["confidence"] = np.nan

    df["parsed_date"] = df["file_name"].apply(parse_date_from_name)
    df["parsed_date"] = df["parsed_date"].apply(
        lambda d: d.strftime("%Y-%m") if pd.notna(d) else None)

    dated = df.dropna(subset=["parsed_date"]).copy()
    topics_payload = None
    alerts = []

    if len(dated) >= 3:
        dated["text_clean"] = dated["text"].apply(strip_meta_tokens)
        vec = TfidfVectorizer(
            max_features=5000,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-]+\b",
        )
        M = vec.fit_transform(dated["text_clean"])
        k = min(5, M.shape[0], M.shape[1])

        if k >= 2:
            nmf = NMF(n_components=k, random_state=RANDOM_STATE,
                      init="nndsvda", max_iter=500)
            W = nmf.fit_transform(M)
            for i in range(k):
                dated[f"Topic_{i+1}"] = W[:, i]
            dated["dominant_topic"] = W.argmax(axis=1) + 1

            counts = pd.crosstab(dated["parsed_date"], dated["dominant_topic"])
            shares = counts.div(counts.sum(axis=1), axis=0).fillna(0)
            months = list(shares.index)

            topics_payload = {
                "months": months,
                "series": [
                    {"name": f"Topic {int(t)}",
                     "data": [round(float(shares.loc[m, t]), 3) for m in months]}
                    for t in shares.columns
                ],
            }

            for i in range(1, len(months)):
                pm, cm = months[i-1], months[i]
                for tid in shares.columns:
                    ps  = float(shares.loc[pm, tid])
                    cs  = float(shares.loc[cm, tid])
                    cnt = int(counts.loc[cm, tid])
                    if (cs - ps) >= EMERGING_MIN_SHARE_JUMP and cnt >= EMERGING_MIN_COUNT:
                        alerts.append({
                            "previous_month": pm,
                            "current_month": cm,
                            "topic": int(tid),
                            "previous_share": round(ps, 3),
                            "current_share": round(cs, 3),
                            "increase": round(cs - ps, 3),
                            "current_count": cnt,
                        })

    out = df[["file_name", "word_count", "predicted_category",
              "confidence", "parsed_date"]].copy()
    out["confidence"] = out["confidence"].round(4)
    out.to_csv(ARTIFACTS / "uploaded_classifications.csv", index=False)

    if alerts:
        pd.DataFrame(alerts).to_csv(
            ARTIFACTS / "emerging_threat_alerts.csv", index=False)

    return {
        "documents": out.replace({np.nan: None}).to_dict(orient="records"),
        "alerts": alerts,
        "topics": topics_payload,
    }


@app.post("/api/analyze-demo")
async def analyze_demo():
    """Run the pipeline on every file in uploads/ — for UI demos."""
    if not DEMO_DIR.exists():
        return JSONResponse(
            {"error": f"Demo folder not found: {DEMO_DIR}"},
            status_code=404,
        )
    files = [p for p in sorted(DEMO_DIR.iterdir())
             if p.is_file() and p.suffix.lower() in EXTRACTORS]
    if not files:
        return JSONResponse(
            {"error": f"No demo files in {DEMO_DIR}"},
            status_code=404,
        )

    docs = []
    for p in files:
        try:
            txt = extract_and_clean(p)
        except Exception:
            continue
        docs.append({
            "file_name": p.name,
            "extension": p.suffix.lower(),
            "word_count": len(txt.split()),
            "char_count": len(txt),
            "text": txt,
        })

    return _run_pipeline(docs)


@app.get("/api/health")
def health():
    try:
        train_classifier()
        return {"status": "ok", "model": "logistic_regression"}
    except Exception as exc:
        return JSONResponse({"status": "err", "error": str(exc)}, status_code=500)


@app.post("/api/analyze")
async def analyze(files: List[UploadFile] = File(...)):
    docs = []
    for uf in files:
        ext = Path(uf.filename).suffix.lower()
        if ext not in EXTRACTORS:
            continue
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(await uf.read())
            tmp_path = Path(tmp.name)

        try:
            txt = extract_and_clean(tmp_path)
        except Exception as exc:
            docs.append({
                "file_name": uf.filename,
                "extension": ext,
                "word_count": 0,
                "char_count": 0,
                "text": "",
                "error": str(exc),
            })
            continue

        docs.append({
            "file_name": uf.filename,
            "extension": ext,
            "word_count": len(txt.split()),
            "char_count": len(txt),
            "text": txt,
        })

    if not docs:
        return {"documents": [], "alerts": [], "topics": None}

    model = train_classifier()
    df = pd.DataFrame(docs)
    df["predicted_category"] = model.predict(df["text"])

    clf = model.named_steps["clf"]
    if hasattr(clf, "predict_proba"):
        probs = model.predict_proba(df["text"])
        df["confidence"] = probs.max(axis=1)
    elif hasattr(clf, "decision_function"):
        scores = model.decision_function(df["text"])
        df["confidence"] = np.max(scores, axis=1)
    else:
        df["confidence"] = np.nan

    df["parsed_date"] = df["file_name"].apply(parse_date_from_name)
    df["parsed_date"] = df["parsed_date"].apply(
        lambda d: d.strftime("%Y-%m") if pd.notna(d) else None)

    dated = df.dropna(subset=["parsed_date"]).copy()
    topics_payload = None
    alerts = []

    if len(dated) >= 3:
        dated["text_clean"] = dated["text"].apply(strip_meta_tokens)
        vec = TfidfVectorizer(
            max_features=5000,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-]+\b",
        )
        M = vec.fit_transform(dated["text_clean"])
        k = min(5, M.shape[0], M.shape[1])

        if k >= 2:
            nmf = NMF(n_components=k, random_state=RANDOM_STATE,
                      init="nndsvda", max_iter=500)
            W = nmf.fit_transform(M)
            for i in range(k):
                dated[f"Topic_{i+1}"] = W[:, i]
            dated["dominant_topic"] = W.argmax(axis=1) + 1

            counts = pd.crosstab(dated["parsed_date"], dated["dominant_topic"])
            shares = counts.div(counts.sum(axis=1), axis=0).fillna(0)
            months = list(shares.index)

            topics_payload = {
                "months": months,
                "series": [
                    {
                        "name": f"Topic {int(t)}",
                        "data": [round(float(shares.loc[m, t]), 3) for m in months],
                    }
                    for t in shares.columns
                ],
            }

            for i in range(1, len(months)):
                pm, cm = months[i-1], months[i]
                for tid in shares.columns:
                    ps  = float(shares.loc[pm, tid])
                    cs  = float(shares.loc[cm, tid])
                    cnt = int(counts.loc[cm, tid])
                    if (cs - ps) >= EMERGING_MIN_SHARE_JUMP and cnt >= EMERGING_MIN_COUNT:
                        alerts.append({
                            "previous_month": pm,
                            "current_month": cm,
                            "topic": int(tid),
                            "previous_share": round(ps, 3),
                            "current_share": round(cs, 3),
                            "increase": round(cs - ps, 3),
                            "current_count": cnt,
                        })

    out = df[["file_name", "word_count", "predicted_category",
              "confidence", "parsed_date"]].copy()
    out["confidence"] = out["confidence"].round(4)
    out.to_csv(ARTIFACTS / "uploaded_classifications.csv", index=False)

    if alerts:
        pd.DataFrame(alerts).to_csv(
            ARTIFACTS / "emerging_threat_alerts.csv", index=False)

    return {
        "documents": out.replace({np.nan: None}).to_dict(orient="records"),
        "alerts": alerts,
        "topics": topics_payload,
    }
