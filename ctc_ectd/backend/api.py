"""
api.py - FastAPI backend for the CTI Threat Analysis UI.
"""

from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(r"D:\Projects\done\thesis\ctc_ectd")
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
UI_DIR = PROJECT_ROOT / "ui"
UPLOAD_DIR = PROJECT_ROOT / "uploads"
MODEL_PATH = ARTIFACT_DIR / "cti_text_classifier.joblib"

UPLOAD_DIR.mkdir(exist_ok=True)


app = FastAPI(title="CTI Threat Analysis API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


_model: Optional[Pipeline] = None


def get_model() -> Pipeline:
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail=f"Model not found at {MODEL_PATH}. Run notebook 03 first.",
            )
        _model = joblib.load(MODEL_PATH)
    return _model


class PredictRequest(BaseModel):
    text: str


class PredictionResponse(BaseModel):
    predicted_category: str
    confidence: float
    top_k: List[dict]


@app.get("/")
def root():
    index = UI_DIR / "index.html"
    if not index.exists():
        raise HTTPException(status_code=404, detail="UI not found.")
    return FileResponse(index)


@app.get("/ui/{filename}")
def ui_asset(filename: str):
    path = UI_DIR / filename
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail=f"{filename} not found.")
    return FileResponse(path)


@app.get("/ui/components/{filename}")
def ui_component(filename: str):
    path = UI_DIR / "components" / filename
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail=f"{filename} not found.")
    return FileResponse(path)


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model_present": MODEL_PATH.exists(),
        "artifacts": sorted(p.name for p in ARTIFACT_DIR.glob("*") if p.is_file()),
    }


@app.get("/api/summary")
def summary():
    cleaned = PROCESSED_DIR / "cleaned_reports.pkl"
    uploaded = PROCESSED_DIR / "uploaded_docs.pkl"

    df = pd.read_pickle(cleaned)["df"] if cleaned.exists() else pd.DataFrame()
    docs = pd.read_pickle(uploaded) if uploaded.exists() else pd.DataFrame()

    label_col = None
    for c in ("category", "label", "attack_type"):
        if c in df.columns:
            label_col = c
            break

    return {
        "training_rows": int(len(df)),
        "uploaded_docs": int(len(docs)),
        "categories": int(df[label_col].nunique()) if label_col else 0,
        "label_column": label_col,
    }


@app.get("/api/label-distribution")
def label_distribution():
    cleaned = PROCESSED_DIR / "cleaned_reports.pkl"
    if not cleaned.exists():
        return {"labels": [], "counts": []}
    df = pd.read_pickle(cleaned)["df"]
    label_col = None
    for c in ("category", "label", "attack_type"):
        if c in df.columns:
            label_col = c
            break
    if not label_col:
        return {"labels": [], "counts": []}
    vc = df[label_col].value_counts()
    return {"labels": vc.index.tolist(), "counts": vc.values.tolist()}


@app.get("/api/model-results")
def model_results():
    path = ARTIFACT_DIR / "model_results.csv"
    if not path.exists():
        return []
    return pd.read_csv(path).fillna("").to_dict(orient="records")


@app.get("/api/classified-docs")
def classified_docs():
    path = ARTIFACT_DIR / "classified_documents.csv"
    if not path.exists():
        return []
    return pd.read_csv(path).fillna("").to_dict(orient="records")


@app.get("/api/emerging-alerts")
def emerging_alerts():
    path = ARTIFACT_DIR / "emerging_threat_alerts.csv"
    if not path.exists():
        return []
    return pd.read_csv(path).fillna("").to_dict(orient="records")


@app.get("/api/monthly-shares")
def monthly_shares():
    path = ARTIFACT_DIR / "monthly_topic_shares.csv"
    if not path.exists():
        return {"months": [], "series": {}}
    df = pd.read_csv(path)
    month_col = df.columns[0]
    topic_cols = [c for c in df.columns if c != month_col]
    return {
        "months": df[month_col].tolist(),
        "series": {t: df[t].fillna(0).tolist() for t in topic_cols},
    }


def _predict(text: str) -> PredictionResponse:
    model = get_model()
    if not text.strip():
        raise HTTPException(status_code=400, detail="Empty text.")

    pred = model.predict([text])[0]
    clf = model.named_steps["clf"]

    if hasattr(clf, "predict_proba"):
        proba = model.predict_proba([text])[0]
        classes = clf.classes_
    elif hasattr(clf, "decision_function"):
        raw = model.decision_function([text])[0]
        e = np.exp(raw - raw.max())
        proba = e / e.sum()
        classes = clf.classes_
    else:
        proba = np.array([1.0])
        classes = np.array([pred])

    order = np.argsort(proba)[::-1]
    top_k = [
        {"category": str(classes[i]), "probability": round(float(proba[i]), 4)}
        for i in order[:5]
    ]
    return PredictionResponse(
        predicted_category=str(pred),
        confidence=top_k[0]["probability"],
        top_k=top_k,
    )


@app.post("/api/predict", response_model=PredictionResponse)
def predict(req: PredictRequest):
    return _predict(req.text)


@app.post("/api/classify-upload", response_model=PredictionResponse)
async def classify_upload(file: UploadFile = File(...)):
    import sys
    scripts_dir = PROJECT_ROOT / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    tmp = UPLOAD_DIR / file.filename
    with tmp.open("wb") as f:
        f.write(await file.read())

    try:
        from document_extractor import extract_and_clean  # type: ignore
        text = extract_and_clean(tmp)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {exc}")

    if not text or len(text.strip()) < 20:
        raise HTTPException(status_code=400, detail="Extracted text too short.")

    return _predict(text)
