"""FastAPI app: upload a PDF, classify relevance, predict techniques, and detect emerging topics."""
from collections import Counter
from pathlib import Path
import pandas as pd

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from classifier import classify, predict_techniques
from pdf_reader import extract_text_from_bytes
from preprocessing import split_sentences
from topic_model import topic_summary

PROJECT_ROOT = Path(__file__).resolve().parent.parent
UI_DIR = PROJECT_ROOT / "ui"
RESULTS_DIR = PROJECT_ROOT / "results"
MAX_BYTES = 15 * 1024 * 1024

app = FastAPI(title="Cyber Threat Detection API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)
app.mount("/ui", StaticFiles(directory=str(UI_DIR), html=True), name="ui")

@app.get("/")
def root():
    return FileResponse(UI_DIR / "index.html")

@app.get("/performance")
def performance():
    path = RESULTS_DIR / "proposed_model_results.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path)
    cols = [c for c in ["model", "classifier", "accuracy", "macro_f1"] if c in df.columns]
    return df[cols].to_dict(orient="records")

@app.get("/health")
def health():
    cls = PROJECT_ROOT / "models" / "classification"
    return {
        "relevance_model": (cls / "sbert_relevance.joblib").exists(),
        "technique_model": (cls / "sbert_technique.joblib").exists(),
        "topic_model":     (PROJECT_ROOT / "models" / "emerging_threat" / "bertopic_model").exists(),
    }

@app.post("/analyse")
async def analyse(file: UploadFile = File(...)):
    if file.content_type not in ("application/pdf", "application/octet-stream"):
        raise HTTPException(400, "Only PDF files are supported")
    
    data = await file.read()
    if not data: raise HTTPException(400, "Empty file")
    if len(data) > MAX_BYTES: raise HTTPException(413, f"Max file size is {MAX_BYTES // 1024 // 1024} MB")

    text, err = extract_text_from_bytes(data)
    if err: raise HTTPException(422, f"Could not read PDF: {err}")

    sents = split_sentences(text)
    if not sents: raise HTTPException(422, "No sentences found in PDF")

    classified = classify(sents)
    rel_sents = [c["sentence"] for c in classified if c["label"] == "relevant"]

    techniques = predict_techniques(rel_sents) if rel_sents else []
    tech_counts = {str(k): int(v) for k, v in Counter(techniques).items()}
    top_tech = max(tech_counts, key=tech_counts.get) if tech_counts else None

    return {
        "file": file.filename,
        "n_sentences": len(sents),
        "n_relevant": len(rel_sents),
        "n_irrelevant": len(sents) - len(rel_sents),
        "relevant_ratio": (len(rel_sents) / len(sents)) if sents else 0,
        "top_technique": top_tech,
        "technique_counts": tech_counts,
        "top_relevant_sentences": rel_sents[:5],
        "topics": topic_summary(sents),
    }
