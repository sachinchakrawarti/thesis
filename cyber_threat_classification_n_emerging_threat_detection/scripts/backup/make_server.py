# D:\Projects\done\thesis\cyber-threat-detection\scripts\make_server.py
"""Generate the complete FastAPI server backend."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SERVER = PROJECT_ROOT / "server"
SERVER.mkdir(parents=True, exist_ok=True)

def write(path: Path, text: str):
    path.write_text(text, encoding="utf-8")
    print("wrote", path.relative_to(PROJECT_ROOT))

# ---------------- requirements.txt ----------------
write(SERVER / "requirements.txt", """\
fastapi==0.115.0
uvicorn[standard]==0.30.6
python-multipart==0.0.9
pymupdf==1.24.10
sentence-transformers==3.1.1
scikit-learn==1.5.2
bertopic==0.16.4
umap-learn==0.5.6
hdbscan==0.8.40
joblib==1.4.2
numpy==1.26.4
pandas==2.2.3
""")

# ---------------- preprocessing.py ----------------
write(SERVER / "preprocessing.py", '''\
"""Text cleaning and sentence splitting."""
import re

def normalize_whitespace(text: str) -> str:
    return re.sub(r"\\s+", " ", text or "").strip()

def split_sentences(text: str):
    text = normalize_whitespace(text)
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\\s+", text)
    return [p.strip() for p in parts if len(p.split()) >= 4]
''')

# ---------------- pdf_reader.py ----------------
write(SERVER / "pdf_reader.py", '''\
"""Extract text from PDF bytes using PyMuPDF."""
import fitz

def extract_text_from_bytes(data: bytes):
    if not data or not data.startswith(b"%PDF"):
        return "", "not a valid PDF"
    doc = None
    try:
        doc = fitz.open(stream=data, filetype="pdf")
        if doc.page_count == 0:
            return "", "0 pages"
        return "\\n".join(page.get_text() for page in doc), None
    except Exception as e:
        return "", f"{type(e).__name__}: {e}"
    finally:
        if doc:
            try:
                doc.close()
            except Exception:
                pass
''')

# ---------------- classifier.py ----------------
write(SERVER / "classifier.py", '''\
"""Load relevance and technique classifiers and predict."""
from pathlib import Path
import joblib
from sentence_transformers import SentenceTransformer

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "classification"
EMBEDDING_NAME = "all-MiniLM-L6-v2"

_sbert = None
_relevance = None
_technique = None

def _get_sbert():
    global _sbert
    if _sbert is None:
        _sbert = SentenceTransformer(EMBEDDING_NAME)
    return _sbert

def _load_relevance():
    global _relevance
    if _relevance is None:
        path = MODEL_DIR / "sbert_relevance.joblib"
        if not path.exists():
            raise FileNotFoundError("sbert_relevance.joblib not found — run notebook first.")
        _relevance = joblib.load(path)
    return _relevance

def _load_technique():
    global _technique
    if _technique is None:
        path = MODEL_DIR / "sbert_technique.joblib"
        if not path.exists():
            return None
        _technique = joblib.load(path)
    return _technique

def classify(sentences):
    if not sentences:
        return []
    emb = _get_sbert().encode(sentences, batch_size=64, normalize_embeddings=True)
    clf = _load_relevance()
    pred = clf.predict(emb)
    scores = clf.decision_function(emb)
    return [
        {"sentence": s, "label": "relevant" if int(p) == 1 else "irrelevant", "score": float(sc)}
        for s, p, sc in zip(sentences, pred, scores)
    ]

def predict_techniques(sentences):
    if not sentences:
        return []
    clf = _load_technique()
    if clf is None:
        return []
    emb = _get_sbert().encode(sentences, batch_size=64, normalize_embeddings=True)
    return clf.predict(emb).tolist()
''')

# ---------------- topic_model.py ----------------
write(SERVER / "topic_model.py", '''\
"""Load BERTopic model and assign topics."""
from pathlib import Path
import pandas as pd

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "emerging_threat"
_topic_model = None

def _load():
    global _topic_model
    if _topic_model is None:
        from bertopic import BERTopic
        path = MODEL_DIR / "bertopic_model"
        if not path.exists():
            raise FileNotFoundError("bertopic_model/ not found — run notebook first.")
        _topic_model = BERTopic.load(str(path))
        
        # Ensure HDBSCAN can predict on new documents
        if hasattr(_topic_model.hdbscan_model, "generate_prediction_data"):
            try:
                _topic_model.hdbscan_model.generate_prediction_data()
            except Exception:
                pass
    return _topic_model

def topic_summary(sentences, min_sentences=5):
    if len(sentences) < min_sentences:
        return {"note": f"too few sentences ({len(sentences)}) for topic modeling"}
    try:
        model = _load()
        topics, _ = model.transform(sentences)
        counts = pd.Series(topics).value_counts().to_dict()
        return {
            "topic_counts": {str(k): int(v) for k, v in counts.items()},
            "top_topics": model.get_topic_info().head(5).to_dict(orient="records"),
        }
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
''')

# ---------------- app.py ----------------
write(SERVER / "app.py", '''\
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
''')

print("\nServer generated. To run:")
print("  cd server")
print("  python -m pip install -r requirements.txt")
print("  uvicorn app:app --reload --port 8000")