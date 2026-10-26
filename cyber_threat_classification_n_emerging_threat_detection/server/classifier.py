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
