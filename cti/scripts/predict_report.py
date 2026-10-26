"""
Standalone CLI and importable module for CTI report prediction.
Usage:
    python scripts/predict_report.py path/to/report.pdf
"""
import sys, json, re, joblib
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from sklearn.preprocessing import normalize
from scipy.sparse import hstack

ART = Path(__file__).parent.parent / "artifacts" / "models"

_vec = None
_clf = None
_sbert = None
_topic_model = None
_vocab = None
_topic_base = None

def _load():
    global _vec, _clf, _sbert, _topic_model, _vocab, _topic_base
    if _vec is not None:
        return
    _vec = joblib.load(ART / "tfidf_vectorizer.pkl")
    _clf = joblib.load(ART / "hybrid_classifier.pkl")
    _sbert = SentenceTransformer("all-MiniLM-L6-v2")
    _topic_model = BERTopic.load(
        str(Path(__file__).parent.parent / "artifacts" / "bertopic_model"),
        embedding_model=_sbert,
    )
    _vocab = set(json.load(open(ART / "vocab_reference.json")))
    _topic_base = {int(k): float(v) for k, v in
                   json.load(open(ART / "topic_baseline.json")).items()}

def extract_text(path):
    p = Path(path)
    if p.suffix.lower() == ".pdf":
        import pdfplumber
        with pdfplumber.open(p) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)
    return p.read_text(encoding="utf-8", errors="ignore")

def _split_sentences(text, min_words=4):
    text = re.sub(r"\s+", " ", text)
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in parts if len(s.split()) >= min_words]

def predict(path_or_text, is_text=False):
    _load()

    if is_text:
        text = path_or_text
        report_name = "pasted text"
    else:
        text = extract_text(path_or_text)
        report_name = str(path_or_text)

    sents = _split_sentences(text)
    if not sents:
        return {"report": report_name, "verdict": "not_relevant",
                "error": "no sentences extracted"}

    emb = _sbert.encode(sents, batch_size=64, convert_to_numpy=True)
    X = hstack([_vec.transform(sents), normalize(emb)]).tocsr()
    preds = _clf.predict(X)
    rel = [s for s, p in zip(sents, preds) if p == 1]
    rel_emb = emb[preds == 1]

    topics = _topic_model.transform(rel, rel_emb)[0] if len(rel) > 0 else []
    tc = {}
    for t in topics:
        t = int(t); tc[t] = tc.get(t, 0) + 1

    reasons, emerging = [], False
    rule_hits = {"outlier": False, "surge": False, "novelty": False}

    if len(topics) > 0 and -1 in tc and tc[-1] / len(topics) > 0.30:
        emerging = True; rule_hits["outlier"] = True
        reasons.append(f"{tc[-1]} outlier sentences")

    if len(topics) > 0:
        for tid, c in tc.items():
            if tid == -1: continue
            surge = (c / len(topics)) / max(_topic_base.get(tid, 1e-9), 1e-9)
            if surge > 3.0 and c >= 3:
                emerging = True; rule_hits["surge"] = True
                reasons.append(f"topic {tid} surge {surge:.1f}x")

    toks = [t for s in rel for t in re.findall(r"\b[a-z]{3,}\b", s.lower())]
    novelty_ratio = 0.0
    if toks:
        novel = sum(1 for t in toks if t not in _vocab)
        novelty_ratio = novel / len(toks)
        if novelty_ratio > 0.40:
            emerging = True; rule_hits["novelty"] = True
            reasons.append(f"{novelty_ratio*100:.0f}% novel tokens")

    dom = None
    if tc:
        dom_tid = max(tc, key=tc.get)
        if dom_tid != -1:
            dom = {"topic_id": dom_tid,
                   "top_words": [w for w, _ in _topic_model.get_topic(dom_tid)][:8],
                   "count": tc[dom_tid],
                   "share": round(tc[dom_tid] / len(topics), 3)}
            reasons.append(f"dominant topic {dom_tid}: " + ", ".join(dom["top_words"]))

    verdict = "not_relevant" if not rel else ("emerging" if emerging else "known")

    return {
        "report": report_name,
        "verdict": verdict,
        "classification": {
            "total_sentences": len(sents),
            "relevant_sentences": len(rel),
            "irrelevant_sentences": len(sents) - len(rel),
            "relevant_fraction": round(len(rel) / len(sents), 3),
        },
        "emerging_detection": {
            "emerging_threat": emerging,
            "novelty_ratio": round(novelty_ratio, 3),
            "rule_hits": rule_hits,
            "dominant_topic": dom,
            "topic_counts": {str(k): v for k, v in tc.items()},
            "reasons": reasons,
        },
        # flat keys for backward compatibility
        "total_sentences": len(sents),
        "relevant_sentences": len(rel),
        "relevant_fraction": round(len(rel) / len(sents), 3),
        "topic_counts": {str(k): v for k, v in tc.items()},
        "dominant_topic": dom,
        "emerging_threat": emerging,
        "novelty_ratio": round(novelty_ratio, 3),
        "rule_hits": rule_hits,
        "reasons": reasons,
    }

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python scripts/predict_report.py <report.pdf|.txt>")
        sys.exit(1)
    print(json.dumps(predict(sys.argv[1]), indent=2))