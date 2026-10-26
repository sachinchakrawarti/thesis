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
