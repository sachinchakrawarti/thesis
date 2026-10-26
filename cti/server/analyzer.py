"""
Analyzer module — loads trained models and runs the full pipeline:
  - relevance classification (TF-IDF + SBERT hybrid)
  - topic assignment (BERTopic)
  - emerging-threat detection (rule-based)
  - attack-type classification (topic → attack mapping)

Used by:
  - server/app.py             (Flask backend)
  - scripts/predict_report.py (CLI)
  - scripts/batch_check.py    (batch CLI)
"""
import json, re, joblib
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from sklearn.preprocessing import normalize
from scipy.sparse import hstack


class CTIAnalyzer:
    """Lazy-loading analyzer. Call .load() once; then .predict_*() repeatedly."""

    def __init__(self, root=None):
        self.root = Path(root).resolve() if root else Path(__file__).resolve().parent.parent
        self.art_dir    = self.root / "artifacts"
        self.models_dir = self.art_dir / "models"
        self.topic_dir  = self.art_dir / "bertopic_model"

        self._loaded = False
        self.vec = None
        self.clf = None
        self.sbert = None
        self.topic_model = None
        self.vocab = None
        self.topic_base = None
        self.topic_to_attack = {}

    def load(self):
        """Load all models into memory. Safe to call multiple times."""
        if self._loaded:
            return

        print("[CTIAnalyzer] Loading models...")

        if not (self.models_dir / "tfidf_vectorizer.pkl").exists():
            raise FileNotFoundError(
                f"Missing models in {self.models_dir}. "
                f"Run scripts/make_inference.py first."
            )
        if not self.topic_dir.exists():
            raise FileNotFoundError(
                f"Missing BERTopic model at {self.topic_dir}. "
                f"Run Notebook 05 first."
            )

        self.vec = joblib.load(self.models_dir / "tfidf_vectorizer.pkl")
        self.clf = joblib.load(self.models_dir / "hybrid_classifier.pkl")
        self.sbert = SentenceTransformer("all-MiniLM-L6-v2")
        self.topic_model = BERTopic.load(
            str(self.topic_dir), embedding_model=self.sbert
        )
        self.vocab = set(json.load(open(self.models_dir / "vocab_reference.json")))
        self.topic_base = {
            int(k): float(v)
            for k, v in json.load(open(self.models_dir / "topic_baseline.json")).items()
        }

        # Optional: attack-type mapping (if present)
        attack_path = self.models_dir / "topic_to_attack.json"
        if attack_path.exists():
            self.topic_to_attack = json.load(open(attack_path))
            print(f"[CTIAnalyzer] Loaded topic->attack map ({len(self.topic_to_attack)} entries).")
        else:
            self.topic_to_attack = {}
            print("[CTIAnalyzer] No topic_to_attack.json — attack-type classification disabled.")

        self._loaded = True
        print(f"[CTIAnalyzer] Ready — {len(self.topic_model.get_topics()) - 1} topics loaded.")

    # ──────────────────────────────────────────────────
    @staticmethod
    def extract_text(path):
        p = Path(path)
        if p.suffix.lower() == ".pdf":
            try:
                import pdfplumber
            except ImportError:
                raise ImportError(
                    "pdfplumber is required for PDF extraction. "
                    "Install with: pip install pdfplumber"
                )
            with pdfplumber.open(p) as pdf:
                return "\n".join(page.extract_text() or "" for page in pdf.pages)
        return p.read_text(encoding="utf-8", errors="ignore")

    @staticmethod
    def split_sentences(text, min_words=4):
        text = re.sub(r"\s+", " ", text)
        parts = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in parts if len(s.split()) >= min_words]

    # ──────────────────────────────────────────────────
    def predict_text(self, text, report_name="pasted text"):
        """Run full pipeline on raw text."""
        self.load()

        sents = self.split_sentences(text)
        if not sents:
            return self._empty_result(report_name, "no sentences extracted")

        # ── 1. Relevance classification ──────────────
        emb = self.sbert.encode(sents, batch_size=64, convert_to_numpy=True)
        X = hstack([self.vec.transform(sents), normalize(emb)]).tocsr()
        preds = self.clf.predict(X)

        rel = [s for s, p in zip(sents, preds) if p == 1]
        rel_emb = emb[preds == 1]

        classification = {
            "total_sentences":    len(sents),
            "relevant_sentences": len(rel),
            "irrelevant_sentences": len(sents) - len(rel),
            "relevant_fraction":  round(len(rel) / len(sents), 3),
        }

        # ── 2. Topic assignment ──────────────────────
        topics = self.topic_model.transform(rel, rel_emb)[0] if len(rel) > 0 else []
        tc = {}
        for t in topics:
            t = int(t)
            tc[t] = tc.get(t, 0) + 1

        # ── 3. Emerging-threat rules ─────────────────
        reasons, emerging = [], False
        rule_hits = {"outlier": False, "surge": False, "novelty": False}

        # Rule 1: outlier fraction
        if len(topics) > 0 and -1 in tc and tc[-1] / len(topics) > 0.30:
            emerging = True
            rule_hits["outlier"] = True
            reasons.append(f"{tc[-1]} outlier sentences")

        # Rule 2: surge (only for rare topics — well-known topics should not trigger)
        if len(topics) > 0:
            for tid, c in tc.items():
                if tid == -1 or c < 3:
                    continue
                base = self.topic_base.get(tid, 1e-9)
                if base > 0.05:      # well-known topic — skip
                    continue
                surge = (c / len(topics)) / max(base, 1e-9)
                if surge > 3.0:
                    emerging = True
                    rule_hits["surge"] = True
                    reasons.append(f"rare topic {tid} surge {surge:.1f}x")

        # Rule 3: novelty
        toks = [t for s in rel for t in re.findall(r"\b[a-z]{3,}\b", s.lower())]
        novelty_ratio = 0.0
        if toks:
            novel = sum(1 for t in toks if t not in self.vocab)
            novelty_ratio = novel / len(toks)
            if novelty_ratio > 0.40:
                emerging = True
                rule_hits["novelty"] = True
                reasons.append(f"{novelty_ratio * 100:.0f}% novel tokens")

        # ── 4. Dominant topic ────────────────────────
        dom = None
        if tc:
            dom_tid = max(tc, key=tc.get)
            if dom_tid != -1:
                dom = {
                    "topic_id": dom_tid,
                    "top_words": [w for w, _ in self.topic_model.get_topic(dom_tid)][:8],
                    "count": tc[dom_tid],
                    "share": round(tc[dom_tid] / len(topics), 3),
                }
                reasons.append(
                    f"dominant topic {dom_tid}: " + ", ".join(dom["top_words"])
                )

        # ── 5. Verdict ───────────────────────────────
        verdict = "not_relevant" if not rel else ("emerging" if emerging else "known")

        # ── 6. Attack-type classification ────────────
        attack_classification = self._classify_attack_type(tc, len(topics))

        # ── 7. Combined result ───────────────────────
        return {
            "report": report_name,
            "verdict": verdict,
            "classification": classification,
            "attack_classification": attack_classification,
            "emerging_detection": {
                "emerging_threat": emerging,
                "novelty_ratio": round(novelty_ratio, 3),
                "rule_hits": rule_hits,
                "dominant_topic": dom,
                "topic_counts": {str(k): v for k, v in tc.items()},
                "reasons": reasons,
            },
            # flat keys for backward compat
            "total_sentences":    classification["total_sentences"],
            "relevant_sentences": classification["relevant_sentences"],
            "relevant_fraction":  classification["relevant_fraction"],
            "topic_counts":       {str(k): v for k, v in tc.items()},
            "dominant_topic":     dom,
            "emerging_threat":    emerging,
            "novelty_ratio":      round(novelty_ratio, 3),
            "rule_hits":          rule_hits,
            "reasons":            reasons,
        }

    def predict_file(self, path):
        """Run pipeline on a PDF or TXT file."""
        text = self.extract_text(path)
        return self.predict_text(text, report_name=str(path))

    # ──────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────
    def _empty_result(self, report_name, error):
        return {
            "report": report_name,
            "verdict": "not_relevant",
            "classification": {
                "total_sentences": 0, "relevant_sentences": 0,
                "irrelevant_sentences": 0, "relevant_fraction": 0.0,
            },
            "attack_classification": {
                "primary_attack_type": "Unknown",
                "confidence": 0.0,
                "all_attack_types": [],
            },
            "emerging_detection": {
                "emerging_threat": False, "novelty_ratio": 0.0,
                "rule_hits": {"outlier": False, "surge": False, "novelty": False},
                "dominant_topic": None, "topic_counts": {}, "reasons": [],
            },
            "error": error,
        }

    def _classify_attack_type(self, topic_counts, total_topics):
        """Map topics to attack categories using topic_to_attack.json."""
        if not self.topic_to_attack or total_topics == 0:
            return {
                "primary_attack_type": "Unknown",
                "confidence": 0.0,
                "all_attack_types": [],
            }

        attack_scores = {}
        for tid, count in topic_counts.items():
            if tid == -1:
                continue
            cat = self.topic_to_attack.get(str(tid), "Unknown")
            if cat == "Unknown":
                continue
            attack_scores[cat] = attack_scores.get(cat, 0) + count

        if not attack_scores:
            return {
                "primary_attack_type": "Unknown",
                "confidence": 0.0,
                "all_attack_types": [],
            }

        ranked = sorted(attack_scores.items(), key=lambda x: -x[1])
        total = sum(attack_scores.values())
        primary_type = ranked[0][0]
        confidence = ranked[0][1] / total if total > 0 else 0.0

        return {
            "primary_attack_type": primary_type,
            "confidence": round(confidence, 3),
            "all_attack_types": [
                {
                    "type": cat,
                    "count": count,
                    "share": round(count / total_topics, 3),
                }
                for cat, count in ranked
            ],
        }


_default = None

def get_analyzer():
    global _default
    if _default is None:
        _default = CTIAnalyzer()
        _default.load()
    return _default