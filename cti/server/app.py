"""
Flask backend for CTI Threat Report Analyzer.
Serves the UI at / and the analysis APIs at /analyze and /batch-analyze.

Run from project root:
    python server/app.py
"""
import os, sys, json, tempfile
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# Make `server` importable as a package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server.analyzer import get_analyzer

ROOT   = Path(__file__).resolve().parent.parent
UI_DIR = ROOT / "ui"

app = Flask(__name__, static_folder=str(UI_DIR))
CORS(app)


# ─────────────────────────────────────────────────────
# Health
# ─────────────────────────────────────────────────────
@app.route("/health")
def health():
    try:
        get_analyzer()
        return jsonify({"status": "ok", "models_loaded": True})
    except Exception as e:
        return jsonify({"status": "error", "detail": str(e)}), 500


# ─────────────────────────────────────────────────────
# Serve UI
# ─────────────────────────────────────────────────────
@app.route("/")
def index():
    index_path = UI_DIR / "index.html"
    if not index_path.exists():
        return (f"UI not found at {index_path}.", 404), 404
    return send_from_directory(str(UI_DIR), "index.html")


@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(str(UI_DIR), filename)


# ─────────────────────────────────────────────────────
# Single-report analysis
# ─────────────────────────────────────────────────────
@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        analyzer = get_analyzer()

        # ── Case 1: uploaded file ─────────────────────
        if "file" in request.files:
            f = request.files["file"]
            if not f.filename:
                return jsonify({"error": "empty filename"}), 400
            suffix = Path(f.filename).suffix.lower()
            if suffix not in (".pdf", ".txt"):
                return jsonify({"error": "only .pdf and .txt accepted"}), 400

            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                f.save(tmp.name)
                tmp_path = tmp.name
            try:
                result = analyzer.predict_file(tmp_path)
                result["report"] = f.filename  # restore original name
            finally:
                os.unlink(tmp_path)

        # ── Case 2: pasted text ───────────────────────
        elif request.is_json and "text" in request.json:
            text = request.json["text"]
            if not text or len(text.strip()) < 20:
                return jsonify({"error": "text too short (min 20 chars)"}), 400
            result = analyzer.predict_text(text, report_name="pasted text")

        else:
            return jsonify({"error": "no file or text provided"}), 400

        return jsonify(result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────────────
# Batch analysis of test_pdfs/
# ─────────────────────────────────────────────────────
@app.route("/batch-analyze", methods=["POST"])
def batch_analyze():
    try:
        analyzer = get_analyzer()
        pdf_dir = ROOT / "test_pdfs"

        if not pdf_dir.exists():
            return jsonify({"error": "test_pdfs/ folder not found"}), 404

        pdfs = sorted(pdf_dir.glob("*.pdf"))
        if not pdfs:
            return jsonify({"error": "no PDFs found in test_pdfs/"}), 404

        # Load expected verdicts if the manifest exists
        expected = {}
        manifest = pdf_dir / "manifest.json"
        if manifest.exists():
            for m in json.load(open(manifest)):
                expected[m["file"]] = m.get("expected_verdict", "?")

        results = []
        for pdf in pdfs:
            r = analyzer.predict_file(str(pdf))
            cls = r.get("classification", {})
            emg = r.get("emerging_detection", {})
            atk = r.get("attack_classification", {})

            results.append({
                "file": pdf.name,
                "expected": expected.get(pdf.name, "?"),
                "predicted": r["verdict"],
                "match": r["verdict"] == expected.get(pdf.name, "?"),
                "total_sentences":    cls.get("total_sentences", 0),
                "relevant_sentences": cls.get("relevant_sentences", 0),
                "emerging_threat":    emg.get("emerging_threat", False),
                "rule_hits":          emg.get("rule_hits", {}),
                "dominant_topic":     (emg.get("dominant_topic") or {}).get("topic_id"),
                "novelty_ratio":      emg.get("novelty_ratio", 0),
                "reasons":            emg.get("reasons", []),

                # ── NEW: attack-type ──────────────────
                "attack_type":        atk.get("primary_attack_type", "Unknown"),
                "attack_confidence":  atk.get("confidence", 0.0),
                "all_attack_types":   atk.get("all_attack_types", []),
            })

        correct = sum(1 for r in results if r["match"])
        emerging_count = sum(1 for r in results if r["emerging_threat"])

        # Aggregate attack-type distribution
        attack_counter = {}
        for r in results:
            at = r.get("attack_type", "Unknown")
            if at and at != "Unknown":
                attack_counter[at] = attack_counter.get(at, 0) + 1

        return jsonify({
            "total": len(results),
            "correct": correct,
            "accuracy": round(correct / len(results), 4) if results else 0,
            "emerging_flags": emerging_count,
            "attack_type_distribution": attack_counter,
            "results": results,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────────────
# Optional debug route — list all registered URLs
# ─────────────────────────────────────────────────────
@app.route("/routes")
def list_routes():
    rules = []
    for r in app.url_map.iter_rules():
        if r.endpoint == "static":
            continue
        rules.append({
            "rule": str(r),
            "methods": sorted(list(r.methods - {"HEAD", "OPTIONS"})),
        })
    return jsonify(sorted(rules, key=lambda x: x["rule"]))


# ─────────────────────────────────────────────────────
# Startup
# ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  CTI Threat Report Analyzer — Backend")
    print("=" * 60)
    print(f"  UI dir      : {UI_DIR}")
    print(f"  Health check: http://127.0.0.1:5000/health")
    print(f"  UI          : http://127.0.0.1:5000/")
    print(f"  Routes list : http://127.0.0.1:5000/routes")
    print("=" * 60)

    try:
        get_analyzer()  # pre-load models
    except Exception as e:
        print(f"\n⚠  Models failed to load: {e}")
        print("   Run scripts/make_inference.py first, then restart.\n")

    app.run(host="127.0.0.1", port=5000, debug=False)