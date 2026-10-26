"""
Minimal Flask backend for the CTI Analyzer UI.
Wraps the trained models from scripts/make_inference.py
Run: python scripts/serve_ui.py
Then open ui/index.html in a browser.
"""
import json, re, os, sys, io, tempfile
from pathlib import Path
import numpy as np
import joblib
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# Add scripts dir to path so we can import predict_report
sys.path.insert(0, str(Path(__file__).parent))

from predict_report import predict, extract_text  # reuse the CLI module

app = Flask(__name__, static_folder="../ui")
CORS(app)

@app.route("/")
def index():
    return send_from_directory("../ui", "index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        # ── Get input text ────────────────────────────
        if "file" in request.files:
            f = request.files["file"]
            suffix = Path(f.filename).suffix.lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                f.save(tmp.name)
                tmp_path = tmp.name
            try:
                text = extract_text(tmp_path)
                report_name = f.filename
            finally:
                os.unlink(tmp_path)
        elif request.is_json and "text" in request.json:
            text = request.json["text"]
            report_name = "pasted text"
        else:
            return jsonify({"error": "No file or text provided"}), 400

        if not text or len(text.strip()) < 20:
            return jsonify({"error": "Report too short (minimum 20 chars)"}), 400

        # ── Run the analyzer ──────────────────────────
        # predict() expects a path; write to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt",
                                         mode="w", encoding="utf-8") as t:
            t.write(text)
            txt_path = t.name
        try:
            result = predict(txt_path)
        finally:
            os.unlink(txt_path)

        # Override report name for display
        result["report"] = report_name

        # ── Ensure both views present ─────────────────
        if "classification" not in result:
            result["classification"] = {
                "total_sentences":    result.get("total_sentences", 0),
                "relevant_sentences": result.get("relevant_sentences", 0),
                "irrelevant_sentences": result.get("total_sentences", 0) - result.get("relevant_sentences", 0),
                "relevant_fraction":  result.get("relevant_fraction", 0),
            }
        if "emerging_detection" not in result:
            result["emerging_detection"] = {
                "emerging_threat": result.get("emerging_threat", False),
                "novelty_ratio":   result.get("novelty_ratio", 0),
                "rule_hits":       result.get("rule_hits", {"outlier": False, "surge": False, "novelty": False}),
                "dominant_topic":  result.get("dominant_topic"),
                "topic_counts":    result.get("topic_counts", {}),
                "reasons":         result.get("reasons", []),
            }

        return jsonify(result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    print("Starting CTI Analyzer UI backend...")
    print("Open http://127.0.0.1:5000/")
    app.run(host="127.0.0.1", port=5000, debug=False)