# D:\Projects\done\thesis\cyber-threat-detection\scripts\make_ui_server.py
"""Generate the FastAPI server and static UI in one shot.

Run from anywhere:
    python scripts/make_ui_server.py

Creates:
    server/app.py
    server/classifier.py
    server/technique.py
    server/topic_model.py
    server/pdf_reader.py
    server/preprocessing.py
    server/requirements.txt
    ui/index.html
    ui/css/style.css
    ui/js/app.js
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SERVER = PROJECT_ROOT / "server"
UI = PROJECT_ROOT / "ui"
(UI / "css").mkdir(parents=True, exist_ok=True)
(UI / "js").mkdir(parents=True, exist_ok=True)
SERVER.mkdir(parents=True, exist_ok=True)


def write(path: Path, text: str):
    path.write_text(text, encoding="utf-8")
    print("wrote", path.relative_to(PROJECT_ROOT))


# ---------------- server/requirements.txt ----------------
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

# ---------------- server/preprocessing.py ----------------
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

# ---------------- server/pdf_reader.py ----------------
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

# ---------------- server/classifier.py ----------------
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
            raise FileNotFoundError(
                "sbert_relevance.joblib not found — run the notebook first."
            )
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
    """Return list of {'sentence', 'label', 'score'} for relevance."""
    if not sentences:
        return []
    emb = _get_sbert().encode(sentences, batch_size=64, normalize_embeddings=True)
    clf = _load_relevance()
    pred = clf.predict(emb)
    scores = clf.decision_function(emb)
    return [
        {"sentence": s,
         "label": "relevant" if int(p) == 1 else "irrelevant",
         "score": float(sc)}
        for s, p, sc in zip(sentences, pred, scores)
    ]


def predict_techniques(sentences):
    """Return technique predictions for the given sentences."""
    if not sentences:
        return []
    clf = _load_technique()
    if clf is None:
        return []
    emb = _get_sbert().encode(sentences, batch_size=64, normalize_embeddings=True)
    return clf.predict(emb).tolist()
''')

# ---------------- server/topic_model.py ----------------
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
            raise FileNotFoundError(
                "bertopic_model/ not found — run the notebook first."
            )
        _topic_model = BERTopic.load(str(path))
    return _topic_model


def topic_summary(sentences, min_sentences=15):
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

# ---------------- server/app.py ----------------
write(SERVER / "app.py", '''\
"""FastAPI app: upload a PDF, classify relevance, predict technique, list topics."""
from collections import Counter
from pathlib import Path

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
    if not data:
        raise HTTPException(400, "Empty file")
    if len(data) > MAX_BYTES:
        raise HTTPException(413, f"File too large (max {MAX_BYTES // 1024 // 1024} MB)")

    text, err = extract_text_from_bytes(data)
    if err:
        raise HTTPException(422, f"Could not read PDF: {err}")

    sents = split_sentences(text)
    if not sents:
        raise HTTPException(422, "No sentences found")

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


@app.get("/sample/{name}")
def sample(name: str):
    path = RESULTS_DIR / name
    if not path.exists():
        raise HTTPException(404, "not found")
    return FileResponse(path)
''')

# ---------------- ui/index.html ----------------
write(UI / "index.html", """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>CTI Analyser</title>
  <link rel="stylesheet" href="css/style.css" />
</head>
<body>
  <header>
    <h1>Cyber Threat Intelligence Analyser</h1>
    <p class="sub">Upload a CTI PDF to classify relevance, detect technique, and view topics.</p>
  </header>

  <main>
    <section class="upload">
      <input type="file" id="pdfInput" accept="application/pdf" />
      <button id="analyseBtn">Analyse</button>
      <span id="status"></span>
    </section>

    <section id="results" class="hidden">
      <div class="cards">
        <div class="card"><h2>Sentences</h2><p id="nSentences">-</p></div>
        <div class="card"><h2>Relevant</h2><p id="nRelevant">-</p></div>
        <div class="card"><h2>Irrelevant</h2><p id="nIrrelevant">-</p></div>
        <div class="card"><h2>Top technique</h2><p id="topTech">-</p></div>
      </div>

      <h2>Technique distribution</h2>
      <div id="techChart" class="chart"></div>

      <h2>Representative relevant sentences</h2>
      <ul id="sentList"></ul>

      <h2>Topics</h2>
      <pre id="topicsBox"></pre>
    </section>
  </main>

  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <script src="js/app.js"></script>
</body>
</html>
""")

# ---------------- ui/css/style.css ----------------
write(UI / "css" / "style.css", """\
:root { --fg:#111827; --muted:#6b7280; --bg:#f9fafb; --card:#ffffff; --accent:#2563eb; }
* { box-sizing: border-box; }
body { font-family: system-ui, sans-serif; color: var(--fg); background: var(--bg); margin: 0; padding: 2rem; }
header h1 { margin: 0 0 .25rem 0; font-size: 1.6rem; }
.sub { color: var(--muted); margin-top: 0; }
.upload { display: flex; gap: .75rem; align-items: center; margin-bottom: 1.5rem; }
button { background: var(--accent); color: #fff; border: 0; padding: .55rem 1.1rem; border-radius: .4rem; cursor: pointer; }
button:disabled { opacity: .55; cursor: not-allowed; }
#status { color: var(--muted); font-size: .9rem; }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 1rem; }
.card { background: var(--card); border-radius: .6rem; padding: 1rem; box-shadow: 0 1px 2px rgba(0,0,0,.05); }
.card h2 { font-size: .8rem; text-transform: uppercase; letter-spacing: .04em; color: var(--muted); margin: 0 0 .35rem 0; }
.card p { font-size: 1.3rem; margin: 0; font-weight: 600; }
.chart { background: var(--card); border-radius: .6rem; padding: 1rem; margin-bottom: 1rem; min-height: 260px; }
#sentList { background: var(--card); border-radius: .6rem; padding: 1rem 1.5rem; line-height: 1.5; }
#topicsBox { background: var(--card); border-radius: .6rem; padding: 1rem; overflow: auto; max-height: 320px; font-size: .85rem; }
.hidden { display: none; }
""")

# ---------------- ui/js/app.js ----------------
write(UI / "js" / "app.js", """\
const input = document.getElementById("pdfInput");
const btn = document.getElementById("analyseBtn");
const status = document.getElementById("status");
const results = document.getElementById("results");

function setStatus(msg) { status.textContent = msg; }

btn.addEventListener("click", async () => {
  const file = input.files[0];
  if (!file) { setStatus("Choose a PDF first"); return; }

  btn.disabled = true;
  setStatus("Uploading and analysing...");

  try {
    const form = new FormData();
    form.append("file", file);
    const res = await fetch("/analyse", { method: "POST", body: form });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || res.statusText);
    }
    const data = await res.json();
    render(data);
    setStatus("Done");
    results.classList.remove("hidden");
  } catch (e) {
    setStatus("Error: " + e.message);
  } finally {
    btn.disabled = false;
  }
});

function render(data) {
  document.getElementById("nSentences").textContent = data.n_sentences;
  document.getElementById("nRelevant").textContent = data.n_relevant;
  document.getElementById("nIrrelevant").textContent = data.n_irrelevant;
  document.getElementById("topTech").textContent = data.top_technique || "-";

  const techLabels = Object.keys(data.technique_counts || {});
  const techValues = techLabels.map(k => data.technique_counts[k]);
  if (techLabels.length) {
    Plotly.newPlot("techChart", [{
      type: "bar", orientation: "h",
      x: techValues, y: techLabels,
      marker: { color: "#9467bd" }
    }], {
      margin: { l: 220, r: 20, t: 20, b: 40 },
      height: 260,
      xaxis: { title: "sentences" }
    });
  } else {
    document.getElementById("techChart").textContent = "No technique predictions";
  }

  const list = document.getElementById("sentList");
  list.innerHTML = "";
  for (const s of data.top_relevant_sentences || []) {
    const li = document.createElement("li");
    li.textContent = s;
    list.appendChild(li);
  }

  document.getElementById("topicsBox").textContent =
    JSON.stringify(data.topics, null, 2);
}
""")

print("\nAll files written.")
print("Next steps:")
print("  1. cd server")
print("  2. python -m pip install -r requirements.txt")
print("  3. uvicorn app:app --reload --port 8000")
print("  4. open http://localhost:8000")