# scripts/setup_ui.py
"""
Set up the CTI Threat Analysis web UI.

Creates:
    ui/index.html
    ui/styles.css
    ui/config.js
    ui/app.js
    backend/api.py
    backend/requirements.txt
    scripts/run_backend.ps1
    scripts/run_ui.ps1

Usage:
    python scripts/setup_ui.py
    python scripts/setup_ui.py --launch
"""

from pathlib import Path
import argparse
import subprocess
import sys
import time
import webbrowser


PROJECT_ROOT = Path(__file__).resolve().parent.parent
UI_DIR       = PROJECT_ROOT / "ui"
BACKEND_DIR  = PROJECT_ROOT / "backend"
SCRIPTS_DIR  = PROJECT_ROOT / "scripts"


# ==================================================================
# File contents
# ==================================================================
INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>CTI Threat Analysis</title>
  <link rel="stylesheet" href="styles.css" />
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
</head>
<body>
  <header class="app-header">
    <div class="brand">
      <div class="logo">CTI</div>
      <div>
        <h1>Threat Analysis Console</h1>
        <p class="subtitle">Document extraction &middot; classification &middot; emerging threats</p>
      </div>
    </div>
    <div class="status" id="status">
      <span class="dot" id="statusDot"></span>
      <span id="statusText">Checking backend&hellip;</span>
    </div>
  </header>

  <main class="layout">
    <section class="panel upload-panel">
      <h2>1 &middot; Upload CTI documents</h2>
      <p class="hint">Drop PDF / DOCX / TXT / HTML files. Filenames with <code>YYYY-MM</code> improve trend detection.</p>

      <label for="fileInput" class="dropzone" id="dropzone">
        <input id="fileInput" type="file" multiple accept=".pdf,.docx,.doc,.txt,.md,.html,.htm" hidden />
        <div class="dz-icon">&#11014;</div>
        <div class="dz-title">Click to choose files</div>
        <div class="dz-sub">or drag &amp; drop them here</div>
        <ul id="fileList" class="file-list"></ul>
      </label>

      <div class="actions">
        <button id="analyzeBtn" class="btn primary" disabled>Analyze</button>
        <button id="resetBtn"   class="btn ghost">Reset</button>
      </div>
    </section>

    <section class="panel kpis-panel">
      <h2>2 &middot; Summary</h2>
      <div class="kpis">
        <div class="kpi"><span class="kpi-label">Documents</span><span class="kpi-value" id="kpiDocs">&mdash;</span></div>
        <div class="kpi"><span class="kpi-label">Categories</span><span class="kpi-value" id="kpiCats">&mdash;</span></div>
        <div class="kpi"><span class="kpi-label">Alerts</span><span class="kpi-value" id="kpiAlerts">&mdash;</span></div>
        <div class="kpi"><span class="kpi-label">Avg confidence</span><span class="kpi-value" id="kpiConf">&mdash;</span></div>
      </div>
    </section>

    <section class="panel results-panel">
      <h2>3 &middot; Classification results</h2>
      <div class="table-wrap">
        <table id="classTable">
          <thead>
            <tr>
              <th>File</th><th>Words</th><th>Predicted category</th>
              <th>Confidence</th><th>Date</th>
            </tr>
          </thead>
          <tbody></tbody>
        </table>
      </div>
    </section>

    <section class="panel charts-panel">
      <h2>4 &middot; Distributions</h2>
      <div class="charts">
        <div class="chart-card">
          <h3>Predicted categories</h3>
          <canvas id="chartCategories"></canvas>
        </div>
        <div class="chart-card">
          <h3>Monthly dominant topic share</h3>
          <canvas id="chartTopics"></canvas>
        </div>
      </div>
    </section>

    <section class="panel alerts-panel">
      <h2>5 &middot; Emerging-threat alerts</h2>
      <p class="hint">Topic share increases &ge; 10&nbsp;pp with &ge; 2 reports in the current month.</p>
      <div class="table-wrap">
        <table id="alertTable">
          <thead>
            <tr>
              <th>Previous month</th><th>Current month</th><th>Topic</th>
              <th>Prev share</th><th>Curr share</th><th>&Delta; (pp)</th><th>Count</th>
            </tr>
          </thead>
          <tbody></tbody>
        </table>
      </div>
      <div class="empty" id="alertsEmpty" hidden>No alerts yet.</div>
    </section>

    <section class="panel logs-panel">
      <h2>6 &middot; Log</h2>
      <pre id="log"></pre>
    </section>
  </main>

  <footer class="app-footer">
    <span>CTI Threat Analysis &middot; local only</span>
    <span id="backendUrl"></span>
  </footer>

  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.3/dist/chart.umd.min.js"></script>
  <script src="config.js"></script>
  <script src="app.js"></script>
</body>
</html>
"""


STYLES_CSS = """:root {
  --bg: #0b1020;
  --panel: #141a2e;
  --panel-2: #1a2138;
  --border: #2a3354;
  --text: #e6ecff;
  --muted: #8b94b8;
  --accent: #5b8def;
  --accent-2: #8b5cf6;
  --ok: #22c55e;
  --warn: #f59e0b;
  --danger: #ef4444;
  --radius: 12px;
  --shadow: 0 10px 30px rgba(0,0,0,.35);
}

* { box-sizing: border-box; }

html, body {
  margin: 0;
  padding: 0;
  background: radial-gradient(1200px 800px at 20% -10%, #16213a 0%, var(--bg) 60%) no-repeat;
  color: var(--text);
  font-family: 'Inter', system-ui, sans-serif;
  min-height: 100vh;
}

.app-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 18px 28px;
  border-bottom: 1px solid var(--border);
  background: rgba(10,15,30,.6);
  backdrop-filter: blur(8px);
  position: sticky; top: 0; z-index: 5;
}

.brand { display: flex; gap: 14px; align-items: center; }
.logo {
  width: 44px; height: 44px; border-radius: 12px;
  background: linear-gradient(135deg, var(--accent), var(--accent-2));
  display: grid; place-items: center;
  font-weight: 700; letter-spacing: .5px;
}
.app-header h1 { font-size: 18px; margin: 0; }
.subtitle { margin: 2px 0 0; color: var(--muted); font-size: 12px; }

.status { display: flex; align-items: center; gap: 8px; font-size: 13px; color: var(--muted); }
.dot { width: 10px; height: 10px; border-radius: 50%; background: var(--warn); box-shadow: 0 0 8px currentColor; }
.dot.ok { background: var(--ok); }
.dot.err { background: var(--danger); }

.layout {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 20px;
  padding: 22px 28px 60px;
  max-width: 1400px; margin: 0 auto;
}
.panel {
  background: linear-gradient(180deg, var(--panel), var(--panel-2));
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 18px 20px;
  box-shadow: var(--shadow);
}
.panel h2 { font-size: 14px; letter-spacing: .4px; text-transform: uppercase; color: var(--muted); margin: 0 0 12px; }
.panel h3 { font-size: 13px; margin: 0 0 8px; color: var(--text); }

.upload-panel { grid-column: 1 / 2; }
.kpis-panel   { grid-column: 2 / 3; }
.results-panel{ grid-column: 1 / -1; }
.charts-panel { grid-column: 1 / -1; }
.alerts-panel { grid-column: 1 / -1; }
.logs-panel   { grid-column: 1 / -1; }

.hint { color: var(--muted); font-size: 12.5px; margin: 0 0 10px; }
.hint code { background: #0f1730; padding: 1px 5px; border-radius: 4px; }

.dropzone {
  display: block; padding: 26px 20px; border: 1.5px dashed var(--border);
  border-radius: var(--radius); cursor: pointer; text-align: center;
  transition: border-color .15s, background .15s;
}
.dropzone.drag { border-color: var(--accent); background: rgba(91,141,239,.07); }
.dz-icon { font-size: 26px; margin-bottom: 6px; color: var(--accent); }
.dz-title { font-weight: 600; margin-bottom: 4px; }
.dz-sub { color: var(--muted); font-size: 12.5px; }

.file-list { list-style: none; padding: 0; margin: 14px 0 0; font-size: 12.5px; color: var(--muted); }
.file-list li { padding: 4px 0; border-bottom: 1px dashed var(--border); display: flex; justify-content: space-between; }
.file-list li:last-child { border-bottom: 0; }

.actions { display: flex; gap: 10px; margin-top: 14px; }
.btn {
  border: 1px solid var(--border); background: #0f1730; color: var(--text);
  border-radius: 8px; padding: 9px 16px; font-weight: 600; font-size: 13px;
  cursor: pointer; transition: transform .05s, background .15s, border-color .15s;
}
.btn:hover:not(:disabled) { background: #16213a; border-color: #3a477a; }
.btn:active:not(:disabled) { transform: translateY(1px); }
.btn:disabled { opacity: .5; cursor: not-allowed; }
.btn.primary { background: linear-gradient(135deg, var(--accent), var(--accent-2)); border-color: transparent; }
.btn.ghost { background: transparent; }

.kpis { display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px; }
.kpi {
  background: #0f1730; border: 1px solid var(--border); border-radius: 10px;
  padding: 12px 14px; display: flex; flex-direction: column; gap: 4px;
}
.kpi-label { color: var(--muted); font-size: 11.5px; letter-spacing: .4px; text-transform: uppercase; }
.kpi-value { font-size: 22px; font-weight: 700; }

.table-wrap { overflow: auto; max-height: 340px; border-radius: 10px; border: 1px solid var(--border); }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
thead th {
  text-align: left; padding: 10px 12px; background: #0f1730; color: var(--muted);
  font-weight: 600; text-transform: uppercase; letter-spacing: .4px; font-size: 11px;
  position: sticky; top: 0;
}
tbody td { padding: 9px 12px; border-top: 1px solid var(--border); }
tbody tr:hover { background: rgba(91,141,239,.05); }

.badge { display: inline-block; padding: 2px 8px; border-radius: 20px; font-size: 11.5px; font-weight: 600; }
.badge.ok    { background: rgba(34,197,94,.15); color: #86efac; }
.badge.warn  { background: rgba(245,158,11,.15); color: #fcd34d; }
.badge.bad   { background: rgba(239,68,68,.15); color: #fca5a5; }
.badge.info  { background: rgba(91,141,239,.15); color: #93c5fd; }

.charts { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
.chart-card {
  background: #0f1730; border: 1px solid var(--border); border-radius: 10px;
  padding: 12px; min-height: 300px;
}
.chart-card canvas { max-height: 280px; }

.empty { color: var(--muted); font-style: italic; padding: 8px 0; }

.logs-panel pre {
  background: #0a0f1e; border: 1px solid var(--border); border-radius: 8px;
  padding: 12px; max-height: 220px; overflow: auto; font-size: 12px;
  color: #c8d3ff; line-height: 1.5;
}

.app-footer {
  display: flex; justify-content: space-between;
  padding: 14px 28px; color: var(--muted); font-size: 12px;
  border-top: 1px solid var(--border);
}

@media (max-width: 900px) {
  .layout { grid-template-columns: 1fr; }
  .upload-panel, .kpis-panel, .results-panel,
  .charts-panel, .alerts-panel, .logs-panel { grid-column: 1 / -1; }
  .charts { grid-template-columns: 1fr; }
}
"""


CONFIG_JS = """// Point at the FastAPI backend started by scripts/run_backend.ps1
window.CTI_CONFIG = {
  API_BASE: "http://127.0.0.1:8000",
  ENDPOINTS: {
    health:   "/api/health",
    analyze:  "/api/analyze"
  }
};
"""


APP_JS = """(() => {
  const cfg = window.CTI_CONFIG;
  const $  = (s) => document.querySelector(s);

  const statusDot   = $("#statusDot");
  const statusText  = $("#statusText");
  const dropzone    = $("#dropzone");
  const fileInput   = $("#fileInput");
  const fileList    = $("#fileList");
  const analyzeBtn  = $("#analyzeBtn");
  const resetBtn    = $("#resetBtn");
  const logEl       = $("#log");
  const backendUrl  = $("#backendUrl");

  let selectedFiles = [];
  let chartCats = null;
  let chartTopics = null;

  const log = (msg, cls = "") => {
    const t = new Date().toLocaleTimeString();
    logEl.innerHTML += `<span class="${cls}">[${t}] ${msg}</span>\\n`;
    logEl.scrollTop = logEl.scrollHeight;
  };

  const setStatus = (kind, text) => {
    statusDot.className = "dot " + (kind === "ok" ? "ok" : kind === "err" ? "err" : "");
    statusText.textContent = text;
  };

  const fmt = (n, d = 3) =>
    (typeof n === "number" ? n.toFixed(d) : "\\u2014");

  async function checkBackend() {
    try {
      const r = await fetch(cfg.API_BASE + cfg.ENDPOINTS.health);
      const j = await r.json();
      setStatus("ok", `Backend online \\u00b7 ${j.model || "model loaded"}`);
      log("Backend health OK");
    } catch (e) {
      setStatus("err", "Backend offline");
      log("Backend not reachable. Start it with scripts\\\\run_backend.ps1", "err");
    }
    backendUrl.textContent = cfg.API_BASE;
  }

  function refreshFileList() {
    fileList.innerHTML = "";
    selectedFiles.forEach((f) => {
      const li = document.createElement("li");
      li.innerHTML = `<span>${f.name}</span><span>${(f.size/1024).toFixed(1)} KB</span>`;
      fileList.appendChild(li);
    });
    analyzeBtn.disabled = selectedFiles.length === 0;
  }

  function addFiles(list) {
    const allowed = [".pdf", ".docx", ".doc", ".txt", ".md", ".html", ".htm"];
    for (const f of list) {
      const ext = "." + f.name.split(".").pop().toLowerCase();
      if (!allowed.includes(ext)) {
        log(`Skipped unsupported file: ${f.name}`, "err");
        continue;
      }
      if (!selectedFiles.some((x) => x.name === f.name && x.size === f.size)) {
        selectedFiles.push(f);
      }
    }
    refreshFileList();
  }

  fileInput.addEventListener("change", (e) => addFiles(e.target.files));

  ["dragenter", "dragover"].forEach((ev) =>
    dropzone.addEventListener(ev, (e) => {
      e.preventDefault(); dropzone.classList.add("drag");
    })
  );
  ["dragleave", "drop"].forEach((ev) =>
    dropzone.addEventListener(ev, (e) => {
      e.preventDefault(); dropzone.classList.remove("drag");
    })
  );
  dropzone.addEventListener("drop", (e) => addFiles(e.dataTransfer.files));

  resetBtn.addEventListener("click", () => {
    selectedFiles = [];
    fileInput.value = "";
    refreshFileList();
    renderTable([]);
    renderAlerts([]);
    renderKpis({});
    destroyCharts();
    log("Reset.");
  });

  analyzeBtn.addEventListener("click", async () => {
    if (!selectedFiles.length) return;
    analyzeBtn.disabled = true;
    analyzeBtn.textContent = "Analyzing\\u2026";

    const form = new FormData();
    selectedFiles.forEach((f) => form.append("files", f, f.name));

    try {
      log(`Uploading ${selectedFiles.length} file(s)\\u2026`);
      const r = await fetch(cfg.API_BASE + cfg.ENDPOINTS.analyze, {
        method: "POST",
        body: form
      });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const data = await r.json();
      log("Analysis complete.");
      handleResults(data);
    } catch (e) {
      log("Error: " + e.message, "err");
      setStatus("err", "Analysis failed");
    } finally {
      analyzeBtn.disabled = false;
      analyzeBtn.textContent = "Analyze";
    }
  });

  function handleResults(data) {
    const docs    = data.documents || [];
    const alerts  = data.alerts || [];
    const topics  = data.topics || null;

    renderKpis({
      docs: docs.length,
      cats: new Set(docs.map((d) => d.predicted_category)).size,
      alerts: alerts.length,
      conf: docs.length
        ? docs.reduce((s, d) => s + (d.confidence || 0), 0) / docs.length
        : 0
    });

    renderTable(docs);
    renderAlerts(alerts);
    renderCategoryChart(docs);
    if (topics) renderTopicsChart(topics);
  }

  function renderKpis(o) {
    $("#kpiDocs").textContent   = o.docs   ?? "\\u2014";
    $("#kpiCats").textContent   = o.cats   ?? "\\u2014";
    $("#kpiAlerts").textContent = o.alerts ?? "\\u2014";
    $("#kpiConf").textContent   = typeof o.conf === "number" ? o.conf.toFixed(3) : "\\u2014";
  }

  function renderTable(docs) {
    const tb = document.querySelector("#classTable tbody");
    tb.innerHTML = "";
    docs.forEach((d) => {
      const conf = d.confidence ?? 0;
      const badge = conf >= 0.8 ? "ok" : conf >= 0.5 ? "warn" : "bad";
      const tr = document.createElement("tr");
      tr.innerHTML =
        `<td>${d.file_name}</td>` +
        `<td>${d.word_count ?? "\\u2014"}</td>` +
        `<td><span class="badge info">${d.predicted_category}</span></td>` +
        `<td><span class="badge ${badge}">${fmt(conf)}</span></td>` +
        `<td>${d.parsed_date ?? "\\u2014"}</td>`;
      tb.appendChild(tr);
    });
  }

  function renderAlerts(alerts) {
    const tb = document.querySelector("#alertTable tbody");
    tb.innerHTML = "";
    const empty = document.getElementById("alertsEmpty");
    if (!alerts.length) {
      empty.hidden = false;
      return;
    }
    empty.hidden = true;
    alerts.forEach((a) => {
      const tr = document.createElement("tr");
      tr.innerHTML =
        `<td>${a.previous_month}</td>` +
        `<td>${a.current_month}</td>` +
        `<td><span class="badge info">T${a.topic}</span></td>` +
        `<td>${fmt(a.previous_share)}</td>` +
        `<td>${fmt(a.current_share)}</td>` +
        `<td><span class="badge warn">+${fmt(a.increase)}</span></td>` +
        `<td>${a.current_count}</td>`;
      tb.appendChild(tr);
    });
  }

  function destroyCharts() {
    if (chartCats)   { chartCats.destroy();   chartCats = null; }
    if (chartTopics) { chartTopics.destroy(); chartTopics = null; }
  }

  function renderCategoryChart(docs) {
    const counts = {};
    docs.forEach((d) => {
      counts[d.predicted_category] = (counts[d.predicted_category] || 0) + 1;
    });
    const labels = Object.keys(counts);
    const values = Object.values(counts);

    if (chartCats) chartCats.destroy();
    chartCats = new Chart(document.getElementById("chartCategories"), {
      type: "bar",
      data: {
        labels,
        datasets: [{
          label: "Documents",
          data: values,
          backgroundColor: "rgba(91,141,239,.65)",
          borderColor: "#5b8def",
          borderWidth: 1,
          borderRadius: 6
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { ticks: { color: "#8b94b8" }, grid: { color: "#2a3354" } },
          y: { ticks: { color: "#8b94b8" }, grid: { color: "#2a3354" },
               beginAtZero: true }
        }
      }
    });
  }

  function renderTopicsChart(topics) {
    const palette = ["#5b8def","#8b5cf6","#22c55e","#f59e0b","#ef4444"];
    const datasets = topics.series.map((s, i) => ({
      label: s.name,
      data: s.data,
      borderColor: palette[i % palette.length],
      backgroundColor: palette[i % palette.length] + "33",
      tension: 0.3,
      fill: false,
      pointRadius: 3
    }));

    if (chartTopics) chartTopics.destroy();
    chartTopics = new Chart(document.getElementById("chartTopics"), {
      type: "line",
      data: { labels: topics.months, datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { labels: { color: "#8b94b8" } } },
        scales: {
          x: { ticks: { color: "#8b94b8" }, grid: { color: "#2a3354" } },
          y: { ticks: { color: "#8b94b8" }, grid: { color: "#2a3354" },
               beginAtZero: true, max: 1 }
        }
      }
    });
  }

  refreshFileList();
  checkBackend();
})();
"""


BACKEND_API_PY = '''"""
FastAPI backend for the CTI Threat Analysis UI.
Wraps scripts/document_extractor.py + a TF-IDF + LogisticRegression classifier.

Run:
    uvicorn backend.api:app --reload --port 8000
"""

from __future__ import annotations

import re
import sys
import tempfile
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR  = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from document_extractor import extract_and_clean, EXTRACTORS  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer   # noqa: E402
from sklearn.linear_model import LogisticRegression           # noqa: E402
from sklearn.decomposition import NMF                         # noqa: E402
from sklearn.pipeline import Pipeline                         # noqa: E402


RANDOM_STATE = 42
EMERGING_MIN_SHARE_JUMP = 0.10
EMERGING_MIN_COUNT = 2

DATASET   = PROJECT_ROOT / "dataset" / "synthetic_cti_reports.csv"
ARTIFACTS = PROJECT_ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)

_model: Optional[Pipeline] = None

app = FastAPI(title="CTI Threat Analysis API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def train_classifier() -> Pipeline:
    global _model
    if _model is not None:
        return _model

    if not DATASET.exists():
        raise FileNotFoundError(f"Training data not found: {DATASET}")

    df = pd.read_csv(DATASET)
    text_col = "text" if "text" in df.columns else df.columns[1]
    label_col = next((c for c in ("category", "label", "attack_type")
                      if c in df.columns), None)
    if label_col is None:
        raise ValueError("No label column found in dataset.")

    title_col = "title" if "title" in df.columns else None
    df = df.dropna(subset=[text_col, label_col]).copy()
    df[text_col] = df[text_col].astype(str).str.strip()

    if title_col:
        df["model_text"] = (df[title_col].fillna("") + " " + df[text_col]).str.strip()
    else:
        df["model_text"] = df[text_col]

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=10000, ngram_range=(1, 2),
            stop_words="english", sublinear_tf=True)),
        ("clf", LogisticRegression(
            max_iter=2000, class_weight="balanced",
            random_state=RANDOM_STATE)),
    ])
    pipe.fit(df["model_text"], df[label_col])
    _model = pipe
    return pipe


DATE_TOKEN_RE = re.compile(r"\\b(19|20)\\d{2}[-_/]?\\d{0,2}\\b")
DATE_RE       = re.compile(r"(20\\d{2})[-_/](\\d{1,2})")


def strip_meta_tokens(text: str) -> str:
    text = DATE_TOKEN_RE.sub(" ", text)
    text = re.sub(r"\\b\\d{1,4}\\b", " ", text)
    return re.sub(r"\\s+", " ", text).strip()


def parse_date_from_name(name: str):
    m = DATE_RE.search(name)
    if not m:
        return None
    try:
        return pd.Timestamp(year=int(m.group(1)),
                            month=int(m.group(2)), day=1)
    except Exception:
        return None


@app.get("/api/health")
def health():
    try:
        train_classifier()
        return {"status": "ok", "model": "logistic_regression"}
    except Exception as exc:
        return JSONResponse({"status": "err", "error": str(exc)}, status_code=500)


@app.post("/api/analyze")
async def analyze(files: List[UploadFile] = File(...)):
    docs = []
    for uf in files:
        ext = Path(uf.filename).suffix.lower()
        if ext not in EXTRACTORS:
            continue
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(await uf.read())
            tmp_path = Path(tmp.name)

        try:
            txt = extract_and_clean(tmp_path)
        except Exception as exc:
            docs.append({
                "file_name": uf.filename,
                "extension": ext,
                "word_count": 0,
                "char_count": 0,
                "text": "",
                "error": str(exc),
            })
            continue

        docs.append({
            "file_name": uf.filename,
            "extension": ext,
            "word_count": len(txt.split()),
            "char_count": len(txt),
            "text": txt,
        })

    if not docs:
        return {"documents": [], "alerts": [], "topics": None}

    model = train_classifier()
    df = pd.DataFrame(docs)
    df["predicted_category"] = model.predict(df["text"])

    clf = model.named_steps["clf"]
    if hasattr(clf, "predict_proba"):
        probs = model.predict_proba(df["text"])
        df["confidence"] = probs.max(axis=1)
    elif hasattr(clf, "decision_function"):
        scores = model.decision_function(df["text"])
        df["confidence"] = np.max(scores, axis=1)
    else:
        df["confidence"] = np.nan

    df["parsed_date"] = df["file_name"].apply(parse_date_from_name)
    df["parsed_date"] = df["parsed_date"].apply(
        lambda d: d.strftime("%Y-%m") if pd.notna(d) else None)

    dated = df.dropna(subset=["parsed_date"]).copy()
    topics_payload = None
    alerts = []

    if len(dated) >= 3:
        dated["text_clean"] = dated["text"].apply(strip_meta_tokens)
        vec = TfidfVectorizer(
            max_features=5000,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
            token_pattern=r"(?u)\\b[a-zA-Z][a-zA-Z\\-]+\\b",
        )
        M = vec.fit_transform(dated["text_clean"])
        k = min(5, M.shape[0], M.shape[1])

        if k >= 2:
            nmf = NMF(n_components=k, random_state=RANDOM_STATE,
                      init="nndsvda", max_iter=500)
            W = nmf.fit_transform(M)
            for i in range(k):
                dated[f"Topic_{i+1}"] = W[:, i]
            dated["dominant_topic"] = W.argmax(axis=1) + 1

            counts = pd.crosstab(dated["parsed_date"], dated["dominant_topic"])
            shares = counts.div(counts.sum(axis=1), axis=0).fillna(0)
            months = list(shares.index)

            topics_payload = {
                "months": months,
                "series": [
                    {
                        "name": f"Topic {int(t)}",
                        "data": [round(float(shares.loc[m, t]), 3) for m in months],
                    }
                    for t in shares.columns
                ],
            }

            for i in range(1, len(months)):
                pm, cm = months[i-1], months[i]
                for tid in shares.columns:
                    ps  = float(shares.loc[pm, tid])
                    cs  = float(shares.loc[cm, tid])
                    cnt = int(counts.loc[cm, tid])
                    if (cs - ps) >= EMERGING_MIN_SHARE_JUMP and cnt >= EMERGING_MIN_COUNT:
                        alerts.append({
                            "previous_month": pm,
                            "current_month": cm,
                            "topic": int(tid),
                            "previous_share": round(ps, 3),
                            "current_share": round(cs, 3),
                            "increase": round(cs - ps, 3),
                            "current_count": cnt,
                        })

    out = df[["file_name", "word_count", "predicted_category",
              "confidence", "parsed_date"]].copy()
    out["confidence"] = out["confidence"].round(4)
    out.to_csv(ARTIFACTS / "uploaded_classifications.csv", index=False)

    if alerts:
        pd.DataFrame(alerts).to_csv(
            ARTIFACTS / "emerging_threat_alerts.csv", index=False)

    return {
        "documents": out.replace({np.nan: None}).to_dict(orient="records"),
        "alerts": alerts,
        "topics": topics_payload,
    }
'''


BACKEND_REQ = """fastapi>=0.111
uvicorn[standard]>=0.30
python-multipart>=0.0.9
pypdf>=4.2
python-docx>=1.1
beautifulsoup4>=4.12
lxml>=5.2
scikit-learn>=1.4
pandas>=2.2
numpy>=1.26
"""


RUN_BACKEND_PS1 = """# scripts/run_backend.ps1
# Starts the FastAPI backend on http://127.0.0.1:8000

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path "$PSScriptRoot\\..").Path
Set-Location $ProjectRoot

Write-Host "Installing backend dependencies (first run only)..." -ForegroundColor Cyan
pip install -r backend\\requirements.txt

Write-Host ""
Write-Host "Starting FastAPI on http://127.0.0.1:8000 ..." -ForegroundColor Green
uvicorn backend.api:app --reload --port 8000
"""


RUN_UI_PS1 = """# scripts/run_ui.ps1
# Serves the static UI on http://127.0.0.1:5500

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path "$PSScriptRoot\\..").Path
Set-Location $ProjectRoot

$Port = 5500
Write-Host "Serving ui/ on http://127.0.0.1:$Port ..." -ForegroundColor Green
python -m http.server $Port --directory ui
"""


# ==================================================================
# Writer
# ==================================================================
def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    print(f"  ✓ {path.relative_to(PROJECT_ROOT)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch", action="store_true",
                        help="Launch backend + UI in new windows")
    args = parser.parse_args()

    print(f"Project root: {PROJECT_ROOT}")
    print("Writing UI + backend files...")

    write(UI_DIR / "index.html",            INDEX_HTML)
    write(UI_DIR / "styles.css",            STYLES_CSS)
    write(UI_DIR / "config.js",             CONFIG_JS)
    write(UI_DIR / "app.js",                APP_JS)
    write(BACKEND_DIR / "api.py",           BACKEND_API_PY)
    write(BACKEND_DIR / "requirements.txt", BACKEND_REQ)
    write(SCRIPTS_DIR / "run_backend.ps1",  RUN_BACKEND_PS1)
    write(SCRIPTS_DIR / "run_ui.ps1",       RUN_UI_PS1)

    print("\n✅ Done. Next steps:")
    print("   1) .\\scripts\\run_backend.ps1     (terminal 1)")
    print("   2) .\\scripts\\run_ui.ps1          (terminal 2)")
    print("   3) Open http://127.0.0.1:5500 in your browser")

    if args.launch:
        print("\nLaunching both servers in new PowerShell windows...")
        subprocess.Popen([
            "powershell", "-NoExit", "-ExecutionPolicy", "Bypass",
            "-File", str(SCRIPTS_DIR / "run_backend.ps1"),
        ], creationflags=subprocess.CREATE_NEW_CONSOLE)
        time.sleep(3)
        subprocess.Popen([
            "powershell", "-NoExit", "-ExecutionPolicy", "Bypass",
            "-File", str(SCRIPTS_DIR / "run_ui.ps1"),
        ], creationflags=subprocess.CREATE_NEW_CONSOLE)
        time.sleep(2)
        webbrowser.open("http://127.0.0.1:5500")

    return 0


if __name__ == "__main__":
    sys.exit(main())