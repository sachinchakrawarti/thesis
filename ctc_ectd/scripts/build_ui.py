"""
build_ui.py
-----------
Generates the CTI Threat Analysis web UI using Tailwind CSS (Play CDN).

Additions in this build:
  * Multi-file upload in Predict tab (POST /api/classify-upload for each file)
  * Emerging-threat preview panel right below Predict
  * Component-based layout under ui/components/

Run:
    cd D:\\Projects\\done\\thesis\\ctc_ectd
    python scripts\\build_ui.py

Then:
    powershell -ExecutionPolicy Bypass -File scripts\\run_ui.ps1
        -> open http://127.0.0.1:8000
"""

from pathlib import Path


PROJECT_ROOT = Path(r"D:\Projects\done\thesis\ctc_ectd")
BACKEND_DIR = PROJECT_ROOT / "backend"
UI_DIR = PROJECT_ROOT / "ui"
UI_COMPONENTS = UI_DIR / "components"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

for d in (BACKEND_DIR, UI_DIR, UI_COMPONENTS, SCRIPTS_DIR):
    d.mkdir(parents=True, exist_ok=True)


# ===========================================================================
# backend/api.py
# ===========================================================================
API_PY = r'''
"""
api.py - FastAPI backend for the CTI Threat Analysis UI.
"""

from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(r"D:\Projects\done\thesis\ctc_ectd")
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
UI_DIR = PROJECT_ROOT / "ui"
UPLOAD_DIR = PROJECT_ROOT / "uploads"
MODEL_PATH = ARTIFACT_DIR / "cti_text_classifier.joblib"

UPLOAD_DIR.mkdir(exist_ok=True)


app = FastAPI(title="CTI Threat Analysis API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


_model: Optional[Pipeline] = None


def get_model() -> Pipeline:
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail=f"Model not found at {MODEL_PATH}. Run notebook 03 first.",
            )
        _model = joblib.load(MODEL_PATH)
    return _model


class PredictRequest(BaseModel):
    text: str


class PredictionResponse(BaseModel):
    predicted_category: str
    confidence: float
    top_k: List[dict]


@app.get("/")
def root():
    index = UI_DIR / "index.html"
    if not index.exists():
        raise HTTPException(status_code=404, detail="UI not found.")
    return FileResponse(index)


@app.get("/ui/{filename}")
def ui_asset(filename: str):
    path = UI_DIR / filename
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail=f"{filename} not found.")
    return FileResponse(path)


@app.get("/ui/components/{filename}")
def ui_component(filename: str):
    path = UI_DIR / "components" / filename
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail=f"{filename} not found.")
    return FileResponse(path)


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model_present": MODEL_PATH.exists(),
        "artifacts": sorted(p.name for p in ARTIFACT_DIR.glob("*") if p.is_file()),
    }


@app.get("/api/summary")
def summary():
    cleaned = PROCESSED_DIR / "cleaned_reports.pkl"
    uploaded = PROCESSED_DIR / "uploaded_docs.pkl"

    df = pd.read_pickle(cleaned)["df"] if cleaned.exists() else pd.DataFrame()
    docs = pd.read_pickle(uploaded) if uploaded.exists() else pd.DataFrame()

    label_col = None
    for c in ("category", "label", "attack_type"):
        if c in df.columns:
            label_col = c
            break

    return {
        "training_rows": int(len(df)),
        "uploaded_docs": int(len(docs)),
        "categories": int(df[label_col].nunique()) if label_col else 0,
        "label_column": label_col,
    }


@app.get("/api/label-distribution")
def label_distribution():
    cleaned = PROCESSED_DIR / "cleaned_reports.pkl"
    if not cleaned.exists():
        return {"labels": [], "counts": []}
    df = pd.read_pickle(cleaned)["df"]
    label_col = None
    for c in ("category", "label", "attack_type"):
        if c in df.columns:
            label_col = c
            break
    if not label_col:
        return {"labels": [], "counts": []}
    vc = df[label_col].value_counts()
    return {"labels": vc.index.tolist(), "counts": vc.values.tolist()}


@app.get("/api/model-results")
def model_results():
    path = ARTIFACT_DIR / "model_results.csv"
    if not path.exists():
        return []
    return pd.read_csv(path).fillna("").to_dict(orient="records")


@app.get("/api/classified-docs")
def classified_docs():
    path = ARTIFACT_DIR / "classified_documents.csv"
    if not path.exists():
        return []
    return pd.read_csv(path).fillna("").to_dict(orient="records")


@app.get("/api/emerging-alerts")
def emerging_alerts():
    path = ARTIFACT_DIR / "emerging_threat_alerts.csv"
    if not path.exists():
        return []
    return pd.read_csv(path).fillna("").to_dict(orient="records")


@app.get("/api/monthly-shares")
def monthly_shares():
    path = ARTIFACT_DIR / "monthly_topic_shares.csv"
    if not path.exists():
        return {"months": [], "series": {}}
    df = pd.read_csv(path)
    month_col = df.columns[0]
    topic_cols = [c for c in df.columns if c != month_col]
    return {
        "months": df[month_col].tolist(),
        "series": {t: df[t].fillna(0).tolist() for t in topic_cols},
    }


def _predict(text: str) -> PredictionResponse:
    model = get_model()
    if not text.strip():
        raise HTTPException(status_code=400, detail="Empty text.")

    pred = model.predict([text])[0]
    clf = model.named_steps["clf"]

    if hasattr(clf, "predict_proba"):
        proba = model.predict_proba([text])[0]
        classes = clf.classes_
    elif hasattr(clf, "decision_function"):
        raw = model.decision_function([text])[0]
        e = np.exp(raw - raw.max())
        proba = e / e.sum()
        classes = clf.classes_
    else:
        proba = np.array([1.0])
        classes = np.array([pred])

    order = np.argsort(proba)[::-1]
    top_k = [
        {"category": str(classes[i]), "probability": round(float(proba[i]), 4)}
        for i in order[:5]
    ]
    return PredictionResponse(
        predicted_category=str(pred),
        confidence=top_k[0]["probability"],
        top_k=top_k,
    )


@app.post("/api/predict", response_model=PredictionResponse)
def predict(req: PredictRequest):
    return _predict(req.text)


@app.post("/api/classify-upload", response_model=PredictionResponse)
async def classify_upload(file: UploadFile = File(...)):
    import sys
    scripts_dir = PROJECT_ROOT / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    tmp = UPLOAD_DIR / file.filename
    with tmp.open("wb") as f:
        f.write(await file.read())

    try:
        from document_extractor import extract_and_clean  # type: ignore
        text = extract_and_clean(tmp)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {exc}")

    if not text or len(text.strip()) < 20:
        raise HTTPException(status_code=400, detail="Extracted text too short.")

    return _predict(text)
'''


# ===========================================================================
# backend/requirements.txt
# ===========================================================================
REQUIREMENTS_TXT = """\
fastapi==0.111.0
uvicorn[standard]==0.30.1
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.5.0
joblib==1.4.2
python-multipart==0.0.9
"""


# ===========================================================================
# ui/index.html - shell
# ===========================================================================
INDEX_HTML = """\
<!DOCTYPE html>
<html lang="en" class="scroll-smooth">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>CTI Automated Classification &amp; Evolving Threat Forecasting</title>
  <script src="https://cdn.jsdelivr.net/npm/@tailwindcss/browser@4"></script>
  <style type="text/tailwindcss">
    @theme {
      --color-brand-500: #2563eb;
      --color-brand-600: #1d4ed8;
      --color-accent-500: #7c3aed;
    }
  </style>
  <style>
    .chart svg { display:block; max-width:100%; }
    .chart { min-height:340px; display:flex; align-items:center; justify-content:center; }
    ::-webkit-scrollbar { height:6px; width:6px; }
    ::-webkit-scrollbar-thumb { background:#cbd5e1; border-radius:999px; }
    @keyframes fadeIn { from { opacity:0; transform:translateY(4px); } to { opacity:1; transform:translateY(0); } }
    .fade-in { animation: fadeIn 0.25s ease-out; }
  </style>
</head>
<body class="bg-slate-100 text-slate-900 antialiased font-sans">

  <div class="bg-slate-950 border-b border-slate-800 py-3 sticky top-0 z-20">
    <div class="max-w-[1240px] mx-auto px-6 flex items-center justify-between gap-4">
      <div class="flex items-center gap-3">
        <div class="bg-gradient-to-br from-brand-500 to-accent-500 text-white font-extrabold tracking-wide px-3 py-2 rounded-xl text-xs">CTI</div>
        <div>
          <div class="text-slate-200 font-semibold text-sm">Threat Intelligence Platform</div>
          <div class="text-slate-400 text-xs">Automated Classification &amp; Evolving Threat Forecasting</div>
        </div>
      </div>
      <div id="status" class="flex items-center gap-2 text-xs text-slate-300 bg-slate-800/80 px-3 py-1.5 rounded-full">
        <span class="dot w-2 h-2 rounded-full bg-slate-500"></span>
        <span id="status-text">Checking API…</span>
      </div>
    </div>
  </div>

  <header class="bg-gradient-to-b from-slate-950 to-slate-900 text-slate-200 py-10 border-b border-slate-800">
    <div class="max-w-[1240px] mx-auto px-6">
      <h1 class="text-2xl md:text-3xl font-bold text-white leading-tight max-w-4xl">
        Cyber Threat Intelligence Automated Classification and Evolving Threat Forecasting
        <span class="bg-gradient-to-r from-blue-300 to-purple-300 bg-clip-text text-transparent">via Temporal Graph Networks</span>
      </h1>
    </div>
  </header>

  <main class="max-w-[1240px] mx-auto px-6 py-7">
    <section class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
      <div class="bg-white border border-slate-200 rounded-xl p-4 shadow-sm flex items-center gap-3">
        <div class="text-2xl">📊</div>
        <div><span class="text-2xl font-bold" id="kpi-training">—</span>
          <div class="text-slate-500 text-[11px] uppercase tracking-wide">Training reports</div></div>
      </div>
      <div class="bg-white border border-slate-200 rounded-xl p-4 shadow-sm flex items-center gap-3">
        <div class="text-2xl">📄</div>
        <div><span class="text-2xl font-bold" id="kpi-uploads">—</span>
          <div class="text-slate-500 text-[11px] uppercase tracking-wide">Uploaded docs</div></div>
      </div>
      <div class="bg-white border border-slate-200 rounded-xl p-4 shadow-sm flex items-center gap-3">
        <div class="text-2xl">🏷️</div>
        <div><span class="text-2xl font-bold" id="kpi-categories">—</span>
          <div class="text-slate-500 text-[11px] uppercase tracking-wide">Categories</div></div>
      </div>
      <div class="bg-white border border-slate-200 rounded-xl p-4 shadow-sm flex items-center gap-3">
        <div class="text-2xl">⚠️</div>
        <div><span class="text-2xl font-bold" id="kpi-alerts">—</span>
          <div class="text-slate-500 text-[11px] uppercase tracking-wide">Emerging alerts</div></div>
      </div>
    </section>

    <nav id="tabs" class="flex gap-1 mb-4 overflow-x-auto border-b border-slate-200">
      <button class="tab" data-tab="predict">🔍 Predict</button>
      <button class="tab" data-tab="overview">📈 Overview</button>
      <button class="tab" data-tab="models">🧪 Models</button>
      <button class="tab" data-tab="docs">📁 Documents</button>
      <button class="tab" data-tab="temporal">🕒 Temporal</button>
      <button class="tab" data-tab="alerts">🚨 Alerts</button>
    </nav>

    <div id="panel-host" class="bg-white border border-slate-200 rounded-xl p-7 shadow-md min-h-[400px]"></div>
  </main>

  <footer class="text-center text-slate-500 text-xs py-6 pb-10 px-6">
    Research pipeline · Cyber Threat Intelligence Automated Classification &amp; Evolving Threat Forecasting via Temporal Graph Networks
    · Backend: <code id="api-base" class="bg-slate-200 px-2 py-0.5 rounded text-[11px]">—</code>
  </footer>

  <script src="/ui/config.js"></script>
  <script src="/ui/app.js"></script>
</body>
</html>
"""


# ===========================================================================
# ui/config.js
# ===========================================================================
CONFIG_JS = r"""
window.CTI_CONFIG = {
  API_BASE: (window.location.protocol === "file:" || window.location.port !== "8000")
    ? "http://127.0.0.1:8000"
    : "",
  ENDPOINTS: {
    health:   "/api/health",
    summary:  "/api/summary",
    labels:   "/api/label-distribution",
    models:   "/api/model-results",
    docs:     "/api/classified-docs",
    alerts:   "/api/emerging-alerts",
    shares:   "/api/monthly-shares",
    predict:  "/api/predict",
    classify: "/api/classify-upload"
  }
};
"""


# ===========================================================================
# ui/app.js - router + multi-file predict + emerging preview
# ===========================================================================
APP_JS = r"""
// app.js - CTI Threat Analysis UI (component-based, multi-file predict)
(() => {
  "use strict";

  const cfg = window.CTI_CONFIG;
  const api = (p) => `${cfg.API_BASE}${p}`;
  const host = document.getElementById("panel-host");

  const apiBaseEl = document.getElementById("api-base");
  if (apiBaseEl) apiBaseEl.textContent = cfg.API_BASE || "(same origin)";

  // -------------------------------------------------------------------------
  // Fetch helpers
  // -------------------------------------------------------------------------
  async function jget(path) {
    const r = await fetch(api(path));
    if (!r.ok) throw new Error(`${path} -> ${r.status}`);
    return r.json();
  }
  async function jpost(path, body, isForm = false) {
    const opts = isForm
      ? { method: "POST", body }
      : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
    const r = await fetch(api(path), opts);
    const data = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(data.detail || `${path} -> ${r.status}`);
    return data;
  }

  // -------------------------------------------------------------------------
  // Health + KPIs
  // -------------------------------------------------------------------------
  async function checkHealth() {
    const dot = document.querySelector("#status .dot");
    const txt = document.getElementById("status-text");
    try {
      const h = await jget(cfg.ENDPOINTS.health);
      dot.className = "dot w-2 h-2 rounded-full bg-emerald-500";
      txt.textContent = h.model_present ? "API ready" : "API up (no model)";
    } catch {
      dot.className = "dot w-2 h-2 rounded-full bg-red-500";
      txt.textContent = "API unreachable";
    }
  }
  async function loadKPIs() {
    try {
      const s = await jget(cfg.ENDPOINTS.summary);
      document.getElementById("kpi-training").textContent = s.training_rows ?? "—";
      document.getElementById("kpi-uploads").textContent = s.uploaded_docs ?? "—";
      document.getElementById("kpi-categories").textContent = s.categories ?? "—";
    } catch {}
    try {
      const a = await jget(cfg.ENDPOINTS.alerts);
      document.getElementById("kpi-alerts").textContent = Array.isArray(a) ? a.length : 0;
    } catch {}
  }

  // -------------------------------------------------------------------------
  // Table + charts
  // -------------------------------------------------------------------------
  function esc(s) {
    return String(s)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
  }
  function fmt(v) {
    if (v === null || v === undefined) return "";
    if (typeof v === "number") return Number.isInteger(v) ? String(v) : v.toFixed(4);
    return String(v);
  }
  function renderTable(el, rows) {
    if (!el) return;
    const thead = el.querySelector("thead");
    const tbody = el.querySelector("tbody");
    thead.innerHTML = ""; tbody.innerHTML = "";
    if (!rows || !rows.length) {
      tbody.innerHTML = "<tr><td class='text-center text-slate-500 italic py-6'>No data available</td></tr>";
      return;
    }
    const cols = Object.keys(rows[0]);
    thead.innerHTML = `<tr>${cols.map(c =>
      `<th class='text-left bg-slate-50 text-slate-500 font-semibold text-[11px] uppercase tracking-wide px-3 py-2.5 border-b border-slate-200'>${esc(c)}</th>`
    ).join("")}</tr>`;
    tbody.innerHTML = rows.map(r =>
      `<tr class='hover:bg-slate-50 transition'>${cols.map(c =>
        `<td class='text-left px-3 py-2.5 border-b border-slate-100'>${esc(fmt(r[c]))}</td>`
      ).join("")}</tr>`
    ).join("");
  }

  function svgEl(tag, attrs) {
    const e = document.createElementNS("http://www.w3.org/2000/svg", tag);
    for (const [k, v] of Object.entries(attrs || {})) e.setAttribute(k, v);
    return e;
  }
  function barChart(el, labels, values, color = "#2563eb") {
    if (!el) return; el.innerHTML = "";
    if (!labels || !labels.length) {
      el.innerHTML = "<p class='text-slate-500 italic'>No data available</p>";
      return;
    }
    const w = el.clientWidth || 800, h = 340, pL = 60, pR = 20, pT = 20, pB = 90;
    const iW = w - pL - pR, iH = h - pT - pB;
    const max = Math.max(...values, 1), bw = iW / labels.length;
    const svg = svgEl("svg", { width: w, height: h });
    svg.appendChild(svgEl("line", { x1: pL, y1: pT, x2: pL, y2: pT + iH, stroke: "#cbd5e1" }));
    svg.appendChild(svgEl("line", { x1: pL, y1: pT + iH, x2: pL + iW, y2: pT + iH, stroke: "#cbd5e1" }));
    for (let g = 1; g <= 4; g++) {
      const y = pT + iH - (g / 4) * iH;
      svg.appendChild(svgEl("line", { x1: pL, y1: y, x2: pL + iW, y2: y, stroke: "#e2e8f0", "stroke-dasharray": "3 3" }));
    }
    labels.forEach((lab, i) => {
      const v = values[i] || 0, hh = (v / max) * iH;
      const x = pL + i * bw + bw * 0.15, y = pT + iH - hh;
      svg.appendChild(svgEl("rect", { x, y, width: bw * 0.7, height: hh, fill: color, rx: 4 }));
      const t = svgEl("text", { x: pL + i * bw + bw / 2, y: h - pB + 18, "text-anchor": "middle", "font-size": "10", fill: "#334155" });
      t.textContent = lab.length > 12 ? lab.slice(0, 11) + "…" : lab;
      svg.appendChild(t);
      const vt = svgEl("text", { x: pL + i * bw + bw / 2, y: y - 4, "text-anchor": "middle", "font-size": "10", fill: "#0f172a", "font-weight": "600" });
      vt.textContent = v; svg.appendChild(vt);
    });
    el.appendChild(svg);
  }
  function lineChart(el, months, series) {
    if (!el) return; el.innerHTML = "";
    const keys = Object.keys(series || {});
    if (!months || !months.length || !keys.length) {
      el.innerHTML = "<p class='text-slate-500 italic'>No data available</p>";
      return;
    }
    const w = el.clientWidth || 800, h = 380, pL = 60, pR = 20, pT = 20, pB = 90;
    const iW = w - pL - pR, iH = h - pT - pB;
    const colors = ["#2563eb", "#ef4444", "#10b981", "#f59e0b", "#8b5cf6", "#06b6d4"];
    const norm = keys.map(k => {
      const vals = series[k] || [];
      const m = Math.max(...vals, 0.0001);
      return vals.map(v => v / m);
    });
    const svg = svgEl("svg", { width: w, height: h });
    svg.appendChild(svgEl("line", { x1: pL, y1: pT + iH, x2: pL + iW, y2: pT + iH, stroke: "#cbd5e1" }));
    svg.appendChild(svgEl("line", { x1: pL, y1: pT, x2: pL, y2: pT + iH, stroke: "#cbd5e1" }));
    const step = months.length > 1 ? iW / (months.length - 1) : iW;
    norm.forEach((vals, ki) => {
      const pts = vals.map((v, i) => `${pL + i * step},${pT + iH - v * iH}`).join(" ");
      svg.appendChild(svgEl("polyline", { points: pts, fill: "none", stroke: colors[ki % colors.length], "stroke-width": "2" }));
    });
    const sL = Math.max(1, Math.floor(months.length / 10));
    months.forEach((m, i) => {
      if (i % sL) return;
      const t = svgEl("text", { x: pL + i * step, y: h - pB + 20, "text-anchor": "middle", "font-size": "10", fill: "#64748b" });
      t.textContent = m; svg.appendChild(t);
    });
    let lx = pL, ly = h - 20;
    keys.forEach((k, ki) => {
      svg.appendChild(svgEl("circle", { cx: lx + 6, cy: ly - 4, r: 5, fill: colors[ki % colors.length] }));
      const t = svgEl("text", { x: lx + 16, y: ly, "font-size": "11", fill: "#334155" });
      t.textContent = k; svg.appendChild(t); lx += 70;
    });
    el.appendChild(svg);
  }

  // -------------------------------------------------------------------------
  // Predict renderer (MULTI-FILE)
  // -------------------------------------------------------------------------
  function renderPredictionCard(fileName, sizeKB, data) {
    const bars = (data.top_k || []).map(k => `
      <div class="grid grid-cols-[140px_1fr_55px] items-center gap-3 mb-1.5">
        <span class="text-xs font-medium text-slate-700 truncate">${esc(k.category)}</span>
        <div class="bg-slate-200 h-2 rounded-full overflow-hidden">
          <span class="block h-full rounded-full bg-gradient-to-r from-brand-500 to-accent-500 transition-all duration-500"
            style="width:${(k.probability * 100).toFixed(1)}%"></span>
        </div>
        <span class="text-[11px] text-slate-500 text-right tabular-nums">${(k.probability * 100).toFixed(1)}%</span>
      </div>`).join("");
    return `
      <div class="bg-slate-50 border border-slate-200 rounded-xl p-4 fade-in">
        <div class="flex items-center justify-between gap-2 mb-3">
          <div class="min-w-0">
            <div class="text-xs text-slate-500 truncate">📎 ${esc(fileName)}</div>
            <div class="text-[10px] text-slate-400">${sizeKB} KB</div>
          </div>
          <span class="shrink-0 bg-gradient-to-br from-brand-500 to-accent-500 text-white font-bold px-2.5 py-1 rounded-md text-xs">${esc(data.predicted_category)}</span>
        </div>
        <div class="text-[11px] text-slate-500 mb-2">Confidence: <strong class="text-slate-700">${(data.confidence * 100).toFixed(2)}%</strong></div>
        ${bars}
      </div>`;
  }
  function renderPredictionError(fileName, msg) {
    return `
      <div class="bg-red-50 border border-red-200 rounded-xl p-4 fade-in">
        <div class="text-xs text-slate-500 truncate mb-1">📎 ${esc(fileName)}</div>
        <p class="text-red-700 text-xs font-medium">${esc(msg)}</p>
      </div>`;
  }

  const renderers = {
    async predict() {
      const fileInput = document.getElementById("file-input");
      const fileList = document.getElementById("file-list");
      const btn = document.getElementById("btn-upload");
      const result = document.getElementById("predict-result");
      const emergingEl = document.getElementById("predict-emerging");
      if (!fileInput || !btn) return;

      const updateFileList = () => {
        const files = Array.from(fileInput.files || []);
        if (files.length) {
          fileList.classList.remove("hidden");
          fileList.innerHTML = files.map(f =>
            `<li class="flex items-center justify-between gap-2 py-1">
               <span class="truncate text-slate-700">📎 ${esc(f.name)}</span>
               <span class="shrink-0 text-slate-400">${(f.size / 1024).toFixed(1)} KB</span>
             </li>`
          ).join("");
          btn.disabled = false;
          btn.textContent = `▶ Classify ${files.length} file${files.length > 1 ? "s" : ""}`;
        } else {
          fileList.classList.add("hidden");
          fileList.innerHTML = "";
          btn.disabled = true;
          btn.textContent = "▶ Classify";
        }
      };

      fileInput.addEventListener("change", updateFileList);

      btn.addEventListener("click", async () => {
        const files = Array.from(fileInput.files || []);
        if (!files.length) return;
        btn.disabled = true;

        result.innerHTML = `
          <div class="space-y-3" id="result-stack">
            ${files.map(f => `
              <div class="bg-slate-50 border border-slate-200 rounded-xl p-4" id="card-${esc(f.name)}">
                <div class="text-xs text-slate-500 truncate mb-1">📎 ${esc(f.name)}</div>
                <div class="text-slate-400 text-xs">⏳ Classifying…</div>
              </div>`).join("")}
          </div>`;

        const stack = document.getElementById("result-stack");

        for (const f of files) {
          const cardId = `card-${f.name}`;
          const card = document.getElementById(cardId);
          try {
            const fd = new FormData(); fd.append("file", f);
            const data = await jpost(cfg.ENDPOINTS.classify, fd, true);
            if (card) {
              card.outerHTML = `<div id="${cardId}">${renderPredictionCard(f.name, (f.size / 1024).toFixed(1), data)}</div>`;
            }
          } catch (e) {
            if (card) {
              card.outerHTML = `<div id="${cardId}">${renderPredictionError(f.name, e.message)}</div>`;
            }
          }
        }

        btn.disabled = false;
      });

      // Load emerging preview
      if (emergingEl) {
        try {
          const alerts = await jget(cfg.ENDPOINTS.alerts);
          if (!alerts.length) {
            emergingEl.innerHTML = "<p class='text-sm text-slate-500 italic'>No emerging alerts on record.</p>";
          } else {
            const top3 = alerts.slice(0, 3);
            emergingEl.innerHTML = top3.map(a => `
              <div class="flex items-start gap-3 py-2 border-b border-slate-100 last:border-0">
                <span class="shrink-0 bg-amber-100 text-amber-800 font-bold text-[10px] px-2 py-0.5 rounded">⚠ Topic ${esc(a.Topic)}</span>
                <div class="min-w-0 text-xs text-slate-600">
                  <div><strong>${esc(a["Previous Month"])}</strong> → <strong>${esc(a["Current Month"])}</strong></div>
                  <div>Share ${(a["Previous Share"] * 100).toFixed(1)}% → <strong>${(a["Current Share"] * 100).toFixed(1)}%</strong>
                       (+${(a["Increase (pp)"] * 100).toFixed(1)} pp · ${esc(a["Current Report Count"])} reports)</div>
                </div>
              </div>`).join("");
          }
        } catch {
          emergingEl.innerHTML = "<p class='text-sm text-slate-500 italic'>Could not load alerts.</p>";
        }
      }
    },

    async overview() {
      try {
        const ld = await jget(cfg.ENDPOINTS.labels);
        barChart(document.getElementById("chart-labels"), ld.labels, ld.counts);
      } catch {}
    },
    async models() {
      try {
        const m = await jget(cfg.ENDPOINTS.models);
        renderTable(document.getElementById("table-models"), m);
      } catch {}
    },
    async docs() {
      try {
        const d = await jget(cfg.ENDPOINTS.docs);
        renderTable(document.getElementById("table-docs"), d);
      } catch {}
    },
    async temporal() {
      try {
        const s = await jget(cfg.ENDPOINTS.shares);
        lineChart(document.getElementById("chart-temporal"), s.months, s.series);
      } catch {}
    },
    async alerts() {
      try {
        const a = await jget(cfg.ENDPOINTS.alerts);
        renderTable(document.getElementById("table-alerts"), a);
      } catch {}
    },
  };

  // -------------------------------------------------------------------------
  // Router
  // -------------------------------------------------------------------------
  async function loadTab(name) {
    document.querySelectorAll(".tab").forEach(t => {
      const active = t.dataset.tab === name;
      t.classList.toggle("active", active);
      t.classList.toggle("text-brand-600", active);
      t.classList.toggle("border-brand-500", active);
      t.classList.toggle("text-slate-500", !active);
      t.classList.toggle("border-transparent", !active);
    });

    host.innerHTML = "<p class='text-slate-400 text-sm'>Loading…</p>";
    try {
      const html = await fetch(`/ui/components/${name}.html`).then(r => r.text());
      host.innerHTML = html;
      await renderers[name]?.();
    } catch (e) {
      host.innerHTML = `<p class='text-red-600 text-sm'>Failed to load ${esc(name)}: ${esc(e.message)}</p>`;
    }
  }

  function setupTabs() {
    document.querySelectorAll(".tab").forEach(btn => {
      btn.classList.add(
        "border-b-2", "transition", "whitespace-nowrap",
        "px-4", "py-3", "text-sm", "font-medium",
        "bg-transparent", "cursor-pointer"
      );
      btn.addEventListener("click", () => loadTab(btn.dataset.tab));
    });
  }

  let resizeTimer = null;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {
      const active = document.querySelector(".tab.active")?.dataset.tab;
      if (active === "overview" || active === "temporal") renderers[active]?.();
    }, 200);
  });

  document.addEventListener("DOMContentLoaded", () => {
    setupTabs();
    checkHealth();
    loadKPIs();
    loadTab("predict");
  });
})();
"""


# ===========================================================================
# ui/components/predict.html - MULTI-FILE + EMERGING PREVIEW
# ===========================================================================
COMPONENT_PREDICT = """\
<div class="fade-in">
  <h2 class="text-lg font-bold mb-1">Classify CTI documents</h2>
  <p class="text-sm text-slate-500 mb-5">Upload one or more files (PDF, DOCX, TXT, MD, HTML). Each will be classified independently.</p>

  <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
    <div>
      <label for="file-input"
        class="flex flex-col items-center justify-center border-2 border-dashed border-slate-300 hover:border-brand-500 rounded-xl p-8 cursor-pointer bg-slate-50 hover:bg-blue-50/50 transition">
        <div class="text-3xl mb-2">📤</div>
        <div class="text-sm font-medium text-slate-700">Click to choose files</div>
        <div class="text-xs text-slate-500 mt-1">Multiple files supported · PDF · DOCX · TXT · MD · HTML</div>
        <input id="file-input" type="file" class="hidden" multiple
          accept=".pdf,.docx,.txt,.md,.html,.htm" />
      </label>

      <ul id="file-list" class="mt-3 text-xs bg-slate-50 border border-slate-200 rounded-lg p-3 divide-y divide-slate-100 hidden"></ul>

      <button id="btn-upload"
        class="mt-4 bg-brand-500 hover:bg-brand-600 text-white font-semibold px-4 py-2 rounded-lg text-sm transition w-full disabled:opacity-50 disabled:cursor-not-allowed"
        disabled>▶ Classify</button>
    </div>

    <div class="bg-slate-50 border border-slate-200 rounded-xl p-5 min-h-[260px] overflow-y-auto max-h-[600px]" id="predict-result">
      <div class="flex flex-col items-center justify-center text-slate-500 h-full min-h-[220px] gap-2">
        <div class="text-3xl opacity-70">🔮</div>
        <p class="text-sm">Predictions will appear here.</p>
      </div>
    </div>
  </div>

  <!-- EMERGING THREAT PREVIEW -->
  <div class="mt-8">
    <div class="flex items-center justify-between mb-3">
      <div>
        <h3 class="text-base font-bold">🚨 Emerging-threat preview</h3>
        <p class="text-sm text-slate-500">Top 3 topic share spikes — see the <strong>Alerts</strong> tab for the full list.</p>
      </div>
    </div>
    <div id="predict-emerging" class="bg-amber-50 border border-amber-200 rounded-xl p-4 text-sm text-slate-700">
      <p class="text-slate-500 italic text-sm">Loading alerts…</p>
    </div>
  </div>
</div>
"""


# ===========================================================================
# ui/components/overview.html
# ===========================================================================
COMPONENT_OVERVIEW = """\
<div class="fade-in">
  <h2 class="text-lg font-bold mb-1">Threat category distribution</h2>
  <p class="text-sm text-slate-500 mb-5">Reports per threat category in the training corpus.</p>
  <div class="chart" id="chart-labels"></div>
</div>
"""


# ===========================================================================
# ui/components/models.html
# ===========================================================================
COMPONENT_MODELS = """\
<div class="fade-in">
  <h2 class="text-lg font-bold mb-1">Baseline model comparison</h2>
  <p class="text-sm text-slate-500 mb-5">Weighted precision / recall / F1 across 3 baselines.</p>
  <div class="overflow-x-auto">
    <table class="w-full text-sm border-collapse" id="table-models"><thead></thead><tbody></tbody></table>
  </div>
</div>
"""


# ===========================================================================
# ui/components/docs.html
# ===========================================================================
COMPONENT_DOCS = """\
<div class="fade-in">
  <h2 class="text-lg font-bold mb-1">Classified uploaded documents</h2>
  <p class="text-sm text-slate-500 mb-5">Per-document predictions from the trained classifier.</p>
  <div class="overflow-x-auto">
    <table class="w-full text-sm border-collapse" id="table-docs"><thead></thead><tbody></tbody></table>
  </div>
</div>
"""


# ===========================================================================
# ui/components/temporal.html
# ===========================================================================
COMPONENT_TEMPORAL = """\
<div class="fade-in">
  <h2 class="text-lg font-bold mb-1">Monthly topic shares</h2>
  <p class="text-sm text-slate-500 mb-5">NMF topic distribution across months.</p>
  <div class="chart" id="chart-temporal"></div>
</div>
"""


# ===========================================================================
# ui/components/alerts.html
# ===========================================================================
COMPONENT_ALERTS = """\
<div class="fade-in">
  <h2 class="text-lg font-bold mb-1">Emerging-threat alerts</h2>
  <p class="text-sm text-slate-500 mb-5">Topics whose monthly share jumped by ≥ 10 pp. Rows with a large <em>Current Report Count</em> are the meaningful ones.</p>
  <div class="overflow-x-auto">
    <table class="w-full text-sm border-collapse" id="table-alerts"><thead></thead><tbody></tbody></table>
  </div>
</div>
"""


# ===========================================================================
# scripts/run_ui.ps1
# ===========================================================================
RUN_UI_PS1 = """\
# run_ui.ps1 - install deps (once) + start FastAPI backend + static UI.

$ErrorActionPreference = "Stop"
$Root = "D:\\Projects\\done\\thesis\\ctc_ectd"

Write-Host ""
Write-Host "CTI Threat Analysis - UI launcher"
Write-Host "Project root: $Root"
Write-Host ""

Set-Location $Root

Write-Host "Installing backend dependencies..."
pip install -r backend\\requirements.txt | Out-Null

if (-not (Test-Path "artifacts\\cti_text_classifier.joblib")) {
    Write-Host ""
    Write-Host "WARNING: model not found at artifacts\\cti_text_classifier.joblib" -ForegroundColor Yellow
    Write-Host "Run notebooks 01 -> 03 first, then restart this launcher." -ForegroundColor Yellow
    Write-Host ""
}

Write-Host "Starting FastAPI on http://127.0.0.1:8000"
Write-Host "Open your browser at: http://127.0.0.1:8000"
Write-Host "Press Ctrl+C to stop."
Write-Host ""

uvicorn backend.api:app --reload --port 8000
"""


# ===========================================================================
# Write
# ===========================================================================
def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.strip() + "\n", encoding="utf-8-sig")
    print(f"Wrote: {path}")


def main() -> None:
    write(BACKEND_DIR / "api.py", API_PY)
    write(BACKEND_DIR / "requirements.txt", REQUIREMENTS_TXT)

    write(UI_DIR / "index.html", INDEX_HTML)
    write(UI_DIR / "config.js", CONFIG_JS)
    write(UI_DIR / "app.js", APP_JS)

    write(UI_COMPONENTS / "predict.html", COMPONENT_PREDICT)
    write(UI_COMPONENTS / "overview.html", COMPONENT_OVERVIEW)
    write(UI_COMPONENTS / "models.html", COMPONENT_MODELS)
    write(UI_COMPONENTS / "docs.html", COMPONENT_DOCS)
    write(UI_COMPONENTS / "temporal.html", COMPONENT_TEMPORAL)
    write(UI_COMPONENTS / "alerts.html", COMPONENT_ALERTS)

    write(SCRIPTS_DIR / "run_ui.ps1", RUN_UI_PS1)

    stale = UI_DIR / "styles.css"
    if stale.exists():
        stale.unlink()
        print(f"Removed stale: {stale}")

    print()
    print("Multi-file Predict + Emerging preview added.")
    print()
    print("Next steps:")
    print("  1) pip install -r backend/requirements.txt")
    print("  2) powershell -ExecutionPolicy Bypass -File scripts/run_ui.ps1")
    print("  3) Open http://127.0.0.1:8000")


if __name__ == "__main__":
    main()