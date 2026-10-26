// =========================================================
// CTI Threat Report Analyzer — Frontend logic
// =========================================================
const API_BASE = "http://127.0.0.1:5000";

// ── DOM refs ─────────────────────────────────────────────
const dropzone    = document.getElementById("dropzone");
const fileInput   = document.getElementById("file-input");
const filenameEl  = document.getElementById("filename");
const inputPanel  = document.getElementById("input-panel");
const loadingPanel = document.getElementById("loading-panel");
const resultsPanel = document.getElementById("results-panel");
const verdictCard = document.getElementById("verdict-card");
const textInput   = document.getElementById("text-input");
const errorBox    = document.getElementById("error-box");

let selectedFile = null;

// ── File drop handling ──────────────────────────────────
dropzone.addEventListener("click", () => fileInput.click());

dropzone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropzone.classList.add("dragover");
});

dropzone.addEventListener("dragleave", () => {
  dropzone.classList.remove("dragover");
});

dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.classList.remove("dragover");
  if (e.dataTransfer.files.length > 0) selectFile(e.dataTransfer.files[0]);
});

fileInput.addEventListener("change", () => {
  if (fileInput.files.length > 0) selectFile(fileInput.files[0]);
});

function selectFile(file) {
  const name = file.name.toLowerCase();
  if (!name.endsWith(".pdf") && !name.endsWith(".txt")) {
    showError("Please upload a PDF or TXT file.");
    return;
  }
  selectedFile = file;
  filenameEl.textContent = `✓ ${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
  filenameEl.classList.remove("hidden");
  hideError();
}

// ── Sample loader ────────────────────────────────────────
document.getElementById("sample-btn").addEventListener("click", () => {
  textInput.value =
`The threat actor exploited CVE-2023-23397 to gain initial access via Outlook.
A malicious DLL named glib-2.0.dll was dropped in the %PUBLIC% folder.
The loader contacted a hardcoded C2 server at 185.220.101.15 over an SSL socket.
Credentials were harvested from compromised cloud accounts using a phishing kit.
Ransomware activity was observed against the subsidiary, with the CyberVolk family encrypting files.
North Korean actors were attributed to a separate campaign targeting financial institutions.`;
  selectedFile = null;
  filenameEl.classList.add("hidden");
});

// ── Single analysis ─────────────────────────────────────
document.getElementById("analyze-btn").addEventListener("click", async () => {
  hideError();
  const text = textInput.value.trim();
  if (!selectedFile && !text) {
    showError("Please upload a file or paste text before analyzing.");
    return;
  }

  inputPanel.classList.add("hidden");
  loadingPanel.classList.remove("hidden");
  resultsPanel.classList.add("hidden");

  try {
    let response;
    if (selectedFile) {
      const fd = new FormData();
      fd.append("file", selectedFile);
      response = await fetch(`${API_BASE}/analyze`, { method: "POST", body: fd });
    } else {
      response = await fetch(`${API_BASE}/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
    }
    if (!response.ok) {
      const errBody = await response.json().catch(() => ({}));
      throw new Error(errBody.error || `HTTP ${response.status}`);
    }
    const result = await response.json();
    renderResult(result);
  } catch (err) {
    loadingPanel.classList.add("hidden");
    inputPanel.classList.remove("hidden");
    showError(`Analysis failed: ${err.message}`);
  }
});

// ── Reset ───────────────────────────────────────────────
document.getElementById("reset-btn").addEventListener("click", () => {
  resultsPanel.classList.add("hidden");
  inputPanel.classList.remove("hidden");
  textInput.value = "";
  selectedFile = null;
  filenameEl.classList.add("hidden");
  fileInput.value = "";
});

// ── Render result card ──────────────────────────────────
function renderResult(r) {
  const verdict = r.verdict || "not_relevant";
  const cls = r.classification || {};
  const emg = r.emerging_detection || {};
  const atk = r.attack_classification || {};

  const banner = {
    known:        { icon: "✔", label: "KNOWN THREAT PATTERN" },
    emerging:     { icon: "⚠", label: "EMERGING THREAT SIGNAL" },
    not_relevant: { icon: "—", label: "NOT RELEVANT" },
  }[verdict] || { icon: "?", label: verdict.toUpperCase() };

  const reportName = r.report ? r.report.split(/[\\/]/).pop() : "pasted text";

  // Attack-type block
  let attackHtml = "";
  if (atk.primary_attack_type && atk.primary_attack_type !== "Unknown") {
    const allTypes = (atk.all_attack_types || []).slice(0, 4).map(a =>
      `<span class="pill gray">${a.type} <b>${a.count}</b></span>`
    ).join("");
    attackHtml = `
      <div class="section">
        <div class="section-title">Attack-Type Classification</div>
        <div class="attack-type-card">
          <div class="type">${atk.primary_attack_type}</div>
          <div class="confidence">Confidence: ${(atk.confidence * 100).toFixed(0)}%</div>
        </div>
        <div class="topic-pills">${allTypes}</div>
      </div>`;
  }

  // Classification metrics
  const clsGrid = `
    <div class="metric-grid">
      <div class="metric"><div class="label">Total sentences</div>
        <div class="value">${cls.total_sentences ?? r.total_sentences ?? 0}</div></div>
      <div class="metric"><div class="label">Relevant sentences</div>
        <div class="value">${cls.relevant_sentences ?? r.relevant_sentences ?? 0}</div></div>
      <div class="metric"><div class="label">Relevant fraction</div>
        <div class="value">${(((cls.relevant_fraction ?? r.relevant_fraction ?? 0) * 100)).toFixed(1)}%</div></div>
    </div>`;

  // Emerging metrics
  const ruleHits = emg.rule_hits || {};
  const hits = Object.entries(ruleHits).filter(([_, v]) => v).map(([k]) => k);
  const emgGrid = `
    <div class="metric-grid">
      <div class="metric"><div class="label">Emerging threat</div>
        <div class="value">${emg.emerging_threat ? "Yes" : "No"}</div></div>
      <div class="metric"><div class="label">Novelty ratio</div>
        <div class="value">${(((emg.novelty_ratio ?? 0) * 100)).toFixed(1)}%</div></div>
      <div class="metric"><div class="label">Rule hits</div>
        <div class="value" style="font-size:13px;">${hits.length ? hits.join(", ") : "none"}</div></div>
    </div>`;

  // Dominant topic
  let domHtml = "";
  const dom = emg.dominant_topic || r.dominant_topic;
  if (dom) {
    const pills = (dom.top_words || []).map(w => `<span class="pill">${w}</span>`).join("");
    domHtml = `
      <div class="section">
        <div class="section-title">Dominant Topic</div>
        <div style="font-size:13px;color:#555;margin-bottom:6px;">
          Topic <b>#${dom.topic_id}</b> — appears in <b>${dom.count}</b> relevant sentences
          ${dom.share ? `(${(dom.share * 100).toFixed(0)}% of relevant)` : ""}
        </div>
        <div class="topic-pills">${pills}</div>
      </div>`;
  }

  // Topic distribution
  let tcHtml = "";
  const tc = emg.topic_counts || r.topic_counts || {};
  if (Object.keys(tc).length > 0) {
    const sorted = Object.entries(tc).sort((a, b) => b[1] - a[1]);
    const pills = sorted.map(([k, v]) =>
      `<span class="pill gray">T${k} <b>${v}</b></span>`
    ).join("");
    tcHtml = `
      <div class="section">
        <div class="section-title">Topic Distribution</div>
        <div class="topic-pills">${pills}</div>
      </div>`;
  }

  // Reasons
  let reasonsHtml = "";
  const reasons = emg.reasons || r.reasons || [];
  if (reasons.length > 0) {
    const items = reasons.map(x => `<li>${x}</li>`).join("");
    reasonsHtml = `
      <div class="section">
        <div class="section-title">Why This Verdict</div>
        <ul class="reasons-list">${items}</ul>
      </div>`;
  }

  verdictCard.innerHTML = `
    <div style="font-size:13px;color:#888;margin-bottom:14px;">
      Report: <b>${reportName}</b>
    </div>
    <div class="verdict-banner ${verdict}">
      <span class="icon">${banner.icon}</span>
      <span>${banner.label}</span>
    </div>
    ${attackHtml}
    <div class="section">
      <div class="section-title">Classification</div>
      ${clsGrid}
    </div>
    <div class="section">
      <div class="section-title">Emerging-Threat Detection</div>
      ${emgGrid}
    </div>
    ${domHtml}
    ${tcHtml}
    ${reasonsHtml}
    <div class="footer-note">Generated by CTI Threat Report Analyzer</div>`;

  loadingPanel.classList.add("hidden");
  resultsPanel.classList.remove("hidden");
}

// ── Batch test ──────────────────────────────────────────
document.getElementById("batch-btn").addEventListener("click", async () => {
  const btn = document.getElementById("batch-btn");
  const el = document.getElementById("batch-results");

  btn.disabled = true;
  btn.textContent = "Running batch test on 20 PDFs — this takes ~60 seconds...";
  el.classList.add("hidden");

  try {
    const resp = await fetch(`${API_BASE}/batch-analyze`, { method: "POST" });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      throw new Error(err.error || `HTTP ${resp.status}`);
    }
    const data = await resp.json();
    renderBatch(data);
    el.classList.remove("hidden");
  } catch (err) {
    el.innerHTML = `<div class="error">Batch failed: ${err.message}</div>`;
    el.classList.remove("hidden");
  } finally {
    btn.disabled = false;
    btn.textContent = "Run Batch Test on 20 PDFs";
  }
});

function renderBatch(data) {
  const el = document.getElementById("batch-results");

  // Summary cards
  const summary = `
    <div class="metric-grid" style="margin-bottom: 20px;">
      <div class="metric"><div class="label">Total PDFs</div>
        <div class="value">${data.total}</div></div>
      <div class="metric"><div class="label">Correct verdicts</div>
        <div class="value">${data.correct} / ${data.total}</div></div>
      <div class="metric"><div class="label">Accuracy</div>
        <div class="value">${(data.accuracy * 100).toFixed(1)}%</div></div>
      <div class="metric"><div class="label">Emerging flags</div>
        <div class="value">${data.emerging_flags}</div></div>
    </div>`;

  // Attack-type distribution
  let attackDist = "";
  if (data.attack_type_distribution && Object.keys(data.attack_type_distribution).length > 0) {
    const items = Object.entries(data.attack_type_distribution)
      .sort((a, b) => b[1] - a[1])
      .map(([type, count]) =>
        `<span class="pill gray">${type} <b>${count}</b></span>`
      ).join("");
    attackDist = `
      <div class="section" style="margin: 20px 0;">
        <div class="section-title">Attack-Type Distribution</div>
        <div class="topic-pills">${items}</div>
      </div>`;
  }

  // Per-report table
  const rows = data.results.map(r => {
    const matchIcon = r.match ? "✓" : "✗";
    const matchColor = r.match ? "#27ae60" : "#c0392b";
    const attackBadge = r.attack_type && r.attack_type !== "Unknown"
      ? `<span class="pill" style="font-size:10px;">${r.attack_type}</span>`
      : `<span style="color:#999;font-size:11px;">—</span>`;
    return `
      <tr>
        <td style="font-family: monospace; font-size: 11px;">${r.file}</td>
        <td>${r.expected}</td>
        <td><span class="badge ${r.predicted}">${r.predicted}</span></td>
        <td style="text-align:center;color:${matchColor};font-weight:700;">${matchIcon}</td>
        <td>${attackBadge}</td>
        <td style="text-align:center;font-size:11px;">${r.emerging_threat ? "⚠ YES" : "no"}</td>
        <td style="font-size:11px;color:#555;">${(r.reasons || []).slice(0, 1).join("; ") || "—"}</td>
      </tr>`;
  }).join("");

  const table = `
    <h3 style="font-size: 14px; margin: 20px 0 10px 0;
               color: var(--primary); text-transform: uppercase;">
      Per-Report Results
    </h3>
    <table>
      <thead>
        <tr>
          <th>File</th><th>Expected</th><th>Predicted</th>
          <th>Match</th><th>Attack Type</th><th>Emerging</th><th>Reason</th>
        </tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>`;

  // Emerging highlight
  const emerging = data.results.filter(r => r.emerging_threat);
  let emgBlock = "";
  if (emerging.length > 0) {
    emgBlock = `
      <h3 style="font-size: 14px; margin: 20px 0 10px 0;
                 color: var(--emerging); text-transform: uppercase;">
        ⚠ ${emerging.length} Emerging Threat(s) Detected
      </h3>
      ${emerging.map(r => `
        <div style="padding:12px; background:#fdecea; border-left:4px solid #c0392b;
                    border-radius:6px; margin-bottom:8px;">
          <div style="font-weight:700; color:#c0392b; margin-bottom:4px;">
            ⚠ ${r.file}  —  ${r.attack_type || "Unknown attack type"}
          </div>
          <div style="font-size:12px; color:#555;">
            Dominant topic: #${r.dominant_topic} ·
            Novelty: ${(r.novelty_ratio * 100).toFixed(1)}% ·
            Rules: ${Object.entries(r.rule_hits).filter(([_, v]) => v).map(([k]) => k).join(", ") || "none"}
          </div>
        </div>`).join("")}`;
  } else {
    emgBlock = `
      <div style="padding:16px; background:#eafaf1; border-left:4px solid #27ae60;
                  border-radius:6px; margin-top:20px; color:#27ae60; font-weight:600;">
        ✓ No emerging threats detected in any of the ${data.total} test PDFs.
      </div>`;
  }

  el.innerHTML = summary + attackDist + table + emgBlock;
}

// ── Helpers ─────────────────────────────────────────────
function showError(msg) {
  errorBox.textContent = msg;
  errorBox.classList.remove("hidden");
}
function hideError() {
  errorBox.classList.add("hidden");
}