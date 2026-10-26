# D:\Projects\done\thesis\cyber-threat-detection\scripts\make_ui.py
"""Generate a professional, dark-mode SOC dashboard UI."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
UI = PROJECT_ROOT / "ui"
(UI / "css").mkdir(parents=True, exist_ok=True)
(UI / "js").mkdir(parents=True, exist_ok=True)

def write(path: Path, text: str):
    path.write_text(text, encoding="utf-8")
    print("wrote", path.relative_to(PROJECT_ROOT))

# ---------------- index.html ----------------
write(UI / "index.html", """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>CTI Analyser</title>
  <link rel="stylesheet" href="/ui/css/style.css" />
</head>
<body>
  <div class="app-container">
    
    <!-- Header -->
    <header class="app-header">
      <div class="logo">
        <span class="icon">🛡️</span> 
        <h1>CTI ANALYSER</h1>
      </div>
      <div class="header-right">
        <span class="badge">Local Demo</span>
        <span class="status-dot"></span> Online
      </div>
    </header>

    <main class="app-main">
      
      <!-- Upload Section -->
      <section class="card">
        <h2 class="card-title">Upload Threat Intelligence Report</h2>
        <div class="upload-zone">
          <input type="file" id="pdfInput" accept="application/pdf" />
          <button id="analyseBtn">Analyse PDF</button>
          <span id="status" class="status-text"></span>
        </div>
      </section>

      <div id="results" class="hidden">
        
        <!-- Metrics Row -->
        <div class="metrics-grid">
          <div class="metric-card">
            <span class="metric-label">Total Sentences</span>
            <span class="metric-value" id="nSentences">-</span>
          </div>
          <div class="metric-card highlight-green">
            <span class="metric-label">Relevant CTI</span>
            <span class="metric-value" id="nRelevant">-</span>
          </div>
          <div class="metric-card highlight-red">
            <span class="metric-label">Irrelevant</span>
            <span class="metric-value" id="nIrrelevant">-</span>
          </div>
          <div class="metric-card highlight-blue">
            <span class="metric-label">Top Technique</span>
            <span class="metric-value text-sm" id="topTech">-</span>
          </div>
        </div>

        <!-- Model Performance -->
        <section class="card">
          <h2 class="card-title">Model Performance</h2>
          <table class="data-table">
            <thead>
              <tr><th>Model</th><th>Accuracy</th><th>Macro-F1</th></tr>
            </thead>
            <tbody id="perfBody">
              <tr><td colspan="3" class="text-muted">Loading performance data...</td></tr>
            </tbody>
          </table>
        </section>

        <div class="grid-2col">
          <!-- Technique Chart -->
          <section class="card">
            <h2 class="card-title">Technique Distribution</h2>
            <div id="techChart" class="chart-container"></div>
          </section>

          <!-- Relevant Sentences -->
          <section class="card">
            <h2 class="card-title">Representative Sentences</h2>
            <ul id="sentList" class="sentence-list"></ul>
          </section>
        </div>

        <!-- Emerging Threats -->
        <section class="card threat-card">
          <h2 class="card-title text-red">🚨 Emerging Threat Monitor</h2>
          <div id="emergingBox" class="alert-box">
            Processing temporal trends...
          </div>
          <div id="topicChart" class="chart-container"></div>
        </section>

      </div>
    </main>

    <footer class="app-footer">
      <p>Prototype Notice: Findings are rule-based temporal signals, not confirmed incidents.</p>
    </footer>
  </div>

  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <script src="/ui/js/app.js"></script>
</body>
</html>
""")

# ---------------- style.css ----------------
write(UI / "css" / "style.css", """\
:root {
  --bg-dark: #0f172a;
  --bg-card: #1e293b;
  --text-main: #f8fafc;
  --text-muted: #94a3b8;
  --border: #334155;
  --accent: #3b82f6;
  --accent-hover: #2563eb;
  --success: #10b981;
  --danger: #ef4444;
  --warning: #f59e0b;
}

* { box-sizing: border-box; }

body {
  font-family: 'Inter', system-ui, -apple-system, sans-serif;
  background-color: var(--bg-dark);
  color: var(--text-main);
  margin: 0;
  padding: 0;
  line-height: 1.5;
}

.app-container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 20px;
}

/* Header */
.app-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 1rem 0;
  border-bottom: 1px solid var(--border);
  margin-bottom: 2rem;
}
.logo { display: flex; align-items: center; gap: 10px; }
.logo h1 { font-size: 1.25rem; margin: 0; letter-spacing: 1px; font-weight: 600; }
.header-right { display: flex; align-items: center; gap: 15px; font-size: 0.9rem; }
.badge { background: var(--border); padding: 4px 10px; border-radius: 12px; font-size: 0.75rem; font-weight: 600; }
.status-dot { width: 8px; height: 8px; background: var(--success); border-radius: 50%; display: inline-block; }

/* Cards */
.card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 1.5rem;
  margin-bottom: 1.5rem;
}
.card-title {
  font-size: 1rem;
  font-weight: 600;
  margin-top: 0;
  margin-bottom: 1rem;
  color: var(--text-main);
  border-bottom: 1px solid var(--border);
  padding-bottom: 0.5rem;
}

/* Upload */
.upload-zone { display: flex; align-items: center; gap: 1rem; }
button {
  background: var(--accent); color: white; border: none; 
  padding: 10px 20px; border-radius: 6px; font-weight: 600; cursor: pointer;
  transition: background 0.2s;
}
button:hover { background: var(--accent-hover); }
button:disabled { opacity: 0.5; cursor: not-allowed; }
.status-text { color: var(--accent); font-size: 0.9rem; font-style: italic; }

/* Metrics Grid */
.metrics-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 1rem;
  margin-bottom: 1.5rem;
}
.metric-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 1rem;
  display: flex;
  flex-direction: column;
  justify-content: center;
}
.metric-label { font-size: 0.8rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 5px; }
.metric-value { font-size: 1.8rem; font-weight: 700; }
.text-sm { font-size: 1.1rem; }
.highlight-green .metric-value { color: var(--success); }
.highlight-red .metric-value { color: var(--danger); }
.highlight-blue .metric-value { color: var(--accent); }

/* Layout */
.grid-2col { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; }

/* Tables */
.data-table { width: 100%; border-collapse: collapse; }
.data-table th, .data-table td { padding: 10px; text-align: left; border-bottom: 1px solid var(--border); font-size: 0.9rem; }
.data-table th { color: var(--text-muted); font-weight: 500; }
.data-table tr:last-child td { border-bottom: none; }

/* Lists */
.sentence-list { list-style: none; padding: 0; margin: 0; }
.sentence-list li { padding: 10px; border-left: 3px solid var(--accent); background: rgba(59, 130, 246, 0.1); margin-bottom: 10px; font-size: 0.9rem; }

/* Charts */
.chart-container { min-height: 250px; width: 100%; }

/* Alerts */
.threat-card { border-color: var(--danger); }
.text-red { color: var(--danger) !important; }
.alert-box {
  padding: 1rem;
  background: rgba(239, 68, 68, 0.1);
  border: 1px solid var(--danger);
  border-radius: 6px;
  color: #fca5a5;
  margin-bottom: 1rem;
  font-weight: 500;
}
.alert-box strong { color: var(--danger); display: block; margin-bottom: 5px; }

/* Footer */
.app-footer { text-align: center; color: var(--text-muted); font-size: 0.8rem; margin-top: 2rem; padding-top: 1rem; border-top: 1px solid var(--border); }
.hidden { display: none; }
.text-muted { color: var(--text-muted); }
""")

# ---------------- app.js ----------------
write(UI / "js" / "app.js", """\
const input = document.getElementById("pdfInput");
const btn = document.getElementById("analyseBtn");
const status = document.getElementById("status");
const results = document.getElementById("results");

// Dark theme for Plotly
const plotlyLayout = {
  paper_bgcolor: 'rgba(0,0,0,0)',
  plot_bgcolor: 'rgba(0,0,0,0)',
  font: { color: '#94a3b8' },
  margin: { l: 150, r: 20, t: 20, b: 40 },
  height: 260
};

document.addEventListener("DOMContentLoaded", async () => {
  try {
    const res = await fetch("/performance");
    const data = await res.json();
    const tbody = document.getElementById("perfBody");
    tbody.innerHTML = "";
    if (!data.length) throw new Error();
    
    data.forEach(row => {
      tbody.innerHTML += `
        <tr>
          <td><strong>${row.model}</strong> <span class="text-muted">(${row.classifier})</span></td>
          <td>${(row.accuracy * 100).toFixed(1)}%</td>
          <td>${(row.macro_f1 * 100).toFixed(1)}%</td>
        </tr>`;
    });
  } catch (e) {
    document.getElementById("perfBody").innerHTML = 
      `<tr><td colspan="3" class="text-muted">Run the notebook to generate performance metrics.</td></tr>`;
  }
});

btn.addEventListener("click", async () => {
  const file = input.files[0];
  if (!file) { status.textContent = "Please choose a PDF."; return; }
  
  btn.disabled = true;
  status.textContent = "Analysing...";
  results.classList.add("hidden");

  try {
    const form = new FormData();
    form.append("file", file);
    const res = await fetch("/analyse", { method: "POST", body: form });
    
    if (!res.ok) throw new Error("Server error");
    const data = await res.json();
    renderResults(data);
    
    status.textContent = "Analysis Complete.";
    results.classList.remove("hidden");
  } catch (e) {
    status.textContent = "Error: " + e.message;
  } finally {
    btn.disabled = false;
  }
});

function renderResults(data) {
  document.getElementById("nSentences").textContent = data.n_sentences;
  document.getElementById("nRelevant").textContent = data.n_relevant;
  document.getElementById("nIrrelevant").textContent = data.n_irrelevant;
  document.getElementById("topTech").textContent = data.top_technique || "None";

  // Technique Chart
  const labels = Object.keys(data.technique_counts || {});
  const values = labels.map(k => data.technique_counts[k]);
  if (labels.length) {
    Plotly.newPlot("techChart", [{
      type: "bar", orientation: "h", x: values, y: labels,
      marker: { color: "#3b82f6" }
    }], plotlyLayout, {displayModeBar: false});
  }

  // Sentences
  const list = document.getElementById("sentList");
  list.innerHTML = "";
  (data.top_relevant_sentences || []).forEach(s => {
    list.innerHTML += `<li>${s}</li>`;
  });

  // Emerging Threats
  const alertBox = document.getElementById("emergingBox");
  const topicChart = document.getElementById("topicChart");
  
  if (data.topics && data.topics.top_topics) {
    alertBox.innerHTML = `
      <strong>⚠️ POTENTIALLY EMERGING: ${data.top_technique || "Threat"} Activity</strong>
      The temporal rule engine detected a rising trend in this technique cluster.
    `;
    const topics = data.topics.top_topics;
    Plotly.newPlot(topicChart, [{
      type: "bar", x: topics.map(t => `T${t.Topic}`), y: topics.map(t => t.Count),
      text: topics.map(t => t.Name.replace(/_/g, " ")), marker: { color: "#ef4444" }
    }], plotlyLayout, {displayModeBar: false});
  } else {
    alertBox.innerHTML = `<strong>No Emerging Threats Detected</strong>`;
    topicChart.innerHTML = "";
  }
}
""")

print("\nDark Mode UI generated. Run the server and refresh the browser.")