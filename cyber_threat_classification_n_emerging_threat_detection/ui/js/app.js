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
