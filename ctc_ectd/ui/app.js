(() => {
  const cfg = window.CTI_CONFIG;
  const $  = (s) => document.querySelector(s);

  const statusDot   = $("#statusDot");
  const statusText  = $("#statusText");
  const dropzone    = $("#dropzone");
  const fileInput   = $("#fileInput");
  const fileList    = $("#fileList");
  const analyzeBtn  = $("#analyzeBtn");
  const resetBtn    = $("#resetBtn");
  const demoBtn     = $("#demoBtn");
  const logEl       = $("#log");
  const backendUrl  = $("#backendUrl");

  let selectedFiles = [];
  let chartCats = null;
  let chartTopics = null;

  const log = (msg, cls = "") => {
    const t = new Date().toLocaleTimeString();
    logEl.innerHTML += `<span class="${cls}">[${t}] ${msg}</span>\n`;
    logEl.scrollTop = logEl.scrollHeight;
  };

  const setStatus = (kind, text) => {
    statusDot.className = "dot " + (kind === "ok" ? "ok" : kind === "err" ? "err" : "");
    statusText.textContent = text;
  };

  const fmt = (n, d = 3) =>
    (typeof n === "number" ? n.toFixed(d) : "\u2014");

  async function checkBackend() {
    try {
      const r = await fetch(cfg.API_BASE + cfg.ENDPOINTS.health);
      const j = await r.json();
      setStatus("ok", `Backend online \u00b7 ${j.model || "model loaded"}`);
      log("Backend health OK");
    } catch (e) {
      setStatus("err", "Backend offline");
      log("Backend not reachable. Start it with scripts\\run_backend.ps1", "err");
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
    analyzeBtn.textContent = "Analyzing\u2026";

    const form = new FormData();
    selectedFiles.forEach((f) => form.append("files", f, f.name));

    try {
      log(`Uploading ${selectedFiles.length} file(s)\u2026`);
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
    $("#kpiDocs").textContent   = o.docs   ?? "\u2014";
    $("#kpiCats").textContent   = o.cats   ?? "\u2014";
    $("#kpiAlerts").textContent = o.alerts ?? "\u2014";
    $("#kpiConf").textContent   = typeof o.conf === "number" ? o.conf.toFixed(3) : "\u2014";
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
        `<td>${d.word_count ?? "\u2014"}</td>` +
        `<td><span class="badge info">${d.predicted_category}</span></td>` +
        `<td><span class="badge ${badge}">${fmt(conf)}</span></td>` +
        `<td>${d.parsed_date ?? "\u2014"}</td>`;
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

  // ---------- demo corpus ----------
  demoBtn.addEventListener("click", async () => {
    demoBtn.disabled = true;
    demoBtn.textContent = "Loading demo\u2026";
    log("POST /api/analyze-demo \u2014 running pipeline on uploads/");

    try {
      const r = await fetch(cfg.API_BASE + cfg.ENDPOINTS.analyzeDemo, {
        method: "POST"
      });
      if (!r.ok) {
        const err = await r.json().catch(() => ({}));
        throw new Error(err.error || `HTTP ${r.status}`);
      }
      const data = await r.json();
      log(`Demo done: ${data.documents.length} docs, ${data.alerts.length} alerts.`);
      handleResults(data);
    } catch (e) {
      log("Demo failed: " + e.message, "err");
    } finally {
      demoBtn.disabled = false;
      demoBtn.textContent = "Load demo corpus";
    }
  });

  refreshFileList();
  checkBackend();
})();
