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
