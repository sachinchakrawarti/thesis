"""
Real-Time CTI Monitoring Dashboard
Live simulation of streaming CTI reports with classification and alerts.

Run:
    streamlit run dashboard.py
"""
import sys, time, random
from pathlib import Path
from datetime import datetime, timedelta
from collections import deque, defaultdict

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.config import (OUTPUT_DIR, MODELS_DIR, RANDOM_STATE,
                        ETS_WEIGHTS)
from src.preprocess import clean_text

# ============================================================
# Page config
# ============================================================
st.set_page_config(
    page_title="CTI Real-Time Monitor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CSS — dark SOC theme
# ============================================================
st.markdown("""
<style>
:root {
    --bg-dark:    #0d1117;
    --bg-panel:   #161b22;
    --border:     #30363d;
    --text:       #c9d1d9;
    --accent:     #58a6ff;
    --ok:         #3fb950;
    --warn:       #d29922;
    --danger:     #f85149;
}
.stApp { background: var(--bg-dark); color: var(--text); }
h1, h2, h3 { color: var(--accent); }
[data-testid="stMetricValue"] { color: var(--accent); font-size: 1.6em; }
[data-testid="stMetricLabel"] { color: #8b949e; }
.metric-card {
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px;
    margin-bottom: 8px;
}
.alert-box {
    background: rgba(248, 81, 73, 0.15);
    border-left: 4px solid var(--danger);
    padding: 8px 12px;
    margin: 4px 0;
    border-radius: 4px;
    font-family: 'Consolas', monospace;
    font-size: 0.85em;
}
.ok-box {
    background: rgba(63, 185, 80, 0.15);
    border-left: 4px solid var(--ok);
    padding: 8px 12px;
    margin: 4px 0;
    border-radius: 4px;
}
.doc-row {
    background: var(--bg-panel);
    border-left: 3px solid var(--accent);
    padding: 6px 10px;
    margin: 3px 0;
    border-radius: 4px;
    font-family: 'Consolas', monospace;
    font-size: 0.82em;
    color: #b0b0b0;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# Load artifacts (cached) — WITH FIX
# ============================================================
@st.cache_resource
def load_artifacts():
    cleaned   = pd.read_csv(OUTPUT_DIR / "cleaned_cti.csv")
    text_col  = "clean_full" if "clean_full" in cleaned.columns else "clean_text"

    clf_path = MODELS_DIR / "best_classifier.pkl"
    if not clf_path.exists():
        clf_path = MODELS_DIR / "SVM.pkl"
    clf = joblib.load(clf_path)

    # Classification vectorizer
    tfidf = joblib.load(MODELS_DIR / "tfidf_vectorizer.pkl")

    # Topic vectorizer — NMF was trained with this one
    tfidf_top_path = MODELS_DIR / "tfidf_topic_vectorizer.pkl"
    if tfidf_top_path.exists():
        tfidf_top = joblib.load(tfidf_top_path)
    else:
        tfidf_top = tfidf

    nmf = joblib.load(MODELS_DIR / "nmf_model.pkl")

    return cleaned, text_col, clf, tfidf, tfidf_top, nmf


cleaned, TEXT_COL, clf, tfidf, tfidf_top, nmf = load_artifacts()
N_TOPICS = nmf.components_.shape[0]

TOPIC_TERMS = {}
try:
    nmf_rows = pd.read_csv(OUTPUT_DIR / "topics" / "nmf_topics.csv")
    for _, r in nmf_rows.iterrows():
        t = r["topic"].replace("NMF_", "topic_")
        TOPIC_TERMS[t] = r["top_terms"]
except Exception:
    pass


# ============================================================
# Sidebar controls
# ============================================================
with st.sidebar:
    st.markdown("### 🎛️ Simulation Controls")
    stream_size    = st.slider("Stream size (docs)", 50, 2000, 300, 50)
    delay_ms       = st.slider("Inter-doc delay (ms)", 0, 500, 80, 10)
    window_size    = st.slider("Rolling window", 50, 500, 150, 25)
    window_step    = st.slider("Topic refresh every N docs", 10, 100, 30, 5)
    ets_threshold  = st.slider("ETS alert threshold", 0.2, 0.9, 0.5, 0.05)

    st.divider()
    st.markdown("### 📊 Model Info")
    st.write(f"**Classifier:** `{type(clf).__name__}`")
    st.write(f"**Topics (K):** `{N_TOPICS}`")
    st.write(f"**Corpus size:** `{len(cleaned):,}`")

    st.divider()
    start_btn = st.button("▶️  START STREAM", use_container_width=True, type="primary")
    stop_btn  = st.button("⏹️  STOP", use_container_width=True)


# ============================================================
# Layout — header + metric strip
# ============================================================
st.markdown("# 🛡️ Real-Time CTI Threat Monitor")
st.markdown(
    "Live simulation of streaming cyber threat intelligence: "
    "classification, topic inference, and emerging-threat alerts."
)

c1, c2, c3, c4, c5 = st.columns(5)
m_docs    = c1.empty()
m_acc     = c2.empty()
m_rate    = c3.empty()
m_alerts  = c4.empty()
m_top_ets = c5.empty()

m_docs.metric("Documents",       "0")
m_acc.metric("Running Accuracy", "—")
m_rate.metric("Throughput",      "0 /s")
m_alerts.metric("Alerts Fired",  "0")
m_top_ets.metric("Top ETS",      "—")


# ============================================================
# Tabs
# ============================================================
tab_live, tab_class, tab_ets, tab_alerts = st.tabs([
    "🔴 Live Stream",
    "📊 Classification",
    "📈 Emerging Threats",
    "🚨 Alerts",
])

with tab_live:
    st.markdown("### Incoming Documents")
    live_log = st.empty()

with tab_class:
    st.markdown("### Predicted Class Distribution Over Time")
    class_plot = st.empty()
    st.markdown("### Running Accuracy")
    acc_plot = st.empty()

with tab_ets:
    st.markdown("### Emerging Threat Score (Live)")
    ets_plot = st.empty()
    st.markdown("### Top Topic Terms (Current)")
    topic_info = st.empty()

with tab_alerts:
    st.markdown("### Alert Feed")
    alert_box = st.empty()


# ============================================================
# Simulation — WITH FIX
# ============================================================
def run_simulation():
    df = cleaned.sample(frac=1, random_state=RANDOM_STATE).head(stream_size)
    df = df.sort_values("date").reset_index(drop=True)

    window          = deque(maxlen=window_size)
    class_counter   = defaultdict(int)
    correct, total  = 0, 0
    alerts          = []
    ets_matrix      = []
    topic_prev      = None
    start_time      = time.time()

    placeholder_lines = []

    for i, row in df.iterrows():
        if stop_btn:
            st.warning("Stopped by user.")
            break

        raw   = str(row[TEXT_COL])
        clean = clean_text(raw)

        # ---- Classification path ----
        X_doc  = tfidf.transform([clean])
        y_pred = clf.predict(X_doc)[0]
        class_counter[y_pred] += 1
        if y_pred == row["category"]:
            correct += 1
        total += 1

        arrival = datetime.now().strftime("%H:%M:%S")
        window.append(clean)

        # ---- Topic refresh path (FIXED — uses tfidf_top) ----
        X_doc_top = tfidf_top.transform([clean])

        if (i + 1) % window_step == 0 and len(window) >= 30:
            X_win_top  = tfidf_top.transform(list(window))
            W_win      = nmf.transform(X_win_top)
            mean_topic = W_win.mean(axis=0)
        else:
            mean_topic = None

        W_doc    = nmf.transform(X_doc_top)
        dominant = int(np.argmax(W_doc[0]))
        dom_label = f"T{dominant+1}"

        # ---- ETS ----
        if mean_topic is not None:
            m = mean_topic / (mean_topic.max() + 1e-9)
            if topic_prev is None:
                growth = np.zeros_like(m)
            else:
                growth = np.clip(m - topic_prev, 0, None)
            persistence = (m > m.mean()).astype(float)
            novelty     = (m > 0.05).astype(float)

            ETS = (
                ETS_WEIGHTS["novelty"]     * novelty +
                ETS_WEIGHTS["growth"]      * growth +
                ETS_WEIGHTS["persistence"] * persistence +
                ETS_WEIGHTS["activity"]    * m +
                ETS_WEIGHTS["term_change"] * np.abs(growth)
            )
            ets_matrix.append(ETS)
            topic_prev = m

            for t_idx, score in enumerate(ETS):
                if score >= ets_threshold:
                    alerts.append({
                        "time":      arrival,
                        "doc_index": i + 1,
                        "topic":     f"T{t_idx+1}",
                        "terms":     TOPIC_TERMS.get(f"topic_{t_idx+1}", ""),
                        "ets":       round(float(score), 4),
                        "predicted": y_pred,
                    })

        mark = "✓" if y_pred == row["category"] else "✗"
        placeholder_lines.append(
            f"`{arrival}` `{row['id']:<12}` "
            f"`{row['category']:<12}` `{y_pred:<12}` "
            f"`{dom_label:<4}` `{mark}`"
        )
        placeholder_lines = placeholder_lines[-25:]

        # ---- Throttled UI update ----
        if i % 3 == 0 or i == len(df) - 1:
            live_log.markdown(
                "".join([
                    '<div class="doc-row">' + ln + "</div>"
                    for ln in placeholder_lines
                ]),
                unsafe_allow_html=True,
            )

            running_acc = correct / total
            elapsed     = max(time.time() - start_time, 1e-6)
            rate        = total / elapsed

            m_docs.metric("Documents", f"{total}")
            m_acc.metric("Running Accuracy", f"{running_acc:.3f}")
            m_rate.metric("Throughput", f"{rate:.1f} /s")
            m_alerts.metric("Alerts Fired", f"{len(alerts)}")

            if ets_matrix:
                top_ets = float(np.max(ets_matrix[-1]))
                m_top_ets.metric("Top ETS", f"{top_ets:.3f}")

            # ---- Class distribution chart ----
            if len(class_counter) > 0:
                cls_df = pd.DataFrame(
                    sorted(class_counter.items()),
                    columns=["Category", "Count"],
                )
                fig = px.bar(cls_df, x="Category", y="Count",
                             color="Category",
                             color_discrete_sequence=px.colors.qualitative.Set2,
                             title=f"Predicted Classes (n={total})")
                fig.update_layout(showlegend=False,
                                  paper_bgcolor="#0d1117",
                                  plot_bgcolor="#0d1117",
                                  font_color="#c9d1d9",
                                  height=350)
                class_plot.plotly_chart(fig, use_container_width=True, key=f"cls_{total}")

            # ---- Running accuracy chart ----
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                y=[correct/t for t in range(1, total+1)],
                mode="lines",
                line=dict(color="#3fb950", width=2),
                name="Accuracy",
            ))
            fig2.update_layout(
                title="Running Classification Accuracy",
                xaxis_title="Document index",
                yaxis_title="Accuracy",
                yaxis=dict(range=[0, 1.05]),
                paper_bgcolor="#0d1117",
                plot_bgcolor="#0d1117",
                font_color="#c9d1d9",
                height=300,
                showlegend=False,
            )
            acc_plot.plotly_chart(fig2, use_container_width=True, key=f"acc_{total}")

            # ---- ETS chart ----
            if ets_matrix:
                ets_arr = np.array(ets_matrix)
                fig3 = go.Figure()
                for t in range(ets_arr.shape[1]):
                    fig3.add_trace(go.Scatter(
                        y=ets_arr[:, t],
                        mode="lines",
                        name=f"T{t+1}",
                        line=dict(width=1.5),
                    ))
                fig3.add_hline(y=ets_threshold, line_dash="dash",
                               line_color="#f85149",
                               annotation_text=f"alert ≥ {ets_threshold}")
                fig3.update_layout(
                    title="Emerging Threat Score per Topic",
                    xaxis_title="Refresh step",
                    yaxis_title="ETS",
                    paper_bgcolor="#0d1117",
                    plot_bgcolor="#0d1117",
                    font_color="#c9d1d9",
                    height=380,
                    legend=dict(orientation="h", y=-0.2),
                )
                ets_plot.plotly_chart(fig3, use_container_width=True, key=f"ets_{total}")

            # ---- Alert feed ----
            if alerts:
                alert_html = []
                for a in alerts[-15:][::-1]:
                    alert_html.append(
                        f'<div class="alert-box">'
                        f'🚨 <b>[{a["time"]}] {a["topic"]}</b> '
                        f'ETS={a["ets"]:.3f} · predicted={a["predicted"]}<br>'
                        f'<span style="color:#8b949e">terms: {a["terms"]}</span>'
                        f'</div>'
                    )
                alert_box.markdown("".join(alert_html),
                                   unsafe_allow_html=True)
            else:
                alert_box.markdown(
                    '<div class="ok-box">✅ No alerts yet.</div>',
                    unsafe_allow_html=True,
                )

        # ---- Simulated delay ----
        if delay_ms > 0:
            time.sleep(delay_ms / 1000.0)

    st.success(
        f"✅ Stream complete. "
        f"Docs={total}, Accuracy={correct/total:.4f}, Alerts={len(alerts)}"
    )


# ============================================================
# Trigger
# ============================================================
if start_btn:
    run_simulation()
else:
    st.info("👈 Configure parameters and press **START STREAM**.")