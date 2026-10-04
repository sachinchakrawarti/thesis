"""
Real-time CTI stream simulation.

Simulates:
    1. Continuous ingestion of CTI reports from a stream
    2. Per-document preprocessing + classification
    3. Rolling window topic modeling
    4. Emerging Threat Score computed incrementally
    5. Live alerts when ETS exceeds threshold

Outputs:
    - output/reports/simulation_log.csv
    - output/reports/simulation_metrics.json
    - output/figures/sim_*.png
"""
import sys, time, json, random
from pathlib import Path
from datetime import datetime, timedelta
from collections import deque, defaultdict

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(r"D:\Projects\done\thesis\Implementation")
sys.path.insert(0, str(ROOT))

from src.config import (OUTPUT_DIR, MODELS_DIR, FIGURES_DIR, REPORTS_DIR,
                        TOPICS_DIR, RANDOM_STATE, ETS_WEIGHTS)
from src.preprocess import clean_text

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 100

# ============================================================
# Configuration
# ============================================================
STREAM_SIZE        = 500       # total documents to stream
WINDOW_SIZE        = 200       # rolling window for topic modeling
WINDOW_STEP        = 50        # recompute topics every N docs
ETS_THRESHOLD      = 0.55      # alert threshold for ETS
DELAY_PER_DOC      = 0.01      # seconds (simulated latency)
TIME_START         = datetime(2025, 1, 1)
TIME_STEP_MINUTES  = 30        # simulated arrival interval

RANDOM = random.Random(RANDOM_STATE)


# ============================================================
# 1. Load pre-trained artifacts
# ============================================================
print("=" * 65)
print("  Real-Time CTI Stream Simulation")
print("=" * 65)

print("\n[1] Loading artifacts...")
dataset   = pd.read_csv(OUTPUT_DIR / "cleaned_cti.csv")
text_col  = "clean_full" if "clean_full" in dataset.columns else "clean_text"
clf       = joblib.load(MODELS_DIR / "best_classifier.pkl") \
            if (MODELS_DIR / "best_classifier.pkl").exists() \
            else joblib.load(MODELS_DIR / "SVM.pkl")
tfidf     = joblib.load(MODELS_DIR / "tfidf_vectorizer.pkl")
nmf       = joblib.load(MODELS_DIR / "nmf_model.pkl")

N_TOPICS  = nmf.components_.shape[0]

print(f"  Dataset     : {dataset.shape}")
print(f"  Classifier  : {type(clf).__name__}")
print(f"  Topics (K)  : {N_TOPICS}")
print(f"  Stream size : {STREAM_SIZE}")
print(f"  Window size : {WINDOW_SIZE}")

# ============================================================
# 2. Build stream (chronological order)
# ============================================================
print("\n[2] Building stream...")
dataset["date"] = pd.to_datetime(dataset["date"])
stream_df = dataset.sort_values("date").head(STREAM_SIZE).reset_index(drop=True)
print(f"  Stream documents: {len(stream_df)}")

# ============================================================
# 3. Simulated stream processing
# ============================================================
print("\n[3] Simulating real-time processing...")

window = deque(maxlen=WINDOW_SIZE)
log_rows = []
ets_history = []          # (doc_index, timestamp, topic, ETS)
alerts = []               # triggered alerts
class_running = defaultdict(int)
correct = 0
total   = 0

stream_start = time.time()

for i, row in stream_df.iterrows():
    # ----- Simulated arrival time -----
    arrival = TIME_START + timedelta(minutes=TIME_STEP_MINUTES * i)

    # ----- Preprocess -----
    raw    = str(row[text_col])
    clean  = clean_text(raw)

    # ----- Classify -----
    X_doc  = tfidf.transform([clean])
    y_pred = clf.predict(X_doc)[0]
    class_running[y_pred] += 1

    if y_pred == row["category"]:
        correct += 1
    total += 1

    # ----- Add to rolling window -----
    window.append((row["id"], clean, arrival))

    # ----- Recompute topics every WINDOW_STEP docs -----
    doc_topics = None
    if (i + 1) % WINDOW_STEP == 0 and len(window) >= 50:
        docs = [w[1] for w in window]
        X    = tfidf.transform(docs)
        W    = nmf.transform(X)             # documents × topics
        doc_topics = W
        mean_topic = W.mean(axis=0)         # prevalence per topic
    else:
        mean_topic = None

    # ----- Compute per-document dominant topic -----
    if doc_topics is None:
        # Fallback: use NMF on current doc alone
        W_doc = nmf.transform(X_doc)
        dominant = int(np.argmax(W_doc[0]))
    else:
        dominant = int(np.argmax(doc_topics[-1]))

    # ----- Compute ETS on the fly (if topics refreshed) -----
    if mean_topic is not None:
        # Normalize
        m = mean_topic / (mean_topic.max() + 1e-9)

        # Growth approximated as deviation from previous mean
        if ets_history:
            prev_mean = ets_history[-1]["mean_topic"]
            growth = np.clip(m - prev_mean, 0, None)
        else:
            growth = np.zeros_like(m)

        persistence = (m > m.mean()).astype(float)
        novelty     = (m > 0.05).astype(float)

        ETS = (
            ETS_WEIGHTS["novelty"]     * novelty +
            ETS_WEIGHTS["growth"]      * growth +
            ETS_WEIGHTS["persistence"] * persistence +
            ETS_WEIGHTS["activity"]    * m +
            ETS_WEIGHTS["term_change"] * np.abs(growth)
        )

        ets_history.append({
            "doc_index":   i,
            "timestamp":   arrival,
            "ETS":         ETS,
            "mean_topic":  m,
        })

        # ----- Fire alerts for topics above threshold -----
        for t_idx, score in enumerate(ETS):
            if score >= ETS_THRESHOLD:
                alerts.append({
                    "doc_index":  i,
                    "timestamp":  arrival.strftime("%Y-%m-%d %H:%M"),
                    "topic":      f"topic_{t_idx+1}",
                    "ETS":        round(float(score), 4),
                    "predicted":  y_pred,
                })

    # ----- Log -----
    log_rows.append({
        "doc_index":      i + 1,
        "id":             row["id"],
        "arrival_time":   arrival.strftime("%Y-%m-%d %H:%M"),
        "true_label":     row["category"],
        "predicted":      y_pred,
        "correct":        int(y_pred == row["category"]),
        "dominant_topic": f"topic_{dominant+1}",
        "window_size":    len(window),
    })

    # ----- Simulated processing delay -----
    if DELAY_PER_DOC > 0:
        time.sleep(DELAY_PER_DOC)

stream_end = time.time()
elapsed    = stream_end - stream_start

print(f"  Processed {total} documents in {elapsed:.2f}s")
print(f"  Throughput: {total/elapsed:.1f} docs/sec")
print(f"  Accuracy  : {correct/total:.4f}")

# ============================================================
# 4. Save logs
# ============================================================
print("\n[4] Saving logs...")
log_df = pd.DataFrame(log_rows)
log_df.to_csv(REPORTS_DIR / "simulation_log.csv", index=False)
print(f"  -> {REPORTS_DIR / 'simulation_log.csv'}")

if alerts:
    alert_df = pd.DataFrame(alerts)
    alert_df.to_csv(REPORTS_DIR / "simulation_alerts.csv", index=False)
    print(f"  -> {REPORTS_DIR / 'simulation_alerts.csv'} ({len(alerts)} alerts)")
else:
    alert_df = pd.DataFrame()
    print("  No alerts fired (adjust ETS_THRESHOLD).")

# ============================================================
# 5. Metrics JSON
# ============================================================
metrics = {
    "stream_size":       total,
    "window_size":       WINDOW_SIZE,
    "window_step":       WINDOW_STEP,
    "elapsed_seconds":   round(elapsed, 2),
    "throughput_docs_s": round(total / elapsed, 2),
    "accuracy":          round(correct / total, 4),
    "alerts_fired":      len(alerts),
    "ets_threshold":     ETS_THRESHOLD,
    "n_topics":          int(N_TOPICS),
    "class_distribution": dict(class_running),
}
(REPORTS_DIR / "simulation_metrics.json").write_text(
    json.dumps(metrics, indent=2), encoding="utf-8"
)
print(f"  -> {REPORTS_DIR / 'simulation_metrics.json'}")

# ============================================================
# 6. Visualizations
# ============================================================
print("\n[5] Generating figures...")

# --- Fig 1: throughput over time ---
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(log_df["doc_index"], np.arange(1, len(log_df)+1) / elapsed,
        color="steelblue")
ax.set_title("Simulated Cumulative Throughput", fontweight="bold")
ax.set_xlabel("Document index")
ax.set_ylabel("Docs processed per second (cumulative avg)")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "sim_throughput.png")
plt.close()
print(f"  -> sim_throughput.png")

# --- Fig 2: running accuracy ---
fig, ax = plt.subplots(figsize=(11, 4))
running_acc = log_df["correct"].expanding().mean()
ax.plot(log_df["doc_index"], running_acc, color="seagreen")
ax.axhline(y=running_acc.iloc[-1], color="red", linestyle="--",
           label=f"Final = {running_acc.iloc[-1]:.4f}")
ax.set_title("Running Classification Accuracy Over Stream", fontweight="bold")
ax.set_xlabel("Document index")
ax.set_ylabel("Accuracy")
ax.legend()
plt.tight_layout()
plt.savefig(FIGURES_DIR / "sim_running_accuracy.png")
plt.close()
print(f"  -> sim_running_accuracy.png")

# --- Fig 3: predicted class distribution over time ---
fig, ax = plt.subplots(figsize=(11, 4))
pred_series = log_df.set_index("doc_index")["predicted"]
block = 50
for cat in pred_series.unique():
    counts = (pred_series == cat).astype(int).rolling(block).sum()
    ax.plot(log_df["doc_index"], counts, label=cat)
ax.set_title(f"Predicted Class Distribution (rolling {block})", fontweight="bold")
ax.set_xlabel("Document index")
ax.set_ylabel("Count in rolling window")
ax.legend(loc="upper right", fontsize=8)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "sim_class_distribution.png")
plt.close()
print(f"  -> sim_class_distribution.png")

# --- Fig 4: ETS over time ---
if ets_history:
    fig, ax = plt.subplots(figsize=(12, 5))
    steps = [h["doc_index"] for h in ets_history]
    for t in range(N_TOPICS):
        scores = [h["ETS"][t] for h in ets_history]
        ax.plot(steps, scores, label=f"topic_{t+1}", alpha=0.7)
    ax.axhline(y=ETS_THRESHOLD, color="red", linestyle="--",
               label=f"Alert threshold = {ETS_THRESHOLD}")
    ax.set_title("Emerging Threat Score Over Stream", fontweight="bold")
    ax.set_xlabel("Document index")
    ax.set_ylabel("ETS")
    ax.legend(loc="upper left", fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "sim_ets_over_time.png")
    plt.close()
    print(f"  -> sim_ets_over_time.png")

# --- Fig 5: alert timeline ---
if not alert_df.empty:
    fig, ax = plt.subplots(figsize=(11, 4))
    topic_order = sorted(alert_df["topic"].unique())
    topic_y = {t: i for i, t in enumerate(topic_order)}
    for _, r in alert_df.iterrows():
        ax.scatter(r["doc_index"], topic_y[r["topic"]],
                   s=40, alpha=0.6, color="crimson")
    ax.set_yticks(list(topic_y.values()))
    ax.set_yticklabels(topic_order)
    ax.set_title(f"Alert Timeline ({len(alert_df)} events)", fontweight="bold")
    ax.set_xlabel("Document index")
    ax.set_ylabel("Topic")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "sim_alert_timeline.png")
    plt.close()
    print(f"  -> sim_alert_timeline.png")

# ============================================================
# 7. Summary
# ============================================================
print("\n" + "=" * 65)
print("  SIMULATION COMPLETE")
print("=" * 65)
print(f"  Documents processed : {total}")
print(f"  Elapsed             : {elapsed:.2f}s")
print(f"  Throughput          : {total/elapsed:.1f} docs/sec")
print(f"  Accuracy            : {correct/total:.4f}")
print(f"  Alerts fired        : {len(alerts)}")
print(f"  ETS threshold       : {ETS_THRESHOLD}")
print()
print("  Saved files:")
print(f"    {REPORTS_DIR / 'simulation_log.csv'}")
print(f"    {REPORTS_DIR / 'simulation_alerts.csv'}")
print(f"    {REPORTS_DIR / 'simulation_metrics.json'}")
print(f"    {FIGURES_DIR}/sim_*.png")
print("=" * 65)