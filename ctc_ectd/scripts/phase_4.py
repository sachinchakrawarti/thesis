# scripts/phase_4.py
"""
PHASE 4 -- Emerging-Pattern Detection & Evaluation
====================================================

Produces:

    artifacts/final/ground_truth_alerts.csv        synthetic ground truth
    artifacts/final/alert_precision_recall.csv     metrics at chosen threshold
    artifacts/final/threshold_sweep.csv            metrics across thresholds
    artifacts/final/threshold_sweep.png            precision/recall vs threshold
    artifacts/final/alert_confusion.png            TP/FP/FN visual
    artifacts/final/alerts_at_best_threshold.csv   final alert list
    artifacts/final/phase_4_summary.txt

Method:
    1. Build ground truth from the known wave schedule:
         - Phishing wave:   2025-09 -> 2025-11
         - Ransomware wave: 2026-01 -> 2026-03
         - (extendable list below)
    2. Fit NMF on cleaned corpus, compute monthly topic shares
    3. Define alert rule: share rises >= threshold AND count >= min_count
    4. Sweep threshold from 0.02 to 0.30 in steps, compute P/R/F1
    5. Pick the F1-maximizing threshold
    6. Report final alerts with TP/FP tags
    7. Save summary + plots

Run:
    python scripts/phase_4.py
    python scripts/phase_4.py --dry-run
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET      = PROJECT_ROOT / "dataset" / "synthetic_cti_reports.csv"
FINAL_DIR    = PROJECT_ROOT / "artifacts" / "final"

RANDOM_STATE = 42
N_TOPICS     = 5

# ------------------------------------------------------------------
# Ground truth -- encoded from known synthetic wave schedule.
# Each entry is a month (YYYY-MM) where a topic-spike SHOULD be flagged.
# We match by (current_month, dominant_keyword) not by topic index,
# because NMF topic numbers are not stable.
# ------------------------------------------------------------------
GROUND_TRUTH_WAVES = [
    # Phishing wave
    {"month": "2025-09", "keyword": "phishing"},
    {"month": "2025-10", "keyword": "phishing"},
    {"month": "2025-11", "keyword": "phishing"},
    # Ransomware wave
    {"month": "2026-01", "keyword": "ransomware"},
    {"month": "2026-02", "keyword": "ransomware"},
    {"month": "2026-03", "keyword": "ransomware"},
]

# Threshold sweep range
THRESHOLDS = [round(x, 2) for x in np.arange(0.02, 0.32, 0.02)]

# Minimum reports required in the current month for an alert to fire
MIN_COUNT = 2


# ==================================================================
# Console
# ==================================================================
class Rep:
    def __init__(self, dry: bool = False):
        self.dry = dry

    def banner(self, text: str) -> None:
        print()
        print("=" * 66)
        print(text)
        print("=" * 66)

    def ok(self, msg: str) -> None:
        print(f"  +  {msg}")

    def info(self, msg: str) -> None:
        print(f"     {msg}")

    def warn(self, msg: str) -> None:
        print(f"  !  {msg}")


# ==================================================================
# Step 1 -- Load + clean (identical to Phase 3)
# ==================================================================
def load_and_clean(rep: Rep) -> pd.DataFrame:
    rep.banner("[1/6] Loading + cleaning dataset")

    if not DATASET.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET}")

    df = pd.read_csv(DATASET)
    df["category"] = df["category"].astype(str).str.strip()
    df["title"]    = df["title"].fillna("").astype(str)
    df["text"]     = df["text"].fillna("").astype(str)
    df["date"]     = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "category"]).reset_index(drop=True)

    df["model_text_raw"] = (
        df["title"].str.strip() + " " + df["text"].str.strip()
    ).str.strip()

    cats = df["category"].unique()
    def strip_labels(text: str) -> str:
        for c in cats:
            text = re.sub(re.escape(c), " ", text, flags=re.IGNORECASE)
        return text

    df["model_text_clean"] = df["model_text_raw"].apply(strip_labels)

    for phrase in [
        r"A recent threat campaign involved",
        r"A medium campaign targeted organizations",
        r"A high campaign targeted organizations",
        r"Security analysts observed",
        r"Analysts observed",
        r"Initial access was linked to",
        r"The activity was associated with",
        r"The investigation reported MITRE ATT&CK technique",
        r"The observed vulnerability identifier was",
        r"Additional indicators included",
        r"and was classified as",
    ]:
        df["model_text_clean"] = df["model_text_clean"].str.replace(
            phrase, " ", case=False, regex=True)

    for pattern in [
        r"CVE-\d{4}-\d+",
        r"T\d{4}(?:\.\d{3})?",
        r"(?:IP|DOMAIN|SHA256|MD5|URL)=\S+",
        r"\b\d+\b",
    ]:
        df["model_text_clean"] = df["model_text_clean"].str.replace(
            pattern, " ", regex=True)

    df["model_text_clean"] = df["model_text_clean"].str.replace(
        r"\s+", " ", regex=True).str.strip()

    rep.ok(f"loaded {len(df)} rows, {df['category'].nunique()} categories")
    return df


# ==================================================================
# Step 2 -- Fit NMF + monthly shares
# ==================================================================
def fit_and_share(df: pd.DataFrame, rep: Rep):
    rep.banner("[2/6] Fitting NMF, computing monthly topic shares")

    vec = TfidfVectorizer(
        max_features=5000,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-]+\b",
    )
    X = vec.fit_transform(df["model_text_clean"])

    nmf = NMF(
        n_components=N_TOPICS,
        random_state=RANDOM_STATE,
        init="nndsvda",
        max_iter=500,
    )
    W = nmf.fit_transform(X)
    W_norm = W / (W.sum(axis=1, keepdims=True) + 1e-12)

    feats = vec.get_feature_names_out()

    # Human-readable label per topic: top keyword
    topic_keywords = {}
    topic_top_words = {}
    for i in range(N_TOPICS):
        top_idx = nmf.components_[i].argsort()[::-1][:10]
        words = [feats[j] for j in top_idx]
        topic_keywords[i + 1] = words[0]
        topic_top_words[i + 1] = words

    # Monthly shares
    df = df.copy()
    df["month"] = df["date"].dt.to_period("M").astype(str)
    for i in range(N_TOPICS):
        df[f"T{i+1}"] = W_norm[:, i]
    df["dominant_topic"] = W_norm.argmax(axis=1) + 1

    monthly_counts = pd.crosstab(df["month"], df["dominant_topic"])
    monthly_shares = monthly_counts.div(
        monthly_counts.sum(axis=1), axis=0).fillna(0)

    print()
    print("Topic keywords (top word per topic):")
    for tid, kw in topic_keywords.items():
        print(f"  T{tid}: {kw}   (top words: {', '.join(topic_top_words[tid][:5])})")

    rep.ok(f"{len(monthly_shares)} months x {N_TOPICS} topics")

    return {
        "nmf": nmf,
        "vectorizer": vec,
        "W_norm": W_norm,
        "df": df,
        "monthly_counts": monthly_counts,
        "monthly_shares": monthly_shares,
        "topic_keywords": topic_keywords,
        "topic_top_words": topic_top_words,
    }


# ==================================================================
# Step 3 -- Ground truth construction
# ==================================================================
def build_ground_truth(models, rep: Rep) -> pd.DataFrame:
    rep.banner("[3/6] Building ground-truth alert set")

    monthly_shares = models["monthly_shares"]
    monthly_counts = models["monthly_counts"]
    topic_keywords = models["topic_keywords"]

    # Map keyword -> topic id
    keyword_to_topic = {}
    for tid, kw in topic_keywords.items():
        keyword_to_topic[kw] = tid

    rows = []
    for wave in GROUND_TRUTH_WAVES:
        month = wave["month"]
        kw = wave["keyword"]

        # Find the topic whose top keyword best matches this wave's keyword
        # by substring in the top-10 words (tolerant matching)
        topic_id = None
        for tid, words in models["topic_top_words"].items():
            if any(kw in w for w in words):
                topic_id = tid
                break

        if topic_id is None:
            rep.warn(f"no topic matches keyword '{kw}' for {month}")
            continue

        if month not in monthly_shares.index:
            rep.warn(f"month {month} not in corpus")
            continue

        share = float(monthly_shares.loc[month, topic_id])
        count = int(monthly_counts.loc[month, topic_id])

        rows.append({
            "month": month,
            "topic_id": topic_id,
            "topic_keyword": topic_keywords[topic_id],
            "expected_share": round(share, 3),
            "expected_count": count,
        })

    gt = pd.DataFrame(rows)
    print()
    print(gt.to_string(index=False))
    rep.ok(f"{len(gt)} expected alert events in ground truth")
    return gt


# ==================================================================
# Step 4 -- Alert rule + threshold sweep
# ==================================================================
def evaluate_alerts(models, gt: pd.DataFrame, threshold: float,
                    min_count: int) -> dict:
    """
    Run the alert rule at a given threshold.
    Returns dict of TPs, FPs, FNs and the fired alerts list.
    """
    shares = models["monthly_shares"]
    counts = models["monthly_counts"]
    keywords = models["topic_keywords"]

    months = list(shares.index)
    fired = []

    for i in range(1, len(months)):
        pm, cm = months[i-1], months[i]
        for tid in shares.columns:
            ps = float(shares.loc[pm, tid])
            cs = float(shares.loc[cm, tid])
            cnt = int(counts.loc[cm, tid])
            inc = cs - ps
            if inc >= threshold and cnt >= min_count:
                fired.append({
                    "previous_month": pm,
                    "current_month": cm,
                    "topic_id": int(tid),
                    "keyword": keywords[int(tid)],
                    "prev_share": round(ps, 3),
                    "curr_share": round(cs, 3),
                    "increase": round(inc, 3),
                    "count": cnt,
                })

    # --- Match against ground truth ---
    gt_set = {(r["month"], r["topic_keyword"]) for _, r in gt.iterrows()}

    tp, fp = 0, 0
    tp_keys = set()
    for f in fired:
        key = (f["current_month"], f["keyword"])
        if key in gt_set:
            f["status"] = "TP"
            tp += 1
            tp_keys.add(key)
        else:
            f["status"] = "FP"
            fp += 1

    fn = len(gt_set - tp_keys)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)

    return {
        "threshold": threshold,
        "min_count": min_count,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "n_fired": len(fired),
        "alerts": fired,
    }


def sweep_thresholds(models, gt: pd.DataFrame, rep: Rep) -> pd.DataFrame:
    rep.banner("[4/6] Threshold sweep")

    rows = []
    best_f1 = -1
    best_threshold = None

    for t in THRESHOLDS:
        r = evaluate_alerts(models, gt, t, MIN_COUNT)
        rows.append({
            "threshold": r["threshold"],
            "tp": r["tp"],
            "fp": r["fp"],
            "fn": r["fn"],
            "precision": r["precision"],
            "recall": r["recall"],
            "f1": r["f1"],
            "n_fired": r["n_fired"],
        })
        if r["f1"] > best_f1:
            best_f1 = r["f1"]
            best_threshold = t

    sweep_df = pd.DataFrame(rows)
    print()
    print(sweep_df.to_string(index=False))
    print()
    rep.ok(f"best F1 = {best_f1:.3f} at threshold = {best_threshold}")
    return sweep_df, best_threshold


# ==================================================================
# Step 5 -- Final alerts at best threshold
# ==================================================================
def final_alerts(models, gt: pd.DataFrame, best_threshold: float,
                 rep: Rep) -> pd.DataFrame:
    rep.banner("[5/6] Final alerts at best threshold")

    r = evaluate_alerts(models, gt, best_threshold, MIN_COUNT)

    print()
    print(f"Threshold = {best_threshold}   min_count = {MIN_COUNT}")
    print(f"  TP={r['tp']}  FP={r['fp']}  FN={r['fn']}")
    print(f"  Precision = {r['precision']:.3f}")
    print(f"  Recall    = {r['recall']:.3f}")
    print(f"  F1        = {r['f1']:.3f}")
    print()

    fired = pd.DataFrame(r["alerts"])
    if not fired.empty:
        print("Alerts fired:")
        print(fired.to_string(index=False))
    else:
        rep.warn("no alerts fired at this threshold")

    return fired, r


# ==================================================================
# Step 6 -- Save artifacts + summary
# ==================================================================
def save_outputs(gt, sweep_df, best_threshold, final_fired,
                 best_metrics, models, rep: Rep) -> None:
    rep.banner("[6/6] Saving artifacts")

    if rep.dry:
        rep.info("dry-run: nothing written")
        return

    FINAL_DIR.mkdir(parents=True, exist_ok=True)

    gt.to_csv(FINAL_DIR / "ground_truth_alerts.csv", index=False)
    sweep_df.to_csv(FINAL_DIR / "threshold_sweep.csv", index=False)
    final_fired.to_csv(FINAL_DIR / "alerts_at_best_threshold.csv", index=False)

    pr_df = pd.DataFrame([{
        "threshold": best_threshold,
        "min_count": MIN_COUNT,
        "tp": best_metrics["tp"],
        "fp": best_metrics["fp"],
        "fn": best_metrics["fn"],
        "precision": best_metrics["precision"],
        "recall": best_metrics["recall"],
        "f1": best_metrics["f1"],
    }])
    pr_df.to_csv(FINAL_DIR / "alert_precision_recall.csv", index=False)

    rep.ok("saved ground_truth_alerts.csv")
    rep.ok("saved threshold_sweep.csv")
    rep.ok("saved alerts_at_best_threshold.csv")
    rep.ok("saved alert_precision_recall.csv")

    # Plots
    try:
        _plot_sweep(sweep_df, best_threshold)
        rep.ok("saved threshold_sweep.png")
    except Exception as exc:
        rep.warn(f"sweep plot failed: {exc}")

    try:
        _plot_confusion(best_metrics)
        rep.ok("saved alert_confusion.png")
    except Exception as exc:
        rep.warn(f"confusion plot failed: {exc}")

    # Summary
    summary = build_summary(gt, sweep_df, best_threshold,
                            final_fired, best_metrics)
    (FINAL_DIR / "phase_4_summary.txt").write_text(summary, encoding="utf-8")
    rep.ok("saved phase_4_summary.txt")


def _plot_sweep(sweep_df, best_threshold) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(sweep_df["threshold"], sweep_df["precision"],
            marker="o", label="Precision", color="#5b8def")
    ax.plot(sweep_df["threshold"], sweep_df["recall"],
            marker="s", label="Recall", color="#22c55e")
    ax.plot(sweep_df["threshold"], sweep_df["f1"],
            marker="^", label="F1", color="#ef4444", linewidth=2)

    ax.axvline(best_threshold, linestyle="--", color="black", alpha=0.5,
               label=f"Chosen ({best_threshold})")

    ax.set_xlabel("Alert threshold (share increase)")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Alert detector threshold sweep")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FINAL_DIR / "threshold_sweep.png", dpi=180)
    plt.close()


def _plot_confusion(metrics) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tp, fp, fn = metrics["tp"], metrics["fp"], metrics["fn"]

    fig, ax = plt.subplots(figsize=(6, 5))
    matrix = np.array([[tp, fp], [fn, 0]])

    import seaborn as sns
    sns.heatmap(matrix, annot=True, fmt="d", cmap="RdYlGn",
                xticklabels=["Fired", "Not fired"],
                yticklabels=["Real", "Not real"],
                cbar_kws={"label": "count"}, ax=ax)
    ax.set_title(f"Alert detection -- TP={tp}  FP={fp}  FN={fn}")
    plt.tight_layout()
    plt.savefig(FINAL_DIR / "alert_confusion.png", dpi=180)
    plt.close()


def build_summary(gt, sweep_df, best_threshold, final_fired,
                  best_metrics) -> str:
    lines = []
    lines.append("PHASE 4 -- EMERGING-PATTERN DETECTION & EVALUATION")
    lines.append("=" * 62)
    lines.append("")
    lines.append("1. Ground truth (synthetic)")
    lines.append("-" * 62)
    for _, row in gt.iterrows():
        lines.append(f"  {row['month']}  topic {row['topic_id']} "
                     f"({row['topic_keyword']})  "
                     f"expected_share={row['expected_share']}  "
                     f"count={row['expected_count']}")
    lines.append("")
    lines.append("2. Alert rule")
    lines.append("-" * 62)
    lines.append(f"  Rule: current_share - previous_share >= threshold")
    lines.append(f"        AND current_count >= {MIN_COUNT}")
    lines.append("")
    lines.append("3. Threshold sweep")
    lines.append("-" * 62)
    for _, row in sweep_df.iterrows():
        marker = "  <-- chosen" if row["threshold"] == best_threshold else ""
        lines.append(
            f"  t={row['threshold']:.2f}  "
            f"TP={row['tp']}  FP={row['fp']}  FN={row['fn']}  "
            f"P={row['precision']:.3f}  R={row['recall']:.3f}  "
            f"F1={row['f1']:.3f}{marker}"
        )
    lines.append("")
    lines.append(f"4. Chosen threshold: {best_threshold}")
    lines.append("-" * 62)
    lines.append(f"  TP={best_metrics['tp']}  "
                 f"FP={best_metrics['fp']}  "
                 f"FN={best_metrics['fn']}")
    lines.append(f"  Precision = {best_metrics['precision']:.3f}")
    lines.append(f"  Recall    = {best_metrics['recall']:.3f}")
    lines.append(f"  F1        = {best_metrics['f1']:.3f}")
    lines.append("")
    lines.append("5. Alerts fired at chosen threshold")
    lines.append("-" * 62)
    if final_fired.empty:
        lines.append("  (none)")
    else:
        for _, row in final_fired.iterrows():
            lines.append(
                f"  {row['previous_month']} -> {row['current_month']}  "
                f"T{row['topic_id']} ({row['keyword']})  "
                f"share {row['prev_share']:.3f} -> {row['curr_share']:.3f}  "
                f"(+{row['increase']:.3f})  count={row['count']}  "
                f"[{row['status']}]"
            )
    lines.append("")
    lines.append("6. Interpretation")
    lines.append("-" * 62)
    lines.append("  Precision = fraction of fired alerts that were real")
    lines.append("  Recall    = fraction of real events that were caught")
    lines.append("  F1        = harmonic mean")
    lines.append("  Lower threshold -> more alerts, higher recall, lower precision")
    lines.append("  Higher threshold -> fewer alerts, higher precision, lower recall")
    lines.append("")
    lines.append("  Ground truth is synthetic; alert calibration on real data")
    lines.append("  would require independently labeled incident reports.")
    return "\n".join(lines)


# ==================================================================
# Main
# ==================================================================
def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 4 alerts")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rep = Rep(dry=args.dry_run)

    print("=" * 66)
    print("PHASE 4 -- Emerging-Pattern Detection & Evaluation")
    print("=" * 66)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Output dir  : {FINAL_DIR}")
    if args.dry_run:
        print("MODE: DRY-RUN")

    try:
        df = load_and_clean(rep)
        models = fit_and_share(df, rep)
        gt = build_ground_truth(models, rep)
        sweep_df, best_threshold = sweep_thresholds(models, gt, rep)
        final_fired, best_metrics = final_alerts(
            models, gt, best_threshold, rep)
        save_outputs(gt, sweep_df, best_threshold, final_fired,
                     best_metrics, models, rep)
    except Exception as exc:
        print()
        print(f"FAILED: {exc}")
        import traceback
        traceback.print_exc()
        return 1

    print()
    print("=" * 66)
    print("PHASE 4 COMPLETE")
    print("=" * 66)
    print("Artifacts in artifacts/final/:")
    print("  ground_truth_alerts.csv")
    print("  threshold_sweep.csv")
    print("  threshold_sweep.png")
    print("  alert_precision_recall.csv")
    print("  alerts_at_best_threshold.csv")
    print("  alert_confusion.png")
    print("  phase_4_summary.txt")
    print()
    print("Next: review phase_4_summary.txt, then start Phase 5.")
    return 0


if __name__ == "__main__":
    sys.exit(main())