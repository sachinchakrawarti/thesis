# scripts/phase_3.py
"""
PHASE 3 -- Topic Modeling Validation
=====================================

Produces:

    artifacts/final/topics_nmf.csv                topic x top words
    artifacts/final/topics_lda.csv                topic x top words
    artifacts/final/nmf_vs_lda.csv                NMI/ARI comparison
    artifacts/final/topic_category_confusion.png  heatmap NMF
    artifacts/final/topic_category_confusion.csv  raw counts
    artifacts/final/nmf_topic_labels.csv          human labels
    artifacts/final/weekly_topic_shares.csv       temporal signal
    artifacts/final/weekly_topic_shares.png       temporal plot
    artifacts/final/phase_3_summary.txt

Method:
    1. Load dataset, reuse cleaned model_text from Phase 2
    2. Fit NMF and LDA with k=5 on the same corpus
    3. Report top words + document counts per topic
    4. Compute NMI and ARI between topics and category labels
    5. Produce a topic x category confusion matrix
    6. Build weekly topic shares from cleaned text
    7. Save a summary with recommendations

Run:
    python scripts/phase_3.py
    python scripts/phase_3.py --dry-run
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET      = PROJECT_ROOT / "dataset" / "synthetic_cti_reports.csv"
FINAL_DIR    = PROJECT_ROOT / "artifacts" / "final"

RANDOM_STATE = 42
N_TOPICS     = 5
TOP_N_WORDS  = 10


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
# Step 1 -- Load + clean (same rules as Phase 2)
# ==================================================================
def load_and_clean(rep: Rep) -> pd.DataFrame:
    rep.banner("[1/7] Loading dataset")

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

    # Clean -- reuse Phase 2 logic
    categories = df["category"].unique()
    def strip_all_labels(text: str) -> str:
        for cat in categories:
            text = re.sub(re.escape(cat), " ", text, flags=re.IGNORECASE)
        return text

    df["model_text_clean"] = df["model_text_raw"].apply(strip_all_labels)

    template_phrases = [
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
    ]
    for phrase in template_phrases:
        df["model_text_clean"] = df["model_text_clean"].str.replace(
            phrase, " ", case=False, regex=True
        )

    df["model_text_clean"] = df["model_text_clean"].str.replace(
        r"CVE-\d{4}-\d+", " ", regex=True)
    df["model_text_clean"] = df["model_text_clean"].str.replace(
        r"T\d{4}(?:\.\d{3})?", " ", regex=True)
    df["model_text_clean"] = df["model_text_clean"].str.replace(
        r"(?:IP|DOMAIN|SHA256|MD5|URL)=\S+", " ", regex=True)
    df["model_text_clean"] = df["model_text_clean"].str.replace(
        r"\b\d+\b", " ", regex=True)
    df["model_text_clean"] = df["model_text_clean"].str.replace(
        r"\s+", " ", regex=True).str.strip()

    rep.ok(f"loaded {len(df)} rows, {df['category'].nunique()} categories")
    return df


# ==================================================================
# Step 2 -- Fit NMF + LDA
# ==================================================================
def fit_models(df: pd.DataFrame, rep: Rep):
    rep.banner("[2/7] Fitting NMF and LDA")

    # NMF on TF-IDF
    tfidf_vec = TfidfVectorizer(
        max_features=5000,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-]+\b",
    )
    X_tfidf = tfidf_vec.fit_transform(df["model_text_clean"])

    nmf = NMF(
        n_components=N_TOPICS,
        random_state=RANDOM_STATE,
        init="nndsvda",
        max_iter=500,
    )
    nmf_doc_topics = nmf.fit_transform(X_tfidf)

    # Normalize so each doc's topic weights sum to 1
    row_sums = nmf_doc_topics.sum(axis=1, keepdims=True) + 1e-12
    nmf_doc_topics_norm = nmf_doc_topics / row_sums

    # LDA on counts
    count_vec = CountVectorizer(
        max_features=5000,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-]+\b",
    )
    X_counts = count_vec.fit_transform(df["model_text_clean"])

    lda = LatentDirichletAllocation(
        n_components=N_TOPICS,
        random_state=RANDOM_STATE,
        learning_method="batch",
        max_iter=20,
    )
    lda_doc_topics = lda.fit_transform(X_counts)

    rep.ok(f"NMF: {X_tfidf.shape[0]} docs x {X_tfidf.shape[1]} features")
    rep.ok(f"LDA: {X_counts.shape[0]} docs x {X_counts.shape[1]} features")

    return {
        "nmf": nmf,
        "lda": lda,
        "tfidf_vec": tfidf_vec,
        "count_vec": count_vec,
        "X_tfidf": X_tfidf,
        "X_counts": X_counts,
        "nmf_doc_topics": nmf_doc_topics_norm,
        "lda_doc_topics": lda_doc_topics,
    }


# ==================================================================
# Step 3 -- Top words per topic
# ==================================================================
def top_words(model, feature_names, topic_id, top_n=TOP_N_WORDS):
    weights = model.components_[topic_id]
    top_idx = weights.argsort()[::-1][:top_n]
    return [feature_names[i] for i in top_idx]


def build_topic_tables(models, df, rep: Rep) -> tuple[pd.DataFrame, pd.DataFrame]:
    rep.banner("[3/7] Extracting top words per topic")

    nmf_topics = []
    lda_topics = []
    nmf_feats = models["tfidf_vec"].get_feature_names_out()
    lda_feats = models["count_vec"].get_feature_names_out()

    nmf_dom = models["nmf_doc_topics"].argmax(axis=1) + 1
    lda_dom = models["lda_doc_topics"].argmax(axis=1) + 1

    for t in range(N_TOPICS):
        nmf_words = top_words(models["nmf"], nmf_feats, t)
        lda_words = top_words(models["lda"], lda_feats, t)

        nmf_topics.append({
            "topic_id": t + 1,
            "top_words": ", ".join(nmf_words),
            "doc_count": int((nmf_dom == t + 1).sum()),
        })
        lda_topics.append({
            "topic_id": t + 1,
            "top_words": ", ".join(lda_words),
            "doc_count": int((lda_dom == t + 1).sum()),
        })

    nmf_df = pd.DataFrame(nmf_topics)
    lda_df = pd.DataFrame(lda_topics)

    print()
    print("NMF topics:")
    for _, row in nmf_df.iterrows():
        print(f"  T{row['topic_id']} ({row['doc_count']:>5} docs): "
              f"{row['top_words']}")

    print()
    print("LDA topics:")
    for _, row in lda_df.iterrows():
        print(f"  T{row['topic_id']} ({row['doc_count']:>5} docs): "
              f"{row['top_words']}")

    return nmf_df, lda_df


# ==================================================================
# Step 4 -- NMI / ARI
# ==================================================================
def evaluate_alignment(models, df, rep: Rep) -> pd.DataFrame:
    rep.banner("[4/7] Topic-category alignment (NMI / ARI)")

    nmf_dom = models["nmf_doc_topics"].argmax(axis=1)
    lda_dom = models["lda_doc_topics"].argmax(axis=1)
    y_true = df["category"].values

    rows = []

    nmf_nmi = normalized_mutual_info_score(y_true, nmf_dom)
    nmf_ari = adjusted_rand_score(y_true, nmf_dom)
    rows.append({
        "method": "NMF",
        "nmi": round(nmf_nmi, 4),
        "ari": round(nmf_ari, 4),
        "k": N_TOPICS,
    })

    lda_nmi = normalized_mutual_info_score(y_true, lda_dom)
    lda_ari = adjusted_rand_score(y_true, lda_dom)
    rows.append({
        "method": "LDA",
        "nmi": round(lda_nmi, 4),
        "ari": round(lda_ari, 4),
        "k": N_TOPICS,
    })

    result = pd.DataFrame(rows)

    print()
    print(result.to_string(index=False))
    print()
    print("Interpretation:")
    print("  NMI = 0.0 : topics are independent of category")
    print("  NMI = 1.0 : topics perfectly recover category")
    print("  ARI = 0.0 : no agreement")
    print("  ARI = 1.0 : perfect agreement")
    print()

    nmf_val = result[result["method"] == "NMF"]["nmi"].values[0]
    if nmf_val >= 0.6:
        rep.ok(f"NMF NMI={nmf_val:.3f}  -- topics strongly align with categories")
    elif nmf_val >= 0.3:
        rep.info(f"NMF NMI={nmf_val:.3f}  -- moderate alignment")
    else:
        rep.warn(f"NMF NMI={nmf_val:.3f}  -- weak alignment; report as limitation")

    return result


# ==================================================================
# Step 5 -- Topic x category confusion
# ==================================================================
def topic_category_confusion(models, df, rep: Rep) -> pd.DataFrame:
    rep.banner("[5/7] Topic x category confusion matrix (NMF)")

    nmf_dom = models["nmf_doc_topics"].argmax(axis=1) + 1
    tmp = pd.DataFrame({
        "topic": nmf_dom,
        "category": df["category"].values,
    })
    ct = pd.crosstab(tmp["topic"], tmp["category"])
    print()
    print(ct.to_string())

    # Best-matching category per topic
    print()
    print("Best-matching category per topic:")
    labels = []
    for topic_id in ct.index:
        row = ct.loc[topic_id]
        best_cat = row.idxmax()
        best_count = int(row.max())
        total = int(row.sum())
        share = best_count / total if total else 0
        labels.append({
            "topic_id": int(topic_id),
            "n_docs": total,
            "best_category": best_cat,
            "best_count": best_count,
            "purity": round(share, 3),
        })
        print(f"  T{topic_id}: {best_cat} "
              f"({best_count}/{total} = {share:.1%})")

    return pd.DataFrame(labels)


# ==================================================================
# Step 6 -- Weekly topic shares (temporal)
# ==================================================================
def weekly_shares(models, df, rep: Rep) -> pd.DataFrame:
    rep.banner("[6/7] Weekly topic shares (NMF)")

    weeks = df["date"].dt.to_period("W").dt.start_time
    tmp = pd.DataFrame(
        models["nmf_doc_topics"],
        columns=[f"Topic_{i+1}" for i in range(N_TOPICS)]
    )
    tmp["week"] = weeks.values

    weekly = tmp.groupby("week").mean().sort_index()
    print()
    print(weekly.round(3).tail(12).to_string())
    rep.ok(f"weekly matrix: {weekly.shape[0]} weeks x "
           f"{weekly.shape[1]} topics")
    return weekly


# ==================================================================
# Step 7 -- Save everything + summary
# ==================================================================
def save_outputs(nmf_df, lda_df, align_df, purity_df,
                 weekly_df, models, df, rep: Rep) -> None:
    rep.banner("[7/7] Saving artifacts")

    if rep.dry:
        rep.info("dry-run: nothing written")
        return

    FINAL_DIR.mkdir(parents=True, exist_ok=True)

    nmf_df.to_csv(FINAL_DIR / "topics_nmf.csv", index=False)
    lda_df.to_csv(FINAL_DIR / "topics_lda.csv", index=False)
    align_df.to_csv(FINAL_DIR / "nmf_vs_lda.csv", index=False)
    purity_df.to_csv(FINAL_DIR / "nmf_topic_labels.csv", index=False)
    weekly_df.to_csv(FINAL_DIR / "weekly_topic_shares.csv")
    rep.ok("saved topics_nmf.csv, topics_lda.csv, nmf_vs_lda.csv")
    rep.ok("saved nmf_topic_labels.csv, weekly_topic_shares.csv")

    # Plots
    try:
        _plot_confusion(models, df)
        rep.ok("saved topic_category_confusion.png")
    except Exception as exc:
        rep.warn(f"confusion plot failed: {exc}")

    try:
        _plot_weekly(weekly_df)
        rep.ok("saved weekly_topic_shares.png")
    except Exception as exc:
        rep.warn(f"weekly plot failed: {exc}")

    # Text summary
    summary = build_summary(nmf_df, lda_df, align_df, purity_df)
    (FINAL_DIR / "phase_3_summary.txt").write_text(summary, encoding="utf-8")
    rep.ok("saved phase_3_summary.txt")


def _plot_confusion(models, df) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    nmf_dom = models["nmf_doc_topics"].argmax(axis=1) + 1
    tmp = pd.DataFrame({
        "topic": nmf_dom,
        "category": df["category"].values,
    })
    ct = pd.crosstab(tmp["topic"], tmp["category"])

    plt.figure(figsize=(11, 6))
    sns.heatmap(ct, annot=True, fmt="d", cmap="Purples",
                cbar_kws={"label": "docs"})
    plt.xlabel("Category")
    plt.ylabel("NMF Topic")
    plt.title("Topic vs category -- NMF alignment")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(FINAL_DIR / "topic_category_confusion.png", dpi=180)
    plt.close()


def _plot_weekly(weekly_df) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 5))
    palette = ["#5b8def", "#8b5cf6", "#22c55e", "#f59e0b", "#ef4444"]
    for i, col in enumerate(weekly_df.columns):
        ax.plot(weekly_df.index, weekly_df[col],
                marker="o", markersize=3, linewidth=1.5,
                label=col, color=palette[i % len(palette)])
    ax.set_xlabel("Week")
    ax.set_ylabel("Mean topic share")
    ax.set_title("Weekly NMF topic shares (cleaned text)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FINAL_DIR / "weekly_topic_shares.png", dpi=180)
    plt.close()


def build_summary(nmf_df, lda_df, align_df, purity_df) -> str:
    lines = []
    lines.append("PHASE 3 -- TOPIC MODELING VALIDATION")
    lines.append("=" * 62)
    lines.append("")
    lines.append("1. NMF topics")
    lines.append("-" * 62)
    for _, row in nmf_df.iterrows():
        lines.append(f"  T{row['topic_id']} ({row['doc_count']:>5} docs): "
                     f"{row['top_words']}")
    lines.append("")
    lines.append("2. LDA topics")
    lines.append("-" * 62)
    for _, row in lda_df.iterrows():
        lines.append(f"  T{row['topic_id']} ({row['doc_count']:>5} docs): "
                     f"{row['top_words']}")
    lines.append("")
    lines.append("3. Alignment with ground-truth categories")
    lines.append("-" * 62)
    for _, row in align_df.iterrows():
        lines.append(f"  {row['method']:<6}  "
                     f"NMI={row['nmi']:.4f}  ARI={row['ari']:.4f}")
    lines.append("")
    lines.append("4. Topic purity (best-matching category)")
    lines.append("-" * 62)
    for _, row in purity_df.iterrows():
        lines.append(f"  T{row['topic_id']}: {row['best_category']:<22} "
                     f"purity={row['purity']:.3f} "
                     f"({row['best_count']}/{row['n_docs']})")
    lines.append("")
    lines.append("5. Interpretation guide")
    lines.append("-" * 62)
    lines.append("  NMI >= 0.6 : topics recover categories well")
    lines.append("  0.3 <= NMI < 0.6 : moderate alignment")
    lines.append("  NMI < 0.3 : weak alignment -- report as limitation")
    lines.append("")
    lines.append("  Topic purity >= 0.7 : topic is dominated by one category")
    lines.append("  Topic purity < 0.4 : topic mixes multiple categories")
    return "\n".join(lines)


# ==================================================================
# Main
# ==================================================================
def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 3 topic validation")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rep = Rep(dry=args.dry_run)

    print("=" * 66)
    print("PHASE 3 -- Topic Modeling Validation")
    print("=" * 66)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Output dir  : {FINAL_DIR}")
    if args.dry_run:
        print("MODE: DRY-RUN")

    try:
        df = load_and_clean(rep)
        models = fit_models(df, rep)
        nmf_df, lda_df = build_topic_tables(models, df, rep)
        align_df = evaluate_alignment(models, df, rep)
        purity_df = topic_category_confusion(models, df, rep)
        weekly_df = weekly_shares(models, df, rep)
        save_outputs(nmf_df, lda_df, align_df, purity_df,
                     weekly_df, models, df, rep)
    except Exception as exc:
        print()
        print(f"FAILED: {exc}")
        import traceback
        traceback.print_exc()
        return 1

    print()
    print("=" * 66)
    print("PHASE 3 COMPLETE")
    print("=" * 66)
    print("Artifacts in artifacts/final/:")
    print("  topics_nmf.csv")
    print("  topics_lda.csv")
    print("  nmf_vs_lda.csv")
    print("  nmf_topic_labels.csv")
    print("  topic_category_confusion.csv")
    print("  topic_category_confusion.png")
    print("  weekly_topic_shares.csv")
    print("  weekly_topic_shares.png")
    print("  phase_3_summary.txt")
    print()
    print("Next: review phase_3_summary.txt, then start Phase 4.")
    return 0


if __name__ == "__main__":
    sys.exit(main())