# scripts/phase_2.py
"""
PHASE 2 -- Leakage Audit & Honest Baselines
============================================

Produces:

    artifacts/final/leakage_audit.csv           category x leak %
    artifacts/final/leakage_audit.png
    artifacts/final/split_comparison.csv        the key thesis table
    artifacts/final/split_comparison.png
    artifacts/final/confusion_matrix_clean_chrono.png
    artifacts/final/per_class_f1_clean_chrono.csv
    artifacts/final/phase_2_summary.txt

Splits evaluated:

    1. Random split + raw text          (reproduces the 1.00 illusion)
    2. Random split + cleaned text      (removes label leakage)
    3. Chronological split + cleaned    (realistic generalization test)
    4. DummyClassifier floor            (most-frequent baseline)

Run:
    python scripts/phase_2.py

Dry-run:
    python scripts/phase_2.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET      = PROJECT_ROOT / "dataset" / "synthetic_cti_reports.csv"
FINAL_DIR    = PROJECT_ROOT / "artifacts" / "final"

RANDOM_STATE = 42
TEST_SIZE    = 0.20


# ==================================================================
# Console helpers
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
# Step 1 -- Load dataset
# ==================================================================
def load_dataset(rep: Rep) -> pd.DataFrame:
    rep.banner("[1/6] Loading dataset")

    if not DATASET.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET}")

    df = pd.read_csv(DATASET)
    rep.ok(f"loaded {len(df)} rows x {df.shape[1]} columns")
    rep.info(f"columns: {df.columns.tolist()}")

    required = ["title", "text", "category", "date"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    df["category"] = df["category"].astype(str).str.strip()
    df["title"]    = df["title"].fillna("").astype(str)
    df["text"]     = df["text"].fillna("").astype(str)
    df["date"]     = pd.to_datetime(df["date"], errors="coerce")

    df = df.dropna(subset=["date", "category"]).reset_index(drop=True)
    df = df[df["category"].str.len() > 0].reset_index(drop=True)

    # Build the raw combined text
    df["model_text_raw"] = (
        df["title"].str.strip() + " " + df["text"].str.strip()
    ).str.strip()

    df = df[df["model_text_raw"].str.len() > 0].reset_index(drop=True)

    rep.ok(f"clean dataset: {len(df)} rows, "
           f"{df['category'].nunique()} categories")
    rep.info(f"date range: {df['date'].min().date()} -> {df['date'].max().date()}")

    return df


# ==================================================================
# Step 2 -- Leakage audit
# ==================================================================
def leakage_audit(df: pd.DataFrame, rep: Rep) -> tuple[pd.DataFrame, pd.DataFrame]:
    rep.banner("[2/6] Leakage audit")

    rows = []
    for cat in sorted(df["category"].unique()):
        mask = df["model_text_raw"].str.contains(
            cat, case=False, regex=False
        )
        hits = int(mask.sum())
        pct = hits / len(df) * 100
        rows.append({
            "category": cat,
            "rows": int((df["category"] == cat).sum()),
            "leak_hits": hits,
            "leak_pct": round(pct, 2),
        })

    audit = pd.DataFrame(rows).sort_values("leak_pct", ascending=False)
    print()
    print(audit.to_string(index=False))

    total_leaks = int(audit["leak_hits"].sum())
    total_rows = len(df)
    overall_pct = total_leaks / total_rows * 100

    rep.ok(f"total rows with label-as-substring: {total_leaks} "
           f"({overall_pct:.2f}% of corpus)")

    # --- Build a cleaned copy of model_text ---------------------
    rep.info("building model_text_clean ...")

    df = df.copy()

    # 1. remove category labels
    def strip_label(text: str, label: str) -> str:
        return re.sub(re.escape(label), " ", text, flags=re.IGNORECASE)

    def clean_row(row):
        text = row["model_text_raw"]
        for cat in df["category"].unique():
            text = strip_label(text, cat)
        return text

    df["model_text_clean"] = df.apply(clean_row, axis=1)

    # 2. strip synthetic template phrases observed in the corpus
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
        r"severity",
    ]
    for phrase in template_phrases:
        df["model_text_clean"] = df["model_text_clean"].str.replace(
            phrase, " ", case=False, regex=True
        )

    # 3. remove MITRE IDs, CVE IDs, IOCs (all leak shortcuts)
    df["model_text_clean"] = df["model_text_clean"].str.replace(
        r"CVE-\d{4}-\d+", " ", regex=True
    )
    df["model_text_clean"] = df["model_text_clean"].str.replace(
        r"T\d{4}(?:\.\d{3})?", " ", regex=True
    )
    df["model_text_clean"] = df["model_text_clean"].str.replace(
        r"(?:IP|DOMAIN|SHA256|MD5|URL)=\S+", " ", regex=True
    )

    # 4. strip standalone digits
    df["model_text_clean"] = df["model_text_clean"].str.replace(
        r"\b\d+\b", " ", regex=True
    )

    # 5. collapse whitespace
    df["model_text_clean"] = df["model_text_clean"].str.replace(
        r"\s+", " ", regex=True
    ).str.strip()

    # --- Re-run leakage check on cleaned text -------------------
    rows_clean = []
    for cat in sorted(df["category"].unique()):
        mask = df["model_text_clean"].str.contains(
            cat, case=False, regex=False
        )
        hits = int(mask.sum())
        rows_clean.append({
            "category": cat,
            "leak_hits_clean": hits,
            "leak_pct_clean": round(hits / len(df) * 100, 2),
        })
    audit_clean = pd.DataFrame(rows_clean)
    audit = audit.merge(audit_clean, on="category", how="left")

    print()
    print("After cleaning:")
    print(audit[["category", "leak_pct", "leak_pct_clean"]].to_string(index=False))

    rep.ok(f"post-cleaning leak total: "
           f"{int(audit['leak_hits_clean'].sum())} rows")

    # --- Save ---------------------------------------------------
    if not rep.dry:
        FINAL_DIR.mkdir(parents=True, exist_ok=True)
        audit.to_csv(FINAL_DIR / "leakage_audit.csv", index=False)
        rep.ok("saved artifacts/final/leakage_audit.csv")
        try:
            _plot_leakage(audit)
            rep.ok("saved artifacts/final/leakage_audit.png")
        except Exception as exc:
            rep.warn(f"plot failed: {exc}")

    return df, audit


def _plot_leakage(audit: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    plot_df = audit.sort_values("leak_pct", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    y = np.arange(len(plot_df))

    ax.barh(y - 0.2, plot_df["leak_pct"],
            height=0.4, label="Raw text", color="#ef4444")
    ax.barh(y + 0.2, plot_df["leak_pct_clean"],
            height=0.4, label="Cleaned text", color="#22c55e")

    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["category"], fontsize=9)
    ax.set_xlabel("Leak %  (rows containing label as substring)")
    ax.set_title("Label leakage before vs after text cleaning")
    ax.legend()
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FINAL_DIR / "leakage_audit.png", dpi=180)
    plt.close()


# ==================================================================
# Step 3 -- Train & evaluate across 4 conditions
# ==================================================================
def build_models() -> dict:
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced",
            random_state=RANDOM_STATE),
        "Multinomial NB": MultinomialNB(),
        "Linear SVM": LinearSVC(
            class_weight="balanced", random_state=RANDOM_STATE),
    }


def make_pipe(clf) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=20000,
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True,
        )),
        ("clf", clf),
    ])


def evaluate(clf, X_tr, y_tr, X_te, y_te) -> dict:
    pipe = make_pipe(clf)
    pipe.fit(X_tr, y_tr)
    y_pred = pipe.predict(X_te)
    return {
        "accuracy":     round(accuracy_score(y_te, y_pred), 4),
        "f1_weighted":  round(f1_score(y_te, y_pred,
                                       average="weighted"), 4),
        "f1_macro":     round(f1_score(y_te, y_pred,
                                       average="macro"), 4),
        "y_pred":       y_pred,
        "pipe":         pipe,
    }


def split_random(df: pd.DataFrame, text_col: str):
    X = df[text_col]
    y = df["category"]
    return train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def split_chrono(df: pd.DataFrame, text_col: str):
    df = df.sort_values("date").reset_index(drop=True)
    cutoff_idx = int(len(df) * (1 - TEST_SIZE))
    cutoff_date = df.loc[cutoff_idx, "date"]
    train = df[df["date"] < cutoff_date]
    test  = df[df["date"] >= cutoff_date]
    return (
        train[text_col], test[text_col],
        train["category"], test["category"],
        cutoff_date,
    )


# ==================================================================
# Step 4 -- Run the four conditions
# ==================================================================
def run_comparison(df: pd.DataFrame, rep: Rep) -> pd.DataFrame:
    rep.banner("[3/6] Running 4-way comparison")

    results = []

    # --- Condition 1: random + raw -------------------------------
    rep.info("condition 1: random split, raw text (baseline illusion)")
    X_tr, X_te, y_tr, y_te = split_random(df, "model_text_raw")
    for name, clf in build_models().items():
        r = evaluate(clf, X_tr, y_tr, X_te, y_te)
        results.append({
            "condition": "Random / Raw",
            "model": name,
            "accuracy": r["accuracy"],
            "f1_weighted": r["f1_weighted"],
            "f1_macro": r["f1_macro"],
            "n_train": len(X_tr),
            "n_test": len(X_te),
        })

    # --- Condition 2: random + clean -----------------------------
    rep.info("condition 2: random split, cleaned text")
    X_tr, X_te, y_tr, y_te = split_random(df, "model_text_clean")
    for name, clf in build_models().items():
        r = evaluate(clf, X_tr, y_tr, X_te, y_te)
        results.append({
            "condition": "Random / Clean",
            "model": name,
            "accuracy": r["accuracy"],
            "f1_weighted": r["f1_weighted"],
            "f1_macro": r["f1_macro"],
            "n_train": len(X_tr),
            "n_test": len(X_te),
        })

    # --- Condition 3: chronological + clean ----------------------
    rep.info("condition 3: chronological split, cleaned text")
    X_tr, X_te, y_tr, y_te, cutoff = split_chrono(df, "model_text_clean")
    rep.info(f"  chronological cutoff: {cutoff.date()}")
    for name, clf in build_models().items():
        r = evaluate(clf, X_tr, y_tr, X_te, y_te)
        results.append({
            "condition": "Chrono / Clean",
            "model": name,
            "accuracy": r["accuracy"],
            "f1_weighted": r["f1_weighted"],
            "f1_macro": r["f1_macro"],
            "n_train": len(X_tr),
            "n_test": len(X_te),
        })

    # --- Condition 4: dummy floor --------------------------------
    rep.info("condition 4: dummy floor (most frequent)")
    X_tr, X_te, y_tr, y_te = split_random(df, "model_text_clean")
    dummy = DummyClassifier(strategy="most_frequent",
                            random_state=RANDOM_STATE)
    dummy.fit(X_tr, y_tr)
    y_pred = dummy.predict(X_te)
    results.append({
        "condition": "Random / Clean / Dummy",
        "model": "Dummy (most_frequent)",
        "accuracy":     round(accuracy_score(y_te, y_pred), 4),
        "f1_weighted":  round(f1_score(y_te, y_pred,
                                       average="weighted"), 4),
        "f1_macro":     round(f1_score(y_te, y_pred,
                                       average="macro"), 4),
        "n_train": len(X_tr),
        "n_test":  len(X_te),
    })

    results_df = pd.DataFrame(results)

    print()
    print(results_df.to_string(index=False))

    # Save
    if not rep.dry:
        results_df.to_csv(FINAL_DIR / "split_comparison.csv", index=False)
        rep.ok("saved artifacts/final/split_comparison.csv")
        try:
            _plot_comparison(results_df)
            rep.ok("saved artifacts/final/split_comparison.png")
        except Exception as exc:
            rep.warn(f"plot failed: {exc}")

    return results_df


def _plot_comparison(df: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conditions = [
        "Random / Raw",
        "Random / Clean",
        "Chrono / Clean",
        "Random / Clean / Dummy",
    ]
    models = df["model"].unique()
    palette = {
        "Logistic Regression": "#5b8def",
        "Multinomial NB":      "#8b5cf6",
        "Linear SVM":          "#22c55e",
        "Dummy (most_frequent)": "#94a3b8",
    }

    x = np.arange(len(conditions))
    width = 0.2

    fig, ax = plt.subplots(figsize=(11, 6))
    for i, m in enumerate(models):
        values = []
        for c in conditions:
            row = df[(df["condition"] == c) & (df["model"] == m)]
            values.append(row["f1_weighted"].values[0]
                          if len(row) else 0)
        offset = (i - (len(models) - 1) / 2) * width
        bars = ax.bar(x + offset, values, width,
                      label=m, color=palette.get(m, "#888"))
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{v:.3f}", ha="center", va="bottom",
                    fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontsize=9)
    ax.set_ylabel("Weighted F1")
    ax.set_ylim(0, 1.05)
    ax.set_title("Classification F1 across evaluation conditions")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FINAL_DIR / "split_comparison.png", dpi=180)
    plt.close()


# ==================================================================
# Step 5 -- Confusion matrix + per-class F1 (chrono/clean)
# ==================================================================
def chrono_diagnostics(df: pd.DataFrame, rep: Rep) -> None:
    rep.banner("[4/6] Chronological diagnostics")

    X_tr, X_te, y_tr, y_te, cutoff = split_chrono(df, "model_text_clean")
    clf = LogisticRegression(
        max_iter=2000, class_weight="balanced",
        random_state=RANDOM_STATE)
    pipe = make_pipe(clf)
    pipe.fit(X_tr, y_tr)
    y_pred = pipe.predict(X_te)

    labels = sorted(df["category"].unique())

    # Per-class report
    report = classification_report(
        y_te, y_pred, labels=labels,
        output_dict=True, zero_division=0)
    per_class = pd.DataFrame(report).T
    per_class = per_class[
        per_class.index.isin(labels)
    ].reset_index().rename(columns={"index": "category"})
    per_class = per_class[
        ["category", "precision", "recall", "f1-score", "support"]
    ].round(3)

    print()
    print(per_class.to_string(index=False))

    if not rep.dry:
        per_class.to_csv(
            FINAL_DIR / "per_class_f1_clean_chrono.csv", index=False)
        rep.ok("saved artifacts/final/per_class_f1_clean_chrono.csv")

        try:
            _plot_confusion(y_te, y_pred, labels)
            rep.ok("saved artifacts/final/confusion_matrix_clean_chrono.png")
        except Exception as exc:
            rep.warn(f"plot failed: {exc}")


def _plot_confusion(y_te, y_pred, labels) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    cm = confusion_matrix(y_te, y_pred, labels=labels)
    plt.figure(figsize=(11, 9))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels,
                cbar_kws={"label": "count"})
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion matrix -- chronological / cleaned text")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(FINAL_DIR / "confusion_matrix_clean_chrono.png",
                dpi=180)
    plt.close()


# ==================================================================
# Step 6 -- Summary text
# ==================================================================
def write_summary(results_df: pd.DataFrame, audit: pd.DataFrame,
                  rep: Rep) -> None:
    rep.banner("[5/6] Writing summary")

    lines = []
    lines.append("PHASE 2 -- LEAKAGE AUDIT & HONEST BASELINES")
    lines.append("=" * 60)
    lines.append("")
    lines.append("1. Leakage audit (raw text)")
    lines.append("-" * 60)
    for _, row in audit.iterrows():
        lines.append(f"  {row['category']:<22} "
                     f"{row['leak_pct']:>6.2f}%  "
                     f"(cleaned: {row['leak_pct_clean']:>6.2f}%)")
    lines.append("")
    lines.append("2. Evaluation conditions (weighted F1)")
    lines.append("-" * 60)
    for _, row in results_df.iterrows():
        lines.append(
            f"  {row['condition']:<24} {row['model']:<22} "
            f"acc={row['accuracy']:.4f}  "
            f"f1_w={row['f1_weighted']:.4f}  "
            f"f1_m={row['f1_macro']:.4f}"
        )
    lines.append("")
    lines.append("3. Headline numbers")
    lines.append("-" * 60)
    raw = results_df[(results_df["condition"] == "Random / Raw")
                     & (results_df["model"] == "Logistic Regression")]
    clean = results_df[(results_df["condition"] == "Random / Clean")
                       & (results_df["model"] == "Logistic Regression")]
    chrono = results_df[(results_df["condition"] == "Chrono / Clean")
                        & (results_df["model"] == "Logistic Regression")]
    dummy = results_df[results_df["model"] == "Dummy (most_frequent)"]

    if len(raw) and len(clean) and len(chrono) and len(dummy):
        lines.append(f"  Random / Raw      F1 = {raw['f1_weighted'].values[0]:.4f}")
        lines.append(f"  Random / Clean    F1 = {clean['f1_weighted'].values[0]:.4f}")
        lines.append(f"  Chrono / Clean    F1 = {chrono['f1_weighted'].values[0]:.4f}")
        lines.append(f"  Dummy floor       F1 = {dummy['f1_weighted'].values[0]:.4f}")
        lines.append("")
        drop = raw['f1_weighted'].values[0] - clean['f1_weighted'].values[0]
        gap  = clean['f1_weighted'].values[0] - chrono['f1_weighted'].values[0]
        lines.append(f"  Leakage drop (raw -> clean):       {drop:+.4f}")
        lines.append(f"  Generalization gap (clean random -> chrono): {gap:+.4f}")

    text = "\n".join(lines)
    print()
    print(text)

    if not rep.dry:
        (FINAL_DIR / "phase_2_summary.txt").write_text(
            text, encoding="utf-8")
        rep.ok("saved artifacts/final/phase_2_summary.txt")


# ==================================================================
# Main
# ==================================================================
def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 2 leakage audit")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rep = Rep(dry=args.dry_run)

    print("=" * 66)
    print("PHASE 2 -- Leakage Audit & Honest Baselines")
    print("=" * 66)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Output dir  : {FINAL_DIR}")
    if args.dry_run:
        print("MODE: DRY-RUN")

    try:
        df = load_dataset(rep)
        df, audit = leakage_audit(df, rep)
        results_df = run_comparison(df, rep)
        chrono_diagnostics(df, rep)
        write_summary(results_df, audit, rep)
    except Exception as exc:
        print()
        print(f"FAILED: {exc}")
        return 1

    print()
    print("=" * 66)
    print("PHASE 2 COMPLETE")
    print("=" * 66)
    print("Artifacts in artifacts/final/:")
    print("  leakage_audit.csv")
    print("  leakage_audit.png")
    print("  split_comparison.csv")
    print("  split_comparison.png")
    print("  per_class_f1_clean_chrono.csv")
    print("  confusion_matrix_clean_chrono.png")
    print("  phase_2_summary.txt")
    print()
    print("Next: review phase_2_summary.txt, then start Phase 3.")
    return 0


if __name__ == "__main__":
    sys.exit(main())