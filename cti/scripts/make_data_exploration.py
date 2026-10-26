"""
make_data_exploration.py
------------------------
Generates notebooks/01_data_exploration.ipynb from scratch.

Run from project root:
    python scripts/make_data_exploration.py

It overwrites the existing notebook with a corrected, review-compliant version.
"""

import json
from pathlib import Path

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"
NOTEBOOK_PATH = NOTEBOOK_DIR / "01_data_exploration.ipynb"

NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------
# Cell helpers
# ------------------------------------------------------------------
def md(cell_id: str, text: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {},
        "source": text.splitlines(keepends=True),
    }


def code(cell_id: str, text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


# ------------------------------------------------------------------
# Notebook cells
# ------------------------------------------------------------------
cells = []

# ---------------------------------------------------------------
# Title
# ---------------------------------------------------------------
cells.append(md("title", """# 01 — Data Exploration (EDA)

**Dataset:** `dataset/SMAH-CTI.csv`
**Feeds thesis section:** Chapter 4.1 — Exploratory Data Analysis

All figures are saved to `artifacts/figs/` and summary statistics are written to
`artifacts/eda_summary.json`. The notebook tolerates the SMAH-CTI schema, where
`doc_id` is the document key and `relevant` is stored as `yes`/`no`.

**Revision notes (post-review):**
- Paths resolved via `PROJECT_ROOT` (no `../` relative paths).
- `raw_rows` captured *immediately* after `read_csv`.
- Labels validated before mapping; unexpected values raise an error.
- Placeholder annotations (e.g. `**`, `N/A`) are treated as missing.
- Empty-data guard before `zip(*data)` in word-frequency plots.
- JSON summary written with `with open(...) as f`.
- Type-token ratio labelled as a corpus-level descriptive statistic.
"""))

# ---------------------------------------------------------------
# 1. Setup
# ---------------------------------------------------------------
cells.append(md("setup_md", "## 1. Setup\n"))
cells.append(code("setup_code", '''from pathlib import Path
import os
import json
import re
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------------
# Project root is the parent of the notebooks/ directory
# ------------------------------------------------------------------
PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()

CSV_PATH  = PROJECT_ROOT / "dataset" / "SMAH-CTI.csv"
ARTIFACTS = PROJECT_ROOT / "artifacts"
FIG_DIR   = ARTIFACTS / "figs"

FIG_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS.mkdir(parents=True, exist_ok=True)

if not CSV_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found at {CSV_PATH}. "
        "Run the notebook from the notebooks/ folder or project root."
    )

print("Project root :", PROJECT_ROOT)
print("Dataset      :", CSV_PATH)
print("Figures dir  :", FIG_DIR)

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 120
'''))

# ---------------------------------------------------------------
# 2. Load raw dataset
# ---------------------------------------------------------------
cells.append(md("load_md", """## 2. Load raw dataset & inspect schema

`raw_rows` is captured **before** any cleaning so the summary can distinguish
original vs. cleaned size.
"""))

cells.append(code("load_code", '''df = pd.read_csv(CSV_PATH, low_memory=False)

# --- Record original size BEFORE cleaning ---
raw_rows = len(df)
raw_columns = df.columns.tolist()

print("Original shape   :", df.shape)
print("Original row count:", raw_rows)
print("Columns          :", raw_columns)
print("\\nData types:\\n", df.dtypes)
print("\\nFirst 3 rows:")
df.head(3)
'''))

# ---------------------------------------------------------------
# 3. Compatibility layer (with label validation)
# ---------------------------------------------------------------
cells.append(md("compat_md", """## 3. Compatibility layer & label validation

- `doc_id` → `document_id`
- Validate `relevant` values **before** mapping; unexpected labels raise an error instead of being silently dropped.
- Drop rows missing `sentence_text` or `relevant`, recording how many were removed.
"""))

cells.append(code("compat_code", '''# --- Rename doc_id -> document_id ---
df = df.rename(columns={"doc_id": "document_id"})

# --- Validate relevant column ---
if "relevant" not in df.columns:
    raise ValueError("Required column 'relevant' is missing from the dataset.")

label_values = (
    df["relevant"]
    .astype("string")
    .str.strip()
    .str.lower()
)

print("Original label value counts:")
print(label_values.value_counts(dropna=False))

allowed = {"yes", "no"}
unexpected = sorted(
    v for v in label_values.dropna().unique() if v not in allowed
)
if unexpected:
    raise ValueError(
        f"Unexpected relevance labels found: {unexpected}. "
        f"Expected only {sorted(allowed)}."
    )

# --- Map to 0/1 ---
df["relevant"] = label_values.map({"yes": 1, "no": 0})

# --- Drop rows missing required fields ---
before_drop = len(df)
df = df.dropna(subset=["sentence_text", "relevant"]).copy()
dropped_rows = before_drop - len(df)
df = df.reset_index(drop=True)

print(f"\\nRows removed for missing text/labels: {dropped_rows}")
print("Rows after normalization:", len(df))
print("\\nClass counts:")
print(df["relevant"].value_counts())
print("Unique documents:", df["document_id"].nunique())
'''))

# ---------------------------------------------------------------
# 4. Missing-value analysis
# ---------------------------------------------------------------
cells.append(md("missing_md", "## 4. Missing-value analysis\n"))
cells.append(code("missing_code", '''missing = df.isna().sum().sort_values(ascending=False)
missing_pct = (missing / len(df) * 100).round(2)

print(pd.DataFrame({"missing": missing, "pct": missing_pct}))

if (missing_pct > 0).any():
    plt.figure(figsize=(8, 4))
    missing_pct[missing_pct > 0].plot(kind="bar", color="salmon")
    plt.ylabel("% missing")
    plt.title("Missing values by column")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_missing_values.png")
    plt.show()
'''))

# ---------------------------------------------------------------
# 5. Class distribution
# ---------------------------------------------------------------
cells.append(md("class_md", "## 5. Class distribution\n"))
cells.append(code("class_code", '''counts = df["relevant"].value_counts().sort_index()
print(counts)
print("\\nProportions:\\n",
      df["relevant"].value_counts(normalize=True).round(4))

vals = [counts.get(0, 0), counts.get(1, 0)]

plt.figure(figsize=(5, 4))
plt.bar(["non-relevant (0)", "relevant (1)"], vals,
        color=["#dd8452", "#4c72b0"])
for i, v in enumerate(vals):
    plt.text(i, v, str(v), ha="center", va="bottom")
plt.ylabel("Count")
plt.title("Class distribution")
plt.tight_layout()
plt.savefig(FIG_DIR / "fig_class_distribution.png")
plt.show()

# --- Robust imbalance ratio ---
if counts.min() > 0 and len(counts) == 2:
    ratio = counts.max() / counts.min()
    print(f"Imbalance ratio: {ratio:.2f}:1")
elif len(counts) < 2:
    print("Imbalance ratio: undefined (only one class present)")
else:
    print("Imbalance ratio: undefined (a class is absent)")
'''))

# ---------------------------------------------------------------
# 6. Sentence-length distribution
# ---------------------------------------------------------------
cells.append(md("sentlen_md", "## 6. Sentence-length distribution\n"))
cells.append(code("sentlen_code", '''df["n_words"] = df["sentence_text"].astype(str).str.split().str.len()
df["n_chars"] = df["sentence_text"].astype(str).str.len()

print(df[["n_words", "n_chars"]].describe().round(2))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(df["n_words"], bins=40, color="#4c72b0")
axes[0].set_title("Words per sentence")
axes[0].set_xlabel("words")
axes[0].set_ylabel("count")

axes[1].hist(df["n_chars"], bins=40, color="#dd8452")
axes[1].set_title("Characters per sentence")
axes[1].set_xlabel("chars")

plt.tight_layout()
plt.savefig(FIG_DIR / "fig_sentence_length.png")
plt.show()

if df["relevant"].nunique() == 2:
    plt.figure(figsize=(7, 4))
    sns.boxplot(x="relevant", y="n_words", data=df,
                hue="relevant", palette="Set2", legend=False)
    plt.xticks([0, 1], ["non-relevant", "relevant"])
    plt.title("Sentence length by class")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_length_by_class.png")
    plt.show()
else:
    print("Only one class present — skipping boxplot.")
'''))

# ---------------------------------------------------------------
# 7. Document-level analysis
# ---------------------------------------------------------------
cells.append(md("doc_md", "## 7. Document-level analysis\n"))
cells.append(code("doc_code", '''doc_counts = (
    df.groupby("document_id")
      .size()
      .sort_values(ascending=False)
)

print("Documents:", doc_counts.shape[0])
print(doc_counts.describe().round(2))

plt.figure(figsize=(8, 4))
doc_counts.head(20).plot(kind="bar", color="#55a868")
plt.title("Top-20 documents by sentence count")
plt.ylabel("# sentences")
plt.tight_layout()
plt.savefig(FIG_DIR / "fig_docs_top20.png")
plt.show()

per_doc = (
    df.groupby(["document_id", "relevant"])
      .size()
      .unstack(fill_value=0)
)
for col in (0, 1):
    if col not in per_doc.columns:
        per_doc[col] = 0
per_doc = per_doc[[1, 0]]
per_doc.columns = ["relevant", "non_relevant"]

per_doc.head(15).plot(kind="bar", stacked=True, figsize=(10, 4))
plt.title("Relevant vs non-relevant sentences — first 15 documents")
plt.tight_layout()
plt.savefig(FIG_DIR / "fig_per_doc_stacked.png")
plt.show()
'''))

# ---------------------------------------------------------------
# 8. Source analysis
# ---------------------------------------------------------------
cells.append(md("source_md", "## 8. Source analysis\n"))
cells.append(code("source_code", '''if "source" in df.columns:
    src = df["source"].value_counts()
    print(src)

    plt.figure(figsize=(7, 4))
    src.plot(kind="bar", color="#c44e52")
    plt.title("Sentences per source")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_sources.png")
    plt.show()

    rel_rate = (
        df.groupby("source")["relevant"].mean().sort_values()
    )
    plt.figure(figsize=(7, 4))
    rel_rate.plot(kind="barh", color="#8172b3")
    plt.xlabel("relevance rate")
    plt.title("Proportion relevant by source")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_relevance_by_source.png")
    plt.show()
else:
    print("No source column.")
'''))

# ---------------------------------------------------------------
# 9. Annotation coverage (with placeholder handling)
# ---------------------------------------------------------------
cells.append(md("anno_md", """## 9. Annotation coverage

`tactic`, `techniques`, `procedure` and `entities` are populated only for relevant
sentences. Placeholder strings such as `**`, `N/A`, `-` or `none` are treated as
missing **before** computing coverage, so the reported numbers reflect real
annotations.
"""))

cells.append(code("anno_code", '''ann_cols = ["tactic", "techniques", "procedure", "entities"]
present = [c for c in ann_cols if c in df.columns]

PLACEHOLDERS = {"**", "n/a", "na", "-", "none", "null", "", "nan"}

if present:
    # Replace known placeholders with NaN first
    ann_df = df[present].copy()
    for col in present:
        ann_df[col] = ann_df[col].apply(
            lambda v: np.nan
            if isinstance(v, str) and v.strip().lower() in PLACEHOLDERS
            else v
        )

    coverage = ann_df.notna().sum() / len(df) * 100
    print("Annotation coverage (%):\\n", coverage.round(2))

    rel_mask = df["relevant"] == 1
    coverage_rel = (
        ann_df[rel_mask].notna().mean() * 100
    )
    print("\\nCoverage among relevant only (%):\\n",
          coverage_rel.round(2))

    # Any remaining suspiciously short values? Helpful for debugging.
    for col in present:
        vals = ann_df[col].dropna().astype(str).str.strip()
        short = vals[vals.str.len() <= 2].unique()[:10]
        if len(short):
            print(f"  Note: short '{col}' values (<=2 chars): {short}")
'''))

# ---------------------------------------------------------------
# 10. Top words per class
# ---------------------------------------------------------------
cells.append(md("topwords_md", "## 10. Top words per class\n"))
cells.append(code("topwords_code", '''import nltk
from nltk.corpus import stopwords

nltk.download("stopwords", quiet=True)
STOP = set(stopwords.words("english"))


def top_words(texts, n=25):
    tokens = []
    for t in texts:
        tokens += [
            w for w in re.findall(r"\\b[a-z]{3,}\\b", str(t).lower())
            if w not in STOP
        ]
    return Counter(tokens).most_common(n)


top_rel = top_words(df.loc[df["relevant"] == 1, "sentence_text"])
top_irrel = top_words(df.loc[df["relevant"] == 0, "sentence_text"])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

panels = [
    (axes[0], top_rel, "Top-25 words — RELEVANT", "#4c72b0"),
    (axes[1], top_irrel, "Top-25 words — NON-RELEVANT", "#dd8452"),
]

for ax, data, title, color in panels:
    if not data:
        ax.set_title(f"{title} — no words available")
        ax.axis("off")
        continue
    words, counts_ = zip(*data)
    positions = range(len(words))[::-1]
    ax.barh(positions, counts_, color=color)
    ax.set_yticks(list(positions))
    ax.set_yticklabels(list(words))
    ax.set_title(title)
    ax.set_xlabel("frequency")

plt.tight_layout()
plt.savefig(FIG_DIR / "fig_top_words.png")
plt.show()
'''))

# ---------------------------------------------------------------
# 11. Word clouds
# ---------------------------------------------------------------
cells.append(md("wc_md", "## 11. Word clouds (optional)\n"))
cells.append(code("wc_code", '''try:
    from wordcloud import WordCloud

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, sub, title, cmap in [
        (axes[0], df[df["relevant"] == 1], "Relevant", "Blues"),
        (axes[1], df[df["relevant"] == 0], "Non-relevant", "Oranges"),
    ]:
        text = " ".join(sub["sentence_text"].astype(str).str.lower())
        if not text.strip():
            ax.set_title(f"{title} — empty corpus")
            ax.axis("off")
            continue
        wc = WordCloud(
            width=800, height=400,
            background_color="white",
            colormap=cmap,
            stopwords=STOP,
            max_words=120,
        ).generate(text)
        ax.imshow(wc)
        ax.axis("off")
        ax.set_title(title)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_wordcloud.png")
    plt.show()
except ImportError:
    print("pip install wordcloud to enable word-cloud plot.")
'''))

# ---------------------------------------------------------------
# 12. Vocabulary statistics
# ---------------------------------------------------------------
cells.append(md("vocab_md", "## 12. Vocabulary statistics\n"))
cells.append(code("vocab_code", '''all_tokens = re.findall(
    r"\\b[a-z]{3,}\\b",
    " ".join(df["sentence_text"].astype(str).str.lower()),
)
freq = Counter(all_tokens)
hapax = sum(1 for w, c in freq.items() if c == 1)
vocab = len(set(all_tokens))
total_tokens = len(all_tokens)

print("Total tokens:", total_tokens)
print("Unique tokens:", vocab)
if total_tokens:
    # NOTE: TTR is a corpus-level descriptive statistic and is not
    # directly comparable across corpora of different sizes.
    print("Type-token ratio (corpus-level):", round(vocab / total_tokens, 4))
    print(f"Hapax legomena: {hapax} ({hapax / vocab * 100:.1f}% of vocab)")
else:
    print("Corpus is empty — cannot compute vocabulary statistics.")
'''))

# ---------------------------------------------------------------
# 13. Save EDA summary
# ---------------------------------------------------------------
cells.append(md("summary_md", "## 13. Save EDA summary\n"))
cells.append(code("summary_code", '''# --- Imbalance ratio handled consistently ---
if counts.min() > 0 and len(counts) == 2:
    imbalance_ratio = round(float(counts.max() / counts.min()), 3)
else:
    imbalance_ratio = None

eda = {
    "raw_rows": int(raw_rows),
    "rows_after_cleaning": int(len(df)),
    "rows_removed": int(raw_rows - len(df)),
    "relevant": int((df["relevant"] == 1).sum()),
    "non_relevant": int((df["relevant"] == 0).sum()),
    "imbalance_ratio": imbalance_ratio,
    "num_documents": int(df["document_id"].nunique()),
    "avg_words": round(float(df["n_words"].mean()), 2),
    "median_words": float(df["n_words"].median()),
    "max_words": int(df["n_words"].max()),
    "vocab_size": int(vocab),
    "type_token_ratio": (
        round(vocab / total_tokens, 4) if total_tokens else None
    ),
    "hapax_pct": round(hapax / vocab * 100, 2) if vocab else None,
    "num_sources": (
        int(df["source"].nunique()) if "source" in df.columns else None
    ),
}

summary_path = ARTIFACTS / "eda_summary.json"
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(eda, f, indent=2)

print("Saved:", summary_path)
print(json.dumps(eda, indent=2))
'''))

# ---------------------------------------------------------------
# 14. Research note
# ---------------------------------------------------------------
cells.append(md("research_note", """## 14. Research note

This notebook only **describes** the dataset. It does not establish that any
particular topic is an emerging threat. That claim requires separate evidence
from the BERTopic and temporal-analysis stages and must be discussed in the
corresponding chapters, not here.
"""))

# ------------------------------------------------------------------
# Assemble + write
# ------------------------------------------------------------------
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Wrote notebook to {NOTEBOOK_PATH}")
print(f"Total cells: {len(cells)}")