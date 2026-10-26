"""
make_preprocessing.py
---------------------
Generates notebooks/02_preprocessing_and_split.ipynb from scratch.

Run from project root:
    python scripts/make_preprocessing.py

Applies every fix from the review of 02_preprocessing_and_split.ipynb:
  - Robust project-root path resolution (no ../ paths)
  - Label validation before mapping (raises on unexpected values)
  - document_id null/blank check before grouped split
  - Bounded-length HTML stripping + diagnostic of matched strings
  - Conflict-aware sentence deduplication
  - Guarded GroupShuffleSplit with label-presence checks
  - Document AND text overlap assertions
  - Safe artifact saving (with open(...) for JSON)
  - raw_rows and rows_after_preprocessing recorded in split_info.json
"""

import json
from pathlib import Path

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"
NOTEBOOK_PATH = NOTEBOOK_DIR / "02_preprocessing_and_split.ipynb"

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
cells.append(md("title", """# 02 — Preprocessing & Document-Level Split

**Inputs:**  `dataset/SMAH-CTI.csv`
**Outputs:** `artifacts/train.csv`, `artifacts/test.csv`, `artifacts/split_info.json`
**Feeds thesis:** Chapter 3.1 — Data Preprocessing and Partitioning

Pipeline:
1. Resolve project root and load CSV.
2. Validate schema, labels, and `document_id` **before** any transformation.
3. Clean `sentence_text` (bounded HTML stripping + whitespace normalisation).
4. Detect and resolve duplicate sentences **with conflicting labels**.
5. Split 80/20 **at document level** (`GroupShuffleSplit`) so no document appears in both partitions.
6. Assert **zero document overlap** and **zero sentence-text overlap**.
7. Persist `train.csv`, `test.csv`, `split_info.json` with `raw_rows` and audit fields.

**Review-driven changes:**
- Paths resolved via `PROJECT_ROOT` (no `../`).
- Labels validated; unexpected values raise `ValueError`.
- `document_id` null/blank check before grouped split.
- Bounded-length HTML regex (avoids stripping legitimate `<…>` content).
- Deduplication raises on conflicting labels instead of keeping the first silently.
- `GroupShuffleSplit` guarded by empty-set and class-presence checks.
- `text_overlap` asserted alongside `document_overlap`.
- JSON written with `with open(...) as f`.
- `raw_rows` and `rows_after_preprocessing` recorded in `split_info.json`.
"""))

# ---------------------------------------------------------------
# 1. Setup
# ---------------------------------------------------------------
cells.append(md("setup_md", """## 1. Setup — robust paths and input validation

Find the project root by walking up from the current working directory until
`dataset/SMAH-CTI.csv` is located. This makes the notebook safe to run from
`notebooks/`, the project root, or any deeper subfolder.
"""))

cells.append(code("setup_code", '''from pathlib import Path
import json
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# ------------------------------------------------------------------
# Locate project root
# ------------------------------------------------------------------
cwd = Path.cwd()
PROJECT_ROOT = next(
    (
        p for p in [cwd, *cwd.parents]
        if (p / "dataset" / "SMAH-CTI.csv").exists()
    ),
    None,
)
if PROJECT_ROOT is None:
    raise FileNotFoundError(
        f"Could not find dataset/SMAH-CTI.csv from {cwd} or any parent."
    )

CSV_PATH  = PROJECT_ROOT / "dataset" / "SMAH-CTI.csv"
ARTIFACTS = PROJECT_ROOT / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)

print("Project root:", PROJECT_ROOT)
print("Dataset     :", CSV_PATH)
print("Artifacts   :", ARTIFACTS)

# ------------------------------------------------------------------
# Load + validate schema
# ------------------------------------------------------------------
df = pd.read_csv(CSV_PATH, low_memory=False)
raw_rows = len(df)

required = {"doc_id", "sentence_text", "relevant"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")

if df.empty:
    raise ValueError("Input dataset is empty.")

print("\\nOriginal shape:", df.shape)
print("Original rows :", raw_rows)
'''))

# ---------------------------------------------------------------
# 2. Rename + validate labels
# ---------------------------------------------------------------
cells.append(md("labels_md", """## 2. Rename columns & validate labels

- `doc_id` → `document_id`
- Validate `relevant` values **before** mapping — unexpected labels raise an error rather than becoming `NaN`.
- Check `document_id` for null/blank values: a grouped split cannot tolerate missing groups.
"""))

cells.append(code("labels_code", '''# --- Rename ---
df = df.rename(columns={"doc_id": "document_id"})

# --- Validate labels ---
labels = df["relevant"].astype("string").str.strip().str.lower()

print("Original label values:")
print(labels.value_counts(dropna=False))

allowed = {"yes", "no"}
unexpected = sorted(
    v for v in labels.dropna().unique() if v not in allowed
)
if unexpected:
    raise ValueError(
        f"Unexpected relevance labels: {unexpected}. "
        f"Expected only {sorted(allowed)}."
    )

df["relevant"] = labels.map({"yes": 1, "no": 0})

# --- Audit missing values before dropping ---
print("\\nMissing document_id :", int(df["document_id"].isna().sum()))
print("Missing sentence_text:", int(df["sentence_text"].isna().sum()))
print("Missing labels       :", int(df["relevant"].isna().sum()))

before = len(df)
df = df.dropna(
    subset=["document_id", "sentence_text", "relevant"]
).copy()

# --- Trim/blank-check document_id ---
df["document_id"] = df["document_id"].astype(str).str.strip()
df = df[df["document_id"].ne("")].copy()

# --- Cast relevant to int ---
df["relevant"] = df["relevant"].astype(int)

if df["relevant"].nunique() < 2:
    raise ValueError("Only one relevance class remains after validation.")

df = df.reset_index(drop=True)

print(f"\\nRows dropped during label validation: {before - len(df)}")
print("Rows after validation:", len(df))
print("\\nLabel counts:")
print(df["relevant"].value_counts().sort_index())
print("Unique documents:", df["document_id"].nunique())
'''))

# ---------------------------------------------------------------
# 3. Text cleaning
# ---------------------------------------------------------------
cells.append(md("clean_md", """## 3. Clean `sentence_text`

- Strip HTML-like tags **only** when bounded in length (`<[^>]{0,80}>`), so long
  threat-content between `<…>` (IOCs, technique IDs) is preserved.
- Collapse whitespace and trim.
- Print a diagnostic of what the regex actually matched so you can confirm it
  is not eating real signal on this dataset.
"""))

cells.append(code("clean_code", '''TAG_RE = re.compile(r"<[^>]{0,80}>")


def clean(text):
    if not isinstance(text, str):
        return ""
    text = TAG_RE.sub(" ", text)
    text = re.sub(r"\\s+", " ", text).strip()
    return text


# --- Diagnostic: sample what the regex matches before applying ---
sample_matches = (
    df["sentence_text"]
      .astype(str)
      .str.findall(TAG_RE.pattern)
      .explode()
      .dropna()
      .unique()
)
print(f"Distinct tag-like matches: {len(sample_matches)}")
print("Examples:", list(sample_matches[:20]))

# --- Apply ---
before = len(df)
df["sentence_text"] = df["sentence_text"].apply(clean)
df = df[df["sentence_text"].str.len() > 0].reset_index(drop=True)
print(f"\\nRows removed by cleaning (empty text): {before - len(df)}")
print("Rows after cleaning:", len(df))
'''))

# ---------------------------------------------------------------
# 4. Deduplicate (conflict-aware)
# ---------------------------------------------------------------
cells.append(md("dedup_md", """## 4. Remove duplicate sentences (conflict-aware)

If two rows share the same cleaned `sentence_text` but disagree on `relevant`,
the notebook **raises** rather than silently keeping the first occurrence.
Inspect those examples and resolve before re-running.

Note (for thesis methods): global sentence deduplication collapses repeated
sentence text across documents to a single row, so the document count can drop
(e.g. 100 → 99). Record that decision in Chapter 3.1.
"""))

cells.append(code("dedup_code", '''# --- Detect conflicting labels before deduplication ---
label_counts = (
    df.groupby("sentence_text")["relevant"]
      .nunique()
)
conflicting = label_counts[label_counts > 1]

print("Unique sentence texts        :", int(label_counts.size))
print("Texts with conflicting labels:", int(len(conflicting)))

if len(conflicting):
    examples = conflicting.index[:10].tolist()
    raise ValueError(
        "Identical sentence text has conflicting relevance labels. "
        f"Inspect these examples before deduplicating: {examples}"
    )

# --- Safe to deduplicate ---
before = len(df)
df = (
    df.drop_duplicates(subset=["sentence_text"], keep="first")
      .reset_index(drop=True)
)
print(f"\\nRows removed as duplicate sentence text: {before - len(df)}")
print("Rows remaining:", len(df))
print("\\nLabel counts after deduplication:")
print(df["relevant"].value_counts().sort_index())
print("Unique documents after deduplication:", df["document_id"].nunique())
'''))

# ---------------------------------------------------------------
# 5. Document-level split
# ---------------------------------------------------------------
cells.append(md("split_md", """## 5. Document-level 80/20 split (`GroupShuffleSplit`)

`GroupShuffleSplit(test_size=0.2)` targets 20% of **documents**, not 20% of
**rows**. Row fractions will drift; that is expected. This cell:

- Requires ≥ 2 distinct documents.
- Splits on `document_id` so no document appears in both partitions.
- Fails if either side is empty.
- Fails if either class is absent from either partition — silent class collapse is
  the classic source of poisoned experiments.
"""))

cells.append(code("split_code", '''if df["document_id"].nunique() < 2:
    raise ValueError("At least two distinct documents are required to split.")

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(
    gss.split(df, y=df["relevant"], groups=df["document_id"])
)

train = df.iloc[train_idx].reset_index(drop=True)
test  = df.iloc[test_idx].reset_index(drop=True)

if train.empty or test.empty:
    raise ValueError("The split produced an empty train or test set.")

# --- Class presence check ---
for name, part in [("train", train), ("test", test)]:
    if part["relevant"].nunique() < 2:
        raise ValueError(
            f"{name} set contains only one class after split; "
            "adjust random_state or split strategy."
        )

print(
    f"Train: {len(train):,} rows "
    f"({len(train) / len(df):.1%}), "
    f"{train['document_id'].nunique():,} documents"
)
print(
    f"Test : {len(test):,} rows "
    f"({len(test) / len(df):.1%}), "
    f"{test['document_id'].nunique():,} documents"
)
print("\\nTrain label proportions:")
print(train["relevant"].value_counts(normalize=True).sort_index().round(4))
print("\\nTest label proportions:")
print(test["relevant"].value_counts(normalize=True).sort_index().round(4))
'''))

# ---------------------------------------------------------------
# 6. Leakage check
# ---------------------------------------------------------------
cells.append(md("leak_md", """## 6. Leakage check

Assert **both** document overlap (the critical leakage vector) and
sentence-text overlap (defensive: catches regressions if deduplication is
later changed).
"""))

cells.append(code("leak_code", '''document_overlap = set(train["document_id"]) & set(test["document_id"])

train_texts = set(train["sentence_text"])
test_texts  = set(test["sentence_text"])
text_overlap = train_texts & test_texts

print("Overlapping documents    :", len(document_overlap))
print("Overlapping sentence texts:", len(text_overlap))

assert not document_overlap, "Document leakage detected!"
assert not text_overlap,     "Sentence-text overlap detected!"
print("\\nNo document or sentence-text overlap.")
'''))

# ---------------------------------------------------------------
# 7. Save artifacts
# ---------------------------------------------------------------
cells.append(md("save_md", """## 7. Save artifacts

- `artifacts/train.csv`
- `artifacts/test.csv`
- `artifacts/split_info.json` (audited: original rows, cleaned rows, sizes, overlaps, positive rates, split config)
"""))

cells.append(code("save_code", '''train_path = ARTIFACTS / "train.csv"
test_path  = ARTIFACTS / "test.csv"
info_path  = ARTIFACTS / "split_info.json"

train.to_csv(train_path, index=False)
test.to_csv(test_path,  index=False)

split_info = {
    "original_rows":           int(raw_rows),
    "rows_after_preprocessing": int(len(df)),
    "train_size":              int(len(train)),
    "test_size":               int(len(test)),
    "train_docs":              int(train["document_id"].nunique()),
    "test_docs":               int(test["document_id"].nunique()),
    "overlap_docs":            int(len(document_overlap)),
    "overlap_sentence_texts":  int(len(text_overlap)),
    "train_pos_rate":          round(float(train["relevant"].mean()), 4),
    "test_pos_rate":           round(float(test["relevant"].mean()), 4),
    "random_state":            42,
    "split_method":            "GroupShuffleSplit",
}

with open(info_path, "w", encoding="utf-8") as f:
    json.dump(split_info, f, indent=2)

print(json.dumps(split_info, indent=2))
print("\\nSaved:")
print(" ", train_path)
print(" ", test_path)
print(" ", info_path)
'''))

# ---------------------------------------------------------------
# 8. Research note
# ---------------------------------------------------------------
cells.append(md("note_md", """## 8. Research note

This notebook produces a **binary relevance** split (`relevant ∈ {0, 1}`). It is
the correct input for the baseline and SBERT relevance-classification
experiments in Chapters 4.2 and 4.3. It does **not** produce multi-class
cyber-threat type labels — those are derived in the BERTopic and temporal
analysis stages and must be documented separately.
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