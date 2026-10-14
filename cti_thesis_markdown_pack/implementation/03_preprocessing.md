# 03. Text Preprocessing

## Goal

Create a consistent text field from report title and body, remove
unusable rows, and prepare labels and dates.

## Minimal preprocessing

``` python
import pandas as pd

df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["category"] = df["category"].astype("string").str.strip()
df["model_text"] = (
    df["title"].fillna("").astype(str) + " " +
    df["text"].fillna("").astype(str)
).str.strip()

df = df.dropna(subset=["date", "category"])
df = df[df["model_text"].str.len() > 0].copy()

print("Rows after cleaning:", len(df))
print("Categories:", df["category"].nunique())
```

## Leakage checks

Before trusting the model, inspect whether the target label appears
directly in titles or bodies:

``` python
for label in sorted(df["category"].dropna().unique()):
    mask = df["model_text"].str.contains(str(label), case=False, regex=False)
    print(label, int(mask.sum()))
```

This is only a simple diagnostic. Review generated templates and
near-duplicate reports too.

## Train/test split

Use a fixed random seed for repeatability. If reports share templates or
near-duplicates, random splitting may put highly similar examples into
both sets. Where possible, group related reports or use a chronological
holdout to test generalization.

## Record

Document rows before/after cleaning, label counts, date coverage, split
method, and any excluded records.
