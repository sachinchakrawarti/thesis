
from pathlib import Path
import json
import re
import pandas as pd

ROOT = Path(".")
INPUT = ROOT / "data/processed/cti_unified_repaired.csv"
OUT = ROOT / "data/experiments"

OUT.mkdir(parents=True, exist_ok=True)


def main():
    if not INPUT.exists():
        raise FileNotFoundError(
            f"{INPUT} not found. Run scripts/fix_primary_tech.py first."
        )

    df = pd.read_csv(INPUT, keep_default_na=False)

    required = {
        "cve_id", "sentence_text", "published", "relevant",
        "severity", "cvss_score", "techniques", "primary_tech",
        "tech_name", "in_kev", "kev_ransomware", "epss"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    # Normalize text fields.
    for col in ["relevant", "severity", "techniques", "sentence_text"]:
        df[col] = df[col].astype(str).str.strip()

    # Parse dates for auditing only.
    dates = pd.to_datetime(df["published"], errors="coerce", utc=True)

    # 1. Binary relevance dataset.
    # Keep only explicitly labelled examples; do not invent negatives.
    relevance = df[df["relevant"].str.lower().isin(["yes", "no"])].copy()
    relevance["target"] = relevance["relevant"].str.lower().map(
        {"no": 0, "yes": 1}
    )
    relevance.to_csv(OUT / "relevance.csv", index=False)

    # 2. Severity dataset.
    allowed_severity = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
    severity = df[df["severity"].str.upper().isin(allowed_severity)].copy()
    severity["target"] = severity["severity"].str.upper()
    severity.to_csv(OUT / "severity.csv", index=False)

    # 3. ATT&CK multi-label dataset.
    # Preserve all valid IDs per record; don't discard secondary techniques.
    attack = df[df["techniques"].str.strip().ne("")].copy()

    def extract_ids(value):
        return sorted(set(re.findall(
            r"(?<![A-Z0-9])T\d{4}(?:\.\d{3})?(?!\d)",
            value.upper()
        )))

    attack["technique_labels"] = attack["techniques"].apply(
        lambda value: ";".join(extract_ids(value))
    )
    attack = attack[attack["technique_labels"].str.strip().ne("")]
    attack.to_csv(OUT / "attack_multilabel.csv", index=False)

    # Audit report.
    report = {
        "input_file": str(INPUT),
        "total_rows": int(len(df)),
        "unique_cve_ids": int(df["cve_id"].nunique()),
        "duplicate_cve_rows": int(df["cve_id"].duplicated().sum()),
        "published_date_count": int(dates.notna().sum()),
        "published_date_min": (
            dates.min().isoformat() if dates.notna().any() else None
        ),
        "published_date_max": (
            dates.max().isoformat() if dates.notna().any() else None
        ),
        "relevance_rows": int(len(relevance)),
        "relevance_distribution": {
            str(k): int(v)
            for k, v in relevance["target"].value_counts().items()
        },
        "unlabelled_relevance_rows": int(
            (~df["relevant"].str.lower().isin(["yes", "no"])).sum()
        ),
        "severity_rows": int(len(severity)),
        "severity_distribution": {
            str(k): int(v)
            for k, v in severity["target"].value_counts().items()
        },
        "attack_mapped_rows": int(len(attack)),
        "unique_attack_ids": int(len(set(
            technique
            for labels in attack["technique_labels"]
            for technique in labels.split(";")
            if technique
        ))),
        "outputs": [
            "relevance.csv",
            "severity.csv",
            "attack_multilabel.csv"
        ],
        "important_warning": (
            "These are prepared datasets, not model results. "
            "Exclude target-defining fields from model features. "
            "Use train-only resampling and fit preprocessing on training data."
        )
    }

    with open(OUT / "dataset_audit.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n=== DATASET AUDIT ===")
    print(json.dumps(report, indent=2))
    print(f"\nFiles saved to: {OUT.resolve()}")


if __name__ == "__main__":
    main()
