from pathlib import Path

import pandas as pd
from datasets import load_dataset

# ============================================================
# Thesis: Cyber Threat Classification and Emerging Threat Detection
# Purpose: Compare CIRCL and ThreatCluster CVE datasets
# ============================================================

OUTPUT_DIR = Path("dataset_comparison")
OUTPUT_DIR.mkdir(exist_ok=True)


def load_all_splits(dataset_name):
    """Load all available splits and combine them into one DataFrame."""
    print(f"\nDownloading: {dataset_name}")

    dataset = load_dataset(dataset_name)
    frames = []

    for split_name, split in dataset.items():
        frame = split.to_pandas()
        frame["source_split"] = split_name
        frames.append(frame)
        print(f"  Split '{split_name}': {len(frame):,} rows")

    if not frames:
        raise ValueError(f"No dataset splits found: {dataset_name}")

    return pd.concat(frames, ignore_index=True)


def normalize_cve(series):
    """Normalize CVE IDs for reliable comparison."""
    return (
        series.astype("string")
        .str.strip()
        .str.upper()
        .replace("", pd.NA)
    )


def main():
    # 1. Load datasets
    circl = load_all_splits(
        "CIRCL/vulnerability-attack-techniques"
    )

    threat = load_all_splits(
        "threatcluster/cve-exploitation-signals"
    )

    # 2. Check expected columns
    print("\nCIRCL columns:")
    print(circl.columns.tolist())

    print("\nThreatCluster columns:")
    print(threat.columns.tolist())

    if "id" not in circl.columns:
        raise KeyError("CIRCL dataset has no expected 'id' column.")

    if "cve_id" not in threat.columns:
        raise KeyError(
            "ThreatCluster dataset has no expected 'cve_id' column."
        )

    # 3. Normalize CVE IDs
    circl["cve_id"] = normalize_cve(circl["id"])
    threat["cve_id"] = normalize_cve(threat["cve_id"])

    # Exclude missing IDs from comparisons
    circl_valid = circl.dropna(subset=["cve_id"]).copy()
    threat_valid = threat.dropna(subset=["cve_id"]).copy()

    # 4. Compare unique CVE IDs
    circl_ids = set(circl_valid["cve_id"].unique())
    threat_ids = set(threat_valid["cve_id"].unique())

    common_ids = circl_ids & threat_ids
    circl_only_ids = circl_ids - threat_ids
    threat_only_ids = threat_ids - circl_ids

    # 5. Print summary
    print("\n" + "=" * 60)
    print("DATASET COMPARISON RESULTS")
    print("=" * 60)

    print(f"CIRCL rows:                  {len(circl):,}")
    print(f"CIRCL unique CVEs:           {len(circl_ids):,}")
    print(f"ThreatCluster rows:          {len(threat):,}")
    print(f"ThreatCluster unique CVEs:   {len(threat_ids):,}")
    print(f"CVEs present in BOTH:        {len(common_ids):,}")
    print(f"CIRCL-only CVEs:             {len(circl_only_ids):,}")
    print(f"ThreatCluster-only CVEs:     {len(threat_only_ids):,}")

    union_count = len(circl_ids | threat_ids)
    if union_count:
        print(
            f"Overlap as % of all unique CVEs: "
            f"{100 * len(common_ids) / union_count:.2f}%"
        )

    # 6. Export unique ID lists
    pd.DataFrame(
        {"cve_id": sorted(common_ids)}
    ).to_csv(OUTPUT_DIR / "overlapping_cve_ids.csv", index=False)

    pd.DataFrame(
        {"cve_id": sorted(circl_only_ids)}
    ).to_csv(OUTPUT_DIR / "circl_only_cve_ids.csv", index=False)

    pd.DataFrame(
        {"cve_id": sorted(threat_only_ids)}
    ).to_csv(OUTPUT_DIR / "threatcluster_only_cve_ids.csv", index=False)

    # 7. Export overlapping records with original columns preserved
    circl_matched = circl_valid[
        circl_valid["cve_id"].isin(common_ids)
    ].copy()

    threat_matched = threat_valid[
        threat_valid["cve_id"].isin(common_ids)
    ].copy()

    # Validate that each dataset has one row per CVE before merging.
    # If duplicates exist, preserve them in separate files instead of
    # creating an accidental many-to-many merge.
    circl_matched.to_csv(
        OUTPUT_DIR / "circl_matching_records.csv", index=False
    )
    threat_matched.to_csv(
        OUTPUT_DIR / "threatcluster_matching_records.csv", index=False
    )

    if (
        not circl_matched["cve_id"].duplicated().any()
        and not threat_matched["cve_id"].duplicated().any()
    ):
        merged = circl_matched.merge(
            threat_matched,
            on="cve_id",
            how="inner",
            suffixes=("_circl", "_threat"),
            validate="one_to_one",
        )
        merged.to_csv(
            OUTPUT_DIR / "merged_matching_records.csv",
            index=False,
        )
        print("\nMerged matched records exported.")
    else:
        print(
            "\nSome CVE IDs have multiple rows. "
            "A one-to-one merge was skipped to avoid duplicating records."
        )
        print(
            "Review the separate matching-record CSV files before "
            "choosing a deduplication strategy."
        )

    # 8. Export a compact summary
    summary = pd.DataFrame(
        [
            {"category": "CIRCL unique CVEs", "count": len(circl_ids)},
            {
                "category": "ThreatCluster unique CVEs",
                "count": len(threat_ids),
            },
            {"category": "CVEs in both", "count": len(common_ids)},
            {"category": "CIRCL only", "count": len(circl_only_ids)},
            {
                "category": "ThreatCluster only",
                "count": len(threat_only_ids),
            },
        ]
    )

    summary.to_csv(OUTPUT_DIR / "comparison_summary.csv", index=False)

    print(f"\nAll outputs saved in: {OUTPUT_DIR.resolve()}")
    print("\nComparison completed successfully.")


if __name__ == "__main__":
    main()
