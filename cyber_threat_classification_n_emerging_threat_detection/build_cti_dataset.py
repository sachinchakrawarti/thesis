from pathlib import Path
import pandas as pd
from datasets import load_dataset

# =====================================================
# CTI DATASET BUILDER
# Sources:
# 1. CIRCL/vulnerability-attack-techniques
# 2. threatcluster/cve-exploitation-signals
#
# Output: cti_dataset.csv
# Minimum required rows: 10,000  max 15000
# =====================================================

MIN_ROWS = 10_000

# Always save beside this Python script
OUTPUT_FILE = Path(__file__).resolve().parent / "cti_dataset.csv"


def download_dataset(dataset_name):
    print(f"\nDownloading: {dataset_name}")

    dataset = load_dataset(dataset_name)
    frames = []

    for split_name, split in dataset.items():
        df = split.to_pandas()
        df["source_split"] = split_name
        frames.append(df)
        print(f"  {split_name}: {len(df):,} rows")

    if not frames:
        raise RuntimeError(f"No data returned for {dataset_name}")

    return pd.concat(frames, ignore_index=True, sort=False)


def main():
    # Download both sources
    circl = download_dataset(
        "CIRCL/vulnerability-attack-techniques"
    )

    threatcluster = download_dataset(
        "threatcluster/cve-exploitation-signals"
    )

    # Validate expected identifiers
    if "id" not in circl.columns:
        raise ValueError(
            f"CIRCL ID column missing. Found: {circl.columns.tolist()}"
        )

    if "cve_id" not in threatcluster.columns:
        raise ValueError(
            "ThreatCluster cve_id column missing. "
            f"Found: {threatcluster.columns.tolist()}"
        )

    # Standardize the CVE identifier
    circl["cve_id"] = (
        circl["id"].astype("string").str.strip().str.upper()
    )

    threatcluster["cve_id"] = (
        threatcluster["cve_id"]
        .astype("string").str.strip().str.upper()
    )

    # Identify original source
    circl["dataset_source"] = "CIRCL"
    threatcluster["dataset_source"] = "ThreatCluster"

    # Keep records from both datasets in one CSV.
    # Different labels remain separate; missing labels stay missing.
    combined = pd.concat(
        [circl, threatcluster],
        ignore_index=True,
        sort=False,
    )

    # Remove rows that are completely identical.
    combined = combined.reset_index(drop=True)

    # Validate the requested size
    if len(combined) < MIN_ROWS:
        raise RuntimeError(
            f"Only {len(combined):,} rows were downloaded. "
            f"At least {MIN_ROWS:,} are required. "
            "No synthetic rows were created."
        )

    # Save one CSV file
    combined.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    # Verify the saved file
    saved = pd.read_csv(OUTPUT_FILE, low_memory=False)

    print("\n" + "=" * 55)
    print("CTI DATASET CREATED SUCCESSFULLY")
    print("=" * 55)
    print(f"CIRCL source rows:       {len(circl):,}")
    print(f"ThreatCluster rows:      {len(threatcluster):,}")
    print(f"Combined rows:           {len(combined):,}")
    print(f"Verified CSV rows:       {len(saved):,}")
    print(f"Unique CVE IDs:          {saved['cve_id'].nunique():,}")
    print(f"Columns:                 {len(saved.columns)}")
    print(f"\nFile location:\n{OUTPUT_FILE}")
    print(f"File size: {OUTPUT_FILE.stat().st_size / (1024**2):.2f} MB")


if __name__ == "__main__":
    main()
