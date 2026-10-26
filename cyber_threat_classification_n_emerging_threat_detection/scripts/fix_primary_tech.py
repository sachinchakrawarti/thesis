
from pathlib import Path
import pandas as pd

# Project paths
INPUT_FILE = Path("data/processed/cti_unified.csv")
OUTPUT_FILE = Path("data/processed/cti_unified_repaired.csv")

def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE, keep_default_na=False)

    required_columns = {"cve_id", "techniques", "tech_name"}
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Normalize technique IDs and derive the first recorded ID.
    techniques = df["techniques"].fillna("").astype(str).str.strip()

    df["primary_tech"] = (
        techniques.str.split(";").str[0].str.strip()
    )
    df.loc[techniques.eq(""), "primary_tech"] = ""

    # Validate IDs against the expected MITRE ATT&CK format.
    valid_id = df["primary_tech"].str.fullmatch(
        r"T\d{4}(?:\.\d{3})?"
    )
    invalid = df["primary_tech"].ne("") & ~valid_id

    if invalid.any():
        examples = df.loc[
            invalid, ["cve_id", "techniques", "primary_tech"]
        ].head(10)
        raise ValueError(
            "Invalid primary technique IDs found:\n"
            + examples.to_string(index=False)
        )

    # Save separately so the original dataset remains unchanged.
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print("Repair completed.")
    print(f"Input rows: {len(df):,}")
    print(f"Rows with technique mappings: {techniques.ne('').sum():,}")
    print(
        "Rows with primary_tech: "
        f"{df['primary_tech'].ne('').sum():,}"
    )
    print(f"Output: {OUTPUT_FILE}")

    print("\nSample:")
    print(
        df.loc[
            df["primary_tech"].ne(""),
            ["cve_id", "techniques", "primary_tech", "tech_name"],
        ].head(10).to_string(index=False)
    )

if __name__ == "__main__":
    main()
