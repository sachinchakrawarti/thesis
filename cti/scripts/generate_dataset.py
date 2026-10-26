"""
Download the public SMAH-CTI JSON dataset and convert it to CSV.

Install:
    python -m pip install requests pandas

Run:
    python convert_smah_cti_to_csv.py

Output:
    SMAH-CTI.csv

The dataset contains sentence-level annotations. Nested fields (such as entities,
tactics, and techniques) are serialized as JSON strings so no annotation detail
is silently discarded.
"""
import json
import sys
from pathlib import Path

import pandas as pd
import requests

URL = "https://raw.githubusercontent.com/davidmohaisen/SMAH-CTI/main/data/SMAH-CTI.json"
OUT = Path("SMAH-CTI.csv")
JSON_FILE = Path("SMAH-CTI.json")


def download_json():
    print("Downloading SMAH-CTI from the official GitHub repository...")
    response = requests.get(URL, timeout=120)
    response.raise_for_status()
    JSON_FILE.write_bytes(response.content)
    print(f"Downloaded {len(response.content):,} bytes to {JSON_FILE.resolve()}")


def load_records(obj):
    # Handle common dataset layouts: list of records, {"data": [...]},
    # {"sentences": [...]}, or a dict whose values are records.
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for key in ("data", "sentences", "records", "instances", "examples"):
            if isinstance(obj.get(key), list):
                return obj[key]
        # Some JSON datasets wrap their rows under a single top-level key.
        list_values = [v for v in obj.values() if isinstance(v, list)]
        if len(list_values) == 1:
            return list_values[0]
        return [obj]
    raise ValueError(f"Unsupported JSON top-level type: {type(obj).__name__}")


def main():
    try:
        if not JSON_FILE.exists():
            download_json()

        with JSON_FILE.open("r", encoding="utf-8") as f:
            obj = json.load(f)

        records = load_records(obj)
        if not records:
            raise ValueError("The JSON file contains no records.")

        # Flatten nested dictionaries into columns. Lists remain JSON strings
        # to preserve multiple labels/entities without losing information.
        df = pd.json_normalize(records, sep=".")
        for column in df.columns:
            df[column] = df[column].apply(
                lambda value: json.dumps(value, ensure_ascii=False)
                if isinstance(value, (list, dict))
                else value
            )

        df.to_csv(OUT, index=False, encoding="utf-8-sig")
        print(f"Success: exported {len(df):,} rows and {len(df.columns):,} columns.")
        print(f"CSV saved to: {OUT.resolve()}")
        print("Note: this is the dataset's native annotation task; it does not automatically")
        print("provide the 15 threat categories from your thesis.")
    except (requests.RequestException, json.JSONDecodeError, OSError, ValueError) as exc:
        print(f"Conversion failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
