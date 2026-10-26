
"""
Unified CTI Dataset Builder
M.Tech Thesis: Temporal Cyber Threat Intelligence Using
Machine Learning for Emerging Threat Detection

Sources:
  1. threatcluster/cve-exploitation-signals
  2. CIRCL/vulnerability-attack-techniques
  3. FIRST EPSS
  4. MITRE ATT&CK Enterprise

Outputs:
  data/processed/cti_unified.csv
  data/processed/cti_unified.parquet
  data/processed/cti_missingness_report.csv
  data/processed/cti_build_summary.json

Install:
  pip install pandas requests datasets pyarrow
"""

import gzip
import json
import re
import time
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from datasets import load_dataset


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
OUTPUT_DIR = BASE_DIR / "data" / "processed"

RAW_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

THREATCLUSTER_DATASET = "threatcluster/cve-exploitation-signals"
CIRCL_DATASET = "CIRCL/vulnerability-attack-techniques"

EPSS_URL = (
    "https://epss.empiricalsecurity.com/"
    "epss_scores-current.csv.gz"
)

MITRE_URL = (
    "https://raw.githubusercontent.com/mitre-attack/"
    "attack-stix-data/master/enterprise-attack/"
    "enterprise-attack.json"
)

OUTPUT_CSV = OUTPUT_DIR / "cti_unified.csv"
OUTPUT_PARQUET = OUTPUT_DIR / "cti_unified.parquet"
OUTPUT_MISSINGNESS = OUTPUT_DIR / "cti_missingness_report.csv"
OUTPUT_SUMMARY = OUTPUT_DIR / "cti_build_summary.json"

EPSS_CACHE = RAW_DIR / "epss_scores-current.csv.gz"
MITRE_CACHE = RAW_DIR / "enterprise-attack.json"

SCHEMA = [
    "cve_id",
    "sentence_text",
    "source",
    "published",
    "relevant",
    "severity",
    "cvss_score",
    "techniques",
    "primary_tech",
    "tech_name",
    "cwe_ids",
    "in_kev",
    "kev_ransomware",
    "epss",
]

USER_AGENT = "CTI-Thesis-Dataset-Builder/1.1"


# ============================================================
# HTTP SESSION AND DOWNLOADS
# ============================================================

def create_session():
    session = requests.Session()

    retries = Retry(
        total=5,
        connect=5,
        read=3,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        respect_retry_after_header=True,
    )

    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    session.headers.update({
        "User-Agent": USER_AGENT,
        "Accept": "*/*",
    })

    return session


SESSION = create_session()


def download_file(url, destination):
    """Download with retries and avoid replacing files with partial data."""
    destination = Path(destination)
    temporary = destination.with_suffix(destination.suffix + ".part")

    destination.parent.mkdir(parents=True, exist_ok=True)

    try:
        print(f"Downloading: {url}")

        with SESSION.get(
            url,
            timeout=(30, 180),
            stream=True,
            allow_redirects=True,
        ) as response:
            response.raise_for_status()

            with open(temporary, "wb") as output:
                for chunk in response.iter_content(256 * 1024):
                    if chunk:
                        output.write(chunk)

        if not temporary.exists() or temporary.stat().st_size == 0:
            raise RuntimeError("Downloaded file is empty.")

        temporary.replace(destination)

        print(f"Saved: {destination}")
        print(f"Size: {destination.stat().st_size:,} bytes")

    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


# ============================================================
# SAFE VALUE HELPERS
# ============================================================

def is_missing(value):
    """Check scalar missing values without failing on lists/arrays."""
    if value is None:
        return True

    if isinstance(value, str):
        return not value.strip() or value.strip().lower() in {
            "nan", "none", "null", "nat"
        }

    try:
        result = pd.isna(value)

        # pd.isna(list/array) can return an array.
        if getattr(result, "ndim", 0) != 0:
            return False

        return bool(result)

    except (TypeError, ValueError):
        return False


def clean_text(value):
    """Safely handle strings, lists, NumPy arrays and dictionaries."""
    if is_missing(value):
        return ""

    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, default=str)

    if isinstance(value, (list, tuple, set)):
        items = []
        for item in value:
            text = clean_text(item)
            if text:
                items.append(text)
        return "; ".join(items)

    # Handle NumPy arrays without requiring NumPy explicitly.
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        try:
            converted = value.tolist()
            if converted is not value:
                return clean_text(converted)
        except (TypeError, ValueError):
            pass

    return str(value).strip()


def first_value(row, candidates, default=None):
    """Find the first non-missing value in the candidate columns."""
    for column in candidates:
        if column not in row.index:
            continue

        value = row[column]

        if not is_missing(value):
            return value

    return default


def clean_cve(value):
    text = clean_text(value)

    match = re.search(
        r"CVE-\d{4}-\d{4,}",
        text,
        flags=re.IGNORECASE,
    )

    return match.group(0).upper() if match else None


def normalise_id_list(value):
    """Normalise IDs from strings, lists, arrays, or dictionaries."""
    if is_missing(value):
        return []

    if isinstance(value, dict):
        candidates = list(value.values())
    elif isinstance(value, (list, tuple, set)):
        candidates = list(value)
    elif hasattr(value, "tolist") and not isinstance(value, str):
        try:
            candidates = value.tolist()
            if not isinstance(candidates, list):
                candidates = [candidates]
        except (TypeError, ValueError):
            candidates = [value]
    else:
        candidates = re.split(r"[;,|\s]+", str(value))

    result = []

    for item in candidates:
        text = clean_text(item)

        if not text:
            continue

        # Flatten nested lists represented as strings where practical.
        for token in re.split(r"[;,|\s]+", text):
            token = token.strip()

            if token and token.lower() not in {
                "nan", "none", "null"
            }:
                if token not in result:
                    result.append(token)

    return result


def as_bool(value):
    if is_missing(value):
        return None

    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        if value == 1:
            return True
        if value == 0:
            return False

    text = clean_text(value).lower()

    if text in {"true", "yes", "y", "1"}:
        return True

    if text in {"false", "no", "n", "0"}:
        return False

    return None


def to_number(value):
    if is_missing(value):
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# ============================================================
# LOAD HUGGING FACE DATASETS
# ============================================================

def load_hf_dataframe(dataset_name):
    print(f"Loading Hugging Face dataset: {dataset_name}")

    dataset = load_dataset(dataset_name)
    frames = []

    for split_name in dataset.keys():
        frame = dataset[split_name].to_pandas()
        frame["_split"] = split_name
        frames.append(frame)

        print(f"  Split {split_name}: {len(frame):,} rows")

    if not frames:
        raise RuntimeError(f"No splits found for {dataset_name}")

    result = pd.concat(frames, ignore_index=True, sort=False)
    result.columns = [str(c).strip() for c in result.columns]

    print(f"  Total rows: {len(result):,}")
    print(f"  Columns: {result.columns.tolist()}")

    return result


# ============================================================
# MITRE ATT&CK TECHNIQUE LOOKUP
# ============================================================

def load_mitre_techniques():
    try:
        if not MITRE_CACHE.exists() or MITRE_CACHE.stat().st_size == 0:
            download_file(MITRE_URL, MITRE_CACHE)

        with open(MITRE_CACHE, "r", encoding="utf-8") as file:
            bundle = json.load(file)

    except Exception as exc:
        print(f"WARNING: MITRE ATT&CK could not be loaded: {exc}")
        print("Technique names will remain blank when unavailable.")
        return {}

    lookup = {}

    for item in bundle.get("objects", []):
        if item.get("type") != "attack-pattern":
            continue

        if item.get("revoked") or item.get("x_mitre_deprecated"):
            continue

        technique_name = item.get("name")

        for reference in item.get("external_references", []):
            if reference.get("source_name") != "mitre-attack":
                continue

            external_id = reference.get("external_id")

            if external_id and technique_name:
                lookup[external_id] = technique_name

    print(f"MITRE technique names loaded: {len(lookup):,}")
    return lookup


# ============================================================
# FIRST EPSS
# ============================================================

def load_epss():
    try:
        download_file(EPSS_URL, EPSS_CACHE)

        try:
            with gzip.open(
                EPSS_CACHE, "rt", encoding="utf-8"
            ) as file:
                frame = pd.read_csv(file, comment="#")

        except (gzip.BadGzipFile, OSError):
            # Handle a server returning plain CSV instead of gzip.
            frame = pd.read_csv(EPSS_CACHE, comment="#")

        frame.columns = [
            str(column).strip().lower()
            for column in frame.columns
        ]

        if not {"cve", "epss"}.issubset(frame.columns):
            raise ValueError(
                f"Unexpected EPSS columns: {frame.columns.tolist()}"
            )

        frame["cve_id"] = frame["cve"].apply(clean_cve)
        frame["epss"] = pd.to_numeric(
            frame["epss"], errors="coerce"
        )

        frame = frame.dropna(subset=["cve_id"])
        frame = frame.drop_duplicates(
            subset=["cve_id"], keep="last"
        )

        result = frame[["cve_id", "epss"]].copy()

        print(f"EPSS records loaded: {len(result):,}")
        return result

    except Exception as exc:
        print(f"WARNING: EPSS download/parsing failed: {exc}")
        print("Continuing with missing EPSS values.")
        return pd.DataFrame(columns=["cve_id", "epss"])


# ============================================================
# THREATCLUSTER NORMALISATION
# ============================================================

def normalise_threatcluster(frame):
    records = []

    for _, row in frame.iterrows():
        cve_id = clean_cve(
            first_value(row, ["cve_id", "cve", "id"])
        )

        if not cve_id:
            continue

        description = clean_text(
            first_value(row, ["description", "summary"], "")
        )

        published = first_value(
            row,
            ["published_date", "published", "published_at"],
        )

        severity = clean_text(
            first_value(
                row,
                ["cvss_v3_severity", "severity"],
                "",
            )
        )

        cvss_score = to_number(
            first_value(
                row,
                ["cvss_v3_score", "cvss_score", "cvss"],
            )
        )

        cwe_ids = normalise_id_list(
            first_value(row, ["cwe_ids", "cwes", "cwe"], [])
        )

        in_kev = as_bool(
            first_value(row, ["in_kev", "known_exploited"])
        )

        ransomware = as_bool(
            first_value(row, ["ransomware_use", "kev_ransomware"])
        )

        has_exploit = as_bool(
            first_value(row, ["has_exploit"])
        )

        # Proxy label only. Unknown signals remain unlabeled.
        if in_kev is True or has_exploit is True:
            relevant = "yes"
        elif in_kev is False and has_exploit is False:
            relevant = "no"
        else:
            relevant = None

        records.append({
            "cve_id": cve_id,
            "sentence_text": description,
            "source": "threatcluster",
            "published": clean_text(published) or None,
            "relevant": relevant,
            "severity": severity or None,
            "cvss_score": cvss_score,
            "techniques": "",
            "primary_tech": "",
            "tech_name": "",
            "cwe_ids": ";".join(cwe_ids),
            "in_kev": in_kev,
            "kev_ransomware": ransomware,
            "epss": None,
        })

    result = pd.DataFrame(records, columns=SCHEMA)
    print(f"Normalised ThreatCluster rows: {len(result):,}")

    return result


# ============================================================
# CIRCL NORMALISATION
# ============================================================

def normalise_circl(frame, technique_lookup):
    records = []

    for _, row in frame.iterrows():
        cve_id = clean_cve(
            first_value(row, ["id", "cve_id", "cve"])
        )

        if not cve_id:
            continue

        # Use description first; fall back to title.
        description = clean_text(
            first_value(row, ["description", "title"], "")
        )

        # Curated techniques preferred; derived mappings are fallback.
        technique_ids = normalise_id_list(
            first_value(row, ["techniques"], [])
        )

        technique_ids = [
            item for item in technique_ids
            if re.fullmatch(r"T\d{4}(?:\.\d{3})?", item)
        ]

        if not technique_ids:
            derived_ids = normalise_id_list(
                first_value(row, ["techniques_derived"], [])
            )

            technique_ids = [
                item for item in derived_ids
                if re.fullmatch(r"T\d{4}(?:\.\d{3})?", item)
            ]

        technique_names = [
            technique_lookup[technique_id]
            for technique_id in technique_ids
            if technique_id in technique_lookup
        ]

        cwe_ids = normalise_id_list(
            first_value(row, ["cwes", "cwe_ids", "cwe"], [])
        )

        records.append({
            "cve_id": cve_id,
            "sentence_text": description,
            "source": "circl",
            "published": None,
            "relevant": None,
            "severity": None,
            "cvss_score": None,
            "techniques": ";".join(technique_ids),
            # Do not mistake primary_impact for an ATT&CK technique ID.
            "primary_tech": "",
            "tech_name": ";".join(technique_names),
            "cwe_ids": ";".join(cwe_ids),
            "in_kev": None,
            "kev_ransomware": None,
            "epss": None,
        })

    result = pd.DataFrame(records, columns=SCHEMA)
    print(f"Normalised CIRCL rows: {len(result):,}")

    return result


# ============================================================
# MERGE RECORDS
# ============================================================

def combine_unique(values):
    result = []

    for value in values:
        if is_missing(value):
            continue

        text = clean_text(value)

        if not text:
            continue

        for item in text.split(";"):
            item = item.strip()

            if item and item not in result:
                result.append(item)

    return ";".join(result) if result else None


def merge_cve_rows(frame):
    """
    Merge records sharing a CVE ID.
    Preserve missing values as missing; don't turn unknown into false.
    """
    rows = []

    for cve_id, group in frame.groupby("cve_id", sort=False):
        record = {"cve_id": cve_id}

        for column in SCHEMA:
            if column == "cve_id":
                continue

            values = [
                value for value in group[column].tolist()
                if not is_missing(value)
            ]

            if not values:
                record[column] = None
                continue

            if column in {
                "sentence_text", "source", "published",
                "relevant", "severity", "cvss_score",
                "in_kev", "kev_ransomware", "epss",
            }:
                # Choose the first known value.
                # For relevance, prefer a non-null proxy label.
                record[column] = values[0]
            else:
                record[column] = combine_unique(values)

        rows.append(record)

    return pd.DataFrame(rows, columns=SCHEMA)


# ============================================================
# MISSINGNESS REPORT
# ============================================================

def create_missingness_report(frame):
    records = []

    for column in frame.columns:
        missing_mask = frame[column].apply(is_missing)
        missing_count = int(missing_mask.sum())

        records.append({
            "column": column,
            "missing_count": missing_count,
            "total_rows": len(frame),
            "missing_percent": round(
                100 * missing_count / max(len(frame), 1), 2
            ),
        })

    return pd.DataFrame(records)


# ============================================================
# MAIN
# ============================================================

def main():
    started = time.time()

    print("=" * 68)
    print("UNIFIED CTI DATASET BUILDER")
    print("=" * 68)

    print("\n[1/6] Loading ThreatCluster...")
    threatcluster = load_hf_dataframe(THREATCLUSTER_DATASET)

    print("\n[2/6] Loading CIRCL...")
    circl = load_hf_dataframe(CIRCL_DATASET)

    print("\n[3/6] Loading MITRE ATT&CK...")
    technique_lookup = load_mitre_techniques()

    print("\n[4/6] Loading FIRST EPSS...")
    epss = load_epss()

    print("\n[5/6] Normalising and merging...")
    tc = normalise_threatcluster(threatcluster)
    cc = normalise_circl(circl, technique_lookup)

    combined = pd.concat(
        [tc, cc],
        ignore_index=True,
        sort=False,
    )

    combined["cve_id"] = combined["cve_id"].apply(clean_cve)
    combined = combined.dropna(subset=["cve_id"])

    # Remove placeholder EPSS column before enrichment.
    combined = combined.drop(columns=["epss"], errors="ignore")

    if not epss.empty:
        combined = combined.merge(
            epss,
            on="cve_id",
            how="left",
            validate="many_to_one",
        )
    else:
        combined["epss"] = None

    unified = merge_cve_rows(combined)

    # Validate output columns.
    unified = unified[SCHEMA].copy()

    # Parse dates but preserve unparseable dates as missing.
    unified["published"] = pd.to_datetime(
        unified["published"],
        errors="coerce",
        utc=True,
    ).dt.strftime("%Y-%m-%d")

    unified["cvss_score"] = pd.to_numeric(
        unified["cvss_score"], errors="coerce"
    )

    unified["epss"] = pd.to_numeric(
        unified["epss"], errors="coerce"
    )

    # Save files.
    unified.to_csv(
        OUTPUT_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    unified.to_parquet(
        OUTPUT_PARQUET,
        index=False,
    )

    missingness = create_missingness_report(unified)
    missingness.to_csv(OUTPUT_MISSINGNESS, index=False)

    summary = {
        "dataset_name": "Unified CTI Dataset",
        "threatcluster_rows_loaded": int(len(threatcluster)),
        "circl_rows_loaded": int(len(circl)),
        "unified_rows": int(len(unified)),
        "unique_cves": int(unified["cve_id"].nunique()),
        "columns": SCHEMA,
        "rows_with_epss": int(unified["epss"].notna().sum()),
        "rows_with_relevance_label": int(
            unified["relevant"].notna().sum()
        ),
        "rows_with_techniques": int(
            unified["techniques"].apply(
                lambda value: not is_missing(value)
            ).sum()
        ),
        "rows_with_published_date": int(
            unified["published"].notna().sum()
        ),
        "mitre_technique_names_loaded": len(technique_lookup),
        "elapsed_seconds": round(time.time() - started, 2),
        "label_definition": (
            "Proxy label: yes if in_kev or has_exploit is true; "
            "no if both are explicitly false; otherwise missing."
        ),
        "epss_note": (
            "Current EPSS snapshot, not a historical daily time series."
        ),
    }

    with open(OUTPUT_SUMMARY, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    print("\n[6/6] COMPLETE")
    print("=" * 68)
    print(f"ThreatCluster source rows: {len(threatcluster):,}")
    print(f"CIRCL source rows:         {len(circl):,}")
    print(f"Unified unique CVEs:       {unified['cve_id'].nunique():,}")
    print(f"Output rows:               {len(unified):,}")
    print(f"Output columns:            {len(unified.columns)}")
    print(f"Rows with EPSS:            {unified['epss'].notna().sum():,}")
    print(
        "Rows with relevance labels: "
        f"{unified['relevant'].notna().sum():,}"
    )
    print(
        "Rows with technique IDs:   "
        f"{unified['techniques'].apply(lambda x: not is_missing(x)).sum():,}"
    )

    print("\nOUTPUT FILES")
    print(f"CSV:          {OUTPUT_CSV}")
    print(f"Parquet:      {OUTPUT_PARQUET}")
    print(f"Missingness:  {OUTPUT_MISSINGNESS}")
    print(f"Summary:      {OUTPUT_SUMMARY}")
    print(f"Elapsed:      {time.time() - started:.1f} seconds")

    print("\nRESEARCH WARNINGS")
    print("- 'relevant' is a proxy label, not verified ground truth.")
    print("- Exclude label-defining exploitation fields from model inputs.")
    print("- Missing ATT&CK labels do not mean no attack technique.")
    print("- Current EPSS scores alone cannot establish historical trends.")
    print("- Verify date coverage and missingness before temporal evaluation.")

    SESSION.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\nPIPELINE FAILED")
        print(f"{type(exc).__name__}: {exc}")
        raise
