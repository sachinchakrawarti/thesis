# scripts/build_cti_dataset.py
"""Build the final CTI dataset with:
   - per-CVE doc_id
   - per-source length caps (CIRCL ≤ 100, ThreatCluster ≤ 50)
   - parent technique IDs only
   - rare-technique pruning
   - writes dataset/cti_dataset.csv
"""
import uuid
from pathlib import Path

import pandas as pd
from datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = PROJECT_ROOT / "dataset" / "cti_dataset.csv"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

MAX_ROWS = 30000
TARGET_YES = 12000       # was 6000 → doubled to keep 40% balance
MIN_LEN = 5
CIRCL_MAX_LEN = 100
TC_MAX_LEN = 50
MIN_TECH_FREQ = 8        # drop technique classes with < 8 examples


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def normalize_severity(sev):
    if not sev:
        return "MEDIUM"
    s = str(sev).strip().upper()
    return s if s in ("CRITICAL", "HIGH", "MEDIUM", "LOW") else "MEDIUM"


def parse_date(val):
    if val is None:
        return None
    try:
        ts = pd.to_datetime(val, errors="coerce")
        return None if pd.isna(ts) else ts.strftime("%Y-%m-%d")
    except Exception:
        return None


def safe_join(val):
    if val is None:
        return ""
    if isinstance(val, list):
        return "; ".join(str(v) for v in val if v)
    return str(val)


def relevance_reason(row):
    if bool(row.get("in_kev")):
        return True, "Actively Exploited (KEV)"
    if bool(row.get("has_exploit")) or (row.get("exploit_count") or 0) > 0:
        return True, "Public Exploit Available"
    if bool(row.get("ransomware_use")):
        return True, "Ransomware-Associated"
    if str(row.get("cvss_v3_severity", "")).strip().upper() == "CRITICAL":
        return True, "High-Risk Vulnerability"
    desc = str(row.get("description", "")).lower()
    keywords = [
        "remote code execution", "authentication bypass", "privilege escalation",
        "arbitrary code", "command injection", "sql injection",
        "deserialization", "path traversal", "buffer overflow", "use after free",
    ]
    if any(kw in desc for kw in keywords):
        return True, "High-Risk Vulnerability"
    return False, "Unclassified"


# ---------------------------------------------------------
# CIRCL — technique labels
# ---------------------------------------------------------
print("Downloading CIRCL ...")
ds_circl = load_dataset("CIRCL/vulnerability-attack-techniques", split="train")

circl_rows = []
for row in ds_circl:
    text = str(row.get("description", "")).strip()
    techs = row.get("techniques") or []
    if not text or not techs:
        continue

    cve_id = row.get("id", "") or f"CVE-UNKNOWN-{uuid.uuid4().hex[:8]}"
    year = cve_id.split("-")[1] if "CVE-" in str(cve_id) else None
    parent_tech = techs[0].split(".")[0]   # T1003.008 -> T1003

    circl_rows.append({
        "uid": f"circl_{cve_id}",
        "source": "circl",
        "doc_id": f"circl_{cve_id}",
        "page_number": 1,
        "sentence_id": 1,
        "context_before": "",
        "sentence_text": text,
        "relevant": "yes",
        "tactic": "",
        "techniques": parent_tech,
        "procedure": "CVE description",
        "entities": safe_join(row.get("affected_products")),
        "date": f"{year}-01-01" if year else None,
        "severity": "HIGH",
        "zero_day": "no",
        "topic_cluster": "Technique Labeled CVE",
    })

df_circl = pd.DataFrame(circl_rows)
print(f"  CIRCL rows: {len(df_circl)}")


# ---------------------------------------------------------
# ThreatCluster — relevance + dates
# ---------------------------------------------------------
print("Downloading ThreatCluster ...")
ds_tc = load_dataset("threatcluster/cve-exploitation-signals", split="train")

tc_rows = []
for row in ds_tc:
    text = str(row.get("description", "")).strip()
    if not text:
        continue

    cve_id = row.get("cve_id", "") or f"CVE-UNKNOWN-{uuid.uuid4().hex[:8]}"
    is_rel, topic = relevance_reason(row)
    date_val = parse_date(row.get("kev_added_date")) or parse_date(row.get("published_date"))
    entities = safe_join(row.get("affected_products")) or safe_join(row.get("affected_vendors"))

    tc_rows.append({
        "uid": f"tc_{cve_id}",
        "source": "threatcluster",
        "doc_id": f"tc_{cve_id}",
        "page_number": 1,
        "sentence_id": 1,
        "context_before": "",
        "sentence_text": text,
        "relevant": "yes" if is_rel else "no",
        "tactic": "",
        "techniques": "",
        "procedure": "CVE description",
        "entities": entities,
        "date": date_val,
        "severity": normalize_severity(row.get("cvss_v3_severity")),
        "zero_day": "no",
        "topic_cluster": topic,
    })

df_tc = pd.DataFrame(tc_rows)
print(f"  ThreatCluster rows: {len(df_tc)}")
print(f"  ThreatCluster YES: {(df_tc['relevant'] == 'yes').sum()}")
print(f"  ThreatCluster NO : {(df_tc['relevant'] == 'no').sum()}")


# ---------------------------------------------------------
# Length filtering — different cap for each source
# ---------------------------------------------------------
def filter_length(df, label, max_len):
    before = len(df)
    df = df.copy()
    df["_len"] = df["sentence_text"].astype(str).str.split().str.len()
    df = df[(df["_len"] >= MIN_LEN) & (df["_len"] <= max_len)].drop(columns=["_len"])
    after = len(df)
    print(f"  {label}: {before} -> {after} rows ({before - after} dropped)")
    return df.reset_index(drop=True)


print("Filtering length ...")
df_circl = filter_length(df_circl, "CIRCL", CIRCL_MAX_LEN)
df_tc    = filter_length(df_tc,    "ThreatCluster", TC_MAX_LEN)


# ---------------------------------------------------------
# Prune rare technique classes
# ---------------------------------------------------------
print(f"Pruning technique classes with < {MIN_TECH_FREQ} examples ...")
tech_counts = df_circl["techniques"].value_counts()
keep_techs = tech_counts[tech_counts >= MIN_TECH_FREQ].index
before = len(df_circl)
df_circl = df_circl[df_circl["techniques"].isin(keep_techs)].reset_index(drop=True)
print(f"  CIRCL: {before} -> {len(df_circl)} rows")
print(f"  Kept {len(keep_techs)} technique classes out of {len(tech_counts)}")


# ---------------------------------------------------------
# Balance
# ---------------------------------------------------------
print("Balancing ...")
df_tc_yes = df_tc[df_tc["relevant"] == "yes"]
df_tc_no  = df_tc[df_tc["relevant"] == "no"]

remaining_yes = max(0, TARGET_YES - len(df_circl))
df_tc_yes_sampled = df_tc_yes.sample(
    n=min(len(df_tc_yes), remaining_yes), random_state=42
)

df_yes = pd.concat([df_circl, df_tc_yes_sampled], ignore_index=True)
target_no = min(len(df_tc_no), MAX_ROWS - len(df_yes))
df_no_sampled = df_tc_no.sample(n=target_no, random_state=42)

df = pd.concat([df_yes, df_no_sampled], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

df["sentence_id"] = df.groupby("doc_id").cumcount() + 1


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------
COLS = [
    "uid", "source", "doc_id", "page_number", "sentence_id", "context_before",
    "sentence_text", "relevant", "tactic", "techniques", "procedure", "entities",
    "date", "severity", "zero_day", "topic_cluster",
]
df = df[COLS]
df.to_csv(OUT_PATH, index=False)


# ---------------------------------------------------------
# Report
# ---------------------------------------------------------
lengths = df["sentence_text"].astype(str).str.split().str.len()
techs = df.loc[df["techniques"] != "", "techniques"]

print("\n" + "=" * 55)
print("DATASET BUILT SUCCESSFULLY")
print("=" * 55)
print(f"Total Rows     : {len(df)}")
print(f"Documents      : {df['doc_id'].nunique()}")
print(f"Relevant (YES) : {(df['relevant'] == 'yes').sum()}")
print(f"Irrelevant (NO): {(df['relevant'] == 'no').sum()}")
print(f"With Techniques: {len(techs)}")
print(f"Distinct Techs : {techs.nunique()}")
print(f"With Dates     : {df['date'].notna().sum()}")
print(f"Date range     : {df['date'].min()}  to  {df['date'].max()}")
print(f"Mean length    : {lengths.mean():.1f} words  (range {lengths.min()}-{lengths.max()})")

print(f"\nTop 10 parent techniques:")
print(techs.value_counts().head(10).to_string())

print(f"\nRelevance reasons (YES only):")
print(df[df['relevant'] == 'yes']['topic_cluster'].value_counts().to_string())

print(f"\nSeverity distribution:")
print(df['severity'].value_counts().to_string())

print("=" * 55)
print(f"Saved to: {OUT_PATH}")