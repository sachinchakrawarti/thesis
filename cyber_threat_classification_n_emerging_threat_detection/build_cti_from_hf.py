#!/usr/bin/env python3
"""
Build a 17-column CVE/CTI dataset (~10,000 rows, last ~3 years) using Hugging Face data.

Base data : Hugging Face dataset  tandevllc/cve_dataset  (config "cve")
            -> cve_id, description text, published, severity, CVSS score, CWE ids
Enrichment: CISA KEV (official JSON)   -> in_kev, kev_ransomware
            FIRST EPSS (official CSV)  -> epss
Rules     : techniques / threat_category / relevant / target are PROXY labels (see below)

Setup:
  pip install datasets pandas requests
Run:
  python build_cti_from_hf.py --inspect          # first: look at the real columns
  python build_cti_from_hf.py --rows 10000 --out cti_dataset.csv
  # optional: make sure all KEV (exploited) CVEs in the window are included
  python build_cti_from_hf.py --rows 10000 --keep-all-kev --out cti_dataset.csv
"""
import argparse
import re

import pandas as pd
import requests
from datasets import load_dataset

HF_DATASET = "tandevllc/cve_dataset"
HF_CONFIG = "cve"
KEV_URL = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"
EPSS_URL = "https://epss.cyentia.com/epss_scores-current.csv.gz"

TECH_NAMES = {
    "T1059": "Command and Scripting Interpreter",
    "T1190": "Exploit Public-Facing Application",
    "T1068": "Exploitation for Privilege Escalation",
    "T1499": "Endpoint Denial of Service",
    "T1078": "Valid Accounts",
    "T1203": "Exploitation for Client Execution",
    "T1005": "Data from Local System",
}
CWE_TECH = {
    "CWE-78": ["T1059", "T1190"], "CWE-77": ["T1059", "T1190"],
    "CWE-94": ["T1059", "T1190"], "CWE-89": ["T1190"],
    "CWE-798": ["T1078"], "CWE-287": ["T1078"], "CWE-269": ["T1068"],
}
DESC_TECH = [
    (r"execute (arbitrary|remote) (code|commands?)|remote code execution", "T1059"),
    (r"remote attackers?|unauthenticated|web (application|interface|server)", "T1190"),
    (r"privilege escalation|elevate privileges|gain (root|elevated) privileges", "T1068"),
    (r"denial of service|crash|resource exhaustion", "T1499"),
    (r"hard-?coded (credentials|password)|default (credentials|password)", "T1078"),
    (r"open (a )?(malicious|crafted) (file|document)|user interaction", "T1203"),
    (r"information disclosure|read (arbitrary|sensitive) files?", "T1005"),
]
CATEGORY_RULES = [
    (r"\bworm\b|self-propagat", "Worm"),
    (r"phish", "Phishing"),
    (r"malware|trojan|backdoor|botnet", "Malware"),
    (r"denial of service|crash", "DoS"),
    (r"execute arbitrary|remote code execution", "Remote Code Execution"),
    (r"privilege", "Privilege Escalation"),
    (r"sql injection|cross-site scripting|injection", "Injection"),
]


def load_base(max_scan):
    """Stream the HF dataset and flatten the nested meta struct."""
    ds = load_dataset(HF_DATASET, HF_CONFIG, split="train", streaming=True)
    rows = []
    for i, ex in enumerate(ds):
        if max_scan and i >= max_scan:
            break
        m = ex.get("meta") or {}
        cv = m.get("cvss") or {}
        rows.append({
            "cve_id": m.get("cve_id"),
            "text": ex.get("text", ""),
            "published": m.get("published"),
            "severity": cv.get("severity"),
            "cvss_score": cv.get("baseScore"),
            "cwes": m.get("cwes") or [],
        })
    return pd.DataFrame(rows)


def clean_text(text, cve_id):
    t = " ".join(str(text).split())
    if cve_id and t.upper().startswith(str(cve_id).upper()):
        t = t[len(cve_id):].lstrip(" :-–—")
    return t


def map_techniques(desc, cwe_list):
    techs = []
    for cwe in cwe_list:
        techs += CWE_TECH.get(cwe, [])
    for pattern, t in DESC_TECH:
        if re.search(pattern, desc, re.I):
            techs.append(t)
    return list(dict.fromkeys(techs))


def categorize(desc, kev_ransom):
    if kev_ransom == "Known":
        return "Ransomware", "CISA-KEV"
    for pattern, cat in CATEGORY_RULES:
        if re.search(pattern, desc, re.I):
            return cat, "keyword-rule"
    return "Other", "default"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=10000)
    ap.add_argument("--start", default="2023-10-01")
    ap.add_argument("--end", default="2026-10-23")
    ap.add_argument("--out", default="cti_dataset.csv")
    ap.add_argument("--max-scan", type=int, default=None, help="only read first N HF rows (testing)")
    ap.add_argument("--keep-all-kev", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--inspect", action="store_true", help="print HF schema + date range and exit")
    a = ap.parse_args()

    if a.inspect:
        ds = load_dataset(HF_DATASET, HF_CONFIG, split="train", streaming=True)
        for i, ex in enumerate(ds):
            print(ex)
            if i >= 2:
                break
        return

    df = load_base(a.max_scan)
    print(f"Loaded {len(df)} CVEs from Hugging Face")
    df["published"] = pd.to_datetime(df["published"], errors="coerce", utc=True)
    print("Published range in HF data:", df["published"].min(), "->", df["published"].max())
    df = df.dropna(subset=["cve_id", "published"]).drop_duplicates("cve_id")
    lo, hi = pd.Timestamp(a.start, tz="UTC"), pd.Timestamp(a.end, tz="UTC") + pd.Timedelta(days=1)
    df = df[(df["published"] >= lo) & (df["published"] < hi)].copy()
    print(f"{len(df)} CVEs inside {a.start}..{a.end}")
    if df.empty:
        raise SystemExit("No rows in that date window: the HF dataset may end earlier. "
                         "Change --start/--end to match the printed range.")

    df["sentence_text"] = [clean_text(t, c) for t, c in zip(df["text"], df["cve_id"])]
    df["published"] = df["published"].dt.strftime("%Y-%m-%d")
    df["severity"] = df["severity"].fillna("").astype(str).str.upper()
    df["cwe_ids"] = df["cwes"].apply(lambda l: "; ".join(sorted(x for x in l if str(x).startswith("CWE-"))))
    df["source"] = "NVD"

    # CISA KEV (required: without it in_kev would be wrong)
    kev = requests.get(KEV_URL, timeout=60).json()["vulnerabilities"]
    kev_map = {k["cveID"]: k.get("knownRansomwareCampaignUse", "Unknown") for k in kev}
    df["in_kev"] = df["cve_id"].isin(kev_map)
    df["kev_ransomware"] = df["cve_id"].map(kev_map).fillna("Unknown")

    # Sample to the requested size
    if len(df) > a.rows:
        if a.keep_all_kev:
            kev_df = df[df["in_kev"]]
            rest = df[~df["in_kev"]].sample(n=max(a.rows - len(kev_df), 0), random_state=a.seed)
            df = pd.concat([kev_df, rest]).head(a.rows)
        else:
            df = df.sample(n=a.rows, random_state=a.seed)
    df = df.sort_values("published").reset_index(drop=True)

    # EPSS
    try:
        epss = pd.read_csv(EPSS_URL, compression="gzip", skiprows=1)[["cve", "epss"]]
        df = df.merge(epss, left_on="cve_id", right_on="cve", how="left").drop(columns="cve")
    except Exception as e:
        print("WARNING: EPSS download failed, epss left empty:", e)
        df["epss"] = None

    # Proxy labels
    tl = [map_techniques(d, c.split("; ") if c else []) for d, c in zip(df["sentence_text"], df["cwe_ids"])]
    df["techniques"] = ["; ".join(t) for t in tl]
    df["primary_tech"] = [t[0] if t else "" for t in tl]
    df["tech_name"] = df["primary_tech"].map(TECH_NAMES).fillna("")
    cats = [categorize(d, k) for d, k in zip(df["sentence_text"], df["kev_ransomware"])]
    df["threat_category"] = [c[0] for c in cats]
    df["threat_category_source"] = [c[1] for c in cats]
    df["relevant"] = (df["in_kev"] | (df["cvss_score"].fillna(0) >= 7.0)).map({True: "yes", False: "no"})
    df["target"] = df["in_kev"].map({True: "yes", False: "no"})

    cols = ["cve_id", "sentence_text", "source", "published", "threat_category", "relevant",
            "severity", "cvss_score", "techniques", "primary_tech", "tech_name", "cwe_ids",
            "in_kev", "kev_ransomware", "epss", "target", "threat_category_source"]
    df[cols].to_csv(a.out, index=False)
    print(f"Saved {len(df)} rows to {a.out}")
    print(df["target"].value_counts())
    print("Rows with no technique:", (df["techniques"] == "").sum())


if __name__ == "__main__":
    main()