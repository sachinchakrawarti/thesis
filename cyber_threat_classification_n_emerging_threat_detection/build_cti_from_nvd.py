#!/usr/bin/env python3
"""
Build a 17-column CTI dataset (~10,000 CVEs, last ~3 years) directly from NVD.
No Hugging Face needed. Includes all CISA-KEV CVEs in range plus a random sample of the rest.

  pip install requests pandas
  python build_cti_from_nvd.py --rows 10000 --api-key YOUR_NVD_KEY --out cti_dataset.csv
Get a free NVD key: https://nvd.nist.gov/developers/request-an-api-key (without it, expect ~15-20 min).
techniques / threat_category / relevant / target are PROXY labels (rule-based), not analyst-verified.
"""

import argparse
import re
import time
from datetime import datetime, timedelta

import pandas as pd
import requests

NVD_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"
KEV_URL = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"
EPSS_URL = "https://epss.cyentia.com/epss_scores-current.csv.gz"

# ---- ATT&CK technique rules: (regex on description, technique ID, name) ----
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
    "CWE-798": ["T1078"], "CWE-287": ["T1078"],
    "CWE-269": ["T1068"],
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

# ---- threat category rules (first match wins) ----
CATEGORY_RULES = [
    (r"\bworm\b|self-propagat", "Worm"),
    (r"phish", "Phishing"),
    (r"malware|trojan|backdoor|botnet", "Malware"),
    (r"denial of service|crash", "DoS"),
    (r"execute arbitrary|remote code execution", "Remote Code Execution"),
    (r"privilege", "Privilege Escalation"),
    (r"sql injection|cross-site scripting|injection", "Injection"),
]


def parse_cve(c):
    desc = next((d["value"] for d in c.get("descriptions", []) if d["lang"] == "en"), "")
    sev, score = "", None
    for key in ("cvssMetricV31", "cvssMetricV40", "cvssMetricV30", "cvssMetricV2"):
        if c.get("metrics", {}).get(key):
            m = c["metrics"][key][0]["cvssData"]
            score = m.get("baseScore")
            sev = m.get("baseSeverity") or c["metrics"][key][0].get("baseSeverity", "")
            break
    cwes = sorted({
        d["value"] for w in c.get("weaknesses", []) for d in w["description"]
        if d["value"].startswith("CWE-")
    })
    return {
        "cve_id": c["id"],
        "sentence_text": " ".join(desc.split()),
        "source": "NVD",
        "published": c["published"][:10],
        "severity": (sev or "").upper(),
        "cvss_score": score,
        "cwe_ids": "; ".join(cwes),
    }


def map_techniques(desc, cwe_ids):
    techs = []
    for cwe in cwe_ids.split("; "):
        techs += CWE_TECH.get(cwe, [])
    for pattern, t in DESC_TECH:
        if re.search(pattern, desc, re.I):
            techs.append(t)
    return list(dict.fromkeys(techs))  # dedupe, keep order


def categorize(desc, kev_ransom):
    if kev_ransom == "Known":
        return "Ransomware", "CISA-KEV"
    for pattern, cat in CATEGORY_RULES:
        if re.search(pattern, desc, re.I):
            return cat, "keyword-rule"
    return "Other", "default"



import random
from datetime import date


def nvd_get(params, headers, delay):
    while True:
        r = requests.get(NVD_URL, params=params, headers=headers, timeout=60)
        if r.status_code in (403, 429, 503):
            time.sleep(30)
            continue
        r.raise_for_status()
        time.sleep(delay)
        return r.json()


def window_params(s, e, **extra):
    p = {"pubStartDate": s.strftime("%Y-%m-%dT00:00:00.000"),
         "pubEndDate": e.strftime("%Y-%m-%dT23:59:59.999")}
    p.update(extra)
    return p


def fetch_sampled(start, end, n_rows, api_key, seed):
    """Sample ~n_rows CVEs evenly across 30-day windows, plus all KEV CVEs in range."""
    rnd = random.Random(seed)
    headers = {"apiKey": api_key} if api_key else {}
    delay = 0.7 if api_key else 6.5
    s0, e0 = datetime.fromisoformat(start), datetime.fromisoformat(end)
    wins, cur = [], s0
    while cur < e0:
        nxt = min(cur + timedelta(days=29), e0)
        wins.append((cur, nxt))
        cur = nxt + timedelta(days=1)
    per_win = max(1, n_rows // len(wins))
    seen, rows = set(), []

    # 1) KEV CVEs first (NVD supports the hasKev filter)
    for s, e in wins:
        idx = 0
        while True:
            d = nvd_get(window_params(s, e, hasKev="", resultsPerPage=2000, startIndex=idx), headers, delay)
            for it in d.get("vulnerabilities", []):
                if it["cve"]["id"] not in seen:
                    seen.add(it["cve"]["id"]); rows.append(parse_cve(it["cve"]))
            idx += d.get("resultsPerPage", 0)
            if idx >= d.get("totalResults", 0) or d.get("resultsPerPage", 0) == 0:
                break
        print(f"KEV {s.date()}..{e.date()} total so far {len(rows)}")

    # 2) random slice from each window
    for s, e in wins:
        total = nvd_get(window_params(s, e, resultsPerPage=1), headers, delay).get("totalResults", 0)
        if total == 0:
            continue
        size = min(per_win + 30, total, 2000)
        start_idx = rnd.randint(0, max(total - size, 0))
        d = nvd_get(window_params(s, e, resultsPerPage=size, startIndex=start_idx), headers, delay)
        for it in d.get("vulnerabilities", []):
            if it["cve"]["id"] not in seen:
                seen.add(it["cve"]["id"]); rows.append(parse_cve(it["cve"]))
        print(f"{s.date()}..{e.date()} window total {total}, rows so far {len(rows)}")
    return rows



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=10000)
    ap.add_argument("--start", default="2023-10-24")
    ap.add_argument("--end", default="2026-10-22")
    ap.add_argument("--out", default="cti_dataset.csv")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    df = pd.DataFrame(fetch_sampled(a.start, a.end, a.rows, a.api_key, a.seed))
    df = df[df["sentence_text"].str.len() > 0].drop_duplicates("cve_id")
    # drop NVD placeholders for rejected CVEs
    df = df[~df["sentence_text"].str.contains(r"\*\* REJECT \*\*", regex=True)]
    if len(df) > a.rows:
        df = df.sample(n=a.rows, random_state=a.seed)  # may trim some KEV rows; see below
    df = df.sort_values("published").reset_index(drop=True)

    kev = requests.get(KEV_URL, timeout=60).json()["vulnerabilities"]
    kev_map = {k["cveID"]: k.get("knownRansomwareCampaignUse", "Unknown") for k in kev}
    df["in_kev"] = df["cve_id"].isin(kev_map)
    df["kev_ransomware"] = df["cve_id"].map(kev_map).fillna("Unknown")

    try:
        epss = pd.read_csv(EPSS_URL, compression="gzip", skiprows=1)[["cve", "epss"]]
        df = df.merge(epss, left_on="cve_id", right_on="cve", how="left").drop(columns="cve")
    except Exception as e:
        print("WARNING: EPSS download failed, epss left empty:", e)
        df["epss"] = None

    tl = [map_techniques(d, c) for d, c in zip(df["sentence_text"], df["cwe_ids"])]
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
    print("Date range:", df["published"].min(), "->", df["published"].max())


if __name__ == "__main__":
    main()