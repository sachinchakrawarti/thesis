"""
Build artifacts/doc_dates.csv from APTnotes metadata + manual entries.
Output: document_id,date (YYYY-MM-DD)

Run from project root:  python scripts/make_doc_dates.py
Re-run after adding manual dates to fill gaps.
"""
import pandas as pd, os
from pathlib import Path

os.makedirs("artifacts", exist_ok=True)

# ─────────────────────────────────────────────────────────
# 1. Unique document_ids in training set
# ─────────────────────────────────────────────────────────
train = pd.read_csv("artifacts/train.csv")
doc_ids = sorted(train["document_id"].astype(str).unique().tolist())
print(f"Documents in SMAH-CTI train: {len(doc_ids)}")

# ─────────────────────────────────────────────────────────
# 2. Try multiple APTnotes mirrors
# ─────────────────────────────────────────────────────────
APTNOTES_URLS = [
    "https://raw.githubusercontent.com/aptnotes/data/master/APTnotes.csv",
    "https://raw.githubusercontent.com/aptnotes/data/main/APTnotes.csv",
    "https://raw.githubusercontent.com/kbandla/APTnotes/master/APTnotes.csv",  # may be 404
    "https://raw.githubusercontent.com/aptnotes/data/master/aptnotes.csv",
]

apt = pd.DataFrame()
for url in APTNOTES_URLS:
    try:
        apt = pd.read_csv(url)
        print(f"APTnotes loaded from: {url}  ({len(apt)} rows)")
        print(f"  columns: {list(apt.columns)}")
        break
    except Exception as e:
        print(f"  failed: {url}")
        print(f"    reason: {type(e).__name__}: {e}")

if apt.empty:
    print("\n⚠  All APTnotes mirrors failed.")
    print("   You can still proceed:")
    print("     (a) Add manual dates below, OR")
    print("     (b) Skip temporal analysis in Notebook 5 (it already prints a")
    print("         clean SKIP message when doc_dates.csv is absent).")

# ─────────────────────────────────────────────────────────
# 3. Build filename → date map
# ─────────────────────────────────────────────────────────
apt_dates = {}
if not apt.empty:
    # Some mirrors use "Filename"; others use "Title", "File", "name", "URL"
    fn_candidates = ["Filename", "File", "Filename.pdf", "name", "Title"]
    fn_col = next((c for c in fn_candidates if c in apt.columns), None)

    date_candidates = ["Year", "year", "Date", "date", "Published"]
    date_col = next((c for c in date_candidates if c in apt.columns), None)

    if fn_col and date_col:
        print(f"Using filename column='{fn_col}', date column='{date_col}'")
        for _, row in apt.iterrows():
            fn = str(row.get(fn_col, "")).strip()
            if not fn:
                continue
            stem = (fn.replace(".pdf", "").replace(".PDF", "")
                      .replace(".doc", "").strip())
            raw_date = row.get(date_col)
            if pd.isna(raw_date):
                continue
            # If date is just a year, make it YYYY-01-01
            try:
                if isinstance(raw_date, (int, float)) or str(raw_date).isdigit():
                    apt_dates[stem] = f"{int(raw_date)}-01-01"
                else:
                    dt = pd.to_datetime(raw_date, errors="coerce")
                    if pd.notna(dt):
                        apt_dates[stem] = dt.strftime("%Y-%m-%d")
            except Exception:
                continue
        print(f"APTnotes dates indexed: {len(apt_dates)} entries")
    else:
        print(f"⚠  Could not find filename/date columns. Available: {list(apt.columns)}")

# ─────────────────────────────────────────────────────────
# 4. Manual dates — fill these for docs missing from APTnotes
# ─────────────────────────────────────────────────────────
# Keys are the exact `document_id` values from artifacts/train.csv.
# Values are ISO dates "YYYY-MM-DD" (year precision acceptable).
# Add entries here after running once and seeing the "Unmatched" list.
#
# Examples (uncomment and fill in for your documents):
#
#   "23_mandiant_report": "2023-06-15",
#   "24_mandiant_report": "2023-11-02",
#   "100_mitre_report":   "2024-03-20",
#
manual_dates = {
    # example — replace with actual dates from vendor sites
    # "23_mandiant_report": "2023-06-15",
}

# ─────────────────────────────────────────────────────────
# 5. Match document_ids → dates
# ─────────────────────────────────────────────────────────
matched, unmatched = [], []
for did in doc_ids:
    did_str = str(did).strip()
    date = None

    # 5a. Exact APTnotes filename match
    if did_str in apt_dates:
        date = apt_dates[did_str]

    # 5b. Fuzzy APTnotes match (substring)
    if date is None:
        for stem, d in apt_dates.items():
            if did_str and stem and (did_str in stem or stem in did_str):
                date = d
                break

    # 5c. Manual override
    if date is None and did_str in manual_dates:
        date = manual_dates[did_str]

    if date:
        matched.append({"document_id": did_str, "date": date})
    else:
        unmatched.append(did_str)

print(f"\nMatched:   {len(matched)}")
print(f"Unmatched: {len(unmatched)}")

if unmatched:
    print("\nUnmatched document_ids (add to `manual_dates` dict above):")
    for u in unmatched[:40]:
        print(f"  \"{u}\": \"YYYY-MM-DD\",")
    if len(unmatched) > 40:
        print(f"  ... and {len(unmatched) - 40} more")

# ─────────────────────────────────────────────────────────
# 6. Save — handles empty case safely
# ─────────────────────────────────────────────────────────
out = pd.DataFrame(matched, columns=["document_id", "date"])
if len(out) > 0:
    out = out.sort_values("date").reset_index(drop=True)

out_path = "artifacts/doc_dates.csv"
out.to_csv(out_path, index=False)

print(f"\nSaved {len(out)} dated documents to {out_path}")

if len(out) > 0:
    print(out.head(20).to_string(index=False))
    print(f"\nNext step: run Notebook 5 Cell 12 to generate the temporal plot.")
else:
    print("\n⚠  0 documents dated.")
    print("   Temporal analysis in Notebook 5 will print a SKIP message.")
    print()
    print("   Options:")
    print("     1. RECOMMENDED — Leave as-is and document the missing")
    print("        temporal analysis as a limitation in Chapter 6.")
    print()
    print("     2. Fill `manual_dates` in this script by looking up")
    print("        publication dates on vendor sites (Mandiant, MITRE,")
    print("        CrowdStrike), then re-run:")
    print("            python scripts/make_doc_dates.py")
    print()
    print("     3. Delete Notebook 5 Cell 12 (temporal analysis) and")
    print("        proceed without it. All other analyses remain complete.")