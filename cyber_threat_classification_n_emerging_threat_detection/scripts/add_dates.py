# scripts/add_dates_v2.py
"""
Add publication dates from CVEListV5 with verbose diagnostics.
"""
import warnings
warnings.filterwarnings('ignore')

import json
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA = PROJECT_ROOT / 'dataset' / 'final_cti_15k_with_attack.csv'
CVELIST = PROJECT_ROOT / 'dataset' / 'cvelistV5'
OUT = PROJECT_ROOT / 'dataset' / 'final_cti_complete.csv'

# 1. Load dataset
df = pd.read_csv(DATA)
print(f'Rows: {len(df):,}')

# 2. Check CVEListV5 exists
print(f'\nChecking CVEListV5...')
print(f'  Path: {CVELIST}')
print(f'  Exists: {CVELIST.exists()}')
if not CVELIST.exists():
    raise FileNotFoundError(f'CVEListV5 not found at {CVELIST}. Run: git clone --depth 1 https://github.com/CVEProject/cvelistV5.git dataset/cvelistV5')

cves_dir = CVELIST / 'cves'
print(f'  cves/ exists: {cves_dir.exists()}')
if cves_dir.exists():
    subdirs = [x.name for x in cves_dir.iterdir() if x.is_dir()]
    print(f'  Year subdirs: {sorted(subdirs)}')

# 3. Check what years we need
df['cve_year'] = df['cve_id'].str.extract(r'CVE-(\d{4})')[0]
print(f'\nCVE years in dataset:')
print(df['cve_year'].value_counts().sort_index().to_string())

# 4. Build lookup — scan only needed years
wanted_years = set(df['cve_year'].dropna())
print(f'\nNeeded years: {sorted(wanted_years)}')

cve_dates = {}
total_files_scanned = 0

for year in sorted(wanted_years):
    year_dir = cves_dir / year
    if not year_dir.exists():
        print(f'  ⚠️ Year {year} not in CVEListV5')
        continue

    # Find all JSON files in this year
    json_files = list(year_dir.glob('**/CVE-*.json'))
    print(f'  Year {year}: {len(json_files):,} JSON files')

    for fp in json_files:
        total_files_scanned += 1
        try:
            with open(fp, encoding='utf-8') as f:
                obj = json.load(f)
            cid = obj.get('cveMetadata', {}).get('cveId', '')
            if cid in set(df['cve_id']):
                pub = obj.get('cveMetadata', {}).get('datePublished', '')
                if pub:
                    cve_dates[cid] = pub[:10]
        except Exception:
            continue

    print(f'    Matches so far: {len(cve_dates):,}')

print(f'\nTotal files scanned: {total_files_scanned:,}')
print(f'Dates found: {len(cve_dates):,} / {len(df):,}')

# 5. Apply dates
df['published'] = df['cve_id'].map(cve_dates).fillna('')
df = df.drop(columns=['cve_year'])

print(f'\nRows with dates: {(df["published"] != "").sum():,} ({(df["published"] != "").mean()*100:.1f}%)')

# 6. Save
df.to_csv(OUT, index=False, encoding='utf-8-sig')
print(f'\n✅ Saved: {OUT}')