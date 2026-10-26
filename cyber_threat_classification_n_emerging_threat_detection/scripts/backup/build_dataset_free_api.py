# scripts/build_dataset_free_api.py
"""
Build dataset using ONLY free APIs — no keys, no signup.
Sources:
  - dbcve.org: CVE descriptions + CVSS severity (bulk CSV)
  - CISA KEV: Known exploited CVEs
  - FIRST EPSS: Exploit probability
  - MITRE ATT&CK: Technique catalog
  - HuggingFace CIRCL: CVE → ATT&CK mappings
"""
import warnings
warnings.filterwarnings('ignore')

import io
import gzip
import json
import requests
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT = PROJECT_ROOT / 'dataset' / 'free_api_cti_dataset.csv'
OUT.parent.mkdir(exist_ok=True)

HEADERS = {'User-Agent': 'Thesis-Research/1.0'}


# ============================================================
# SOURCE 1 — dbcve.org (CVE descriptions + CVSS, bulk CSV)
# ============================================================
print('[1/4] Fetching CVE data from dbcve.org...')
try:
    r = requests.get('https://dbcve.org/api/v1/export/enriched.csv',
                     headers=HEADERS, timeout=120, stream=True)
    r.raise_for_status()
    # Save raw CSV first
    raw_path = PROJECT_ROOT / 'dataset' / 'dbcve_enriched.csv'
    with open(raw_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    df_cve = pd.read_csv(raw_path, low_memory=False)
    print(f'  Loaded: {len(df_cve):,} CVEs')
    print(f'  Columns: {df_cve.columns.tolist()[:15]}')
except Exception as e:
    print(f'  ❌ Failed: {e}')
    df_cve = pd.DataFrame()


# ============================================================
# SOURCE 2 — CISA KEV (no key)
# ============================================================
print('\n[2/4] Fetching CISA KEV...')
try:
    r = requests.get('https://www.cisa.gov/sites/default/files/feeds/'
                     'known_exploited_vulnerabilities.json',
                     headers=HEADERS, timeout=60)
    kev = {x['cveID']: x for x in r.json().get('vulnerabilities', [])}
    print(f'  KEV entries: {len(kev):,}')
except Exception as e:
    print(f'  ❌ Failed: {e}')
    kev = {}


# ============================================================
# SOURCE 3 — FIRST EPSS (no key, gzipped CSV)
# ============================================================
print('\n[3/4] Fetching EPSS scores...')
try:
    r = requests.get('https://epss.empiricalsecurity.com/epss_scores-current.csv.gz',
                     headers=HEADERS, timeout=120)
    r.raise_for_status()
    with gzip.open(io.BytesIO(r.content), 'rt') as f:
        epss_df = pd.read_csv(f, skiprows=1)  # skip comment line
    epss = dict(zip(epss_df['cve'], epss_df['epss'].astype(float)))
    print(f'  EPSS entries: {len(epss):,}')
except Exception as e:
    print(f'  ❌ Failed: {e}')
    epss = {}


# ============================================================
# SOURCE 4 — MITRE ATT&CK catalog (GitHub raw, no key)
# ============================================================
print('\n[4/4] Fetching MITRE ATT&CK catalog...')
try:
    r = requests.get('https://raw.githubusercontent.com/mitre/cti/master/'
                     'enterprise-attack/enterprise-attack.json',
                     headers=HEADERS, timeout=120)
    attack = r.json()
    tech_names = {}
    for obj in attack.get('objects', []):
        if obj.get('revoked') or obj.get('x_mitre_deprecated'):
            continue
        if obj.get('type') != 'attack-pattern':
            continue
        ext = next((ref.get('external_id') for ref in obj.get('external_references', [])
                    if ref.get('source_name') == 'mitre-attack'), None)
        if ext:
            tech_names[ext] = obj.get('name', '')
    print(f'  ATT&CK techniques: {len(tech_names):,}')
except Exception as e:
    print(f'  ❌ Failed: {e}')
    tech_names = {}


# ============================================================
# MERGE
# ============================================================
print('\nMerging...')

if df_cve.empty:
    raise RuntimeError('No CVE data. Check dbcve.org endpoint.')

df = df_cve.copy()

# Normalize column names
rename_map = {
    'id': 'cve_id', 'cve': 'cve_id',
    'cvss_v3_severity': 'severity', 'severity': 'severity',
    'description': 'sentence_text', 'summary': 'sentence_text',
}
df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

# Ensure cve_id exists
if 'cve_id' not in df.columns:
    for col in df.columns:
        if 'cve' in col.lower():
            df['cve_id'] = df[col]
            break

# Ensure sentence_text exists
if 'sentence_text' not in df.columns:
    for col in df.columns:
        if 'desc' in col.lower() or 'summary' in col.lower():
            df['sentence_text'] = df[col]
            break

# Ensure severity exists
if 'severity' not in df.columns:
    for col in df.columns:
        if 'severity' in col.lower() or 'cvss' in col.lower():
            df['severity'] = df[col]
            break

# Add enrichments
df['in_kev'] = df['cve_id'].map(lambda x: x in kev if pd.notna(x) else False)
df['kev_ransomware'] = df['cve_id'].map(
    lambda x: kev.get(x, {}).get('knownRansomwareCampaignUse') == 'Known' if pd.notna(x) else False
)
df['epss'] = df['cve_id'].map(lambda x: epss.get(x) if pd.notna(x) else None)

# Add ATT&CK from HuggingFace CIRCL (already on disk)
print('\nLoading CIRCL ATT&CK mappings...')
try:
    from datasets import load_dataset
    circl = load_dataset('CIRCL/vulnerability-attack-techniques', split='train')
    cve_to_tech = {}
    for row in circl:
        cid = row.get('id')
        techs = row.get('techniques') or []
        if cid and techs:
            parents = list(set(str(t).split('.')[0] for t in techs if t))
            cve_to_tech[cid] = parents
    print(f'  CIRCL mappings: {len(cve_to_tech):,} CVEs')

    df['techniques'] = df['cve_id'].map(
        lambda x: '; '.join(cve_to_tech.get(x, [])) if pd.notna(x) else ''
    )
    df['primary_tech'] = df['techniques'].map(
        lambda x: x.split(';')[0].strip() if x else ''
    )
    df['tech_name'] = df['primary_tech'].map(tech_names)
except Exception as e:
    print(f'  ⚠️ CIRCL failed: {e}')
    df['techniques'] = ''
    df['primary_tech'] = ''
    df['tech_name'] = ''


# ============================================================
# BUILD BINARY TARGET (rule-based)
# ============================================================
print('\nBuilding binary relevance target...')
df['relevant'] = df.apply(
    lambda r: 'yes' if (r.get('in_kev', False) or r.get('kev_ransomware', False)
                        or (r.get('severity', '').upper() == 'CRITICAL'))
    else 'no',
    axis=1
)


# ============================================================
# FINAL COLUMNS
# ============================================================
print('\nFinalizing...')
FINAL_COLS = [
    'cve_id', 'sentence_text', 'severity', 'relevant',
    'primary_tech', 'tech_name', 'techniques',
    'in_kev', 'kev_ransomware', 'epss',
]
final_cols = [c for c in FINAL_COLS if c in df.columns]
df = df[final_cols].dropna(subset=['sentence_text'])
df = df.drop_duplicates(subset=['cve_id'], keep='first')


# ============================================================
# REPORT
# ============================================================
print('\n' + '='*60)
print('FREE-API CTI DATASET — SUMMARY')
print('='*60)
print(f'Rows     : {len(df):,}')
print(f'Columns  : {len(df.columns)}')

if 'severity' in df.columns:
    print(f'\n--- Severity ---')
    print(df['severity'].value_counts().head(10).to_string())

if 'relevant' in df.columns:
    print(f'\n--- Binary (relevant) ---')
    print(df['relevant'].value_counts().to_string())

if 'primary_tech' in df.columns:
    TOP8 = ['T1059', 'T1190', 'T1005', 'T1068', 'T1203', 'T1189', 'T1499', 'T1078']
    att = df[df['primary_tech'].isin(TOP8)]
    print(f'\n--- 8-class ATT&CK ---')
    print(att['primary_tech'].value_counts().to_string())
    print(f'Total: {len(att):,}')

print(f'\n--- Enrichment coverage ---')
print(f'  KEV     : {df["in_kev"].sum():,}')
print(f'  EPSS    : {df["epss"].notna().sum():,}')

df.to_csv(OUT, index=False, encoding='utf-8-sig')
print(f'\n✅ Saved: {OUT}')
print(f'   {len(df):,} rows × {len(df.columns)} columns')