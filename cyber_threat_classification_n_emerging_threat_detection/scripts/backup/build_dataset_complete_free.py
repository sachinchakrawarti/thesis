# scripts/build_dataset_fast.py
"""
Fast 15K dataset builder — skips HuggingFace matching.
Reads CVEListV5 once, samples 15K, adds all enrichments.
Runtime: ~3-5 minutes.
"""
import warnings
warnings.filterwarnings('ignore')

import json
import gzip
import io
import random
import requests
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CVELIST = PROJECT_ROOT / 'dataset' / 'cvelistV5'
OUT = PROJECT_ROOT / 'dataset' / 'final_cti_dataset.csv'
OUT.parent.mkdir(exist_ok=True)
HEADERS = {'User-Agent': 'Thesis-Research/1.0'}
TARGET = 15000
random.seed(42)

SEV_TARGETS = {'LOW': 2500, 'MEDIUM': 5000, 'HIGH': 5000, 'CRITICAL': 2500}

# ============================================================
# STEP 1 — Scan CVEListV5 ONCE, group by severity
# ============================================================
print('[1/4] Scanning CVEListV5 (this is the slow part — 3-5 min)...')

cve_files = list(CVELIST.glob('cves/**/CVE-*.json'))
print(f'  Total CVE files: {len(cve_files):,}')

by_severity = {'LOW': [], 'MEDIUM': [], 'HIGH': [], 'CRITICAL': []}
processed = 0

for i, fp in enumerate(cve_files):
    if i % 50000 == 0 and i > 0:
        total = sum(len(v) for v in by_severity.values())
        print(f'  ...{i:,}/{len(cve_files):,} files, {total:,} candidates so far')
    try:
        with open(fp, encoding='utf-8') as f:
            obj = json.load(f)
        meta = obj.get('cveMetadata', {})
        if meta.get('state') != 'PUBLISHED':
            continue
        cna = obj.get('containers', {}).get('cna', {})
        desc = next(
            (d.get('value', '') for d in cna.get('descriptions', [])
             if d.get('lang', '').startswith('en')),
            ''
        )
        if not desc or len(desc) < 20:
            continue

        severity, cvss_score = '', None
        for metric in cna.get('metrics', []):
            for k in ['cvssV3_1', 'cvssV3_0', 'cvssV4_0']:
                if k in metric:
                    severity = (metric[k].get('baseSeverity', '') or '').upper()
                    cvss_score = metric[k].get('baseScore')
                    if severity in by_severity:
                        break
            if severity in by_severity:
                break
        if severity not in by_severity:
            continue

        cwes = []
        for pt in cna.get('problemTypes', []):
            for d in pt.get('descriptions', []):
                if d.get('cweId'):
                    cwes.append(d['cweId'])

        by_severity[severity].append({
            'cve_id': meta.get('cveId', ''),
            'sentence_text': desc,
            'source': 'cvelistv5',
            'published': (meta.get('datePublished', '') or '')[:10],
            'severity': severity,
            'cvss_score': cvss_score,
            'cwe_ids': '; '.join(cwes),
        })
    except Exception:
        continue

total = sum(len(v) for v in by_severity.values())
print(f'\n  ✅ Found {total:,} valid CVEs')
for sev in ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']:
    print(f'    {sev:10s} {len(by_severity[sev]):>7,}')

# ============================================================
# STEP 2 — Sample 15K balanced
# ============================================================
print(f'\n[2/4] Sampling {TARGET:,} rows...')
sampled = []
for sev, target in SEV_TARGETS.items():
    pool = by_severity[sev]
    take = min(target, len(pool))
    if take < len(pool):
        sampled.extend(random.sample(pool, take))
    else:
        sampled.extend(pool)
    print(f'  {sev:10s} {take:>5,} / {len(pool):>7,}')

# If we undershot, fill from any pool
if len(sampled) < TARGET:
    used_ids = {r['cve_id'] for r in sampled}
    extra = []
    for sev in by_severity:
        for r in by_severity[sev]:
            if r['cve_id'] not in used_ids:
                extra.append(r)
    random.shuffle(extra)
    sampled.extend(extra[:TARGET - len(sampled)])

random.shuffle(sampled)
df = pd.DataFrame(sampled[:TARGET])
print(f'  ✅ Sampled: {len(df):,} rows')

# ============================================================
# STEP 3 — KEV + EPSS + CIRCL
# ============================================================
print('\n[3/4] Fetching enrichments...')

# KEV
try:
    r = requests.get(
        'https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json',
        headers=HEADERS, timeout=60
    )
    kev = {x['cveID']: x for x in r.json().get('vulnerabilities', [])}
    print(f'  KEV: {len(kev):,}')
except Exception as e:
    print(f'  KEV failed: {e}')
    kev = {}

df['in_kev'] = df['cve_id'].map(lambda x: x in kev)
df['kev_ransomware'] = df['cve_id'].map(
    lambda x: (kev.get(x, {}).get('knownRansomwareCampaignUse', '') or '').lower() == 'known'
)

# EPSS
try:
    r = requests.get(
        'https://epss.empiricalsecurity.com/epss_scores-current.csv.gz',
        headers=HEADERS, timeout=120
    )
    with gzip.open(io.BytesIO(r.content), 'rt') as f:
        epss_df = pd.read_csv(f, skiprows=1)
    epss = dict(zip(epss_df['cve'], epss_df['epss'].astype(float)))
    print(f'  EPSS: {len(epss):,}')
except Exception as e:
    print(f'  EPSS failed: {e}')
    epss = {}

df['epss'] = df['cve_id'].map(lambda x: epss.get(x))

# CIRCL
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
    print(f'  CIRCL: {len(cve_to_tech):,} mappings')
except Exception as e:
    print(f'  CIRCL failed: {e}')
    cve_to_tech = {}

df['techniques'] = df['cve_id'].map(lambda x: '; '.join(cve_to_tech.get(x, [])))
df['primary_tech'] = df['techniques'].map(
    lambda x: x.split(';')[0].strip() if x and x.strip() else ''
)

# MITRE names
try:
    r = requests.get(
        'https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json',
        headers=HEADERS, timeout=120
    )
    tech_names = {}
    for obj in r.json().get('objects', []):
        if obj.get('type') != 'attack-pattern' or obj.get('revoked'):
            continue
        ext = next((ref.get('external_id') for ref in obj.get('external_references', [])
                    if ref.get('source_name') == 'mitre-attack'), None)
        if ext:
            tech_names[ext] = obj.get('name', '')
    print(f'  ATT&CK names: {len(tech_names):,}')
except Exception:
    tech_names = {}

df['tech_name'] = df['primary_tech'].map(tech_names)

# ============================================================
# STEP 4 — Binary target + save
# ============================================================
print('\n[4/4] Building binary target + saving...')
df['relevant'] = df.apply(
    lambda r: 'yes' if (
        r['in_kev'] or r['kev_ransomware'] or r['severity'] == 'CRITICAL'
    ) else 'no',
    axis=1
)

COLS = [
    'cve_id', 'sentence_text', 'source', 'published',
    'relevant', 'severity', 'cvss_score',
    'techniques', 'primary_tech', 'tech_name', 'cwe_ids',
    'in_kev', 'kev_ransomware', 'epss',
]
df = df[COLS].dropna(subset=['sentence_text']).drop_duplicates(subset=['cve_id'])

print('\n' + '='*60)
print('FINAL DATASET')
print('='*60)
print(f'Rows     : {len(df):,}')
print(f'Columns  : {len(df.columns)}')
print(f'\n--- Severity ---')
print(df['severity'].value_counts().to_string())
print(f'\n--- Binary ---')
print(df['relevant'].value_counts().to_string())

TOP8 = ['T1059', 'T1190', 'T1005', 'T1068', 'T1203', 'T1189', 'T1499', 'T1078']
att = df[df['primary_tech'].isin(TOP8)]
print(f'\n--- ATT&CK top-8 ---')
print(att['primary_tech'].value_counts().to_string())
print(f'Total: {len(att):,}')
print(f'\nKEV: {df["in_kev"].sum()}')
print(f'EPSS: {df["epss"].notna().sum()}')

df.to_csv(OUT, index=False, encoding='utf-8-sig')
print(f'\n✅ Saved: {OUT}')
print(f'   {len(df):,} rows × {len(df.columns)} columns')



