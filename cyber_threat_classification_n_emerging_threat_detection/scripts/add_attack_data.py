# scripts/add_attack_data.py
"""
Force-include all 1,086 CIRCL ATT&CK-labeled CVEs into the dataset.
Result: 15,000 rows total with ~1,086 ATT&CK-labeled (7.2%).
"""
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from pathlib import Path
from datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA = PROJECT_ROOT / 'dataset' / 'final_cti_dataset.csv'
OUT = PROJECT_ROOT / 'dataset' / 'final_cti_15k_with_attack.csv'

# Load current 15K
df = pd.read_csv(DATA)
print(f'Current: {len(df):,} rows')
print(f'With ATT&CK: {(df["primary_tech"].notna() & (df["primary_tech"] != "")).sum()}')

# Load ALL CIRCL CVEs (1,086 with ATT&CK)
print('\nLoading CIRCL...')
circl = load_dataset('CIRCL/vulnerability-attack-techniques', split='train')

# Load MITRE names
import requests
r = requests.get(
    'https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json',
    timeout=120
)
tech_names = {}
for obj in r.json().get('objects', []):
    if obj.get('type') != 'attack-pattern' or obj.get('revoked'):
        continue
    ext = next((ref.get('external_id') for ref in obj.get('external_references', [])
                if ref.get('source_name') == 'mitre-attack'), None)
    if ext:
        tech_names[ext] = obj.get('name', '')

# Build CIRCL rows
circl_rows = []
for row in circl:
    cid = row.get('id')
    desc = str(row.get('description', '') or '').strip()
    techs = row.get('techniques') or []
    if not cid or not desc or not techs:
        continue
    parents = list(set(str(t).split('.')[0] for t in techs if t))
    if not parents:
        continue
    primary = parents[0]
    circl_rows.append({
        'cve_id': cid,
        'sentence_text': desc,
        'source': 'circl',
        'published': '',
        'relevant': 'yes',
        'severity': 'HIGH',
        'cvss_score': None,
        'techniques': '; '.join(parents),
        'primary_tech': primary,
        'tech_name': tech_names.get(primary, ''),
        'cwe_ids': '',
        'in_kev': False,
        'kev_ransomware': False,
        'epss': None,
    })

df_circl = pd.DataFrame(circl_rows)
print(f'CIRCL rows: {len(df_circl):,}')

# Remove duplicates (CIRCL CVEs already in df)
existing_cves = set(df['cve_id'])
df_circl_new = df_circl[~df_circl['cve_id'].isin(existing_cves)]
print(f'New CIRCL CVEs: {len(df_circl_new):,}')

# Replace bottom 1086 rows with CIRCL rows (force-include)
# Keep all 13,914 non-ATT&CK rows + all 1,086 CIRCL rows
df_non_attack = df[~(df['primary_tech'].notna() & (df['primary_tech'] != ''))]
df_combined = pd.concat([df_non_attack, df_circl_new], ignore_index=True)

# Trim to 15,000 if over
if len(df_combined) > 15000:
    df_combined = df_combined.sample(n=15000, random_state=42).reset_index(drop=True)
elif len(df_combined) < 15000:
    # Fill with extra eromang rows that aren't already in
    used = set(df_combined['cve_id'])
    remaining = df[~df['cve_id'].isin(used)]
    needed = 15000 - len(df_combined)
    if len(remaining) > 0:
        fill = remaining.sample(n=min(needed, len(remaining)), random_state=42)
        df_combined = pd.concat([df_combined, fill], ignore_index=True)

print(f'\nFinal: {len(df_combined):,} rows')
print(f'ATT&CK rows: {(df_combined["primary_tech"].notna() & (df_combined["primary_tech"] != "")).sum()}')

# Distribution check
print('\n--- Severity ---')
print(df_combined['severity'].value_counts().to_string())
print('\n--- ATT&CK top-8 ---')
TOP8 = ['T1059', 'T1190', 'T1005', 'T1068', 'T1203', 'T1189', 'T1499', 'T1078']
att = df_combined[df_combined['primary_tech'].isin(TOP8)]
print(att['primary_tech'].value_counts().to_string())
print(f'Total top-8: {len(att):,}')

df_combined.to_csv(OUT, index=False, encoding='utf-8-sig')
print(f'\n✅ Saved: {OUT}')