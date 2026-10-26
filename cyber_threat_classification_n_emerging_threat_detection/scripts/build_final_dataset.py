# scripts/build_final_dataset.py
"""
FINAL DATASET BUILDER — no API keys, no scanning, no crashes.
Source: eromang/cyberscale-training-cves (30,641 balanced CVEs from cvelistV5)
Enrichments: CISA KEV + FIRST EPSS + CIRCL ATT&CK + MITRE names
Target: 15,000 rows balanced across 4 severity classes.
"""

import warnings
warnings.filterwarnings('ignore')

import gzip
import io
import random
import requests
import pandas as pd
from pathlib import Path
from datasets import load_dataset

# ============================================================
# CONFIG
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT = PROJECT_ROOT / 'dataset' / 'final_cti_dataset.csv'
OUT.parent.mkdir(exist_ok=True)

HEADERS = {'User-Agent': 'Thesis-Research/1.0'}
TARGET = 15000
RANDOM_SEED = 42
SEV_TARGETS = {'LOW': 2500, 'MEDIUM': 5000, 'HIGH': 5000, 'CRITICAL': 2500}

random.seed(RANDOM_SEED)


# ============================================================
# STEP 1 — Load base dataset (30,641 balanced CVEs)
# ============================================================
def load_base():
    print('[1/5] Loading eromang/cyberscale-training-cves...')
    try:
        ds = load_dataset('eromang/cyberscale-training-cves', split='train')
        print(f'  Loaded {len(ds):,} rows')
        print(f'  Columns: {ds.column_names}')
        df = ds.to_pandas()
    except Exception as e:
        print(f'  ❌ Failed: {e}')
        print('  Trying alternative dataset: eromang/cyberscale-holdout-cves')
        try:
            ds = load_dataset('eromang/cyberscale-holdout-cves', split='train')
            print(f'  Loaded {len(ds):,} rows')
            df = ds.to_pandas()
        except Exception as e2:
            print(f'  ❌ Both failed: {e2}')
            raise

    # Rename to our schema
    df = df.rename(columns={
        'description': 'sentence_text',
        'cwe': 'cwe_ids',
    })

    # Ensure cvss_score is numeric
    df['cvss_score'] = pd.to_numeric(df.get('cvss_score'), errors='coerce')

    # Derive severity from CVSS score if not present
    if 'cvss_severity' not in df.columns or df['cvss_severity'].isna().all():
        def score_to_sev(s):
            if pd.isna(s):
                return ''
            s = float(s)
            if s == 0:
                return 'NONE'
            if s < 4:
                return 'LOW'
            if s < 7:
                return 'MEDIUM'
            if s < 9:
                return 'HIGH'
            return 'CRITICAL'
        df['severity'] = df['cvss_score'].apply(score_to_sev)
    else:
        df['severity'] = df['cvss_severity'].astype(str).str.upper().str.strip()

    # Filter to 4 valid classes
    df = df[df['severity'].isin(['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'])].copy()

    # Ensure required fields
    df['source'] = df.get('source', 'cyberscale')
    if 'published' not in df.columns:
        df['published'] = ''
    df['cwe_ids'] = df['cwe_ids'].fillna('').astype(str)

    # Drop rows with empty descriptions
    df = df[df['sentence_text'].notna()]
    df = df[df['sentence_text'].astype(str).str.len() >= 20]

    # Drop duplicate CVEs
    df = df.drop_duplicates(subset='cve_id', keep='first')

    print(f'  After cleaning: {len(df):,} rows')
    print(f'  Severity: {dict(df["severity"].value_counts())}')
    return df


# ============================================================
# STEP 2 — CISA KEV
# ============================================================
def add_kev(df):
    print('\n[2/5] Fetching CISA KEV...')
    try:
        r = requests.get(
            'https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json',
            headers=HEADERS,
            timeout=60,
        )
        r.raise_for_status()
        kev = {x['cveID']: x for x in r.json().get('vulnerabilities', [])}
        print(f'  KEV entries: {len(kev):,}')
    except Exception as e:
        print(f'  ❌ KEV failed: {e}')
        kev = {}

    df['in_kev'] = df['cve_id'].map(lambda x: x in kev)
    df['kev_ransomware'] = df['cve_id'].map(
        lambda x: (kev.get(x, {}).get('knownRansomwareCampaignUse', '') or '').lower() == 'known'
    )
    print(f'  Rows in KEV: {df["in_kev"].sum()}')
    return df


# ============================================================
# STEP 3 — EPSS
# ============================================================
def add_epss(df):
    print('\n[3/5] Fetching EPSS scores...')
    try:
        r = requests.get(
            'https://epss.empiricalsecurity.com/epss_scores-current.csv.gz',
            headers=HEADERS,
            timeout=120,
        )
        r.raise_for_status()
        with gzip.open(io.BytesIO(r.content), 'rt') as f:
            epss_df = pd.read_csv(f, skiprows=1)
        epss = dict(zip(epss_df['cve'], epss_df['epss'].astype(float)))
        print(f'  EPSS scores: {len(epss):,}')
    except Exception as e:
        print(f'  ❌ EPSS failed: {e}')
        epss = {}

    df['epss'] = df['cve_id'].map(lambda x: epss.get(x))
    print(f'  Rows with EPSS: {df["epss"].notna().sum()}')
    return df


# ============================================================
# STEP 4 — CIRCL ATT&CK
# ============================================================
def add_attack(df):
    print('\n[4/5] Loading CIRCL ATT&CK mappings...')
    try:
        circl = load_dataset('CIRCL/vulnerability-attack-techniques', split='train')
        cve_to_tech = {}
        for row in circl:
            cid = row.get('id')
            techs = row.get('techniques') or []
            if cid and techs:
                parents = list(set(str(t).split('.')[0] for t in techs if t))
                cve_to_tech[cid] = parents
        print(f'  CIRCL mappings: {len(cve_to_tech):,}')
    except Exception as e:
        print(f'  ❌ CIRCL failed: {e}')
        cve_to_tech = {}

    df['techniques'] = df['cve_id'].map(lambda x: '; '.join(cve_to_tech.get(x, [])))
    df['primary_tech'] = df['techniques'].map(
        lambda x: x.split(';')[0].strip() if x and x.strip() else ''
    )

    # MITRE names
    print('  Fetching MITRE names...')
    try:
        r = requests.get(
            'https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json',
            headers=HEADERS,
            timeout=120,
        )
        tech_names = {}
        for obj in r.json().get('objects', []):
            if obj.get('type') != 'attack-pattern' or obj.get('revoked'):
                continue
            ext = next(
                (ref.get('external_id') for ref in obj.get('external_references', [])
                 if ref.get('source_name') == 'mitre-attack'),
                None,
            )
            if ext:
                tech_names[ext] = obj.get('name', '')
        print(f'  ATT&CK names: {len(tech_names):,}')
    except Exception as e:
        print(f'  ❌ MITRE names failed: {e}')
        tech_names = {}

    df['tech_name'] = df['primary_tech'].map(tech_names)
    print(f'  Rows with ATT&CK: {(df["primary_tech"] != "").sum()}')
    return df


# ============================================================
# STEP 5 — Binary target + sample to 15K + save
# ============================================================
def build_binary_and_sample(df):
    print('\n[5/5] Building binary target + sampling...')

    df['relevant'] = df.apply(
        lambda r: 'yes' if (
            r['in_kev'] or r['kev_ransomware'] or r['severity'] == 'CRITICAL'
        ) else 'no',
        axis=1,
    )

    # Sample balanced
    sampled = []
    for sev, target in SEV_TARGETS.items():
        pool = df[df['severity'] == sev]
        if len(pool) == 0:
            print(f'  {sev:10s} 0 available (skipping)')
            continue
        take = min(target, len(pool))
        sampled.append(pool.sample(n=take, random_state=RANDOM_SEED))
        print(f'  {sev:10s} {take:>5,} / {len(pool):>6,}')

    if not sampled:
        raise RuntimeError('No samples taken — severity distribution is empty')

    df_out = pd.concat(sampled).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    # If undershot, fill from remaining
    if len(df_out) < TARGET:
        used = set(df_out['cve_id'])
        remaining = df[~df['cve_id'].isin(used)]
        needed = TARGET - len(df_out)
        if len(remaining) > 0:
            fill = remaining.sample(n=min(needed, len(remaining)), random_state=RANDOM_SEED)
            df_out = pd.concat([df_out, fill]).reset_index(drop=True)
            print(f'  Filled to {len(df_out):,} with extra rows')

    return df_out


# ============================================================
# MAIN
# ============================================================
def main():
    # 1. Load
    df = load_base()

    # 2. Enrich
    df = add_kev(df)
    df = add_epss(df)
    df = add_attack(df)

    # 3. Binary + sample
    df_out = build_binary_and_sample(df)

    # 4. Select final columns
    COLS = [
        'cve_id', 'sentence_text', 'source', 'published',
        'relevant', 'severity', 'cvss_score',
        'techniques', 'primary_tech', 'tech_name', 'cwe_ids',
        'in_kev', 'kev_ransomware', 'epss',
    ]
    df_out = df_out[[c for c in COLS if c in df_out.columns]]

    # 5. Report
    print('\n' + '=' * 60)
    print('FINAL DATASET — SUMMARY')
    print('=' * 60)
    print(f'Rows     : {len(df_out):,}')
    print(f'Columns  : {len(df_out.columns)}')

    print(f'\n--- Source ---')
    print(df_out['source'].value_counts().to_string())

    print(f'\n--- Severity ---')
    print(df_out['severity'].value_counts().to_string())

    print(f'\n--- Binary (relevant) ---')
    print(df_out['relevant'].value_counts().to_string())

    TOP8 = ['T1059', 'T1190', 'T1005', 'T1068', 'T1203', 'T1189', 'T1499', 'T1078']
    att = df_out[df_out['primary_tech'].isin(TOP8)]
    print(f'\n--- 8-class ATT&CK ---')
    print(att['primary_tech'].value_counts().to_string())
    print(f'Total: {len(att):,}')

    print(f'\n--- Enrichment ---')
    print(f'  KEV     : {df_out["in_kev"].sum():,}')
    print(f'  EPSS    : {df_out["epss"].notna().sum():,} ({df_out["epss"].notna().mean()*100:.1f}%)')
    print(f'  Dates   : {df_out["published"].notna().sum():,}')

    # 6. Save
    df_out.to_csv(OUT, index=False, encoding='utf-8-sig')
    print(f'\n✅ Saved: {OUT}')
    print(f'   {len(df_out):,} rows × {len(df_out.columns)} columns')


if __name__ == '__main__':
    main()