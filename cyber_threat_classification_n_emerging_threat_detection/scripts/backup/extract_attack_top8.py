# scripts/build_final_dataset.py
"""
FINAL dataset builder — one CSV with everything:
  - Binary relevance    (relevant: yes/no)
  - 4-class severity    (LOW/MEDIUM/HIGH/CRITICAL)
  - 8-class ATT&CK      (primary_tech: T1059, T1190, ...)
  - Temporal            (date: for LDA emerging threat detection)
"""
import warnings
warnings.filterwarnings('ignore')

import re
import uuid
import pandas as pd
from pathlib import Path
from datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT = PROJECT_ROOT / 'dataset' / 'final_cti_dataset.csv'
OUT.parent.mkdir(exist_ok=True)

# The 8 ATT&CK techniques we support
TOP8_TECHS = ['T1059', 'T1190', 'T1005', 'T1068', 'T1203', 'T1189', 'T1499', 'T1078']


def normalize_cve(v):
    if pd.isna(v):
        return None
    m = re.search(r'CVE-\d{4}-\d{4,}', str(v).strip().upper())
    return m.group(0) if m else None


def safe_join(v):
    if v is None:
        return ''
    if isinstance(v, (list, tuple)):
        return '; '.join(str(x) for x in v if x)
    return str(v)


def parse_date(v):
    if v is None or (isinstance(v, str) and not v.strip()):
        return ''
    try:
        ts = pd.to_datetime(v, errors='coerce')
        if pd.isna(ts):
            return ''
        return ts.strftime('%Y-%m-%d')
    except Exception:
        return ''


# ============================================================
# SOURCE 1 — CIRCL (ATT&CK techniques + descriptions)
# ============================================================
print('[1/2] Loading CIRCL...')
circl_rows = []
try:
    ds = load_dataset('CIRCL/vulnerability-attack-techniques', split='train')
    for row in ds:
        text = str(row.get('description') or '').strip()
        techs = row.get('techniques') or []
        if not text:
            continue

        # Extract parent techniques (T1059.001 -> T1059)
        parent_techs = []
        for t in techs:
            parent = str(t).split('.')[0].strip()
            if parent and parent not in parent_techs:
                parent_techs.append(parent)

        # primary_tech = first technique that's in our top-8
        primary = ''
        for p in parent_techs:
            if p in TOP8_TECHS:
                primary = p
                break
        # if none match, use first parent
        if not primary and parent_techs:
            primary = parent_techs[0]

        cve_id = row.get('id') or f'CVE-UNK-{uuid.uuid4().hex[:8]}'
        pub = parse_date(row.get('published_date') or row.get('date'))

        circl_rows.append({
            'uid': f'circl_{cve_id}',
            'source': 'circl',
            'cve_id': normalize_cve(cve_id) or cve_id,
            'sentence_text': text,
            'date': pub,
            'severity': 'HIGH',
            'relevant': 'yes',
            'techniques': '; '.join(parent_techs),
            'primary_tech': primary,
            'cwe_ids': safe_join(row.get('cwes')),
            'has_exploit': False,
            'in_kev': False,
            'ransomware_use': False,
        })
    print(f'  CIRCL: {len(circl_rows):,} rows')
except Exception as e:
    print(f'  CIRCL failed: {e}')


# ============================================================
# SOURCE 2 — ThreatCluster (severity + exploit + dates)
# ============================================================
print('[2/2] Loading ThreatCluster...')
tc_rows = []
try:
    ds = load_dataset('threatcluster/cve-exploitation-signals', split='train')
    for row in ds:
        text = str(row.get('description') or '').strip()
        if not text:
            continue
        cve_id = row.get('cve_id') or f'CVE-UNK-{uuid.uuid4().hex[:8]}'

        sev = str(row.get('cvss_v3_severity') or '').strip().upper()
        if sev not in ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL'):
            sev = 'MEDIUM'

        in_kev = bool(row.get('in_kev'))
        has_exp = bool(row.get('has_exploit')) or (row.get('exploit_count') or 0) > 0
        rans = bool(row.get('ransomware_use'))

        is_relevant = in_kev or has_exp or rans or sev == 'CRITICAL'

        pub = parse_date(row.get('published_date') or row.get('kev_added_date'))

        tc_rows.append({
            'uid': f'tc_{cve_id}',
            'source': 'threatcluster',
            'cve_id': normalize_cve(cve_id) or cve_id,
            'sentence_text': text,
            'date': pub,
            'severity': sev,
            'relevant': 'yes' if is_relevant else 'no',
            'techniques': '',
            'primary_tech': '',
            'cwe_ids': safe_join(row.get('cwe_ids')),
            'has_exploit': has_exp,
            'in_kev': in_kev,
            'ransomware_use': rans,
        })
    print(f'  ThreatCluster: {len(tc_rows):,} rows')
except Exception as e:
    print(f'  ThreatCluster failed: {e}')


# ============================================================
# COMBINE
# ============================================================
print('\nCombining...')
df = pd.concat([pd.DataFrame(circl_rows), pd.DataFrame(tc_rows)], ignore_index=True)

# Drop exact duplicate text
df = df.drop_duplicates(subset=['sentence_text'], keep='first')

COLS = [
    'uid', 'source', 'cve_id', 'sentence_text', 'date',
    'severity', 'relevant',
    'techniques', 'primary_tech',
    'cwe_ids', 'has_exploit', 'in_kev', 'ransomware_use',
]
df = df[COLS]


# ============================================================
# REPORT
# ============================================================
print('\n' + '='*60)
print('FINAL CTI DATASET — SUMMARY')
print('='*60)
print(f'Rows     : {len(df):,}')
print(f'Columns  : {len(COLS)}')

print(f'\n--- Source ---')
print(df['source'].value_counts().to_string())

print(f'\n--- Binary (relevant) ---')
print(df['relevant'].value_counts().to_string())

print(f'\n--- 4-class severity ---')
print(df['severity'].value_counts().to_string())

print(f'\n--- 8-class ATT&CK (primary_tech) ---')
att = df[df['primary_tech'].isin(TOP8_TECHS)]
print(f'Rows in top-8: {len(att):,}')
print(att['primary_tech'].value_counts().to_string())

print(f'\n--- Date coverage (for LDA) ---')
has_date = df[df['date'] != '']
print(f'With date: {len(has_date):,} / {len(df):,} ({len(has_date)/len(df)*100:.1f}%)')
if len(has_date) > 0:
    print(f'Date range: {has_date["date"].min()} to {has_date["date"].max()}')

print(f'\n--- Column completeness ---')
for c in COLS:
    if c in ('has_exploit', 'in_kev', 'ransomware_use'):
        n = df[c].astype(bool).sum()
        print(f'  {c:18s} {n:>7,} True   ({n/len(df)*100:.1f}%)')
    else:
        n = (df[c].astype(str).str.strip() != '').sum()
        print(f'  {c:18s} {n:>7,} filled ({n/len(df)*100:.1f}%)')

# ============================================================
# SAVE
# ============================================================
df.to_csv(OUT, index=False, encoding='utf-8-sig')
print(f'\n✅ Saved: {OUT}')
print(f'   {len(df):,} rows × {len(COLS)} columns')