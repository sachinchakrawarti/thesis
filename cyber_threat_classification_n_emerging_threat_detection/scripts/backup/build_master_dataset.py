# scripts/build_master_dataset.py
"""
Build a master dataset from MULTIPLE HuggingFace sources.
NO forced 6-class mapping. Preserve whatever category labels exist.
"""
import warnings
warnings.filterwarnings('ignore')

import uuid
import re
import pandas as pd
from pathlib import Path
from datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / 'dataset'
OUT_DIR.mkdir(exist_ok=True)

def normalize_cve(v):
    if pd.isna(v): return None
    m = re.search(r'CVE-\d{4}-\d{4,}', str(v).strip().upper())
    return m.group(0) if m else None

# ============================================================
# 1. CIRCL — techniques + description
# ============================================================
print('\n[1/4] Loading CIRCL...')
circl_rows = []
try:
    ds = load_dataset('CIRCL/vulnerability-attack-techniques', split='train')
    for row in ds:
        text = str(row.get('description') or '').strip()
        techs = row.get('techniques') or []
        if not text or not techs:
            continue
        cve_id = row.get('id') or f'CVE-UNK-{uuid.uuid4().hex[:6]}'
        
        # REAL label: use the MITRE technique ID as the category!
        for tech in techs:
            parent = str(tech).split('.')[0]
            circl_rows.append({
                'uid': f'circl_{cve_id}_{tech}',
                'source': 'circl',
                'cve_id': normalize_cve(cve_id),
                'sentence_text': text,
                'real_category': parent,  # e.g. T1059
                'label_type': 'mitre_technique',
                'severity': 'HIGH',
                'relevant': 'yes',
                'extra': str(tech),
            })
    print(f'  CIRCL: {len(circl_rows):,} rows, {len(set(r["real_category"] for r in circl_rows))} techniques')
except Exception as e:
    print(f'  CIRCL failed: {e}')

# ============================================================
# 2. ThreatCluster — severity + KEV
# ============================================================
print('\n[2/4] Loading ThreatCluster...')
tc_rows = []
try:
    ds = load_dataset('threatcluster/cve-exploitation-signals', split='train')
    for row in ds:
        text = str(row.get('description') or '').strip()
        if not text:
            continue
        cve_id = row.get('cve_id') or f'CVE-UNK-{uuid.uuid4().hex[:6]}'
        sev = str(row.get('cvss_v3_severity') or '').strip().upper()
        if sev not in ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL'):
            sev = 'MEDIUM'
        
        # REAL label: severity (LOW/MED/HIGH/CRIT)
        tc_rows.append({
            'uid': f'tc_{cve_id}',
            'source': 'threatcluster',
            'cve_id': normalize_cve(cve_id),
            'sentence_text': text,
            'real_category': sev,
            'label_type': 'cvss_severity',
            'severity': sev,
            'relevant': 'yes' if (row.get('in_kev') or row.get('has_exploit') or row.get('ransomware_use')) else 'no',
            'extra': '',
        })
    print(f'  ThreatCluster: {len(tc_rows):,} rows, {len(set(r["real_category"] for r in tc_rows))} severities')
except Exception as e:
    print(f'  ThreatCluster failed: {e}')

# ============================================================
# 3. AlicanKiraz0 — threat categories
# ============================================================
print('\n[3/4] Loading AlicanKiraz0...')
alican_rows = []
try:
    ds = load_dataset('AlicanKiraz0/Cyber-Threat-Intelligence-Text-Dataset', split='train')
    print(f'  Columns: {ds.column_names}')
    
    # Try to find the label column
    label_col = None
    text_col = None
    for c in ds.column_names:
        if any(k in c.lower() for k in ['label', 'class', 'type', 'category', 'threat']):
            label_col = c
        if any(k in c.lower() for k in ['text', 'content', 'post', 'body', 'description']):
            text_col = c
    print(f'  Using text_col={text_col}, label_col={label_col}')
    
    for row in ds:
        text = str(row.get(text_col) or '').strip() if text_col else ''
        label = str(row.get(label_col) or '').strip() if label_col else ''
        if not text or len(text) < 20:
            continue
        
        # REAL label: whatever the dataset says
        alican_rows.append({
            'uid': f'alican_{uuid.uuid4().hex[:12]}',
            'source': 'alican_kiraz',
            'cve_id': None,
            'sentence_text': text,
            'real_category': label if label else 'unknown',
            'label_type': 'cti_category',
            'severity': 'MEDIUM',
            'relevant': 'yes',
            'extra': '',
        })
    print(f'  AlicanKiraz0: {len(alican_rows):,} rows')
    if alican_rows:
        cats = pd.Series([r['real_category'] for r in alican_rows])
        print(f'  Categories found: {cats.value_counts().head(20).to_dict()}')
except Exception as e:
    print(f'  AlicanKiraz0 failed: {e}')

# ============================================================
# 4. HackerAI — threat categories
# ============================================================
print('\n[4/4] Loading HackerAI...')
hai_rows = []
try:
    ds = load_dataset('HackerAI/Cyber-Threat-Classification', split='train')
    print(f'  Columns: {ds.column_names}')
    
    label_col = None
    text_col = None
    for c in ds.column_names:
        if any(k in c.lower() for k in ['label', 'class', 'type', 'category', 'threat']):
            label_col = c
        if any(k in c.lower() for k in ['text', 'content', 'post', 'body', 'description']):
            text_col = c
    print(f'  Using text_col={text_col}, label_col={label_col}')
    
    for row in ds:
        text = str(row.get(text_col) or '').strip() if text_col else ''
        label = str(row.get(label_col) or '').strip() if label_col else ''
        if not text or len(text) < 20:
            continue
        
        hai_rows.append({
            'uid': f'hai_{uuid.uuid4().hex[:12]}',
            'source': 'hackerai',
            'cve_id': None,
            'sentence_text': text,
            'real_category': label if label else 'unknown',
            'label_type': 'cti_category',
            'severity': 'MEDIUM',
            'relevant': 'yes',
            'extra': '',
        })
    print(f'  HackerAI: {len(hai_rows):,} rows')
    if hai_rows:
        cats = pd.Series([r['real_category'] for r in hai_rows])
        print(f'  Categories found: {cats.value_counts().head(20).to_dict()}')
except Exception as e:
    print(f'  HackerAI failed: {e}')

# ============================================================
# COMBINE
# ============================================================
print('\n' + '='*60)
print('COMBINING')
print('='*60)

frames = [pd.DataFrame(x) for x in [circl_rows, tc_rows, alican_rows, hai_rows] if x]
if not frames:
    raise RuntimeError('No data loaded from any source')

df = pd.concat(frames, ignore_index=True)
df = df.drop_duplicates(subset=['sentence_text', 'real_category'])

# ============================================================
# REPORT
# ============================================================
print(f'\n📊 MASTER DATASET')
print('='*60)
print(f'Total rows: {len(df):,}')
print(f'\nBy source:')
print(df['source'].value_counts().to_string())
print(f'\nBy label_type:')
print(df['label_type'].value_counts().to_string())

print(f'\n--- MITRE techniques (from CIRCL) ---')
print(df[df['label_type']=='mitre_technique']['real_category'].value_counts().head(20).to_string())

print(f'\n--- Severities (from ThreatCluster) ---')
print(df[df['label_type']=='cvss_severity']['real_category'].value_counts().to_string())

print(f'\n--- CTI categories (from AlicanKiraz0 + HackerAI) ---')
cti = df[df['label_type']=='cti_category']
if len(cti) > 0:
    print(f'  Total: {len(cti):,} rows across {cti["real_category"].nunique()} categories')
    print(cti['real_category'].value_counts().head(30).to_string())
else:
    print('  ❌ No data from category-based datasets')

# ============================================================
# SAVE
# ============================================================
out = OUT_DIR / 'master_dataset.csv'
df.to_csv(out, index=False, encoding='utf-8-sig')
print(f'\n✅ Saved: {out}')
print(f'   Rows: {len(df):,}')