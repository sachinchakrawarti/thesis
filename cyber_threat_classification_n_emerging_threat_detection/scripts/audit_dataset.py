# scripts/audit_dataset.py
"""Complete audit of cti_unified.csv — every column, every task."""
import pandas as pd
import numpy as np
from pathlib import Path

df = pd.read_csv('data/processed/cti_unified.csv')

print('=' * 70)
print('FULL DATASET AUDIT')
print('=' * 70)
print(f'File     : data/processed/cti_unified.csv')
print(f'Rows     : {len(df):,}')
print(f'Columns  : {len(df.columns)}')
print(f'Memory   : {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB')

# ============================================================
# 1. COLUMN-BY-COLUMN MISSINGNESS
# ============================================================
print('\n' + '=' * 70)
print('1. COLUMN-BY-COLUMN COMPLETENESS')
print('=' * 70)
print(f'{"Column":<18} {"Non-null":>10} {"Null":>10} {"% Filled":>10} {"Type":<12}')
print('-' * 70)

for col in df.columns:
    if df[col].dtype == 'object':
        # Empty strings count as null
        nonnull = df[col].astype(str).str.strip().ne('').sum()
    else:
        nonnull = df[col].notna().sum()
    null = len(df) - nonnull
    pct = nonnull / len(df) * 100
    dtype = str(df[col].dtype)[:12]
    print(f'{col:<18} {nonnull:>10,} {null:>10,} {pct:>9.1f}% {dtype:<12}')

# ============================================================
# 2. TASK A — BINARY RELEVANCE
# ============================================================
print('\n' + '=' * 70)
print('2. TASK A — BINARY RELEVANCE (target: relevant)')
print('=' * 70)

bin_df = df[df['relevant'].notna()].copy()
print(f'Usable rows: {len(bin_df):,}')
print(f'\nDistribution:')
print(bin_df['relevant'].value_counts().to_string())

yes = (bin_df['relevant'] == 'yes').sum()
no = (bin_df['relevant'] == 'no').sum()
if no > 0:
    print(f'\nImbalance ratio: {no/yes:.1f}:1 (no:yes)')
    print(f'Majority baseline accuracy: {no/len(bin_df)*100:.2f}%')
    
    # Rebalanced target
    target_yes = yes * 20  # 20x oversample
    print(f'\nRebalanced (20× yes): {target_yes:,} yes + {min(target_yes, no):,} no')
    print(f'Rebalanced total: {target_yes + min(target_yes, no):,} rows')

# ============================================================
# 3. TASK B — SEVERITY CLASSIFICATION
# ============================================================
print('\n' + '=' * 70)
print('3. TASK B — 4-CLASS SEVERITY (target: severity)')
print('=' * 70)

sev_df = df[df['severity'].notna()].copy()
print(f'Usable rows: {len(sev_df):,}')
print(f'\nDistribution:')
print(sev_df['severity'].value_counts().to_string())

for cls in ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']:
    n = (sev_df['severity'] == cls).sum()
    pct = n / len(sev_df) * 100 if len(sev_df) > 0 else 0
    print(f'  {cls:10s}: {n:>6,} ({pct:.1f}%)')

# ============================================================
# 4. TASK C — ATT&CK TECHNIQUE
# ============================================================
print('\n' + '=' * 70)
print('4. TASK C — ATT&CK TECHNIQUE (target: primary_tech)')
print('=' * 70)

# Real ATT&CK rows (contain T-pattern)
att_mask = df['techniques'].astype(str).str.contains(r'T\d', na=False)
att_df = df[att_mask].copy()
print(f'Rows with ATT&CK IDs: {len(att_df):,}')

# Extract parent technique
def extract_primary(tech_str):
    if not tech_str or str(tech_str).strip() == '':
        return ''
    first = str(tech_str).split(';')[0].strip()
    return first.split('.')[0] if first else ''

att_df['parent_tech'] = att_df['techniques'].apply(extract_primary)

# Count per technique
tech_counts = att_df['parent_tech'].value_counts()
print(f'\nAll techniques ({len(tech_counts)} distinct):')
print(tech_counts.head(20).to_string())

# Top-8 filter
TOP8 = ['T1059', 'T1190', 'T1005', 'T1068', 'T1203', 'T1189', 'T1499', 'T1078']
top8_df = att_df[att_df['parent_tech'].isin(TOP8)]
print(f'\nTop-8 filter ({len(top8_df):,} rows):')
for tech in TOP8:
    n = (top8_df['parent_tech'] == tech).sum()
    print(f'  {tech:8s}: {n:>4,}')

# ============================================================
# 5. TASK D — TEMPORAL / LDA
# ============================================================
print('\n' + '=' * 70)
print('5. TASK D — TEMPORAL (target: published)')
print('=' * 70)

date_df = df[df['published'].notna() & (df['published'].astype(str).str.strip() != '')].copy()
print(f'Rows with date: {len(date_df):,}')

if len(date_df) > 0:
    date_df['published'] = pd.to_datetime(date_df['published'], errors='coerce')
    date_df = date_df[date_df['published'].notna()]
    print(f'Valid dates: {len(date_df):,}')
    print(f'Date range: {date_df["published"].min()} to {date_df["published"].max()}')
    
    date_df['year'] = date_df['published'].dt.year
    print(f'\nRows per year:')
    print(date_df['year'].value_counts().sort_index().to_string())
    
    date_df['month'] = date_df['published'].dt.to_period('M').astype(str)
    unique_months = date_df['month'].nunique()
    print(f'\nUnique months: {unique_months}')
    print(f'Months span: {date_df["month"].min()} to {date_df["month"].max()}')

# ============================================================
# 6. OTHER COLUMNS
# ============================================================
print('\n' + '=' * 70)
print('6. OTHER COLUMNS')
print('=' * 70)

# EPSS
print(f'EPSS:')
epss_vals = pd.to_numeric(df['epss'], errors='coerce')
print(f'  Valid: {epss_vals.notna().sum():,}')
print(f'  Range: {epss_vals.min():.4f} to {epss_vals.max():.4f}')
print(f'  Mean:  {epss_vals.mean():.4f}')

# KEV
print(f'\nKEV:')
print(f'  in_kev True: {(df["in_kev"] == True).sum():,}')  # noqa: E712
print(f'  kev_ransomware True: {(df["kev_ransomware"] == True).sum():,}')  # noqa: E712

# CWE
print(f'\nCWE:')
has_cwe = df['cwe_ids'].astype(str).str.strip().ne('').sum()
print(f'  Rows with CWE: {has_cwe:,} ({has_cwe/len(df)*100:.1f}%)')

# CVSS
print(f'\nCVSS score:')
cvss = pd.to_numeric(df['cvss_score'], errors='coerce')
print(f'  Valid: {cvss.notna().sum():,}')
print(f'  Range: {cvss.min():.1f} to {cvss.max():.1f}')
print(f'  Mean:  {cvss.mean():.2f}')

# Sources
print(f'\nSources:')
print(df['source'].value_counts().to_string())

# ============================================================
# 7. FINAL VERDICT
# ============================================================
print('\n' + '=' * 70)
print('7. FINAL VERDICT — TASK READINESS')
print('=' * 70)

checks = {
    'Binary': (len(bin_df) >= 10000) and yes >= 500,
    'Severity': len(sev_df) >= 10000,
    'ATT&CK top-8': len(top8_df) >= 500,
    'LDA (dates)': len(date_df) >= 10000,
}

for task, ok in checks.items():
    status = '✅ READY' if ok else '❌ INSUFFICIENT'
    print(f'{task:<20s}: {status}')

print('\n' + '=' * 70)
print('AUDIT COMPLETE')
print('=' * 70)