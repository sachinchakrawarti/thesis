# scripts/check_data.py
import pandas as pd
from pathlib import Path

p = Path('dataset/cti_dataset_clean.csv')
if not p.exists():
    hits = list(Path('.').rglob('cti_dataset_clean.csv'))
    if hits:
        p = hits[0]
        print('found:', p)
    else:
        raise FileNotFoundError('cti_dataset_clean.csv not found. Run Notebook 01 first.')

df = pd.read_csv(p)
print(f'\nLoaded: {df.shape[0]:,} rows x {df.shape[1]} cols')
print('Columns:', df.columns.tolist())

for col in ['topic_cluster','techniques','severity','zero_day',
            'tactic','source','procedure','relevant']:
    print(f'\n=== {col} ===')
    if col not in df.columns:
        print('  (missing)')
        continue
    nn = df[col].notna().sum()
    print(f'  non-null: {nn:,} / {len(df):,} ({nn/len(df)*100:.1f}%)')
    if nn > 0:
        print(df[col].value_counts().head(15).to_string())

if 'doc_id' in df.columns:
    print(f'\ndoc_id unique groups: {df["doc_id"].nunique():,}')