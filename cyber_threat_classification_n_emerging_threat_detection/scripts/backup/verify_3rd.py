# scripts/verify_3rd.py
"""Verify a HuggingFace dataset actually contains the 6 threat labels."""
import sys
from datasets import load_dataset, get_dataset_config_names
from collections import Counter

TARGET = {'ddos', 'botnet', 'malware', 'phishing', 'spam', 'ransomware'}

def verify(ds_id):
    print(f'\n{"="*75}\n{ds_id}\n{"="*75}')
    try:
        configs = get_dataset_config_names(ds_id)
        config = configs[0] if configs else None
        print(f'configs: {configs[:5]}')
        ds = load_dataset(ds_id, config, split='train', streaming=True)
    except Exception as e:
        print(f'❌ cannot load: {e}')
        return

    print(f'columns: {ds.column_names}')

    # sample and count target labels
    label_hits = Counter()
    text_cols = set()
    sample = []
    for i, row in enumerate(ds):
        if i >= 3000:
            break
        for col in ds.column_names:
            v = row.get(col)
            if isinstance(v, str):
                vl = v.lower().strip()
                if vl in TARGET:
                    label_hits[col] += 1
                if len(v) > 50:
                    text_cols.add(col)
        if i < 3:
            sample.append(row)

    print(f'\n--- Label hits in first 3000 rows ---')
    if label_hits:
        print(f'✅ LABELS FOUND:')
        for col, cnt in label_hits.most_common():
            print(f'   column "{col}": {cnt} matching rows')
    else:
        print('❌ No direct match with target labels')
        print(f'   text columns: {text_cols}')
        # check unique values in likely label columns
        for col in ds.column_names:
            if any(k in col.lower() for k in ['label', 'class', 'type', 'category']):
                try:
                    vals = set()
                    ds2 = load_dataset(ds_id, config, split='train', streaming=True)
                    for j, r in enumerate(ds2):
                        if j >= 200: break
                        vals.add(str(r.get(col)))
                    print(f'   column "{col}" sample values: {list(vals)[:10]}')
                except Exception:
                    pass

    print(f'\n--- Sample rows ---')
    for i, s in enumerate(sample):
        print(f'row {i}: {dict(s)}')

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('usage: python scripts/verify_3rd.py <dataset_id> [<id2> ...]')
        sys.exit(1)
    for ds_id in sys.argv[1:]:
        verify(ds_id)