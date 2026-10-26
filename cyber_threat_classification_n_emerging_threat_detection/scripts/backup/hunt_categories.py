# scripts/hunt_categories.py
"""
Hunter: scan HuggingFace for CTI datasets, inspect their real label columns,
report every distinct category found. No assumptions. No forced mapping.
"""
import warnings
warnings.filterwarnings('ignore')

import sys
from collections import Counter
from datasets import load_dataset, get_dataset_config_names
from huggingface_hub import list_datasets

# ---------------------------------------------------------
# CONFIG — edit this list to add/remove candidates
# ---------------------------------------------------------
CANDIDATES = [
    # Original datasets
    'CIRCL/vulnerability-attack-techniques',
    'threatcluster/cve-exploitation-signals',
    # Newly found
    'Shomi28/cyber-threat-intelligence',
    'reloading0101/threat-intelligence-dataset',
    'DatasetSubmission/HackerSignal',
    'AI4Sec/cti-bench',
    'sarahwei/cyber_MITRE_CTI_dataset_v15',
    '11-47/GOT_Defense_1',
    'AlicanKiraz0/Cybersecurity-Dataset-Fenrir-v2.1',
    # Add more if you find them
    # 'Kaggle/...',
]

# Keywords that signal a category/label column
LABEL_HINTS = ['label', 'class', 'category', 'type', 'threat', 'attack',
               'severity', 'tactic', 'technique', 'cwe', 'cve']

# Keywords that signal the text column
TEXT_HINTS = ['text', 'content', 'description', 'sentence', 'body',
              'post', 'summary', 'prompt', 'instruction']

# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------
def find_columns(columns, hints):
    """Return columns whose name contains any hint."""
    return [c for c in columns if any(h in c.lower() for h in hints)]


def inspect_dataset(ds_id, max_configs=2, sample_size=500):
    """Load a dataset, inspect its label columns, return (labels_by_col, text_col)."""
    print(f'\n{"="*70}')
    print(f'DATASET: {ds_id}')
    print(f'{"="*70}')

    try:
        configs = get_dataset_config_names(ds_id)
    except Exception as e:
        print(f'  ❌ Cannot list configs: {e}')
        return {}, None

    if not configs:
        configs = [None]

    all_labels = {}
    chosen_text_col = None

    for cfg in configs[:max_configs]:
        print(f'\n  Config: {cfg!r}')
        try:
            ds = load_dataset(ds_id, cfg, split='train', streaming=True)
        except Exception as e:
            print(f'    ❌ Cannot load: {e}')
            continue

        cols = ds.column_names
        print(f'    Columns ({len(cols)}): {cols}')

        label_cols = find_columns(cols, LABEL_HINTS)
        text_cols  = find_columns(cols, TEXT_HINTS)
        print(f'    Label-like columns: {label_cols}')
        print(f'    Text-like columns : {text_cols}')

        if text_cols and not chosen_text_col:
            chosen_text_col = text_cols[0]

        # Sample rows and count values per label column
        counters = {c: Counter() for c in label_cols}
        total = 0
        for i, row in enumerate(ds):
            if i >= sample_size:
                break
            total += 1
            for c in label_cols:
                v = row.get(c)
                if v is None:
                    continue
                counters[c][str(v)] += 1

        print(f'    Sampled {total} rows')
        for c, cnt in counters.items():
            if not cnt:
                continue
            print(f'\n    --- Column "{c}" — {len(cnt)} distinct values ---')
            for val, freq in cnt.most_common(20):
                # Truncate long values
                v_disp = val if len(val) < 60 else val[:57] + '...'
                print(f'      {freq:>5}  {v_disp}')
            all_labels[c] = dict(cnt)

    return all_labels, chosen_text_col


def classify_columns(labels_by_col):
    """Given label counts, decide what kind of task each column supports."""
    verdicts = []
    for col, counts in labels_by_col.items():
        n_distinct = len(counts)
        total = sum(counts.values())
        if total == 0:
            continue

        # Heuristics
        if n_distinct == 2:
            kind = 'BINARY'
        elif n_distinct <= 10:
            kind = f'MULTI-CLASS ({n_distinct})'
        elif n_distinct <= 50:
            kind = f'MANY-CLASS ({n_distinct})'
        else:
            kind = f'HIGH-CARDINALITY ({n_distinct})'

        # Check if any value looks like a paper category
        paper_cats = ['ddos', 'botnet', 'malware', 'phishing', 'spam', 'ransomware']
        hits = []
        for v in counts:
            for cat in paper_cats:
                if cat in v.lower():
                    hits.append(cat)
                    break
        hits = sorted(set(hits))

        verdicts.append({
            'column': col,
            'distinct': n_distinct,
            'kind': kind,
            'paper_cats_found': hits,
        })
    return verdicts


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    print('='*70)
    print('HUNTING CATEGORIES IN HUGGINGFACE CTI DATASETS')
    print('='*70)

    results = {}
    for ds_id in CANDIDATES:
        labels, text_col = inspect_dataset(ds_id)
        results[ds_id] = {'labels': labels, 'text_col': text_col}

    # -----------------------------------------------------
    # FINAL SUMMARY
    # -----------------------------------------------------
    print('\n\n' + '='*70)
    print('FINAL SUMMARY — WHICH DATASETS HAVE WHAT')
    print('='*70)

    print(f'\n{"DATASET":<50} {"KIND":<20} {"PAPER CATS"}')
    print('-'*100)
    for ds_id, data in results.items():
        verdicts = classify_columns(data['labels'])
        if not verdicts:
            print(f'{ds_id:<50} {"(no labels)":<20}')
            continue
        for v in verdicts:
            cats = ','.join(v['paper_cats_found']) if v['paper_cats_found'] else '-'
            print(f'{ds_id:<50} {v["kind"]:<20} {cats}')

    # -----------------------------------------------------
    # BEST CANDIDATES FOR PAPER'S 6 CATEGORIES
    # -----------------------------------------------------
    print('\n' + '='*70)
    print('CANDIDATES FOR THE PAPER\'S 6 CATEGORIES')
    print('='*70)
    best = []
    for ds_id, data in results.items():
        for v in classify_columns(data['labels']):
            if len(v['paper_cats_found']) >= 2:
                best.append({
                    'dataset': ds_id,
                    'column': v['column'],
                    'n_paper_cats': len(v['paper_cats_found']),
                    'paper_cats': v['paper_cats_found'],
                    'n_distinct': v['distinct'],
                })
    best.sort(key=lambda x: (-x['n_paper_cats'], -x['n_distinct']))

    if not best:
        print('  ❌ No dataset has ≥2 of the paper\'s 6 categories.')
        print('  → Proceed with severity or MITRE technique tasks.')
    else:
        print(f'  {len(best)} dataset(s) with paper categories:\n')
        for b in best:
            print(f'  ⭐ {b["dataset"]}')
            print(f'     column={b["column"]}, classes={b["n_distinct"]}')
            print(f'     paper categories found: {b["paper_cats"]}')
            print()

    # -----------------------------------------------------
    # FULL DUMP OF EVERY LABEL
    # -----------------------------------------------------
    print('\n' + '='*70)
    print('FULL LABEL DUMP (all datasets, all label columns)')
    print('='*70)
    for ds_id, data in results.items():
        if not data['labels']:
            continue
        print(f'\n### {ds_id}')
        for col, counts in data['labels'].items():
            print(f'\n  --- {col} ({len(counts)} distinct) ---')
            for val, freq in sorted(counts.items(), key=lambda x: -x[1])[:15]:
                v_disp = val if len(val) < 70 else val[:67] + '...'
                print(f'    {freq:>6}  {v_disp}')


if __name__ == '__main__':
    main()