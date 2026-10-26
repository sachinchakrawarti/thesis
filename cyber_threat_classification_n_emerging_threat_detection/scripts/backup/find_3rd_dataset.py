# scripts/find_3rd_dataset.py
"""Search HuggingFace for a dataset containing the paper's 6 threat categories."""
from huggingface_hub import list_datasets
import requests

TARGET_LABELS = ['ddos', 'botnet', 'malware', 'phishing', 'spam', 'ransomware']
QUERIES = [
    'cyber threat intelligence', 'threat classification', 'threat type',
    'attack classification', 'malware classification', 'phishing',
    'ransomware', 'ddos', 'botnet', 'spam',
    'hacker forum', 'darkweb', 'dark web', 'nulled',
    'cybersecurity', 'cti', 'apt', 'security posts',
    'network intrusion', 'iot attack',
]

print('=' * 75)
print('SEARCHING HUGGINGFACE FOR 6-CLASS THREAT DATASETS')
print('=' * 75)

all_results = {}
for q in QUERIES:
    print(f'\n--- Query: "{q}" ---')
    try:
        for ds in list_datasets(search=q, limit=25):
            ds_id = ds.id
            # fetch full metadata to inspect description/tags
            try:
                info = requests.get(
                    f'https://huggingface.co/api/datasets/{ds_id}',
                    timeout=5
                ).json()
            except Exception:
                info = {}
            desc = (info.get('cardData') or {}).get('description', '') or ''
            tags = ' '.join(info.get('tags', []) or [])
            blob = (ds_id + ' ' + desc + ' ' + tags).lower()

            hits = [t for t in TARGET_LABELS if t in blob]
            if len(hits) >= 2:
                all_results[ds_id] = {
                    'hits': hits,
                    'n_hits': len(hits),
                    'downloads': info.get('downloads', 0),
                    'description': desc[:150],
                }
                print(f'  ✅ {ds_id:65s} labels={hits}')
    except Exception as e:
        print(f'  ❌ error: {e}')

print('\n' + '=' * 75)
print(f'FOUND {len(all_results)} CANDIDATES')
print('=' * 75)

# Group by how many target labels matched
for n in [6, 5, 4, 3, 2]:
    group = {k: v for k, v in all_results.items() if v['n_hits'] == n}
    if group:
        print(f'\n### {n}/6 labels matched ({len(group)} datasets):')
        for ds_id, v in sorted(group.items(), key=lambda x: -x[1]['downloads']):
            print(f'  {ds_id:65s}  labels={v["hits"]}  downloads={v["downloads"]}')
            if v['description']:
                print(f'    → {v["description"]}')

print('\n\n=== TOP 10 TO VERIFY FIRST ===')
top10 = sorted(all_results.items(), key=lambda x: (-x[1]['n_hits'], -x[1]['downloads']))[:10]
for ds_id, v in top10:
    print(f'  python scripts/verify_3rd.py {ds_id}')