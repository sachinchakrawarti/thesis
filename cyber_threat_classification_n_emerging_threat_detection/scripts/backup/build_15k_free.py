# scripts/build_15k_free.py
import json, gzip, io, random, requests, pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CVELIST = PROJECT_ROOT / 'dataset' / 'cvelistV5'
OUT = PROJECT_ROOT / 'dataset' / 'final_cti_15k.csv'
HEADERS = {'User-Agent': 'Thesis-Research/1.0'}
TARGET = 15000
random.seed(42)

# 1. Parse CVEListV5 (free, local files)
print('[1/4] Parsing local CVEListV5...')
rows = []
for fp in CVELIST.glob('cves/**/CVE-*.json'):
    try:
        obj = json.load(open(fp, encoding='utf-8'))
        meta = obj.get('cveMetadata', {})
        if meta.get('state') != 'PUBLISHED':
            continue
        cna = obj.get('containers', {}).get('cna', {})
        desc = next((d['value'] for d in cna.get('descriptions', []) if d.get('lang','').startswith('en')), '')
        if not desc or len(desc) < 20:
            continue
        sev, score = '', None
        for metric in cna.get('metrics', []):
            for k in ['cvssV3_1','cvssV3_0','cvssV4_0']:
                if k in metric:
                    sev = (metric[k].get('baseSeverity','') or '').upper()
                    score = metric[k].get('baseScore')
                    if sev in ('LOW','MEDIUM','HIGH','CRITICAL'): break
            if sev in ('LOW','MEDIUM','HIGH','CRITICAL'): break
        if sev not in ('LOW','MEDIUM','HIGH','CRITICAL'):
            continue
        rows.append({'cve_id': meta['cveId'], 'sentence_text': desc, 'source': 'cvelistv5',
                     'published': (meta.get('datePublished','') or '')[:10],
                     'severity': sev, 'cvss_score': score})
    except Exception:
        continue
print(f'  Parsed {len(rows):,} valid CVEs')

df = pd.DataFrame(rows).drop_duplicates('cve_id')

# 2. Free enrichments (KEV + EPSS)
print('[2/4] Fetching CISA KEV...')
kev = {x['cveID']: x for x in requests.get('https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json', headers=HEADERS).json().get('vulnerabilities', [])}
print(f'  {len(kev):,} KEV entries')

print('[3/4] Fetching EPSS...')
r = requests.get('https://epss.empiricalsecurity.com/epss_scores-current.csv.gz', headers=HEADERS)
with gzip.open(io.BytesIO(r.content), 'rt') as f:
    epss_df = pd.read_csv(f, skiprows=1)
epss = dict(zip(epss_df['cve'], epss_df['epss'].astype(float)))

# 3. Enrich
df['in_kev'] = df['cve_id'].map(lambda x: x in kev)
df['kev_ransomware'] = df['cve_id'].map(lambda x: (kev.get(x,{}).get('knownRansomwareCampaignUse','') or '').lower() == 'known')
df['epss'] = df['cve_id'].map(epss)
df['relevant'] = df.apply(lambda r: 'yes' if (r['in_kev'] or r['kev_ransomware'] or r['severity']=='CRITICAL') else 'no', axis=1)
df['techniques'] = ''; df['primary_tech'] = ''; df['tech_name'] = ''
df['cwe_ids'] = ''  # CWE parsing omitted for brevity; add from JSON if needed

# 4. Sample to 15,000 and save
df = df.sample(n=min(TARGET, len(df)), random_state=42)
COLS = ['cve_id','sentence_text','source','published','relevant','severity','cvss_score',
        'techniques','primary_tech','tech_name','cwe_ids','in_kev','kev_ransomware','epss']
df = df[COLS]
df.to_csv(OUT, index=False, encoding='utf-8-sig')
print(f'\n✅ Saved {len(df):,} rows to {OUT}')
print(df['severity'].value_counts())
print(df['relevant'].value_counts())