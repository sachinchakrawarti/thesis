#!/usr/bin/env python3
"""Create up to 15,000 unique CVE records using official NVD, CISA KEV and FIRST EPSS.

pip install requests
python build_15000_verified_cves.py --target 15000
Optional: set NVD_API_KEY environment variable for faster NVD API requests.
No ATT&CK technique labels are guessed; supply verified mappings separately.
"""
import argparse
import csv
import datetime as dt
import json
import os
import re
import time
from pathlib import Path
import requests

NVD = 'https://services.nvd.nist.gov/rest/json/cves/2.0'
KEV = 'https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json'
EPSS = 'https://api.first.org/data/v1/epss'
FIELDS = ['cve_id','sentence_text','source','published','relevant','severity','cvss_score','techniques','primary_tech','tech_name','cwe_ids','in_kev','kev_ransomware','epss']

def get_json(s, url, params=None):
    for attempt in range(7):
        try:
            response = s.get(url, params=params, timeout=100)
            if response.status_code in (429, 500, 502, 503, 504):
                wait = min(120, 5 * 2**attempt)
                print(f'HTTP {response.status_code}; retry in {wait}s: {url}', flush=True)
                time.sleep(wait)
                continue
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError):
            if attempt == 6:
                raise
            time.sleep(min(90, 3 * 2**attempt))
    raise RuntimeError('Request failed: ' + url)

def parse_cve(c, kev):
    cid = c.get('id', '')
    if not re.fullmatch(r'CVE-\d{4}-\d{4,}', cid):
        return None
    if c.get('vulnStatus', '').lower() in ('rejected', 'rejected or withdrawn'):
        return None
    description = next((d.get('value', '').strip() for d in c.get('descriptions', []) if d.get('lang') == 'en'), '')
    if not description or description.startswith('** REJECT **'):
        return None
    score, severity = None, ''
    metrics = c.get('metrics', {})
    # Prefer CVSS 3.1/3.0 for consistent four-class classification; use 4.0 as fallback.
    for version in ('cvssMetricV31','cvssMetricV30','cvssMetricV40'):
        candidates = metrics.get(version) or []
        if candidates:
            item = next((m for m in candidates if m.get('type') == 'Primary'), candidates[0])
            data = item.get('cvssData', {})
            score = data.get('baseScore')
            severity = (data.get('baseSeverity') or item.get('baseSeverity') or '').upper()
            break
    if score is not None and not severity:
        score = float(score)
        severity = 'NONE' if score == 0 else 'LOW' if score < 4 else 'MEDIUM' if score < 7 else 'HIGH' if score < 9 else 'CRITICAL'
    # NONE is not one of the requested four classes: leave it unlabeled.
    if severity not in ('LOW','MEDIUM','HIGH','CRITICAL'):
        severity = ''
    cwes = sorted({d.get('value') for w in c.get('weaknesses', []) for d in w.get('description', []) if re.fullmatch(r'CWE-\d+', d.get('value', ''))})
    kev_record = kev.get(cid)
    ransomware = (kev_record or {}).get('knownRansomwareCampaignUse', '') == 'Known'
    return {
        'cve_id': cid, 'sentence_text': description, 'source': 'nvd',
        'published': c.get('published', '')[:10],
        'relevant': 'yes' if kev_record is not None or severity == 'CRITICAL' else 'no',
        'severity': severity, 'cvss_score': '' if score is None else score,
        'techniques': '', 'primary_tech': '', 'tech_name': '',
        'cwe_ids': ';'.join(cwes), 'in_kev': kev_record is not None,
        'kev_ransomware': ransomware, 'epss': ''
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', type=int, default=15000)
    ap.add_argument('--out', default='verified_cve_15000.csv')
    ap.add_argument('--earliest-year', type=int, default=2018)
    ap.add_argument('--require-severity', action='store_true', help='Exclude CVEs without a four-class CVSS 3.x/4.x severity')
    ap.add_argument('--nvd-api-key', default=os.getenv('NVD_API_KEY'))
    args = ap.parse_args()
    if args.target < 1 or args.earliest_year < 1999:
        ap.error('Invalid target or earliest year')
    session = requests.Session()
    session.headers.update({'User-Agent':'AcademicVerifiedCveDataset/1.0', 'Accept':'application/json'})
    if args.nvd_api_key:
        session.headers['apiKey'] = args.nvd_api_key
    kev_payload = get_json(session, KEV)
    kev = {x['cveID']: x for x in kev_payload.get('vulnerabilities', []) if x.get('cveID')}
    rows = {}
    end = dt.datetime.now(dt.timezone.utc)
    earliest = dt.datetime(args.earliest_year, 1, 1, tzinfo=dt.timezone.utc)
    while len(rows) < args.target and end > earliest:
        start = max(earliest, end - dt.timedelta(days=119))
        params = {'pubStartDate':start.strftime('%Y-%m-%dT%H:%M:%S.000Z'),
                  'pubEndDate':end.strftime('%Y-%m-%dT%H:%M:%S.000Z'),
                  'resultsPerPage':2000, 'startIndex':0}
        print(f'Fetching NVD {start.date()} to {end.date()} | {len(rows)}/{args.target}', flush=True)
        while len(rows) < args.target:
            payload = get_json(session, NVD, params)
            items = payload.get('vulnerabilities', [])
            if not items:
                break
            for item in items:
                row = parse_cve(item.get('cve', {}), kev)
                if row and (not args.require_severity or row['severity']):
                    rows.setdefault(row['cve_id'], row)
                    if len(rows) >= args.target:
                        break
            params['startIndex'] += len(items)
            if params['startIndex'] >= payload.get('totalResults', 0):
                break
            time.sleep(0.7 if args.nvd_api_key else 6.2)
        end = start - dt.timedelta(milliseconds=1)
        time.sleep(0.7 if args.nvd_api_key else 6.2)
    ids = list(rows)
    print(f'Enriching {len(ids)} CVEs with FIRST EPSS...', flush=True)
    for i in range(0, len(ids), 100):
        data = get_json(session, EPSS, {'cve':','.join(ids[i:i+100])})
        for item in data.get('data', []):
            cid = item.get('cve')
            if cid in rows:
                rows[cid]['epss'] = item.get('epss', '')
        if i % 1000 == 0:
            print(f'EPSS: {min(i+100,len(ids))}/{len(ids)}', flush=True)
        time.sleep(0.2)
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows.values())
    manifest = {
        'created_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
        'row_count':len(rows), 'requested_count':args.target,
        'sources':{'nvd':NVD,'cisa_kev':KEV,'first_epss':EPSS},
        'definitions':{
            'relevant':'yes iff in CISA KEV at collection or CVSS severity CRITICAL; heuristic, not ground truth',
            'kev_ransomware':'true iff CISA KEV knownRansomwareCampaignUse is Known; Unknown is encoded false in this requested schema',
            'epss':'latest available EPSS at collection, not necessarily publication-time EPSS',
            'techniques':'intentionally blank: NVD does not supply verified CVE-to-ATT&CK mappings',
            'severity':'CVSS v3.1 preferred, then v3.0, then v4.0; blank if missing or out of four-class scope',
            'source':'nvd; enrichment from CISA and FIRST',
            'published':'NVD publication date in UTC'},
        'limitations':['Collection is most-recent-first, not random or class-balanced.',
                       'The 15,000 target is reached only if enough valid CVEs are available.',
                       'KEV and EPSS are current snapshots: avoid future-information leakage in historical evaluation.',
                       'No verified ATT&CK technique training labels are produced.',
                       'Boolean kev_ransomware false can mean CISA says Unknown, not confirmed absence.']}
    dest.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'Saved {len(rows)} unique CVEs to {dest.resolve()}')
    if len(rows) != args.target:
        print('WARNING: requested target not reached; check --earliest-year and --require-severity')

if __name__ == '__main__':
    main()
