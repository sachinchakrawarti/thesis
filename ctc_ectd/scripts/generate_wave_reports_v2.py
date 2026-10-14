# scripts/generate_wave_reports_v2.py
"""
Generate realistic multi-doc-per-month clusters so the emerging-threat
detector has genuine share fractions to trend.

Scenario:
    Baseline months: 1 phishing + 1 ransomware + 1 botnet  (share = 1/3 each)
    Phishing ramps up over 3 months → share climbs to 4/6, 5/6, 6/6
    Then ransomware ramps up over 3 months → its share climbs

Clean up old PDFs first (optional):  del uploads\*.pdf
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_sample_pdfs import build_pdf, UPLOAD_DIR


TEMPLATES = {
    "phishing": {
        "title": "Phishing Campaign", "category": "Phishing",
        "severity": "High", "sector": "Banking", "country": "United States",
        "actor": "APT-Alpha", "malware": "ShadowRAT",
        "vector": "Spearphishing Link", "cve": "CVE-2025-44444",
        "mitre": "T1566.002", "mitre_name": "Spearphishing Link",
        "text": ("Phishing emails targeting banking customers harvested "
                 "credentials and MFA tokens through adversary-in-the-middle "
                 "pages. Spearphishing links redirected victims to lookalike "
                 "SSO portals. New C2 infrastructure was rotated frequently."),
    },
    "ransomware": {
        "title": "Ransomware Incident", "category": "Ransomware",
        "severity": "Critical", "sector": "Healthcare",
        "country": "United Kingdom", "actor": "NightWolf",
        "malware": "RedLineX", "vector": "Exposed Service",
        "cve": "CVE-2024-22222", "mitre": "T1486",
        "mitre_name": "Data Encrypted for Impact",
        "text": ("Ransomware operators exploited unpatched VPN appliances, "
                 "exfiltrated patient records, then encrypted backups and "
                 "file shares. RedLineX was the deployed payload."),
    },
    "botnet": {
        "title": "Botnet Activity", "category": "Botnet",
        "severity": "Medium", "sector": "Telecom", "country": "Brazil",
        "actor": "ShadowNet", "malware": "Mirai-Variant",
        "vector": "Exploit Kit", "cve": "CVE-2025-11111",
        "mitre": "T1583.005", "mitre_name": "Botnet",
        "text": ("IoT routers and cameras recruited into a DDoS-for-hire "
                 "botnet. Twenty-seven C2 domains and twelve hardcoded IPs "
                 "were seized during takedown."),
    },
    "ddos": {
        "title": "DDoS Attack", "category": "DDoS",
        "severity": "High", "sector": "Government", "country": "Germany",
        "actor": "StormByte", "malware": "SilverWorm",
        "vector": "Botnet", "cve": "N/A", "mitre": "T1498",
        "mitre_name": "Network Denial of Service",
        "text": ("Volumetric DDoS peaked at 620 Gbps against government "
                 "portals. DNS amplification and HTTP floods were used."),
    },
    "credential_theft": {
        "title": "Credential Theft", "category": "Credential Theft",
        "severity": "High", "sector": "Financial Services",
        "country": "Singapore", "actor": "DarkHydra",
        "malware": "DarkLoader", "vector": "Drive-by Download",
        "cve": "CVE-2024-33333", "mitre": "T1555.003",
        "mitre_name": "Credentials from Web Browsers",
        "text": ("Infostealer scraped browser credentials, session cookies, "
                 "and crypto wallet files. Exfiltration to C2 over HTTPS."),
    },
}

# (month, [topic, topic, ...])
# Phishing share: 1/3 → 1/3 → 2/4 → 3/5 → 4/5 → 5/5  (clear rise)
# Ransomware then ramps up: 1/3 → 2/4 → 3/4 → 4/4
PLAN = [
    ("2026-01", ["phishing", "ransomware", "botnet"]),
    ("2026-02", ["phishing", "ransomware", "botnet"]),
    ("2026-03", ["phishing", "phishing", "ransomware", "ddos"]),
    ("2026-04", ["phishing", "phishing", "phishing", "ransomware", "botnet"]),
    ("2026-05", ["phishing", "phishing", "phishing", "phishing",
                 "ransomware"]),
    ("2026-06", ["phishing", "phishing", "phishing", "phishing", "phishing"]),
    ("2026-07", ["ransomware", "ransomware", "phishing", "botnet"]),
    ("2026-08", ["ransomware", "ransomware", "ransomware", "phishing"]),
    ("2026-09", ["ransomware", "ransomware", "ransomware", "ransomware"]),
]


def main():
    n = 0
    for month, topics in PLAN:
        for i, topic in enumerate(topics, 1):
            rep = dict(TEMPLATES[topic], date=month)
            fname = f"{month}_{topic}_cluster_{i:02d}.pdf"
            build_pdf(rep, UPLOAD_DIR / fname)
            print(f"  ✓ {fname}")
            n += 1
    print(f"\n✅ Added {n} multi-doc cluster reports")


if __name__ == "__main__":
    main()