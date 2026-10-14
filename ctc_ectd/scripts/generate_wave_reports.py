# scripts/generate_wave_reports.py
"""
Add a cluster of same-category reports across adjacent months to
demonstrate emerging-threat detection.

Wave 1: Phishing        — 2025-09, 2025-10, 2025-11  (rising)
Wave 2: Ransomware      — 2026-01, 2026-02, 2026-03  (rising)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_sample_pdfs import build_pdf, UPLOAD_DIR


PHISHING_WAVE = {
    "title": "Escalating Phishing Wave Against Banking Sector",
    "category": "Phishing",
    "severity": "High",
    "sector": "Banking",
    "country": "United States",
    "actor": "APT-Alpha",
    "malware": "ShadowRAT",
    "vector": "Spearphishing Link",
    "cve": "CVE-2025-44444",
    "mitre": "T1566.002",
    "mitre_name": "Spearphishing Link",
    "text": (
        "A sustained phishing wave was observed against banking customers. "
        "Adversary-in-the-middle pages harvested credentials and MFA tokens. "
        "Spearphishing links redirected victims to lookalike SSO portals. "
        "Volume increased sharply versus prior months, with new C2 "
        "infrastructure rotated every 48 hours. Mitigation: phishing-resistant "
        "MFA, conditional access, rapid session revocation."
    ),
}

RANSOMWARE_WAVE = {
    "title": "Ransomware Surge Against Healthcare Providers",
    "category": "Ransomware",
    "severity": "Critical",
    "sector": "Healthcare",
    "country": "United Kingdom",
    "actor": "NightWolf",
    "malware": "RedLineX",
    "vector": "Exposed Service",
    "cve": "CVE-2024-22222",
    "mitre": "T1486",
    "mitre_name": "Data Encrypted for Impact",
    "text": (
        "A renewed ransomware surge targeted healthcare providers. Operators "
        "exploited unpatched VPN appliances, exfiltrated patient records, and "
        "encrypted backups. Ransom demands escalated compared to earlier "
        "campaigns. Multiple victims reported the same RedLineX payload "
        "variant. Mitigation: patch edge appliances, isolate backups, "
        "monitor for T1486 behaviour."
    ),
}

WAVES = [
    (PHISHING_WAVE, ["2025-09", "2025-10", "2025-11"]),
    (RANSOMWARE_WAVE, ["2026-01", "2026-02", "2026-03"]),
]


def main():
    total = 0
    for template, months in WAVES:
        slug = template["category"].lower()
        for i, month in enumerate(months, 1):
            rep = dict(template, date=month)
            fname = f"{month}_{slug}_wave_{i:02d}.pdf"
            path = UPLOAD_DIR / fname
            build_pdf(rep, path)
            print(f"  ✓ {fname}")
            total += 1
    print(f"\n✅ Added {total} wave reports to {UPLOAD_DIR}")


if __name__ == "__main__":
    main()