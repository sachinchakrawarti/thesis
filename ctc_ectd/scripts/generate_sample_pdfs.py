# scripts/generate_sample_pdfs.py
"""
Generate 10 realistic synthetic CTI PDF reports in uploads/.

Requires:
    pip install reportlab
"""

from pathlib import Path
from datetime import datetime
import random

try:
    from reportlab.lib.pagesizes import LETTER
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib.enums import TA_JUSTIFY
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    )
    from reportlab.lib import colors
except ImportError:
    raise SystemExit(
        "Install reportlab first:  pip install reportlab"
    )


PROJECT_ROOT = Path(r"D:\Sachin Chakrawarti\Learn\done\thesis\ctc_ectd")
UPLOAD_DIR = PROJECT_ROOT / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)


# ------------------------------------------------------------------
# 10 hand-crafted CTI scenarios (dates go from old to recent so the
# emerging-threat detector has something to trend on)
# ------------------------------------------------------------------
REPORTS = [
    {
        "date": "2024-01",
        "title": "Phishing Campaign Targeting Banking Sector",
        "category": "Phishing",
        "severity": "High",
        "sector": "Banking",
        "country": "United States",
        "actor": "APT-Alpha",
        "malware": "ShadowRAT",
        "vector": "Spearphishing Attachment",
        "cve": "CVE-2024-11111",
        "mitre": "T1566.001",
        "mitre_name": "Spearphishing Attachment",
        "text": (
            "A targeted phishing campaign was observed against multiple banking "
            "institutions across the United States. The threat actor APT-Alpha "
            "delivered malicious Microsoft Word documents via spearphishing emails. "
            "The attachments exploited CVE-2024-11111 to drop ShadowRAT, giving the "
            "actor remote control of victim machines. Analysts identified credential "
            "harvesting pages mimicking the victims' SSO login screens. Mitigation: "
            "block the identified sender domains, enable MFA, and deploy attachment "
            "sandboxing."
        ),
    },
    {
        "date": "2024-03",
        "title": "Ransomware Deployment Against Healthcare Providers",
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
            "NightWolf ransomware operators exploited an unpatched VPN appliance "
            "(CVE-2024-22222) at multiple UK healthcare providers. After gaining "
            "initial access, the group deployed RedLineX to exfiltrate patient data "
            "before encrypting file shares and backup servers. Ransom notes demanded "
            "payment in Monero. Affected systems included electronic health record "
            "portals and imaging databases. Mitigation: patch VPN appliances, segment "
            "backups, and monitor for T1486 behaviour."
        ),
    },
    {
        "date": "2024-05",
        "title": "Credential Theft via Infostealer Targeting Financial Services",
        "category": "Credential Theft",
        "severity": "High",
        "sector": "Financial Services",
        "country": "Singapore",
        "actor": "DarkHydra",
        "malware": "DarkLoader",
        "vector": "Drive-by Download",
        "cve": "CVE-2024-33333",
        "mitre": "T1555.003",
        "mitre_name": "Credentials from Web Browsers",
        "text": (
            "DarkHydra deployed DarkLoader via compromised advertising networks "
            "to steal credentials from financial services employees in Singapore. "
            "The loader injected into browsers to scrape saved passwords, session "
            "cookies, and cryptocurrency wallet files. Analysts observed exfiltration "
            "to attacker-controlled C2 over HTTPS with domain fronting. Mitigation: "
            "block the identified C2 domains, enforce browser credential isolation, "
            "and rotate compromised sessions."
        ),
    },
    {
        "date": "2024-07",
        "title": "DDoS Attack Against Government Services",
        "category": "DDoS",
        "severity": "High",
        "sector": "Government",
        "country": "Germany",
        "actor": "StormByte",
        "malware": "SilverWorm",
        "vector": "Botnet",
        "cve": "N/A",
        "mitre": "T1498",
        "mitre_name": "Network Denial of Service",
        "text": (
            "A volumetric DDoS attack disrupted public-facing German government "
            "portals for approximately four hours. StormByte coordinated a botnet of "
            "IoT devices and compromised web servers, peaking at 620 Gbps. Attack "
            "vectors included DNS amplification, NTP reflection, and HTTP floods. "
            "Service was restored after upstream scrubbing was enabled. Mitigation: "
            "deploy anycast scrubbing, rate-limit DNS, and take down the botnet C2 "
            "infrastructure."
        ),
    },
    {
        "date": "2024-09",
        "title": "Web Shell Activity Targeting Retail eCommerce",
        "category": "Web Attack",
        "severity": "High",
        "sector": "Retail",
        "country": "Australia",
        "actor": "BlackOrchid",
        "malware": "SilverWorm",
        "vector": "Web Shell",
        "cve": "CVE-2024-44444",
        "mitre": "T1505.003",
        "mitre_name": "Web Shell",
        "text": (
            "BlackOrchid compromised multiple eCommerce platforms in Australia via "
            "an unpatched CMS plugin (CVE-2024-44444). The group uploaded PHP web "
            "shells disguised as image files, then pivoted to the payment database. "
            "Stolen data included customer PII and partial card data. Analysts "
            "identified persistence via scheduled tasks and hidden admin accounts. "
            "Mitigation: patch CMS plugins, restrict upload directories, and hunt "
            "for T1505.003 indicators."
        ),
    },
    {
        "date": "2024-11",
        "title": "Data Breach via Compromised Third-Party Vendor",
        "category": "Data Breach",
        "severity": "Critical",
        "sector": "Technology",
        "country": "United States",
        "actor": "APT-Beta",
        "malware": "GhostShell",
        "vector": "Supply Chain",
        "cve": "CVE-2024-55555",
        "mitre": "T1195.002",
        "mitre_name": "Compromise Software Supply Chain",
        "text": (
            "APT-Beta compromised a managed service provider to reach downstream "
            "technology customers in the United States. The actor trojanised a "
            "software update (CVE-2024-55555) with GhostShell, harvesting credentials "
            "and intellectual property. More than forty organisations were affected. "
            "Mitigation: verify software signatures, monitor vendor access, and "
            "rotate leaked credentials. Attack chain matched MITRE T1195.002."
        ),
    },
    {
        "date": "2025-01",
        "title": "Insider Threat — Privileged User Data Exfiltration",
        "category": "Insider Threat",
        "severity": "Medium",
        "sector": "Financial Services",
        "country": "India",
        "actor": "Insider",
        "malware": "N/A",
        "vector": "Insider Abuse",
        "cve": "N/A",
        "mitre": "T1078",
        "mitre_name": "Valid Accounts",
        "text": (
            "A privileged employee at an Indian financial services firm exfiltrated "
            "sensitive customer data over several weeks. The insider used valid "
            "credentials (T1078) to access the data warehouse outside business hours "
            "and copied records to personal cloud storage. Detection came from DLP "
            "alerts and anomalous egress volume. Mitigation: enforce least privilege, "
            "monitor privileged sessions, and tighten DLP rules."
        ),
    },
    {
        "date": "2025-03",
        "title": "Botnet Command-and-Control Infrastructure Takedown",
        "category": "Botnet",
        "severity": "Medium",
        "sector": "Telecommunications",
        "country": "Brazil",
        "actor": "ShadowNet",
        "malware": "Mirai-Variant",
        "vector": "Exploit Kit",
        "cve": "CVE-2025-11111",
        "mitre": "T1583.005",
        "mitre_name": "Botnet",
        "text": (
            "A Mirai variant used CVE-2025-11111 to recruit IoT routers and cameras "
            "into a botnet spanning Brazil and neighbouring countries. The botnet "
            "was rented to third parties for DDoS-for-hire services. Takedown "
            "operations seized 27 C2 domains and 12 hardcoded IPs. Mitigation: "
            "firmware updates, disable default credentials, and subscribe to the "
            "provided IOC feed."
        ),
    },
    {
        "date": "2025-05",
        "title": "Vulnerability Exploitation in Edge Devices",
        "category": "Vulnerability",
        "severity": "Critical",
        "sector": "Energy",
        "country": "Canada",
        "actor": "Unknown",
        "malware": "N/A",
        "vector": "Exposed Service",
        "cve": "CVE-2025-22222",
        "mitre": "T1190",
        "mitre_name": "Exploit Public-Facing Application",
        "text": (
            "Edge devices at multiple Canadian energy operators were mass-scanned "
            "and exploited via CVE-2025-22222, a critical authentication bypass. "
            "Attackers deployed lightweight loaders and attempted lateral movement "
            "toward OT networks. No confirmed impact on industrial control systems "
            "was observed. Mitigation: patch edge devices immediately, restrict "
            "management interfaces, and monitor for T1190 behaviour."
        ),
    },
    {
        "date": "2025-07",
        "title": "Multi-Stage Phishing and Credential Theft Campaign",
        "category": "Credential Theft",
        "severity": "High",
        "sector": "Financial Services",
        "country": "United States",
        "actor": "APT-Alpha",
        "malware": "ShadowRAT",
        "vector": "Spearphishing Link",
        "cve": "CVE-2025-33333",
        "mitre": "T1566.002",
        "mitre_name": "Spearphishing Link",
        "text": (
            "APT-Alpha launched a multi-stage phishing campaign against US financial "
            "services. Stage 1 used spearphishing links (T1566.002) to redirect "
            "victims to adversary-in-the-middle pages that harvested credentials and "
            "MFA tokens. Stage 2 used those sessions to enrol attacker devices and "
            "persist via ShadowRAT. Analysts noted the campaign reused infrastructure "
            "from earlier Credential Theft operations. Mitigation: phishing-resistant "
            "MFA, conditional access, and rapid session revocation."
        ),
    },
]


# ------------------------------------------------------------------
def build_pdf(report: dict, out_path: Path):
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TitleStyle", parent=styles["Title"], fontSize=18, leading=22,
        textColor=colors.HexColor("#1a1a1a"), spaceAfter=12,
    )
    h2 = ParagraphStyle(
        "H2", parent=styles["Heading2"], fontSize=13, leading=16,
        textColor=colors.HexColor("#003366"), spaceBefore=14, spaceAfter=6,
    )
    body = ParagraphStyle(
        "Body", parent=styles["BodyText"], fontSize=10.5, leading=15,
        alignment=TA_JUSTIFY, spaceAfter=6,
    )

    doc = SimpleDocTemplate(
        str(out_path), pagesize=LETTER,
        leftMargin=0.9 * inch, rightMargin=0.9 * inch,
        topMargin=0.9 * inch, bottomMargin=0.9 * inch,
        title=report["title"],
        author="CTI Synthetic Feed",
    )

    story = []

    story.append(Paragraph("Cyber Threat Intelligence Report", title_style))
    story.append(Paragraph(report["title"], h2))
    story.append(Spacer(1, 6))

    meta_rows = [
        ["Date", report["date"]],
        ["Threat Category", report["category"]],
        ["Severity", report["severity"]],
        ["Target Sector", report["sector"]],
        ["Country", report["country"]],
        ["Threat Actor", report["actor"]],
        ["Malware", report["malware"]],
        ["Attack Vector", report["vector"]],
        ["CVE", report["cve"]],
        ["MITRE ATT&CK", f'{report["mitre"]} — {report["mitre_name"]}'],
    ]
    meta_table = Table(meta_rows, colWidths=[1.8 * inch, 4.5 * inch])
    meta_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#eef3f9")),
        ("BOX", (0, 0), (-1, -1), 0.4, colors.grey),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 12))

    story.append(Paragraph("Executive Summary", h2))
    story.append(Paragraph(report["text"], body))

    story.append(Paragraph("Recommended Mitigations", h2))
    mitigations = [
        "Patch affected software to the latest vendor release.",
        "Block known adversary infrastructure at the perimeter.",
        "Enable MFA and rotate potentially compromised credentials.",
        "Hunt for the listed MITRE ATT&CK technique in EDR telemetry.",
        "Brief the affected business unit and update the risk register.",
    ]
    for m in mitigations:
        story.append(Paragraph(f"• {m}", body))

    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "This document is a synthetic test report generated for pipeline "
        "validation. It contains no real threat information.",
        ParagraphStyle("Footer", parent=body, fontSize=8.5,
                       textColor=colors.grey, spaceBefore=10),
    ))

    doc.build(story)


# ------------------------------------------------------------------
def main():
    random.seed(42)
    created = []
    for i, rep in enumerate(REPORTS, start=1):
        # Embed date in filename so the extractor can pick it up
        slug = rep["category"].lower().replace(" ", "_")
        fname = f'{rep["date"]}_{slug}_report_{i:02d}.pdf'
        path = UPLOAD_DIR / fname
        build_pdf(rep, path)
        created.append(path)
        print(f"  ✓ {path.name}")

    print(f"\n✅ Created {len(created)} PDF reports in {UPLOAD_DIR}")


if __name__ == "__main__":
    main()