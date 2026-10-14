# scripts/generate_sample_pdfs.py
"""
Generate 10 synthetic CTI PDF reports into uploads/.
Requires:  pip install reportlab
"""
from pathlib import Path
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
)
from reportlab.lib import colors

PROJECT_ROOT = Path(__file__).resolve().parent.parent
UPLOAD_DIR = PROJECT_ROOT / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

REPORTS = [
    ("2024-01", "Phishing Campaign Targeting Banking Sector", "Phishing",
     "High", "Banking", "United States", "APT-Alpha", "ShadowRAT",
     "Spearphishing Attachment", "CVE-2024-11111",
     "T1566.001", "Spearphishing Attachment",
     "A targeted phishing campaign was observed against multiple banking "
     "institutions. The threat actor APT-Alpha delivered malicious Word "
     "documents via spearphishing emails exploiting CVE-2024-11111 to drop "
     "ShadowRAT. Analysts identified credential harvesting pages mimicking "
     "SSO login screens."),
    ("2024-03", "Ransomware Against Healthcare Providers", "Ransomware",
     "Critical", "Healthcare", "United Kingdom", "NightWolf", "RedLineX",
     "Exposed Service", "CVE-2024-22222",
     "T1486", "Data Encrypted for Impact",
     "NightWolf ransomware operators exploited an unpatched VPN appliance "
     "(CVE-2024-22222) at multiple UK healthcare providers. The group "
     "deployed RedLineX to exfiltrate patient data before encrypting file "
     "shares and backup servers."),
    ("2024-05", "Credential Theft Targeting Financial Services",
     "Credential Theft", "High", "Financial Services", "Singapore",
     "DarkHydra", "DarkLoader", "Drive-by Download", "CVE-2024-33333",
     "T1555.003", "Credentials from Web Browsers",
     "DarkHydra deployed DarkLoader via compromised advertising networks to "
     "steal credentials from financial services employees. The loader "
     "injected into browsers to scrape saved passwords, session cookies, "
     "and crypto wallet files."),
    ("2024-07", "DDoS Attack Against Government Services", "DDoS",
     "High", "Government", "Germany", "StormByte", "SilverWorm",
     "Botnet", "N/A", "T1498", "Network Denial of Service",
     "A volumetric DDoS attack disrupted public-facing German government "
     "portals for approximately four hours. StormByte coordinated a botnet "
     "of IoT devices, peaking at 620 Gbps."),
    ("2024-09", "Web Shell Activity Targeting Retail", "Web Attack",
     "High", "Retail", "Australia", "BlackOrchid", "SilverWorm",
     "Web Shell", "CVE-2024-44444", "T1505.003", "Web Shell",
     "BlackOrchid compromised multiple eCommerce platforms in Australia via "
     "an unpatched CMS plugin (CVE-2024-44444). PHP web shells were uploaded "
     "disguised as image files."),
    ("2024-11", "Data Breach via Compromised Vendor", "Data Breach",
     "Critical", "Technology", "United States", "APT-Beta", "GhostShell",
     "Supply Chain", "CVE-2024-55555",
     "T1195.002", "Compromise Software Supply Chain",
     "APT-Beta compromised a managed service provider to reach downstream "
     "technology customers. The actor trojanised a software update with "
     "GhostShell. More than forty organisations were affected."),
    ("2025-01", "Insider Threat Data Exfiltration", "Insider Threat",
     "Medium", "Financial Services", "India", "Insider", "N/A",
     "Insider Abuse", "N/A", "T1078", "Valid Accounts",
     "A privileged employee at an Indian financial services firm "
     "exfiltrated sensitive customer data over several weeks using valid "
     "credentials to access the data warehouse outside business hours."),
    ("2025-03", "Botnet C2 Infrastructure Takedown", "Botnet",
     "Medium", "Telecommunications", "Brazil", "ShadowNet", "Mirai-Variant",
     "Exploit Kit", "CVE-2025-11111", "T1583.005", "Botnet",
     "A Mirai variant used CVE-2025-11111 to recruit IoT routers and "
     "cameras into a botnet spanning Brazil. Takedown operations seized 27 "
     "C2 domains and 12 hardcoded IPs."),
    ("2025-05", "Vulnerability Exploitation in Edge Devices", "Vulnerability",
     "Critical", "Energy", "Canada", "Unknown", "N/A",
     "Exposed Service", "CVE-2025-22222",
     "T1190", "Exploit Public-Facing Application",
     "Edge devices at multiple Canadian energy operators were mass-scanned "
     "and exploited via CVE-2025-22222, a critical authentication bypass. "
     "Attackers deployed lightweight loaders and attempted lateral movement "
     "toward OT networks."),
    ("2025-07", "Multi-Stage Phishing Credential Campaign",
     "Credential Theft", "High", "Financial Services", "United States",
     "APT-Alpha", "ShadowRAT", "Spearphishing Link", "CVE-2025-33333",
     "T1566.002", "Spearphishing Link",
     "APT-Alpha launched a multi-stage phishing campaign against US "
     "financial services. Stage 1 used spearphishing links to redirect "
     "victims to adversary-in-the-middle pages that harvested credentials "
     "and MFA tokens."),
]


def build_pdf(report, out_path):
    (date, title, category, severity, sector, country,
     actor, malware, vector, cve, mitre, mitre_name, body) = report

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "T", parent=styles["Title"], fontSize=17, leading=21,
        spaceAfter=10)
    h2 = ParagraphStyle(
        "H2", parent=styles["Heading2"], fontSize=13,
        textColor=colors.HexColor("#003366"),
        spaceBefore=12, spaceAfter=6)
    body_style = ParagraphStyle(
        "B", parent=styles["BodyText"], fontSize=10.5,
        leading=15, alignment=TA_JUSTIFY, spaceAfter=6)

    doc = SimpleDocTemplate(
        str(out_path), pagesize=LETTER,
        leftMargin=0.9*inch, rightMargin=0.9*inch,
        topMargin=0.9*inch, bottomMargin=0.9*inch,
        title=title, author="CTI Synthetic Feed")

    story = [
        Paragraph("Cyber Threat Intelligence Report", title_style),
        Paragraph(title, h2),
        Spacer(1, 6),
    ]

    meta = [
        ["Threat Category", category],
        ["Severity", severity],
        ["Target Sector", sector],
        ["Country", country],
        ["Threat Actor", actor],
        ["Malware", malware],
        ["Attack Vector", vector],
        ["CVE", cve],
        ["MITRE ATT&CK", f"{mitre} — {mitre_name}"],
    ]
    t = Table(meta, colWidths=[1.8*inch, 4.5*inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (0,-1), colors.HexColor("#eef3f9")),
        ("BOX", (0,0), (-1,-1), 0.4, colors.grey),
        ("INNERGRID", (0,0), (-1,-1), 0.25, colors.lightgrey),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("FONTSIZE", (0,0), (-1,-1), 10),
        ("LEFTPADDING", (0,0), (-1,-1), 8),
        ("RIGHTPADDING", (0,0), (-1,-1), 8),
        ("TOPPADDING", (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
    ]))
    story.append(t)
    story.append(Spacer(1, 12))

    story.append(Paragraph("Executive Summary", h2))
    story.append(Paragraph(body, body_style))

    story.append(Paragraph("Recommended Mitigations", h2))
    for m in [
        "Patch affected software to the latest vendor release.",
        "Block known adversary infrastructure at the perimeter.",
        "Enable MFA and rotate potentially compromised credentials.",
        "Hunt for the listed MITRE ATT&CK technique in EDR telemetry.",
        "Brief the affected business unit and update the risk register.",
    ]:
        story.append(Paragraph(f"• {m}", body_style))

    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "This document is a synthetic test report generated for pipeline "
        "validation. It contains no real threat information.",
        ParagraphStyle("F", parent=body_style, fontSize=8.5,
                       textColor=colors.grey, spaceBefore=10)))

    doc.build(story)


def main():
    for i, rep in enumerate(REPORTS, start=1):
        slug = rep[2].lower().replace(" ", "_")
        fname = f"{rep[0]}_{slug}_report_{i:02d}.pdf"
        path = UPLOAD_DIR / fname
        build_pdf(rep, path)
        print(f"  + {fname}")
    print(f"\nCreated {len(REPORTS)} PDFs in {UPLOAD_DIR}")


if __name__ == "__main__":
    main()