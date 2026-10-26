"""
Generates 20 synthetic CTI PDFs for testing the analyzer UI.
Output: test_pdfs/report_01.pdf ... report_20.pdf
        test_pdfs/manifest.json  (expected verdicts)
Run from project root:  python scripts/make_20_pdfs.py
"""
import os, json
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.colors import HexColor
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                 Table, TableStyle, PageBreak)

OUT = Path("test_pdfs")
OUT.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────
# 20 test reports
# ─────────────────────────────────────────────────────────────
REPORTS = [
    # ---- KNOWN threat patterns (10) ----
    {"id": 1, "category": "apt", "expected": "known",
     "title": "APT28 Spear Phishing Campaign",
     "text": """Executive Summary
This report describes an APT28 spear phishing campaign observed in Q3 2024.
The threat actor used a crafted email attachment to deliver a malicious DLL.
The dropper executed a PowerShell script that connected to a command-and-control
server hosted at 185.220.101.15. The actor exploited CVE-2023-23397 for initial
access. Post-exploitation activity included credential harvesting and lateral
movement via WMI.

Technical Details
The malicious attachment uses a macro to download a second-stage payload from
a compromised WordPress site. The payload is a Remote Access Trojan supporting
keylogging and screen capture. C2 communication uses HTTPS with certificate
pinning to evade interception. Persistence is established via a registry run key
and a scheduled task created with schtasks.exe."""},

    {"id": 2, "category": "ransomware", "expected": "known",
     "title": "LockBit 3.0 Ransomware Activity",
     "text": """Incident Overview
A LockBit 3.0 affiliate targeted a healthcare organization through exploited
VPN credentials. The actor used double extortion: files were encrypted with
AES-256, and 500GB of data was exfiltrated to a leak site hosted on Tor.

Encryption Behavior
The ransomware encrypts files matching known extensions and drops a ransom
note named README_LOCKBIT.txt in each directory. The note contains a victim
ID and a Tor negotiation portal. Payment is demanded in Bitcoin.

Affiliate Model
LockBit operates as a Ransomware-as-a-Service with affiliates purchasing
access to the builder. Affiliates keep 80% of ransom payments; the core group
retains 20%."""},

    {"id": 3, "category": "apt", "expected": "known",
     "title": "Sandworm CaddyWiper ICS Attack",
     "text": """Threat Summary
The Sandworm APT group (attributed to Russian GRU Unit 74455) deployed a new
variant of CaddyWiper malware against an operational technology environment.

Attack Chain
Initial access was gained through a hypervisor hosting a SCADA management
instance for a power substation. The actor performed reconnaissance on
industrial control system devices before deploying CaddyWiper, which
overwrites the master boot record of the target systems.

Impact
The wiper disabled safety mechanisms at the substation, resulting in a
temporary loss of monitoring capability. Recovery required physical
restoration of the affected control systems."""},

    {"id": 4, "category": "apt", "expected": "known",
     "title": "Lazarus Group Cryptocurrency Theft",
     "text": """Campaign Overview
North Korean actors attributed to Lazarus Group targeted cryptocurrency
exchanges and DeFi platforms in 2024, stealing an estimated $1.4 billion.

Initial Access
The campaign used fake job offers to deliver malware via PDF lures sent
through LinkedIn. The PDF contained an embedded link to a malicious
JavaScript file that installed a backdoor on the victim's system.

Money Laundering
Stolen funds were laundered through mixers, cross-chain bridges, and
over-the-counter brokers. Some funds were traced to North Korean state
accounts via blockchain analysis."""},

    {"id": 5, "category": "rat", "expected": "known",
     "title": "AsyncRAT Campaign Analysis",
     "text": """Malware Overview
AsyncRAT is a Remote Access Trojan available as open-source on GitHub.
The malware supports keylogging, screen capture, remote shell, and file
transfer operations.

Distribution
AsyncRAT is distributed via phishing emails with HTML attachments containing
obfuscated JavaScript. When opened, the script downloads and executes the
AsyncRAT payload from a cloud storage bucket.

C2 Infrastructure
The malware communicates with a C2 server over HTTPS. The C2 address is
hardcoded but can be updated via a configuration file. Multiple C2 servers
rotate every 48 hours to evade takedown."""},

    {"id": 6, "category": "botnet", "expected": "known",
     "title": "Mirai Variant DDoS Attack",
     "text": """Botnet Analysis
A Mirai variant infected IoT devices (routers, cameras, DVRs) through default
credentials. The botnet launched a volumetric DDoS attack against a financial
institution, peaking at 1.2 Tbps.

C2 Infrastructure
The botnet uses a custom C2 protocol over TCP with an encrypted handshake.
The C2 infrastructure rotates domains every 24 hours using a domain-generation
algorithm. Bots periodically download updated C2 configurations.

Takedown Attempt
A coordinated takedown attempt against the C2 domain succeeded in disrupting
one server, but the botnet immediately re-established through the DGA."""},

    {"id": 7, "category": "phishing", "expected": "known",
     "title": "Microsoft 365 Credential Phishing",
     "text": """Campaign Overview
A phishing campaign impersonated Microsoft 365, targeting enterprise users
across multiple sectors. Victims received emails requesting credential
re-verification due to a fabricated security alert.

Landing Page
The landing page cloned the Microsoft login portal, including the logo,
font, and layout. The page used HTTPS with a valid certificate issued to a
compromised domain. After credential entry, victims were redirected to the
real Microsoft login page.

Credential Use
Harvested credentials were tested within 15 minutes of the phishing email
using automated tools. Successful logins triggered an Office 365 inbox rule
to forward future emails to the attacker."""},

    {"id": 8, "category": "supply_chain", "expected": "known",
     "title": "3CX Supply Chain Compromise",
     "text": """Incident Overview
The 3CX softphone application was compromised via a supply chain attack.
Multiple versions of the installer contained a malicious DLL named
ffmpeg.dll that loaded a shellcode payload.

Attack Chain
The attackers compromised a 3CX build server and modified the installer
binaries. The installer was distributed through 3CX's official download
page. Once installed, the malicious DLL downloaded a second-stage payload
from a C2 infrastructure.

Attribution
The attack was attributed to the Lazarus Group based on overlaps in
infrastructure and tooling with prior North Korean campaigns."""},

    {"id": 9, "category": "ics", "expected": "known",
     "title": "Unitronics PLC Exploitation",
     "text": """ICS Incident
IRGC-affiliated actors targeted Unitronics Vision Series PLCs at
U.S.-based Water and Wastewater Systems facilities. The actors
authenticated to internet-connected devices using default or no passwords.

Impact
The attackers defaced the PLC displays and disrupted operations at multiple
facilities. The intrusion was detected through network monitoring when
unusual Modbus commands were observed.

Remediation
Affected devices were reconfigured with strong passwords and removed from
direct internet exposure. The vendor released firmware updates to address
the default-credential issue."""},

    {"id": 10, "category": "rat", "expected": "known",
     "title": "Cobalt Strike Beacon Deployment",
     "text": """Threat Summary
Adversaries deployed Cobalt Strike beacons across a corporate network via
a compromised endpoint. The beacon was configured with a 60-second sleep
cycle and HTTPS malleable C2 profile.

Lateral Movement
Lateral movement used stolen credentials and PsExec to deploy the beacon
to additional hosts. Credentials were dumped from LSASS memory using a
Mimikatz-based tool.

Detection
The activity was detected via network inspection for the Cobalt Strike
default certificate and via EDR alerts on LSASS access attempts."""},

    # ---- EMERGING threats (5) ----
    {"id": 11, "category": "novel", "expected": "emerging",
     "title": "ShadowForge Loader — New Malware Family",
     "text": """Alert: New Malware
A previously unknown loader named ShadowForge was observed in the wild.
The malware uses polymorphic encryption and evades endpoint detection by
hooking system calls at runtime. Affiliates advertise the loader on
underground forums as a subscription service.

Novel Behaviors
Unlike previous loaders, ShadowForge uses indirect syscalls to hide
API calls from user-mode hooks. The loader also implements sleep obfuscation
with RC4 encryption of its memory space during idle periods.

Affiliate Model
The loader is offered on Telegram as a subscription. Pricing starts at
$500/month for basic access and $1500/month for the premium tier with
anti-analysis features."""},

    {"id": 12, "category": "novel", "expected": "emerging",
     "title": "CrimsonVeil Ransomware — VMware ESXi",
     "text": """New Ransomware Family
A new ransomware family called CrimsonVeil was observed encrypting VMware
ESXi hosts. The operators used a leaked ESXiArgs exploit to gain initial
access. The ransom note contained a Tor negotiation portal and a leaked
sample of stolen data as proof.

Technical Details
CrimsonVeil uses a hybrid encryption scheme (AES-256 + RSA-4096). The
ransomware targets ESXi datastores and encrypts .vmdk, .vmx, and .vswp
files. It terminates VM processes before encryption to ensure file locks
are released.

Initial Access
The operators exploited a known ESXiArgs vulnerability that had not been
patched on the victim's hosts."""},

    {"id": 13, "category": "novel", "expected": "emerging",
     "title": "SilentVault Backdoor — DNS-over-HTTPS C2",
     "text": """Emerging Threat
An unknown APT cluster tracked as UNC7777 exploited a zero-day in
Fortinet FortiOS to deploy a new backdoor called SilentVault.

C2 Innovation
SilentVault uses DNS-over-HTTPS (DoH) for command and control, which
evades traditional DNS monitoring. The backdoor queries known DoH
resolvers (Cloudflare, Google) with encrypted payloads.

Persistence
Persistence is established through a modified systemd service on Linux
and a scheduled task on Windows. The backdoor survives reboots and
updates itself from a hardcoded C2."""},

    {"id": 14, "category": "novel", "expected": "emerging",
     "title": "ObsidianSpy Infostealer",
     "text": """New Infostealer
A new infostealer called ObsidianSpy was distributed through malvertising.
The malware targets browser credential stores and cryptocurrency wallet
extensions. Affiliates use a subscription model with tiered pricing.

Distribution
ObsidianSpy is distributed through Google Ads for popular software (e.g.,
Notepad++, VLC) that redirect to lookalike download pages. The installer
contains the payload, which immediately exfiltrates data.

Targets
The malware targets Chrome, Edge, Firefox, and Brave credential stores,
plus MetaMask, Phantom, and TronLink wallet extensions."""},

    {"id": 15, "category": "novel", "expected": "emerging",
     "title": "Novel AiTM MFA Bypass Technique",
     "text": """Emerging Technique
Researchers identified a novel technique for bypassing multi-factor
authentication using adversary-in-the-middle proxies.

Attack Flow
The attack abuses OAuth device-code flow: the victim receives a fake MFA
prompt that redirects to a real Microsoft authentication portal, but the
attacker's proxy captures the session token. Affected organizations span
multiple sectors including finance and healthcare.

Detection
Standard MFA logs may not reveal this attack since the authentication
completes successfully. Detection requires correlating device-code grants
with anomalous login patterns."""},

    # ---- IRRELEVANT (5) ----
    {"id": 16, "category": "irrelevant", "expected": "not_relevant",
     "title": "Quarterly Business Review",
     "text": """Business Update
Our company reported a 12% increase in Q3 revenue compared to the same
period last year. The board approved the quarterly dividend of $0.50 per
share, payable to shareholders of record.

Operational Highlights
The new office in Berlin opened in September, accommodating 150 employees.
Customer satisfaction scores reached an all-time high of 4.6 out of 5.
Headcount grew by 8% year-over-year.

Looking Ahead
The leadership team will present the annual strategy at the shareholder
meeting in January. Registration details will be shared via the corporate
portal."""},

    {"id": 17, "category": "irrelevant", "expected": "not_relevant",
     "title": "Weather and Travel Advisory",
     "text": """Weekend Forecast
Saturday will bring rain showers across the region, with temperatures
dropping to 12 degrees Celsius. Sunday is expected to be clear with
highs of 18 degrees.

Travel Tips
If traveling this weekend, allow extra time for potential delays on
highway 101. The mountain passes may have snow at higher elevations.
Bring chains if driving through the Sierras.

Local Events
The annual fall festival will be held in the downtown plaza on Saturday,
featuring local food vendors and live music."""},

    {"id": 18, "category": "irrelevant", "expected": "not_relevant",
     "title": "Product Launch Announcement",
     "text": """New Product
We are excited to announce the launch of our latest smartphone, featuring
a 6.7-inch OLED display, an upgraded triple-camera system, and a larger
5000 mAh battery.

Availability
The device will be available in three colors: Midnight Black, Ocean Blue,
and Pearl White. Pre-orders begin next Friday, with shipping starting
the following week.

Pricing
The base model starts at $899, with the Pro model at $1,099. Trade-in
options are available for previous-generation devices."""},

    {"id": 19, "category": "irrelevant", "expected": "not_relevant",
     "title": "Team Building Retreat",
     "text": """Event Summary
The engineering team recently completed a two-day team-building retreat
at the mountain lodge. Activities included problem-solving exercises,
a ropes course, and a group dinner.

Feedback
Participant feedback was overwhelmingly positive. Many team members
mentioned that the off-site format allowed deeper conversations than
the usual office environment.

Next Steps
The planning committee will organize a similar event next quarter.
Suggestions for activities are welcome and can be submitted through
the HR portal."""},

    {"id": 20, "category": "irrelevant", "expected": "not_relevant",
     "title": "Book Club Meeting Notes",
     "text": """Meeting Summary
The book club met on Thursday to discuss this month's selection,
a historical novel set in 19th-century Europe. Twenty members
attended, with several joining virtually.

Discussion Highlights
The group debated the author's use of narrative perspective and the
role of secondary characters. The discussion was lively and ran
over by 30 minutes.

Next Month
The next book will be announced by email. The meeting location
remains the community library. Refreshments will be provided."""},
]

# ─────────────────────────────────────────────────────────────
# PDF generator
# ─────────────────────────────────────────────────────────────
def generate_pdf(report, out_path):
    doc = SimpleDocTemplate(
        out_path, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm,
        title=report["title"],
        author="CTI Test Corpus"
    )

    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("H1", parent=styles["Heading1"],
                        fontSize=18, textColor=HexColor("#1f3a93"),
                        spaceAfter=12)
    h2 = ParagraphStyle("H2", parent=styles["Heading2"],
                        fontSize=13, textColor=HexColor("#1f3a93"),
                        spaceAfter=8)
    body = ParagraphStyle("Body", parent=styles["BodyText"],
                          fontSize=10, leading=15, spaceAfter=6)
    meta = ParagraphStyle("Meta", parent=styles["BodyText"],
                          fontSize=9, textColor=HexColor("#666666"))

    flow = []
    flow.append(Paragraph(report["title"], h1))
    flow.append(Paragraph(
        f"Category: <b>{report['category']}</b> &nbsp; | &nbsp; "
        f"Expected verdict: <b>{report['expected']}</b> &nbsp; | &nbsp; "
        f"Report ID: {report['id']:02d}",
        meta))
    flow.append(Spacer(1, 0.5*cm))

    # Split paragraphs on double newlines
    for block in report["text"].strip().split("\n\n"):
        lines = block.strip().split("\n")
        if not lines:
            continue
        # If first line is short and title-like, treat as heading
        if len(lines[0].split()) < 8 and lines[0][0].isupper():
            flow.append(Paragraph(lines[0], h2))
            body_text = " ".join(lines[1:])
            if body_text.strip():
                flow.append(Paragraph(body_text, body))
        else:
            flow.append(Paragraph(" ".join(lines), body))
        flow.append(Spacer(1, 0.2*cm))

    flow.append(Spacer(1, 1*cm))
    flow.append(Paragraph(
        "This is a synthetic test document generated for the CTI Threat "
        "Report Analyzer evaluation. It is not a real threat report and "
        "should not be used for security decisions.",
        meta))

    doc.build(flow)


# ─────────────────────────────────────────────────────────────
# Generate all 20 PDFs
# ─────────────────────────────────────────────────────────────
print("Generating 20 test PDFs...\n")

manifest = []
for r in REPORTS:
    filename = f"report_{r['id']:02d}.pdf"
    path = OUT / filename
    generate_pdf(r, str(path))
    manifest.append({
        "file": filename,
        "id": r["id"],
        "title": r["title"],
        "category": r["category"],
        "expected_verdict": r["expected"],
        "size_kb": round(path.stat().st_size / 1024, 1),
    })
    print(f"  ✓ {filename}  ({r['title']})")

json.dump(manifest, open(OUT / "manifest.json", "w"), indent=2)

print(f"\n✅ Generated {len(REPORTS)} PDFs in `{OUT}/`")
print(f"   Manifest: {OUT}/manifest.json")
print("\nExpected verdicts:")
for r in REPORTS:
    print(f"  report_{r['id']:02d}.pdf  → {r['expected']}  ({r['category']})")