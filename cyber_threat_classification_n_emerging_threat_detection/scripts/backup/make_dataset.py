# D:\Projects\done\thesis\cyber-threat-detection\scripts\make_dataset.py
"""Generate a realistic 10,000-row CTI dataset for Classification and Emerging Threat detection."""
import random
import uuid
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = PROJECT_ROOT / "dataset" / "SMAH-CTI.csv"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# Reproducibility
random.seed(42)

# --- Vocabulary ---
ACTORS = ["APT28", "APT29", "Lazarus Group", "FIN7", "Sandworm", "Wizard Spider", "LockBit", "BlackCat", "Cl0p", "Hive"]
MALWARE = ["Emotet", "TrickBot", "QakBot", "IcedID", "Cobalt Strike", "Mimikatz", "PlugX", "Remcos", "AgentTesla", "Formbook", "Carbanak"]
SECTORS = ["healthcare", "finance", "energy", "government", "manufacturing", "education", "retail", "telecom", "defense", "logistics"]
CVES = ["CVE-2023-23397", "CVE-2023-4966", "CVE-2024-3400", "CVE-2024-21762", "CVE-2023-38831", "CVE-2021-44228", "CVE-2023-34362", "CVE-2022-30190"]
PROTOCOLS = ["HTTPS", "DNS", "TLS", "SMTP", "SMB", "RDP", "SSH"]
SOURCES = ["Vendor Blog", "OSINT Feed", "Dark Web", "Internal"]

# --- Text Templates ---
# (Tactic, Technique, Template, Priority, Topic Cluster)
POSITIVE_TEMPLATES = [
    ("Initial Access", "T1566 Phishing", "The threat actor utilized a spear-phishing campaign targeting {sector} entities, deploying a malicious ZIP archive containing a weaponized LNK file.", "High", "Phishing Campaigns"),
    ("Initial Access", "T1190 Exploit Public-Facing", "The {actor} group exploited an unpatched vulnerability ({cve}) in public-facing servers to gain initial access.", "Critical", "Zero-Day Exploits"),
    ("Execution", "T1204 User Execution", "{malware} was executed after the victim opened a malicious attachment disguised as an invoice.", "Medium", "Malware Delivery"),
    ("Persistence", "T1053 Scheduled Task", "Once executed, {malware} established persistence by creating a scheduled task masquerading as a legitimate system update.", "High", "C2 & Persistence"),
    ("Persistence", "T1547 Boot or Logon Autostart", "The malware modified registry run keys to ensure {malware} executed automatically upon system reboot.", "High", "C2 & Persistence"),
    ("Privilege Escalation", "T1068 Exploitation", "{malware} exploited a local privilege escalation flaw to gain SYSTEM-level access on the compromised host.", "High", "Evasion Techniques"),
    ("Defense Evasion", "T1027 Obfuscated Files", "{malware} employed heavy obfuscation and disabled endpoint detection and response (EDR) agents to evade analysis.", "High", "Evasion Techniques"),
    ("Credential Access", "T1003 OS Credential Dumping", "Operators used {malware} to dump LSASS process memory and harvest credentials from endpoints.", "High", "Credential Theft"),
    ("Lateral Movement", "T1021 Remote Services", "The attackers moved laterally across the network via RDP using the compromised administrative credentials.", "High", "Lateral Movement"),
    ("Command and Control", "T1071 Application Layer Protocol", "Command-and-control traffic from {malware} was observed over {proto} to attacker-controlled infrastructure.", "High", "C2 & Persistence"),
    ("Exfiltration", "T1567 Exfiltration Over Web Service", "Sensitive data was exfiltrated by {malware} over an encrypted channel to a cloud storage service.", "Critical", "Data Theft"),
    ("Impact", "T1486 Data Encrypted for Impact", "The {actor} ransomware encrypted production systems at a {sector} organization, demanding payment in cryptocurrency.", "Critical", "Ransomware Surge"),
    ("Impact", "T1490 Inhibit System Recovery", "Backups were targeted for deletion using the vssadmin tool to prevent recovery of encrypted files.", "High", "Ransomware Surge"),
]

NEGATIVE_TEMPLATES = [
    "Secureworks analysts recommend that organizations patch their perimeter firewalls and enforce multi-factor authentication to mitigate these risks.",
    "This document outlines the general compliance policies for the organization regarding data retention and employee privacy.",
    "The quarterly earnings report showed revenue growth in the {sector} division, exceeding analyst expectations.",
    "All employees must complete the annual safety training by the end of the month.",
    "We are pleased to announce the launch of our new product line next quarter, featuring enhanced security controls.",
    "Forecasts indicate moderate rainfall across the {sector} region through the end of the week.",
]

def generate_dataset(n_docs=1000, sentences_per_doc=(8, 15), positive_ratio=0.35):
    rows = []
    start_date = datetime(2023, 1, 1)
    
    for doc_i in range(n_docs):
        doc_id = f"doc_{doc_i:04d}"
        source = random.choice(SOURCES)
        # Assign a date to this document (spread across 2 years)
        doc_date = start_date + timedelta(days=random.randint(0, 730))
        
        n_sents = random.randint(*sentences_per_doc)
        page = 1
        context = ""
        
        for s_i in range(1, n_sents + 1):
            is_positive = random.random() < positive_ratio
            
            if is_positive:
                tactic, technique, template, priority, topic = random.choice(POSITIVE_TEMPLATES)
                actor = random.choice(ACTORS)
                malware = random.choice(MALWARE)
                sector = random.choice(SECTORS)
                cve = random.choice(CVES)
                proto = random.choice(PROTOCOLS)
                
                text = template.format(
                    actor=actor, malware=malware, sector=sector, cve=cve, proto=proto
                )
                
                relevant = "yes"
                severity = priority
                zero_day = "yes" if "unpatched" in text or "0-day" in text else "no"
                entities = f"{actor}; {malware}; {sector}"
            else:
                text = random.choice(NEGATIVE_TEMPLATES).format(sector=random.choice(SECTORS))
                relevant = "no"
                tactic = ""
                technique = ""
                entities = ""
                severity = "Low"
                zero_day = "no"
                topic = "N/A"

            uid = f"{doc_id}-p{page}-s{s_i}-{str(uuid.uuid4())[:8]}"
            rows.append({
                "uid": uid,
                "source": source,
                "doc_id": doc_id,
                "page_number": page,
                "sentence_id": s_i,
                "context_before": context,
                "sentence_text": text,
                "relevant": relevant,
                "tactic": tactic,
                "techniques": technique,
                "procedure": f"actor={actor}; malware={malware}" if relevant == "yes" else "",
                "entities": entities,
                "date": doc_date.strftime("%Y-%m-%d"),
                "severity": severity,
                "zero_day": zero_day,
                "topic_cluster": topic
            })
            
            context = text
            if s_i % 5 == 0:
                page += 1
                
    return pd.DataFrame(rows)

if __name__ == "__main__":
    print("Generating 10,000+ realistic CTI sentences...")
    df = generate_dataset()
    
    # Shuffle the dataset so dates and topics are mixed
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    df.to_csv(OUT_PATH, index=False)
    
    print(f"\nSuccessfully wrote {len(df)} rows to {OUT_PATH}")
    print(f"Relevant sentences: {(df['relevant'] == 'yes').sum()}")
    print(f"Irrelevant sentences: {(df['relevant'] == 'no').sum()}")
    print(f"Zero-day flags: {(df['zero_day'] == 'yes').sum()}")
    print("\n--- First 3 rows ---")
    print(df[['date', 'sentence_text', 'techniques', 'topic_cluster']].head(3).to_string())