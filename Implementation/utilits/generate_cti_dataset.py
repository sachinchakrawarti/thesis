import os
import random
import hashlib
import pandas as pd
import numpy as np

from faker import Faker
from datetime import datetime, timedelta


# ============================================================
# 1. CONFIGURATION
# ============================================================

NUM_RECORDS = 10000

OUTPUT_DIR = "data/raw"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "synthetic_cti_reports.csv")

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
Faker.seed(SEED)

fake = Faker()

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. CTI CATEGORIES
# ============================================================

CATEGORIES = [
    "Malware",
    "Phishing",
    "Ransomware",
    "DDoS",
    "Data Breach",
    "Credential Theft",
    "Vulnerability",
    "Botnet",
    "Web Attack",
    "Insider Threat"
]


# ============================================================
# 3. CYBERSECURITY VOCABULARY
# ============================================================

THREAT_ACTORS = [
    "APT-Alpha",
    "ShadowFox",
    "DarkHydra",
    "NightWolf",
    "SilverSpider",
    "RedFalcon",
    "StormByte",
    "BlackOrchid"
]

MALWARE = [
    "ShadowRAT",
    "DarkLoader",
    "NightStealer",
    "RedLineX",
    "StormCrypt",
    "BlackRAT",
    "GhostLoader",
    "SilverWorm"
]

ATTACK_VECTORS = [
    "Spear Phishing",
    "Malicious Attachment",
    "Compromised Website",
    "Credential Stuffing",
    "Remote Desktop",
    "Exposed Service",
    "Malicious Link",
    "Drive-by Download",
    "Supply Chain",
    "Brute Force"
]

TARGET_SECTORS = [
    "Banking",
    "Healthcare",
    "Government",
    "Education",
    "Manufacturing",
    "Telecommunications",
    "Retail",
    "Energy",
    "Technology",
    "Financial Services"
]

COUNTRIES = [
    "India",
    "United States",
    "United Kingdom",
    "Germany",
    "France",
    "Japan",
    "Australia",
    "Canada",
    "Singapore",
    "Brazil"
]

SEVERITIES = [
    "Low",
    "Medium",
    "High",
    "Critical"
]


# ============================================================
# 4. CATEGORY-SPECIFIC KEYWORDS
# ============================================================

CATEGORY_KEYWORDS = {

    "Malware": [
        "malware",
        "payload",
        "trojan",
        "remote access",
        "command and control",
        "PowerShell",
        "persistence",
        "executable"
    ],

    "Phishing": [
        "phishing",
        "spear-phishing",
        "malicious email",
        "credential harvesting",
        "malicious link",
        "fake login page",
        "social engineering",
        "email campaign"
    ],

    "Ransomware": [
        "ransomware",
        "file encryption",
        "extortion",
        "encrypted files",
        "ransom demand",
        "backup deletion",
        "data exfiltration",
        "double extortion"
    ],

    "DDoS": [
        "DDoS",
        "distributed denial of service",
        "traffic flood",
        "network flooding",
        "bot traffic",
        "service disruption",
        "HTTP flood",
        "UDP flood"
    ],

    "Data Breach": [
        "data breach",
        "data leakage",
        "stolen records",
        "database exposure",
        "sensitive information",
        "customer records",
        "unauthorized access",
        "information disclosure"
    ],

    "Credential Theft": [
        "credential theft",
        "password stealing",
        "credential harvesting",
        "authentication",
        "login credentials",
        "password",
        "session token",
        "account takeover"
    ],

    "Vulnerability": [
        "vulnerability",
        "CVE",
        "security flaw",
        "remote code execution",
        "privilege escalation",
        "patch",
        "exploit",
        "zero-day"
    ],

    "Botnet": [
        "botnet",
        "command and control",
        "infected devices",
        "bot",
        "distributed attack",
        "malicious infrastructure",
        "C2 server",
        "automated traffic"
    ],

    "Web Attack": [
        "web attack",
        "SQL injection",
        "XSS",
        "cross-site scripting",
        "web shell",
        "malicious request",
        "application attack",
        "server compromise"
    ],

    "Insider Threat": [
        "insider threat",
        "employee",
        "internal access",
        "unauthorized employee",
        "privileged account",
        "internal misuse",
        "data transfer",
        "sensitive files"
    ]
}


# ============================================================
# 5. MITRE ATT&CK TECHNIQUES
# ============================================================

MITRE_TECHNIQUES = [
    ("T1566", "Phishing"),
    ("T1566.001", "Spearphishing Attachment"),
    ("T1566.002", "Spearphishing Link"),
    ("T1059.001", "PowerShell"),
    ("T1059", "Command and Scripting Interpreter"),
    ("T1071.001", "Web Protocols"),
    ("T1083", "File and Directory Discovery"),
    ("T1003", "OS Credential Dumping"),
    ("T1021.001", "Remote Services"),
    ("T1190", "Exploit Public-Facing Application"),
    ("T1486", "Data Encrypted for Impact"),
    ("T1498", "Network Denial of Service"),
    ("T1041", "Exfiltration Over C2 Channel"),
    ("T1053", "Scheduled Task/Job"),
    ("T1547.001", "Registry Run Keys / Startup Folder")
]


# ============================================================
# 6. HELPER FUNCTIONS
# ============================================================

def generate_cve():
    """
    Generate a synthetic CVE-style identifier.
    Clearly synthetic; not intended to represent a real vulnerability.
    """
    year = random.randint(2021, 2026)
    number = random.randint(10000, 99999)

    return f"CVE-{year}-{number}"


def generate_hash():
    """
    Generate a synthetic SHA-256-like value.
    """
    random_text = fake.uuid4()

    return hashlib.sha256(
        random_text.encode()
    ).hexdigest()


def generate_indicators():
    """
    Generate synthetic CTI indicators.
    """

    ip = fake.ipv4()

    domain = (
        fake.domain_word()
        + random.choice([".com", ".net", ".org"])
    )

    file_hash = generate_hash()

    return f"IP={ip}; DOMAIN={domain}; SHA256={file_hash}"


def generate_date():
    """
    Generate dates between 2021 and 2026.
    """

    start = datetime(2021, 1, 1)
    end = datetime(2026, 9, 1)

    delta = end - start

    random_days = random.randint(0, delta.days)

    return start + timedelta(days=random_days)


# ============================================================
# 7. GENERATE CTI TEXT
# ============================================================

def generate_cti_text(
    category,
    threat_actor,
    malware,
    attack_vector,
    target_sector,
    country,
    severity,
    cve_id,
    mitre_id,
    mitre_name
):

    keyword = random.choice(
        CATEGORY_KEYWORDS[category]
    )

    templates = [

        (
            f"A {severity.lower()} {category.lower()} campaign "
            f"targeted organizations in the {target_sector} sector "
            f"located in {country}. The activity was associated with "
            f"{threat_actor} and involved {attack_vector.lower()}. "
            f"Security analysts observed {keyword} activity and "
            f"identified the use of {malware}. The investigation "
            f"reported MITRE ATT&CK technique {mitre_id} "
            f"({mitre_name})."
        ),

        (
            f"Security researchers identified a {category.lower()} "
            f"incident affecting {target_sector} organizations. "
            f"The threat actor {threat_actor} reportedly used "
            f"{attack_vector.lower()} to gain access to targeted "
            f"systems. Observed behavior included {keyword}, "
            f"followed by malicious activity involving {malware}. "
            f"The activity was mapped to {mitre_id}."
        ),

        (
            f"A recent threat campaign involved {category.lower()} "
            f"activity against the {target_sector} sector. "
            f"Initial access was linked to "
            f"{attack_vector.lower()}. Analysts observed "
            f"{keyword} and additional activity associated with "
            f"{malware}. The campaign affected systems in "
            f"{country} and was classified as {severity.lower()} "
            f"severity."
        ),

        (
            f"Threat intelligence monitoring detected {category.lower()} "
            f"behavior involving {threat_actor}. The campaign used "
            f"{attack_vector.lower()} against organizations in the "
            f"{target_sector} sector. Researchers identified "
            f"{keyword}, while {malware} was observed during the "
            f"investigation. The activity corresponds to MITRE "
            f"ATT&CK {mitre_id}."
        )
    ]

    text = random.choice(templates)

    # Add synthetic technical details
    technical_details = (
        f" The observed vulnerability identifier was {cve_id}. "
        f"Additional indicators included network addresses, "
        f"domains and file hashes collected during the investigation."
    )

    return text + technical_details


# ============================================================
# 8. GENERATE ONE RECORD
# ============================================================

def generate_record(record_id):

    category = random.choice(CATEGORIES)

    threat_actor = random.choice(THREAT_ACTORS)

    malware = random.choice(MALWARE)

    attack_vector = random.choice(ATTACK_VECTORS)

    target_sector = random.choice(TARGET_SECTORS)

    country = random.choice(COUNTRIES)

    severity = random.choice(SEVERITIES)

    date = generate_date()

    cve_id = generate_cve()

    mitre_id, mitre_name = random.choice(
        MITRE_TECHNIQUES
    )

    indicators = generate_indicators()

    text = generate_cti_text(
        category=category,
        threat_actor=threat_actor,
        malware=malware,
        attack_vector=attack_vector,
        target_sector=target_sector,
        country=country,
        severity=severity,
        cve_id=cve_id,
        mitre_id=mitre_id,
        mitre_name=mitre_name
    )

    title = (
        f"{category} activity targeting "
        f"{target_sector} organizations"
    )

    # Synthetic URL
    url = (
        f"https://cti-reports.example.org/"
        f"report/{record_id}"
    )

    return {
        "id": record_id,
        "title": title,
        "text": text,
        "category": category,
        "date": date.strftime("%Y-%m-%d"),
        "source": f"CTI-Source-{random.randint(1, 15)}",
        "url": url,
        "severity": severity,
        "threat_actor": threat_actor,
        "malware": malware,
        "attack_vector": attack_vector,
        "target_sector": target_sector,
        "country": country,
        "cve_id": cve_id,
        "mitre_technique": mitre_id,
        "mitre_technique_name": mitre_name,
        "indicators": indicators
    }


# ============================================================
# 9. GENERATE DATASET
# ============================================================

print("Generating synthetic CTI dataset...")

records = []

for i in range(1, NUM_RECORDS + 1):

    record = generate_record(i)

    records.append(record)

    if i % 1000 == 0:
        print(f"{i:,} records generated...")


# ============================================================
# 10. CREATE DATAFRAME
# ============================================================

df = pd.DataFrame(records)


# ============================================================
# 11. DATA CLEANING / VALIDATION
# ============================================================

df["date"] = pd.to_datetime(df["date"])

# Sort chronologically
df = df.sort_values("date")

# Reset index
df = df.reset_index(drop=True)

# Remove duplicate IDs
df = df.drop_duplicates(subset=["id"])


# ============================================================
# 12. SAVE DATASET
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8"
)


# ============================================================
# 13. DISPLAY DATASET INFORMATION
# ============================================================

print("\n==========================================")
print("DATASET GENERATED SUCCESSFULLY")
print("==========================================")

print(f"Rows       : {len(df):,}")
print(f"Columns    : {len(df.columns)}")
print(f"Output     : {OUTPUT_FILE}")

print("\nColumns:")
print(df.columns.tolist())

print("\nCategory Distribution:")
print(df["category"].value_counts())

print("\nDate Range:")
print(df["date"].min())
print(df["date"].max())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nFirst 5 Records:")
print(
    df[
        [
            "id",
            "title",
            "category",
            "date",
            "severity",
            "threat_actor"
        ]
    ].head()
)

print("\nDataset generation completed.")