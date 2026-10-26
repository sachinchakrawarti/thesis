# Dataset Description and Sources

## 1. Source Datasets

This research uses two publicly available datasets to develop a dataset for Cyber Threat Classification and Emerging Threat Detection.

### Dataset 1: CIRCL Vulnerability Attack Techniques

Source link: [CIRCL/Vulnerability-Attack-Techniques on Hugging Face](https://huggingface.co/datasets/CIRCL/vulnerability-attack-techniques?utm_source=chatgpt.com)

| Attribute                     | Description                                                                                                   |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------- |
| Dataset name                  | `CIRCL/vulnerability-attack-techniques`                                                                       |
| Data type                     | Structured CVE vulnerability descriptions and attack technique labels                                         |
| `id`                          | CVE identifier                                                                                                |
| `title`                       | Vulnerability title                                                                                           |
| `description`                 | Textual description of the vulnerability                                                                      |
| `exploitation_techniques`     | Exploitation-related MITRE ATT&CK technique identifiers                                                       |
| `primary_impact`              | Primary impact-related technique labels                                                                       |
| `secondary_impact`            | Secondary impact-related technique labels                                                                     |
| `techniques`                  | Available technique annotations                                                                               |
| `techniques_derived`          | Derived or mapped technique information                                                                       |
| `cvss_vector`, `cvss_version` | CVSS scoring vector and version information                                                                   |
| `cwes`                        | Common Weakness Enumeration identifiers                                                                       |
| `affected_products`, `cpes`   | Affected products and platform information                                                                    |
| Publication date              | No explicit publication-date field in the listed schema; CVE ID year alone is not the actual publication date |

Purpose in this research: This dataset provides vulnerability descriptions and technique annotations that can support cyber-threat classification. Records should be checked for valid labels before being treated as ground truth.

### Dataset 2: ThreatCluster CVE Exploitation Signals

Source link: [threatcluster/cve-exploitation-signals on Hugging Face](https://huggingface.co/datasets/threatcluster/cve-exploitation-signals?utm_source=chatgpt.com)

| Attribute                               | Description                                                              |
| --------------------------------------- | ------------------------------------------------------------------------ |
| Dataset name                            | `threatcluster/cve-exploitation-signals`                                 |
| Data type                               | CVE descriptions, vulnerability severity, and exploitation indicators    |
| Number of rows                          | 60,879 rows in the reported training split                               |
| `cve_id`                                | CVE identifier                                                           |
| `description`                           | Textual description of the vulnerability                                 |
| `published_date`                        | CVE publication date                                                     |
| `cvss_v3_score`                         | CVSS v3 numerical severity score                                         |
| `cvss_v3_severity`                      | Severity category, such as Critical, High, Medium, or Low                |
| `cwe_ids`                               | Associated weakness identifiers                                          |
| `affected_vendors`, `affected_products` | Affected vendors and products                                            |
| `in_kev`                                | Indicates inclusion in the Known Exploited Vulnerabilities (KEV) catalog |
| `kev_added_date`                        | Date added to the KEV catalog                                            |
| `kev_due_date`                          | Remediation due date, where available                                    |
| `ransomware_use`                        | Ransomware-related exploitation indicator                                |
| `has_exploit`                           | Indicates whether exploit evidence is recorded                           |
| `exploit_count`                         | Number of recorded exploits                                              |

Purpose in this research: This dataset provides timestamped vulnerability and exploitation-related signals that can help identify relevant threats and analyze changes in threat activity over time.

## 2. Proposed Research Dataset

The two source datasets are processed and combined into a research dataset named SMAH-CTI67.csv.

| Attribute               | Description                                                                                                             |
| ----------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| Dataset name            | `SMAH-CTI67.csv`                                                                                                        |
| Source datasets         | CIRCL Vulnerability Attack Techniques and ThreatCluster CVE Exploitation Signals                                        |
| Data type               | Structured cyber-threat intelligence and vulnerability text                                                             |
| Maximum target size     | 15,000 rows, depending on the number of eligible records                                                                |
| Target positive records | 6,000 relevant records, subject to source availability and filtering                                                    |
| `uid`                   | Unique record identifier generated during dataset construction                                                          |
| `source`                | Original source dataset (`circl` or `threatcluster`)                                                                    |
| `doc_id`                | Document or record group identifier                                                                                     |
| `page_number`           | Page number field                                                                                                       |
| `sentence_id`           | Sentence or record sequence identifier                                                                                  |
| `context_before`        | Context preceding the main text; empty for the current source records                                                   |
| `sentence_text`         | Main vulnerability description used as text input                                                                       |
| `relevant`              | Research relevance label (`yes` or `no`) assigned according to source-specific rules                                    |
| `tactic`                | MITRE ATT&CK tactic field; currently empty where not populated                                                          |
| `techniques`            | Available technique label; CIRCL processing currently retains only the first technique                                  |
| `procedure`             | Supporting description or procedure text                                                                                |
| `entities`              | Affected products or vendors                                                                                            |
| `date`                  | Date selected during processing; currently may represent a KEV-added date, publication date, or estimated CVE-year date |
| `severity`              | Normalized severity category                                                                                            |
| `zero_day`              | Zero-day indicator; currently set to `no` for every record and therefore not independently verified                     |
| `topic_cluster`         | Source label or heuristic reason used to describe the record                                                            |

### Dataset construction workflow

flowchart TD
    A["Dataset 1: CIRCL Vulnerability Attack Techniques"] --> C["Data Loading and Filtering"]
    B["Dataset 2: ThreatCluster CVE Exploitation Signals"] --> C

    C --> D["Text and Field Normalization"]
    D --> E["Relevance Label Assignment"]
    E --> F["Record Sampling and Combination"]
    F --> G["SMAH-CTI67.csv"]

    G --> H["Cyber-Threat Classification"]
    G --> I["Temporal Topic Analysis"]
    I --> J["BERTopic Clustering"]
    J --> K["Emerging Threat Candidate Detection"]



Column	Purpose	Task
uid	unique row ID	all
source	origin (threatcluster / circl)	all
cve_id	CVE identifier	all
sentence_text	description (the input)	all
date	publication date	all
severity	LOW / MEDIUM / HIGH / CRITICAL	binary + severity
relevant	yes / no	binary
techniques	ATT&CK IDs (e.g. T1059)	ATT&CK
primary_tech	first technique only	ATT&CK
cwe_ids	CWE classification	reference
has_exploit	True/False	reference
in_kev	True/False	reference
ransomware_use	True/False	reference