# Cyber Threat Intelligence (CTI) Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)

An automated, end-to-end Cyber Threat Intelligence framework designed to ingest, process, enrich, and correlate threat data from heterogeneous sources to produce actionable intelligence. Developed as part of a Master's Thesis research project.

---

## 📌 Abstract / Overview

Modern Security Operations Centers (SOCs) face an overwhelming volume of unstandardized threat data. This framework addresses the challenge by automating the extraction of Indicators of Compromise (IoCs), mapping threat actor Tactics, Techniques, and Procedures (TTPs) to the **MITRE ATT&CK®** framework, and outputting standardized **STIX 2.1** objects for seamless integration into SIEM and SOAR platforms.

**Key Thesis Contributions:**
* Automated ingestion of structured (STIX/TAXII, MISP) and unstructured (OSINT blogs, reports) data.
* NLP-driven entity extraction for novel threat vectors.
* Dynamic threat scoring and deduplication mechanism.

---

## 🏗️ System Architecture

```text
[ Raw Data Feeds ] ---> [ Ingestion Pipeline ] ---> [ NLP / Entity Extraction ]
  (OSINT, STIX, API)         (Parser/Filter)            (IoCs & TTP Mapping)
                                                                |
                                                                v
[ Standardized STIX 2.1 ] <-- [ Enrichment & Scoring ] <-- [ Correlation Engine ]
         |
         v
[ SIEM / SOAR Export ]
