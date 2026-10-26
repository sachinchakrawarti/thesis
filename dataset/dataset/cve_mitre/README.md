---
license: mit
task_categories:
  - text-classification
  - zero-shot-classification
language:
  - en
size_categories:
  - 10K<n<100K
tags:
  - security
  - mitre-attack
  - cve
  - cybersecurity
  - vulnerability
  - threat-intelligence
pretty_name: MITRE CVE Community Pack
---

# MITRE CVE Community Pack

A comprehensive dataset for training multi-label classifiers to predict MITRE ATT&CK techniques from vulnerability descriptions.

## Dataset Description

This dataset contains vulnerability descriptions paired with MITRE ATT&CK technique labels, designed for training and evaluating multi-label text classification models in the cybersecurity domain.

### Key Features

- **Dual-source data**: 10,000 real CVE descriptions and 10,000 synthetic vulnerability descriptions
- **Multi-label classification**: Each vulnerability can be associated with multiple MITRE ATT&CK techniques
- **Cybersecurity-focused**: Specifically curated for security research and threat intelligence applications
- **Ready for training**: Compatible with Hugging Face Datasets and Transformers libraries

## Dataset Structure

### Data Files

- **`real_10k.jsonl`**: 10,000 real vulnerability descriptions from CVE (Common Vulnerabilities and Exposures)
- **`synth_10k.jsonl`**: 10,000 synthetically generated vulnerability descriptions
- **`attaquemitre2.csv`**: MITRE ATT&CK technique reference (ID, name, description, platforms)

### Data Format

Each line in the JSONL files contains:

```json
{
  "text": "Description of the vulnerability...",
  "labels": ["T1078", "T1203", "T1059"]
}
```

- **`text`**: Natural language description of the vulnerability
- **`labels`**: List of MITRE ATT&CK technique IDs (multi-label)

### Label Space

The dataset uses MITRE ATT&CK technique IDs as labels (e.g., T1078, T1203, T1059). The `attaquemitre2.csv` file provides detailed information about each technique including:
- Technique ID
- Technique name
- Description
- Target platforms

## Usage

### Loading the Dataset

```python
import json
from datasets import Dataset

# Load real CVE data
real_data = []
with open('real_10k.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        real_data.append(json.loads(line))

# Load synthetic data
synth_data = []
with open('synth_10k.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        synth_data.append(json.loads(line))

# Combine datasets
all_data = real_data + synth_data
dataset = Dataset.from_list(all_data)
```

### Training Example (ModernBERT)

```python
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import Dataset

# Load data
texts = [item['text'] for item in all_data]
all_labels = sorted(list(set(l for item in all_data for l in item['labels'])))
label2id = {l: i for i, l in enumerate(all_labels)}
id2label = {i: l for i, l in enumerate(all_labels)}

# Create multi-hot encoded labels
labels = np.zeros((len(texts), len(all_labels)), dtype=np.float32)
for i, item in enumerate(all_data):
    for lbl in item['labels']:
        labels[i, label2id[lbl]] = 1.0

# Split data
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.1, random_state=42
)

# Tokenize
tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-large")
train_ds = Dataset.from_dict({"text": train_texts, "labels": train_labels}).map(
    lambda x: tokenizer(x["text"], truncation=True, max_length=1024),
    batched=True
)

# Model setup
model = AutoModelForSequenceClassification.from_pretrained(
    "answerdotai/ModernBERT-large",
    num_labels=len(id2label),
    problem_type="multi_label_classification",
    id2label=id2label,
    label2id=label2id
)
```

## Dataset Statistics

- **Total samples**: 20,000 (10,000 real + 10,000 synthetic)
- **Average labels per sample**: ~1.2 techniques
- **Unique techniques**: Covers a wide range of MITRE ATT&CK tactics
- **Language**: English
- **Domain**: Cybersecurity / Vulnerability Assessment

## Applications

- **Vulnerability analysis**: Automatically categorize CVEs by ATT&CK techniques
- **Threat intelligence**: Enhance CTI platforms with automated labeling
- **Security research**: Train models for technique prediction
- **Incident response**: Quickly identify potential attack techniques in vulnerabilities
- **Risk assessment**: Prioritize vulnerabilities based on associated techniques

## Considerations

- **Multi-label nature**: Each vulnerability may map to multiple ATT&CK techniques
- **Class imbalance**: Some techniques are more common than others
- **Synthetic data**: The `synth_10k.jsonl` file contains AI-generated descriptions for training augmentation
- **Label granularity**: Uses technique-level labels (e.g., T1078) rather than tactic or sub-technique

## Reference

- **MITRE ATT&CK Framework**: https://attack.mitre.org/
- **CVE Database**: https://cve.mitre.org/

## Citation

If you use this dataset, please cite:
```
@dataset{mitre_cve_community_pack,
  title={MITRE CVE Community Pack},
  author={Community},
  year={2024},
  url={https://huggingface.co/datasets/[your-username]/MITRE_CVE_Community_Pack}
}
```

## License

MIT License

---

**Note**: This dataset is designed for research and educational purposes in cybersecurity. Always validate predictions with human experts in production security environments.
