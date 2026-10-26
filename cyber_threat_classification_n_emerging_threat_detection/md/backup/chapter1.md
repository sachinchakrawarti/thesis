# Chapter 1: Introduction

## 1.1 Background and Motivation

### 1.1.1 The New World Disaster of Cyber Threat Overload

In the 21st century, the volume of cyber threat intelligence (CTI) data has become a major issue of concern regarding organizational security. Even though the world has become more digitally connected than ever before due to technological progress, security analysts have become overwhelmed by unstructured data from vendor reports, dark web forums, and security blogs. CTI is described as the collection, analysis, and dissemination of information regarding current and potential cyber threats. It indicates perceived weaknesses and attack vectors that malicious actors use to compromise systems.

Empirical evidence has demonstrated that unprocessed CTI can lead to many negative consequences, such as:

- Delayed incident response.
- Failure to detect zero-day exploits.
- Worsened data breaches.
- Financial loss and reputational damage.
- Compromised infrastructure.
- Elevated risk of ransomware attacks.

Constant information overload also affects the productivity of cybersecurity teams and their ability to secure critical infrastructure. As the communication process is becoming more digitalized, threat actors are becoming more vocal about their tactics on the internet (e.g., hacker forums), and in many cases, they share vulnerabilities more openly than during physical meetings.

### 1.1.2 Weaknesses of Standard Threat Classification Measures

Years ago, cybersecurity systems used standardized measurements in the form of rule-based detection, such as:

- Signature-based IDS (Intrusion Detection Systems).
- Keyword search.
- Manual review by analysts.

As much as these instruments are valid and acceptable in cybersecurity research, there are a number of limitations associated with them:

1. Reliance on exact keyword matches.
2. Failure to understand semantic context (e.g., missing a threat described using synonyms).
3. Low repetitiveness and scalability.
4. Lack of capacity to offer real-time monitoring of massive text streams.

Such limitations bring about the necessity of using alternative methods which will enable continuous, automated, and scalable detection of cyber threat indicators.

### 1.1.3 Vigor of Computational Threat Intelligence

The sheer increase in the development of the internet and underground forums (e.g., dark web, Reddit) has produced huge amounts of textual data. These sites are often used by attackers to communicate:

- Exploit codes.
- Malware updates.
- Command and Control (C2) discussions.
- Exfiltrated data.

Computational Threat Intelligence has developed into an interdisciplinary system of using Artificial Intelligence and Natural Language Processing to analyze digital footprints and extract actionable information about a threat actor.

Initial methodologies of calculation involved:

- Lexicon-based sentiment analysis.
- Bag-of-Words (BOW) models.
- Traditional machine learning classifiers.

Nevertheless, these approaches do not represent more contextual relationships in language, especially when dealing with complex cybersecurity terminology.

### 1.1.4 Revolution in Transformer in NLP

Transformer-based architecture introductions became a paradigm change in Natural Language Processing. Transformers, in contrast to recurrent or convolutional neural networks, use self-attention mechanisms that allow:

- Long-distance dependency modeling.
- Semantic perception through contextual understanding.
- Parallelized training.
- Dynamic contextual embeddings.

There are models like Sentence-BERT (SBERT) and RoBERTa which exhibit better contextual understanding than traditional word embeddings. Contextual modeling is critical in the case of cyber threat detection since the manifestations of threats are frequently:

- Indirect.
- Obfuscated.
- Context-dependent.
- Not overtly malicious, but subtle.

### 1.1.5 Requirement of Emerging Threat Detection

Traditional classification models are black-box models that focus only on static categories (e.g., "Malware" or "Phishing"). They do not offer reasoning about how the threat landscape is changing. In modern cybersecurity, transparency and temporal awareness are of concern:

- Zero-day detection.
- Trend analysis.
- Algorithmic bias toward known threats.
- Proactive defense.

Topic Modeling techniques, specifically BERTopic, strive to explain and group hidden patterns in the data. By analyzing the frequency of topics over time, we can map:

- Global topic importance (overall threat landscape).
- Local explanation of emerging trends.

Interpretability and temporal awareness are not options in the context of cybersecurity; they are requirements.

### 1.1.6 Transitioning between Detection and Mitigation

The majority of computational systems end at the classification output. It is, however, true that prediction without actionable visualization is of little use in society. Premature identification should be accompanied by an elaborated dashboard to assist in action.

The study will thus include a **FastAPI-based Web Dashboard** mapping the severity of the predicted threats to suitable categories of intervention:

- Visual alerts for zero-day exploits.
- Interactive MITRE ATT&CK distribution charts.
- Trending topic timelines for emerging threats.

This changes the paradigm from passive analytics to active, real-time support.

![Fig. 1.1 Proposed Workflow for Cyber Threat Classification and Emerging Threat Detection](assets/Figure 1.1 Proposed Workflow for Cyber Threat Classification and Emerging Threat Detection.png)

*Fig. 1.1 Proposed workflow for CTI collection, classification, and emerging threat detection.*

<div style="page-break-after: always;"></div>

## 1.2 Problem Statement

Despite the developments in computational approaches to threat detection, the current CTI systems have five significant drawbacks:

1. **Binary Simplification:** Threat relevance is often treated as a binary issue, and systems disregard the severity or temporal progression of attacks.
2. **Shallow Context Modeling:** The classic ML models do not learn fine-grained contextual relations. They rely on TF-IDF and fail to understand the semantic meaning of cybersecurity language.
3. **Failure to Attend to Temporal Indicators:** The frequency of attacks, patterns in terms of time, and the evolution of topics are frequently overlooked. Systems cannot identify a sudden rise in a specific malware campaign.
4. **Lack of Explainability:** Deep learning models do not offer reasoning transparency in their predictions. Analysts cannot see *why* the model flagged a sentence.
5. **Lack of Intervention Mapping:** The outcomes of the detection are not converted into support mechanisms (like visual alerts, trending charts, or downloadable reports) which can be implemented.

Thus, it is necessary to have a thorough, fine-grained, temporally aware, and user-friendly AI model for the detection of cyber threats.

<div style="page-break-after: always;"></div>

## 1.3 Research Objectives

The overall aim of the study is to develop and test an automated AI system on transformer and topic modeling, grounded on contextual and temporal modeling, to predict cyber threats at early stages and identify emerging campaigns.

**Specific Objectives:**

1. Create a binary relevance classification model (Relevant vs. Irrelevant) using Sentence-BERT.
2. Fine-tune a multi-class SVM classifier to map text to MITRE ATT&CK techniques.
3. Apply BERTopic (UMAP + HDBSCAN) to learn and group semantic topics.
4. Develop a temporal rule-based engine to detect emerging threat trends.
5. Build a FastAPI web-based recommendation system for analyst intervention.

<div style="page-break-after: always;"></div>

## 1.4 Research Questions

1. To what extent can cyber threat expressions that are subtle be identified by transformer-based contextual embeddings?
2. Can multi-class technique classification lead to a significant improvement in actionable intelligence?
3. Would BERTopic temporal analysis give operationally interpretable results for emerging threats?
4. What is the way to map the results of classification to a structured web-based mitigation plan?
5. How well does the proposed model work with unstructured, noisy PDF data?

<div style="page-break-after: always;"></div>

## 1.5 Scope and Significance of the Study

### Scope

This study focuses on:

- Cybersecurity text in English (PDF reports).
- Binary and multi-class threat detection.
- Fusion of semantic text and temporal topic features.
- BERTopic-based emerging threat detection.
- Design of an interactive web dashboard.
- The system should be used as a supportive AI tool, and not an automated diagnostic or mitigation tool.

### Significance

- **Scientific Significance:** Integrates fine-grained semantic classification with temporal topic modeling, developing computational threat intelligence research.
- **Technological Significance:** Shows that it is able to combine contextual embeddings and temporal signals effectively in a real-world pipeline.
- **Ethical Significance:** Further increases open and responsible AI in sensitive areas like cyber defense.
- **Societal Significance:** Allows preventing vulnerable organizations at an early stage of exploitation and offering them a structured intervention via the UI.

<div style="page-break-after: always;"></div>

## 1.6 Research Contributions

This thesis contributes:

1. A cyber threat classification framework based on fine-grained transformer-based relevance classification.
2. Context-embeddings and fusion of multi-class technique classification.
3. BERTopic Explainability and Temporal Trend Detection of Emerging Threats.
4. A mitigation recommendation dashboard of pre-emptive action.
5. Fully experimental validation with skewed PDF data.
6. A scalable, web-based architecture to be used in real-world security operations.
