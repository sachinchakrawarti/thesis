# Cyber Threat Classification and Emerging Threat Detection Using Sentence-BERT and BERTopic

## 1. Topic

This research focuses on classifying cyber threat intelligence (CTI) text and identifying emerging cybersecurity threat patterns using natural language processing and topic modeling techniques. It combines semantic text representation through Sentence-BERT with unsupervised topic discovery using BERTopic to analyze real-world cyber threat reports.

## 2. Purpose of the Study

The main purpose of this research is to develop and evaluate a method for analyzing cyber threat intelligence reports efficiently. The study aims to:

- Classify CTI text using machine learning and semantic text representations.
- Identify meaningful threat-related themes from real cybersecurity reports.
- Analyze changes in topic prevalence over time to identify potential emerging threat patterns.
- Compare traditional text classification with embedding-based classification.
- Evaluate the effectiveness, computational cost, and usefulness of the proposed methods.

## 3. Dataset Source

The study will use the SMAH-CTI dataset, available from its official GitHub repository:

Dataset repository: [SMAH-CTI on GitHub](https://github.com/davidmohaisen/SMAH-CTI?utm_source=chatgpt.com)

SMAH-CTI is a sentence-level cyber threat intelligence dataset derived from 100 publicly available CTI reports. It contains 12,304 sentence instances, including 4,302 sentences labelled as relevant and 8,002 labelled as non-relevant, along with entity annotations. Its source collections include APTNotes, CrowdStrike, Mandiant, MITRE, and Microsoft security reports.

The dataset provides sentence text, source and document identifiers, relevance labels, and available annotations related to tactics, techniques, procedures, and entities. These fields support the analysis of CTI relevance and threat-related content.

The main text input for this research will be the `sentence_text` field. The `relevant` field will be used for the primary supervised classification task. The tactic and technique annotations may support additional analysis where the annotations are suitable.

## 4. Proposed Methodology

### 4.1 Data Preprocessing

The dataset will be loaded and examined for missing values, duplicate records, inconsistent labels, and text-quality issues. The `sentence_text` field will be cleaned while preserving important cybersecurity terms, malware names, vulnerability identifiers, and other technical information. The data will be split at the document level where possible to reduce data leakage between training and testing sets.

### 4.2 Cyber Threat Classification

Two approaches will be compared:

- TF-IDF + SVM (Baseline): TF-IDF converts text into numerical features, and a Support Vector Machine classifies sentences based on those features.
- Sentence-BERT + SVM: The `all-MiniLM-L6-v2` Sentence-BERT model converts sentences into semantic embeddings. These embeddings are then passed to an SVM classifier.

The primary classification task will distinguish CTI-relevant sentences from non-relevant sentences using the dataset's existing relevance labels.

### 4.3 Emerging-Threat Detection Using BERTopic

BERTopic will be applied to discover meaningful themes and patterns in cyber threat intelligence text. Sentence embeddings will represent the semantic meaning of the text, while clustering and topic representation will help identify groups of related threat information.

The discovered topics will be examined using their keywords, representative sentences, topic sizes, and relevance to cybersecurity. Where reliable publication dates can be obtained and linked to the reports, topic prevalence will be analyzed over time to identify potential emerging-threat signals.

### 4.4 Evaluation

The classification models will be evaluated using:

- Accuracy
- Precision
- Recall
- Macro-F1 score
- Processing time

BERTopic will be assessed through topic coherence, topic diversity, representative-text inspection, and the usefulness of the identified themes. If reliable temporal metadata is available, the study will also examine changes in topic prevalence and the quality of potential emerging-threat signals.

## 5. Expected Outcome

The expected outcome is a comparative evaluation of traditional and semantic machine-learning approaches for CTI text classification, together with a topic-modeling pipeline that discovers meaningful cybersecurity themes. The research will investigate whether combining Sentence-BERT and BERTopic can provide useful insights into patterns and potential changes in the cyber threat landscape.

Dataset limitation: The SMAH-CTI sentence-level data does not itself guarantee that every record has a publication date. Temporal analysis will therefore depend on obtaining reliable dates from the original reports or associated metadata.