# Chapter 6 – Conclusion and Future Work

## 6.1 Introduction

Cyber Threat Intelligence (CTI) plays an important role in identifying, understanding, and responding to evolving cybersecurity threats. The increasing volume of textual threat intelligence generated through security reports, vulnerability disclosures, incident analyses, malware reports, and other sources makes manual analysis increasingly difficult.

This research proposed a framework for automated CTI analysis that combines machine learning-based threat classification, topic modeling, temporal analysis, and emerging threat scoring. The primary objective was to develop a computational approach capable of analyzing textual CTI data, identifying predefined threat categories, discovering latent threat-related topics, tracking their evolution over time, and highlighting candidate emerging threat patterns.

This chapter summarizes the research work, discusses the major findings, presents the contributions of the study, identifies its limitations, and describes possible directions for future research.

---

## 6.2 Research Summary

The research focused on the development of a framework titled:

> **Real-Time Cyber Threat Classification and Emerging Threat Detection Using Machine Learning and Temporal Topic Modeling**

The study was organized into several major stages.

First, CTI documents were collected and organized into a structured dataset containing textual information, threat categories, temporal information, and source metadata.

Second, the collected text was preprocessed to remove irrelevant content and convert the documents into a suitable representation for machine learning and topic modeling.

Third, machine learning algorithms were applied to classify CTI documents into predefined threat categories.

Fourth, topic modeling techniques were used to identify latent themes within the CTI corpus.

Fifth, temporal analysis was applied to examine how discovered topics changed across different time periods.

Finally, an emerging threat scoring mechanism was designed to combine multiple signals, including novelty, growth, persistence, activity, and changes in important terms.

The overall research workflow can be summarized as:

```text
CTI Data Collection
        ↓
Dataset Preparation
        ↓
Text Preprocessing
        ↓
Feature Extraction
        ↓
Threat Classification
        ↓
Topic Modeling
        ↓
Temporal Topic Analysis
        ↓
Emerging Threat Scoring
        ↓
Evaluation and Analysis
```

The framework therefore combines supervised and unsupervised approaches to provide complementary perspectives on CTI information.

---

## 6.3 Achievement of Research Objectives

The research objectives defined in Chapter 1 were addressed through the methodology and experimental implementation.

### Objective 1: CTI Dataset Collection and Preparation

A structured CTI dataset was prepared containing textual threat information and temporal metadata. Data validation and preprocessing procedures were applied to improve the consistency and usability of the dataset.

### Objective 2: Text Preprocessing and Feature Extraction

Natural Language Processing techniques were applied to transform raw CTI text into a machine-readable representation. Text cleaning, normalization, tokenization, and feature extraction were incorporated into the processing pipeline.

TF-IDF-based representations were used for traditional machine learning classification, while document-term representations were used for topic modeling.

### Objective 3: Automated Threat Classification

Machine learning models were implemented to classify CTI documents into predefined threat categories.

The evaluated models included:

* Naive Bayes
* Logistic Regression
* Support Vector Machine
* Random Forest

The performance of these models was assessed using standard classification metrics.

### Objective 4: Comparative Evaluation of Classification Models

The classification models were compared using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion matrix

This comparison provided information about the behavior of different machine learning algorithms on the selected CTI dataset.

### Objective 5: Latent Topic Discovery

Topic modeling was applied to identify latent themes within the CTI corpus.

LDA and NMF were considered for discovering groups of terms that frequently occur together and may represent meaningful cybersecurity topics.

### Objective 6: Temporal Topic Analysis

The discovered topics were associated with temporal information to determine how their prevalence changed across different periods.

Temporal analysis enabled the identification of:

* Increasing topics
* Decreasing topics
* Persistent topics
* Newly appearing topics
* Temporally fluctuating topics

### Objective 7: Emerging Threat Detection

An emerging threat scoring mechanism was designed to identify candidate patterns that exhibit characteristics associated with emerging activity.

The scoring mechanism incorporated multiple factors:

$$
ETS_k(t)
=
w_NN_k(t)
+
w_GG_k(t)
+
w_SS_k(t)
+
w_AA_k(t)
+
w_TT_k(t)
$$

where novelty, growth, persistence, activity, and term change contribute to the overall score.

### Objective 8: Framework Evaluation

The complete framework was evaluated through classification experiments, topic modeling analysis, temporal analysis, emerging threat scoring, and error analysis.

The evaluation provides a basis for determining the usefulness and limitations of the proposed approach.

---

## 6.4 Major Findings

The major findings of the research can be summarized into four areas: threat classification, topic discovery, temporal analysis, and emerging threat detection.

### 6.4.1 Threat Classification

The classification experiments demonstrated the applicability of machine learning techniques to automated CTI categorization.

Traditional text representations such as TF-IDF can provide useful features for distinguishing threat categories when the dataset contains sufficiently representative terminology.

However, classification performance depends strongly on:

* Dataset quality.
* Number of training examples.
* Class distribution.
* Label consistency.
* Vocabulary overlap between categories.
* Feature representation.

Therefore, classification results obtained from a particular dataset should not automatically be generalized to all CTI sources.

### 6.4.2 Topic Discovery

Topic modeling provided an unsupervised mechanism for discovering latent themes within the CTI corpus.

The discovered topics can reveal relationships between cybersecurity concepts that may not be explicitly represented by predefined classification labels.

The quality of discovered topics depends on:

* Number of topics.
* Preprocessing strategy.
* Vocabulary selection.
* Dataset size.
* Topic modeling algorithm.
* Semantic characteristics of the corpus.

Topic coherence and human interpretation were therefore considered together when analyzing topic quality.

### 6.4.3 Temporal Topic Evolution

Temporal analysis extended conventional topic modeling by examining how cybersecurity themes change over time.

A topic that becomes increasingly prevalent may indicate a change in the type or frequency of CTI reporting associated with that theme.

Similarly, persistent topics may represent established cybersecurity concerns, while newly appearing topics may provide signals for further investigation.

Temporal analysis is therefore useful for understanding the dynamic nature of cybersecurity information.

### 6.4.4 Emerging Threat Detection

The proposed emerging threat scoring mechanism provides a method for prioritizing topics that demonstrate multiple signals of emerging activity.

Instead of relying only on topic frequency, the framework considers:

* Novelty.
* Growth.
* Persistence.
* Activity.
* Term changes.

This multi-factor approach provides a broader representation of emerging behavior.

However, a high score should be interpreted as a **candidate emerging-threat signal** rather than confirmation of a new cyber threat or zero-day vulnerability.

Additional validation using independent CTI sources and expert analysis remains necessary.

---

## 6.5 Contributions of the Research

The research makes several contributions to the area of automated Cyber Threat Intelligence analysis.

### 6.5.1 Integrated CTI Analysis Framework

The study presents an integrated framework combining:

* Machine learning-based classification.
* Topic modeling.
* Temporal analysis.
* Emerging threat scoring.

These components provide complementary analytical capabilities within a single workflow.

### 6.5.2 Automated Threat Classification

The framework provides an automated mechanism for assigning CTI documents to predefined threat categories.

This can reduce the amount of manual effort required for initial organization and analysis of large collections of textual CTI information.

### 6.5.3 Temporal Topic Analysis

The research incorporates temporal information into topic analysis rather than treating the CTI corpus as a static collection of documents.

This allows changes in threat-related topics to be examined across different time periods.

### 6.5.4 Emerging Threat Prioritization

The proposed scoring mechanism provides a systematic method for prioritizing candidate emerging topics based on multiple temporal and textual characteristics.

It is designed as an analytical support mechanism rather than an autonomous threat-confirmation system.

### 6.5.5 Experimental Evaluation

The framework provides an experimental methodology for evaluating:

* Classification performance.
* Topic quality.
* Temporal behavior.
* Emerging-threat signals.
* Error characteristics.

This provides a foundation for further development and comparison with alternative approaches.

### 6.5.6 Reproducible Implementation

The implementation is organized as a modular data science workflow so that dataset preparation, feature extraction, classification, topic modeling, temporal analysis, and scoring can be independently examined and reproduced.

---

## 6.6 Research Limitations

Although the proposed framework provides an integrated approach to CTI analysis, several limitations remain.

### 6.6.1 Dataset Dependency

The performance of the framework depends strongly on the characteristics of the selected CTI dataset.

A dataset containing limited threat categories, a narrow source distribution, or insufficient temporal coverage may restrict the generalizability of the results.

### 6.6.2 Label Dependence

Supervised classification requires labeled training data.

Incorrect, incomplete, or inconsistent labels can negatively affect model performance.

### 6.6.3 Topic Interpretability

Topic modeling does not guarantee that every discovered topic corresponds to a clearly defined cybersecurity concept.

Some topics may contain generic or overlapping terminology and require human interpretation.

### 6.6.4 Temporal Data Limitations

Temporal analysis depends on the accuracy and completeness of document dates.

Changes in publication frequency may also influence observed topic trends without necessarily representing changes in actual cyber threat activity.

### 6.6.5 Emerging Threat Validation

The emerging threat scoring mechanism identifies candidate signals rather than independently confirming real-world threats.

A high score may result from increased reporting, duplicated information, media attention, or other changes in the underlying corpus.

### 6.6.6 Limited Contextual Understanding

Traditional machine learning and topic modeling approaches may not fully capture the contextual relationships between cybersecurity entities.

For example, the same term may have different meanings depending on its surrounding context.

### 6.6.7 Scalability

As the number of CTI documents and vocabulary size increase, feature extraction and topic modeling can become computationally demanding.

Large-scale operational deployment may therefore require optimized processing pipelines and distributed infrastructure.

### 6.6.8 Real-Time Processing Constraints

The framework is designed with timely CTI analysis in mind, but actual real-time performance depends on the data ingestion mechanism, processing architecture, model complexity, and available computational resources.

If the experimental implementation is evaluated only on historical data, the results should be considered an offline evaluation rather than proof of operational real-time capability.

---

## 6.7 Future Work

Several improvements can be explored in future research.

### 6.7.1 Transformer-Based CTI Representation

Future implementations can incorporate transformer-based language models to obtain contextual representations of CTI documents.

Potential approaches include:

* BERT.
* RoBERTa.
* Domain-adapted transformer models.
* Cybersecurity-specific language models.

These representations may improve the ability to understand contextual relationships between cybersecurity terms.

### 6.7.2 Hybrid Classification Models

Future work can investigate hybrid approaches combining traditional machine learning, deep learning, and transformer-based representations.

For example:

```text
CTI Text
   ↓
Transformer Representation
   ↓
Feature Combination
   ↓
Machine Learning Classifier
   ↓
Threat Category
```

Such approaches can be compared against the traditional TF-IDF-based models used in the current study.

### 6.7.3 Dynamic Topic Modeling

The current temporal analysis can be extended using dynamic topic modeling techniques.

Future research can investigate models that explicitly represent topic evolution rather than independently estimating topics across temporal intervals.

This may provide a more detailed understanding of how cybersecurity topics emerge, change, merge, or disappear.

### 6.7.4 Real-Time CTI Stream Processing

Future versions of the framework can integrate continuous CTI feeds instead of relying primarily on static datasets.

A possible architecture is:

```text
Continuous CTI Sources
        ↓
Message Queue / Stream
        ↓
Real-Time Preprocessing
        ↓
Threat Classification
        ↓
Incremental Topic Analysis
        ↓
Temporal Monitoring
        ↓
Emerging Threat Score
        ↓
Alert / Dashboard
```

Technologies such as Kafka, streaming APIs, or event-driven architectures could be investigated for large-scale implementations.

### 6.7.5 Integration with External Threat Intelligence

Future work can integrate multiple CTI sources to improve validation.

Potential sources include:

* Vulnerability databases.
* Malware intelligence.
* Security advisories.
* Incident reports.
* Threat intelligence feeds.
* Attack technique databases.

Cross-source validation may reduce false emerging-threat signals caused by changes in a single source.

### 6.7.6 Knowledge Graph Integration

A future version could represent relationships between:

* Threat actors.
* Malware.
* Vulnerabilities.
* Indicators of compromise.
* Attack techniques.
* Campaigns.
* Organizations.

A knowledge graph could complement topic modeling by explicitly representing relationships between cybersecurity entities.

### 6.7.7 Explainable AI

Explainability techniques can be incorporated to help analysts understand why a classification or emerging-threat signal was generated.

For classification, explanations could identify important terms or features contributing to the prediction.

For emerging-threat detection, the system could report:

```text
Candidate Topic
      ↓
Why was it detected?
      ├── Increased prevalence
      ├── New terminology
      ├── Persistent activity
      ├── Increased document frequency
      └── Related threat categories
```

This would improve transparency and analyst usability.

### 6.7.8 Improved Emerging Threat Scoring

The current scoring framework can be further investigated using learned weights rather than manually specified weights.

Future studies could compare:

* Fixed weighting.
* Data-driven weighting.
* Supervised scoring.
* Anomaly detection.
* Ranking-based approaches.

The effectiveness of these alternatives can be evaluated using historical threat events.

### 6.7.9 Human-in-the-Loop Validation

Security analysts can be incorporated into the framework to validate candidate emerging threats.

A human-in-the-loop system could follow:

```text
Automated Detection
        ↓
Candidate Emerging Signal
        ↓
Analyst Review
        ↓
Validation / Rejection
        ↓
Feedback
        ↓
Model Improvement
```

Such an approach may improve the reliability of emerging-threat identification while retaining automation.

### 6.7.10 Multilingual CTI Analysis

Most CTI analysis systems focus primarily on English-language data.

Future research can investigate multilingual CTI processing to support threat intelligence published in different languages.

Multilingual transformer models and language-specific preprocessing techniques could be evaluated for this purpose.

### 6.7.11 Large-Scale Evaluation

Future studies should evaluate the framework using larger datasets collected over longer periods and from multiple independent sources.

This would provide a stronger basis for evaluating:

* Generalization.
* Scalability.
* Temporal robustness.
* Cross-source performance.
* Emerging-threat detection reliability.

### 6.7.12 Operational Security Dashboard

The framework can eventually be extended into an analyst-oriented dashboard displaying:

* Current threat categories.
* Classification statistics.
* Topic distributions.
* Temporal trends.
* Emerging threat scores.
* Related keywords.
* Source information.
* Historical comparisons.

Such a dashboard could make the research framework more useful for practical CTI analysis.

---

## 6.8 Final Conclusion

This research investigated an automated approach for analyzing Cyber Threat Intelligence using machine learning and temporal topic modeling.

The proposed framework integrates multiple analytical components into a unified workflow. Machine learning models are used to classify CTI documents into predefined threat categories, while topic modeling identifies latent cybersecurity themes. Temporal analysis then examines how these themes change over time, and the proposed emerging threat scoring mechanism combines multiple signals to identify candidate patterns that warrant further investigation.

The research demonstrates that CTI analysis can benefit from combining supervised classification with unsupervised and temporal analysis. Classification provides structured information about known threat categories, while topic modeling provides an opportunity to discover themes that may not be represented by predefined labels. Temporal analysis adds an additional dimension by identifying changes in threat-related information over time.

The proposed emerging threat scoring mechanism further extends this analysis by considering novelty, growth, persistence, activity, and term changes. However, the resulting signals should be interpreted as analytical indicators rather than definitive evidence of newly discovered or zero-day threats.

Overall, the research provides a foundation for automated and data-driven CTI analysis. The framework can be further enhanced through transformer-based representations, dynamic topic modeling, continuous CTI streams, external intelligence integration, explainable AI, knowledge graphs, and human-in-the-loop validation.

Future development toward a continuously operating CTI platform could provide more comprehensive support for security analysts by combining automated document processing, threat classification, temporal intelligence, and emerging-pattern prioritization within a single analytical environment.

## 6.9 Chapter Summary

This chapter presented the conclusion and future directions of the research.

The major outcomes of the study include:

* Development of an integrated CTI analysis framework.
* Automated classification of CTI documents.
* Discovery of latent cybersecurity topics.
* Temporal analysis of threat-related topics.
* Development of an emerging threat scoring mechanism.
* Experimental evaluation of classification and topic modeling components.
* Identification of limitations and potential sources of error.
* Identification of future improvements involving transformers, dynamic topic modeling, real-time streams, knowledge graphs, explainable AI, and human-in-the-loop validation.

The research provides a foundation for further investigation into automated, temporal, and intelligent Cyber Threat Intelligence analysis.
