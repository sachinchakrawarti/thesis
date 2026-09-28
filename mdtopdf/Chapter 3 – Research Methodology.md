# Chapter 3 – Research Methodology

## 3.1 Proposed Methodology

The proposed research methodology is designed to automate the analysis of Cyber Threat Intelligence (CTI) by combining Natural Language Processing (NLP), Machine Learning (ML), and temporal topic modeling techniques. The methodology focuses on two complementary objectives: **classification of known cyber threats** and **identification of potentially emerging threat patterns**.

The overall research workflow consists of the following major stages:

1. Dataset collection
2. Dataset integration and organization
3. Data preprocessing
4. Feature extraction and text representation
5. Cyber threat classification
6. Temporal topic modeling
7. Emerging threat detection
8. Performance evaluation
9. Comparative analysis and interpretation

The proposed methodology treats threat classification and emerging-threat detection as related but distinct analytical tasks. The classification component uses labeled CTI data to identify predefined threat categories, while the temporal topic modeling component analyzes the evolution of threat-related topics over time.

The overall workflow can be represented as:

```text
Cyber Threat Intelligence Sources
              |
              v
       Dataset Collection
              |
              v
      Dataset Organization
              |
              v
       Data Preprocessing
              |
              v
       Feature Extraction
              |
        +-----+-----+
        |           |
        v           v
Threat Classification   Temporal Topic Modeling
        |           |
        v           v
Known Threat Classes   Topic Distributions
        |           |
        |           v
        |      Temporal Analysis
        |           |
        |           v
        |    Emerging Threat Signals
        |           |
        +-----+-----+
              |
              v
       Evaluation & Analysis
```

The methodology is designed to support systematic analysis of textual CTI while maintaining a clear separation between supervised classification and unsupervised temporal analysis.

---

## 3.2 Dataset Collection

Dataset collection is the first major stage of the proposed methodology. The quality and relevance of the dataset directly influence the performance of the classification and topic modeling components.

The research focuses primarily on **textual Cyber Threat Intelligence reports and threat-related documents**. Such documents may contain information about malware, phishing, ransomware, vulnerabilities, denial-of-service attacks, exploit techniques, threat actors, campaigns, and other cybersecurity events.

### 3.2.1 Dataset Requirements

The dataset should contain sufficient textual information to support both supervised classification and temporal topic analysis.

The desired dataset fields include:

| Field      | Description                         |
| ---------- | ----------------------------------- |
| `id`       | Unique identifier for each document |
| `title`    | Title of the CTI report or document |
| `text`     | Main textual content                |
| `category` | Threat category or class label      |
| `date`     | Publication or observation date     |
| `source`   | Source of the CTI document          |
| `url`      | Reference URL where available       |

Not every source is required to provide all fields. When certain attributes are unavailable, the dataset may be standardized using the information available from the source.

### 3.2.2 Dataset Sources

Potential CTI data sources include:

* Cybersecurity reports
* Security advisories
* Malware analysis reports
* Vulnerability descriptions
* Cybersecurity incident reports
* Security blogs
* Threat intelligence publications
* Publicly available threat datasets
* Structured threat intelligence feeds

For this research, preference is given to datasets that provide both **textual information and temporal information**, because the proposed methodology requires documents to be analyzed according to their time of publication or observation.

### 3.2.3 Dataset Organization

Collected documents are combined into a standardized tabular format. Each document is treated as an individual observation.

A simplified representation is:

```text
+------+----------------+-------------------+------------+------------+
| ID   | Title          | Text              | Category   | Date       |
+------+----------------+-------------------+------------+------------+
| 001  | Report A       | Threat report...  | Malware    | 2024-01-05 |
| 002  | Report B       | Security alert... | Phishing   | 2024-01-08 |
| 003  | Report C       | Attack report...  | Ransomware | 2024-01-14 |
+------+----------------+-------------------+------------+------------+
```

The final dataset is divided into components required for supervised classification and temporal topic analysis.

### 3.2.4 Dataset Quality Considerations

Before model development, the dataset is examined for:

* Missing values
* Duplicate documents
* Invalid dates
* Empty text fields
* Incorrect or inconsistent labels
* Extremely short documents
* Formatting errors
* Class imbalance
* Temporal coverage

Data quality is particularly important because duplicate or incorrectly labeled documents can introduce bias into classification results and distort topic distributions.

---

## 3.3 Data Preprocessing

Raw CTI documents frequently contain irrelevant formatting, duplicated information, HTML elements, URLs, special characters, and cybersecurity-specific identifiers. Therefore, preprocessing is required before feature extraction and model training.

The preprocessing pipeline consists of several stages.

### 3.3.1 Data Cleaning

The collected dataset is first examined for incomplete and invalid records. Records with missing essential information may be removed or handled according to the availability of alternative information.

Duplicate documents are identified and removed to prevent repeated observations from influencing the model.

### 3.3.2 Text Normalization

Text normalization converts documents into a consistent representation.

Typical operations include:

* Converting text to lowercase where appropriate
* Removing unnecessary HTML tags
* Removing redundant whitespace
* Normalizing punctuation
* Removing irrelevant formatting
* Handling special characters
* Standardizing textual representations

However, cybersecurity-specific information should not be removed indiscriminately.

For example, identifiers such as:

```text
CVE-2024-XXXX
192.168.1.10
malware-family-name
example-domain.com
```

may contain useful cybersecurity information.

Therefore, preprocessing rules should preserve meaningful security-related tokens where they are relevant to the research task.

### 3.3.3 Tokenization

Tokenization divides a document into individual tokens.

For example:

```text
Original:
"Ransomware attacks targeted enterprise systems."

Tokens:
["ransomware", "attacks", "targeted", "enterprise", "systems"]
```

Tokenization provides the basic representation required by many NLP techniques.

### 3.3.4 Stop-Word Removal

Stop words are common words that generally provide limited discriminative information in traditional text classification.

Examples include:

```text
the
is
and
of
to
in
```

Removing stop words can reduce the dimensionality of the feature space.

However, domain-specific analysis should be used when deciding which words to remove because some common words may become meaningful in specific cybersecurity contexts.

### 3.3.5 Stemming and Lemmatization

Stemming reduces words to simplified root forms, whereas lemmatization attempts to convert words into their linguistically meaningful base forms.

For example:

```text
attacking
attacked
attacks

→ attack
```

Lemmatization may be preferred when maintaining semantic consistency is important.

### 3.3.6 Date Processing

Temporal information is essential for the proposed emerging-threat detection component.

Dates are therefore converted into a standardized format such as:

```text
YYYY-MM-DD
```

Documents can subsequently be grouped into temporal intervals such as:

* Daily
* Weekly
* Monthly
* Quarterly

The selected interval depends on the volume and temporal distribution of the dataset.

### 3.3.7 Preprocessing Pipeline

The complete preprocessing workflow can be represented as:

```text
Raw CTI Documents
        |
        v
Missing Value Handling
        |
        v
Duplicate Removal
        |
        v
Text Cleaning
        |
        v
Normalization
        |
        v
Tokenization
        |
        v
Stop-Word Processing
        |
        v
Lemmatization
        |
        v
Date Standardization
        |
        v
Clean CTI Dataset
```

---

## 3.4 Feature Extraction

Feature extraction converts preprocessed textual information into numerical representations that can be processed by Machine Learning algorithms.

Since ML models generally require numerical input, appropriate text representation is an essential component of the proposed methodology.

### 3.4.1 TF-IDF Representation

Term Frequency-Inverse Document Frequency (TF-IDF) is used as a primary traditional representation for textual CTI.

The TF-IDF value of a term $t$ in document $d$ can be expressed as:

$$
TFIDF(t,d) = TF(t,d) \times IDF(t)
$$

where:

$$
IDF(t) = \log\left(\frac{N}{df(t)}\right)
$$

Here:

* $N$ represents the total number of documents.
* $df(t)$ represents the number of documents containing term $t$.
* $TF(t,d)$ represents the frequency of term $t$ in document $d$.

TF-IDF assigns higher weights to terms that are important within a document but relatively uncommon across the complete corpus.

### 3.4.2 N-Gram Features

N-grams can be used to capture combinations of consecutive words.

For example:

```text
Text:
"distributed denial service attack"

Unigrams:
distributed
denial
service
attack

Bigrams:
distributed denial
denial service
service attack
```

N-gram features can capture meaningful cybersecurity phrases that may not be represented adequately by individual words.

Examples include:

```text
zero day
denial service
remote code execution
credential theft
```

### 3.4.3 Word Embeddings

Dense word representations such as Word2Vec and Doc2Vec can be used to represent semantic relationships between words or documents.

These approaches can provide richer representations than simple frequency-based methods. However, their effectiveness depends on the training corpus and the ability of the embeddings to capture cybersecurity-specific terminology.

### 3.4.4 Transformer-Based Representations

Transformer-based language models can generate contextual representations of CTI documents. These representations consider the surrounding context of words rather than treating each term independently.

Transformer representations may be used as an advanced feature representation or as a comparative modeling approach.

The selection of feature representation depends on the experimental design, computational resources, and characteristics of the dataset.

---

## 3.5 Threat Classification

The threat classification component is designed to categorize CTI documents into predefined threat classes.

The classification process consists of:

```text
Clean CTI Dataset
       |
       v
Feature Representation
       |
       v
Training / Testing Split
       |
       v
Model Training
       |
       v
Threat Prediction
       |
       v
Performance Evaluation
```

### 3.5.1 Classification Categories

The exact categories depend on the available labeled dataset.

Possible categories include:

* Malware
* Phishing
* Ransomware
* DDoS
* Vulnerability Exploitation
* Credential Attacks
* Botnets
* Other cyber threats

Only categories supported by the selected dataset should be used in the final experimental implementation.

### 3.5.2 Machine Learning Algorithms

Several supervised ML algorithms may be evaluated.

#### Naïve Bayes

Naïve Bayes is a probabilistic classification algorithm that assumes conditional independence between features. It is computationally efficient and has traditionally been used for text classification.

#### Logistic Regression

Logistic Regression estimates the probability of a document belonging to a particular class. It is commonly used as a baseline for high-dimensional text classification.

#### Support Vector Machine

Support Vector Machine (SVM) attempts to identify a decision boundary that separates classes in feature space. SVM is particularly suitable for high-dimensional sparse representations such as TF-IDF.

#### Random Forest

Random Forest combines multiple decision trees to produce a classification decision. It can model nonlinear relationships and provides an ensemble-based alternative to linear classifiers.

The final selection of models should be based on experimental comparison rather than assuming that one algorithm will always perform best.

### 3.5.3 Training and Testing

The labeled dataset is divided into training and testing subsets.

The training data are used to learn the classification model, while the test data are reserved for evaluating its performance on previously unseen documents.

A stratified split may be used when class distributions are uneven so that the relative representation of classes is maintained across the subsets.

### 3.5.4 Classification Output

For an input CTI document $d$, the classification model produces a predicted class:

$$
\hat{y} = f(d)
$$

where:

* $d$ is the input CTI document.
* $f$ represents the trained classification model.
* $\hat{y}$ represents the predicted threat category.

The predicted class can subsequently be used as an additional analytical attribute during CTI analysis.

---

## 3.6 Temporal Topic Modeling

Temporal topic modeling is used to discover latent themes in CTI data and investigate how those themes change over time.

Unlike supervised classification, topic modeling does not require every document to have a predefined threat label. This makes it useful for exploring patterns that may not correspond directly to known threat categories.

The temporal topic modeling pipeline consists of:

```text
Clean CTI Documents
        |
        v
Document-Time Association
        |
        v
Feature / Document-Term Matrix
        |
        v
Topic Modeling
        |
        v
Topic-Term Distributions
        |
        v
Topic-Document Distributions
        |
        v
Temporal Grouping
        |
        v
Topic Evolution Analysis
```

### 3.6.1 Document-Time Association

Each CTI document is associated with its publication or observation date.

For example:

```text
Document 1 → January 2024
Document 2 → January 2024
Document 3 → February 2024
Document 4 → March 2024
```

This allows the corpus to be divided into temporal periods.

### 3.6.2 Latent Dirichlet Allocation

Latent Dirichlet Allocation (LDA) can be used to discover latent topics.

For a corpus containing $K$ topics, LDA estimates:

* Topic distribution for each document.
* Word distribution for each topic.

The resulting topics can be interpreted by examining their highest-probability terms.

For example:

```text
Topic 1:
ransomware, encryption, victim, extortion, payload

Topic 2:
phishing, email, credential, campaign, login

Topic 3:
vulnerability, exploit, patch, remote, execution
```

The actual topics will depend on the selected dataset and model parameters.

### 3.6.3 Non-Negative Matrix Factorization

NMF may also be used as an alternative topic modeling approach.

Given a document-term matrix $X$, NMF decomposes it into two non-negative matrices:

$$
X \approx WH
$$

where:

* $W$ represents document-topic relationships.
* $H$ represents topic-term relationships.

Comparing LDA and NMF can provide additional insight into the stability and interpretability of discovered topics.

### 3.6.4 Temporal Topic Distribution

After topics are extracted, their prevalence is calculated for different time periods.

For topic $k$ and time period $t$, the topic prevalence can be represented as:

$$
P(k,t) = \frac{\text{documents associated with topic }k\text{ during }t}
{\text{total documents during }t}
$$

The resulting values can be visualized as temporal trends.

For example:

```text
Time →       T1     T2     T3     T4

Topic A     0.12   0.15   0.21   0.34
Topic B     0.30   0.27   0.22   0.18
Topic C     0.08   0.09   0.13   0.25
```

Such trends can be investigated for potential changes in the cyber threat landscape.

---

## 3.7 Emerging Threat Detection

The emerging-threat detection component uses temporal changes in CTI topics to identify potentially significant new or increasing patterns.

The objective is not to automatically declare every newly appearing topic as a confirmed cyber threat. Instead, the methodology generates **emerging-threat signals** that can be examined by analysts.

### 3.7.1 Topic Growth

For a topic $k$, its change between two consecutive periods can be calculated as:

$$
\Delta P_k(t) = P(k,t) - P(k,t-1)
$$

A positive value indicates that the relative prevalence of the topic has increased.

### 3.7.2 Growth Rate

A relative growth measure can also be calculated:

$$
GrowthRate_k(t) =
\frac{P(k,t)-P(k,t-1)}
{P(k,t-1)+\epsilon}
$$

where $\epsilon$ is a small value used to avoid division by zero.

This measure can help identify topics showing unusually large increases.

### 3.7.3 Emerging Topic Criteria

A topic may be considered a candidate emerging pattern when one or more of the following conditions are observed:

1. The topic appears for the first time in a later temporal period.
2. The topic shows a sustained increase in prevalence.
3. The topic demonstrates a substantial increase compared with previous periods.
4. New cybersecurity-specific terms become strongly associated with the topic.
5. The topic becomes increasingly associated with known threat categories or indicators.

These criteria are analytical signals rather than definitive evidence of a newly discovered attack.

### 3.7.4 Emerging Threat Detection Workflow

```text
Temporal CTI Data
        |
        v
Topic Extraction
        |
        v
Topic Distribution by Time
        |
        v
Calculate Topic Changes
        |
        v
Identify Increasing / New Topics
        |
        v
Analyze Important Terms
        |
        v
Generate Emerging-Threat Signals
        |
        v
Analyst Interpretation
```

### 3.7.5 Relationship with Threat Classification

Threat classification and emerging-threat detection provide complementary information.

Classification answers:

> **"What known threat category does this document belong to?"**

Temporal topic modeling addresses:

> **"What themes are changing or becoming increasingly prominent over time?"**

The combination can therefore provide a broader analytical view of CTI.

For example, a group of documents may be classified as belonging to a known threat category while temporal topic analysis reveals that a new vulnerability, malware family, or attack technique is increasingly associated with those documents.

---

## 3.8 Evaluation Metrics

The proposed methodology requires separate evaluation procedures for threat classification and temporal topic modeling.

### 3.8.1 Accuracy

Accuracy measures the proportion of correctly classified observations among all observations.

$$
Accuracy =
\frac{TP + TN}
{TP + TN + FP + FN}
$$

where:

* $TP$ = True Positive
* $TN$ = True Negative
* $FP$ = False Positive
* $FN$ = False Negative

Accuracy provides an overall measure of classification correctness but may be insufficient for imbalanced datasets.

### 3.8.2 Precision

Precision measures the proportion of predicted positive instances that are actually positive.

$$
Precision =
\frac{TP}
{TP + FP}
$$

Higher precision indicates fewer false-positive predictions.

### 3.8.3 Recall

Recall measures the proportion of actual positive instances that are correctly identified.

$$
Recall =
\frac{TP}
{TP + FN}
$$

Recall is particularly relevant when failing to identify a threat can have significant consequences.

### 3.8.4 F1-Score

F1-score is the harmonic mean of precision and recall.

$$
F1 =
2 \times
\frac{Precision \times Recall}
{Precision + Recall}
$$

F1-score provides a balanced measure when both precision and recall are important.

### 3.8.5 Confusion Matrix

A confusion matrix provides a class-wise representation of classification results.

For a multiclass threat classification problem, the matrix can be represented as:

```text
                    Predicted
              A       B       C       D
Actual A      TP      -       -       -
Actual B      -       TP      -       -
Actual C      -       -       TP      -
Actual D      -       -       -       TP
```

The confusion matrix can help identify which threat categories are frequently confused with one another.

### 3.8.6 Macro and Weighted Metrics

For multiclass classification, macro-averaged and weighted-averaged precision, recall, and F1-score may be reported.

**Macro averaging** gives equal importance to each class.

**Weighted averaging** considers the number of samples belonging to each class.

These metrics are useful when the dataset contains imbalanced threat categories.

### 3.8.7 Topic Coherence

Topic coherence can be used to evaluate the interpretability of discovered topics.

A higher coherence value generally indicates that the top words within a topic have stronger semantic or statistical relationships.

Topic coherence can be used to compare different numbers of topics or different topic modeling approaches.

### 3.8.8 Topic Diversity

Topic diversity measures how distinct the words used by different topics are.

High topic diversity indicates that different topics contain relatively different vocabularies rather than repeatedly using the same terms.

### 3.8.9 Temporal Trend Analysis

Emerging-threat detection can be evaluated using temporal characteristics such as:

* Topic frequency change
* Topic growth rate
* Topic persistence
* First appearance of topics
* Frequency of emerging-topic signals
* Relationship between emerging topics and known threat categories

Because an emerging topic is not necessarily a confirmed threat, these measures should primarily be interpreted as indicators of temporal change.

### 3.8.10 Comparative Evaluation

The final evaluation compares different classification models and, where applicable, different topic modeling configurations.

The comparison considers:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion matrix
* Training and inference characteristics
* Topic coherence
* Topic diversity
* Temporal stability
* Interpretability

The results obtained from these evaluations are presented and analyzed in Chapter 5.

---

## Chapter Summary

This chapter presented the proposed methodology for automated Cyber Threat Intelligence analysis. The methodology begins with CTI dataset collection and preparation, followed by text preprocessing and feature extraction.

Machine Learning algorithms are then used to classify CTI documents into predefined threat categories. In parallel, topic modeling is applied to discover latent themes, and temporal analysis is used to investigate how these themes evolve over time.

The emerging-threat detection component analyzes changes in topic prevalence and related textual patterns to generate candidate emerging-threat signals. Finally, classification and topic-modeling performance are evaluated using appropriate quantitative and analytical measures.

The methodology establishes the foundation for the experimental implementation and results presented in the subsequent chapters.
