# Chapter 2 – Literature Review

## 2.1 Cyber Threat Intelligence

Cyber Threat Intelligence (CTI) has become an important component of modern cybersecurity because of the increasing complexity, frequency, and diversity of cyber attacks. CTI focuses on collecting, processing, analyzing, and disseminating information related to cyber threats so that organizations can make informed decisions regarding cybersecurity operations.

Traditional cybersecurity systems primarily focus on detecting individual security events, whereas CTI provides contextual information that can help explain the nature of a threat, the techniques used by an attacker, the targeted assets, and potential consequences. This contextual information can support security monitoring, incident response, vulnerability management, risk assessment, and strategic decision-making.

CTI is commonly divided into strategic, tactical, operational, and technical intelligence. Strategic intelligence provides high-level information about the broader threat landscape and potential organizational impact. Tactical intelligence focuses on attacker tactics, techniques, and procedures (TTPs). Operational intelligence provides information about ongoing or planned campaigns, while technical intelligence contains technical indicators such as IP addresses, domains, URLs, file hashes, and malware signatures.

The CTI lifecycle generally consists of planning and direction, collection, processing, analysis, dissemination, and feedback. Each stage contributes to converting raw cybersecurity information into actionable intelligence. However, the continuous growth of CTI sources has created challenges related to data volume, heterogeneity, timeliness, and analysis.

Recent research has increasingly investigated automated CTI analysis using Natural Language Processing (NLP), Machine Learning (ML), Deep Learning (DL), and knowledge-based approaches. These techniques can process large volumes of textual information and extract useful patterns that may be difficult to identify manually.

The base research underlying this thesis demonstrates the use of Machine Learning and topic modeling for automated cyber threat classification and emerging-threat analysis. The study highlights the potential of combining classification with topic-based analysis to support automated CTI processing.

---

## 2.2 CTI Data Sources

The effectiveness of an automated CTI system depends significantly on the quality and diversity of its data sources. Cyber threat information is distributed across multiple sources, including structured databases, security reports, vulnerability repositories, security blogs, online forums, incident reports, and threat intelligence platforms.

### 2.2.1 Security Reports

Security reports published by cybersecurity organizations often contain detailed descriptions of attacks, malware, vulnerabilities, threat actors, and observed attack techniques. These reports are particularly useful for NLP-based CTI research because they contain rich textual descriptions.

### 2.2.2 Vulnerability Databases

Vulnerability databases provide structured information about known security vulnerabilities. The Common Vulnerabilities and Exposures (CVE) system is an important example. Vulnerability records can contain descriptions, affected products, severity information, and references.

Such information can be used to investigate vulnerability trends and relationships between vulnerabilities and observed attack techniques.

### 2.2.3 Malware Reports

Malware analysis reports provide information about malicious software, including malware families, capabilities, infection mechanisms, indicators of compromise, and behaviors. These reports are valuable sources for studying malware-related threats.

### 2.2.4 Security Blogs and Advisories

Cybersecurity companies and research organizations regularly publish security blogs and advisories describing newly discovered threats, vulnerabilities, campaigns, and attack techniques. These sources can provide relatively recent information about changes in the threat landscape.

### 2.2.5 Online Forums and Underground Sources

Cybersecurity-related forums and underground communities can contain information about vulnerabilities, malicious tools, credentials, campaigns, and attacker discussions. However, such sources can contain noisy, incomplete, duplicated, or unreliable information and therefore require careful preprocessing and validation.

### 2.2.6 News and Incident Reports

Cybersecurity news and incident reports provide information about major attacks, breaches, ransomware campaigns, and security incidents. Their temporal characteristics make them useful for studying the emergence and evolution of cyber threats.

### 2.2.7 CTI Platforms and Structured Feeds

Threat intelligence platforms can provide structured indicators and relationships between threat entities. Examples include feeds containing domains, IP addresses, URLs, hashes, malware identifiers, and other indicators.

For this research, the primary focus is on **textual CTI reports and threat-related documents**, as textual sources provide sufficient information for threat classification and temporal topic analysis.

---

## 2.3 Cyber Threat Classification

Cyber threat classification is the process of assigning cyber threat information to predefined categories based on characteristics contained in the data. Classification can help security analysts organize large collections of threat reports and prioritize information for further investigation.

Threat categories may include malware, phishing, ransomware, denial-of-service attacks, vulnerability exploitation, credential attacks, botnets, and other forms of malicious activity.

Traditional threat classification relies heavily on manually defined rules, signatures, and expert analysis. Although such approaches remain useful for specific detection tasks, they can become difficult to maintain when threat descriptions and attack techniques continuously change.

Machine Learning provides an alternative by learning classification patterns from previously labeled examples. A typical ML-based CTI classification pipeline includes data collection, preprocessing, feature extraction, model training, prediction, and evaluation.

Text representation methods such as Bag-of-Words, Term Frequency-Inverse Document Frequency (TF-IDF), Word2Vec, and Doc2Vec have been investigated for converting threat-related text into numerical representations.

The quality of a classification system depends on several factors, including dataset size, label quality, class balance, text representation, model selection, and the similarity between training and unseen data.

A major limitation of supervised classification is its dependence on predefined classes. If a new type of threat does not correspond to an existing training category, a conventional classifier may incorrectly assign it to an existing class. This limitation motivates the use of complementary techniques such as topic modeling and temporal analysis.

---

## 2.4 NLP for CTI

Natural Language Processing (NLP) provides techniques for automatically processing and analyzing human language. Since a large portion of CTI is available as textual information, NLP has become an important technology for automated threat intelligence analysis.

A typical NLP pipeline for CTI may include:

1. Text cleaning
2. Tokenization
3. Lowercase conversion
4. Removal of punctuation and irrelevant characters
5. Stop-word removal
6. Stemming or lemmatization
7. Named entity extraction
8. Feature extraction
9. Text representation

Cybersecurity text presents several challenges that distinguish it from general-domain text. CTI documents frequently contain technical terminology, product names, malware families, vulnerability identifiers, IP addresses, domain names, file hashes, abbreviations, and specialized security terminology.

Consequently, conventional NLP preprocessing techniques must be carefully adapted. For example, indiscriminate removal of special characters may remove meaningful information such as CVE identifiers, IP addresses, URLs, or hashes.

### 2.4.1 Text Representation

Traditional NLP-based CTI systems commonly use sparse representations such as Bag-of-Words and TF-IDF.

The TF-IDF representation assigns a weight to a term based on its frequency within a document and its frequency across the complete collection. It can be represented as:

$$
TFIDF(t,d) = TF(t,d) \times IDF(t)
$$

where $TF(t,d)$ represents the frequency of term $t$ in document $d$, and $IDF(t)$ represents the inverse document frequency of the term.

Word embedding approaches such as Word2Vec and Doc2Vec provide dense representations and can capture semantic relationships between words or documents. However, these representations may not fully capture context-dependent meanings.

The development of contextual language models has subsequently enabled more advanced representations of cybersecurity text.

---

## 2.5 Machine Learning for CTI

Machine Learning has been extensively investigated for automated cybersecurity analysis. In CTI, ML techniques can be applied to classification, clustering, anomaly detection, entity extraction, malware analysis, and threat prediction.

### 2.5.1 Supervised Learning

Supervised learning uses labeled examples to train a model to predict predefined categories. Common algorithms include:

* Naïve Bayes
* Logistic Regression
* Support Vector Machine (SVM)
* Decision Tree
* Random Forest
* Gradient Boosting
* k-Nearest Neighbors (k-NN)

These algorithms have been used for text classification because of their relatively straightforward implementation and computational efficiency.

SVM-based methods can be particularly effective for high-dimensional sparse text representations. Random Forest can capture nonlinear relationships and is less dependent on assumptions about the underlying feature distribution.

### 2.5.2 Unsupervised Learning

Unsupervised learning operates without predefined labels. It can be used to discover hidden structures and groups within CTI datasets.

Clustering algorithms such as K-Means and hierarchical clustering can group documents based on their feature representations. Such approaches can be useful when labeled CTI data are limited.

However, interpreting automatically generated clusters can be difficult because the resulting groups may not directly correspond to meaningful cybersecurity categories.

### 2.5.3 Evaluation of ML-Based CTI Classification

Common evaluation metrics include accuracy, precision, recall, and F1-score.

Accuracy is defined as:

$$
Accuracy = \frac{TP + TN}{TP + TN + FP + FN}
$$

Precision is:

$$
Precision = \frac{TP}{TP + FP}
$$

Recall is:

$$
Recall = \frac{TP}{TP + FN}
$$

The F1-score combines precision and recall:

$$
F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}
$$

For imbalanced CTI datasets, relying only on accuracy can be misleading. Therefore, precision, recall, F1-score, and class-wise performance should also be considered.

---

## 2.6 Deep Learning for CTI

Deep Learning extends traditional Machine Learning by using multi-layer neural networks to learn increasingly complex representations from data. Deep learning approaches have been applied to several cybersecurity tasks, including malware detection, intrusion detection, threat classification, and security-text analysis.

### 2.6.1 Convolutional Neural Networks

Convolutional Neural Networks (CNNs) can identify local patterns in sequential data and have been applied to text classification. In CTI analysis, CNN-based architectures can learn local word or character patterns that may be associated with specific threat categories.

### 2.6.2 Recurrent Neural Networks

Recurrent Neural Networks (RNNs) are designed to process sequential information. Variants such as Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) can capture dependencies across sequences and have been used in cybersecurity text classification and event analysis.

However, recurrent architectures process sequences sequentially, which can make training less efficient for very long documents.

### 2.6.3 Limitations of Traditional Deep Learning

Although CNN, RNN, LSTM, and GRU models can learn useful representations automatically, they may face limitations when processing long documents or modeling relationships between distant words.

These limitations contributed to the development and adoption of attention-based architectures and Transformer models.

---

## 2.7 Transformer Models

Transformer models have significantly changed the field of Natural Language Processing by introducing attention-based mechanisms for modeling relationships between tokens.

The Transformer architecture was introduced to address limitations associated with sequential processing in recurrent neural networks. Its self-attention mechanism allows the model to consider relationships between different parts of a sequence simultaneously.

A simplified scaled dot-product attention mechanism is expressed as:

$$
Attention(Q,K,V) =
softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

where $Q$, $K$, and $V$ represent the query, key, and value matrices, respectively, and $d_k$ represents the dimensionality of the key vectors.

Transformer-based models such as BERT and its variants have demonstrated strong performance in many NLP tasks. Domain-specific models have also been developed to better represent specialized language.

For cybersecurity applications, Transformer models can be used for:

* Threat report classification
* Malware-related text analysis
* Vulnerability information extraction
* Threat entity recognition
* Attack technique identification
* Security event classification
* CTI document representation

One important advantage of Transformer-based representations is their ability to capture contextual relationships between words. The meaning of a cybersecurity term can depend strongly on its surrounding context, and contextual representations can capture this information more effectively than traditional word representations.

However, Transformer models may require substantial computational resources and large datasets for effective training or fine-tuning. Their computational requirements can also make deployment challenging in resource-constrained environments.

For this research, Transformer-based representations can be considered as an advanced feature representation or comparative approach alongside conventional Machine Learning methods.

---

## 2.8 Topic Modeling

Topic modeling is an unsupervised learning technique used to discover latent thematic structures within a collection of documents. Unlike supervised classification, topic modeling does not require predefined document labels.

Topic modeling is particularly relevant to CTI because threat reports may contain information about previously unknown or changing threat categories. By identifying groups of frequently co-occurring terms, topic models can provide insights into the underlying themes present in a collection of threat documents.

### 2.8.1 Latent Dirichlet Allocation

Latent Dirichlet Allocation (LDA) is one of the most widely used probabilistic topic modeling techniques.

LDA assumes that:

* Each document is associated with a mixture of topics.
* Each topic is represented by a probability distribution over words.
* Documents are generated through combinations of latent topics.

The probability of a document can be represented conceptually as a mixture of its underlying topics.

LDA has been applied to cybersecurity text to identify major themes and investigate changes in cyber threat discussions. Its interpretable topic-word distributions make it useful for exploratory analysis.

### 2.8.2 Non-Negative Matrix Factorization

Non-Negative Matrix Factorization (NMF) is another approach for extracting latent topics from document-term matrices.

Given a non-negative matrix $X$, NMF attempts to approximate it as:

$$
X \approx WH
$$

where $W$ represents document-topic relationships and $H$ represents topic-term relationships.

NMF can produce interpretable topics and can be applied to TF-IDF representations.

### 2.8.3 Dynamic and Temporal Topic Modeling

Traditional topic modeling generally treats a document collection as static. However, cyber threats evolve continuously. Temporal topic modeling addresses this limitation by considering the time associated with documents.

Documents can be grouped into time intervals such as:

* Daily
* Weekly
* Monthly
* Quarterly
* Yearly

Topics can then be extracted for different time periods and compared to identify changes in topic prevalence.

For example, if a topic related to a particular vulnerability, malware family, or attack technique shows a significant increase over successive periods, it may represent an emerging trend requiring further investigation.

Temporal topic analysis therefore provides an important foundation for the emerging-threat detection component of this research.

---

## 2.9 Emerging Threat Detection

Emerging threat detection refers to the identification of new, changing, or increasingly important cyber threat patterns. Unlike conventional classification, emerging-threat detection is concerned not only with known categories but also with changes that may indicate the development of new threats.

Cyber threats can emerge through several mechanisms, including:

* New vulnerabilities
* New malware families
* Changes in malware behavior
* New attack techniques
* New threat-actor campaigns
* Increasing exploitation of existing vulnerabilities
* Changes in attacker terminology
* New combinations of existing techniques

Traditional supervised classification can identify threats that belong to known classes, but it may not explicitly identify new patterns outside the training distribution.

Topic modeling provides a complementary approach. By monitoring topic distributions over time, it is possible to identify topics whose frequency, vocabulary, or relationships change substantially.

### 2.9.1 Temporal Trend Analysis

Let $T_k(t)$ represent the prevalence of topic $k$ during time period $t$. A temporal trend can be represented as:

$$
Trend_k(t) = T_k(t) - T_k(t-1)
$$

A sustained increase in a topic's prevalence may indicate growing attention or activity associated with that topic.

However, an increase in a topic does not automatically imply the existence of a confirmed new cyber threat. Topic changes can also result from news coverage, repeated reporting, major security announcements, or changes in data collection. Therefore, emerging-topic detection should be treated as a signal-generation mechanism rather than definitive threat confirmation.

### 2.9.2 Emerging Threat Detection Using Topic Modeling

The base research associated with this thesis investigated the use of LDA and NMF for analyzing topic distributions over time and identifying emerging threats. Its findings demonstrate that topic modeling can complement supervised threat classification by providing information about changing threat-related themes.

This research direction is particularly relevant because emerging threats may not have sufficient labeled examples for conventional supervised learning. Temporal topic modeling can instead identify changes in the distribution of textual themes without requiring every emerging pattern to have a predefined label.

---

## 2.10 Research Gap

The literature indicates that Machine Learning, Deep Learning, NLP, Transformer models, and topic modeling have each been applied to different aspects of cybersecurity and CTI analysis. However, several challenges remain.

### 2.10.1 Gap in Automated Classification and Emerging-Threat Analysis

Many supervised approaches focus primarily on assigning documents to predefined threat categories. Such systems can perform well for known classes but may provide limited information about previously unseen or emerging patterns.

### 2.10.2 Gap in Temporal Analysis

A considerable portion of CTI classification research treats documents independently or considers the dataset as a static collection. Cyber threats, however, change over time. A framework that explicitly considers temporal changes can provide additional information about threat evolution.

### 2.10.3 Gap in Integration of Classification and Topic Modeling

Threat classification and topic modeling are often considered separate analytical tasks. Classification provides predefined categories, while topic modeling discovers latent themes. Integrating both approaches can provide complementary information about known and evolving threats.

### 2.10.4 Gap in Handling Unstructured CTI

A significant amount of CTI is available in unstructured textual form. Extracting useful representations from such data remains challenging because cybersecurity text contains specialized terminology, identifiers, abbreviations, and domain-specific language.

### 2.10.5 Gap in Emerging Threat Identification

Existing classification models generally depend on historical labels. Consequently, they may have difficulty identifying new threat patterns that are not represented in the training dataset. Temporal topic analysis provides a possible approach for identifying changes in threat-related discussions and activity.

### 2.10.6 Gap in Timely CTI Analysis

The rapidly changing cybersecurity environment requires efficient processing of newly available information. Systems that require extensive manual preprocessing and analysis may introduce delays between the appearance of threat information and its interpretation.

### 2.10.7 Proposed Research Direction

Based on the identified gaps, this research proposes an integrated framework combining **Machine Learning-based cyber threat classification with temporal topic modeling for emerging threat detection**.

The proposed research aims to:

1. Process and normalize textual CTI data.
2. Extract meaningful textual features.
3. Classify CTI documents into predefined cyber threat categories.
4. Discover latent threat-related topics.
5. Track topic distributions across time.
6. Identify significant changes and potentially emerging threat patterns.
7. Evaluate classification and temporal-analysis performance.
8. Provide an integrated analytical framework for automated CTI analysis.

The proposed approach therefore addresses the gap between **known-threat classification** and **emerging-threat pattern detection** by combining supervised and unsupervised analytical techniques within a temporal CTI framework.

---

## Chapter Summary

This chapter reviewed the major research areas relevant to automated Cyber Threat Intelligence analysis. The discussion covered CTI concepts and data sources, cyber threat classification, Natural Language Processing, Machine Learning, Deep Learning, Transformer models, topic modeling, and emerging-threat detection.

The literature demonstrates that Machine Learning can effectively support automated classification of known cyber threat categories, while topic modeling can help discover latent structures within unstructured CTI data. Temporal analysis provides an additional dimension for examining how these topics change over time.

The review also identified limitations in existing approaches, particularly the dependence on predefined categories, limited temporal analysis, and the separation between threat classification and emerging-threat detection. These observations provide the foundation for the methodology proposed in the subsequent chapters of this thesis.
