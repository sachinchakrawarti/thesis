# Chapter 5 – Experimental Results and Analysis

## 5.1 Experimental Setup

The experimental evaluation was conducted to assess the performance of the proposed Cyber Threat Intelligence (CTI) analysis framework. The framework integrates machine learning-based threat classification, topic modeling, temporal analysis, and emerging threat scoring.

The experiments were designed to evaluate the following aspects:

1. Quality and distribution of the collected CTI dataset.
2. Performance of machine learning models for threat classification.
3. Comparative performance of different classification algorithms.
4. Quality and interpretability of discovered topics.
5. Temporal evolution of cybersecurity-related topics.
6. Identification of candidate emerging threat patterns.
7. Effectiveness of the proposed emerging threat scoring mechanism.
8. Computational and practical limitations of the framework.

The experimental workflow follows the sequence:

```text
CTI Dataset
     ↓
Dataset Validation
     ↓
Preprocessing
     ↓
Feature Extraction
     ↓
Classification Experiments
     ↓
Topic Modeling Experiments
     ↓
Temporal Topic Analysis
     ↓
Emerging Threat Scoring
     ↓
Performance Evaluation
     ↓
Comparative Analysis
```

The experiments were performed using the implementation described in Chapter 4. The exact dataset version, preprocessing configuration, model parameters, and software versions used for the final experiments are documented to ensure reproducibility.

### 5.1.1 Experimental Configuration

The major experimental parameters are summarized below.

| Parameter               | Configuration                                        |
| ----------------------- | ---------------------------------------------------- |
| Programming Language    | Python                                               |
| Text Representation     | TF-IDF                                               |
| N-gram Range            | 1–2                                                  |
| Train-Test Split        | 80:20                                                |
| Random State            | 42                                                   |
| Classification Models   | Naive Bayes, Logistic Regression, SVM, Random Forest |
| Topic Models            | LDA, NMF                                             |
| Temporal Unit           | Based on dataset coverage                            |
| Evaluation Metrics      | Accuracy, Precision, Recall, F1-score                |
| Topic Evaluation        | Coherence, Topic Diversity, Interpretability         |
| Visualization           | Matplotlib / Seaborn                                 |
| Development Environment | Jupyter Notebook                                     |

The final values should be updated according to the actual configuration used in the experiments.

---

## 5.2 Hardware and Software Requirements

The proposed framework requires a general-purpose computing environment capable of processing textual datasets and training machine learning and topic modeling models.

### 5.2.1 Hardware Requirements

The minimum and recommended configurations are presented below.

| Component        | Requirement                             |
| ---------------- | --------------------------------------- |
| Processor        | Modern multi-core CPU                   |
| RAM              | Minimum 8 GB; 16 GB or more recommended |
| Storage          | Minimum 10 GB available space           |
| GPU              | Optional for traditional ML and LDA/NMF |
| Operating System | Windows / Linux / macOS                 |

For the proposed experiments, CPU-based execution is sufficient for traditional machine learning, TF-IDF feature extraction, LDA, and NMF for moderate-sized datasets.

If transformer-based representations or large-scale deep learning models are incorporated in future experiments, GPU acceleration may be beneficial.

### 5.2.2 Software Requirements

| Software         | Purpose                                 |
| ---------------- | --------------------------------------- |
| Python           | Primary programming language            |
| Jupyter Notebook | Experimental development                |
| Pandas           | Dataset processing                      |
| NumPy            | Numerical computation                   |
| scikit-learn     | Machine learning and feature extraction |
| NLTK / spaCy     | Natural Language Processing             |
| Gensim           | Topic modeling where applicable         |
| Matplotlib       | Visualization                           |
| Seaborn          | Statistical visualization               |
| Git              | Version control                         |

The exact software versions should be recorded in the final experimental environment.

---

## 5.3 Dataset Distribution Analysis

The dataset distribution was analyzed before model training to understand its size, class composition, temporal coverage, and potential imbalance.

### 5.3.1 Dataset Size

The final dataset contains:

* **Total documents:** `[N]`
* **Training documents:** `[N_train]`
* **Testing documents:** `[N_test]`
* **Number of threat categories:** `[C]`
* **Number of temporal periods:** `[T]`

Replace the bracketed values with the actual experimental results.

### 5.3.2 Class Distribution

The distribution of documents across threat categories is an important factor in supervised classification.

| Threat Category            | Number of Documents | Percentage |
| -------------------------- | ------------------: | ---------: |
| Malware                    |               `[N]` |     `[X]%` |
| Phishing                   |               `[N]` |     `[X]%` |
| Ransomware                 |               `[N]` |     `[X]%` |
| DDoS                       |               `[N]` |     `[X]%` |
| Vulnerability Exploitation |               `[N]` |     `[X]%` |
| Other                      |               `[N]` |     `[X]%` |
| **Total**                  |           **`[N]`** |   **100%** |

The final categories should correspond to the labels available in the selected dataset.

### 5.3.3 Class Imbalance

Class imbalance occurs when some threat categories contain substantially more documents than others.

The class distribution was examined using:

* Number of documents per class.
* Percentage contribution of each class.
* Majority-to-minority class ratio.
* Macro and weighted evaluation metrics.

If substantial imbalance exists, accuracy alone may not provide a complete representation of model performance. Therefore, precision, recall, and F1-score are also considered.

### 5.3.4 Temporal Distribution

The temporal distribution of documents was analyzed to determine the availability of CTI information across different periods.

| Time Period | Number of Documents |
| ----------- | ------------------: |
| Period 1    |               `[N]` |
| Period 2    |               `[N]` |
| Period 3    |               `[N]` |
| Period 4    |               `[N]` |
| ...         |                 ... |

The temporal distribution is particularly important because the proposed emerging-threat analysis depends on observing changes across time.

### 5.3.5 Dataset Quality Analysis

The dataset was examined for:

* Missing values.
* Duplicate records.
* Empty documents.
* Invalid dates.
* Incorrect labels.
* Extremely short documents.
* Duplicate textual content.
* Inconsistent formatting.

The number of records removed during preprocessing should be reported.

| Data Quality Check | Records Identified | Records Removed/Handled |
| ------------------ | -----------------: | ----------------------: |
| Missing text       |              `[N]` |                   `[N]` |
| Missing date       |              `[N]` |                   `[N]` |
| Duplicate records  |              `[N]` |                   `[N]` |
| Empty documents    |              `[N]` |                   `[N]` |
| Invalid labels     |              `[N]` |                   `[N]` |

---

## 5.4 Final Dataset Format

After data cleaning and preprocessing, the CTI dataset was converted into a structured format suitable for classification and temporal topic analysis.

The final dataset contains the following fields:

| Field        | Data Type   | Description                       |
| ------------ | ----------- | --------------------------------- |
| `id`         | String      | Unique document identifier        |
| `title`      | String      | CTI document title                |
| `text`       | Text        | Original or extracted CTI content |
| `clean_text` | Text        | Preprocessed text                 |
| `category`   | Categorical | Threat class                      |
| `date`       | Date        | Publication/observation date      |
| `source`     | String      | Source of CTI information         |
| `url`        | String      | Reference URL                     |
| `period`     | Categorical | Temporal grouping                 |

The cleaned text is used for feature extraction, while the date and period fields are used for temporal topic analysis.

---

## 5.5 Dataset Availability

The dataset used in the experimental study should be documented according to its source, licensing conditions, and reproducibility requirements.

Where redistribution is permitted, the processed dataset may be included with the project repository. If redistribution is restricted, the thesis should provide:

* Dataset source.
* Dataset collection procedure.
* Collection period.
* Number of documents.
* Preprocessing procedure.
* Dataset schema.
* Relevant source references.

The dataset should not contain sensitive personal information or restricted information that cannot legally or ethically be redistributed.

The final thesis should clearly distinguish between:

1. Original publicly available data.
2. Data collected by the researcher.
3. Preprocessed data.
4. Experiment-generated features and results.

---

## 5.6 Model Implementation

The classification and topic modeling experiments were implemented using the workflow described in Chapter 4.

### 5.6.1 Feature Extraction

TF-IDF was used to transform preprocessed CTI text into numerical feature vectors.

The resulting matrix is represented as:

$$
X \in \mathbb{R}^{N \times V}
$$

where:

* $N$ represents the number of documents.
* $V$ represents the vocabulary size.

The feature matrix was subsequently supplied to the classification algorithms.

### 5.6.2 Classification Models

The following models were evaluated:

1. Naive Bayes
2. Logistic Regression
3. Support Vector Machine
4. Random Forest

The objective was not only to evaluate individual models but also to understand how different algorithms behave on CTI text classification.

### 5.6.3 Training Configuration

The dataset was divided into training and testing subsets.

```text
Complete Dataset
       │
       ├────────────── 80% ──────────────┐
       │                                 │
       ▼                                 ▼
 Training Dataset                  Testing Dataset
       │                                 │
       ▼                                 ▼
 Model Training                   Final Evaluation
```

Where appropriate, stratified sampling was used to maintain approximately similar class proportions between training and testing sets.

---

## 5.7 Classification Performance

The performance of the classification models was evaluated using accuracy, precision, recall, and F1-score.

### 5.7.1 Accuracy

Accuracy measures the proportion of correctly classified documents.

$$
Accuracy =
\frac{TP+TN}
{TP+TN+FP+FN}
$$

For multiclass classification, accuracy represents the proportion of correctly predicted samples across all classes.

### 5.7.2 Precision

Precision measures the proportion of predicted positive samples that are actually positive.

$$
Precision =
\frac{TP}
{TP+FP}
$$

### 5.7.3 Recall

Recall measures the proportion of actual positive samples correctly identified by the model.

$$
Recall =
\frac{TP}
{TP+FN}
$$

### 5.7.4 F1-Score

The F1-score is the harmonic mean of precision and recall.

$$
F1 =
2\times
\frac{Precision\times Recall}
{Precision+Recall}
$$

For multiclass classification, macro-average and weighted-average F1-scores can be reported.

### 5.7.5 Overall Classification Results

The final classification results should be reported in the following format.

| Model               |   Accuracy |  Precision |     Recall |   F1-Score |
| ------------------- | ---------: | ---------: | ---------: | ---------: |
| Naive Bayes         | `[XX.XX]%` | `[XX.XX]%` | `[XX.XX]%` | `[XX.XX]%` |
| Logistic Regression | `[XX.XX]%` | `[XX.XX]%` | `[XX.XX]%` | `[XX.XX]%` |
| SVM                 | `[XX.XX]%` | `[XX.XX]%` | `[XX.XX]%` | `[XX.XX]%` |
| Random Forest       | `[XX.XX]%` | `[XX.XX]%` | `[XX.XX]%` | `[XX.XX]%` |

The values should be replaced with the actual experimental measurements.

---

## 5.8 Comparative Model Analysis

The classification models were compared using their performance across the selected evaluation metrics.

The comparison focuses on:

* Overall accuracy.
* Precision.
* Recall.
* F1-score.
* Class-wise performance.
* Robustness to class imbalance.
* Computational requirements.

A comparative table can be used to summarize the results.

| Model               |  Accuracy |  Macro F1 | Weighted F1 | Training Time |
| ------------------- | --------: | --------: | ----------: | ------------: |
| Naive Bayes         | `[XX.XX]` | `[XX.XX]` |   `[XX.XX]` |    `[XX] sec` |
| Logistic Regression | `[XX.XX]` | `[XX.XX]` |   `[XX.XX]` |    `[XX] sec` |
| SVM                 | `[XX.XX]` | `[XX.XX]` |   `[XX.XX]` |    `[XX] sec` |
| Random Forest       | `[XX.XX]` | `[XX.XX]` |   `[XX.XX]` |    `[XX] sec` |

The comparison should be interpreted according to the experimental results rather than assuming that a particular algorithm will perform best before experimentation.

### 5.8.1 Class-Wise Performance

Class-wise evaluation provides additional information about categories that are easier or more difficult to classify.

| Threat Category | Precision |    Recall |  F1-Score | Support |
| --------------- | --------: | --------: | --------: | ------: |
| Category 1      | `[XX.XX]` | `[XX.XX]` | `[XX.XX]` |   `[N]` |
| Category 2      | `[XX.XX]` | `[XX.XX]` | `[XX.XX]` |   `[N]` |
| Category 3      | `[XX.XX]` | `[XX.XX]` | `[XX.XX]` |   `[N]` |
| Category 4      | `[XX.XX]` | `[XX.XX]` | `[XX.XX]` |   `[N]` |

Differences between classes may result from differences in sample size, terminology, topic overlap, and the similarity of threat descriptions.

---

## 5.9 Confusion Matrix Analysis

A confusion matrix was generated to analyze the classification behavior of the selected models.

For a multiclass classification problem, the confusion matrix is represented as:

$$
C_{ij}
=
\text{Number of samples belonging to class }i
\text{ predicted as class }j
$$

An example structure is:

| Actual / Predicted | Class A | Class B | Class C | Class D |
| ------------------ | ------: | ------: | ------: | ------: |
| Class A            |   `[N]` |   `[N]` |   `[N]` |   `[N]` |
| Class B            |   `[N]` |   `[N]` |   `[N]` |   `[N]` |
| Class C            |   `[N]` |   `[N]` |   `[N]` |   `[N]` |
| Class D            |   `[N]` |   `[N]` |   `[N]` |   `[N]` |

The diagonal values represent correctly classified documents, while off-diagonal values represent classification errors.

### 5.9.1 Misclassification Analysis

Misclassification can occur when different threat categories share similar terminology.

For example, documents discussing credential theft may contain terminology associated with both phishing and malware. Similarly, vulnerability exploitation can occur as part of ransomware or other attack campaigns.

The confusion matrix therefore provides information about semantic overlap between threat categories.

---

## 5.10 Precision-Recall-F1 Analysis

Precision, recall, and F1-score were analyzed to obtain a more detailed understanding of classification performance.

A model with high precision but relatively low recall may correctly classify its positive predictions while missing a substantial number of relevant documents.

Conversely, a model with high recall but lower precision may identify more relevant documents while generating additional false positives.

For CTI classification, both types of errors can affect downstream analysis. Therefore, F1-score and class-wise results are considered along with accuracy.

The final results should be reported as:

| Model               | Macro Precision | Macro Recall |  Macro F1 | Weighted F1 |
| ------------------- | --------------: | -----------: | --------: | ----------: |
| Naive Bayes         |       `[XX.XX]` |    `[XX.XX]` | `[XX.XX]` |   `[XX.XX]` |
| Logistic Regression |       `[XX.XX]` |    `[XX.XX]` | `[XX.XX]` |   `[XX.XX]` |
| SVM                 |       `[XX.XX]` |    `[XX.XX]` | `[XX.XX]` |   `[XX.XX]` |
| Random Forest       |       `[XX.XX]` |    `[XX.XX]` | `[XX.XX]` |   `[XX.XX]` |

---

## 5.11 Topic Modeling Results

Topic modeling was performed to discover latent themes within the CTI corpus.

Both LDA and NMF can be evaluated to determine their suitability for the selected dataset.

### 5.11.1 Discovered Topics

The top terms associated with each topic were extracted from the trained topic model.

| Topic   | Top Terms                    | Interpretation     |
| ------- | ---------------------------- | ------------------ |
| Topic 1 | `[term1, term2, term3, ...]` | `[Interpretation]` |
| Topic 2 | `[term1, term2, term3, ...]` | `[Interpretation]` |
| Topic 3 | `[term1, term2, term3, ...]` | `[Interpretation]` |
| Topic 4 | `[term1, term2, term3, ...]` | `[Interpretation]` |
| Topic 5 | `[term1, term2, term3, ...]` | `[Interpretation]` |

Topic interpretations are based on the semantic relationship between the representative terms and the underlying documents.

### 5.11.2 Topic Interpretability

Topic interpretability was assessed by examining:

* Co-occurrence of representative terms.
* Semantic consistency.
* Relevance to cybersecurity.
* Representative documents.
* Stability across experiments.

Topics containing unrelated or highly generic terms may indicate that the selected topic number or preprocessing configuration requires further adjustment.

---

## 5.12 Topic Coherence Analysis

Topic coherence was used to quantitatively evaluate the semantic consistency of discovered topics.

The coherence score provides an indication of how frequently the top terms associated with a topic occur together or relate semantically within the corpus.

The results can be reported as:

| Number of Topics | LDA Coherence | NMF Coherence |
| ---------------: | ------------: | ------------: |
|                5 |     `[XX.XX]` |     `[XX.XX]` |
|               10 |     `[XX.XX]` |     `[XX.XX]` |
|               15 |     `[XX.XX]` |     `[XX.XX]` |
|               20 |     `[XX.XX]` |     `[XX.XX]` |

The final number of topics should be selected using coherence together with interpretability and temporal stability.

A higher coherence value generally indicates stronger semantic consistency, although coherence alone should not determine the final topic configuration.

---

## 5.13 Temporal Topic Analysis

The temporal analysis examined how discovered topics changed across the selected time periods.

The topic prevalence for topic $k$ at time $t$ is represented by:

$$
P(k,t)
=
\frac{\sum_{d\in D_t}\theta_{d,k}}
{|D_t|}
$$

The difference between two consecutive periods is:

$$
\Delta P_k(t)
=
P(k,t)-P(k,t-1)
$$

### 5.13.1 Topic Distribution Over Time

The temporal topic matrix is reported as:

| Time Period | Topic 1 | Topic 2 | Topic 3 | Topic 4 |
| ----------- | ------: | ------: | ------: | ------: |
| Period 1    |   `[X]` |   `[X]` |   `[X]` |   `[X]` |
| Period 2    |   `[X]` |   `[X]` |   `[X]` |   `[X]` |
| Period 3    |   `[X]` |   `[X]` |   `[X]` |   `[X]` |
| Period 4    |   `[X]` |   `[X]` |   `[X]` |   `[X]` |

The resulting trends are visualized using time-series plots and heatmaps.

### 5.13.2 Topic Growth

The growth rate of a topic is calculated using:

$$
G_k(t)
=
\frac{P(k,t)-P(k,t-1)}
{P(k,t-1)+\epsilon}
$$

Topics with substantial positive changes are examined further to determine whether the increase corresponds to meaningful cybersecurity activity.

### 5.13.3 Topic Persistence

A topic that appears across several consecutive periods may represent a persistent threat theme rather than a short-lived anomaly.

Persistence is therefore considered alongside growth and novelty when identifying candidate emerging threats.

---

## 5.14 Emerging Threat Detection Results

The emerging threat detection component identifies candidate topics exhibiting unusual or increasing activity.

The analysis combines:

* Novelty.
* Growth.
* Persistence.
* Activity.
* Term changes.

The combined score is represented as:

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

where the weights sum to one.

### 5.14.1 Emerging Threat Score Results

The final results should be presented using:

| Topic   | Period     | Novelty | Growth | Persistence | Activity | Term Change |   ETS |
| ------- | ---------- | ------: | -----: | ----------: | -------: | ----------: | ----: |
| Topic 1 | `[Period]` |   `[X]` |  `[X]` |       `[X]` |    `[X]` |       `[X]` | `[X]` |
| Topic 2 | `[Period]` |   `[X]` |  `[X]` |       `[X]` |    `[X]` |       `[X]` | `[X]` |
| Topic 3 | `[Period]` |   `[X]` |  `[X]` |       `[X]` |    `[X]` |       `[X]` | `[X]` |

The threshold used for candidate identification should also be reported.

### 5.14.2 Candidate Emerging Topics

Topics exceeding the experimentally selected threshold are considered candidate emerging-threat signals.

| Rank by Score | Topic       | Representative Terms | Period     |   ETS | Interpretation     |
| ------------: | ----------- | -------------------- | ---------- | ----: | ------------------ |
|             1 | Topic `[X]` | `[terms]`            | `[period]` | `[X]` | `[interpretation]` |
|             2 | Topic `[X]` | `[terms]`            | `[period]` | `[X]` | `[interpretation]` |
|             3 | Topic `[X]` | `[terms]`            | `[period]` | `[X]` | `[interpretation]` |

The table should be populated only after the actual experiments have been completed.

### 5.14.3 Emerging Threat Trend Visualization

The temporal behavior of candidate topics can be visualized using:

* Topic prevalence curves.
* Emerging score curves.
* Heatmaps.
* Term-frequency trends.
* Topic evolution diagrams.

These visualizations provide additional evidence for determining whether an increase represents a persistent pattern or a temporary fluctuation.

### 5.14.4 Interpretation of Emerging Signals

A high Emerging Threat Score indicates that the corresponding topic exhibits multiple characteristics associated with emerging activity.

However, the score should not be interpreted as proof of a confirmed new cyber threat, zero-day vulnerability, or active attack.

The proposed framework functions as an analytical prioritization mechanism. Candidate signals require validation using additional CTI sources, security advisories, vulnerability information, incident evidence, or expert analysis.

---

## 5.15 Real-Time Detection Analysis

The proposed framework is designed to support timely CTI analysis by separating data ingestion, classification, topic analysis, and emerging-threat scoring into modular components.

The practical detection time depends on:

* Dataset size.
* Document length.
* Feature dimensionality.
* Classification model.
* Topic modeling algorithm.
* Number of topics.
* Temporal aggregation interval.
* Hardware configuration.

### 5.15.1 Processing Time

The implementation can measure:

| Component                  | Processing Time |
| -------------------------- | --------------: |
| Data Loading               |      `[XX] sec` |
| Preprocessing              |      `[XX] sec` |
| Feature Extraction         |      `[XX] sec` |
| Classification             |      `[XX] sec` |
| Topic Modeling             |      `[XX] sec` |
| Temporal Analysis          |      `[XX] sec` |
| Emerging Score Calculation |      `[XX] sec` |
| **Total**                  |  **`[XX] sec`** |

These values should be obtained from the actual implementation.

### 5.15.2 Real-Time Capability

If the implementation processes incoming documents continuously or in short intervals, the framework may be evaluated for near-real-time operation.

If the experiments are performed only on a static historical dataset, the results should instead be described as **offline experimental evaluation**.

Therefore, the term "real-time" should be used only where the implementation and experimental measurements support the claim.

---

## 5.16 Error Analysis

Error analysis was performed to understand the limitations of the classification and emerging-threat detection components.

### 5.16.1 Classification Errors

Common sources of classification errors include:

* Similar terminology across threat categories.
* Multi-threat documents.
* Ambiguous descriptions.
* Insufficient training examples.
* Class imbalance.
* Generic cybersecurity terminology.
* Inconsistent labeling.

For example, a report describing credential theft through malicious email attachments may contain terminology associated with phishing, malware, and credential attacks simultaneously.

### 5.16.2 Topic Modeling Errors

Topic modeling may produce topics containing:

* Generic cybersecurity terms.
* Overlapping topics.
* Weakly interpretable topics.
* Highly correlated topics.
* Rare terms with limited context.

These issues can be influenced by preprocessing, vocabulary size, number of topics, and corpus characteristics.

### 5.16.3 Emerging Threat Detection Errors

Emerging-threat scoring may produce false positives when a topic experiences a temporary increase due to:

* A major security event.
* Publication of multiple reports about the same incident.
* Duplicate or highly similar reports.
* Changes in reporting frequency.
* Media attention.
* Sudden changes in dataset composition.

Consequently, temporal growth alone should not be interpreted as proof of an emerging cyber threat.

---

## 5.17 Discussion

The experimental results provide evidence for evaluating the major components of the proposed framework.

### 5.17.1 Classification Performance

The classification experiments determine whether conventional machine learning models can effectively categorize CTI documents using textual features.

Differences between models can be analyzed in terms of their ability to distinguish cybersecurity terminology and handle overlapping threat categories.

The final discussion should reference the actual accuracy, precision, recall, and F1-score values obtained from the experiments.

### 5.17.2 Topic Modeling Performance

The topic modeling results demonstrate whether meaningful latent cybersecurity themes can be extracted from the CTI corpus.

The combination of coherence scores and qualitative interpretation provides a more complete evaluation than using a single numerical measure.

### 5.17.3 Temporal Analysis

Temporal analysis extends topic modeling by identifying how threat-related themes change over time.

This is particularly relevant to CTI because threat activity and security discussions are dynamic rather than static.

A topic that remains stable across several periods may represent an established threat theme, whereas a newly appearing and rapidly increasing topic may warrant additional investigation.

### 5.17.4 Emerging Threat Detection

The emerging-threat component combines multiple signals rather than relying solely on topic frequency.

The inclusion of novelty, growth, persistence, activity, and term changes provides a multidimensional representation of emerging behavior.

Nevertheless, the scoring mechanism should be considered a prioritization method rather than an autonomous threat-confirmation system.

### 5.17.5 Integration of Components

The major advantage of the framework is the integration of supervised and unsupervised analysis.

Classification answers:

> **What predefined threat category does the document represent?**

Topic modeling answers:

> **What latent themes are present in the CTI corpus?**

Temporal analysis answers:

> **How are these themes changing over time?**

Emerging-threat scoring answers:

> **Which changing themes warrant further investigation?**

This combination provides complementary views of CTI information.

---

## 5.18 Limitations

Several limitations should be considered when interpreting the experimental results.

### 5.18.1 Dataset Dependency

The quality and diversity of the dataset directly affect classification and topic modeling performance.

A dataset dominated by particular threat categories or sources may not generalize to other CTI environments.

### 5.18.2 Label Quality

Supervised classification depends on the correctness of the available labels. Incorrect or inconsistent labels can reduce model performance.

### 5.18.3 Topic Interpretability

Topic modeling is inherently exploratory. Some automatically generated topics may require human interpretation and may not correspond to clearly defined cybersecurity concepts.

### 5.18.4 Temporal Coverage

Emerging-threat analysis requires sufficient temporal coverage. A short observation period may not provide enough information to distinguish persistent trends from temporary fluctuations.

### 5.18.5 Emerging Threat Validation

The framework identifies candidate emerging patterns rather than independently confirming new threats.

External evidence and expert validation remain necessary.

### 5.18.6 Model Generalization

Performance obtained on one CTI dataset should not automatically be generalized to other datasets, languages, sources, or operational environments.

### 5.18.7 Computational Limitations

Topic modeling and high-dimensional text processing can become computationally expensive as dataset size and vocabulary increase.

---

## 5.19 Summary of Experimental Findings

The experimental evaluation is designed to answer the following research questions:

| Research Question                                                        | Evaluation Component       |
| ------------------------------------------------------------------------ | -------------------------- |
| Can CTI documents be automatically classified?                           | Classification experiments |
| Which ML models perform effectively?                                     | Comparative model analysis |
| Can meaningful CTI topics be discovered?                                 | LDA/NMF topic modeling     |
| How do CTI topics change over time?                                      | Temporal topic analysis    |
| Can increasing or novel patterns be identified?                          | Emerging threat scoring    |
| Can multiple signals be combined into a useful prioritization mechanism? | Emerging Threat Score      |
| What are the major sources of error?                                     | Error analysis             |

The final experimental results should be interpreted in relation to the research objectives defined in Chapter 1.

The results demonstrate the practical behavior of the proposed framework across classification, topic discovery, temporal analysis, and emerging-threat detection. The findings also identify limitations that must be considered when applying the framework to larger or operational CTI environments.

## Chapter Summary

This chapter presented the experimental setup and analysis of the proposed CTI framework. The dataset was examined in terms of its size, class distribution, temporal coverage, and data quality. Multiple machine learning models were evaluated for threat classification using accuracy, precision, recall, F1-score, and confusion matrix analysis.

Topic modeling was subsequently evaluated using topic coherence and qualitative interpretation. Temporal topic analysis was used to identify changes in cybersecurity-related themes across time periods. The emerging-threat scoring mechanism combined novelty, growth, persistence, activity, and term-change signals to identify candidate emerging patterns.

The chapter also examined processing requirements, classification errors, topic-modeling limitations, temporal fluctuations, and the distinction between candidate emerging-threat signals and confirmed cybersecurity incidents.

The next chapter presents the overall conclusions of the research, major contributions, limitations, and possible directions for future work.
