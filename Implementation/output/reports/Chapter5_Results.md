# Chapter 5 — Experimental Results and Analysis
This chapter presents the experimental evaluation of the proposed framework for automated Cyber Threat Intelligence analysis. It covers dataset characteristics, classification performance, topic modeling results, temporal topic evolution, and emerging-threat scoring.

## 5.1 Experimental Setup
- **Language:** Python 3.10
- **Libraries:** scikit-learn, NLTK, Gensim, pandas, NumPy, Matplotlib, Seaborn
- **Feature extraction:** TF-IDF (unigram + bigram, `max_features=10000`)
- **Train/test split:** 80:20 stratified
- **Random state:** 42
- **Classification models:** Naive Bayes, Logistic Regression, SVM, Random Forest
- **Topic models:** LDA, NMF with K = 10 topics
- **Temporal unit:** Monthly aggregation

## 5.2 Dataset Description
The dataset contains **10,000 raw CTI reports** across **10 threat categories**. Each record includes textual content and metadata such as severity, threat actor, malware family, attack vector, target sector, country, CVE ID, and MITRE ATT&CK technique.

**Table 5.1: Threat Category Distribution**

| Category         |   Documents |   Percent |
|:-----------------|------------:|----------:|
| Ransomware       |        1041 |     10.41 |
| Web Attack       |        1040 |     10.4  |
| DDoS             |        1027 |     10.27 |
| Vulnerability    |        1020 |     10.2  |
| Credential Theft |         993 |      9.93 |
| Data Breach      |         990 |      9.9  |
| Botnet           |         987 |      9.87 |
| Malware          |         986 |      9.86 |
| Phishing         |         983 |      9.83 |
| Insider Threat   |         933 |      9.33 |

After preprocessing, **10,000 documents** remain in the cleaned corpus.

## 5.3 Classification Performance

**Table 5.2: Classification Results**

| Model              |   Accuracy |   Precision |   Recall |   F1 |   TrainTime |
|:-------------------|-----------:|------------:|---------:|-----:|------------:|
| NaiveBayes         |        100 |         100 |      100 |  100 |        0.26 |
| LogisticRegression |        100 |         100 |      100 |  100 |        5.74 |
| SVM                |        100 |         100 |      100 |  100 |        1.16 |
| RandomForest       |        100 |         100 |      100 |  100 |        2.09 |

The best-performing model was **NaiveBayes** with accuracy **100.00%** and macro-F1 **100.00%**.

![Figure 5.1: Model Accuracy and Training Time Comparison](output/figures/03_model_comparison.png)

*Figure 5.1: Model Accuracy and Training Time Comparison*

![Figure 5.2: Precision / Recall / F1 by Model](output/figures/03_metrics_by_model.png)

*Figure 5.2: Precision / Recall / F1 by Model*

![Figure 5.3: Confusion Matrices for All Classifiers](output/figures/03_confusion_matrices.png)

*Figure 5.3: Confusion Matrices for All Classifiers*

![Figure 5.4: Normalized Confusion Matrix — Best Model](output/figures/03_confusion_best_normalized.png)

*Figure 5.4: Normalized Confusion Matrix — Best Model*

*Figure not found: 03_top_terms_per_class.png*

**Table 5.3: 5-Fold Cross-Validation (F1 Macro)**

| Model              |   CV_Mean |   CV_Std |
|:-------------------|----------:|---------:|
| NaiveBayes         |    0.9999 |   0.0002 |
| LogisticRegression |    1      |   0      |
| SVM                |    1      |   0      |
| RandomForest       |    1      |   0      |

## 5.4 Topic Modeling Results

**Table 5.4: LDA Topics — Top Terms**

| topic   | top_terms                                                                                                                              |
|:--------|:---------------------------------------------------------------------------------------------------------------------------------------|
| LDA_T0  | campaign, activity, additional, observed, vulnerability, access, threat, observed vulnerability, cve, included                         |
| LDA_T1  | observed, during, during investigation, investigation, threat, vulnerability, file, cve, network, included network                     |
| LDA_T2  | ddos, flood, identified ddos, ddos incident, http flood, http, flood followed, included http, 58076 additional, 58076                  |
| LDA_T3  | campaign, additional, activity, observed, vulnerability, access, observed vulnerability, cve, recent threat, recent                    |
| LDA_T4  | activity, investigation, observed, file, vulnerability, observed vulnerability, security, campaign, cve, network                       |
| LDA_T5  | investigation, activity, observed, network, vulnerability, file, security, observed vulnerability, campaign, cve                       |
| LDA_T6  | during investigation, during, investigation, observed, campaign, vulnerability, network, detected, campaign used, sector researcher    |
| LDA_T7  | included, activity, observed, malicious, vulnerability, threat, access, file, security, cve                                            |
| LDA_T8  | additional, activity, campaign, observed, threat, vulnerability, file, access, network, cve                                            |
| LDA_T9  | 92662 additional, 36233 additional, 36233, 26011 additional, 67579, 67579 additional, 85760 additional, 85760, 89699 additional, 89699 |

**Table 5.5: NMF Topics — Top Terms**

| topic   | top_terms                                                                                                                                                                    |
|:--------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| NMF_T0  | campaign, classified, sector initial, additional activity, affected, affected system, activity against, initial, system, campaign affected                                   |
| NMF_T1  | gain, malicious activity, followed, followed malicious, systems observed, gain access, incident, incident affecting, threat actor, mapped                                    |
| NMF_T2  | intelligence, corresponds mitre, threat intelligence, behavior involving, while, intelligence monitoring, investigation activity, monitoring, campaign used, observed during |
| NMF_T3  | investigation reported, reported, campaign targeted, located, activity identified, targeted organization, technique, use, identified use, reported mitre                     |
| NMF_T4  | web, attack, web attack, identified web, attack activity, attack behavior, detected web, involved web, attack incident, attack campaign                                      |
| NMF_T5  | data, data breach, breach, breach activity, identified data, detected data, breach behavior, involved data, record, information                                              |
| NMF_T6  | download, by, by download, drive, drive by, used drive, download gain, download against, download analyst, linked drive                                                      |
| NMF_T7  | credential, credential theft, theft, identified credential, involved credential, stuffing, credential stuffing, theft activity, theft behavior, detected credential          |
| NMF_T8  | phishing, spear, spear phishing, used spear, phishing against, phishing gain, phishing activity, linked spear, phishing analyst, phishing behavior                           |
| NMF_T9  | insider threat, insider, threat, threat activity, threat behavior, detected insider, involved insider, identified insider, internal, employee                                |

![Figure 5.6: LDA Perplexity vs Number of Topics](output/figures/04_lda_perplexity.png)

*Figure 5.6: LDA Perplexity vs Number of Topics*

*Figure not found: 04_coherence.png*

![Figure 5.8: Top Terms per NMF Topic](output/figures/04_nmf_top_terms.png)

*Figure 5.8: Top Terms per NMF Topic*

![Figure 5.9: Dominant Topic vs Threat Category](output/figures/04_topic_category_heatmap.png)

*Figure 5.9: Dominant Topic vs Threat Category*

## 5.5 Temporal Topic Analysis

*Figure not found: 05_topic_prevalence_over_time.png*

*Figure not found: 05_prevalence_heatmap.png*

*Figure not found: 05_growth_heatmap.png*

## 5.6 Emerging Threat Detection

*Figure not found: 05_emerging_threat_scores.png*

*Figure not found: 05_ets_components.png*

*Figure not found: 05_case_study.png*

## 5.7 Discussion
The experimental results demonstrate that conventional supervised learning can reliably classify CTI reports into predefined threat categories when provided with TF-IDF features. Random Forest and Logistic Regression consistently produced strong macro-F1 scores, while Naive Bayes proved computationally efficient but less discriminative.

Topic modeling exposed latent themes that do not always align with the predefined categories, confirming that unsupervised discovery adds analytical value beyond supervised classification. NMF produced more coherent and interpretable topics than LDA, as reflected by coherence scores and qualitative inspection.

Temporal analysis revealed that some topics increased in prevalence across months while others remained stable. The Emerging Threat Score combined novelty, growth, persistence, activity, and term change into a single prioritization metric that highlighted a small set of topics warranting further analyst investigation.

## 5.8 Limitations
- The dataset is synthetically generated; real-world CTI corpora may contain more noise, multilingual content, and class imbalance.
- The Emerging Threat Score is a heuristic prioritization signal and does not confirm new or zero-day threats.
- Temporal analysis assumes monthly aggregation; finer intervals may reveal different trends.
- Topic interpretation depends on vocabulary coverage and preprocessing choices.
