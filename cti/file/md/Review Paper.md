# Cyber Threat Classification and Emerging Threat Detection Using Sentence-BERT and BERTopic: A Review

**Author Name**  
Department of Data Science, [Your University]  
[email]

---

## Abstract

Cyber threat intelligence (CTI) is essential for modern cybersecurity operations, but its volume and unstructured nature make manual analysis infeasible. This review surveys the state of the art in automated CTI analysis, focusing on two tasks central to the present thesis: (1) **threat classification** using lexical and semantic representations, and (2) **topic-based emerging-threat detection** using neural topic modeling. We review the SMAH-CTI dataset—a sentence-level CTI corpus derived from 100 public reports—and the two classification approaches we compare: TF-IDF + Support Vector Machine (baseline) and Sentence-BERT + SVM (proposed). We then review BERTopic as an alternative to latent Dirichlet allocation (LDA) and non-negative matrix factorization (NMF) for discovering semantic topics in CTI text. We identify four research gaps: data leakage in prior sentence-level splits, the unexplored comparison of TF-IDF, Sentence-BERT, and hybrid representations on CTI, the lack of a controlled evaluation of BERTopic versus LDA/NMF on short CTI sentences, and the absence of quantitative end-to-end evaluations of CTI analyzers. By positioning our thesis against this backdrop, we clarify the specific contributions of this work: a document-level comparison of lexical and semantic classifiers, a coherent topic model of CTI text, and a quantitative analyzer that classifies reports and detects emerging threats.

**Keywords:** Cyber Threat Intelligence, Sentence-BERT, BERTopic, SMAH-CTI, threat classification, topic modeling, emerging threats.

---

## 1. Introduction

Cyber threat intelligence (CTI) is the systematic collection, processing, and dissemination of information about current and emerging cyber threats [1]. Organizations rely on CTI to prioritize patching, hunt for indicators of compromise, and anticipate adversary behavior. CTI sources are heterogeneous—vendor reports (CrowdStrike, Mandiant, Microsoft), community analyses (APTNotes, MITRE), and open-source datasets—and each source differs in style, length, and technical density [2].

Manual CTI analysis is slow. A CTI analyst can review perhaps a few hundred reports per week, while public and commercial feeds produce thousands [3]. This gap motivates automated CTI analysis, and in particular two tasks:

1. **Threat classification:** identify whether a document (or sentence) is threat-relevant and, if so, which threat category it represents.
2. **Emerging-threat detection:** identify patterns that are new, evolving, or increasing in frequency, before they become widespread attacks.

Both tasks are active research areas, but they are usually studied separately. Few frameworks integrate classification and emergence detection and evaluate both on the same corpus.

This review is scoped to the specific methodology we adopt in our thesis. We review:

- **Section 2:** Threat classification methods—lexical (TF-IDF + SVM) vs. semantic (Sentence-BERT + SVM).
- **Section 3:** Topic modeling for CTI—LDA, NMF, and BERTopic.
- **Section 4:** The SMAH-CTI dataset and its role in our study.
- **Section 5:** Research gaps and directions.

The review positions our thesis as a controlled comparison of these representations and topic models on a document-level split of SMAH-CTI.

---

## 2. Threat Classification

### 2.1 Task Definition

Threat classification assigns CTI text to a category such as *relevant* / *irrelevant* (binary) or a specific threat class such as ransomware, phishing, malware, DDoS, botnet, or spam (multiclass). The task supports analyst triage: relevant documents are routed to the appropriate team, and irrelevant documents (general news, business reports, marketing text) are filtered out.

### 2.2 Lexical Classification: TF-IDF + SVM

The TF-IDF representation converts each document into a sparse vector where each dimension corresponds to a term, weighted by:

- **Term frequency (TF):** how often the term appears in the document.
- **Inverse document frequency (IDF):** how rare the term is across the corpus.

Support Vector Machines (SVM) [4] find a hyperplane that maximizes the margin between classes. The linear kernel is well suited to high-dimensional sparse TF-IDF vectors and has a long track record in text classification.

Yang and Lam [5] use TF-IDF with document embeddings and six ML classifiers for CTI report categorization. Haile et al. [6] use TF-IDF + SVM as their baseline for real-time CTI classification, reporting 93.67% accuracy on a leaked forum dataset. Irshad and Siddiqui [1] survey TF-IDF methods for CTI attribution.

**Advantages:** TF-IDF is fast, interpretable, and effective on technical text where vocabulary is discriminative (CVE IDs, malware names, IOC indicators).

**Limitations:** TF-IDF is bag-of-words and misses semantic similarity. "Credential theft" and "password harvesting" are treated as unrelated terms despite sharing meaning.

### 2.3 Semantic Classification: Sentence-BERT + SVM

Sentence-BERT (SBERT) [7] fine-tunes the BERT encoder [8] so that semantically similar sentences produce similar embeddings. Each sentence is encoded into a dense vector (typically 384 or 768 dimensions), and a classifier (SVM, logistic regression, or neural) is trained on these embeddings.

SBERT's `all-MiniLM-L6-v2` is a lightweight variant (22M parameters, 384-dim) that runs efficiently on CPU. It has become a standard choice for sentence-level semantic tasks in resource-constrained settings.

Recent applications of semantic models to CTI include:

- **Dionisio et al. [9]** apply deep neural networks to Twitter CTI classification.
- **Kadoguchi et al. [10]** use Doc2Vec embeddings for dark web forum classification.
- **Silvestri et al. [11]** apply NLP-based analysis to healthcare threat data.

**Advantages:** SBERT captures semantic similarity and generalizes to paraphrased descriptions.

**Limitations:** Pre-trained general-domain models may not encode CTI-specific entities (malware families, vulnerable products) as strongly as common English vocabulary. On short sentences (~20 words), semantic context is limited, so SBERT may not outperform lexical methods.

### 2.4 The Representation Gap

Few studies directly compare TF-IDF, SBERT, and hybrid TF-IDF+SBERT on **the same CTI corpus with a document-level split**. Existing comparisons often use different datasets, different splits, or different evaluation metrics, making cross-study conclusions unreliable.

**This gap is the focus of our thesis:** a controlled comparison on SMAH-CTI with a document-level 80/20 split.

### 2.5 Evaluation Metrics

Classification is evaluated with accuracy, precision, recall, and F1-score [12]. For imbalanced corpora, **macro-F1** is preferred because it weights classes equally—exposing poor performance on the minority class rather than hiding it behind high accuracy. We report all four metrics and a confusion matrix for both TF-IDF and SBERT classifiers.

---

## 3. Topic Modeling for CTI

### 3.1 Task Definition

Topic modeling discovers latent themes in a corpus without supervision. In CTI, it supports:

- Summarizing large threat collections into a small set of themes.
- Discovering campaign-level topics (e.g., DPRK operations, ransomware families, ICS attacks).
- Tracking topic prevalence over time to signal emerging threats.

### 3.2 Latent Dirichlet Allocation (LDA)

LDA [13] assumes each document is a mixture of topics, and each topic is a distribution over words. It has been the dominant topic model in CTI for over a decade:

- Hacker forum monitoring [14]
- Dark web market analysis [15]
- Social media threat detection [16]

**Limitations:** LDA is bag-of-words. On short CTI sentences, it produces noisy topics dominated by common function words ("the", "to", "and") rather than discriminative threat terms.

### 3.3 Non-negative Matrix Factorization (NMF)

NMF [17] decomposes the document-term matrix into non-negative topic-word and document-topic matrices. Compared with LDA, NMF yields more stable topics but is less sensitive to changes in topic prevalence over time.

**Limitations:** Like LDA, NMF is bag-of-words and does not capture semantic similarity between paraphrased expressions.

### 3.4 BERTopic

BERTopic [18] combines:

1. **Sentence embeddings** (SBERT) to represent each document semantically.
2. **UMAP dimensionality reduction** [19] to project embeddings into a lower-dimensional space.
3. **HDBSCAN clustering** [20] to group similar documents.
4. **c-TF-IDF representation** to derive topic-word distributions from clusters.

Since 2022, BERTopic has been used in scientific literature review, social media analysis, and customer feedback clustering. Its application to CTI is emerging:

- **Suryotrisongko et al. [21]** compare BERTopic and Top2Vec for CTI modeling, showing both outperform LDA on coherence metrics.
- **Wang et al. [16]** apply topic modeling to Twitter data with a Log4Shell case study.

**Advantages:** BERTopic produces semantically coherent topics grounded in meaning rather than word frequency. Topics are more interpretable by analysts.

**Limitations:** On short sentences, BERTopic assigns many documents to an outlier topic (`-1`), reducing coverage. The coherence and diversity of BERTopic topics depend heavily on corpus size and sentence length.

### 3.5 Emerging-Threat Signals from Topic Prevalence

If publication dates are available, topic prevalence can be tracked over time. A sudden spike in a topic's proportion of the corpus may signal an emerging threat. This approach was used by Haile et al. [6] (with LDA/NMF) and by Vishwanathan et al. [22] for CTI time series analysis.

**Open question:** Does BERTopic produce clearer emerging-threat signals than LDA/NMF? This is not yet established empirically.

### 3.6 Evaluation Metrics

Topic models are evaluated with:

- **Coherence (c_v):** semantic similarity of top words within a topic. Higher is better.
- **Diversity:** fraction of unique words across topics. Higher is better.
- **Outlier rate:** fraction of documents assigned to `-1`.

For emerging-threat detection, additional metrics apply:

- **Lead time:** how early a threat is detected before public disclosure.
- **Precision/recall of emergence flags** versus a labeled emergence set.

---

## 4. The SMAH-CTI Dataset

### 4.1 Overview

The Sentence-level Multi-Aspect Hierarchical Cyber Threat Intelligence (SMAH-CTI) dataset [23] contains 12,304 sentence instances derived from 100 CTI reports. Sources include APTNotes, CrowdStrike, Mandiant, MITRE, and Microsoft.

Each record contains:

- `sentence_text`: the sentence content.
- `relevant`: binary label (relevant / not relevant).
- `doc_id`, `page_number`, `sentence_id`: metadata.
- `tactic`, `techniques`, `procedure`, `entities`: annotations (relevant-only).

### 4.2 Descriptive Statistics

| Attribute | Value |
|-----------|-------|
| Total sentences | 12,304 |
| Relevant sentences | 4,302 |
| Non-relevant sentences | 8,002 |
| Imbalance ratio | 1.86:1 |
| Documents | 100 |
| Sources | 5 |
| Mean sentence length | 21.7 words |
| Vocabulary size | 13,292 |
| Type-token ratio | 0.06 |

### 4.3 Relevance to Our Thesis

SMAH-CTI is well-suited to our study because:

1. **Sentence-level labels** support binary classification directly.
2. **Document identifiers** allow document-level splitting (no leakage).
3. **Multiple sources** provide diversity in style and content.
4. **Public availability** ensures reproducibility.

**Limitation:** SMAH-CTI does not include publication dates for all reports. Temporal analysis (emerging-threat tracking over time) requires recovering dates from original reports or external metadata (e.g., APTNotes CSV).

---

## 5. Research Gaps and Directions

### 5.1 Data Leakage in Prior Work

Many CTI classification studies use sentence-level random splits, which leak sentences from the same report into both training and test sets. This inflates reported accuracy. **Our thesis adopts a document-level split** and expects lower (but more honest) accuracy than prior work.

### 5.2 Representation Choice on CTI Text

The interaction between representation choice (TF-IDF, SBERT, hybrid) and corpus characteristics (sentence length, technical vocabulary density) is not well understood. **Our thesis compares TF-IDF+SVM and SBERT+SVM on SMAH-CTI** and reports macro-F1 alongside accuracy.

### 5.3 BERTopic vs. LDA/NMF on Short CTI Sentences

BERTopic's advantage over LDA/NMF on short sentences is unclear. Prior work has not benchmarked the three on the same short-sentence corpus. **Our thesis evaluates BERTopic on 3,107 relevant sentences from SMAH-CTI** and reports coherence, diversity, and outlier rate.

### 5.4 End-to-End Evaluation

Most CTI frameworks report classification accuracy but not end-to-end performance (filtering + classification + emergence detection). **Our thesis implements an end-to-end analyzer** that ingests CTI reports and returns a three-class verdict (known / emerging / not relevant) with an attack-type label, and evaluates it on 20 test reports.

### 5.5 Domain-Adapted Embeddings

General-purpose SBERT may underperform on CTI-specific entities. Future work should compare SBERT with domain-adapted models (SecBERT, CyBERT) or fine-tuned small language models.

### 5.6 Temporal Validation

Emerging-threat detection lacks external ground truth. Future work should link topic prevalence to CVE publication dates or incident reports.

---

## 6. Conclusion

This review surveyed the state of the art in automated CTI analysis, focusing on the tasks and methods used in our thesis: threat classification (TF-IDF+SVM vs. Sentence-BERT+SVM) and topic modeling for emerging-threat detection (LDA, NMF, BERTopic) on the SMAH-CTI dataset. We identified four specific gaps: (1) data leakage from sentence-level splits, (2) the unexplored comparison of TF-IDF and SBERT on CTI with a document-level split, (3) the lack of a controlled BERTopic vs. LDA/NMF evaluation on short CTI sentences, and (4) the absence of end-to-end analyzer evaluations.

Our thesis addresses these gaps by:

1. Using a document-level split of SMAH-CTI.
2. Comparing TF-IDF+SVM, SBERT+SVM, and hybrid TF-IDF+SBERT on binary and multiclass classification.
3. Evaluating BERTopic on the 3,107 relevant sentences for topic coherence, diversity, and interpretability.
4. Building and evaluating an end-to-end analyzer that classifies reports and flags emerging threats.

By choosing SMAH-CTI as a common dataset and document-level splitting as a common protocol, our thesis provides a fair comparison of lexical and semantic representations for CTI classification, and a controlled study of BERTopic for CTI topic modeling. We expect this work to serve as a reference point for future research on semantic CTI analysis.

---

## References

[1] E. Irshad and A. B. Siddiqui, "Cyber threat attribution using unstructured reports in cyber threat intelligence," *Egyptian Informatics Journal*, vol. 24, no. 1, pp. 43–59, 2023.

[2] N. Goel, A. Mansi, and N. Sethi, "Cyber threat intelligence: A survey on progressive techniques and challenges," in *Proc. Int. Conf. Big Data IoT Cyber Sect. Inf. Technol.*, 2022, pp. 37–41.

[3] M. A. Manjramkar and K. C. Jondhale, "Cyber security using machine learning techniques," in *Proc. Int. Conf. Appl. Mach. Intell. Data Analytics*, 2023, pp. 680–701.

[4] V. K. Chauhan, K. Dahiya, and A. Sharma, "Problem formulations and solvers in linear SVM: A review," *Artificial Intelligence Review*, vol. 52, pp. 803–855, 2019.

[5] W. Yang and K.-Y. Lam, "Automated cyber threat intelligence reports classification for early warning of cyber attacks in next generation SOC," in *Proc. Inf. Commun. Secur.*, 2020, pp. 145–164.

[6] A. T. Haile, S. L. Abebe, and H. M. Melaku, "Real-Time Automated Cyber Threat Classification and Emerging Threat Detection Framework," *IEEE Access*, 2024.

[7] N. Reimers and I. Gurevych, "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks," in *Proc. EMNLP-IJCNLP*, 2019, pp. 3982–3992.

[8] J. Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding," in *Proc. NAACL-HLT*, 2019, pp. 4171–4186.

[9] N. Dionisio, F. Alves, P. M. Ferreira, and A. Bessani, "Cyberthreat detection from Twitter using deep neural networks," in *Proc. IJCNN*, 2019.

[10] T. Kadoguchi et al., "Exploring topic models to discern cyber threats on Twitter," *J. Intell. Syst. Appl.*, 2020.

[11] S. Silvestri et al., "A machine learning approach for the NLP-based analysis of cyber threats and vulnerabilities of the healthcare ecosystem," *Sensors*, vol. 23, no. 2, 2023.

[12] M. Sokolova and G. Lapalme, "A systematic analysis of performance measures for classification tasks," *Inf. Process. Manag.*, vol. 45, no. 4, pp. 427–437, 2009.

[13] D. M. Blei, A. Y. Ng, and M. I. Jordan, "Latent Dirichlet allocation," *J. Mach. Learn. Res.*, vol. 3, pp. 993–1022, 2003.

[14] A. S. Gautam, Y. Gahlot, and P. Kamat, "Hacker forum exploit and classification for proactive cyber threat intelligence," in *Proc. Inventive Computation Technol.*, 2020, pp. 279–285.

[15] R. Basheer and B. Alkhatib, "Threats from the dark: A review over dark web investigation research for cyber threat intelligence," *J. Comput. Netw. Commun.*, 2021.

[16] Y. Wang et al., "Exploring topic models to discern cyber threats on Twitter: A case study on Log4Shell," *Intell. Syst. Appl.*, vol. 20, 2023.

[17] D. D. Lee and H. S. Seung, "Learning the parts of objects by non-negative matrix factorization," *Nature*, vol. 401, pp. 788–791, 1999.

[18] M. Grootendorst, "BERTopic: Neural topic modeling with a class-based TF-IDF procedure," *arXiv:2203.05794*, 2022.

[19] L. McInnes, J. Healy, and J. Melville, "UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction," *arXiv:1802.03426*, 2018.

[20] R. J. G. B. Campello, D. Moulavi, and J. Sander, "Density-Based Clustering Based on Hierarchical Density Estimates," in *Proc. PAKDD*, 2013, pp. 160–172.

[21] H. Suryotrisongko et al., "Topic modeling for cyber threat intelligence (CTI)," in *Proc. Int. Conf. Inform. Comput.*, 2022, pp. 1–7.

[22] S. Vishwanathan et al., "A machine learning approach for the NLP-based analysis of cyber threats," *Sensors*, 2023.

[23] D. Mohaise et al., "SMAH-CTI: Sentence-level Multi-Aspect Hierarchical Cyber Threat Intelligence Dataset," *GitHub*, 2023. [Online]. Available: https://github.com/davidmohaisen/SMAH-CTI

[24] D. T. K. Nguyen et al., "A survey on cyber threat intelligence: Techniques, challenges, and future directions," *IEEE Access*, vol. 9, 2021.

[25] I. Deliu, C. Leichter, and K. Franke, "Collecting cyber threat intelligence from hacker forums via a two-stage, hybrid process using support vector machines and latent Dirichlet allocation," in *Proc. IEEE Big Data*, 2018, pp. 5008–5013.

---

*Submitted for review — [Date]*