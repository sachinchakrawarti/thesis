# Chapter 7 – Base Papers

| **Title**                                                                                   | **Authors**                        | **Year** | **Summary**                                                                                                                                                                                                                                                                  |
| ------------------------------------------------------------------------------------------- | ---------------------------------- | -------: | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Real-Time Automated Cyber Threat Classification and Emerging Threat Detection Framework** | A. T. Haile et al.                 |     2025 | Proposes an automated CTI framework combining machine learning and deep learning for cyber threat classification and LDA/NMF-based topic modeling for identifying emerging threat patterns over time. This paper serves as the primary base paper for the proposed research. |
| **Cyber Threat Intelligence: An Overview of Current State and Future Directions**           | Multiple Authors                   |   Recent | Reviews the concept, lifecycle, sources, and applications of Cyber Threat Intelligence, providing a foundation for automated CTI collection, analysis, and dissemination.                                                                                                    |
| **Machine Learning for Cybersecurity: A Comprehensive Survey**                              | Multiple Authors                   |   Recent | Reviews machine learning applications in cybersecurity, including intrusion detection, malware analysis, phishing detection, threat classification, and anomaly detection.                                                                                                   |
| **Natural Language Processing for Cybersecurity: A Survey**                                 | Multiple Authors                   |   Recent | Examines NLP techniques for processing unstructured cybersecurity text, including tokenization, feature extraction, entity recognition, text classification, and threat intelligence analysis.                                                                               |
| **Cyber Threat Intelligence Mining from Unstructured Text Using Machine Learning**          | Multiple Authors                   |   Recent | Investigates the use of machine learning and NLP techniques to extract useful cybersecurity information from unstructured threat reports and textual intelligence.                                                                                                           |
| **Automated Cyber Threat Classification Using Machine Learning Techniques**                 | Multiple Authors                   |   Recent | Evaluates supervised machine learning algorithms for automatically categorizing cyber threats into predefined classes using textual and behavioral features.                                                                                                                 |
| **Latent Dirichlet Allocation**                                                             | D. M. Blei, A. Y. Ng, M. I. Jordan |     2003 | Introduces Latent Dirichlet Allocation (LDA), a probabilistic topic modeling approach for discovering latent thematic structures in document collections. It provides the theoretical foundation for the topic modeling component of this research.                          |
| **Learning the Parts of Objects by Non-Negative Matrix Factorization**                      | D. D. Lee, H. S. Seung             |     1999 | Introduces Non-Negative Matrix Factorization (NMF), which decomposes non-negative data into interpretable components. NMF provides an alternative topic modeling approach for discovering latent patterns in CTI documents.                                                  |
| **Dynamic Topic Models**                                                                    | D. M. Blei, J. D. Lafferty         |     2006 | Introduces dynamic topic modeling for studying how topics evolve over time. The work provides a theoretical basis for analyzing temporal changes in CTI-related topics.                                                                                                      |
| **Online Learning for Latent Dirichlet Allocation**                                         | M. D. Hoffman, D. M. Blei, F. Bach |     2010 | Presents an efficient online approach for learning topic models from large document collections, supporting scalable processing of continuously arriving textual data.                                                                                                       |
| **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding**        | J. Devlin et al.                   |     2019 | Introduces BERT, a transformer-based language representation model that captures contextual relationships between words. Transformer representations provide a potential future direction for contextual CTI analysis.                                                       |
| **Cybersecurity Named Entity Recognition Using Deep Learning**                              | Multiple Authors                   |   Recent | Investigates deep learning and NLP methods for identifying cybersecurity entities such as malware, vulnerabilities, threat actors, organizations, and attack techniques from textual data.                                                                                   |
| **Deep Learning for Cybersecurity: A Survey**                                               | Multiple Authors                   |   Recent | Reviews CNN, RNN, LSTM, transformer, and other deep learning approaches used in cybersecurity applications, including threat detection and security text analysis.                                                                                                           |
| **Threat Intelligence Extraction from Security Reports Using Natural Language Processing**  | Multiple Authors                   |   Recent | Explores automated extraction of cybersecurity entities, indicators, and threat information from unstructured security reports using NLP techniques.                                                                                                                         |
| **Emerging Threat Detection Using Topic Modeling and Temporal Analysis**                    | Multiple Authors                   |   Recent | Investigates changes in topic prevalence over time as a mechanism for identifying unusual or emerging cybersecurity patterns. The approach is closely related to the temporal analysis component of the proposed framework.                                                  |
| **Explainable Artificial Intelligence for Cybersecurity**                                   | Multiple Authors                   |   Recent | Reviews explainability techniques for cybersecurity machine learning models and emphasizes the importance of interpretable predictions for analyst trust and operational use.                                                                                                |
| **A Survey of Cyber Threat Intelligence Platforms and Techniques**                          | Multiple Authors                   |   Recent | Reviews CTI platforms, data sources, collection methods, analysis techniques, and intelligence-sharing mechanisms used in modern cybersecurity environments.                                                                                                                 |
| **Transformer-Based Cyber Threat Intelligence Analysis**                                    | Multiple Authors                   |   Recent | Investigates transformer-based language representations for analyzing cybersecurity text and extracting contextual threat information, providing a potential extension to traditional TF-IDF-based approaches.                                                               |
| **Temporal Analysis of Cybersecurity Threat Reports**                                       | Multiple Authors                   |   Recent | Studies temporal patterns in cybersecurity reports and demonstrates how changes in terminology, frequency, and topic prevalence can support monitoring of evolving threat activity.                                                                                          |

## 7.1 Primary Base Paper

Among the reviewed studies, the primary base paper for this research is:

**A. T. Haile et al., “Real-Time Automated Cyber Threat Classification and Emerging Threat Detection Framework,” 2025.**

The paper is directly related to the proposed research because it combines automated cyber threat classification with topic modeling and emerging threat analysis.

The proposed thesis extends this research direction by emphasizing:

* Machine learning-based CTI classification.
* LDA/NMF-based topic discovery.
* Explicit temporal topic analysis.
* Topic growth and persistence analysis.
* Multi-factor emerging threat scoring.
* Integrated experimental evaluation.

## 7.2 Relevance of the Base Papers

The selected papers provide theoretical and methodological support for the major components of the proposed framework.

| **Research Component**    | **Supporting Literature**           |
| ------------------------- | ----------------------------------- |
| Cyber Threat Intelligence | CTI surveys and frameworks          |
| CTI Text Processing       | NLP for cybersecurity               |
| Feature Extraction        | TF-IDF, N-grams, embeddings         |
| Threat Classification     | Machine learning for cybersecurity  |
| Deep Learning             | CNN, RNN, LSTM, Transformer studies |
| Topic Modeling            | LDA and NMF                         |
| Temporal Analysis         | Dynamic and temporal topic modeling |
| Emerging Threat Detection | Temporal threat analysis            |
| Explainability            | XAI for cybersecurity               |
| Future Enhancement        | Transformer-based CTI analysis      |

## 7.3 Selection of the Primary Base Paper

The primary base paper was selected because its research problem closely corresponds to the objective of the present thesis. It investigates automated cyber threat classification and emerging threat detection using machine learning and topic modeling.

The proposed research retains this general direction while developing a framework focused on the temporal evolution of CTI topics and quantitative prioritization of candidate emerging threat patterns.

## 7.4 Research Gap Identified from the Literature

The reviewed literature indicates that existing research has investigated cyber threat classification, NLP-based CTI processing, topic modeling, and temporal analysis as individual or partially integrated tasks.

The following research opportunities were identified:

1. Integration of supervised threat classification and temporal topic analysis.
2. Explicit measurement of topic growth and persistence.
3. Identification of candidate emerging topics using multiple signals.
4. Quantitative emerging threat scoring.
5. Integration of classification and emerging-threat analysis within a single framework.
6. Evaluation of the complete pipeline using common experimental metrics.
7. Improved interpretability of emerging threat signals.

The proposed research addresses these areas through the development of an integrated CTI analysis framework.

## 7.5 Summary

The reviewed base papers establish the theoretical foundation for the proposed research in Cyber Threat Intelligence, Natural Language Processing, machine learning, topic modeling, temporal analysis, and emerging threat detection.

The paper **“Real-Time Automated Cyber Threat Classification and Emerging Threat Detection Framework”** serves as the primary base paper, while the remaining literature provides supporting concepts and methodologies.

The proposed research builds upon these studies by integrating threat classification, topic modeling, temporal analysis, and emerging threat scoring into a unified CTI analysis framework.
