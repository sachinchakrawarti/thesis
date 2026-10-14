# Base Paper Review

## Paper identity

**Title:** Real-Time Automated Cyber Threat Classification and Emerging
Threat Detection Framework\
**Authors:** Alemayehu Tilahun Haile, Surafel Lemma Abebe, Henock
Mulugeta Melaku\
**Publication:** IEEE Open Journal of the Computer Society, 2025\
**Source:** User-provided PDF `11037544_copy(3).pdf`. Verify the
complete bibliographic metadata, DOI, volume, issue, and pages from the
PDF title page before final submission.

## Research problem

The paper addresses automated classification of cyber-threat reports and
detection of potentially emerging threat patterns from CTI text.

## Methods described in the provided paper

-   Text representation approaches discussed include Bag-of-Words,
    TF-IDF, and embedding-based representations.
-   Classification experiments include machine-learning and
    deep-learning approaches.
-   The paper discusses threat classes including DDoS, botnets, malware,
    phishing, spam, and ransomware.
-   LDA and NMF topic modeling are used to inspect changes in CTI topic
    distributions over time.
-   Topic distributions are plotted weekly in the described temporal
    analysis.

## Findings to report carefully

The paper reports SVM with binary Bag-of-Words features as a strong
classifier in its own experimental setting, with 93.67% binary accuracy
and 96.35% multiclass accuracy. These figures belong to the paper's
dataset and experiment; they are not results of the present project.

The paper's discussion indicates that LDA topic proportions can be more
responsive to changing topic prevalence, while NMF provides a
comparatively stable topic view in the reported analysis. The authors
caution that topic changes do not establish causation and should be
checked against incident data.

## Strengths

-   Combines classification and temporal topic analysis.
-   Uses more than one topic-modeling approach.
-   Presents topic trends over time rather than only static topic lists.
-   Recognizes the need to validate interpretations with incident
    evidence.

## Limitations / points to investigate

-   Confirm the dataset construction and representativeness from the
    full paper.
-   Confirm exactly how emerging topics and alerts are operationally
    defined.
-   Determine whether reported trends are validated against independent
    incident labels.
-   Distinguish offline experiments from a truly live streaming
    deployment.

## Relationship to this project

  -----------------------------------------------------------------------
  Base-paper element                  Current project status
  ----------------------------------- -----------------------------------
  TF-IDF text classification          Planned/implemented in notebook;
                                      verify final run

  SVM baseline                        Included as LinearSVC candidate

  LDA topic modeling                  Included in proposed implementation
                                      documentation; run and save
                                      evidence

  NMF topic modeling                  Included in proposed implementation
                                      documentation; run and save
                                      evidence

  Weekly topic trends                 Included in proposed implementation
                                      documentation; verify date
                                      distribution

  Emerging-pattern alerts             Heuristic extension; must be
                                      described as candidate alerts

  Real-world validation               Not established by synthetic data
  -----------------------------------------------------------------------

## Summary

The paper is the methodological starting point. The present work should
be described as a prototype implementation using synthetic CTI-style
reports, not as a reproduction of the paper's experimental results.
