# Chapter 3: Research Methodology

## 3.1 Introduction

Describe the experimental prototype and research workflow.

## 3.2 Research Design

The work combines supervised text classification and unsupervised topic
analysis.

## 3.3 Dataset Collection and Description

Document that the current dataset is synthetic. State record count,
fields, label distribution, and date range based on the final notebook
outputs.

## 3.4 Data Preprocessing

Explain text joining, missing-value handling, date parsing, label
cleanup, and leakage checks.

## 3.5 Feature Extraction

Explain TF-IDF for classification. For LDA use count-vector features;
for NMF use TF-IDF in the proposed implementation.

## 3.6 Classification Models

Describe Logistic Regression, Multinomial Naive Bayes, and Linear SVM
only if they were run.

## 3.7 Train/Test Protocol

Record random seed, test size, stratification, and any chronological or
group-based holdout. Explain why random splits may overestimate
performance on templated synthetic text.

## 3.8 Evaluation Metrics

Define accuracy, precision, recall, F1-score, and confusion matrix.

## 3.9 Topic Modeling

Describe LDA and NMF, number of topics, vectorization settings, random
seeds, and how topic words were inspected.

## 3.10 Temporal Analysis

Explain how report dates are mapped to weeks and how mean topic
distributions are computed.

## 3.11 Emerging-Pattern Alert Logic

State the exact baseline window and threshold used by the final code.
Explain that alerts are heuristic candidate signals.

## 3.12 Tools and Environment

Record actual OS, Python version, packages, CPU/RAM if required by
institution.

## 3.13 Ethical and Reproducibility Considerations

Discuss synthetic data, privacy, repeatability, and limits of
interpretation.

## 3.14 Chapter Summary

Summarize the reproducible pipeline.
