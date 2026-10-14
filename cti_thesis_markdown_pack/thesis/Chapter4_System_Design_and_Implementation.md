# Chapter 4: System Design and Implementation

## 4.1 Introduction

Describe the system modules and their responsibilities.

## 4.2 Proposed Architecture

``` text
Synthetic CTI Reports
        |
        v
Data Validation and Preprocessing
        |
        +-------------------------+
        |                         |
        v                         v
TF-IDF Classification       Topic Modeling
        |                    (LDA / NMF)
        v                         |
Predicted Category                v
                           Weekly Topic Shares
                                   |
                                   v
                         Candidate Change Alerts
                                   |
                                   v
                         Analyst Review / Output
```

## 4.3 Dataset Loading

Describe file path configuration, required columns, and schema
validation.

## 4.4 Preprocessing Module

Describe joining title and body, removing unusable rows, parsing dates,
and checking labels.

## 4.5 Classification Module

Describe the scikit-learn Pipeline, TF-IDF, candidate classifiers, and
saved pipeline.

## 4.6 Topic Modeling Module

Describe LDA count features and NMF TF-IDF features, topic count,
top-word extraction, and manual interpretation.

## 4.7 Temporal Analysis Module

Describe weekly grouping and average topic weights.

## 4.8 Emerging-Pattern Module

Describe the exact change threshold and how a candidate alert is
recorded. Do not call it a confirmed threat.

## 4.9 Artifact and Output Management

List saved files that actually exist, such as model joblib, confusion
matrix, weekly trend plots, and candidate alert CSV.

## 4.10 Example Workflow

Walk through one sample report, from input to predicted category and
topic analysis. Mark it as an illustrative example if synthetic.

## 4.11 Implementation Challenges

Document real errors encountered and their fixes.

## 4.12 Chapter Summary

Summarize implemented modules and identify anything not implemented.
