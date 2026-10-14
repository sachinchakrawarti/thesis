# Chapter 1: Introduction

## 1.1 Background

Cybersecurity teams receive reports about malware, phishing, ransomware,
botnets, DDoS activity, vulnerabilities, and attacker behavior. Manually
reviewing large volumes of text can be slow and inconsistent. NLP and
machine learning can help organize reports, while temporal analysis can
help analysts notice changing themes.

## 1.2 Cyber Threat Intelligence

Define CTI and explain strategic, operational, tactical, and technical
intelligence using verified sources. Explain that the project focuses on
text reports and category/topic analysis.

## 1.3 Problem Statement

CTI-style reports may be difficult to categorize consistently, and
static topic summaries do not show how themes change over time. This
project prototypes automated category prediction and weekly topic-trend
analysis.

## 1.4 Research Motivation

Explain the need to reduce repetitive manual triage and help prioritize
reports for review. Avoid claiming the system replaces analysts.

## 1.5 Objectives

1.  Prepare and inspect a CTI-style report dataset.
2.  Build TF-IDF-based threat-category classifiers.
3.  Evaluate models using precision, recall, F1-score, and confusion
    matrices.
4.  Discover topics using LDA and NMF.
5.  Compare weekly topic distributions.
6.  Flag candidate changes for human review.
7.  Document limitations and future validation requirements.

## 1.6 Scope

The current prototype uses a synthetic dataset and notebook-based
processing. State clearly whether a user interface, streaming ingestion,
or live deployment was actually implemented.

## 1.7 Contributions

List only completed contributions. Possible contribution: an integrated
prototype combining baseline classification, topic discovery, weekly
trends, and heuristic review alerts.

## 1.8 Thesis Organization

Briefly describe Chapters 1--7.

## References needed

Use verified CTI, NLP, text classification, topic modeling, and
emerging-threat literature. Link every major claim to a real reference.
