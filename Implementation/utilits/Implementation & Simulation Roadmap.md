Absolutely. For your thesis, the implementation should be built **step-by-step from dataset preparation to simulated real-time emerging-threat detection**. The important part is to make sure every experiment can later produce figures/tables for Chapter 5.

# Implementation & Simulation Roadmap

Your overall pipeline should be:

```text
CTI Dataset
    ↓
1. Dataset Collection
    ↓
2. Dataset Validation
    ↓
3. Data Preprocessing
    ↓
4. Exploratory Data Analysis
    ↓
5. Feature Extraction
    ↓
6. Threat Classification
    ↓
7. Classification Evaluation
    ↓
8. Topic Modeling
    ↓
9. Temporal Topic Modeling
    ↓
10. Emerging Threat Detection
    ↓
11. Simulated Real-Time Detection
    ↓
12. Performance Evaluation
    ↓
13. Error Analysis
    ↓
14. Visualization
    ↓
15. Final Experimental Results
```

---

## Phase 0 — Finalize the Research Design

Before coding, freeze these decisions.

### Proposed title

**Real-Time Cyber Threat Classification and Emerging Threat Detection Using Machine Learning and Temporal Topic Modeling**

### Main research components

| Component          | Method                                                       |
| ------------------ | ------------------------------------------------------------ |
| Text preprocessing | NLP                                                          |
| Feature extraction | TF-IDF                                                       |
| Classification     | NB, Logistic Regression, SVM, Random Forest                  |
| Topic modeling     | LDA + NMF                                                    |
| Temporal analysis  | Time-window topic distributions                              |
| Emerging detection | Topic growth + novelty + frequency                           |
| Simulation         | Chronological incoming CTI reports                           |
| Evaluation         | Accuracy, Precision, Recall, F1, confusion matrix, coherence |

Do **not** start with BERT, LLMs, knowledge graphs, etc. They can become future work or an optional experiment.

Your core implementation should remain manageable.

---

# Phase 1 — Dataset Collection

## Step 1. Collect CTI reports

Your dataset should contain textual threat intelligence.

Recommended basic structure:

```text
data/
├── raw/
│   └── cti_reports.csv
├── processed/
│   └── cti_cleaned.csv
└── final/
    └── cti_final.csv
```

Minimum fields:

```text
id
title
text
category
date
source
url
```

Example:

```text
id: CTI0001
title: LockBit ransomware campaign
text: ...
category: Ransomware
date: 2024-01-15
source: Threat Report
```

### Target

Try to obtain **at least 3,000–10,000 documents** if feasible.

But don't artificially inflate the dataset. A smaller, well-cleaned dataset is preferable to a huge noisy dataset.

---

# Phase 2 — Dataset Validation

## Step 2. Inspect the raw dataset

Your notebook should answer:

* How many documents?
* How many categories?
* How many documents per category?
* How many missing values?
* How many duplicate reports?
* Date range?
* Average document length?
* Minimum/maximum document length?
* How many reports per year/month?

Create:

```text
Dataset Shape
Missing Values
Duplicate Count
Class Distribution
Temporal Distribution
Text Length Distribution
```

### Output

For Chapter 5:

**Table 5.1 — Dataset Statistics**

```text
Total Documents
Unique Categories
Date Range
Average Text Length
Duplicate Documents
Missing Documents
```

---

# Phase 3 — Data Cleaning

## Step 3. Clean CTI text

Important: **do not blindly remove cybersecurity indicators.**

For example:

```text
CVE-2024-XXXX
192.168.1.10
example.com
powershell
cmd.exe
T1059
APT29
malware.exe
```

These may contain valuable threat information.

Your preprocessing pipeline:

```text
Raw Text
   ↓
HTML removal
   ↓
URL normalization
   ↓
Whitespace normalization
   ↓
Lowercase where appropriate
   ↓
Tokenization
   ↓
Stopword handling
   ↓
Lemmatization/Stemming
   ↓
Remove meaningless tokens
   ↓
Clean CTI Text
```

Keep the original text too:

```text
text_original
text_clean
```

This makes your experiment reproducible.

---

# Phase 4 — Exploratory Data Analysis

## Step 4. Understand the dataset

Generate:

### Figure 1

Class distribution.

```text
Ransomware       ███████████
Malware          █████████
Phishing         ███████
DDoS             █████
Vulnerability    ████
...
```

### Figure 2

Reports over time:

```text
Reports
  │
  │        █
  │    █   █
  │ █  █   █
  │ █  █ █ █
  └──────────────→ Time
```

### Figure 3

Document length distribution.

### Figure 4

Top cybersecurity terms.

These figures become useful in **Chapter 5.3 Dataset Distribution Analysis**.

---

# Phase 5 — Train/Test Strategy

This is VERY important.

Because your research includes **temporal analysis**, don't rely only on random train/test splitting.

Use two evaluation strategies.

### Experiment A — Standard classification

```text
Dataset
   ↓
80% Training
20% Testing
```

Use stratification if appropriate.

### Experiment B — Temporal simulation

```text
Earlier CTI → Historical data
Later CTI   → Simulated incoming data
```

For example:

```text
2021 ──────── 2022 ──────── 2023 ──────── 2024
     Training             Simulation
```

This better represents your research problem.

---

# Phase 6 — Feature Extraction

## Step 5. TF-IDF

Start with TF-IDF.

```text
CTI Text
   ↓
Tokenization
   ↓
TF-IDF
   ↓
Feature Matrix
```

Experiment with:

```text
ngram_range = (1,2)
min_df
max_df
max_features
```

Don't blindly tune dozens of parameters.

Save:

```text
X_train
X_test
y_train
y_test
tfidf_vectorizer
```

---

# Phase 7 — Threat Classification

## Step 6. Implement baseline models

Start with four models:

### Model 1

**Multinomial Naïve Bayes**

### Model 2

**Logistic Regression**

### Model 3

**Linear SVM**

### Model 4

**Random Forest**

Optional:

### Model 5

**XGBoost**

Your experiment:

```text
TF-IDF
   │
   ├── Naïve Bayes
   ├── Logistic Regression
   ├── SVM
   ├── Random Forest
   └── XGBoost
```

---

# Phase 8 — Classification Evaluation

## Step 7. Evaluate every model

Calculate:

```text
Accuracy
Precision
Recall
F1-score
Macro F1
Weighted F1
```

For each model.

Create:

### Table

| Model   | Accuracy | Precision | Recall |     F1 |
| ------- | -------: | --------: | -----: | -----: |
| NB      |   actual |    actual | actual | actual |
| LR      |   actual |    actual | actual | actual |
| SVM     |   actual |    actual | actual | actual |
| RF      |   actual |    actual | actual | actual |
| XGBoost |   actual |    actual | actual | actual |

**Do not invent these numbers.**

They come directly from your notebook.

---

# Phase 9 — Confusion Matrix

## Step 8. Analyze classification errors

Generate confusion matrices.

```text
                 Predicted
             Ransom Malware Phish
Actual
Ransom          82      8      2
Malware          7     91      3
Phishing         4      5     88
```

The actual values will come from your experiment.

Analyze:

* Which categories are confused?
* Why?
* Are categories semantically similar?
* Are minority classes harder to classify?

This becomes **Chapter 5.9**.

---

# Phase 10 — Topic Modeling

Now move to the second major part of your research.

## Step 9. Build topic models

Use:

### LDA

```text
Clean CTI Documents
       ↓
Document-Term Matrix
       ↓
LDA
       ↓
Topics
       ↓
Top Terms
```

### NMF

```text
TF-IDF Matrix
     ↓
NMF
     ↓
Topics
```

Start with:

```text
K = 5
K = 10
K = 15
K = 20
```

Compare coherence and interpretability.

---

# Phase 11 — Topic Interpretation

## Step 10. Name the topics

The algorithm will produce something like:

```text
Topic 1:
ransomware, encryption, victim, file, ransom

Topic 2:
vulnerability, exploit, patch, CVE, remote

Topic 3:
phishing, email, credential, malicious, link
```

You can then assign human-readable descriptions:

```text
Topic 1 → Ransomware Activity
Topic 2 → Vulnerability Exploitation
Topic 3 → Phishing Campaigns
```

Important:

**The model generates the topics; the researcher interprets them.**

---

# Phase 12 — Topic Coherence

## Step 11. Select useful topic configuration

Calculate topic coherence.

Example experiment:

```text
Number of Topics

5
10
15
20
```

Then:

```text
K    Coherence
5    actual
10   actual
15   actual
20   actual
```

Use coherence + qualitative interpretability rather than blindly selecting the maximum.

---

# Phase 13 — Temporal Topic Modeling

This is one of the most important parts of your thesis.

## Step 12. Add time windows

Group CTI reports by:

```text
Year
```

or preferably:

```text
Month
```

depending on dataset size.

Example:

```text
2023-01
2023-02
2023-03
...
2024-12
```

Then calculate topic prevalence.

For topic `k` and time period `t`:

$$
P(k,t)=
\frac{1}{|D_t|}
\sum_{d\in D_t}\theta_{d,k}
$$

Where:

* `D_t` = documents during time `t`
* `θ` = document-topic probability
* `P(k,t)` = topic prevalence

---

# Phase 14 — Topic Evolution

## Step 13. Calculate topic changes

Calculate:

$$
\Delta P_k(t)=P(k,t)-P(k,t-1)
$$

And:

$$
G_k(t)=
\frac{P(k,t)-P(k,t-1)}
{P(k,t-1)+\epsilon}
$$

This tells you:

> Which topics are increasing rapidly?

Example:

```text
Month       Ransomware    Phishing    Vulnerability

Jan            0.12         0.21          0.18
Feb            0.14         0.20          0.19
Mar            0.19         0.23          0.31
Apr            0.27         0.22          0.42
```

You can then visualize topic evolution.

---

# Phase 15 — Emerging Threat Detection

## Step 14. Define candidate signals

Your framework can combine:

### 1. Novelty

Does the topic/term appear for the first time?

### 2. Growth

Is its prevalence increasing?

### 3. Frequency

Is it appearing repeatedly?

### 4. Temporal persistence

Does the increase continue?

### 5. Association

Is it associated with other threat indicators?

---

# Phase 16 — Emerging Threat Score

## Step 15. Implement your proposed scoring mechanism

For example:

$$
ETS_k(t)=
w_NN_k(t)+
w_GG_k(t)+
w_SS_k(t)+
w_AA_k(t)+
w_TT_k(t)
$$

where:

* `N` = novelty
* `G` = growth
* `S` = frequency/salience
* `A` = association
* `T` = temporal persistence

with:

$$
\sum w_i=1
$$

For your first implementation, keep it simple.

For example:

```text
Novelty             0.20
Growth              0.30
Frequency           0.20
Association         0.15
Persistence         0.15
```

**But these weights are experimental parameters, not established facts.**

You should test sensitivity later rather than claiming these are universally optimal.

---

# Phase 17 — Simulated Real-Time Detection

This is where your title's **“Real-Time”** component becomes experimentally meaningful.

You don't need to build a production SOC.

Instead, create a **replay simulation**.

## Step 16. Sort reports chronologically

```python
df = df.sort_values("date")
```

Then process reports sequentially.

Conceptually:

```text
Historical Data
       ↓
Initialize Model
       ↓
Incoming Report
       ↓
Preprocess
       ↓
Classify
       ↓
Assign Topic
       ↓
Update Temporal Statistics
       ↓
Calculate Emerging Score
       ↓
Generate Signal
       ↓
Next Report
```

---

# Phase 18 — Streaming Simulation

## Step 17. Simulate batches

Instead of processing thousands of reports one by one initially, use time batches:

```text
Batch 1 → January
Batch 2 → February
Batch 3 → March
Batch 4 → April
...
```

At each batch:

```text
New Reports
    ↓
Classification
    ↓
Topic Assignment
    ↓
Topic Distribution
    ↓
Growth Calculation
    ↓
Emerging Score
    ↓
Signal
```

This is easier to reproduce and explain.

---

# Phase 19 — Emerging Threat Alert

## Step 18. Define an experimental threshold

For example:

```text
If ETS >= threshold
        ↓
Candidate Emerging Threat
```

But call it:

> **Candidate Emerging Threat Signal**

not:

> Confirmed New Threat

because increased textual activity does not necessarily prove that a genuinely new cyber threat exists.

---

# Phase 20 — Simulation Output

For every time window, save:

```text
date
topic_id
topic_terms
topic_prevalence
growth_rate
novelty_score
frequency_score
persistence_score
emerging_threat_score
alert
```

Example:

| Date    | Topic   | Growth |    ETS | Signal |
| ------- | ------- | -----: | -----: | ------ |
| 2024-01 | Topic 3 | actual | actual | No     |
| 2024-02 | Topic 3 | actual | actual | No     |
| 2024-03 | Topic 3 | actual | actual | Yes    |
| 2024-04 | Topic 3 | actual | actual | Yes    |

Again, **actual values come from your experiment**.

---

# Phase 21 — Evaluate the Simulation

This part is important for your thesis.

You need to answer:

### Classification

> Can CTI reports be automatically categorized?

### Topic modeling

> Can meaningful threat topics be discovered?

### Temporal modeling

> Can changes in threat topics be detected?

### Emerging detection

> Can rapidly increasing/new topic patterns be identified?

### Simulation

> Can the framework process chronologically arriving CTI data and generate candidate signals?

---

# Phase 22 — Compare Methods

You can create experiments like:

```text
Experiment 1
TF-IDF + Naïve Bayes

Experiment 2
TF-IDF + Logistic Regression

Experiment 3
TF-IDF + SVM

Experiment 4
TF-IDF + Random Forest

Experiment 5
LDA temporal analysis

Experiment 6
NMF temporal analysis

Experiment 7
Emerging Threat Score

Experiment 8
Chronological replay simulation
```

This gives your thesis a clear experimental story.

---

# Phase 23 — Visualization

Your final notebook should generate approximately:

### Classification

1. Class distribution
2. Model comparison
3. Accuracy comparison
4. Precision/Recall/F1
5. Confusion matrix

### Topic modeling

6. Topic coherence
7. Top terms per topic
8. Topic distribution
9. Topic prevalence

### Temporal

10. Topic evolution
11. Topic growth
12. New topic appearance

### Emerging threats

13. Emerging Threat Score over time
14. Candidate emerging-topic alerts
15. Top emerging terms

### Simulation

16. Incoming CTI volume
17. Processing time
18. Detection signals over time

---

# Phase 24 — Error Analysis

## Step 24. Investigate failures

Don't just report accuracy.

Look at:

```text
False Positive
False Negative
Wrong Category
Ambiguous Topic
Short Document
Duplicate Information
New Terminology
Rare Threat
```

For example:

```text
Actual: Ransomware
Predicted: Malware
```

Then inspect why.

This makes the thesis much stronger academically.

---

# Phase 25 — Final Notebook Structure

Your notebook can finally look like this:

```text
cyber-threat-intelligence-framework.ipynb

01. Research Objective
02. Environment Setup
03. Import Libraries

04. Dataset Loading
05. Dataset Validation
06. Dataset Statistics

07. Data Cleaning
08. Text Preprocessing

09. Exploratory Data Analysis

10. Train-Test Preparation
11. TF-IDF Feature Extraction

12. Naive Bayes
13. Logistic Regression
14. SVM
15. Random Forest
16. XGBoost

17. Classification Evaluation
18. Confusion Matrix
19. Model Comparison

20. LDA Topic Modeling
21. NMF Topic Modeling
22. Topic Interpretation
23. Topic Coherence

24. Temporal Dataset Preparation
25. Temporal Topic Distribution
26. Topic Evolution
27. Topic Growth

28. Emerging Threat Indicators
29. Emerging Threat Score
30. Candidate Emerging Threat Detection

31. Real-Time Replay Simulation
32. Incoming CTI Processing
33. Temporal Updating
34. Alert Generation

35. Simulation Evaluation
36. Error Analysis
37. Limitations

38. Final Visualizations
39. Export Results
40. Conclusion
```

---

# Phase 26 — Folder Structure

I recommend your project become:

```text
cyber-threat-intelligence-framework/
│
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── final/
│
├── notebook/
│   └── cyber-threat-intelligence-framework.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── features.py
│   ├── classification.py
│   ├── topic_modeling.py
│   ├── temporal_analysis.py
│   ├── emerging_detection.py
│   └── simulation.py
│
├── models/
│   ├── classifier/
│   ├── vectorizer/
│   └── topic_models/
│
├── results/
│   ├── tables/
│   ├── figures/
│   ├── classification/
│   ├── topics/
│   ├── temporal/
│   └── simulation/
│
├── reports/
│
├── README.md
└── requirements.txt
```

---

# Final Implementation Order

**Don't implement everything at once.**

Follow this exact order:

```text
STEP 1   Dataset
   ↓
STEP 2   Dataset validation
   ↓
STEP 3   Preprocessing
   ↓
STEP 4   EDA
   ↓
STEP 5   TF-IDF
   ↓
STEP 6   Classification
   ↓
STEP 7   Classification evaluation
   ↓
STEP 8   LDA
   ↓
STEP 9   NMF
   ↓
STEP 10  Topic coherence
   ↓
STEP 11  Temporal topic analysis
   ↓
STEP 12  Topic growth
   ↓
STEP 13  Emerging-threat indicators
   ↓
STEP 14  Emerging Threat Score
   ↓
STEP 15  Chronological replay simulation
   ↓
STEP 16  Simulation evaluation
   ↓
STEP 17  Error analysis
   ↓
STEP 18  Final graphs/tables
   ↓
STEP 19  Chapter 5 results
```

### Most important recommendation

**First get Steps 1–7 working completely.** Then build topic modeling. Then temporal analysis. Finally build the replay simulation.

That way, if the emerging-threat component doesn't work perfectly, you still have a complete, reproducible **CTI classification + topic modeling + temporal analysis** thesis rather than having the entire project depend on one experimental component.
