# Temporal Topic Modeling — Explanation with Examples

## 1. What Is Topic Modeling?

Topic modeling is a Natural Language Processing (NLP) technique used to discover common themes or topics in a collection of text documents.

For example, suppose we have these CTI reports:

| Report | Report text | Possible topic |
|---|---|---|
| R1 | Attackers send fake emails to steal passwords. | Phishing |
| R2 | Malicious software encrypts files and demands payment. | Ransomware |
| R3 | Attackers exploit a server vulnerability to gain access. | Exploitation |
| R4 | Fake login pages collect employee credentials. | Phishing |

Topic modeling can discover that R1 and R4 discuss similar subjects, even though their wording is different.

Common topic modeling methods include:
- Latent Dirichlet Allocation (LDA)
- Non-negative Matrix Factorization (NMF)
- BERTopic

Topic modeling is generally unsupervised: it discovers patterns without requiring a predefined category label for each topic.

## 2. What Is Temporal Topic Modeling?

Temporal Topic Modeling combines topic modeling with time information.

It studies how topics change over time by analyzing documents with dates or timestamps.

In simple words:

**Topic Modeling:** What subjects appear in the reports?

**Temporal Topic Modeling:** How do those subjects change over time?

For your CTI thesis, it can help analyze whether discussions of phishing, ransomware, credential theft, or another topic become more or less frequent over successive weeks or months.

## 3. Simple Real-Life Example

Imagine a news organization publishes reports about technology.

During January, most articles discuss artificial intelligence. In February, articles about data breaches become more frequent. In March, ransomware receives much more attention.

A temporal analysis could show this change:

| Month | AI reports | Data breach reports | Ransomware reports |
|---|---:|---:|---:|
| January | 50 | 10 | 5 |
| February | 40 | 25 | 10 |
| March | 30 | 35 | 30 |

Interpretation:

- AI coverage decreases from January to March.
- Data breach coverage increases gradually.
- Ransomware coverage increases sharply in March.

This indicates changing topics over time. It does not, by itself, prove that ransomware attacks actually increased in the real world.

## 4. Example Related to Your CTI Thesis

Suppose your system collects the following reports.

| Report date | CTI report |
|---|---|
| January 5 | A phishing email targets employees to steal passwords. |
| January 12 | Attackers distribute ransomware to encrypt company files. |
| February 3 | Several reports describe credential theft through fake login pages. |
| February 15 | Attackers use phishing to obtain corporate credentials. |
| March 2 | Reports describe ransomware attacks against multiple organizations. |
| March 10 | New reports discuss ransomware delivery through compromised accounts. |

A topic modeling algorithm might identify topics such as:

- Topic 1: phishing, email, credentials, login
- Topic 2: ransomware, encryption, files, payload
- Topic 3: exploitation, vulnerability, server, access

These labels are illustrative. The algorithm actually produces groups of words and weights; a researcher interprets the topic and gives it a human-readable name.

Next, the system measures each topic's frequency in January, February, and March.

If the ransomware topic becomes substantially more common in March, the system can flag it for investigation.

**Important distinction:** A topic becoming more common in reports may indicate increased reporting, greater attention, duplicate reporting, or a real change in threat activity. Additional evidence is needed to distinguish these explanations.

## 5. How Temporal Topic Modeling Works

The process can be represented as follows:

CTI Reports with Dates  
↓  
Text Preprocessing  
↓  
TF-IDF or Another Text Representation  
↓  
Topic Modeling (for example, NMF or LDA)  
↓  
Topic Assignment or Topic Weights  
↓  
Grouping Reports by Week or Month  
↓  
Measuring Topic Frequency Over Time  
↓  
Detecting Changes and Reviewing Emerging Patterns

### Step 1: Collect reports

Collect CTI reports containing a report identifier, publication date, and report text.

Example:

```text
report_id: CTI001
date: 2026-01-05
text: Attackers use phishing emails to steal credentials.
```

### Step 2: Preprocess the text

Typical operations include:
- Converting text to lowercase.
- Removing unnecessary punctuation and whitespace.
- Tokenizing text.
- Removing common stop words when appropriate.
- Handling duplicate reports.

Cybersecurity terminology and indicators must be handled carefully. Removing important technical terms may reduce model quality.

### Step 3: Convert text into numerical features

TF-IDF converts report text into numerical feature vectors that reflect the importance of terms within the document collection.

For example, terms such as `phishing`, `credential`, and `ransomware` may help distinguish different report themes.

### Step 4: Discover topics

NMF or LDA can identify recurring word patterns across the reports.

A possible NMF topic might have these high-weight words:

```text
ransomware, encryption, files, payload, ransom
```

A researcher may interpret this as a ransomware-related topic.

### Step 5: Add the time dimension

Group reports by publication month or week.

Calculate how much of the reporting activity in each period is associated with each topic.

For example:

| Month | Phishing share | Ransomware share |
|---|---:|---:|
| January | 60% | 20% |
| February | 45% | 30% |
| March | 25% | 55% |

These values are illustrative, not experimental results.

### Step 6: Identify changes

Compare topic frequency across periods.

For example, ransomware's share increases from 30% in February to 55% in March.

The system may flag this increase for analyst review. A robust system should consider the number of reports, reporting-source changes, duplicate reports, and whether the increase is statistically meaningful.

## 6. Python Example Using Pandas

This small example shows the temporal analysis step after reports have already been assigned to topics.

It does not train a topic model; it demonstrates how to calculate topic frequencies over time.

```python
import pandas as pd

data = {
    "date": [
        "2026-01-05",
        "2026-01-12",
        "2026-02-03",
        "2026-02-15",
        "2026-03-02",
        "2026-03-10"
    ],
    "topic": [
        "Phishing",
        "Ransomware",
        "Phishing",
        "Phishing",
        "Ransomware",
        "Ransomware"
    ]
}

df = pd.DataFrame(data)

df["date"] = pd.to_datetime(df["date"])
df["month"] = df["date"].dt.to_period("M").astype(str)

monthly_counts = pd.crosstab(
    df["month"],
    df["topic"]
)

monthly_shares = monthly_counts.div(
    monthly_counts.sum(axis=1),
    axis=0
)

print("Monthly topic counts:")
print(monthly_counts)

print("\nMonthly topic shares:")
print(monthly_shares.round(2))
```

In this example, the topic labels are supplied manually. In a complete research pipeline, a topic model would first generate topic assignments or weights from the report text.

## 7. What Is the Difference Between Topic Modeling and Temporal Topic Modeling?

| Feature | Topic Modeling | Temporal Topic Modeling |
|---|---|---|
| Main question | What themes exist? | How do themes change over time? |
| Uses report text | Yes | Yes |
| Uses dates | Not necessarily | Yes |
| Discovers topics | Yes | Yes, or uses topics tracked over time |
| Measures topic changes | No, not by itself | Yes |
| Can support emerging-pattern analysis | Limited by itself | Yes, as one source of evidence |

A basic workflow that fits NMF once and then compares monthly topic shares is a useful **temporal topic analysis prototype**. It is not necessarily a fully dynamic topic model that learns how the topics themselves evolve over time.

## 8. How It Fits into the CTI Thesis

Your proposed framework has two complementary analytical tasks.

### A. Threat Classification

Question: **What type of threat does this report describe?**

Example output:

```text
Input: CTI report text
Predicted category: Phishing
```

A supervised model such as Logistic Regression, Multinomial Naive Bayes, or Linear SVM can perform this task.

### B. Temporal Topic Modeling and Emerging-Pattern Analysis

Question: **Which report themes are changing, appearing, or becoming unusually frequent over time?**

Example output:

```text
Topic: Ransomware-related activity
Period: March 2026
Observation: Topic share increased compared with February.
Action: Flag for analyst review.
```

The observation is a signal to investigate, not proof of a newly discovered attack.

## 9. Important Research Limitations

1. Synthetic reports may not represent real-world CTI language or topic frequencies.
2. Publication dates may differ from the dates when the underlying attacks occurred.
3. Repeated or syndicated reports can create artificial increases in topic frequency.
4. A fixed NMF model may assign new reports to existing topics even when a genuinely new topic appears.
5. Topic labels require human interpretation.
6. A rising topic share does not prove that real attacks are increasing.
7. To demonstrate emerging-threat detection, evaluate alerts on time-ordered data and, where possible, compare them with independently verified events.

## 10. Conclusion

Temporal Topic Modeling analyzes how the themes found in text documents change over time.

For the CTI thesis, the process can discover recurring cybersecurity themes, measure changes in their frequency, and generate candidate alerts for further investigation.

The key distinction is:

- **Threat classification:** Identifies the predicted category of an individual report.
- **Temporal topic analysis:** Measures how themes across a collection of reports change over time.
- **Emerging-pattern detection:** Uses those changes, along with suitable thresholds and additional evidence, to flag patterns that may deserve investigation.

Together, these components form a practical starting point for a machine learning-based Cyber Threat Intelligence analysis framework.