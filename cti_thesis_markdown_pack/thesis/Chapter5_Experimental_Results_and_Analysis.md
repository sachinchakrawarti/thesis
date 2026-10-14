# Chapter 5: Experimental Results and Analysis

> Fill all numerical values from the final notebook. Do not leave
> placeholders in the submitted version.

## 5.1 Experimental Setup

State machine/software configuration, package versions, dataset version,
random seed, and split.

## 5.2 Dataset Distribution

Insert row count, class count, class distribution, date range, number of
unique dates, and number of weeks.

## 5.3 Preprocessing Results

State the number of records before and after cleaning and any
exclusions.

## 5.4 Classification Performance

Insert an actual classification report for each trained model.

  Model                               Accuracy   Macro precision   Macro recall   Macro F1
  ------------------------- ------------------ ----------------- -------------- ----------
  Logistic Regression         Fill from output                                  
  Multinomial Naive Bayes     Fill from output                                  
  Linear SVM                  Fill from output                                  

## 5.5 Confusion Matrix

Insert the saved confusion-matrix figure and explain important errors.

## 5.6 Leakage and Robustness Checks

Discuss explicit category cues, template repetition, duplicate text, and
the limits of random splitting. If no additional robustness experiment
was run, state this as a limitation.

## 5.7 Topic Modeling Results

Insert LDA and NMF top words and explain how topics were interpreted. Do
not assign semantic labels from top words alone without inspecting
reports.

## 5.8 Weekly Topic Trends

Insert the LDA/NMF plots. Explain changes, sample volume, and potential
synthetic-generation effects.

## 5.9 Emerging-Pattern Candidate Alerts

Report the actual alert rule, count of generated candidate alerts, and
examples reviewed. Do not claim confirmed emerging threats without
external ground truth.

## 5.10 Error Analysis

Discuss classification errors, ambiguous reports, short text, label
leakage, and unstable topics as observed.

## 5.11 Discussion

Compare findings with the base paper only where datasets and methods are
sufficiently comparable. The base paper's metrics are not directly
comparable with results on this synthetic dataset.

## 5.12 Limitations

Synthetic data, no real-world validation, heuristic alerts, and notebook
prototype status.

## 5.13 Chapter Summary

Summarize what the experiments establish and what they do not establish.
