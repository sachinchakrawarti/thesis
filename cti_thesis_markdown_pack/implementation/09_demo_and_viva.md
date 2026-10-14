# 09. Demonstration and Viva Preparation

## Five-minute demonstration

1.  Show dataset fields and class distribution.
2.  Show preprocessing that joins report title and body.
3.  Train or load the saved text-classification pipeline.
4.  Enter one report and show the predicted category.
5.  Show evaluation metrics and confusion matrix.
6.  Show LDA/NMF top words and weekly topic plots.
7.  Explain one candidate alert and its limitations.

## Example inference

``` python
import joblib

pipe = joblib.load("artifacts/cti_text_classifier.joblib")
report = "Suspicious email campaign uses a fake login page to collect credentials."
print(pipe.predict([report])[0])
```

The output is a model prediction, not an authoritative security
determination.

## Common viva questions

**What is CTI?** Cyber Threat Intelligence is analyzed information about
threats, threat actors, techniques, indicators, and context used to
support security decisions.

**Why TF-IDF?** It represents text numerically while giving greater
weight to terms that are informative within the corpus.

**Why LDA and NMF?** They provide complementary topic-discovery
approaches; LDA uses a probabilistic topic mixture, while NMF factorizes
a non-negative document-term representation.

**What is temporal topic analysis?** It examines how the prevalence of
discovered themes changes across time periods.

**Does a topic spike prove a new attack?** No. It is a candidate signal
that requires analyst review and external validation.

**What is the main limitation?** The current synthetic dataset cannot
establish real-world generalization.
