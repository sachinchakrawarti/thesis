# CTI Model Artifacts

## 1. Overview

This directory stores trained machine learning models and related artifacts for the research project:

**Real-Time Cyber Threat Classification and Emerging Threat Detection Using Machine Learning and Temporal Topic Modeling**

The artifacts support the classification of Cyber Threat Intelligence (CTI) reports into predefined threat categories.

## 2. Trained Model

**Model filename:** `cti_text_classifier.joblib`

**Model format:** Joblib serialization

**Model components:**
- TF-IDF vectorizer for converting report text into numerical features.
- Trained machine learning classifier selected from the baseline experiments.

The classifier is selected based on weighted F1-score on the held-out test dataset.

## 3. Model Training

The model is trained using the Jupyter Notebook:

`notebooks/cti_threat_analysis.ipynb`

The training workflow includes:

1. Loading the CTI report dataset.
2. Cleaning report text and target labels.
3. Extracting text features using TF-IDF.
4. Training Logistic Regression, Multinomial Naive Bayes, and Linear SVM.
5. Evaluating classification performance.
6. Selecting and saving the best-performing baseline pipeline.

## 4. Loading the Model

```python
import joblib

model = joblib.load(
    "artifacts/cti_text_classifier.joblib"
)

report = """
A phishing email attempts to steal employee credentials.
"""

prediction = model.predict([report])[0]

print("Predicted threat category:", prediction)
```

Run this code from the project root directory, or update the model path accordingly.

## 5. Important Notes

- The model file is created only after the training and saving cells run successfully.
- The model is trained on the dataset used by the notebook; it is not a pretrained general-purpose cybersecurity model.
- Predictions depend on the quality, size, and representativeness of the training dataset.
- The initial synthetic dataset is intended for pipeline development. Validate the model on real or public CTI data before making real-world performance claims.
- Record the dataset version, training configuration, evaluation metrics, and model version for reproducibility.
- Only load Joblib files from trusted sources because deserializing untrusted files can execute malicious code.

## 6. Research Scope

This artifact supports the **CTI threat classification module**. Temporal topic modeling and emerging-pattern detection are separate analytical components and must be evaluated independently.

## 7. Expected Directory Structure

```text
ctc_ectd/
├── dataset/
│   └── synthetic_cti_reports.csv
├── notebooks/
│   └── cti_threat_analysis.ipynb
└── artifacts/
    ├── README.md
    └── cti_text_classifier.joblib
```

The `.joblib` file will appear after successful model training and saving.