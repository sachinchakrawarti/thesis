# 04. Threat Classification

## Goal

Predict the threat category from a report title and body.

## Baseline pipeline

TF-IDF is fitted on the training data only through a scikit-learn
Pipeline. This avoids fitting the vocabulary on the test set.

``` python
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

X = df["model_text"]
y = df["category"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Multinomial Naive Bayes": MultinomialNB(),
    "Linear SVM": LinearSVC()
}

pipelines = {}
for name, classifier in models.items():
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True, stop_words="english",
            ngram_range=(1, 2), max_features=50000
        )),
        ("classifier", classifier)
    ])
    pipe.fit(X_train, y_train)
    pipelines[name] = pipe
    print(name, "trained")
```

## Important

Do not copy a metric from an earlier run. Recalculate metrics from the
exact dataset, split, and code used for the final thesis. If the result
is unexpectedly perfect, investigate label leakage, templates, duplicate
content, and split design.

## Save selected pipeline

``` python
from pathlib import Path
import joblib

artifact_dir = Path("artifacts")
artifact_dir.mkdir(parents=True, exist_ok=True)

# Select only after comparing evaluation results.
best_name = "Linear SVM"
joblib.dump(pipelines[best_name], artifact_dir / "cti_text_classifier.joblib")
print("Saved:", artifact_dir / "cti_text_classifier.joblib")
```

Change `best_name` only after evaluating the models.
