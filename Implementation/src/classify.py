"""Train and evaluate multiple classifiers."""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix)
from .config import TEST_SIZE, RANDOM_STATE

def _models():
    return {
        "NaiveBayes":         MultinomialNB(),
        "LogisticRegression": LogisticRegression(max_iter=1000, n_jobs=-1),
        "SVM":                LinearSVC(),
        "RandomForest":       RandomForestClassifier(n_estimators=200,
                                                     n_jobs=-1,
                                                     random_state=RANDOM_STATE),
    }

def train_all_models(X, y):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    results = []
    fitted  = {}
    for name, clf in _models().items():
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        results.append({
            "Model":     name,
            "Accuracy":  round(accuracy_score(y_te, y_pred), 4),
            "Precision": round(precision_score(y_te, y_pred, average="macro",
                                               zero_division=0), 4),
            "Recall":    round(recall_score(y_te, y_pred, average="macro",
                                            zero_division=0), 4),
            "F1":        round(f1_score(y_te, y_pred, average="macro",
                                        zero_division=0), 4),
            "CM":        confusion_matrix(y_te, y_pred, labels=np.unique(y)),
        })
        fitted[name] = (clf, y_te, y_pred)
    return results, fitted
