# 05. Classification Evaluation

## Metrics

-   **Precision:** among reports predicted as a category, how many were
    correct?
-   **Recall:** among reports truly in a category, how many were found?
-   **F1-score:** harmonic mean of precision and recall.
-   **Accuracy:** proportion of all test predictions that were correct.
-   **Confusion matrix:** shows which categories are confused with one
    another.

## Evaluation code

``` python
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

for name, pipe in pipelines.items():
    pred = pipe.predict(X_test)
    print("\\n", name)
    print("Accuracy:", accuracy_score(y_test, pred))
    print(classification_report(y_test, pred, zero_division=0))

best_pipe = pipelines[best_name]
best_pred = best_pipe.predict(X_test)
labels = sorted(y.unique())
cm = confusion_matrix(y_test, best_pred, labels=labels)

plt.figure(figsize=(11, 8))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels,
            yticklabels=labels, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("artifacts/confusion_matrix.png", dpi=200)
plt.show()
```

## Reporting rule

Report the split, seed, number of test examples, metrics, and
limitations. Do not report training performance as test performance.
Avoid describing a synthetic-data score as real-world effectiveness.
