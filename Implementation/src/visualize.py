"""Visualization helpers."""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
from .config import FIGURES_DIR

sns.set_style("whitegrid")

def plot_confusion_matrices(fitted_models, X, y):
    for name, (clf, y_te, y_pred) in fitted_models.items():
        labels = np.unique(y_te)
        cm = confusion_matrix(y_te, y_pred, labels=labels)
        plt.figure(figsize=(7, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=labels, yticklabels=labels)
        plt.title(f"Confusion Matrix - {name}")
        plt.xlabel("Predicted"); plt.ylabel("Actual")
        plt.tight_layout()
        out = FIGURES_DIR / f"cm_{name}.png"
        plt.savefig(out, dpi=150); plt.close()
        print(f"Saved: {out}")
