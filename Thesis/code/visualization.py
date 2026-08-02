from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_training_curve(loss_history):
    plt.figure(figsize=(8, 5))
    plt.plot(loss_history, color='tab:blue')
    plt.title('Validation Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.tight_layout()
    plt.savefig('results/training_loss.png')
    plt.close()


def plot_confusion_matrix(cm):
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('results/confusion_matrix.png')
    plt.close()


def save_confusion_from_predictions(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm)
    return cm


def show_sample_forecast(series_values, predictions):
    plt.figure(figsize=(10, 4))
    plt.plot(series_values, label='Observed')
    plt.plot(predictions, label='Forecast')
    plt.title('Forecast Trace')
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/forecast_trace.png')
    plt.close()
