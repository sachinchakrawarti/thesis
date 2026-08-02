# Chapter 5: Implementation

## 5.1 Implementation Overview

This chapter documents the complete implementation of the proposed research pipeline. The implementation is designed to be modular, reproducible, and easy to extend. It uses Python as the primary language and organizes the research workflow into clearly separated packages.

## 5.2 Folder Structure

The complete project structure is:

```text
Thesis/
├── thesis.md
├── references.bib
├── images/
├── figures/
├── tables/
├── datasets/
├── code/
│   ├── data/
│   ├── models/
│   ├── preprocessing/
│   ├── training/
│   ├── evaluation/
│   ├── utils/
│   ├── config/
│   └── notebooks/
├── results/
├── notebooks/
├── bibliography/
└── README.md
```

## 5.3 Python Packages and Environment

The implementation relies on widely used scientific and deep-learning libraries.

### Core dependencies

- Python 3.10+
- PyTorch
- scikit-learn
- pandas
- NumPy
- matplotlib
- seaborn
- PyYAML
- tqdm

### Environment configuration

A reproducible environment is maintained via `requirements.txt`, which lists the specific package versions required for the experiment. The training configuration is stored in `config.yaml` to allow consistent experiment management across deployments.

## 5.4 Model Pipeline

The model pipeline has the following modules:

1. `dataset.py`: data loading, synthetic or external market data generation, window creation, and train-validation-test splitting.
2. `model.py`: transformer encoder and regime classifier definitions.
3. `preprocessing/`: normalization, windowing, and feature engineering utilities.
4. `training/`: training loops, optimizer setup, and checkpoint management.
5. `evaluation.py`: metrics, confusion matrix generation, and comparison logic.
6. `visualization.py`: loss curves, training curves, and ROC visuals.
7. `train.py`: training entry point.
8. `test.py`: evaluation entry point.
9. `predict.py`: inference entry point.

## 5.5 Hyperparameters

A representative configuration is shown below.

| Hyperparameter | Value |
|---|---:|
| Input window length | 60 |
| Embedding dimension | 128 |
| Number of attention heads | 4 |
| Transformer layers | 3 |
| Dropout | 0.2 |
| Batch size | 32 |
| Learning rate | 0.0005 |
| Epochs | 40 |
| Loss weights ($\lambda_1, \lambda_2$) | 0.5, 0.5 |

## 5.6 Training Procedure

The training procedure follows a standard sequence:

1. Build a validation-safe data split.
2. Normalize input features.
3. Initialize the transformer model and optimizer.
4. For each epoch, optimize the composite loss.
5. Save the model with the best validation score.
6. Re-load the best checkpoint for final evaluation.

The implementation also includes a reproducibility seed setting to support consistent experiment outcomes.

## 5.7 Evaluation Procedure

The evaluation procedure includes:

- Accuracy, precision, recall, F1 score, AUC,
- Confusion matrix visualization,
- Loss curve plotting,
- Baseline comparison with random forest, LSTM, and classical statistical baselines.

A rolling time split rather than a random split is preferred so that the test set behaves like a realistic temporal deployment setting.

## 5.8 Reproducibility Notes

The code is modular so that experiments can be rerun on a different dataset by editing only the configuration file and the data loader. This is important because financial research is often sensitive to data source and sampling frequency.

## 5.9 Chapter Summary

Chapter 5 details how the methodology is implemented in a clean, reproducible Python ecosystem. The software structure mirrors the methodological structure of the thesis, dividing the work into data, model, training, evaluation, and visualization components. Chapter 6 presents the experimental results and discusses their interpretation.
