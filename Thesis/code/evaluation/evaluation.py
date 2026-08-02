from __future__ import annotations

import torch
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, TensorDataset

from code.data.dataset import prepare_data
from code.models.model import RegimeTransformer
from code.visualization import save_confusion_from_predictions


def evaluate_model():
    with open('code/config/config.yaml', 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    data = prepare_data(window_size=cfg['window_size'])
    test_x, test_y, test_r = data['test']
    dataset = TensorDataset(test_x, test_y, test_r)
    loader = DataLoader(dataset, batch_size=cfg['batch_size'])

    model = RegimeTransformer(
        input_dim=cfg['input_dim'],
        embedding_dim=cfg['embedding_dim'],
        num_heads=cfg['num_heads'],
        num_layers=cfg['num_layers'],
        dropout=cfg['dropout'],
        num_classes=cfg['num_classes']
    )
    model.load_state_dict(torch.load('results/model_state.pt', map_location='cpu'))
    model.eval()

    preds = []
    labels = []
    with torch.no_grad():
        for x, _, r in loader:
            regime_logits, _ = model(x)
            pred = torch.argmax(regime_logits, dim=1)
            preds.extend(pred.tolist())
            labels.extend(r.tolist())

    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, average='macro', zero_division=0)
    rec = recall_score(labels, preds, average='macro', zero_division=0)
    f1 = f1_score(labels, preds, average='macro', zero_division=0)

    metrics = {
        'accuracy': round(acc, 4),
        'precision': round(prec, 4),
        'recall': round(rec, 4),
        'f1': round(f1, 4)
    }

    save_confusion_from_predictions(labels, preds)
    print(metrics)
    return metrics
