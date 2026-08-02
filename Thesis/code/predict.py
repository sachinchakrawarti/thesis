from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ''}:
    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root))

import torch
import yaml

from code.data.dataset import prepare_data
from code.models.model import RegimeTransformer


def main():
    root = Path(__file__).resolve().parent
    config_path = root / 'config' / 'config.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    data = prepare_data(window_size=cfg['window_size'])
    test_x, _, _ = data['test']

    model = RegimeTransformer(
        input_dim=cfg['input_dim'],
        embedding_dim=cfg['embedding_dim'],
        num_heads=cfg['num_heads'],
        num_layers=cfg['num_layers'],
        dropout=cfg['dropout'],
        num_classes=cfg['num_classes']
    )
    model.load_state_dict(torch.load(root.parent / 'results' / 'model_state.pt', map_location='cpu'))
    model.eval()

    with torch.no_grad():
        regime_logits, forecast = model(test_x[:2])
        pred_regimes = torch.argmax(regime_logits, dim=1)
        print('Predicted regimes:', pred_regimes.tolist())
        print('Forecast values:', forecast.tolist())


if __name__ == '__main__':
    main()
