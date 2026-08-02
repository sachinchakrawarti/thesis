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
from code.training.train_loop import train_model
from code.visualization import plot_training_curve


def main():
    root = Path(__file__).resolve().parent
    config_path = root / 'config' / 'config.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(cfg['seed'])
    data = prepare_data(window_size=cfg['window_size'])
    model = RegimeTransformer(
        input_dim=cfg['input_dim'],
        embedding_dim=cfg['embedding_dim'],
        num_heads=cfg['num_heads'],
        num_layers=cfg['num_layers'],
        dropout=cfg['dropout'],
        num_classes=cfg['num_classes']
    )
    loss_history = train_model(model, data, cfg)
    plot_training_curve(loss_history)


if __name__ == '__main__':
    main()
