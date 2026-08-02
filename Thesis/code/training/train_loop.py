from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset

from code.utils.helpers import ensure_directory


def train_model(model, data, cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    train_x, train_y, train_r = data['train']
    val_x, val_y, val_r = data['val']

    train_dataset = TensorDataset(train_x, train_y, train_r)
    val_dataset = TensorDataset(val_x, val_y, val_r)

    train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg['batch_size'])

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'])
    criterion_class = torch.nn.CrossEntropyLoss()
    criterion_reg = torch.nn.MSELoss()

    history = []
    best_val_loss = float('inf')
    best_state = None

    for epoch in range(cfg['epochs']):
        model.train()
        total_loss = 0.0
        for x, y, r in train_loader:
            x = x.to(device)
            y = y.to(device)
            r = r.to(device)

            optimizer.zero_grad()
            regime_logits, forecast = model(x)
            loss = (
                cfg['lambda_regime'] * criterion_class(regime_logits, r)
                + cfg['lambda_forecast'] * criterion_reg(forecast, y)
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y, r in val_loader:
                x = x.to(device)
                y = y.to(device)
                r = r.to(device)
                regime_logits, forecast = model(x)
                val_loss += (
                    cfg['lambda_regime'] * criterion_class(regime_logits, r)
                    + cfg['lambda_forecast'] * criterion_reg(forecast, y)
                ).item()

        avg_train_loss = total_loss / max(1, len(train_loader))
        avg_val_loss = val_loss / max(1, len(val_loader))
        history.append(avg_val_loss)

        print(f'epoch={epoch + 1}, train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}')

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    ensure_directory('results')
    if best_state is not None:
        torch.save(best_state, 'results/model_state.pt')
    else:
        torch.save(model.state_dict(), 'results/model_state.pt')
    print('Training complete. Model saved to results/model_state.pt')
    return history
