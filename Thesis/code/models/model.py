from __future__ import annotations

import torch
import torch.nn as nn


class TokenMixer(nn.Module):
    def __init__(self, input_dim, embedding_dim, num_heads, num_layers, dropout):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.transformer(x)
        x = self.norm(x)
        return x


class RegimeTransformer(nn.Module):
    def __init__(self, input_dim, embedding_dim, num_heads, num_layers, dropout, num_classes):
        super().__init__()
        self.token_mixer = TokenMixer(input_dim, embedding_dim, num_heads, num_layers, dropout)
        self.regime_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim // 2, num_classes)
        )
        self.forecast_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim // 2, 1)
        )

    def forward(self, x):
        context = self.token_mixer(x)
        pooled = context.mean(dim=1)
        regime_logits = self.regime_head(pooled)
        forecast = self.forecast_head(pooled)
        return regime_logits, forecast.squeeze(-1)
