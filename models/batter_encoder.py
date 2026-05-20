"""12-layer transformer encoder over the 400-pitch batter sequence."""

import math
import torch
import torch.nn as nn
from .embeddings import PitchEmbedding


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 400):
        super().__init__()
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class BatterEncoder(nn.Module):
    def __init__(
        self,
        d_model:         int   = 256,
        nhead:           int   = 8,
        num_layers:      int   = 12,
        dim_feedforward: int   = 1024,
        dropout:         float = 0.1,
    ):
        super().__init__()
        self.embedding = PitchEmbedding(d_model, n_mask_flags=2)
        self.pos_enc   = SinusoidalPositionalEncoding(d_model)
        encoder_layer  = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, batter_seq: torch.Tensor) -> torch.Tensor:
        """
        batter_seq: (B, 400, 30)  — columns 0-29 of the stored array (no label cols)
        Returns:    (B, 400, d_model)
        """
        x = self.pos_enc(self.embedding(batter_seq))
        return self.transformer(x)
