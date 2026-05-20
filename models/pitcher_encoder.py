"""Hierarchical pitcher encoder: pitch-level within each appearance,
then appearance-level across K appearances."""

import torch
import torch.nn as nn
from .embeddings import PitchEmbedding


class HierarchicalPitcherEncoder(nn.Module):
    """
    Level 1: encode each appearance's pitches with a shallow transformer,
             then mean-pool to a single appearance embedding.
    Level 2: concat with the 7-dim context vector, project, then encode
             across K appearances with another shallow transformer.
    """

    def __init__(
        self,
        d_model:       int   = 256,
        d_pitch:       int   = 128,
        n_pitch_layers: int  = 4,
        n_app_layers:  int   = 3,
        nhead_pitch:   int   = 4,
        nhead_app:     int   = 4,
        dim_ff_pitch:  int   = 512,
        dim_ff_app:    int   = 512,
        context_dim:   int   = 7,
        dropout:       float = 0.1,
    ):
        super().__init__()
        self.embedding = PitchEmbedding(d_pitch, n_mask_flags=0)

        pitch_layer = nn.TransformerEncoderLayer(
            d_model=d_pitch, nhead=nhead_pitch,
            dim_feedforward=dim_ff_pitch,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.pitch_encoder = nn.TransformerEncoder(pitch_layer, num_layers=n_pitch_layers)

        self.app_proj = nn.Sequential(
            nn.Linear(d_pitch + context_dim, d_model),
            nn.LayerNorm(d_model),
        )

        app_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead_app,
            dim_feedforward=dim_ff_app,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.app_encoder = nn.TransformerEncoder(app_layer, num_layers=n_app_layers)

    def forward(
        self,
        pitcher_pitches:    torch.Tensor,  # (B, K, T_max, 28)
        pitcher_pitch_mask: torch.Tensor,  # (B, K, T_max)  True = padding
        pitcher_context:    torch.Tensor,  # (B, K, 7)
        pitcher_app_mask:   torch.Tensor,  # (B, K)          True = padding
    ) -> torch.Tensor:
        """Returns (B, K, d_model)."""
        B, K, T, F = pitcher_pitches.shape

        # Level 1: encode each appearance independently (batch over B*K)
        pitches_flat    = pitcher_pitches.view(B * K, T, F)
        pitch_mask_flat = pitcher_pitch_mask.view(B * K, T)

        embedded = self.embedding(pitches_flat)                           # (B*K, T, d_pitch)
        encoded  = self.pitch_encoder(
            embedded, src_key_padding_mask=pitch_mask_flat,
        )                                                                  # (B*K, T, d_pitch)

        # Mean pool over non-padded positions
        valid  = ~pitch_mask_flat                                          # (B*K, T)
        counts = valid.sum(dim=1, keepdim=True).clamp(min=1).float()
        app_embed = (encoded * valid.unsqueeze(-1)).sum(dim=1) / counts   # (B*K, d_pitch)
        app_embed = app_embed.view(B, K, -1)                              # (B, K, d_pitch)

        # Level 2: project + encode across K appearance tokens
        app_tokens = self.app_proj(
            torch.cat([app_embed, pitcher_context], dim=-1)               # (B, K, d_pitch+7)
        )                                                                  # (B, K, d_model)

        return self.app_encoder(
            app_tokens, src_key_padding_mask=pitcher_app_mask,
        )                                                                  # (B, K, d_model)
