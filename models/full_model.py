"""Assembled pitch outcome model.

Forward pass:
  1. Batter encoder  → (B, 400, d_model)
  2. Pitcher encoder → (B, K,   d_model)
  3. Cross-attention (batter attends to pitcher) → (B, 400, d_model)
  4. Take final position → (B, d_model)
  5. Classification heads → pitch outcome (10) + hit location (10)
  6. Physics head → MoG params for exit velocity + launch angle
"""

import torch
import torch.nn as nn

from .batter_encoder import BatterEncoder
from .pitcher_encoder import HierarchicalPitcherEncoder

N_PITCH_OUTCOME  = 10
N_HIT_LOCATION   = 10
N_MOG_COMPONENTS = 2

# release_speed(0), pfx_x(4), pfx_z(5), plate_x(6), plate_z(7) within CONTINUOUS_COLS
PHYSICS_FEAT_IDX = [0, 4, 5, 6, 7]


class PitchOutcomeModel(nn.Module):
    def __init__(
        self,
        d_model: int   = 256,
        nhead:   int   = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.batter_encoder  = BatterEncoder(d_model=d_model, nhead=nhead, dropout=dropout)
        self.pitcher_encoder = HierarchicalPitcherEncoder(d_model=d_model, dropout=dropout)

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead,
            dropout=dropout, batch_first=True,
        )
        self.cross_attn_norm = nn.LayerNorm(d_model)

        self.outcome_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.ReLU(),
            nn.Linear(d_model // 2, N_PITCH_OUTCOME),
        )
        self.location_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.ReLU(),
            nn.Linear(d_model // 2, N_HIT_LOCATION),
        )

        # Output layout per variable: [logit_w × K, mu × K, log_sigma × K]
        physics_out = 2 * 3 * N_MOG_COMPONENTS
        self.physics_head = nn.Sequential(
            nn.Linear(d_model + len(PHYSICS_FEAT_IDX), 256), nn.ReLU(),
            nn.Linear(256, 128),                              nn.ReLU(),
            nn.Linear(128, physics_out),
        )

    def forward(
        self,
        batter_seq:         torch.Tensor,  # (B, 400, 30)
        pitcher_pitches:    torch.Tensor,  # (B, K, T_max, 28)
        pitcher_pitch_mask: torch.Tensor,  # (B, K, T_max)  True = padding
        pitcher_context:    torch.Tensor,  # (B, K, 7)
        pitcher_app_mask:   torch.Tensor,  # (B, K)          True = padding
    ) -> dict:
        batter_out  = self.batter_encoder(batter_seq)
        pitcher_out = self.pitcher_encoder(
            pitcher_pitches, pitcher_pitch_mask, pitcher_context, pitcher_app_mask,
        )

        attn_out, _ = self.cross_attention(
            query=batter_out, key=pitcher_out, value=pitcher_out,
            key_padding_mask=pitcher_app_mask,
        )
        batter_out = self.cross_attn_norm(batter_out + attn_out)

        final_repr = batter_out[:, -1, :]

        K = N_MOG_COMPONENTS
        physics_params = self.physics_head(
            torch.cat([final_repr, batter_seq[:, -1, PHYSICS_FEAT_IDX]], dim=-1)
        )

        return {
            'pitch_outcome_logits': self.outcome_head(final_repr),
            'hit_location_logits':  self.location_head(final_repr),
            'ev_params':            physics_params[:, :3 * K],
            'la_params':            physics_params[:, 3 * K:],
        }


def run_model(model: PitchOutcomeModel, batch: dict) -> dict:
    return model(
        batch['batter_seq'],
        batch['pitcher_pitches'],
        batch['pitcher_pitch_mask'],
        batch['pitcher_context'],
        batch['pitcher_app_mask'],
    )
