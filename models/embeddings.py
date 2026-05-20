"""Pitch token embedding layer shared by batter and pitcher encoders.

Input tensor column layout (per pitch, matches ALL_COLS / PITCH_FEAT_COLS order):
  [0  :11]  feat_continuous          (11 cols, standardized floats)
  [11 :15]  feat_outcome_continuous  (4 cols, zeroed when masked)
  [15 :28]  cat_*                    (13 cols, integer-coded categoricals)
  [28 :30]  mask flags               (2 cols, batter only; omit for pitcher)

Categorical column order must match CATEGORICAL_COLS in data/preprocess.py:
  pitch_type, pitch_outcome, hit_location_class,
  balls, strikes, outs_when_up, on_1b, on_2b, on_3b,
  inning, stand, p_throws, zone
"""

import torch
import torch.nn as nn

N_CONTINUOUS   = 11
N_OUTCOME_CONT = 4
N_CATEGORICAL  = 13
FEAT_DIM       = N_CONTINUOUS + N_OUTCOME_CONT  # 15

# Vocab sizes must cover all integer values produced by build_vocabs() + fillna(0)
CAT_VOCAB_SIZES = [13, 10, 10,  5,  4,  4,  2,  2,  2, 11,  3,  2, 15]
CAT_EMBED_DIMS  = [16, 16,  8,  4,  4,  4,  4,  4,  4,  8,  4,  4,  8]
TOTAL_CAT_DIM   = sum(CAT_EMBED_DIMS)  # 92


class PitchEmbedding(nn.Module):
    """Embed one pitch token (or a batch) into d_model-dimensional space.

    n_mask_flags: 2 for batter sequences (pitch_outcome_mask, hit_location_mask),
                  0 for pitcher appearances (all pitches are fully observed).
    """

    def __init__(self, d_model: int, n_mask_flags: int = 2):
        super().__init__()
        self.n_mask_flags = n_mask_flags
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, embed_dim)
            for vocab_size, embed_dim in zip(CAT_VOCAB_SIZES, CAT_EMBED_DIMS)
        ])
        n_scalars = FEAT_DIM + n_mask_flags
        self.projection = nn.Linear(n_scalars + TOTAL_CAT_DIM, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., F)  →  (..., d_model)"""
        cont = x[..., :FEAT_DIM]                                          # (..., 15)
        cats = x[..., FEAT_DIM:FEAT_DIM + N_CATEGORICAL].long()           # (..., 13)

        cat_embeds = torch.cat(
            [emb(cats[..., i]) for i, emb in enumerate(self.cat_embeddings)],
            dim=-1,
        )                                                                  # (..., 92)

        if self.n_mask_flags > 0:
            masks = x[..., FEAT_DIM + N_CATEGORICAL:
                          FEAT_DIM + N_CATEGORICAL + self.n_mask_flags]
            scalars = torch.cat([cont, masks], dim=-1)                    # (..., 17)
        else:
            scalars = cont                                                 # (..., 15)

        return self.norm(self.projection(torch.cat([scalars, cat_embeds], dim=-1)))
