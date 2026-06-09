# Models

All model components for Phase 1. Built with PyTorch `nn.TransformerEncoderLayer` — no external transformer libraries.

## Files

### `embeddings.py` — `PitchEmbedding`

Shared embedding layer used by both the batter and pitcher encoders. Each pitch token is embedded by:
1. Concatenating 15 standardized continuous features + optional mask flags
2. Looking up learned embeddings for each of 13 categorical features (separate `nn.Embedding` per variable)
3. Projecting the concatenation to `d_model` via a linear layer + LayerNorm

```python
# Batter: includes 2 mask flags (pitch_outcome_mask, hit_location_mask)
PitchEmbedding(d_model=256, n_mask_flags=2)

# Pitcher: all pitches are historical, no masking
PitchEmbedding(d_model=128, n_mask_flags=0)
```

### `batter_encoder.py` — `BatterEncoder`

12-layer Pre-LN transformer encoder over the 400-pitch batter sequence. Input: `(B, 400, 30)`. Output: `(B, 400, d_model)`.

- Sinusoidal positional encoding
- Sub-token masking is applied by the Dataset before the tensor reaches this module

### `pitcher_encoder.py` — `HierarchicalPitcherEncoder`

Two-level encoder that handles starter/reliever asymmetry without explicit role labels.

**Level 1 (pitch-level):** For each of the K most recent appearances, encode all pitches with a 4-layer transformer and mean-pool to a single appearance embedding.

**Level 2 (appearance-level):** Concatenate each appearance embedding with a 7-dim context vector (rest days, workload, game state on entry), project to `d_model`, then encode across K appearances with a 3-layer transformer.

### `full_model.py` — `PitchOutcomeModel`

Assembles all components:

1. **Batter encoder** → `(B, 400, d_model)`
2. **Pitcher encoder** → `(B, K, d_model)`
3. **Cross-attention** (batter attends to pitcher) → `(B, 400, d_model)`
4. Final pitch position → classification heads + physics head

**Classification heads:** Two FC layers → 10-class pitch outcome + 10-class hit location.

**Physics head:** Conditioned on encoder output + 5 pitch physics features (velocity, movement, plate location). Outputs 2-component mixture-of-Gaussians parameters for exit velocity and launch angle independently.

Also exports `run_model(model, batch) -> dict` — a convenience wrapper used by training and evaluation to avoid repeating the 5-argument model call.

## Key Constants

| Constant | Value | Meaning |
|---|---|---|
| `N_PITCH_OUTCOME` | 10 | Outcome classes (Ball, Strike, Single, …) |
| `N_HIT_LOCATION` | 10 | Location classes (Pitcher, …, Right Field, None) |
| `N_MOG_COMPONENTS` | 2 | MoG components for physics head |
| `PHYSICS_FEAT_IDX` | `[0,4,5,6,7]` | Indices into CONTINUOUS_COLS for physics conditioning |
