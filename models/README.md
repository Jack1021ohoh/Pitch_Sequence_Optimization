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

12-layer Pre-LN transformer encoder over the 400-pitch batter sequence. Input: `(B, 400, 30)`. Returns a tuple `(encoded, embedded)`, each `(B, 400, d_model)`:

- `encoded` — the context-mixed transformer output.
- `embedded` — the pre-transformer pitch embeddings. The final token is the un-smoothed representation of the masked pitch; the classification heads consume it directly as a skip connection to recover the sharp per-pitch signal (location, zone, pitch type, velocity) that the 12-layer attention stack blurs.

Other details:
- Sinusoidal positional encoding (added before the transformer, not to the returned `embedded`)
- Sub-token masking is applied by the Dataset before the tensor reaches this module

### `pitcher_encoder.py` — `HierarchicalPitcherEncoder`

Two-level encoder that handles starter/reliever asymmetry without explicit role labels.

**Level 1 (pitch-level):** For each of the K most recent appearances, encode all pitches with a 4-layer transformer and mean-pool to a single appearance embedding.

**Level 2 (appearance-level):** Concatenate each appearance embedding with a 7-dim context vector (rest days, workload, game state on entry), project to `d_model`, then encode across K appearances with a 3-layer transformer.

### `full_model.py` — `PitchOutcomeModel`

Assembles all components:

1. **Batter encoder** → `(encoded, embedded)`, each `(B, 400, d_model)`
2. **Pitcher encoder** → `(B, K, d_model)`
3. **Cross-attention** (batter attends to pitcher) → `(B, 400, d_model)`
4. Final pitch position → classification heads + physics head

**Classification heads:** Two FC layers → 10-class pitch outcome + 10-class hit location. Each head takes a `2 * d_model` input: the cross-attention final representation concatenated with the raw final-pitch embedding (`embedded[:, -1]`). The raw skip gives sharp access to location/zone/pitch-type that the deep encoder smooths out, which helps rare contact classes (e.g. doubles, triples). No label leak: the final pitch's outcome/location features are already masked by the Dataset before embedding.

**Physics head:** Conditioned on encoder output + 5 pitch physics features (velocity, movement, plate location). Outputs 2-component mixture-of-Gaussians parameters for exit velocity and launch angle independently.

Also exports:
- `run_model(model, batch) -> dict` — a convenience wrapper used by training and evaluation to avoid repeating the 5-argument model call.
- `load_model_weights(model, state_dict)` — loads weights and raises a clear error if the checkpoint predates an architecture change (e.g. the `2 * d_model` heads), instead of a cryptic size-mismatch traceback. Used by `train.py --resume`, `evaluate.py`, and `run_search.py`. Checkpoints from before that change must be retrained, not resumed.

## Key Constants

| Constant | Value | Meaning |
|---|---|---|
| `N_PITCH_OUTCOME` | 10 | Outcome classes (Ball, Strike, Single, …) |
| `N_HIT_LOCATION` | 10 | Location classes (Pitcher, …, Right Field, None) |
| `N_MOG_COMPONENTS` | 2 | MoG components for physics head |
| `PHYSICS_FEAT_IDX` | `[0,4,5,6,7]` | Indices into CONTINUOUS_COLS for physics conditioning |
