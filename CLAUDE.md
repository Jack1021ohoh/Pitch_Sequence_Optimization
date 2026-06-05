# Pitch Sequence Transformer — Claude Instructions

## Data splits

| Split | Years       |
|-------|-------------|
| Train | 2021–2023   |
| Val   | 2024        |
| Test  | 2025        |

Limited to 2021–2025 due to computing constraints.

Scalers and vocabularies are **fit on training data only** (`preprocess.py`) and applied to val/test without refitting.

## Column naming conventions

Encoded columns in the processed parquets follow strict prefixes:

| Prefix | Meaning |
|--------|---------|
| `feat_` | Standardized continuous feature |
| `cat_`  | Integer-encoded categorical feature |
| `pitch_outcome_mask`, `hit_location_mask` | 0 = visible, 1 = masked (applied to final pitch at training time) |
| `pitch_outcome_label`, `hit_location_label` | Integer class label; -1 = unknown (not a valid training target) |

## Adding a new feature

All column lists are defined once in `data/preprocess.py` (`CONTINUOUS_COLS`, `OUTCOME_CONTINUOUS_COLS`, `CATEGORICAL_COLS`) and imported by the sequence builders. **Always edit `preprocess.py` first** — the builders derive their column lists from these constants automatically.

After changing `preprocess.py`, re-run the full pipeline:
1. `python data/preprocess.py`
2. `python data/build_batter_sequences.py`
3. `python data/build_pitcher_appearances.py`

## Sequence boundary rules

- Batter sequences are 400-pitch sliding windows.
- Windows that span a **season boundary** are excluded. The check is: `years[end_row - 399] == years[end_row]`. This prevents a sequence from connecting the last pitch of one season to the first pitch of the next.
- Postseason is excluded at fetch time (`SEASON_END = '09-30'`).

## Pitcher context vector

`days_since_last_appearance` (slot 0 of the 7-dim context vector) is **capped at 30 days**. This prevents off-season gaps (~180 days) from producing outlier values that dwarf all in-season values. Any gap > 30 days is effectively treated as "start of season."

## Vocabs

`data/artifacts/vocabs.json` maps raw values to integers for all categorical features. Unknown values fall back to 0 at encode time (safe default — 0 is a valid index for all vocabularies). Labels use -1 for unknowns so they can be masked out of the loss.

## Batter array layout

Stored as `(T, 32)` `.npy` per batter. Model receives columns `[0:30]` (30-dim input); columns `[30:32]` are labels only.

| Cols    | Content |
|---------|---------|
| `[0:11]`  | `feat_*` — 11 standardized continuous features |
| `[11:15]` | `feat_outcome_*` — 4 outcome-continuous features (zeroed on final pitch) |
| `[15:28]` | `cat_*` — 13 integer-encoded categoricals |
| `[28:30]` | mask flags: `pitch_outcome_mask`, `hit_location_mask` (set to 1 on final pitch) |
| `[30:32]` | labels: `pitch_outcome_label`, `hit_location_label` |

Pitcher appearance arrays are `(T, 28)` — same layout as `[0:28]` above, no mask flags.

## Key constants (must stay in sync across files)

| Constant | Value | Defined in |
|----------|-------|-----------|
| `SEQ_LEN` | 400 | `training/dataset.py` |
| `K_APPEARANCES` | 7 | `training/dataset.py` |
| `MAX_PITCH_LEN` | 150 | `training/dataset.py` |
| `PITCHER_F` | 28 | `training/dataset.py` |
| `N_PITCH_OUTCOME` | 10 | `models/full_model.py` |
| `N_HIT_LOCATION` | 10 | `models/full_model.py` |
| `N_MOG_COMPONENTS` | 2 | `models/full_model.py` |
| `PHYSICS_FEAT_IDX` | `[0,4,5,6,7]` | `models/full_model.py` |
| `CAT_VOCAB_SIZES` | 13 entries | `models/embeddings.py` |

`PHYSICS_FEAT_IDX` indexes into `CONTINUOUS_COLS`: release_speed(0), pfx_x(4), pfx_z(5), plate_x(6), plate_z(7).

`CAT_VOCAB_SIZES` must match the number of distinct values produced by `build_vocabs()` in `preprocess.py`. If `CATEGORICAL_COLS` changes, update both.

## Contact labels

Physics loss (EV/LA NLL) is computed only on contact pitches. Contact is defined as `pitch_outcome_label in {2, 3, 4, 5, 9}` — Single, Double, Triple, Home Run, Field Out.

## Loss weights

`MultiTaskLoss` default weights: `w_cls=0.7`, `w_phy=0.2`, `w_cal=0.1`.
- `cls_loss`: focal loss on outcome + location (ignore_index=-1)
- `phy_loss`: MoG NLL on EV + LA, contact pitches only
- `cal_loss`: soft ECE on outcome head only (calibration regularizer)

Classification uses `focal_loss` (`training/loss.py`), not plain CE. Two knobs:
- `gamma` (default 2.0): focal focusing parameter; `(1 - p_t)^gamma` down-weights easy/frequent classes. `gamma=0` → (weighted) CE.
- `class_weight_power` (CLI `--class-weight-power`, default 0.0): exponent applied to the inverse-freq class weights from `class_weights.json` before they become the focal alpha. `0` → uniform weights (focal alone handles imbalance), `0.5` → sqrt softening, `1` → raw inverse-freq.

Default config is **focal-only** (`gamma=2.0`, `class_weight_power=0.0`). The two mechanisms are meant to be used one at a time, not stacked.

## Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `DATA_DIR` | `data` | Root for all processed data |
| `CKPT_DIR` | `checkpoints` | Where `best.pt` and `latest.pt` are saved |
| `BASELINE_DIR` | `baselines` | Where LightGBM model files are saved/loaded |

Set these before training on Colab. `best.pt` is saved on val top-4 outcome improvement; `latest.pt` is saved every epoch. Checkpoint keys: `epoch`, `model`, `optimizer`, `scheduler`, `metrics`.

**Checkpoint architecture compatibility:** the classification heads take a `2 * d_model` input (cross-attn final repr | raw final-pitch embedding skip). Checkpoints saved before this change have `d_model`-wide heads and cannot be loaded — they must be retrained, not resumed. All loaders (`train.py --resume`, `evaluate.py`, `run_search.py`) go through `load_model_weights()` in `models/full_model.py`, which converts the size-mismatch into a clear "retrain from scratch" error.
