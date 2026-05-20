# Pitch Sequence Transformer — Claude Instructions

## Data splits

| Split | Years       |
|-------|-------------|
| Train | 2015–2022   |
| Val   | 2023        |
| Test  | 2024–2025   |

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
