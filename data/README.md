# Data Pipeline

Run all scripts from the project root (`pitch_sequence/`) with the virtual environment active.

## Scripts

### 1. `fetch_statcast.py`

Downloads raw Statcast pitch-by-pitch data from Baseball Savant via pybaseball.

```bash
python data/fetch_statcast.py
```

- Fetches regular-season data only (March 15 – September 30) to exclude postseason.
- Saves one parquet per year to `data/raw/statcast_{year}.parquet`.
- Skips years that are already downloaded — safe to re-run after failures.
- Years fetched: 2015–2025.

```bash
# Inspect events/description value distributions across all downloaded files
python data/fetch_statcast.py inspect
```

---

### 2. `preprocess.py`

Cleans raw data, fits scalers and vocabularies on training data, and encodes all three splits.

```bash
python data/preprocess.py
```

**Outputs:**
- `data/processed/pitches_train.parquet`
- `data/processed/pitches_val.parquet`
- `data/processed/pitches_test.parquet`
- `data/artifacts/scaler.pkl` — StandardScaler for continuous features
- `data/artifacts/outcome_scaler.pkl` — StandardScaler for outcome continuous features
- `data/artifacts/vocabs.json` — integer mappings for all categorical features

**What it does:**
- Filters invalid rows (missing keys, unsupported pitch types, intent walks, errors)
- Maps `events` and `description` to 10 outcome classes (Ball, Strike, Single, Double, Triple, Home Run, Strikeout, Walk, Hit by Pitch, Field Out)
- Maps `hit_location` to 9 position classes (Pitcher, Catcher, ..., Right Field) + None
- Standardizes 11 continuous features (velocity, spin, movement, etc.)
- Encodes 13 categorical features as integers using fixed vocabularies

---

### 3. `build_batter_sequences.py`

Builds per-batter pitch history arrays and a global sequence index for training.

```bash
python data/build_batter_sequences.py
```

**Outputs:**
- `data/sequences/batter_pitches/{batter_id}.npy` — shape `(T, F)`, one row per pitch
- `data/sequences/batter_index.parquet` — columns: `batter_id`, `end_row`, `game_date`, `split`

**How sequences work:**
- At training time, the Dataset loads `batter_pitches/{batter_id}.npy` and slices a 400-pitch window ending at `end_row`.
- Only batters with ≥ 400 pitches are included.
- Windows that span a season boundary (e.g., end of 2021 into start of 2022) are excluded.

---

### 4. `build_pitcher_appearances.py`

Builds per-pitcher appearance records for the hierarchical pitcher encoder.

```bash
python data/build_pitcher_appearances.py
```

**Output:**
- `data/sequences/pitcher_appearances.pkl`

**Structure:** `{pitcher_id (int) → list of AppearanceRecord}`, sorted by date.

Each `AppearanceRecord` is a dict:
```python
{
    'game_date': pd.Timestamp,
    'game_pk':   int,
    'pitches':   np.ndarray,   # shape (num_pitches, num_features)
    'context':   np.ndarray,   # shape (7,) — appearance-level context
}
```

**Context vector layout:**
```
[0] days_since_last_appearance   (capped at 30 — avoids cross-season outliers)
[1] pitches_thrown_last_3_days
[2] appearances_last_7_days
[3] inning_entered
[4] score_differential
[5] outs_when_entered
[6] runners_on_when_entered
```

---

### 5. `build_re24_table.py`

Computes the RE24 (Run Expectancy by 24 base-out states) table from training data.
Only needed for Phase 2 (MCTS). Can be skipped for Phase 1 training.

```bash
python data/build_re24_table.py
```

**Output:** `data/re24_table.json`
```json
{
  "re24":       { "0": 0.481, "1": 0.268, ... },
  "run_values": { "home_run": 1.37, "strikeout": -0.28, ... }
}
```

State encoding: `state = on_1b + on_2b×2 + on_3b×4 + outs×8` (range 0–23).

---

### `utils.py`

Shared constants and helpers imported by all scripts above. Not run directly.

- `TRAIN_YEARS = list(range(2015, 2023))`
- `VAL_YEARS = [2023]`
- `TEST_YEARS = [2024, 2025]`
- `load_statcast_years(raw_dir, years, columns=None)` — loads and concatenates yearly parquets

---

## Output Directory Layout

```
data/
├── raw/
│   ├── statcast_2015.parquet
│   └── ...
├── processed/
│   ├── pitches_train.parquet
│   ├── pitches_val.parquet
│   └── pitches_test.parquet
├── artifacts/
│   ├── scaler.pkl
│   ├── outcome_scaler.pkl
│   └── vocabs.json
├── sequences/
│   ├── batter_index.parquet
│   ├── batter_pitches/
│   │   ├── 112345.npy
│   │   └── ...
│   └── pitcher_appearances.pkl
└── re24_table.json
```
