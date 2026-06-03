# Baselines

LightGBM baseline for pitch outcome and hit location prediction. Operates on individual pitches with no sequential context — each pitch is a flat 22-feature vector. Used to quantify how much the transformer gains from modeling 400-pitch batter history.

## Files

### `train_lightgbm.py`

Trains two LightGBM classifiers and saves them to `$BASELINE_DIR`.

**Model 1 — Pitch outcome:** 10-class classifier over all pitches.

**Model 2 — Hit location:** 10-class classifier over contact pitches only (`pitch_outcome_label ∈ {2, 3, 4, 5, 9}`).

**Input features (22 total):**

| Group | Columns | Count |
|-------|---------|-------|
| Continuous | `feat_release_speed`, `feat_release_spin_rate`, `feat_release_pos_x`, `feat_release_pos_z`, `feat_pfx_x`, `feat_pfx_z`, `feat_plate_x`, `feat_plate_z`, `feat_vx0`, `feat_vy0`, `feat_vz0` | 11 |
| Categorical | `cat_pitch_type`, `cat_balls`, `cat_strikes`, `cat_outs_when_up`, `cat_on_1b`, `cat_on_2b`, `cat_on_3b`, `cat_inning`, `cat_stand`, `cat_p_throws`, `cat_zone` | 11 |

`cat_pitch_outcome` and `cat_hit_location_class` are excluded (they encode the targets). Outcome-continuous features (`feat_launch_speed` etc.) are excluded (unavailable at throw time). LightGBM's native categorical support is used — no one-hot encoding needed.

**Class weights** from `data/artifacts/class_weights.json` are applied as sample weights to handle class imbalance.

```bash
# Local
python baselines/train_lightgbm.py

# Override paths (e.g. Colab + Drive):
import os
os.environ['DATA_DIR']     = '/content/drive/MyDrive/pitch_sequence/data'
os.environ['BASELINE_DIR'] = '/content/drive/MyDrive/pitch_sequence/baselines'
%run baselines/train_lightgbm.py
```

**Outputs:**

| File | Description |
|------|-------------|
| `baselines/lgbm_outcome.txt` | Saved outcome model (LightGBM text format) |
| `baselines/lgbm_location.txt` | Saved hit location model (LightGBM text format) |

Model files are excluded from version control (see `.gitignore`).

### `evaluate_lightgbm.py`

Loads the saved models and reports full metrics on the test set, matching the format of `evaluation/evaluate.py`.

```bash
# Local
python baselines/evaluate_lightgbm.py

# Override paths (e.g. Colab + Drive):
import os
os.environ['DATA_DIR']     = '/content/drive/MyDrive/pitch_sequence/data'
os.environ['BASELINE_DIR'] = '/content/drive/MyDrive/pitch_sequence/baselines'
%run baselines/evaluate_lightgbm.py
```

**Metrics reported:**

| Metric | Description |
|--------|-------------|
| Top-4 recall | Fraction of pitches where true class is in top-4 predictions |
| Top-1 accuracy | Fraction of pitches where top prediction is correct |
| Log-loss | Cross-entropy between predicted probabilities and true labels |
| Brier score | Mean squared error between predicted probabilities and one-hot labels |
| ECE | Expected calibration error (10 bins) |
| Per-class Top-4 recall | Top-4 recall broken down by class |

## LightGBM Config

| Parameter | Value |
|-----------|-------|
| `num_leaves` | 127 |
| `learning_rate` | 0.05 |
| `n_estimators` | 500 |
| `min_child_samples` | 20 |
| Early stopping | 20 rounds on val loss |

## Evaluation Metric

**Top-4 recall per class** — for each class c, the fraction of true class-c pitches where c appears in the model's top-4 predicted probabilities. The 42 Analytics paper labels this metric "precision" (Section 4.1) but the computation is recall. Chosen because Ball and Strike dominate (~75% of pitches combined), making top-1 accuracy uninformative for rare but strategically important events like Singles, Doubles, and Home Runs.

## Data Requirements

Reads directly from the processed parquets — no re-preprocessing needed.

| File | Size |
|------|------|
| `data/processed/pitches_train.parquet` | ~387 MB |
| `data/processed/pitches_val.parquet` | ~143 MB |
| `data/processed/pitches_test.parquet` | ~146 MB |
| `data/artifacts/class_weights.json` | < 1 KB |
