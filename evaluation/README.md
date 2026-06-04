# Evaluation

Metrics and evaluation script for Phase 1.

## Usage

```bash
python -m evaluation.evaluate --checkpoint checkpoints/best.pt --split test
```

## Files

### `metrics.py`

All metric functions take concatenated tensors over the full split (not per-batch averages) and return `float`.

| Function | Description |
|----------|-------------|
| `top_k_precision(logits, labels, k)` | Fraction of samples where true label is in the top-k predicted classes. Ignores `label == -1`. |
| `log_loss_score(logits, labels)` | Cross-entropy loss (same as training CE). |
| `brier_score(logits, labels, n_classes)` | Mean squared error between predicted probabilities and one-hot targets. |
| `expected_calibration_error(logits, labels, n_bins=10)` | Weighted mean \|confidence − accuracy\| across 10 equal-width bins. |
| `mog_nll(params, targets)` | Float wrapper around `training.loss.mog_nll`. Reports mean NLL of the predicted mixture on contact pitches. |
| `physics_mae(params, targets, scale)` | MAE between predicted MoG mean and target, in original units. `scale` is `outcome_scaler.scale_[i]` to undo standardization. |
| `mog_mean(params)` | Expected value of the MoG: `Σ w_k μ_k`. Used by `physics_mae`. |

All functions filter out `ignore_index=-1` labels before computing.

### `evaluate.py`

Loads a checkpoint, runs inference on one split, and prints a formatted report.

**`ev_scale` / `la_scale`** are read from `data/artifacts/outcome_scaler.pkl` (`scaler.scale_[0]` and `[1]`) to convert standardized predictions back to mph and degrees.

**Physics metrics** are reported only when `is_contact` is true for at least one sample in the split.

**Report format:**

```
── Pitch Outcome ───────────────────────────────
  Top-4 recall : ...
  Top-1 precision : ...
  Log-loss        : ...
  Brier score     : ...
  ECE             : ...

── Hit Location ─────────────────────────────────
  ...

── Physics (contact pitches only) ───────────────
  EV NLL          : ...
  LA NLL          : ...
  EV MAE (mph)    : ...
  LA MAE (deg)    : ...
```

**Arguments:**

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint` | *(required)* | Path to `.pt` checkpoint |
| `--split` | `test` | `train`, `val`, or `test` |
| `--batch-size` | 128 | Inference batch size |
| `--workers` | 4 | DataLoader workers (use 0 on Windows) |

`DATA_DIR` is read from the environment (default: `data`).

## Test Set Results

Evaluated on 2025 regular-season data. Transformer uses `best.pt`; LightGBM uses `lgbm_outcome.txt` / `lgbm_location.txt` trained on 2021–2023 and early-stopped on 2024 val.

### Transformer

#### Pitch Outcome

| Metric | Score |
|--------|-------|
| Top-4 recall | 0.9627 |
| Top-1 precision | 0.5817 |
| Log-loss | 1.0475 |
| Brier score | 0.5337 |
| ECE | 0.0267 |

**Per-class Top-4 recall:**

| Class | Recall |
|-------|--------|
| Ball | 0.9795 |
| Strike | 0.9577 |
| Single | 0.9820 |
| Double | 0.3752 |
| Triple | 0.0000 |
| Home Run | 0.5652 |
| Strikeout | 0.9926 |
| Walk | 0.9853 |
| Hit by Pitch | 0.9757 |
| Field Out | 0.9979 |

#### Hit Location

| Metric | Score |
|--------|-------|
| Top-4 recall | 0.6912 |
| Top-1 precision | 0.2035 |
| Log-loss | 2.0492 |
| Brier score | 0.8559 |
| ECE | 0.0386 |

**Per-class Top-4 recall:**

| Class | Recall |
|-------|--------|
| Pitcher | 0.5115 |
| Catcher | 0.2221 |
| First Base | 0.6612 |
| Second Base | 0.6472 |
| Third Base | 0.6585 |
| Shortstop | 0.6276 |
| Left Field | 0.7462 |
| Center Field | 0.7515 |
| Right Field | 0.7258 |
| None | 0.7673 |

#### Physics (contact pitches only)

| Metric | Score |
|--------|-------|
| EV NLL | 0.2032 |
| LA NLL | 1.6356 |
| EV MAE | 9.99 mph |
| LA MAE | 20.07° |

### LightGBM Baseline

#### Pitch Outcome

| Metric | Score |
|--------|-------|
| Top-4 recall | 0.9581 |
| Top-1 accuracy | 0.5757 |
| Log-loss | 1.1137 |
| Brier score | 0.5573 |
| ECE | 0.0461 |

**Per-class Top-4 recall:**

| Class | Recall |
|-------|--------|
| Ball | 0.9738 |
| Strike | 0.9521 |
| Single | 0.9886 |
| Double | 0.4856 |
| Triple | 0.0091 |
| Home Run | 0.2463 |
| Strikeout | 1.0000 |
| Walk | 0.9898 |
| Hit by Pitch | 0.9489 |
| Field Out | 0.9970 |

#### Hit Location

| Metric | Score |
|--------|-------|
| Top-4 recall | 0.6417 |
| Top-1 accuracy | 0.1845 |
| Log-loss | 2.1330 |
| Brier score | 0.8687 |
| ECE | 0.0167 |

**Per-class Top-4 recall:**

| Class | Recall |
|-------|--------|
| Pitcher | 0.5976 |
| Catcher | 0.1663 |
| First Base | 0.6199 |
| Second Base | 0.6386 |
| Third Base | 0.6234 |
| Shortstop | 0.6045 |
| Left Field | 0.6910 |
| Center Field | 0.6372 |
| Right Field | 0.6646 |
| None | 0.7284 |
