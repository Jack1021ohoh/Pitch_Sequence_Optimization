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
  Top-4 precision : ...
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
