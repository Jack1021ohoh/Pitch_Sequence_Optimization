# Training

All training components for Phase 1.

## Files

### `dataset.py` — `PitchSequenceDataset`

Loads one sample per row in `batter_index.parquet`. Each sample is a 400-pitch batter window with the final pitch masked, plus K pitcher appearances.

**Batter array layout** (32 cols per pitch, stored in `.npy`):

| Cols    | Content |
|---------|---------|
| `[0:11]`  | `feat_*` — standardized continuous features |
| `[11:15]` | `feat_outcome_*` — continuous outcome features (zeroed on final pitch) |
| `[15:28]` | `cat_*` — integer-encoded categoricals |
| `[28:30]` | mask flags (set to 1 on final pitch) |
| `[30:32]` | labels (extracted, excluded from model input) |

Model receives columns `[0:30]` (30-dim per pitch).

**Masking** (applied in `__getitem__` on the final pitch):
- Outcome-continuous cols zeroed out
- `cat_pitch_outcome` and `cat_hit_location` set to 0
- `pitch_outcome_mask` and `hit_location_mask` set to 1

**Pitcher lookup** is O(1): a `_pitcher_game_pos` dict built at `__init__` maps `(pitcher_id, game_pk) → index` so `_load_pitcher_data` slices the last K appearances in constant time.

Contact labels (EV/LA targets are only valid for these): `{2, 3, 4, 5, 9}` (Single, Double, Triple, HR, Field Out).

### `loss.py` — `MultiTaskLoss`

```
total = 0.7 × (outcome_CE + location_CE)
      + 0.2 × (EV_NLL + LA_NLL)   [contact pitches only]
      + 0.1 × ECE                  [outcome head only]
```

**`mog_nll(params, targets)`** — mean NLL of a K-component mixture of Gaussians. `params` layout: `[logit_w × K | mu × K | log_sigma × K]`. log_sigma is clamped to `[-3, 3]`. Used by both training (returns tensor for backprop) and evaluation (wrapped to return float).

**`soft_ece(logits, labels)`** — vectorized ECE via `torch.bucketize` + `scatter_add_`. Used as a differentiable calibration regularizer during training.

**`MultiTaskLoss.forward`** returns a dict with keys `loss`, `cls_loss`, `phy_loss`, `cal_loss`. The physics and calibration terms are skipped (zero) if there are no contact pitches or no valid labels in the batch.

`CrossEntropyLoss` is constructed with `ignore_index=-1` to skip unknown labels.

### `train.py` — training loop

```bash
# Local
python -m training.train --epochs 20 --batch-size 64 --workers 0

# Colab
import os
os.environ['DATA_DIR'] = '/content/drive/MyDrive/pitch_sequence/data'
os.environ['CKPT_DIR'] = '/content/drive/MyDrive/pitch_sequence/checkpoints'
%run training/train.py --workers 2
```

**Arguments:**

| Flag | Default | Description |
|------|---------|-------------|
| `--epochs` | 20 | Number of training epochs |
| `--batch-size` | 64 | Batch size |
| `--lr` | 1e-4 | Peak learning rate |
| `--workers` | 4 | DataLoader workers (use 0 on Windows) |
| `--warmup-steps` | 1000 | Linear warmup steps |
| `--grad-clip` | 1.0 | Gradient norm clip |
| `--d-model` | 256 | Batter encoder width |
| `--resume` | None | Path to checkpoint to resume from |

**Scheduler:** Linear warmup from `1e-3 × lr` to `lr` over `--warmup-steps`, then cosine annealing to 0.

**Checkpoints** (saved to `CKPT_DIR`):
- `latest.pt` — saved every epoch
- `best.pt` — saved when val top-4 outcome precision improves

Checkpoint dict keys: `epoch`, `model`, `optimizer`, `scheduler`, `metrics`.
