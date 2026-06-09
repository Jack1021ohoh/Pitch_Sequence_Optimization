# Experiment Record — Focal Loss + Raw Final-Pitch Skip vs. Baseline

**Date:** 2026-06-06
**Outcome:** the focal-loss + raw-skip variants did **not** beat the weighted-CE baseline. **Baseline is kept as the final model** (checkpoint: `training/checkpoints_weighted_ece.zip`). The focal/skip code is preserved on the `focal-skip-experiment` branch; `main` is reverted to the baseline architecture so the baseline checkpoint loads.

## Runs compared

| # | Name | Loss | Architecture |
|---|------|------|--------------|
| ① | **Baseline** | weighted cross-entropy (raw inverse-freq) | single-width (`d_model`) heads, no skip |
| ② | Focal-only | focal `gamma=2.0`, `class_weight_power=0.0` (uniform alpha) | `2*d_model` heads + raw final-pitch skip |
| ③ | Weighted focal | focal `gamma=2.0`, `class_weight_power=1.0` (raw inverse-freq alpha) | `2*d_model` heads + raw final-pitch skip |

Common: `w_cls=0.7 / w_phy=0.2 / w_cal=0.1`, `d_model=256`, AdamW `lr=1e-4` (warmup→cosine), batch 256, early stopping (patience 3). Eval split: test (2025 regular season), 509,062 samples.

① is the figure recorded in `evaluation/README.md`. ② and ③ were resumed from earlier checkpoints rather than trained from scratch — see caveats.

## Test Set Results

### Pitch Outcome — aggregate

| Metric | ① Baseline | ② Focal-only | ③ Weighted focal |
|--------|-----------|--------------|------------------|
| Top-4 recall | 0.9627 | **0.9805** | 0.9585 |
| Top-1 accuracy | 0.5817 | **0.6833** | 0.5793 |
| Log-loss | 1.0475 | **0.8655** | 1.0871 |
| Brier score | 0.5337 | **0.4485** | 0.5499 |
| ECE | **0.0267** | 0.1028 | 0.0871 |

### Pitch Outcome — per-class Top-4 recall

| Class | ① Baseline | ② Focal-only | ③ Weighted focal |
|-------|-----------|--------------|------------------|
| Ball | 0.9795 | 0.9958 | 0.9762 |
| Strike | 0.9577 | 1.0000 | 0.9483 |
| Single | 0.9820 | 0.8998 | **0.9832** |
| Double | 0.3752 | 0.3851 | **0.4778** |
| Triple | 0.0000 | 0.0000 | 0.0000 |
| Home Run | **0.5652** | 0.3170 | 0.4723 |
| Strikeout | 0.9926 | 0.9908 | **0.9960** |
| Walk | 0.9853 | 0.9818 | **0.9910** |
| Hit by Pitch | 0.9757 | 0.9632 | **0.9779** |
| Field Out | 0.9979 | **0.9997** | 0.9979 |
| *Double+HR avg* | *0.4702* | *0.3511* | ***0.4751*** |

### Hit Location — aggregate

| Metric | ① Baseline | ② Focal-only | ③ Weighted focal |
|--------|-----------|--------------|------------------|
| Top-4 recall | 0.6912 | **0.7338** | 0.6869 |
| Top-1 accuracy | 0.2035 | **0.2613** | 0.2023 |
| Log-loss | 2.0492 | **1.9612** | 2.0564 |
| Brier score | 0.8559 | **0.8326** | 0.8564 |
| ECE | 0.0386 | **0.0170** | 0.0321 |

### Hit Location — per-class Top-4 recall

| Class | ① Baseline | ② Focal-only | ③ Weighted focal |
|-------|-----------|--------------|------------------|
| Pitcher | 0.5115 | 0.2027 | **0.5989** |
| Catcher | 0.2221 | 0.0109 | **0.2845** |
| First Base | **0.6612** | 0.4543 | 0.6399 |
| Second Base | 0.6472 | **0.6873** | 0.6400 |
| Third Base | 0.6585 | **0.6660** | 0.6454 |
| Shortstop | 0.6276 | **0.7668** | 0.6251 |
| Left Field | 0.7462 | **0.8680** | 0.7378 |
| Center Field | 0.7515 | **0.9420** | 0.7382 |
| Right Field | 0.7258 | **0.8115** | 0.7261 |
| None | **0.7673** | 0.3951 | 0.7625 |

### Physics — contact pitches (head unchanged across runs)

| Metric | ① Baseline | ② Focal-only | ③ Weighted focal |
|--------|-----------|--------------|------------------|
| EV NLL | 0.2032 | 0.1991 | 0.1952 |
| LA NLL | 1.6356 | 1.6326 | 1.6354 |
| EV MAE (mph) | 9.99 | 9.97 | 9.87 |
| LA MAE (deg) | 20.07 | 20.07 | 20.09 |

## Training logs (②, ③)

**② Focal-only** — resumed at epoch 4; `best.pt` = epoch 4 (val loss never improved after).
```
Epoch 005 | 5986s | train loss 1.6563 (cls 1.8423 phy 1.7756 cal 0.1160) | val loss 1.7053 | val top-4 outcome 0.9806 location 0.7324
Epoch 006 | 5966s | train loss 1.6257 (cls 1.8066 phy 1.7474 cal 0.1160) | val loss 1.7063 | val top-4 outcome 0.9805 location 0.7319
Epoch 007 | 5966s | train loss 1.5971 (cls 1.7734 phy 1.7209 cal 0.1161) | val loss 1.7177 | val top-4 outcome 0.9807 location 0.7291
Early stopping: no improvement for 3 epochs.
```

**③ Weighted focal** — resumed at epoch 5; `best.pt` = epoch 4.
```
Epoch 006 | 6025s | train loss 1.9306 (cls 2.2351 phy 1.7809 cal 0.0985) | val loss 2.0177 | val top-4 outcome 0.9601 location 0.6792
Epoch 007 | 5963s | train loss 1.9376 (cls 2.2414 phy 1.7944 cal 0.0974) | val loss 2.0373 | val top-4 outcome 0.9492 location 0.6801
Early stopping: no improvement for 3 epochs.
```

## Takeaway & decision

The three runs trace a **frequent-vs-rare tradeoff dial**, not a clear improvement:

- **② Focal-only** wins the aggregate scoreboard (outcome top-1 0.683, best log-loss/Brier; location best across the board) — but at the cost of the rare classes it was meant to help: **HR collapses to 0.317**, Single drops to 0.900, and outcome **calibration degrades badly (ECE 0.103)**. The sparse location classes (Pitcher, Catcher, None) also crater.
- **③ Weighted focal** recovers the rare classes — **best Doubles (0.478)** and best on most sparse classes — but its aggregate metrics fall back to roughly the baseline, and HR (0.472) stays below the baseline.
- **① Baseline** has the **best calibration by a wide margin (ECE 0.027)**, the **best Home Run recall (0.565)**, and a rare-class (Double+HR) average (0.470) essentially tied with ③ (0.475) — while ② is far worse (0.351).

The focal + skip additions never broke the rare-class frontier; they only shuffled which rare class did better (HR ↔ Double). Given that **Phase 2 uses this model as an MCTS simulator** — where calibrated outcome probabilities drive the run-expectancy reward, and high-run-value events (HR) matter disproportionately — the baseline's calibration and HR strength make it the soundest choice.

**Triple is effectively irreducible** (0.000 in all three): it depends on ballpark, defense, and baserunning far more than on the pitch.

**Decision: keep ① (baseline) as the final model.** Checkpoint `training/checkpoints_weighted_ece.zip` (old single-width-head architecture). `main` is reverted to that architecture so the checkpoint loads via `evaluate.py` / `run_search.py`; the focal/skip code remains on the `focal-skip-experiment` branch.

### Caveats
- **Confounded comparison.** ② and ③ each changed three things at once vs. the baseline (loss, head width + skip, weighting), so regressions can't be cleanly attributed to a single factor.
- **Resumed, not from-scratch.** ② and ③ were resumed from earlier checkpoints, and because each config shifts the loss scale, the carried-over `best_val_loss` means `best.pt` may not reflect a clean run of that exact config. A rigorous comparison would retrain each config from scratch with a separate `CKPT_DIR` (not done here due to compute limits).

### Re-evaluating the baseline
The baseline checkpoint requires the baseline architecture (current `main` after the revert):
```bash
python -m evaluation.evaluate --checkpoint <path>/best.pt --split test
```
