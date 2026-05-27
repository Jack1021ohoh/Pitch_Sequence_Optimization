# Pitch Sequence Transformer

An improved transformer-based model for MLB pitch outcome prediction and optimal pitch sequencing, extending the 42 Analytics paper [*Transformer-Based Baseball Modeling for Pitch Outcome Prediction and Strategy Optimization*](https://www.sloansportsconference.com/research-papers/transformer-based-baseball-modeling-for-pitch-outcome-prediction-and-strategy-optimization).

## What's New vs. the Paper

| # | Limitation in Paper | Our Fix |
|---|---|---|
| 1 | Pitcher absent from model | Hierarchical pitcher encoder (appearance-based) |
| 2 | Starter/reliever asymmetry ignored | Appearance-level tokenization handles it implicitly |
| 3 | Continuous predictions (EV, LA) failed | Conditional mixture-of-Gaussians physics head |
| 4 | One-hot categorical encoding | Learned embeddings per categorical variable |
| 5 | Greedy single-pitch selection | MCTS with RE24 rewards for multi-step planning (Phase 2) |

## Architecture

```
Batter Sequence (400 pitches)     Pitcher Sequence (K appearances)
         │                                      │
         ▼                                      ▼
  Batter Encoder                  Hierarchical Pitcher Encoder
  (12-layer Transformer)          (pitch-level → appearance-level)
         │                                      │
         └──────────────┬───────────────────────┘
                        ▼
               Cross-Attention Layer
               (Batter ← Pitcher)
                        │
            ┌───────────┴───────────┐
            ▼                       ▼
   Classification Head      Physics Regression Head
   - Pitch outcome (10)     - Exit velocity (MoG)
   - Hit location (10)      - Launch angle (MoG)
                             (conditional on contact)
```

## Phases

- **Phase 1** *(implemented)* — Train the transformer: batter encoder + hierarchical pitcher encoder + cross-attention + classification and physics heads.
- **Phase 2** *(implemented)* — MCTS pitch sequencer using the Phase 1 model as a simulator. No additional training required.

## Setup

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1       # Windows
# source .venv/bin/activate      # macOS / Linux
pip install -r requirements.txt
# PyTorch is pre-installed on Colab; for local use see requirements.txt
```

## Data Pipeline

Run in order from the project root:

```bash
python data/fetch_statcast.py             # download raw Statcast data (2015–2025)
python data/preprocess.py                 # clean, encode, save train/val/test parquets
python data/build_batter_sequences.py     # sliding 400-pitch windows per batter
python data/build_pitcher_appearances.py  # per-game appearance records per pitcher
python data/build_re24_table.py           # RE24 run expectancy table (Phase 2 only)
python data/build_pitch_library.py        # per-pitcher (pitch_type, zone) feature lookup (Phase 2 only)
```

See [`data/README.md`](data/README.md) for details on each script.

## Training

```bash
# Local
python -m training.train --epochs 20 --batch-size 64 --workers 0

# Colab
import os
os.environ['DATA_DIR'] = '/content/drive/MyDrive/pitch_sequence/data'
os.environ['CKPT_DIR'] = '/content/drive/MyDrive/pitch_sequence/checkpoints'
%run training/train.py --workers 2
```

Checkpoints are saved to `CKPT_DIR` (`best.pt` by val top-4 outcome precision, `latest.pt` each epoch).

See [`training/README.md`](training/README.md) for all options and the loss function.

## Evaluation

```bash
python -m evaluation.evaluate --checkpoint checkpoints/best.pt --split test
```

Reports top-4 precision, log-loss, Brier score, and ECE for pitch outcome and hit location; MAE and NLL for exit velocity and launch angle on contact pitches.

See [`evaluation/README.md`](evaluation/README.md) for metric definitions.

## MCTS Pitch Sequencer (Phase 2)

```bash
python -m mcts.run_search --checkpoint checkpoints/best.pt \
                           --batter-id 592450 --pitcher-id 605483 \
                           --n-iter 500 --top-k 10
```

Omit `--batter-id` / `--pitcher-id` to sample a random at-bat from `--split` (default: test).
Outputs a ranked table of `(pitch_type, zone)` actions by visit count, Q-value, and share.

See [`mcts/README.md`](mcts/README.md) for algorithm details and all options.

## Data

- **Source:** MLB Statcast via [pybaseball](https://github.com/jldbc/pybaseball)
- **Split:** Train 2021–2023 | Val 2024 | Test 2025 (limited to 2021–2025 due to computing constraints)
- **Storage:** Data files are excluded from this repo (see `.gitignore`). Store on Google Drive for Colab access.

```python
from google.colab import drive
drive.mount('/content/drive')
DATA_DIR = '/content/drive/MyDrive/pitch_sequence/data'
```

## File Structure

```
pitch_sequence/
├── data/
│   ├── fetch_statcast.py
│   ├── preprocess.py
│   ├── build_batter_sequences.py
│   ├── build_pitcher_appearances.py
│   ├── build_re24_table.py
│   └── utils.py
├── models/
│   ├── embeddings.py        # PitchEmbedding (shared)
│   ├── batter_encoder.py    # 12-layer transformer
│   ├── pitcher_encoder.py   # hierarchical pitcher encoder
│   └── full_model.py        # assembled model + run_model helper
├── training/
│   ├── dataset.py           # PitchSequenceDataset
│   ├── loss.py              # MultiTaskLoss
│   └── train.py             # training loop
├── evaluation/
│   ├── metrics.py           # all metric functions
│   └── evaluate.py          # evaluation script
├── mcts/
│   ├── state.py             # AtBatState dataclass + outcome transitions
│   ├── simulator.py         # PitchSimulator: model wrapper for single-pitch rollouts
│   ├── node.py              # MCTSNode with UCB1 selection
│   ├── search.py            # UCT search loop (selection → expansion → rollout → backprop)
│   └── run_search.py        # CLI entry point
├── requirements.txt
└── README.md
```
