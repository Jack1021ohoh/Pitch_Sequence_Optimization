# Pitch Sequence Transformer

An improved transformer-based model for MLB pitch outcome prediction and optimal pitch sequencing, extending the 42 Analytics paper *"Transformer-Based Baseball Modeling for Pitch Outcome Prediction and Strategy Optimization"*.

## What's New vs. the Paper

| # | Limitation in Paper | Our Fix |
|---|---|---|
| 1 | Pitcher absent from model | Hierarchical pitcher encoder (appearance-based) |
| 2 | Starter/reliever asymmetry ignored | Appearance-level tokenization handles it implicitly |
| 3 | Continuous predictions (EV, LA) failed | Conditional mixture-of-Gaussians physics head |
| 4 | One-hot categorical encoding | Learned embeddings per categorical variable |
| 5 | Greedy single-pitch selection | MCTS with RE24 rewards for multi-step planning |

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
   - Hit location (9)       - Launch angle (MoG)
                             (conditional on contact)
```

## Phases

- **Phase 1** — Train the transformer model (batter encoder + pitcher encoder + heads).
- **Phase 2** — MCTS pitch sequencer using the Phase 1 model as a simulator. No additional training required.

## Setup

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1       # Windows
# source .venv/bin/activate      # macOS / Linux
pip install -r requirements.txt
```

## Data Pipeline

Run in order from the project root:

```bash
python data/fetch_statcast.py          # download raw Statcast data (2015–2025)
python data/preprocess.py              # clean, encode, save train/val/test parquets
python data/build_batter_sequences.py  # sliding 400-pitch windows per batter
python data/build_pitcher_appearances.py  # per-game appearance records per pitcher
python data/build_re24_table.py        # RE24 run expectancy table (Phase 2 only)
```

See [`data/README.md`](data/README.md) for details on each script.

## Data

- **Source:** MLB Statcast via [pybaseball](https://github.com/jldbc/pybaseball)
- **Split:** Train 2015–2022 | Val 2023 | Test 2024–2025
- **Storage:** Data files are excluded from this repo (see `.gitignore`). Store processed data on Google Drive for Colab access.

```python
# Colab mount
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
├── models/          # (Phase 1 — to be implemented)
├── training/        # (Phase 1 — to be implemented)
├── mcts/            # (Phase 2 — to be implemented)
├── evaluation/      # (to be implemented)
├── requirements.txt
└── README.md
```
