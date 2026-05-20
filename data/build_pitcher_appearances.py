"""Build a per-pitcher appearance dataset for the hierarchical pitcher encoder.

An "appearance" is all pitches thrown by a pitcher in a single game.
For each appearance we store:
  - A pitch array of shape (num_pitches, num_features)
  - A context vector of shape (7,) with appearance-level features

Outputs:
  data/sequences/pitcher_appearances.pkl
    dict: { pitcher_id (int) -> list of AppearanceRecord (sorted by date) }

Each AppearanceRecord is a dict:
  {
    'game_date':  pd.Timestamp,
    'game_pk':    int,
    'pitches':    np.ndarray  shape (T, F),
    'context':    np.ndarray  shape (7,),
  }

Context vector layout (7 scalars):
  [0] days_since_last_appearance
  [1] pitches_thrown_last_3_days
  [2] appearances_last_7_days
  [3] inning_entered
  [4] score_differential
  [5] outs_when_entered
  [6] runners_on_when_entered
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from utils import SPLITS
from preprocess import CONTINUOUS_COLS, OUTCOME_CONTINUOUS_COLS, CATEGORICAL_COLS

# Pitch-level features per appearance; outcomes visible (all pitches are historical)
PITCH_FEAT_COLS = (
    [f'feat_{c}' for c in CONTINUOUS_COLS] +
    [f'feat_{c}' for c in OUTCOME_CONTINUOUS_COLS] +
    [f'cat_{c}'  for c in CATEGORICAL_COLS]
)


def _compute_context(
    appearance_df:   pd.DataFrame,
    pitcher_history: list,
    game_date:       pd.Timestamp,
) -> np.ndarray:
    # Cap at 30: any gap longer than a month is effectively "start of season"
    days_since = min((game_date - pitcher_history[-1]['game_date']).days, 30) if pitcher_history else 0

    cutoff_3d = game_date - pd.Timedelta(days=3)
    cutoff_7d = game_date - pd.Timedelta(days=7)
    pitches_last_3d = 0
    apps_last_7d    = 0
    for rec in reversed(pitcher_history):
        if rec['game_date'] <= cutoff_7d:
            break
        apps_last_7d += 1
        if rec['game_date'] > cutoff_3d:
            pitches_last_3d += len(rec['pitches'])

    first = appearance_df.iloc[0]
    runners_on = int(first['cat_on_1b'] > 0) + int(first['cat_on_2b'] > 0) + int(first['cat_on_3b'] > 0)

    return np.array([
        days_since,
        pitches_last_3d,
        apps_last_7d,
        int(first['inning']),
        int(first['score_differential']),
        int(first['outs_when_up']),
        runners_on,
    ], dtype=np.float32)


def build_pitcher_appearances(
    processed_dir: str = 'data/processed',
    output_dir:    str = 'data/sequences',
) -> None:
    sort_cols = ['pitcher', 'game_date', 'game_pk', 'at_bat_number', 'pitch_number']
    extra_cols = ['inning', 'score_differential', 'outs_when_up']
    needed_cols = sort_cols + extra_cols + PITCH_FEAT_COLS

    frames = []
    for split in SPLITS:
        p = Path(processed_dir) / f'pitches_{split}.parquet'
        if not p.exists():
            print(f'Warning: {p} not found, skipping.')
            continue
        frames.append(pd.read_parquet(p, columns=needed_cols))

    if not frames:
        raise FileNotFoundError(f'No processed parquet files found in {processed_dir}.')

    df_all = pd.concat(frames, ignore_index=True)
    df_all['game_date'] = pd.to_datetime(df_all['game_date'])
    df_all = df_all.sort_values(sort_cols).reset_index(drop=True)

    missing = [c for c in PITCH_FEAT_COLS if c not in df_all.columns]
    for c in missing:
        df_all[c] = 0.0
    if 'score_differential' not in df_all.columns:
        df_all['score_differential'] = 0

    appearances: dict[int, list] = {}

    for pitcher_id, pitcher_df in df_all.groupby('pitcher', sort=False):
        pitcher_df = pitcher_df.reset_index(drop=True)
        history: list[dict] = []

        # pitcher_df is already sorted by game_date; groupby(sort=False) preserves order
        for (game_pk, game_date), game_df in pitcher_df.groupby(
            ['game_pk', 'game_date'], sort=False
        ):
            game_date = pd.Timestamp(game_date)
            record = {
                'game_date': game_date,
                'game_pk':   int(game_pk),
                'pitches':   game_df[PITCH_FEAT_COLS].to_numpy(dtype=np.float32),
                'context':   _compute_context(game_df, history, game_date),
            }
            history.append(record)

        appearances[int(pitcher_id)] = history

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(output_dir) / 'pitcher_appearances.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump(appearances, f)

    total_apps = sum(len(v) for v in appearances.values())
    print(f'Saved {len(appearances):,} pitchers, {total_apps:,} total appearances -> {output_path}')


if __name__ == '__main__':
    build_pitcher_appearances()
