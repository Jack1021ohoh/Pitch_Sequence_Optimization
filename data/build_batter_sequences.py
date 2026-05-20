"""Build per-batter pitch history files and a sequence index for training.

For each batter with >= MIN_PITCHES pitches, we:
  1. Save their full pitch history as a numpy array (one row per pitch).
  2. Record every valid sequence endpoint in a global index.

At training time the Dataset loads batter_pitches/{batter_id}.npy and
slices out a 400-pitch window ending at the indexed row.

Outputs:
  data/sequences/batter_pitches/{batter_id}.npy   -- shape (T, F)
  data/sequences/batter_index.parquet             -- (batter_id, end_row, split)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from utils import SPLITS, load_statcast_years
from preprocess import CONTINUOUS_COLS, OUTCOME_CONTINUOUS_COLS, CATEGORICAL_COLS

SEQ_LEN     = 400
MIN_PITCHES = SEQ_LEN

# Derive prefixed column names from the canonical lists in preprocess.py
FEAT_CONTINUOUS         = [f'feat_{c}' for c in CONTINUOUS_COLS]
FEAT_OUTCOME_CONTINUOUS = [f'feat_{c}' for c in OUTCOME_CONTINUOUS_COLS]
FEAT_CATEGORICAL        = [f'cat_{c}'  for c in CATEGORICAL_COLS]
FEAT_MASK               = ['pitch_outcome_mask', 'hit_location_mask']
LABEL_COLS              = ['pitch_outcome_label', 'hit_location_label']

# Column order written into each .npy file; Dataset applies masking at load time
ALL_COLS = FEAT_CONTINUOUS + FEAT_OUTCOME_CONTINUOUS + FEAT_CATEGORICAL + FEAT_MASK + LABEL_COLS


def build_batter_sequences(
    processed_dir: str = 'data/processed',
    output_dir:    str = 'data/sequences',
    min_pitches:   int = MIN_PITCHES,
) -> None:
    pitch_dir = Path(output_dir) / 'batter_pitches'
    pitch_dir.mkdir(parents=True, exist_ok=True)

    sort_cols = ['batter', 'game_date', 'game_pk', 'at_bat_number', 'pitch_number']
    needed_cols = sort_cols + ALL_COLS

    frames = []
    for split in SPLITS:
        p = Path(processed_dir) / f'pitches_{split}.parquet'
        if not p.exists():
            print(f'Warning: {p} not found, skipping.')
            continue
        df = pd.read_parquet(p, columns=needed_cols)
        df['split'] = split
        frames.append(df)

    if not frames:
        raise FileNotFoundError(f'No processed parquet files found in {processed_dir}.')

    df_all = pd.concat(frames, ignore_index=True)
    df_all['game_date'] = pd.to_datetime(df_all['game_date'])
    df_all = df_all.sort_values(
        ['batter', 'game_date', 'game_pk', 'at_bat_number', 'pitch_number']
    ).reset_index(drop=True)

    missing_cols = [c for c in ALL_COLS if c not in df_all.columns]
    if missing_cols:
        print(f'Warning: filling missing columns with 0: {missing_cols}')
        for c in missing_cols:
            df_all[c] = 0

    index_records = []

    for batter_id, batter_df in df_all.groupby('batter', sort=False):
        batter_df = batter_df.reset_index(drop=True)
        T = len(batter_df)
        if T < min_pitches:
            continue

        np.save(pitch_dir / f'{batter_id}.npy', batter_df[ALL_COLS].to_numpy(dtype=np.float32))

        # Extract to Python lists before the loop to avoid per-row .loc overhead
        dates  = batter_df['game_date'].tolist()
        splits = batter_df['split'].tolist()
        years  = batter_df['game_date'].dt.year.tolist()
        index_records.extend(
            {'batter_id': batter_id, 'end_row': end_row,
             'game_date': dates[end_row], 'split': splits[end_row]}
            for end_row in range(min_pitches - 1, T)
            # Exclude windows that span an off-season gap
            if years[end_row - min_pitches + 1] == years[end_row]
        )

    index_df  = pd.DataFrame(index_records)
    index_path = Path(output_dir) / 'batter_index.parquet'
    index_df.to_parquet(index_path, index=False)

    for split in SPLITS:
        n = (index_df['split'] == split).sum()
        print(f'  {split}: {n:,} sequences')
    print(f'Batter index saved -> {index_path}')
    print(f'Per-batter arrays  -> {pitch_dir}')


if __name__ == '__main__':
    build_batter_sequences()
