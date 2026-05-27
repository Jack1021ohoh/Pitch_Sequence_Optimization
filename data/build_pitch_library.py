"""Build per-pitcher pitch feature library for MCTS action space.

For each (pitcher, pitch_type, zone) seen in training, computes the mean
standardized continuous feature vector. Used by the MCTS simulator to convert
a (pitch_type, zone) action into a model-ready 11-dim feature vector.

Run after preprocess.py:
  python data/build_pitch_library.py

Output: data/artifacts/pitch_library.pkl
  {pitcher_id (int) -> {pitch_type (str) -> {zone (int) -> np.ndarray(11,),
                                             '_mean'     -> np.ndarray(11,)}}}

'_mean' is the pitch-type average across all zones — used as a fallback when
a pitcher has not thrown a specific (pitch_type, zone) combination enough times.
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.preprocess import CONTINUOUS_COLS

MIN_PITCHES_PER_TYPE = 20  # drop (pitcher, pitch_type) pairs below this threshold
MIN_PITCHES_PER_ZONE = 3   # zones below this fall back to '_mean'

FEAT_COLS = [f'feat_{c}' for c in CONTINUOUS_COLS]


def build_pitch_library(processed_path: str) -> dict:
    print(f'Loading {processed_path}...')
    df = pd.read_parquet(
        processed_path,
        columns=['pitcher', 'pitch_type', 'zone'] + FEAT_COLS,
    )
    df = df.dropna(subset=FEAT_COLS)
    df['pitcher'] = df['pitcher'].astype(int)
    df['zone']    = df['zone'].astype(int)

    library = {}
    for (pitcher_id, pitch_type), pt_group in df.groupby(['pitcher', 'pitch_type']):
        if len(pt_group) < MIN_PITCHES_PER_TYPE:
            continue
        pitcher_id = int(pitcher_id)
        if pitcher_id not in library:
            library[pitcher_id] = {}

        pt_mean = pt_group[FEAT_COLS].mean().values.astype(np.float32)
        zones   = {'_mean': pt_mean}
        for zone, z_group in pt_group.groupby('zone'):
            if len(z_group) >= MIN_PITCHES_PER_ZONE:
                zones[int(zone)] = z_group[FEAT_COLS].mean().values.astype(np.float32)

        library[pitcher_id][pitch_type] = zones

    return library


def main(
    processed_path: str = 'data/processed/pitches_train.parquet',
    artifact_dir:   str = 'data/artifacts',
) -> None:
    library = build_pitch_library(processed_path)

    n_pitchers  = len(library)
    n_pt_combos = sum(len(pts) for pts in library.values())
    n_zones     = sum(
        len(zones) - 1  # subtract the '_mean' entry
        for pts in library.values()
        for zones in pts.values()
    )
    print(f'Library: {n_pitchers:,} pitchers | '
          f'{n_pt_combos:,} pitch-type entries | '
          f'{n_zones:,} zone-specific entries')

    out = Path(artifact_dir) / 'pitch_library.pkl'
    with open(out, 'wb') as f:
        pickle.dump(library, f)
    print(f'Saved -> {out}')


if __name__ == '__main__':
    main()
