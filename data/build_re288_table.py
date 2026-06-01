"""Compute the RE288 (Run Expectancy by 288 count × base-out states) table.

288 states = 12 count states (0-0 … 3-2) × 24 base-out states.

Key format: "{balls}-{strikes}-{base_out_index}"
  base_out_index = on_1b + on_2b*2 + on_3b*4 + outs*8   (0-23)

Outputs:
  data/artifacts/re288_table.json   {"0-0-0": 0.432, "0-0-1": 0.812, ...}
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from utils import TRAIN_YEARS, load_statcast_years

RE288_COLUMNS = [
    'game_pk', 'inning', 'inning_topbot',
    'at_bat_number', 'pitch_number',
    'on_1b', 'on_2b', 'on_3b',
    'outs_when_up', 'bat_score', 'post_bat_score',
    'balls', 'strikes',
]


def _prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for base in ['on_1b', 'on_2b', 'on_3b']:
        df[base] = df[base].notna().astype(int)
    for col in ['outs_when_up', 'bat_score', 'post_bat_score', 'balls', 'strikes']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    df['balls']       = df['balls'].clip(0, 3)
    df['strikes']     = df['strikes'].clip(0, 2)
    df['outs_when_up'] = df['outs_when_up'].clip(0, 2)
    df['runs_on_play'] = (df['post_bat_score'] - df['bat_score']).clip(lower=0)
    df['base_out'] = (
        df['on_1b'] + df['on_2b'] * 2 + df['on_3b'] * 4 + df['outs_when_up'] * 8
    )
    df['re288_key'] = (
        df['balls'].astype(str) + '-' +
        df['strikes'].astype(str) + '-' +
        df['base_out'].astype(str)
    )
    return df


def compute_re288(df: pd.DataFrame) -> dict:
    df = df.sort_values(
        ['game_pk', 'inning', 'inning_topbot', 'at_bat_number', 'pitch_number']
    )
    half_key = ['game_pk', 'inning', 'inning_topbot']
    df['inning_total_runs'] = df.groupby(half_key)['runs_on_play'].transform('sum')
    df['cum_runs']          = df.groupby(half_key)['runs_on_play'].cumsum() - df['runs_on_play']
    df['runs_remaining']    = (df['inning_total_runs'] - df['cum_runs']).clip(lower=0)
    return df.groupby('re288_key')['runs_remaining'].mean().to_dict()


def main(
    raw_dir:     str = 'data/raw',
    output_path: str = 'data/artifacts/re288_table.json',
) -> None:
    print('Loading raw data...')
    df = load_statcast_years(raw_dir, TRAIN_YEARS, columns=RE288_COLUMNS)
    df = _prepare_df(df)

    print('Computing RE288 table...')
    re288 = compute_re288(df)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(re288, f, indent=2)

    n = len(re288)
    print(f'Saved {n} RE288 entries -> {output_path}')
    print(f'Expected up to 288; missing states default to 0.0 at runtime.')

    print('\nSample (0-0 counts, 0 outs):')
    for base_out in range(8):
        key = f'0-0-{base_out}'
        print(f'  {key}: {re288.get(key, 0.0):.4f}')


if __name__ == '__main__':
    main()
