"""Compute the RE24 (Run Expectancy by 24 base-out states) table from Statcast data.

The 24 states are: (on_1b, on_2b, on_3b) x (outs) = 8 base configs x 3 out states.

Outputs:
  data/re24_table.json
    {
      "re24":       {state_int: expected_runs, ...},
      "run_values": {outcome: run_value_delta, ...}
    }

State encoding:
  state = on_1b + on_2b * 2 + on_3b * 4 + outs * 8     (range 0-23)
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from utils import TRAIN_YEARS, load_statcast_years

RE24_COLUMNS = [
    'game_pk', 'inning', 'inning_topbot',
    'at_bat_number', 'pitch_number',
    'on_1b', 'on_2b', 'on_3b',
    'outs_when_up', 'bat_score', 'post_bat_score',
    'events', 'description',
]

TERMINAL_EVENTS = {
    'single', 'double', 'triple', 'home_run',
    'strikeout', 'strikeout_double_play', 'walk',
    'hit_by_pitch', 'field_out', 'grounded_into_double_play',
    'double_play', 'force_out', 'sac_fly', 'fielders_choice',
    'fielders_choice_out', 'other_out',
}


def base_out_state(on_1b: int, on_2b: int, on_3b: int, outs: int) -> int:
    return int(on_1b) + int(on_2b) * 2 + int(on_3b) * 4 + int(outs) * 8


def decode_state(state: int) -> dict:
    return {
        'on_1b': (state >> 0) & 1,
        'on_2b': (state >> 1) & 1,
        'on_3b': (state >> 2) & 1,
        'outs':  (state >> 3) & 3,
    }


def _prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    """Binarize bases, coerce numerics, compute runs_on_play and state."""
    df = df.copy()
    for base in ['on_1b', 'on_2b', 'on_3b']:
        df[base] = df[base].notna().astype(int)
    df['outs_when_up']   = pd.to_numeric(df['outs_when_up'],   errors='coerce').fillna(0).astype(int)
    df['bat_score']      = pd.to_numeric(df['bat_score'],      errors='coerce').fillna(0)
    df['post_bat_score'] = pd.to_numeric(df['post_bat_score'], errors='coerce').fillna(0)
    df['runs_on_play']   = (df['post_bat_score'] - df['bat_score']).clip(lower=0)
    # Vectorized state encoding (avoids row-by-row apply)
    df['state'] = df['on_1b'] + df['on_2b'] * 2 + df['on_3b'] * 4 + df['outs_when_up'] * 8
    return df


def compute_re24(df: pd.DataFrame) -> dict:
    """RE24 table: state (0-23) -> expected runs to end of half-inning."""
    df = df.sort_values(['game_pk', 'inning', 'inning_topbot', 'at_bat_number', 'pitch_number'])

    half_inning_key = ['game_pk', 'inning', 'inning_topbot']
    df['inning_total_runs'] = df.groupby(half_inning_key)['runs_on_play'].transform('sum')
    df['cum_runs']          = df.groupby(half_inning_key)['runs_on_play'].cumsum() - df['runs_on_play']
    df['runs_remaining']    = (df['inning_total_runs'] - df['cum_runs']).clip(lower=0)

    re24 = df.groupby('state')['runs_remaining'].mean().to_dict()
    for s in range(24):
        re24.setdefault(s, 0.0)
    return re24


def compute_run_values(df: pd.DataFrame, re24: dict) -> dict:
    """Average RE24 delta per terminal event, for MCTS rewards."""
    df = df[df['events'].isin(TERMINAL_EVENTS)].copy()
    df['re_before'] = df['state'].map(re24).fillna(0)

    # outs_after is in {1,2,3}; map to RE24 of empty-base state at that out count
    re_after_map = {o: re24.get(o * 8, 0) for o in range(4)}

    run_values = {}
    for event, subset in df.groupby('events'):
        outs_after = (subset['outs_when_up'] + 1).clip(upper=3)
        re_after   = outs_after.map(re_after_map)
        run_values[event] = float((subset['runs_on_play'] + re_after - subset['re_before']).mean())

    run_values['ball']   = -0.02
    run_values['strike'] =  0.02
    return run_values


def main(
    raw_dir:     str = 'data/raw',
    output_path: str = 'data/re24_table.json',
) -> None:
    print('Loading raw data...')
    df = load_statcast_years(raw_dir, TRAIN_YEARS, columns=RE24_COLUMNS)
    df = _prepare_df(df)

    print('Computing RE24 table...')
    re24 = compute_re24(df)

    print('Computing per-event run values...')
    run_values = compute_run_values(df, re24)

    output = {
        're24':       {str(k): v for k, v in re24.items()},
        'run_values': run_values,
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print('\nRE24 table (runs expected per state):')
    for outs in range(3):
        row = [f'{re24[base_out_state(b1, b2, b3, outs)]:.3f}'
               for b1 in [0, 1] for b2 in [0, 1] for b3 in [0, 1]]
        print(f'  {outs} outs: {row}')
    print(f'\nSaved -> {output_path}')


if __name__ == '__main__':
    main()
