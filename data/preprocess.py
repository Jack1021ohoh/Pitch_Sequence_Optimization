"""Clean raw Statcast data, map to outcome/location classes,
fit feature scalers and categorical vocabularies, and encode all features.

Run this once on training data before building sequences.
Outputs:
  data/processed/pitches_train.parquet
  data/processed/pitches_val.parquet
  data/processed/pitches_test.parquet
  data/artifacts/scaler.pkl
  data/artifacts/outcome_scaler.pkl
  data/artifacts/vocabs.json
"""

import json
import pickle
import sys
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from utils import TRAIN_YEARS, VAL_YEARS, TEST_YEARS, load_statcast_years

# ---------------------------------------------------------------------------
# Feature column definitions
# ---------------------------------------------------------------------------

CONTINUOUS_COLS = [
    'release_speed', 'release_spin_rate',
    'release_pos_x', 'release_pos_z',
    'pfx_x', 'pfx_z',
    'plate_x', 'plate_z',
    'vx0', 'vy0', 'vz0',
]

# Masked on the final pitch of each sequence (prediction targets)
OUTCOME_CONTINUOUS_COLS = ['launch_speed', 'launch_angle', 'hc_x', 'hc_y']

CATEGORICAL_COLS = [
    'pitch_type', 'pitch_outcome', 'hit_location_class',
    'balls', 'strikes', 'outs_when_up',
    'on_1b', 'on_2b', 'on_3b',
    'inning', 'stand', 'p_throws', 'zone',
]

# ---------------------------------------------------------------------------
# Outcome mappings
# ---------------------------------------------------------------------------

EVENTS_TO_OUTCOME = {
    'single':                    'Single',
    'double':                    'Double',
    'triple':                    'Triple',
    'home_run':                  'Home Run',
    'strikeout':                 'Strikeout',
    'strikeout_double_play':     'Strikeout',
    'walk':                      'Walk',
    'hit_by_pitch':              'Hit by Pitch',
    'field_out':                 'Field Out',
    'grounded_into_double_play': 'Field Out',
    'double_play':               'Field Out',
    'triple_play':               'Field Out',
    'fielders_choice':           'Field Out',
    'fielders_choice_out':       'Field Out',
    'force_out':                 'Field Out',
    'sac_fly':                   'Field Out',
    'sac_fly_double_play':       'Field Out',
    'sac_bunt':                  'Field Out',
    'sac_bunt_double_play':      'Field Out',
    'other_out':                 'Field Out',
    'catcher_interf':            'Field Out',
}

DESCRIPTION_TO_OUTCOME = {
    'ball':                    'Ball',
    'blocked_ball':            'Ball',
    'pitchout':                'Ball',
    'called_strike':           'Strike',
    'swinging_strike':         'Strike',
    'swinging_strike_blocked': 'Strike',
    'foul':                    'Strike',
    'foul_tip':                'Strike',
    'foul_bunt':               'Strike',
    'missed_bunt':             'Strike',
    'hit_by_pitch':            'Hit by Pitch',
}

OUTCOME_CLASSES = [
    'Ball', 'Strike', 'Single', 'Double', 'Triple',
    'Home Run', 'Strikeout', 'Walk', 'Hit by Pitch', 'Field Out',
]

HIT_LOCATION_MAP = {
    1: 'Pitcher',    2: 'Catcher',     3: 'First Base',
    4: 'Second Base', 5: 'Third Base', 6: 'Shortstop',
    7: 'Left Field', 8: 'Center Field', 9: 'Right Field',
}
# float keys because Statcast stores hit_location as float when NaNs are present
_FLOAT_HIT_LOCATION_MAP = {float(k): v for k, v in HIT_LOCATION_MAP.items()}

HIT_LOCATION_CLASSES = list(HIT_LOCATION_MAP.values()) + ['None']

VALID_PITCH_TYPES = {'FF', 'SL', 'CH', 'CU', 'SI', 'FC', 'FS', 'ST', 'KC', 'SV', 'CS', 'FO'}

# ---------------------------------------------------------------------------
# Data loading and cleaning
# ---------------------------------------------------------------------------

def filter_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows that cannot be used for modeling."""
    required = ['batter', 'pitcher', 'game_pk', 'at_bat_number',
                 'pitch_number', 'game_date', 'pitch_type']
    df = df.dropna(subset=required)
    df = df[df['pitch_type'].isin(VALID_PITCH_TYPES)]
    df = df[~df['description'].isin([
        'pickoff', 'caught_stealing_2b', 'caught_stealing_3b',
        'caught_stealing_home', 'pickoff_1b', 'pickoff_2b', 'pickoff_3b',
    ])]
    df = df[~df['events'].isin(['intent_walk', 'field_error'])]

    # Vectorized outcome mapping: events take priority over description
    outcome = df['events'].map(EVENTS_TO_OUTCOME)
    no_event = outcome.isna()
    outcome = outcome.where(~no_event, df['description'].map(DESCRIPTION_TO_OUTCOME))
    df = df.assign(pitch_outcome=outcome)
    df = df[df['pitch_outcome'].notna()]

    df = df[df['inning'].between(1, 15)]
    return df.reset_index(drop=True)


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add outcome class columns, clean base occupancy and count fields."""
    df = df.copy()

    df['hit_location_class'] = (
        pd.to_numeric(df['hit_location'], errors='coerce')
        .map(_FLOAT_HIT_LOCATION_MAP)
        .fillna('None')
    )

    for base in ['on_1b', 'on_2b', 'on_3b']:
        df[base] = df[base].notna().astype(int)

    df['inning'] = df['inning'].clip(upper=10).astype(int)
    df['score_differential'] = (df['fld_score'] - df['bat_score']).fillna(0).astype(int)

    for col in ['balls', 'strikes', 'outs_when_up']:
        df[col] = df[col].fillna(0).astype(int)

    df['zone'] = pd.to_numeric(df['zone'], errors='coerce').fillna(0).clip(0, 14).astype(int)
    df['game_date'] = pd.to_datetime(df['game_date'])
    df = df.sort_values(['batter', 'game_date', 'game_pk', 'at_bat_number', 'pitch_number'])
    return df.reset_index(drop=True)

# ---------------------------------------------------------------------------
# Artifact fitting
# ---------------------------------------------------------------------------

def fit_scaler(df: pd.DataFrame, cols: list) -> StandardScaler:
    present = [c for c in cols if c in df.columns]
    scaler = StandardScaler()
    scaler.fit(df[present].fillna(0))
    return scaler


def build_vocabs() -> dict:
    vocabs = {}
    vocabs['pitch_type'] = {pt: i + 1 for i, pt in enumerate(sorted(VALID_PITCH_TYPES))}
    vocabs['pitch_type']['<UNK>'] = 0
    vocabs['pitch_outcome']      = {cls: i for i, cls in enumerate(OUTCOME_CLASSES)}
    vocabs['hit_location_class'] = {cls: i for i, cls in enumerate(HIT_LOCATION_CLASSES)}
    vocabs['balls']        = {i: i for i in range(5)}
    vocabs['strikes']      = {i: i for i in range(4)}
    vocabs['outs_when_up'] = {i: i for i in range(4)}
    vocabs['inning']       = {i: i for i in range(1, 11)}
    vocabs['on_1b']        = {0: 0, 1: 1}
    vocabs['on_2b']        = {0: 0, 1: 1}
    vocabs['on_3b']        = {0: 0, 1: 1}
    vocabs['stand']        = {'L': 0, 'R': 1, 'S': 2}
    vocabs['p_throws']     = {'L': 0, 'R': 1}
    vocabs['zone']         = {i: i for i in range(15)}
    return vocabs


def save_artifacts(scaler, outcome_scaler, vocabs, artifact_dir: str) -> None:
    out = Path(artifact_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / 'scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open(out / 'outcome_scaler.pkl', 'wb') as f:
        pickle.dump(outcome_scaler, f)
    json_vocabs = {col: {str(k): v for k, v in mapping.items()} for col, mapping in vocabs.items()}
    with open(out / 'vocabs.json', 'w') as f:
        json.dump(json_vocabs, f, indent=2)
    print(f'Artifacts saved -> {out}')


def load_artifacts(artifact_dir: str) -> tuple:
    out = Path(artifact_dir)
    with open(out / 'scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open(out / 'outcome_scaler.pkl', 'rb') as f:
        outcome_scaler = pickle.load(f)
    with open(out / 'vocabs.json') as f:
        raw = json.load(f)
    vocabs = {}
    for col, mapping in raw.items():
        restored = {}
        for k, v in mapping.items():
            try:
                restored[int(k)] = v
            except ValueError:
                restored[k] = v
        vocabs[col] = restored
    return scaler, outcome_scaler, vocabs

# ---------------------------------------------------------------------------
# Encoding and saving
# ---------------------------------------------------------------------------

def encode_and_save(df: pd.DataFrame, scaler, outcome_scaler, vocabs, output_path: str) -> None:
    df = df.copy()

    feat_df = pd.DataFrame(
        scaler.transform(df[CONTINUOUS_COLS].fillna(0)),
        columns=[f'feat_{c}' for c in CONTINUOUS_COLS],
        index=df.index,
    )
    outcome_cols = [c for c in OUTCOME_CONTINUOUS_COLS if c in df.columns]
    outcome_df = pd.DataFrame(
        outcome_scaler.transform(df[outcome_cols].fillna(0)),
        columns=[f'feat_{c}' for c in outcome_cols],
        index=df.index,
    )
    cat_df = pd.DataFrame(
        {f'cat_{col}': df[col].map(vocabs.get(col, {})).fillna(0).astype(int)
         for col in CATEGORICAL_COLS},
        index=df.index,
    )
    df = pd.concat([df, feat_df, outcome_df, cat_df], axis=1)

    df['pitch_outcome_mask'] = 0
    df['hit_location_mask']  = 0

    # Labels use -1 for unknowns (invalid training targets); cat_ columns use 0 (valid masked input)
    df['pitch_outcome_label'] = df['pitch_outcome'].map(vocabs['pitch_outcome']).fillna(-1).astype(int)
    df['hit_location_label']  = df['hit_location_class'].map(vocabs['hit_location_class']).fillna(-1).astype(int)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f'Saved {len(df):,} rows -> {output_path}')

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _load_split(raw_dir: str, years: list) -> pd.DataFrame:
    df = load_statcast_years(raw_dir, years)
    df = filter_data(df)
    return add_derived_columns(df)


def main(
    raw_dir:      str = 'data/raw',
    output_dir:   str = 'data/processed',
    artifact_dir: str = 'data/artifacts',
) -> None:
    print('Loading training data...')
    df_train = _load_split(raw_dir, TRAIN_YEARS)

    print('Fitting artifacts on training data...')
    outcome_cols = [c for c in OUTCOME_CONTINUOUS_COLS if c in df_train.columns]
    scaler         = fit_scaler(df_train, CONTINUOUS_COLS)
    outcome_scaler = fit_scaler(df_train, outcome_cols)
    vocabs         = build_vocabs()
    save_artifacts(scaler, outcome_scaler, vocabs, artifact_dir)

    print('Encoding train...')
    encode_and_save(df_train, scaler, outcome_scaler, vocabs, f'{output_dir}/pitches_train.parquet')
    del df_train

    print('Encoding val...')
    df_val = _load_split(raw_dir, VAL_YEARS)
    encode_and_save(df_val, scaler, outcome_scaler, vocabs, f'{output_dir}/pitches_val.parquet')
    del df_val

    print('Encoding test...')
    df_test = _load_split(raw_dir, TEST_YEARS)
    encode_and_save(df_test, scaler, outcome_scaler, vocabs, f'{output_dir}/pitches_test.parquet')
    del df_test

    print('Done.')


if __name__ == '__main__':
    main()
