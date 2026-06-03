"""LightGBM baseline for pitch outcome and hit location prediction.

Trains two models on per-pitch features (no sequential context):
  1. Pitch outcome classifier  — 10 classes, all pitches
  2. Hit location classifier   — 10 classes, contact pitches only

Evaluates with top-4 precision per class, matching the 42 Analytics paper metric
(Tables 1 & 2).  Models saved to $BASELINE_DIR/lgbm_outcome.txt and
$BASELINE_DIR/lgbm_location.txt.

Usage:
    python baselines/train_lightgbm.py
    DATA_DIR=data BASELINE_DIR=baselines python baselines/train_lightgbm.py

    # Override paths (e.g. Colab + Drive):
    import os
    os.environ['DATA_DIR']     = '/content/drive/MyDrive/pitch_sequence/data'
    os.environ['BASELINE_DIR'] = '/content/drive/MyDrive/pitch_sequence/baselines'
    %run baselines/train_lightgbm.py
"""

import json
import os
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OUTCOME_CLASSES = [
    'Ball', 'Strike', 'Single', 'Double', 'Triple',
    'Home Run', 'Strikeout', 'Walk', 'Hit by Pitch', 'Field Out',
]
HIT_LOCATION_CLASSES = [
    'Pitcher', 'Catcher', 'First Base', 'Second Base', 'Third Base',
    'Shortstop', 'Left Field', 'Center Field', 'Right Field', 'None',
]

# Continuous pitch-physics features available at throw time
CONT_FEATURES = [
    'feat_release_speed', 'feat_release_spin_rate',
    'feat_release_pos_x', 'feat_release_pos_z',
    'feat_pfx_x', 'feat_pfx_z',
    'feat_plate_x', 'feat_plate_z',
    'feat_vx0', 'feat_vy0', 'feat_vz0',
]
# Game-state + pitch-type categoricals (exclude cat_pitch_outcome and
# cat_hit_location_class — they directly encode the targets)
CAT_FEATURES = [
    'cat_pitch_type', 'cat_balls', 'cat_strikes', 'cat_outs_when_up',
    'cat_on_1b', 'cat_on_2b', 'cat_on_3b',
    'cat_inning', 'cat_stand', 'cat_p_throws', 'cat_zone',
]
ALL_FEATURES = CONT_FEATURES + CAT_FEATURES

DATA_DIR     = os.environ.get('DATA_DIR', 'data')
TRAIN_PATH   = f'{DATA_DIR}/processed/pitches_train.parquet'
VAL_PATH     = f'{DATA_DIR}/processed/pitches_val.parquet'
TEST_PATH    = f'{DATA_DIR}/processed/pitches_test.parquet'
WEIGHTS_PATH = f'{DATA_DIR}/artifacts/class_weights.json'
OUTPUT_DIR   = os.environ.get('BASELINE_DIR', 'baselines')

LGB_PARAMS = dict(
    num_leaves=127,
    learning_rate=0.05,
    n_estimators=500,
    min_child_samples=20,
    random_state=42,
    verbose=-1,
)

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_outcome_split(path: str) -> tuple[pd.DataFrame, pd.Series]:
    cols = ALL_FEATURES + ['pitch_outcome_label']
    df = pd.read_parquet(path, columns=cols)
    df = df[df['pitch_outcome_label'] != -1]
    return df[ALL_FEATURES], df['pitch_outcome_label'].astype(int)


def load_location_split(path: str) -> tuple[pd.DataFrame, pd.Series]:
    cols = ALL_FEATURES + ['hit_location_label']
    df = pd.read_parquet(path, columns=cols)
    # hit_location_label is already -1 for non-contact pitches (set in preprocess.py)
    df = df[df['hit_location_label'] != -1]
    return df[ALL_FEATURES], df['hit_location_label'].astype(int)


def make_sample_weights(y: pd.Series, class_weights: list[float]) -> np.ndarray:
    w = np.array(class_weights)
    return w[y.values]

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def top_k_recall_per_class(
    y_true: np.ndarray,
    proba: np.ndarray,
    k: int = 4,
) -> dict[int, float]:
    top_k = np.argsort(proba, axis=1)[:, -k:]
    results = {}
    for c in range(proba.shape[1]):
        mask = y_true == c
        if mask.sum() == 0:
            results[c] = float('nan')
        else:
            results[c] = float((top_k[mask] == c).any(axis=1).mean())
    return results


def print_recall_table(
    recalls: dict[int, float],
    class_names: list[str],
    title: str,
) -> None:
    print(f'\n{title}')
    print(f'  {"Class":<18} {"Top-4 Recall":>16}')
    print('  ' + '-' * 36)
    for i, name in enumerate(class_names):
        val = recalls.get(i, float('nan'))
        val_str = f'{val:.3f}' if not np.isnan(val) else '   N/A'
        print(f'  {name:<18} {val_str:>16}')

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    X_tr: pd.DataFrame, y_tr: pd.Series, w_tr: np.ndarray,
    X_val: pd.DataFrame, y_val: pd.Series, w_val: np.ndarray,
    label: str,
) -> lgb.LGBMClassifier:
    model = lgb.LGBMClassifier(**LGB_PARAMS)
    print(f'\nTraining {label} model  '
          f'(train={len(y_tr):,}  val={len(y_val):,})')
    model.fit(
        X_tr, y_tr,
        sample_weight=w_tr,
        eval_set=[(X_val, y_val)],
        eval_sample_weight=[w_val],
        categorical_feature=CAT_FEATURES,
        callbacks=[
            lgb.early_stopping(stopping_rounds=20, verbose=True),
            lgb.log_evaluation(period=50),
        ],
    )
    return model

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    Path(OUTPUT_DIR).mkdir(exist_ok=True)

    with open(WEIGHTS_PATH) as f:
        class_weights = json.load(f)

    # ------------------------------------------------------------------ #
    # Model 1: Pitch outcome                                               #
    # ------------------------------------------------------------------ #
    print('Loading outcome data...')
    X_tr,  y_tr  = load_outcome_split(TRAIN_PATH)
    X_val, y_val = load_outcome_split(VAL_PATH)

    w_tr  = make_sample_weights(y_tr,  class_weights['outcome'])
    w_val = make_sample_weights(y_val, class_weights['outcome'])

    outcome_model = train(X_tr, y_tr, w_tr, X_val, y_val, w_val, 'outcome')

    out_path = f'{OUTPUT_DIR}/lgbm_outcome.txt'
    outcome_model.booster_.save_model(out_path)
    print(f'Saved -> {out_path}')

    # ------------------------------------------------------------------ #
    # Model 2: Hit location (contact pitches only)                        #
    # ------------------------------------------------------------------ #
    print('\nLoading hit location data...')
    X_tr_l,  y_tr_l  = load_location_split(TRAIN_PATH)
    X_val_l, y_val_l = load_location_split(VAL_PATH)

    w_tr_l  = make_sample_weights(y_tr_l,  class_weights['location'])
    w_val_l = make_sample_weights(y_val_l, class_weights['location'])

    location_model = train(
        X_tr_l, y_tr_l, w_tr_l,
        X_val_l, y_val_l, w_val_l,
        'hit location',
    )

    loc_path = f'{OUTPUT_DIR}/lgbm_location.txt'
    location_model.booster_.save_model(loc_path)
    print(f'Saved -> {loc_path}')


if __name__ == '__main__':
    main()
