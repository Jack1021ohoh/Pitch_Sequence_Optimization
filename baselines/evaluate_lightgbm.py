"""Evaluate saved LightGBM baseline models on the test set.

Loads lgbm_outcome.txt and lgbm_location.txt from $BASELINE_DIR and
reports the same metrics as evaluation/evaluate.py for direct comparison.

Usage:
    python baselines/evaluate_lightgbm.py
    DATA_DIR=data BASELINE_DIR=baselines python baselines/evaluate_lightgbm.py

    # Override paths (e.g. Colab + Drive):
    import os
    os.environ['DATA_DIR']     = '/content/drive/MyDrive/pitch_sequence/data'
    os.environ['BASELINE_DIR'] = '/content/drive/MyDrive/pitch_sequence/baselines'
    %run baselines/evaluate_lightgbm.py
"""

import os
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.metrics import log_loss as sk_log_loss

sys.path.insert(0, str(Path(__file__).parent))
from train_lightgbm import (
    ALL_FEATURES,
    HIT_LOCATION_CLASSES,
    OUTPUT_DIR,
    OUTCOME_CLASSES,
    TEST_PATH,
    load_location_split,
    load_outcome_split,
    top_k_recall_per_class,
)

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def top_k_recall(y_true: np.ndarray, proba: np.ndarray, k: int = 4) -> float:
    top_k = np.argsort(proba, axis=1)[:, -k:]
    return float((top_k == y_true[:, None]).any(axis=1).mean())


def brier_score(y_true: np.ndarray, proba: np.ndarray) -> float:
    one_hot = np.eye(proba.shape[1])[y_true]
    return float(np.mean(np.sum((proba - one_hot) ** 2, axis=1)))


def expected_calibration_error(
    y_true: np.ndarray, proba: np.ndarray, n_bins: int = 10
) -> float:
    confidences = proba.max(axis=1)
    predictions = proba.argmax(axis=1)
    correct = (predictions == y_true).astype(float)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        in_bin = (confidences > lo) & (confidences <= hi)
        n = in_bin.sum()
        if n > 0:
            ece += (n / len(y_true)) * abs(
                confidences[in_bin].mean() - correct[in_bin].mean()
            )
    return ece

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_section(
    title: str,
    y_true: np.ndarray,
    proba: np.ndarray,
    class_names: list[str],
) -> None:
    pad = '─' * max(0, 48 - len(title) - 4)
    print(f'\n── {title} {pad}')
    print(f'  Top-4 recall    : {top_k_recall(y_true, proba, k=4):.4f}')
    print(f'  Top-1 accuracy  : {top_k_recall(y_true, proba, k=1):.4f}')
    print(f'  Log-loss        : {sk_log_loss(y_true, proba):.4f}')
    print(f'  Brier score     : {brier_score(y_true, proba):.4f}')
    print(f'  ECE             : {expected_calibration_error(y_true, proba):.4f}')
    print('\n  Per-class Top-4 recall:')
    recalls = top_k_recall_per_class(y_true, proba)
    for i, name in enumerate(class_names):
        val = recalls.get(i, float('nan'))
        v = f'{val:.4f}' if not np.isnan(val) else '   N/A'
        print(f'    {name:<16}: {v}')

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    outcome_path  = f'{OUTPUT_DIR}/lgbm_outcome.txt'
    location_path = f'{OUTPUT_DIR}/lgbm_location.txt'

    print(f'Loading models from {OUTPUT_DIR}...')
    outcome_model  = lgb.Booster(model_file=outcome_path)
    location_model = lgb.Booster(model_file=location_path)

    print('Loading test data...')
    X_te,   y_te   = load_outcome_split(TEST_PATH)
    X_te_l, y_te_l = load_location_split(TEST_PATH)

    proba_out = outcome_model.predict(X_te)
    proba_loc = location_model.predict(X_te_l)

    print_section('Pitch Outcome (Test 2025)', y_te.values,   proba_out, OUTCOME_CLASSES)
    print_section('Hit Location (Test 2025)',  y_te_l.values, proba_loc, HIT_LOCATION_CLASSES)
    print()


if __name__ == '__main__':
    main()
