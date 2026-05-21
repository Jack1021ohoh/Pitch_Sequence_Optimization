"""Shared constants and utilities for all data pipeline scripts."""

from pathlib import Path

import pandas as pd

TRAIN_YEARS = list(range(2021, 2024))
VAL_YEARS   = [2024]
TEST_YEARS  = [2025]
SPLITS      = ('train', 'val', 'test')


def load_statcast_years(
    raw_dir: str,
    years:   list,
    columns: list = None,
) -> pd.DataFrame:
    """Load and concatenate Statcast parquet files for the given years."""
    frames = []
    for year in years:
        path = Path(raw_dir) / f'statcast_{year}.parquet'
        if not path.exists():
            print(f'Warning: {path} not found, skipping.')
            continue
        frames.append(pd.read_parquet(path, columns=columns))
    if not frames:
        raise FileNotFoundError(f'No parquet files found in {raw_dir} for years {years}')
    return pd.concat(frames, ignore_index=True)
