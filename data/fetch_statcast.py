"""Fetch Statcast pitch-by-pitch data from pybaseball and save as parquet files."""

import sys
import time
from pathlib import Path

import pandas as pd
import pybaseball

pybaseball.cache.enable()

SEASON_START = '03-15'
SEASON_END   = '09-30'


def fetch_year(year: int, output_dir: Path) -> None:
    output_path = output_dir / f'statcast_{year}.parquet'
    if output_path.exists():
        print(f'[{year}] Already exists, skipping.')
        return

    start_dt = f'{year}-{SEASON_START}'
    end_dt   = f'{year}-{SEASON_END}'
    print(f'[{year}] Fetching {start_dt} -> {end_dt} ...')

    df = pybaseball.statcast(start_dt=start_dt, end_dt=end_dt, verbose=True)
    df.to_parquet(output_path, index=False)
    print(f'[{year}] Saved {len(df):,} rows -> {output_path}')


def fetch_all(
    start_year: int = 2015,
    end_year:   int = 2025,
    output_dir: str = 'data/raw',
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for year in range(start_year, end_year + 1):
        try:
            fetch_year(year, out)
        except Exception as e:
            print(f'[{year}] Error: {e}')
        time.sleep(3)


def inspect_values(raw_dir: str = 'data/raw') -> None:
    """Print value counts for events and description across all fetched parquet files."""
    files = sorted(Path(raw_dir).glob('statcast_*.parquet'))
    if not files:
        print(f'No parquet files found in {raw_dir}')
        return

    df = pd.concat(
        [pd.read_parquet(f, columns=['events', 'description']) for f in files],
        ignore_index=True,
    )
    print(f'Loaded {len(df):,} rows from {len(files)} files.\n')
    print('=== events ===')
    print(df['events'].value_counts(dropna=False).to_string())
    print('\n=== description ===')
    print(df['description'].value_counts(dropna=False).to_string())


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'inspect':
        inspect_values()
    else:
        fetch_all()
