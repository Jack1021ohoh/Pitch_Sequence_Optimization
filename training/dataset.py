"""PyTorch Dataset for pitch sequence modeling.

Each sample corresponds to one row in batter_index.parquet:
  - A 400-pitch batter window ending at end_row, with the final pitch masked.
  - The K most recent pitcher appearances before this game.

Batter array column layout (32 cols, stored in .npy):
  [0 :11]  feat_continuous
  [11:15]  feat_outcome_continuous  (zeroed on final pitch at load time)
  [15:28]  cat_*
  [28:30]  mask flags               (set to 1 on final pitch at load time)
  [30:32]  labels                   (extracted then excluded from model input)

Model receives columns [0:30] (30-dim per pitch).
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

SEQ_LEN       = 400
K_APPEARANCES = 7    # pitcher history window
MAX_PITCH_LEN = 150  # max pitches per appearance
PITCHER_F     = 28   # feature cols in pitcher_appearances.pkl

# Column indices within the stored 32-col batter array
_OUTCOME_CONT_START = 11
_OUTCOME_CONT_END   = 15
_CAT_PITCH_OUTCOME  = 16
_CAT_HIT_LOCATION   = 17
_MASK_OUTCOME       = 28
_MASK_HIT           = 29
_LABEL_OUTCOME      = 30
_LABEL_HIT          = 31

# Single=2, Double=3, Triple=4, Home Run=5, Field Out=9 in OUTCOME_CLASSES
_CONTACT_LABELS = frozenset([2, 3, 4, 5, 9])


class PitchSequenceDataset(Dataset):
    def __init__(
        self,
        index_path:               str,
        batter_pitch_dir:         str,
        pitcher_appearances_path: str,
        split:                    str,
        k_appearances:            int = K_APPEARANCES,
        max_pitch_len:            int = MAX_PITCH_LEN,
    ):
        self.batter_pitch_dir = Path(batter_pitch_dir)
        self.k_appearances    = k_appearances
        self.max_pitch_len    = max_pitch_len
        self._array_cache: dict = {}

        index = pd.read_parquet(index_path)
        self.index = index[index['split'] == split].reset_index(drop=True)

        with open(pitcher_appearances_path, 'rb') as f:
            self.pitcher_appearances: dict = pickle.load(f)

        # O(1) lookup: (pitcher_id, game_pk) → index in appearances list
        self._pitcher_game_pos = {
            pid: {a['game_pk']: i for i, a in enumerate(apps)}
            for pid, apps in self.pitcher_appearances.items()
        }

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict:
        row        = self.index.iloc[idx]
        batter_id  = int(row['batter_id'])
        end_row    = int(row['end_row'])
        pitcher_id = int(row['pitcher_id'])
        game_pk    = int(row['game_pk'])

        arr    = self._load_batter_array(batter_id)
        window = arr[end_row - SEQ_LEN + 1:end_row + 1].copy()  # (400, 32)

        pitch_outcome_label = int(window[-1, _LABEL_OUTCOME])
        hit_location_label  = int(window[-1, _LABEL_HIT])
        ev_target           = float(window[-1, _OUTCOME_CONT_START])
        la_target           = float(window[-1, _OUTCOME_CONT_START + 1])
        is_contact          = pitch_outcome_label in _CONTACT_LABELS

        window[-1, _OUTCOME_CONT_START:_OUTCOME_CONT_END] = 0.0
        window[-1, _CAT_PITCH_OUTCOME] = 0
        window[-1, _CAT_HIT_LOCATION]  = 0
        window[-1, _MASK_OUTCOME]      = 1.0
        window[-1, _MASK_HIT]          = 1.0

        batter_seq = torch.from_numpy(window[:, :30])  # (400, 30)

        pitcher_pitches, pitcher_pitch_mask, pitcher_context, pitcher_app_mask = \
            self._load_pitcher_data(pitcher_id, game_pk)

        return {
            'batter_seq':           batter_seq,
            'pitcher_pitches':      pitcher_pitches,
            'pitcher_pitch_mask':   pitcher_pitch_mask,
            'pitcher_context':      pitcher_context,
            'pitcher_app_mask':     pitcher_app_mask,
            'pitch_outcome_label':  torch.tensor(pitch_outcome_label, dtype=torch.long),
            'hit_location_label':   torch.tensor(hit_location_label,  dtype=torch.long),
            'ev_target':            torch.tensor(ev_target, dtype=torch.float32),
            'la_target':            torch.tensor(la_target, dtype=torch.float32),
            'is_contact':           torch.tensor(is_contact, dtype=torch.bool),
        }

    def _load_batter_array(self, batter_id: int) -> np.ndarray:
        if batter_id not in self._array_cache:
            self._array_cache[batter_id] = np.load(
                self.batter_pitch_dir / f'{batter_id}.npy', mmap_mode='r'
            )
        return self._array_cache[batter_id]

    def _load_pitcher_data(self, pitcher_id: int, game_pk: int):
        K     = self.k_appearances
        T_max = self.max_pitch_len

        appearances = self.pitcher_appearances.get(pitcher_id, [])
        # O(1) lookup of where the current game sits in the pitcher's history
        idx  = self._pitcher_game_pos.get(pitcher_id, {}).get(game_pk, len(appearances))
        past = appearances[max(0, idx - K):idx]

        pitcher_pitches    = torch.zeros(K, T_max, PITCHER_F)
        pitcher_pitch_mask = torch.ones(K, T_max, dtype=torch.bool)
        pitcher_context    = torch.zeros(K, 7)
        pitcher_app_mask   = torch.ones(K, dtype=torch.bool)

        for i, app in enumerate(past):
            pitches = app['pitches']
            T = min(len(pitches), T_max)
            pitcher_pitches[i, :T]    = torch.from_numpy(pitches[:T])
            pitcher_pitch_mask[i, :T] = False
            pitcher_context[i]        = torch.from_numpy(app['context'])
            pitcher_app_mask[i]       = False

        # Guard against all-padding (e.g. pitcher with no prior appearances)
        if pitcher_app_mask.all():
            pitcher_app_mask[0]      = False
            pitcher_pitch_mask[0, 0] = False

        return pitcher_pitches, pitcher_pitch_mask, pitcher_context, pitcher_app_mask


def make_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        pin_memory=torch.cuda.is_available(),
    )
