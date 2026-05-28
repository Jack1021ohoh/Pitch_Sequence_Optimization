"""MCTS pitch-sequence search — CLI entry point.

Usage:
  python -m mcts.run_search --checkpoint checkpoints/best.pt
                            [--batter-id 592450 --pitcher-id 605483]
                            [--split test] [--n-iter 500] [--c 1.4]

If --batter-id / --pitcher-id are omitted, a random sample from --split is used.
DATA_DIR is read from the environment (default: 'data').
"""

import argparse
import json
import os
import pickle
import random
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.full_model  import PitchOutcomeModel
from training.dataset   import PitchSequenceDataset
from data.preprocess    import load_artifacts
from mcts.state         import AtBatState
from mcts.simulator     import (PitchSimulator,
                                _CAT_BALLS, _CAT_STRIKES, _CAT_OUTS,
                                _CAT_ON_1B, _CAT_ON_2B, _CAT_ON_3B,
                                _CAT_INNING, _CAT_STAND, _CAT_P_THROWS)
from mcts.node          import MCTSNode
from mcts.search        import mcts_search

_STAND_MAP  = {0: 'L', 1: 'R', 2: 'S'}
_THROWS_MAP = {0: 'L', 1: 'R'}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint',   default=None)
    p.add_argument('--batter-id',    type=int,   default=None)
    p.add_argument('--pitcher-id',   type=int,   default=None)
    p.add_argument('--split',        default='test', choices=['train', 'val', 'test'])
    p.add_argument('--n-iter',       type=int,   default=500)
    p.add_argument('--c',            type=float, default=1.4)
    p.add_argument('--max-rollout',  type=int,   default=12)
    p.add_argument('--top-k',        type=int,   default=10)
    p.add_argument('--seed',         type=int,   default=42)
    p.add_argument('--balls',        type=int,   default=None, choices=[0,1,2,3])
    p.add_argument('--strikes',      type=int,   default=None, choices=[0,1,2])
    p.add_argument('--outs',         type=int,   default=None, choices=[0,1,2])
    p.add_argument('--inning',       type=int,   default=None)
    p.add_argument('--on-1b',        action='store_true')
    p.add_argument('--on-2b',        action='store_true')
    p.add_argument('--on-3b',        action='store_true')
    return p.parse_args()


def _pitcher_tensors(dataset: PitchSequenceDataset, pitcher_id: int, game_pk: int, device) -> dict:
    pitches, pitch_mask, context, app_mask = dataset._load_pitcher_data(pitcher_id, game_pk)
    return {
        'pitcher_pitches':    pitches.unsqueeze(0).to(device),
        'pitcher_pitch_mask': pitch_mask.unsqueeze(0).to(device),
        'pitcher_context':    context.unsqueeze(0).to(device),
        'pitcher_app_mask':   app_mask.unsqueeze(0).to(device),
    }


def _root_window(arr: np.ndarray, end_row: int) -> np.ndarray:
    """400-pitch history ending at end_row-1 (no masks set)."""
    window = np.zeros((400, 30), dtype=np.float32)
    start  = end_row - 400
    if start >= 0:
        window[:] = arr[start:end_row, :30]
    else:
        chunk = arr[0:end_row, :30]
        window[-len(chunk):] = chunk
    return window


def _initial_state(arr: np.ndarray, end_row: int) -> tuple[AtBatState, str, str]:
    row = arr[end_row]
    state = AtBatState(
        balls      = int(row[_CAT_BALLS]),
        strikes    = int(row[_CAT_STRIKES]),
        outs       = int(row[_CAT_OUTS]),
        on_1b      = bool(row[_CAT_ON_1B]),
        on_2b      = bool(row[_CAT_ON_2B]),
        on_3b      = bool(row[_CAT_ON_3B]),
        inning     = int(row[_CAT_INNING]),
        score_diff = 0,
    )
    stand   = _STAND_MAP.get(int(row[_CAT_STAND]),    'R')
    throws  = _THROWS_MAP.get(int(row[_CAT_P_THROWS]), 'R')
    return state, stand, throws


def _print_report(root: MCTSNode, top_k: int) -> None:
    ranked = root.ranked_actions()[:top_k]
    print(f'\n── MCTS Recommendations ({root.visits} simulations) ─────────────────')
    print(f'  {"#":<4} {"Pitch":<12} {"Zone":<6} {"Visits":<8} {"Q (↑ pitcher)":<16} {"Share"}')
    print('  ' + '─' * 55)
    for rank, action in enumerate(ranked, 1):
        child           = root.children[action]
        pitch_type, zone = action
        share           = 100 * child.visits / root.visits if root.visits else 0
        print(f'  {rank:<4} {pitch_type:<12} {zone:<6} {child.visits:<8} '
              f'{child.q_value:<16.4f} {share:.1f}%')
    print()


def main():
    args     = parse_args()
    random.seed(args.seed)
    data_dir = Path(os.environ.get('DATA_DIR', 'data'))
    ckpt_dir = Path(os.environ.get('CKPT_DIR', 'checkpoints'))
    device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── Artifacts ────────────────────────────────────────────────────
    _, _, vocabs = load_artifacts(str(data_dir / 'artifacts'))
    with open(data_dir / 'artifacts' / 'pitch_library.pkl', 'rb') as f:
        pitch_library = pickle.load(f)
    with open(data_dir / 'artifacts' / 're24_table.json') as f:
        run_values = json.load(f)['run_values']

    # ── Model ────────────────────────────────────────────────────────
    ckpt_path = Path(args.checkpoint) if args.checkpoint else ckpt_dir / 'best.pt'
    ckpt  = torch.load(ckpt_path, map_location=device)
    model = PitchOutcomeModel().to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f'Loaded checkpoint (epoch {ckpt["epoch"]})')

    # ── Dataset (pitcher appearance lookup) ─────────────────────────
    dataset = PitchSequenceDataset(
        index_path               = str(data_dir / 'sequences' / 'batter_index.parquet'),
        batter_pitch_dir         = str(data_dir / 'sequences' / 'batter_pitches'),
        pitcher_appearances_path = str(data_dir / 'sequences' / 'pitcher_appearances.pkl'),
        split=args.split,
    )
    index = dataset.index

    # ── Sample selection ─────────────────────────────────────────────
    if args.batter_id is not None and args.pitcher_id is not None:
        mask = (index['batter_id'] == args.batter_id) & (index['pitcher_id'] == args.pitcher_id)
        if not mask.any():
            print(f'No {args.split}-split match: batter {args.batter_id} vs pitcher {args.pitcher_id}')
            sys.exit(1)
        row = index[mask].iloc[0]
    else:
        row = index.sample(1, random_state=args.seed).iloc[0]

    batter_id  = int(row['batter_id'])
    pitcher_id = int(row['pitcher_id'])
    game_pk    = int(row['game_pk'])
    end_row    = int(row['end_row'])
    print(f'Batter {batter_id} vs Pitcher {pitcher_id} | game {game_pk}')

    if pitcher_id not in pitch_library:
        print(f'Pitcher {pitcher_id} not in pitch library (insufficient training data).')
        sys.exit(1)

    # ── Build root state ─────────────────────────────────────────────
    arr    = np.load(data_dir / 'sequences' / 'batter_pitches' / f'{batter_id}.npy')
    window = _root_window(arr, end_row)
    state, stand, throws = _initial_state(arr, end_row)

    if args.on_1b or args.on_2b or args.on_3b or \
            any(v is not None for v in [args.balls, args.strikes, args.outs, args.inning]):
        state = replace(
            state,
            balls   = args.balls   if args.balls   is not None else state.balls,
            strikes = args.strikes if args.strikes is not None else state.strikes,
            outs    = args.outs    if args.outs    is not None else state.outs,
            inning  = args.inning  if args.inning  is not None else state.inning,
            on_1b   = args.on_1b  or state.on_1b,
            on_2b   = args.on_2b  or state.on_2b,
            on_3b   = args.on_3b  or state.on_3b,
        )

    print(f'Count: {state.balls}-{state.strikes} | Outs: {state.outs} | '
          f'Inning: {state.inning} | Batter: {stand} | Pitcher: {throws}')
    runners = ('_' if not state.on_1b else '1') + \
              ('_' if not state.on_2b else '2') + \
              ('_' if not state.on_3b else '3')
    print(f'Runners: {runners}')

    p_tensors = _pitcher_tensors(dataset, pitcher_id, game_pk, device)
    simulator = PitchSimulator(model, pitch_library, run_values, vocabs, device)
    actions   = simulator.available_actions(pitcher_id)
    print(f'Repertoire: {len(actions)} (pitch_type, zone) actions\n')

    root = MCTSNode(
        state           = state,
        batter_window   = window,
        action          = None,
        untried_actions = actions,
    )

    # ── Search ───────────────────────────────────────────────────────
    print(f'Running {args.n_iter} MCTS iterations (c={args.c})...')
    mcts_search(
        root, simulator, p_tensors, pitcher_id, stand, throws,
        actions=actions,
        n_iter=args.n_iter, c=args.c, max_rollout_depth=args.max_rollout,
    )

    _print_report(root, args.top_k)


if __name__ == '__main__':
    main()
