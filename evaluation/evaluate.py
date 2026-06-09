"""Evaluate a trained model checkpoint on the test set.

Usage:
  python -m evaluation.evaluate --checkpoint checkpoints/best.pt [--split test]

DATA_DIR is read from the environment (default: 'data').
"""

import argparse
import os
import pickle
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.full_model import PitchOutcomeModel, N_PITCH_OUTCOME, N_HIT_LOCATION, run_model
from training.dataset import PitchSequenceDataset, make_dataloader
from evaluation.metrics import (
    top_k_precision, per_class_top4, log_loss_score, brier_score,
    expected_calibration_error, mog_nll, physics_mae,
)

OUTCOME_NAMES  = [
    'Ball', 'Strike', 'Single', 'Double', 'Triple',
    'Home Run', 'Strikeout', 'Walk', 'Hit by Pitch', 'Field Out',
]
LOCATION_NAMES = [
    'Pitcher', 'Catcher', 'First Base', 'Second Base', 'Third Base',
    'Shortstop', 'Left Field', 'Center Field', 'Right Field', 'None',
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--split',      default='test', choices=['train', 'val', 'test'])
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--workers',    type=int, default=4)
    return p.parse_args()


@torch.no_grad()
def evaluate(model, loader, device, ev_scale: float, la_scale: float) -> dict:
    model.eval()

    outcome_logits_all, outcome_labels_all = [], []
    loc_logits_all,     loc_labels_all     = [], []
    ev_params_all,      ev_targets_all     = [], []
    la_params_all,      la_targets_all     = [], []
    contact_mask_all                       = []

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        preds = run_model(model, batch)

        outcome_logits_all.append(preds['pitch_outcome_logits'].cpu())
        outcome_labels_all.append(batch['pitch_outcome_label'].cpu())
        loc_logits_all.append(preds['hit_location_logits'].cpu())
        loc_labels_all.append(batch['hit_location_label'].cpu())
        ev_params_all.append(preds['ev_params'].cpu())
        ev_targets_all.append(batch['ev_target'].cpu())
        la_params_all.append(preds['la_params'].cpu())
        la_targets_all.append(batch['la_target'].cpu())
        contact_mask_all.append(batch['is_contact'].cpu())

    outcome_logits = torch.cat(outcome_logits_all)
    outcome_labels = torch.cat(outcome_labels_all)
    loc_logits     = torch.cat(loc_logits_all)
    loc_labels     = torch.cat(loc_labels_all)
    contact_mask   = torch.cat(contact_mask_all)

    results = {
        'outcome_top4':          top_k_precision(outcome_logits, outcome_labels, k=4),
        'outcome_top1':          top_k_precision(outcome_logits, outcome_labels, k=1),
        'outcome_logloss':       log_loss_score(outcome_logits, outcome_labels),
        'outcome_brier':         brier_score(outcome_logits, outcome_labels, N_PITCH_OUTCOME),
        'outcome_ece':           expected_calibration_error(outcome_logits, outcome_labels),
        'outcome_per_class_top4': per_class_top4(outcome_logits, outcome_labels, N_PITCH_OUTCOME, k=4),
        'location_top4':         top_k_precision(loc_logits, loc_labels, k=4),
        'location_top1':         top_k_precision(loc_logits, loc_labels, k=1),
        'location_logloss':      log_loss_score(loc_logits, loc_labels),
        'location_brier':        brier_score(loc_logits, loc_labels, N_HIT_LOCATION),
        'location_ece':          expected_calibration_error(loc_logits, loc_labels),
        'location_per_class_top4': per_class_top4(loc_logits, loc_labels, N_HIT_LOCATION, k=4),
    }

    if contact_mask.any():
        ev_p = torch.cat(ev_params_all)[contact_mask]
        la_p = torch.cat(la_params_all)[contact_mask]
        ev_t = torch.cat(ev_targets_all)[contact_mask]
        la_t = torch.cat(la_targets_all)[contact_mask]
        results.update({
            'ev_nll':     mog_nll(ev_p, ev_t),
            'la_nll':     mog_nll(la_p, la_t),
            'ev_mae_mph': physics_mae(ev_p, ev_t, scale=ev_scale),
            'la_mae_deg': physics_mae(la_p, la_t, scale=la_scale),
        })

    return results


def print_report(results: dict):
    print('\n── Pitch Outcome ───────────────────────────────')
    print(f'  Top-4 recall    : {results["outcome_top4"]:.4f}')
    print(f'  Top-1 accuracy  : {results["outcome_top1"]:.4f}')
    print(f'  Log-loss        : {results["outcome_logloss"]:.4f}')
    print(f'  Brier score     : {results["outcome_brier"]:.4f}')
    print(f'  ECE             : {results["outcome_ece"]:.4f}')
    print('\n  Per-class Top-4 recall:')
    for name, val in zip(OUTCOME_NAMES, results['outcome_per_class_top4']):
        v = f'{val:.4f}' if val is not None else '  N/A'
        print(f'    {name:<16}: {v}')

    print('\n── Hit Location ─────────────────────────────────')
    print(f'  Top-4 recall    : {results["location_top4"]:.4f}')
    print(f'  Top-1 accuracy  : {results["location_top1"]:.4f}')
    print(f'  Log-loss        : {results["location_logloss"]:.4f}')
    print(f'  Brier score     : {results["location_brier"]:.4f}')
    print(f'  ECE             : {results["location_ece"]:.4f}')
    print('\n  Per-class Top-4 recall:')
    for name, val in zip(LOCATION_NAMES, results['location_per_class_top4']):
        v = f'{val:.4f}' if val is not None else '  N/A'
        print(f'    {name:<16}: {v}')

    if 'ev_nll' in results:
        print('\n── Physics (contact pitches only) ───────────────')
        print(f'  EV NLL          : {results["ev_nll"]:.4f}')
        print(f'  LA NLL          : {results["la_nll"]:.4f}')
        print(f'  EV MAE (mph)    : {results["ev_mae_mph"]:.2f}')
        print(f'  LA MAE (deg)    : {results["la_mae_deg"]:.2f}')
    print()


def main():
    args     = parse_args()
    data_dir = Path(os.environ.get('DATA_DIR', 'data'))
    device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(data_dir / 'artifacts' / 'outcome_scaler.pkl', 'rb') as f:
        outcome_scaler = pickle.load(f)
    ev_scale = float(outcome_scaler.scale_[0])
    la_scale = float(outcome_scaler.scale_[1])

    ckpt  = torch.load(args.checkpoint, map_location=device)
    model = PitchOutcomeModel().to(device)
    model.load_state_dict(ckpt['model'])
    print(f'Loaded checkpoint: {args.checkpoint} (epoch {ckpt["epoch"]})')

    ds = PitchSequenceDataset(
        index_path=str(data_dir / 'sequences' / 'batter_index.parquet'),
        batter_pitch_dir=str(data_dir / 'sequences' / 'batter_pitches'),
        pitcher_appearances_path=str(data_dir / 'sequences' / 'pitcher_appearances.pkl'),
        split=args.split,
    )
    loader = make_dataloader(ds, args.batch_size, shuffle=False, num_workers=args.workers)
    print(f'Evaluating on {args.split} split ({len(ds):,} samples)...')

    print_report(evaluate(model, loader, device, ev_scale, la_scale))


if __name__ == '__main__':
    main()
