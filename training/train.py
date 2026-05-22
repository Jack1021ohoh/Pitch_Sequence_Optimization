"""Main training script.

Usage:
  python -m training.train [--epochs N] [--batch-size N] [--lr LR] [--workers N]

Data directory is read from the DATA_DIR environment variable (default: 'data').
Checkpoints are saved to the CKPT_DIR environment variable (default: 'checkpoints').

On Colab:
  import os
  os.environ['DATA_DIR']  = '/content/drive/MyDrive/pitch_sequence/data'
  os.environ['CKPT_DIR']  = '/content/drive/MyDrive/pitch_sequence/checkpoints'
  %run training/train.py
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.full_model import PitchOutcomeModel, run_model
from training.dataset import PitchSequenceDataset, make_dataloader
from training.loss import MultiTaskLoss
from evaluation.metrics import top_k_precision


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs',       type=int,   default=20)
    p.add_argument('--batch-size',   type=int,   default=64)
    p.add_argument('--lr',           type=float, default=1e-4)
    p.add_argument('--workers',      type=int,   default=4,
                   help='DataLoader workers. Use 0 on Windows.')
    p.add_argument('--warmup-steps', type=int,   default=1000)
    p.add_argument('--grad-clip',    type=float, default=1.0)
    p.add_argument('--d-model',      type=int,   default=256)
    p.add_argument('--patience',     type=int,   default=5,
                   help='Early stopping patience (epochs). 0 to disable.')
    p.add_argument('--resume',       type=str,   default=None)
    return p.parse_args()


def build_dataloaders(data_dir: Path, batch_size: int, num_workers: int):
    index_path  = data_dir / 'sequences' / 'batter_index.parquet'
    pitch_dir   = data_dir / 'sequences' / 'batter_pitches'
    appearances = data_dir / 'sequences' / 'pitcher_appearances.pkl'

    def make(split, shuffle):
        ds = PitchSequenceDataset(
            index_path=str(index_path),
            batter_pitch_dir=str(pitch_dir),
            pitcher_appearances_path=str(appearances),
            split=split,
        )
        return make_dataloader(ds, batch_size, shuffle, num_workers)

    return make('train', shuffle=True), make('val', shuffle=False)


def build_scheduler(optimizer, warmup_steps: int, total_steps: int):
    warmup = LinearLR(optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=max(total_steps - warmup_steps, 1))
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])


def train_epoch(model, loader, criterion, optimizer, scheduler, device, grad_clip):
    model.train()
    total     = {k: 0.0 for k in ('loss', 'cls_loss', 'phy_loss', 'cal_loss')}
    n_batches = 0

    pbar = tqdm(loader, desc='  train', leave=False)
    for batch in pbar:
        batch  = {k: v.to(device) for k, v in batch.items()}
        losses = criterion(run_model(model, batch), batch)

        optimizer.zero_grad()
        losses['loss'].backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        for k in total:
            total[k] += losses[k].item()
        n_batches += 1
        pbar.set_postfix(loss=f'{losses["loss"].item():.4f}')

    return {k: v / n_batches for k, v in total.items()}


@torch.no_grad()
def val_epoch(model, loader, criterion, device):
    model.eval()
    total                                  = {k: 0.0 for k in ('loss', 'cls_loss', 'phy_loss', 'cal_loss')}
    all_outcome_logits, all_outcome_labels = [], []
    all_loc_logits,     all_loc_labels     = [], []
    n_batches = 0

    for batch in tqdm(loader, desc='    val', leave=False):
        batch = {k: v.to(device) for k, v in batch.items()}
        preds = run_model(model, batch)
        losses = criterion(preds, batch)

        for k in total:
            total[k] += losses[k].item()
        n_batches += 1

        all_outcome_logits.append(preds['pitch_outcome_logits'].cpu())
        all_outcome_labels.append(batch['pitch_outcome_label'].cpu())
        all_loc_logits.append(preds['hit_location_logits'].cpu())
        all_loc_labels.append(batch['hit_location_label'].cpu())

    metrics = {k: v / n_batches for k, v in total.items()}
    metrics['top4_outcome']  = top_k_precision(torch.cat(all_outcome_logits), torch.cat(all_outcome_labels), k=4)
    metrics['top4_location'] = top_k_precision(torch.cat(all_loc_logits),     torch.cat(all_loc_labels),     k=4)
    return metrics


def save_checkpoint(path, epoch, model, optimizer, scheduler, metrics):
    torch.save({
        'epoch':     epoch,
        'model':     model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'metrics':   metrics,
    }, path)


def main():
    args     = parse_args()
    data_dir = Path(os.environ.get('DATA_DIR', 'data'))
    ckpt_dir = Path(os.environ.get('CKPT_DIR', 'checkpoints'))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    print('Building dataloaders...')
    train_loader, val_loader = build_dataloaders(data_dir, args.batch_size, args.workers)
    total_steps = args.epochs * len(train_loader)

    print('Building model...')
    model     = PitchOutcomeModel(d_model=args.d_model).to(device)
    criterion = MultiTaskLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = build_scheduler(optimizer, args.warmup_steps, total_steps)

    start_epoch = 0
    best_val_loss = float('inf')

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch   = ckpt['epoch'] + 1
        best_val_loss = ckpt['metrics'].get('loss', float('inf'))
        if best_val_loss != best_val_loss:  # NaN check
            best_val_loss = float('inf')
        print(f'Resumed from epoch {ckpt["epoch"]}')

    print(f'Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')

    epochs_no_improve = 0

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, scheduler, device, args.grad_clip)
        val_metrics   = val_epoch(model, val_loader, criterion, device)
        print(
            f'Epoch {epoch:03d} | {time.time()-t0:.0f}s | '
            f'train loss {train_metrics["loss"]:.4f} '
            f'(cls {train_metrics["cls_loss"]:.4f} '
            f'phy {train_metrics["phy_loss"]:.4f} '
            f'cal {train_metrics["cal_loss"]:.4f}) | '
            f'val loss {val_metrics["loss"]:.4f} | '
            f'val top-4 outcome {val_metrics["top4_outcome"]:.4f} '
            f'location {val_metrics["top4_location"]:.4f}'
        )

        save_checkpoint(ckpt_dir / 'latest.pt', epoch, model, optimizer, scheduler, val_metrics)

        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            epochs_no_improve = 0
            save_checkpoint(ckpt_dir / 'best.pt', epoch, model, optimizer, scheduler, val_metrics)
            print(f'  -> New best val loss: {best_val_loss:.4f}')
        else:
            epochs_no_improve += 1
            if args.patience > 0 and epochs_no_improve >= args.patience:
                print(f'Early stopping: no improvement for {args.patience} epochs.')
                break

    print('Training complete.')


if __name__ == '__main__':
    main()
