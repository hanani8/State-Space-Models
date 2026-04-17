#!/usr/bin/env python3
"""
Dataset statistics for SpeechCommands v0.02.

Usage:
    python dataset_info.py [--data_root ./data] [--config config.yaml]
                           [--classes yes,no,go,stop] [--split all|train|val|test]
                           [--no_bar]
"""
import os
import sys
import argparse
from collections import Counter

import torchaudio

ALL_CLASSES = [
    'backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five',
    'follow', 'forward', 'four', 'go', 'happy', 'house', 'learn', 'left',
    'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila',
    'six', 'stop', 'three', 'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero'
]

SPLIT_MAP = {'train': 'training', 'val': 'validation', 'test': 'testing'}
W = 64


def get_labels(dataset):
    """Read class labels from file paths — no audio loading required."""
    if hasattr(dataset, '_walker'):
        return [os.path.basename(os.path.dirname(p)) for p in dataset._walker]
    # Fallback for future torchaudio versions (slow)
    print("  Note: _walker unavailable, iterating dataset (slow)...")
    return [dataset[i][2] for i in range(len(dataset))]


def print_split(split_name, all_labels, classes, bar=True):
    counts     = Counter(all_labels)
    sel        = {c: counts.get(c, 0) for c in classes}
    sel_total  = sum(sel.values())
    all_total  = len(all_labels)
    other      = all_total - sel_total
    max_n      = max(sel.values()) if sel else 1
    BAR_W      = 24

    print(f"\n{'─'*W}")
    print(f"  {split_name}  │  {sel_total:,} selected  /  {all_total:,} total")
    print(f"{'─'*W}")
    print(f"  {'Class':<16}  {'Count':>7}  {'%':>6}  Distribution")
    print(f"  {'-'*16}  {'-'*7}  {'-'*6}  {'-'*BAR_W}")

    for cls in classes:
        n   = sel[cls]
        pct = 100 * n / sel_total if sel_total else 0
        b   = ('█' * int(BAR_W * n / max_n)) if bar else ''
        print(f"  {cls:<16}  {n:>7,}  {pct:>5.1f}%  {b}")

    if other > 0:
        pct_other = 100 * other / all_total
        print(f"  {'[other]':<16}  {other:>7,}  ——      excluded ({pct_other:.1f}% of split)")

    print(f"  {'-'*16}  {'-'*7}")
    print(f"  {'TOTAL':<16}  {sel_total:>7,}  100%")


def main():
    parser = argparse.ArgumentParser(description="Dataset statistics for SpeechCommands v0.02")
    parser.add_argument('--data_root', default='./data', help='Root data directory')
    parser.add_argument('--config', default='config.yaml',
                        help='Read data.classes from config if --classes not given')
    parser.add_argument('--classes', type=str, default=None,
                        help='Comma-separated class subset (default: read from config, else all 35)')
    parser.add_argument('--split', default='all',
                        choices=['all', 'train', 'val', 'test'],
                        help='Which split(s) to analyse')
    parser.add_argument('--no_bar', action='store_true', help='Disable ASCII bar chart')
    args = parser.parse_args()

    # ── resolve class list + training config ───────────────────────────────────
    subset_fraction = None
    if args.classes:
        classes = [c.strip() for c in args.classes.split(',')]
    else:
        try:
            import yaml
            with open(args.config) as f:
                cfg = yaml.safe_load(f)
            classes         = cfg.get('data', {}).get('classes', None)
            subset_fraction = cfg.get('data', {}).get('subset_fraction', None)
        except Exception:
            classes = None

    if classes is None:
        classes = list(ALL_CLASSES)
    else:
        invalid = set(classes) - set(ALL_CLASSES)
        if invalid:
            sys.exit(f"ERROR — unknown classes: {sorted(invalid)}\nValid: {ALL_CLASSES}")
        # preserve canonical ordering so indices are deterministic
        classes = [c for c in ALL_CLASSES if c in set(classes)]

    # ── header ──────────────────────────────────────────────────────────────────
    print("=" * W)
    print("  SpeechCommands v0.02 — Dataset Statistics")
    print("=" * W)
    print(f"  Root    : {os.path.abspath(args.data_root)}")
    print(f"  Classes : {len(classes)} / {len(ALL_CLASSES)}")
    print()
    for i, cls in enumerate(classes):
        end = '\n' if (i + 1) % 5 == 0 or i == len(classes) - 1 else '   '
        print(f"  [{i:2d}] {cls:<12}", end=end)

    # ── per-split stats ─────────────────────────────────────────────────────────
    splits = ['train', 'val', 'test'] if args.split == 'all' else [args.split]

    grand_sel = grand_all = 0
    train_sel = None
    for split in splits:
        print(f"\nLoading {split} split...", end='\r')
        ds     = torchaudio.datasets.SPEECHCOMMANDS(
            root=args.data_root, download=False, subset=SPLIT_MAP[split]
        )
        labels  = get_labels(ds)
        sel_cnt = sum(1 for l in labels if l in set(classes))
        grand_sel += sel_cnt
        grand_all += len(labels)
        if split == 'train':
            train_sel = sel_cnt
        print_split(split.capitalize(), labels, classes, bar=not args.no_bar)

    # ── grand total + effective training size ───────────────────────────────────
    print(f"\n{'═'*W}")
    frac = 100 * grand_sel / grand_all if grand_all else 0
    print(f"  Grand total : {grand_sel:,} selected / {grand_all:,} total  ({frac:.1f}%)")

    if train_sel is not None and subset_fraction is not None:
        if subset_fraction <= 0.0 or subset_fraction >= 1.0:
            eff = train_sel if subset_fraction >= 1.0 else 1
            note = "full dataset" if subset_fraction >= 1.0 else f"WARNING: subset_fraction={subset_fraction} → only 1 sample!"
        else:
            eff  = max(1, int(train_sel * subset_fraction))
            note = f"subset_fraction={subset_fraction}"
        print(f"  Effective train samples used ({note}): {eff:,} / {train_sel:,}")

    print(f"{'═'*W}")


if __name__ == '__main__':
    main()
