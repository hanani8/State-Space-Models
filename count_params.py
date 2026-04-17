#!/usr/bin/env python3
"""
Count and compare trainable parameters for every experiment in run_all_experiments.sh.

Usage:
    python count_params.py [--config config.yaml] [--classes yes,no,go,stop]
"""
import sys
import os
import copy
import argparse
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train import build_model

ALL_CLASSES = [
    'backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five',
    'follow', 'forward', 'four', 'go', 'happy', 'house', 'learn', 'left',
    'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila',
    'six', 'stop', 'three', 'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero'
]

# Mirrors exactly the 11 runs in run_all_experiments.sh
EXPERIMENTS = [
    {"num":  1, "name": "LSTM + Raw Waveform",        "model": "lstm",        "input_type": "raw",  "ssm_type": None},
    {"num":  2, "name": "LSTM + Conv Frontend",        "model": "lstm",        "input_type": "conv", "ssm_type": None},
    {"num":  3, "name": "LSTM + MFCC",                 "model": "lstm",        "input_type": "mfcc", "ssm_type": None},
    {"num":  4, "name": "Transformer + Raw Waveform",  "model": "transformer", "input_type": "raw",  "ssm_type": None},
    {"num":  5, "name": "Transformer + Conv Frontend", "model": "transformer", "input_type": "conv", "ssm_type": None},
    {"num":  6, "name": "Transformer + MFCC",          "model": "transformer", "input_type": "mfcc", "ssm_type": None},
    {"num":  7, "name": "S4D + Raw Waveform",          "model": "s4",          "input_type": "raw",  "ssm_type": "s4d"},
    {"num":  8, "name": "S4D + Conv Frontend",         "model": "s4",          "input_type": "conv", "ssm_type": "s4d"},
    {"num":  9, "name": "S4D + MFCC",                  "model": "s4",          "input_type": "mfcc", "ssm_type": "s4d"},
    {"num": 10, "name": "S4  + Raw Waveform",          "model": "s4",          "input_type": "raw",  "ssm_type": "s4"},
    {"num": 11, "name": "S4  + MFCC",                  "model": "s4",          "input_type": "mfcc", "ssm_type": "s4"},
]


def count_params(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    return trainable, total


def main():
    parser = argparse.ArgumentParser(description="Count model parameters for all experiments")
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    parser.add_argument('--classes', type=str, default=None,
                        help='Comma-separated class subset (default: read from config, else all 35)')
    args = parser.parse_args()

    with open(args.config) as f:
        base_config = yaml.safe_load(f)

    if args.classes:
        classes = [c.strip() for c in args.classes.split(',')]
    else:
        classes = base_config.get('data', {}).get('classes', None)

    num_classes = len(classes) if classes else len(ALL_CLASSES)

    print(f"\nParameter Count — All Experiments  (config: {args.config})")
    if classes:
        print(f"Classes ({num_classes}): {', '.join(classes)}")
    else:
        print(f"Classes: all {len(ALL_CLASSES)}")
    print()

    results = []
    for exp in EXPERIMENTS:
        cfg = copy.deepcopy(base_config)
        cfg['model']      = exp['model']
        cfg['input_type'] = exp['input_type']
        if exp['ssm_type'] is not None:
            cfg['s4']['ssm_type'] = exp['ssm_type']

        try:
            model = build_model(cfg, num_classes, device='cpu')
            trainable, total = count_params(model)
            results.append({**exp, 'trainable': trainable, 'total': total, 'error': None})
        except Exception as e:
            results.append({**exp, 'trainable': None, 'total': None, 'error': str(e)})

    # ── table ───────────────────────────────────────────────────────────────────
    W_NAME, W_NUM = 28, 14
    header  = f"  {'#':>3}  {'Experiment':<{W_NAME}}  {'Trainable':>{W_NUM}}  {'Total':>{W_NUM}}  {'Size':>9}"
    divider = f"  {'-'*3}  {'-'*W_NAME}  {'-'*W_NUM}  {'-'*W_NUM}  {'-'*9}"

    print(header)
    print(divider)
    for r in results:
        prefix = f"  {r['num']:>3}  {r['name']:<{W_NAME}}"
        if r['error']:
            print(f"{prefix}  ERROR: {r['error'][:55]}")
        else:
            size_mb = r['total'] * 4 / 1e6
            print(f"{prefix}  {r['trainable']:>{W_NUM},}  {r['total']:>{W_NUM},}  {size_mb:>8.2f}M")
    print(divider)

    # ── summary by family ───────────────────────────────────────────────────────
    print("\nSummary by family (trainable params):")
    families = [
        ('LSTM',        [r for r in results if r['model'] == 'lstm']),
        ('Transformer', [r for r in results if r['model'] == 'transformer']),
        ('S4D',         [r for r in results if r['ssm_type'] == 's4d']),
        ('S4',          [r for r in results if r['ssm_type'] == 's4']),
    ]
    for fname, subset in families:
        valid = [r for r in subset if r['trainable'] is not None]
        if not valid:
            continue
        params = [r['trainable'] for r in valid]
        mn, mx, avg = min(params), max(params), sum(params) // len(params)
        print(f"  {fname:<14}  min={mn:>12,}  max={mx:>12,}  avg={avg:>12,}")


if __name__ == '__main__':
    main()
