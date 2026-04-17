#!/usr/bin/env python3
"""
Find the maximum safe batch size for every experiment in run_all_experiments.sh
without triggering an OOM.

Strategy (per experiment):
  1. Warmup pass at bs=1 to prime CUDA kernels and allocate optimizer state.
  2. Exponential growth (1 → 2 → 4 → …) until OOM or --limit is reached.
  3. Binary search between the last good and first OOM to find the exact max.
  4. Report peak GPU memory at the found max batch size.

The full model (preprocessor + backbone) is built identically to training, and
each probe runs a real forward + backward pass so activation memory is counted.

Usage:
    python find_batch_size.py [--config config.yaml] [--device cuda]
                              [--limit 1024] [--classes yes,no,go,stop]
"""
import sys, os, gc, copy, argparse, time, yaml
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import build_model

ALL_CLASSES = [
    'backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five',
    'follow', 'forward', 'four', 'go', 'happy', 'house', 'learn', 'left',
    'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila',
    'six', 'stop', 'three', 'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero'
]

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


# ── memory helpers ────────────────────────────────────────────────────────────

def clear_memory(device):
    gc.collect()
    if device != 'cpu':
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()


def peak_allocated(device):
    return torch.cuda.max_memory_allocated(device) if device != 'cpu' else 0


# ── single batch probe ────────────────────────────────────────────────────────

def try_batch(model, batch_size, num_classes, max_length, device):
    """
    Run one forward + backward pass at `batch_size`.
    Returns (success: bool, peak_bytes: int).
    Peak is measured only on success (it's meaningless after OOM).
    """
    clear_memory(device)
    if device != 'cpu':
        torch.cuda.reset_peak_memory_stats(device)

    try:
        model.train()  # cuDNN RNN backward requires training mode (saves dropout masks etc.)
        x    = torch.randn(batch_size, max_length, device=device)
        y    = torch.randint(0, num_classes, (batch_size,), device=device)
        out  = model(x)
        loss = nn.CrossEntropyLoss()(out, y)
        loss.backward()

        peak = peak_allocated(device)
        del x, y, out, loss
        model.zero_grad(set_to_none=True)
        clear_memory(device)
        return True, peak

    except RuntimeError as e:
        if 'out of memory' not in str(e).lower():
            raise
        model.zero_grad(set_to_none=True)
        clear_memory(device)
        return False, 0

    except MemoryError:
        model.zero_grad(set_to_none=True)
        clear_memory(device)
        return False, 0


# ── search ────────────────────────────────────────────────────────────────────

def find_max_batch_size(model, num_classes, max_length, device, limit=1024):
    """
    Returns (max_batch_size, peak_bytes_at_max).
    max_batch_size == 0 means even bs=1 OOMs.
    """
    # Warmup: prime CUDA kernels so first real probe isn't penalised
    try_batch(model, 1, num_classes, max_length, device)

    # Phase 1 — exponential growth
    lo, hi = 0, None
    bs = 1
    while bs <= limit:
        ok, _ = try_batch(model, bs, num_classes, max_length, device)
        if ok:
            lo = bs
            bs = bs * 2
        else:
            hi = bs
            break

    if lo == 0:
        return 0, 0  # bs=1 already OOMs

    if hi is None:
        # Never hit OOM within limit — lo is the highest power-of-2 tried
        _, peak = try_batch(model, lo, num_classes, max_length, device)
        return lo, peak

    # Phase 2 — binary search between lo (good) and hi (OOM)
    while lo < hi - 1:
        mid = (lo + hi) // 2
        ok, _ = try_batch(model, mid, num_classes, max_length, device)
        if ok:
            lo = mid
        else:
            hi = mid

    _, peak = try_batch(model, lo, num_classes, max_length, device)
    return lo, peak


# ── helpers ───────────────────────────────────────────────────────────────────

def prev_power_of_two(n):
    """Largest power of 2 that is ≤ n."""
    if n <= 0:
        return 0
    p = 1
    while p * 2 <= n:
        p *= 2
    return p


def safe_recommended(max_bs, headroom=0.75):
    """Largest power-of-2 at or below headroom * max_bs (≥1 if possible)."""
    if max_bs <= 0:
        return 0
    target = max(1, int(max_bs * headroom))
    return prev_power_of_two(target)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Find max batch size per experiment without OOM"
    )
    parser.add_argument('--config',  default='config.yaml')
    parser.add_argument('--device',  default=None,
                        help='Override device (default: read from config)')
    parser.add_argument('--limit',   type=int, default=1024,
                        help='Largest batch size to probe (default: 1024)')
    parser.add_argument('--classes', type=str, default=None,
                        help='Comma-separated class subset (default: from config)')
    args = parser.parse_args()

    with open(args.config) as f:
        base_config = yaml.safe_load(f)

    # ── device ────────────────────────────────────────────────────────────────
    device = args.device or base_config.get('device', 'cuda')
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available — falling back to CPU (OOM detection is unreliable on CPU)")
        device = 'cpu'

    # ── classes / problem size ────────────────────────────────────────────────
    if args.classes:
        classes = [c.strip() for c in args.classes.split(',')]
    else:
        classes = base_config.get('data', {}).get('classes', None)

    num_classes = len(classes) if classes else len(ALL_CLASSES)
    max_length  = base_config['data']['max_length']
    default_bs  = base_config['data']['batch_size']

    # ── banner ────────────────────────────────────────────────────────────────
    if device != 'cpu' and torch.cuda.is_available():
        props    = torch.cuda.get_device_properties(device)
        total_gb = props.total_memory / 1e9
        dev_str  = f"{props.name}  ({total_gb:.1f} GB total)"
    else:
        dev_str  = "CPU"
        total_gb = None

    print(f"\n{'━'*72}")
    print(f"  Batch Size Finder — forward + backward probe")
    print(f"{'━'*72}")
    print(f"  Device      : {dev_str}")
    print(f"  Config      : {args.config}")
    print(f"  Classes     : {num_classes}  |  max_length={max_length}")
    print(f"  Default BS  : {default_bs}  |  search limit={args.limit}")
    print(f"{'━'*72}\n")

    results = []
    for exp in EXPERIMENTS:
        cfg = copy.deepcopy(base_config)
        cfg['model']      = exp['model']
        cfg['input_type'] = exp['input_type']
        if exp['ssm_type'] is not None:
            cfg['s4']['ssm_type'] = exp['ssm_type']

        label = f"[{exp['num']:2d}/11] {exp['name']}"
        print(f"  {label:<38}", end='', flush=True)
        t0 = time.time()

        try:
            model = build_model(cfg, num_classes, device)

            max_bs, peak_bytes = find_max_batch_size(
                model, num_classes, max_length, device, limit=args.limit
            )

            elapsed   = time.time() - t0
            peak_gb   = peak_bytes / 1e9
            pct_vram  = (peak_gb / total_gb * 100) if total_gb else None
            rec_bs    = safe_recommended(max_bs)
            default_ok = max_bs >= default_bs if max_bs > 0 else False

            results.append({
                **exp,
                'max_bs':     max_bs,
                'peak_gb':    peak_gb,
                'pct_vram':   pct_vram,
                'rec_bs':     rec_bs,
                'default_ok': default_ok,
                'elapsed':    elapsed,
                'error':      None,
            })

            status = f"max={max_bs:>4}  peak={peak_gb:.2f} GB"
            if pct_vram is not None:
                status += f" ({pct_vram:.0f}%)"
            status += f"  rec={rec_bs:>4}  [{elapsed:.1f}s]"
            print(status)

        except Exception as e:
            elapsed = time.time() - t0
            results.append({**exp, 'max_bs': None, 'peak_gb': None, 'pct_vram': None,
                             'rec_bs': None, 'default_ok': False,
                             'elapsed': elapsed, 'error': str(e)})
            print(f"ERROR — {str(e)[:55]}  [{elapsed:.1f}s]")

        finally:
            try:
                del model
            except Exception:
                pass
            clear_memory(device)

    # ── summary table ─────────────────────────────────────────────────────────
    W = 28
    print(f"\n{'━'*72}")
    print(f"  {'#':>3}  {'Experiment':<{W}}  {'Max BS':>6}  {'Peak Mem':>10}  {'OK?':>4}  {'Safe BS':>7}")
    print(f"  {'-'*3}  {'-'*W}  {'-'*6}  {'-'*10}  {'-'*4}  {'-'*7}")

    any_unsafe = False
    for r in results:
        if r['error']:
            print(f"  {r['num']:>3}  {r['name']:<{W}}  {'ERROR':>6}")
            continue

        if r['max_bs'] == 0:
            bs_str   = 'OOM@1'
            peak_str = '—'
            ok_str   = ' ✗!'
            rec_str  = 'N/A'
            any_unsafe = True
        else:
            bs_str   = str(r['max_bs'])
            pct      = f"({r['pct_vram']:.0f}%)" if r['pct_vram'] is not None else ''
            peak_str = f"{r['peak_gb']:.2f} GB {pct}"
            ok_str   = '  ✓' if r['default_ok'] else ' ✗!'
            rec_str  = str(r['rec_bs'])
            if not r['default_ok']:
                any_unsafe = True

        print(f"  {r['num']:>3}  {r['name']:<{W}}  {bs_str:>6}  {peak_str:>10}  {ok_str:>4}  {rec_str:>7}")

    print(f"  {'-'*3}  {'-'*W}  {'-'*6}  {'-'*10}  {'-'*4}  {'-'*7}")
    print(f"\n  OK?    ✓ = default batch_size ({default_bs}) fits  |  ✗! = reduce batch_size")
    print(f"  Safe BS = largest power-of-2 ≤ 75% of max (leaves headroom for dataloader workers)")
    if any_unsafe:
        print(f"\n  ✗! experiments will OOM with the current config batch_size={default_bs}.")
        print(f"     Set a per-experiment batch_size or use the Safe BS values above.")
    print(f"{'━'*72}\n")


if __name__ == '__main__':
    main()
