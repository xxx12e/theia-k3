"""
Tuned THEIA: the tuned-Transformer recipe applied to the THEIA backbone
(paper App. G; symmetric-tuning control).

The paper compares THEIA's 9.2-min wall-clock under default lr=1e-3 against
the Transformer's 51.5-min matched and 28.9-min tuned (lr=1e-4 + warmup).
For a symmetric comparison, the exact same tuned recipe is applied to
THEIA's IsisV9 backbone:

  - lr = 1e-4   (was 1e-3)
  - betas = (0.9, 0.98)  (was AdamW default)
  - linear warmup 5 epochs then cosine
  - weight_decay = 0.01
  - batch = 2048   (was 4096)
  - grad_clip = 1.0
  - 150-epoch cap (was 200)
  - same Kleene early-stop: 12/12 + acc>99.9% on 2 consecutive checks

Backbone architecture: byte-identical to theia_5seed_v2.py IsisV9 (via
_theia_model_def import). No restart mechanism (matches theia_5seed_v2.py).

Output:
  tuned_theia_5seed_results.json    (per-seed P1 trajectory + Kleene + wall-clock)
  tuned_theia_seed{N}_log.txt        (per-seed stdout dump)
  tuned_theia_5seed_report.md        (gate decision + per-seed table)

Usage:
    python tuned_theia_5seed.py --pilot         # seed 42 only
    python tuned_theia_5seed.py                 # all 5 seeds
    python tuned_theia_5seed.py --seeds 42 123  # specific seeds
"""
import argparse
import json
import math
import os
import sys
import time

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import GradScaler, autocast

from _theia_model_def import (
    IsisV9, build_dataset, run_kleene,
    DEVICE, BATCH as _DEFAULT_BATCH,    # default 4096; overridden to 2048 below
)

# --- tuned recipe (matches tf8l_5seed_tuned.py) ---
PEAK_LR        = 1e-4
BETAS          = (0.9, 0.98)
WEIGHT_DECAY   = 0.01
WARMUP_EPOCHS  = 5
GRAD_CLIP      = 1.0
BATCH          = 2048               # override _theia_model_def's 4096
MAX_EPOCHS     = 150                # tuned recipe cap (paper used 200 default)
CLASS_WEIGHTS  = [1.0, 1.0, 2.0]    # same as theia_5seed_v2

# Early-stop criterion (matches theia_5seed_v2 except cadence; tuned uses ep%5)
EARLY_STOP_OVERALL = 0.999
KLEENE_EVERY      = 5      # tuned recipe: every 5 ep after acc>0.95
KLEENE_AFTER_ACC  = 0.95
PILOT_WALLCLOCK_CAP_MIN = 60  # pilot hard cap: abort if 12/12 not reached

SEEDS_ALL = [42, 123, 256, 777, 999]


def lr_lambda(epoch):
    """Linear warmup 5 epochs, then cosine to 0 over remaining (MAX_EPOCHS - 5)."""
    if epoch < WARMUP_EPOCHS:
        return (epoch + 1) / WARMUP_EPOCHS
    progress = (epoch - WARMUP_EPOCHS) / max(1, MAX_EPOCHS - WARMUP_EPOCHS)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def run_seed(seed, log_path=None, is_pilot=False):
    """Run tuned-recipe THEIA training for one seed; returns metrics + wall-clock.

    If is_pilot=True, aborts training if 12/12 is not reached within
    PILOT_WALLCLOCK_CAP_MIN (60 min wall-clock)."""
    log_f = open(log_path, 'w', encoding='utf-8') if log_path else None
    def log(m):
        print(m)
        if log_f:
            log_f.write(m + '\n'); log_f.flush()

    log('=' * 72)
    log(f'  Tuned THEIA — seed {seed}')
    log(f'  Recipe: lr={PEAK_LR} beta={BETAS} warmup={WARMUP_EPOCHS}ep wd={WEIGHT_DECAY} '
        f'bs={BATCH} grad_clip={GRAD_CLIP} max_ep={MAX_EPOCHS}')
    log('=' * 72)

    # Determinism setup (matches theia_5seed_v2.py L32-36)
    torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    log(f'  Building dataset (2M, seed {seed}) ...')
    AF, BF, DF, SB, S_UNK, A_UNK, B_UNK, D_UNK, AR, RL, OP, TARGET, N = build_dataset(seed)
    split = int(N * 0.8)

    # Re-seed before model init (matches theia_5seed_v2.py L302)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = IsisV9().to(DEVICE)
    params = sum(p.numel() for p in model.parameters())
    log(f'  Params: {params:,}')
    assert params == 2_751_232, f'param count mismatch {params}'

    cw = torch.tensor(CLASS_WEIGHTS, device=DEVICE)
    opt = optim.AdamW(model.parameters(), lr=PEAK_LR, betas=BETAS, weight_decay=WEIGHT_DECAY)
    sched = optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = GradScaler('cuda')

    def gb(idx):
        return (AF[idx], BF[idx], DF[idx], SB[idx], S_UNK[idx],
                A_UNK[idx], B_UNK[idx], D_UNK[idx],
                AR[idx], RL[idx], OP[idx])

    # Tracking
    best_acc = 0.0
    converge_epoch = MAX_EPOCHS
    converge_wall_min = None
    first_12_12_epoch = None
    first_12_12_wall_min = None
    first_999_epoch = None
    first_999_wall_min = None
    last_kleene_passed = 0
    final_kleene = {}
    kleene_streak = 0
    prev_passed = -1

    t_start = time.time()
    early_stopped = False

    # Per-epoch eval (cheap)
    def eval_overall():
        model.eval()
        with torch.no_grad():
            tidx = torch.arange(split, N, device=DEVICE)
            preds = []
            for j in range(0, len(tidx), BATCH * 4):
                bi = tidx[j:j + BATCH * 4]
                with autocast('cuda'):
                    out = model(*gb(bi))
                    p = model.classify(out)
                preds.append(p)
            return (torch.cat(preds) == TARGET[split:]).float().mean().item()

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        perm = torch.randperm(split, device=DEVICE)
        tl = 0.0; nb = 0
        for i in range(0, split, BATCH):
            idx = perm[i:i + BATCH]
            with autocast('cuda'):
                out = model(*gb(idx))
                logits = out @ model.sv.weight.T
                loss = F.cross_entropy(logits, TARGET[idx], weight=cw)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(opt); scaler.update()
            tl += loss.item(); nb += 1
        sched.step()
        acc = eval_overall()
        if acc > best_acc:
            best_acc = acc
        wall_min = (time.time() - t_start) / 60.0

        if first_999_epoch is None and acc >= EARLY_STOP_OVERALL:
            first_999_epoch = epoch
            first_999_wall_min = wall_min

        # Pilot hard cap: 60 min wall-clock
        if is_pilot and wall_min > PILOT_WALLCLOCK_CAP_MIN and first_12_12_epoch is None:
            log(f'\n  *** PILOT HARD CAP: wall_min={wall_min:.2f} > {PILOT_WALLCLOCK_CAP_MIN}, '
                f'12/12 not reached — aborting to save GPU time')
            early_stopped = True
            converge_epoch = -1  # signal: hit cap
            converge_wall_min = wall_min  # mark abort time
            break

        # Kleene check — every 5 ep after best > 0.95 (tuned cadence)
        if epoch % KLEENE_EVERY == 0 and best_acc > KLEENE_AFTER_ACC:
            kleene, passed = run_kleene(model)
            last_kleene_passed = passed
            final_kleene = kleene
            log(f'  ep {epoch:3d} loss={tl/nb:.4f} acc={acc:.4f} best={best_acc:.4f} '
                f'lr={opt.param_groups[0]["lr"]:.2e} kleene={passed}/12 wall={wall_min:.2f}m')

            if passed == 12 and first_12_12_epoch is None:
                first_12_12_epoch = epoch
                first_12_12_wall_min = wall_min
                log(f'    *** first 12/12 Kleene at ep {epoch} ({wall_min:.2f} min)')

            # Paper convergence: 12/12 + acc>0.999 on 2 consecutive checks
            if passed == 12 and acc > EARLY_STOP_OVERALL and prev_passed == 12:
                kleene_streak += 1
                if kleene_streak >= 1:
                    converge_epoch = epoch
                    converge_wall_min = wall_min
                    log(f'    *** paper-criterion converged @ {epoch}: 2x 12/12 + acc>0.999 '
                        f'({wall_min:.2f} min)')
                    early_stopped = True
                    break
            else:
                kleene_streak = 0
            prev_passed = passed
        else:
            if epoch % 10 == 0:
                log(f'  ep {epoch:3d} loss={tl/nb:.4f} acc={acc:.4f} best={best_acc:.4f} '
                    f'lr={opt.param_groups[0]["lr"]:.2e} wall={wall_min:.2f}m')

    total_min = (time.time() - t_start) / 60.0

    # Final Kleene + per-class
    final_kleene, final_passed = run_kleene(model)
    model.eval()
    with torch.no_grad():
        tidx = torch.arange(split, N, device=DEVICE)
        preds_all = []
        for j in range(0, len(tidx), BATCH * 4):
            bi = tidx[j:j + BATCH * 4]
            with autocast('cuda'):
                preds_all.append(model.classify(model(*gb(bi))))
        preds_all = torch.cat(preds_all)
        labels = TARGET[split:]
        per_class = {}
        for vid, vn in [(0, 'False'), (1, 'True'), (2, 'Unknown')]:
            m = labels == vid
            if m.sum() > 0:
                per_class[vn] = (preds_all[m] == labels[m]).float().mean().item()

    result = {
        'seed': seed,
        'recipe': {
            'lr': PEAK_LR, 'betas': list(BETAS), 'wd': WEIGHT_DECAY,
            'warmup_epochs': WARMUP_EPOCHS, 'grad_clip': GRAD_CLIP,
            'batch': BATCH, 'max_epochs': MAX_EPOCHS,
            'class_weights_FTU': CLASS_WEIGHTS,
        },
        'params': params,
        'first_12_12_epoch': first_12_12_epoch,
        'first_12_12_wall_min': first_12_12_wall_min,
        'first_999_epoch': first_999_epoch,
        'first_999_wall_min': first_999_wall_min,
        'converge_epoch_paper_criterion': converge_epoch,
        'converge_wall_min_paper_criterion': converge_wall_min,
        'early_stopped': early_stopped,
        'final_overall_acc': best_acc,
        'final_kleene_passed': final_passed,
        'final_kleene_per_rule': final_kleene,
        'per_class_acc': per_class,
        'total_wall_min': total_min,
    }

    log('')
    log(f'  Seed {seed} DONE: total {total_min:.2f} min '
        f'(first 12/12 @ {first_12_12_wall_min} min, '
        f'paper-converge @ {converge_wall_min} min, final acc {best_acc:.4f}, '
        f'kleene {final_passed}/12)')

    if log_f: log_f.close()
    del model
    torch.cuda.empty_cache()
    return result


def aggregate(results, baseline_theia_min=9.2, tuned_tf_min=28.9):
    """Compute mean/std/ratio table."""
    valid = [r for r in results if r and r.get('first_12_12_wall_min') is not None]
    if not valid:
        return {'n_seeds': 0, 'note': 'no seed reached 12/12'}
    seeds = [r['seed'] for r in valid]
    times_12 = [r['first_12_12_wall_min'] for r in valid]
    times_conv = [r['converge_wall_min_paper_criterion'] for r in valid if r.get('converge_wall_min_paper_criterion') is not None]
    accs = [r['final_overall_acc'] for r in valid]
    return {
        'n_seeds': len(seeds), 'seeds': seeds,
        'first_12_12': {
            'mean': float(np.mean(times_12)),
            'std':  float(np.std(times_12, ddof=1)) if len(times_12) > 1 else 0.0,
            'min':  float(np.min(times_12)),
            'max':  float(np.max(times_12)),
            'per_seed': times_12,
        },
        'paper_converge': {
            'mean': float(np.mean(times_conv)) if times_conv else None,
            'std':  float(np.std(times_conv, ddof=1)) if len(times_conv) > 1 else 0.0,
            'n':    len(times_conv),
            'per_seed': times_conv,
        },
        'final_overall_acc_mean': float(np.mean(accs)),
        'final_overall_acc_std':  float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
        'baseline_default_lr_THEIA_min': baseline_theia_min,
        'tuned_TF_TFTuned_min':          tuned_tf_min,
        'narrowing_ratio_tuned_TF_over_tuned_THEIA': (
            tuned_tf_min / float(np.mean(times_12)) if times_12 else None
        ),
        'tuned_THEIA_vs_default_THEIA_ratio': (
            float(np.mean(times_12)) / baseline_theia_min if times_12 else None
        ),
    }


def gate_decide_pilot(result):
    """Pilot (seed 42) gate: maps Kleene 12/12 wall-clock time to an action.

    The 60-min hard cap is handled in run_seed;
    result['converge_epoch_paper_criterion'] == -1 signals abort due to cap."""
    if result is None:
        return 'NO_RESULT', 'pilot returned None'
    # Check for hard-cap abort
    if result.get('converge_epoch_paper_criterion') == -1:
        wt = result.get('converge_wall_min_paper_criterion', PILOT_WALLCLOCK_CAP_MIN)
        return 'STOP_FOR_REVIEW_TIMEOUT', (
            f'seed 42 hit {PILOT_WALLCLOCK_CAP_MIN}-min hard cap without reaching 12/12 '
            f'(aborted at {wt:.2f} min, final {result["final_kleene_passed"]}/12) — '
            f'STOP, do not extend, paper revision needed'
        )
    t = result.get('first_12_12_wall_min')
    if t is None:
        return 'NO_CONVERGE', (
            f'seed 42 did NOT reach 12/12 in {MAX_EPOCHS} epochs '
            f'(final {result["final_kleene_passed"]}/12) — STOP, paper revision needed'
        )
    if t <= 15:
        return 'EXTEND_FULL_PLUS_LR_SWEEP', f'seed 42 12/12 @ {t:.2f} min ≤ 15 min — H3 stands, run 5 seeds + lr sweep'
    if t <= 30:
        return 'EXTEND_FULL_NO_LR_SWEEP', f'seed 42 12/12 @ {t:.2f} min in (15, 30] — H3 partial, run 5 seeds, skip lr sweep'
    if t <= 60:
        return 'EXTEND_FULL_DEGRADE', f'seed 42 12/12 @ {t:.2f} min in (30, 60] — H3 degrade to "default lr advantage", run 5 seeds for confirmation'
    return 'STOP_FOR_REVIEW', f'seed 42 12/12 @ {t:.2f} min > 60 min — STOP, do not extend, write pilot to paper scope'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pilot', action='store_true', help='Only seed 42 (pilot)')
    p.add_argument('--seeds', type=int, nargs='+', default=None,
                   help='Specific seeds to run')
    args = p.parse_args()

    if args.pilot:
        seeds_to_run = [42]
    elif args.seeds:
        seeds_to_run = args.seeds
    else:
        seeds_to_run = SEEDS_ALL

    print(f'\nTuned THEIA run: seeds = {seeds_to_run}, pilot_mode = {args.pilot}')
    results = []
    for seed in seeds_to_run:
        log_path = f'tuned_theia_seed{seed}_log.txt'
        r = run_seed(seed, log_path=log_path, is_pilot=args.pilot)
        results.append(r)

    out = {
        'recipe': {
            'lr': PEAK_LR, 'betas': list(BETAS), 'wd': WEIGHT_DECAY,
            'warmup_epochs': WARMUP_EPOCHS, 'grad_clip': GRAD_CLIP,
            'batch': BATCH, 'max_epochs': MAX_EPOCHS,
        },
        'seeds_run': seeds_to_run,
        'pilot_mode': args.pilot,
        'per_seed': results,
    }

    if args.pilot and len(results) == 1:
        gate, reason = gate_decide_pilot(results[0])
        out['pilot_gate'] = gate
        out['pilot_gate_reason'] = reason
        print(f'\n{"="*72}')
        print(f'  PILOT GATE: {gate}')
        print(f'  Reason:    {reason}')
        print(f'{"="*72}')
    else:
        out['aggregate'] = aggregate(results)

    out_path = 'tuned_theia_5seed_results.json' if not args.pilot else 'tuned_theia_pilot_results.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nSaved: {out_path}')


if __name__ == '__main__':
    main()
