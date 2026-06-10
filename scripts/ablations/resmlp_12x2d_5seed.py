"""
12-block × 2d ResMLP, full 5 seeds.

The existing ResMLP grid (4 and 8 blocks) maxes at 97.95% ± 2.03% on the
500-step chain. Does scaling to 12 blocks push ResMLP over 99%, or is the
4-domain-split prior still required?

Config: n_blocks=12, expansion=2d, d=226, step_params=2,747,003 (target 2,751,232,
delta -0.15%, parameter-matched to full IsisV9).

Recipe: identical to resmlp_8x4d_5seed.py:
  - 3-phase Gumbel pipeline (phase1/2/3 from theia_chain_v3_5seed)
  - AdamW lr=1e-3 cosine, batch 4096, FP16
  - EPOCHS_PER_TRY=150, MAX_RESTARTS=3, auto-restart ON
  - 5/10/50/100/500 step chain eval on 10K test chains

Decision thresholds on 500-step:
  - 5/5 ≥99%: depth alone closes the gap
  - 4/5 ≥99%: scaling helps but the architectural prior wins on robustness
  - ≤3/5 ≥99%: depth alone cannot close the gap

Output: multi_seed_results/resmlp_grid_sweep/12x2d_5seed.{json,md}
"""
import json
import os
import sys
import time

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import numpy as np
import torch

from resmlp_grid_sweep import (
    ResMLPChain, count_params,
    phase1_with_restart_count,
)
from theia_chain_v3_5seed import (
    DEVICE, phase2, phase3, evaluate,
)

# Config
CONFIG_NAME = '12blocks_x2d'
N_BLOCKS = 12
EXPANSION = 2
D = 226
EXPECTED_STEP_PARAMS = 2_747_003   # within 0.15% of IsisV9's 2,751,232

ALL_SEEDS = [42, 123, 256, 777, 999]
TEST_STEPS = [5, 10, 50, 100, 500]

OUT_DIR = 'multi_seed_results/resmlp_grid_sweep'


def run_5_seeds():
    print(f'\n{"="*72}')
    print(f'  12 blocks x 2d full 5-seed run')
    print(f'  Config: n_blocks={N_BLOCKS}, expansion={EXPANSION}d, d={D}')
    print(f'  Seeds: {ALL_SEEDS}')
    print(f'{"="*72}')

    _probe = ResMLPChain(D, N_BLOCKS, EXPANSION)
    n_step = count_params(_probe.step)
    print(f'  step_params: {n_step:,}  (expected {EXPECTED_STEP_PARAMS:,}, '
          f'delta vs THEIA 2,751,232: {(n_step-2_751_232)/2_751_232*100:+.2f}%)')
    assert n_step == EXPECTED_STEP_PARAMS, f'param mismatch: got {n_step}, expected {EXPECTED_STEP_PARAMS}'
    del _probe

    out = {
        'config': CONFIG_NAME, 'n_blocks': N_BLOCKS, 'expansion': EXPANSION, 'd': D,
        'step_params': n_step,
        'seeds': ALL_SEEDS,
        'p1': [], 'p2': [], 'p3': [],
        'elapsed': [],
        'p1_restart_count': [],
        'p1_converge_epoch': [],
        'results': {str(s): [] for s in TEST_STEPS},
    }

    for si, seed in enumerate(ALL_SEEDS):
        print(f'\n--- 12x2d seed {si+1}/{len(ALL_SEEDS)}: {seed} ---')
        torch.manual_seed(seed); np.random.seed(seed)
        model = ResMLPChain(D, N_BLOCKS, EXPANSION).to(DEVICE)
        t0 = time.time()
        p1, n_restart = phase1_with_restart_count(model.step, seed)
        p2 = phase2(model, train_seed=seed)
        p3 = phase3(model, train_seed=seed)
        elapsed = time.time() - t0
        out['p1'].append(p1)
        out['p2'].append(p2)
        out['p3'].append(p3)
        out['elapsed'].append(elapsed)
        out['p1_restart_count'].append(n_restart)
        out['p1_converge_epoch'].append(None)
        print(f'  Training: {elapsed:.0f}s | P1={p1:.2%} P2={p2:.2%} P3={p3:.2%} | restarts={n_restart}')

        for steps in TEST_STEPS:
            acc, s1, _ = evaluate(model, steps, seed=seed + 5000, N=10000)
            out['results'][str(steps)].append(acc)
            tag = 'PASS' if acc > 0.99 else ('WARN' if acc > 0.95 else 'FAIL')
            print(f'  {steps:>4d}-step: {acc:.2%} (step1={s1:.2%}) {tag}')

        del model; torch.cuda.empty_cache()

    return out


def report(out):
    seeds = out['seeds']
    accs500 = out['results']['500']
    rests = out['p1_restart_count']
    elapsed = out['elapsed']
    p1 = out['p1']
    p3 = out['p3']

    restart_seed_set = {seeds[i] for i, r in enumerate(rests) if r > 0}
    strict_idx = [i for i, s in enumerate(seeds) if s not in restart_seed_set]
    strict_vals = [accs500[i] for i in strict_idx]
    n_ge_99_strict = sum(1 for v in strict_vals if v >= 0.99)
    n_ge_99_all = sum(1 for v in accs500 if v >= 0.99)

    def mstd(v):
        if not v: return None
        return {
            'mean': float(np.mean(v)),
            'std':  float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
            'min':  float(np.min(v)),
            'max':  float(np.max(v)),
            'median': float(np.median(v)),
            'n': len(v),
        }

    aggregates = {
        '500_strict':  mstd(strict_vals),
        '500_as_spec': mstd(accs500),
        'strict_seeds': sorted(set(seeds) - restart_seed_set),
        'restart_seeds': sorted(restart_seed_set),
        'n_ge_99_strict': n_ge_99_strict,
        'n_ge_99_all':    n_ge_99_all,
        'per_step_5seed_mean': {str(s): mstd(out['results'][str(s)]) for s in TEST_STEPS},
        'wallclock': mstd(elapsed),
    }
    out['aggregates'] = aggregates

    json_path = os.path.join(OUT_DIR, '12x2d_5seed.json')
    with open(json_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\n  Saved: {json_path}')

    # --- Markdown report ---
    L = [f'# ResMLP 12 blocks x 2d (d={D}) — Full 5-seed\n']
    L.append(f'**Date**: {time.strftime("%Y-%m-%d %H:%M:%S")}')
    L.append(f'**Config**: n_blocks={N_BLOCKS}, expansion={EXPANSION}d, d={D}, '
             f'step_params={out["step_params"]:,} (delta vs THEIA 2,751,232: '
             f'{(out["step_params"]-2_751_232)/2_751_232*100:+.2f}%)')
    L.append(f'**Protocol**: 3-phase Gumbel pipeline, AdamW lr=1e-3 cosine, batch 4096, '
             f'EPOCHS_PER_TRY=150, MAX_RESTARTS=3 (auto-restart ON), FP16')
    L.append('')
    L.append('## Per-seed (sorted by seed)\n')
    L.append('| Seed | P1 | P3 | restarts | t(s) | 5 | 10 | 50 | 100 | **500** |')
    L.append('|---|---|---|---|---|---|---|---|---|---|')
    for i, sd in enumerate(seeds):
        L.append(f'| {sd} | {p1[i]:.2%} | {p3[i]:.2%} | {rests[i]} | {elapsed[i]:.0f} | '
                 f'{out["results"]["5"][i]:.2%} | {out["results"]["10"][i]:.2%} | '
                 f'{out["results"]["50"][i]:.2%} | {out["results"]["100"][i]:.2%} | '
                 f'**{accs500[i]:.2%}** |')
    L.append('')

    L.append('## Aggregate (500-step)\n')
    s = aggregates['500_strict']; a = aggregates['500_as_spec']
    L.append('| Aggregate | n | mean | std (ddof=1) | min | max | count >=99% |')
    L.append('|---|---|---|---|---|---|---|')
    L.append(f'| **strict** (no restart) | {s["n"]} | {s["mean"]:.2%} | '
             f'{s["std"]:.2%} | {s["min"]:.2%} | {s["max"]:.2%} | '
             f'**{aggregates["n_ge_99_strict"]}/{s["n"]}** |')
    L.append(f'| **as-spec** (all 5)      | {a["n"]} | {a["mean"]:.2%} | '
             f'{a["std"]:.2%} | {a["min"]:.2%} | {a["max"]:.2%} | '
             f'**{aggregates["n_ge_99_all"]}/{a["n"]}** |')
    L.append('')
    if not aggregates['restart_seeds']:
        L.append('All seeds completed without P1 auto-restart -> strict ≡ as-spec.')
    else:
        L.append(f'Restart-triggered seeds (excluded from strict): {aggregates["restart_seeds"]}')
    L.append('')

    L.append('## Comparison with depth/expansion grid\n')
    L.append('| Config | step_params | mean ± std (500) | >=99% |')
    L.append('|---|---|---|---|')
    grid = {
        '4x4d (d=280)':     ('multi_seed_results/resmlp_backbone_ablation/raw.json',     2_848_547),
        '4x2d (d=383)':     ('multi_seed_results/resmlp_grid_sweep/4blocks_x2d/config_raw.json', None),
        '8x2d (d=276)':     ('multi_seed_results/resmlp_grid_sweep/8x2d_5seed.json',     2_774_659),
        '8x4d (d=198)':     ('multi_seed_results/resmlp_grid_sweep/8x4d_5seed.json',     2_780_707),
    }
    for cname, (cpath, p_known) in grid.items():
        if not os.path.exists(cpath):
            L.append(f'| {cname} | (missing) | -- | -- |')
            continue
        cd = json.load(open(cpath))
        c500 = cd.get('results', {}).get('500', [])
        if not c500: c500 = cd.get('500step', [])
        if not c500:
            L.append(f'| {cname} | -- | (no 500-step) | -- |')
            continue
        m = float(np.mean(c500)); st = float(np.std(c500, ddof=1)) if len(c500) > 1 else 0.0
        n99 = sum(1 for v in c500 if v >= 0.99)
        ps = f'{p_known:,}' if p_known else '?'
        L.append(f'| {cname} | {ps} | {m:.2%} ± {st:.2%} | {n99}/{len(c500)} |')
    L.append(f'| **12x2d (d={D}) — THIS RUN** | {out["step_params"]:,} | '
             f'{a["mean"]:.2%} ± {a["std"]:.2%} | **{aggregates["n_ge_99_all"]}/5** |')
    L.append('')

    L.append('## Gate D verdict\n')
    if aggregates['n_ge_99_all'] >= 5:
        L.append('**5/5 >=99%**: depth alone closes the gap — falsifies H1 '
                 '(architectural prior alone is not the explanation).')
    elif aggregates['n_ge_99_all'] >= 4:
        L.append('**4/5 >=99%**: depth helps but architectural prior still wins on '
                 'robustness — H1 partially weakened.')
    else:
        L.append(f'**{aggregates["n_ge_99_all"]}/5 >=99%**: depth alone INSUFFICIENT — '
                 'H1 stands.')
    L.append('')

    md_path = os.path.join(OUT_DIR, '12x2d_5seed.md')
    open(md_path, 'w', encoding='utf-8').write('\n'.join(L))
    print(f'  Saved: {md_path}')

    print(f'\n{"="*72}')
    print(f'  12x2d FINAL (500-step)')
    print(f'{"="*72}')
    for sd, ac, r in zip(seeds, accs500, rests):
        print(f'  seed {sd:>4}: {ac*100:6.2f}%  restart={r}')
    print(f'\n  AGGREGATE (as-spec): {a["mean"]*100:.2f}% +/- {a["std"]*100:.2f}%, '
          f'{aggregates["n_ge_99_all"]}/5 >=99%')


def main():
    out = run_5_seeds()
    report(out)


if __name__ == '__main__':
    main()
