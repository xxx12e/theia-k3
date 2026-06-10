"""
8×4d (d=198) ResMLP — full 5 seeds {42, 123, 256, 777, 999}.

Completes the 2×2 (depth × expansion) ResMLP grid:
  4 blocks × 4d (d=280) - existing baseline
  4 blocks × 2d (d=383) - existing
  8 blocks × 2d (d=276) - existing
  8 blocks × 4d (d=198) - THIS RUN

Protocol: identical to 8×2d completion (resmlp_8x2d_completion.py):
  - 3-phase Gumbel pipeline (phase1/2/3 from theia_chain_v3_5seed)
  - AdamW lr=1e-3 cosine, batch 4096, FP16
  - EPOCHS_PER_TRY=150, MAX_RESTARTS=3, auto-restart ON
  - 5/10/50/100/500 step chain eval on 10K test chains
  - Restart count via stdout tee (phase1_with_restart_count)
  - Stop gate: DISABLED (complete distribution required)

Param target: ≈ 2,774,659 ± 3% (match 8×2d)
Actual at d=198: 2,780,707 (+0.22% vs 8×2d, +1.07% vs THEIA)

Output:
  multi_seed_results/resmlp_grid_sweep/8x4d_5seed.json
  multi_seed_results/resmlp_grid_sweep/8x4d_5seed.md
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
    DEVICE, SEEDS, phase2, phase3, evaluate,
)

# Config
CONFIG_NAME = '8blocks_x4d'
N_BLOCKS = 8
EXPANSION = 4
D = 198
EXPECTED_STEP_PARAMS = 2_780_707

ALL_SEEDS = [42, 123, 256, 777, 999]
TEST_STEPS = [5, 10, 50, 100, 500]

OUT_DIR = 'multi_seed_results/resmlp_grid_sweep'


def run_5_seeds():
    print(f'\n{"="*72}')
    print(f'  8×4d full 5-seed run')
    print(f'  Config: n_blocks={N_BLOCKS}, expansion={EXPANSION}d, d={D}')
    print(f'  Seeds: {ALL_SEEDS}')
    print(f'  Stop gate: DISABLED (complete distribution required)')
    print(f'{"="*72}')

    # Sanity: param count
    _probe = ResMLPChain(D, N_BLOCKS, EXPANSION)
    n_step = count_params(_probe.step)
    print(f'  step_params: {n_step:,}  (expected {EXPECTED_STEP_PARAMS:,})')
    assert n_step == EXPECTED_STEP_PARAMS, (
        f'Param mismatch: got {n_step}, expected {EXPECTED_STEP_PARAMS}')
    del _probe

    out = {
        'config': CONFIG_NAME, 'n_blocks': N_BLOCKS, 'expansion': EXPANSION, 'd': D,
        'step_params': n_step,
        'seeds': ALL_SEEDS,
        'p1': [], 'p2': [], 'p3': [],
        'elapsed': [],
        'p1_restart_count': [],
        'p1_converge_epoch': [],   # left as None; converge epoch lives in the stdout log
        'results': {str(s): [] for s in TEST_STEPS},
    }

    for si, seed in enumerate(ALL_SEEDS):
        print(f'\n--- 8×4d seed {si+1}/{len(ALL_SEEDS)}: {seed} ---')
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
        # P1 converge epoch appears in phase1's stdout; not captured here
        # (parse the log file if needed).
        out['p1_converge_epoch'].append(None)
        print(f'  Training: {elapsed:.0f}s | P1={p1:.2%} P2={p2:.2%} P3={p3:.2%} '
              f'| restarts={n_restart}')

        for steps in TEST_STEPS:
            acc, s1, _ = evaluate(model, steps, seed=seed + 5000, N=10000)
            out['results'][str(steps)].append(acc)
            tag = 'PASS' if acc > 0.99 else 'WARN' if acc > 0.95 else 'FAIL'
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

    # Aggregates
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
            'n':    len(v),
        }

    aggregates = {
        '500_strict':   mstd(strict_vals),
        '500_as_spec':  mstd(accs500),
        'strict_seeds': sorted(set(seeds) - restart_seed_set),
        'restart_seeds': sorted(restart_seed_set),
        'n_ge_99_strict': n_ge_99_strict,
        'n_ge_99_all':    n_ge_99_all,
        'per_step_5seed_mean': {
            str(s): mstd(out['results'][str(s)]) for s in TEST_STEPS
        },
        'wallclock': mstd(elapsed),
    }
    out['aggregates'] = aggregates

    # Save JSON
    json_path = os.path.join(OUT_DIR, '8x4d_5seed.json')
    with open(json_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\n  Saved: {json_path}')

    # --- Markdown report ---
    L = []
    L.append('# ResMLP 8×4d (d=198) — Full 5-seed Distribution\n')
    L.append(f'**Date**: {time.strftime("%Y-%m-%d %H:%M:%S")}')
    L.append(f'**Config**: n_blocks={N_BLOCKS}, expansion={EXPANSION}d, d={D}, '
             f'step_params={out["step_params"]:,}')
    L.append(f'**Protocol**: 3-phase Gumbel pipeline, AdamW lr=1e-3 cosine, batch 4096, '
             f'EPOCHS_PER_TRY=150, MAX_RESTARTS=3 (auto-restart ON), FP16')
    L.append('')
    L.append('## Per-seed full results (sorted by seed)\n')
    L.append('| Seed | P1 acc | P3 acc | restarts | t(s) | 5-step | 10-step | 50-step | 100-step | **500-step** |')
    L.append('|---|---|---|---|---|---|---|---|---|---|')
    for i, seed in enumerate(seeds):
        L.append(f'| {seed} | {p1[i]:.2%} | {p3[i]:.2%} | {rests[i]} | '
                 f'{elapsed[i]:.0f} | {out["results"]["5"][i]:.2%} | '
                 f'{out["results"]["10"][i]:.2%} | {out["results"]["50"][i]:.2%} | '
                 f'{out["results"]["100"][i]:.2%} | **{accs500[i]:.2%}** |')
    L.append('')
    L.append('## 500-step sorted by accuracy\n')
    sorted_pairs = sorted(zip(seeds, accs500), key=lambda x: x[1])
    L.append('| Rank | Seed | 500-step |')
    L.append('|---|---|---|')
    for r, (s, a) in enumerate(sorted_pairs, 1):
        L.append(f'| {r} (min)' if r == 1 else (f'| {r} (max)' if r == len(sorted_pairs) else f'| {r}') +
                 f' | {s} | {a:.2%} |')
    L.append('')

    L.append('## Aggregate statistics (500-step)\n')
    s = aggregates['500_strict']; a = aggregates['500_as_spec']
    L.append('| Aggregate | n | mean | sample std (ddof=1) | min | max | median | count ≥99% |')
    L.append('|---|---|---|---|---|---|---|---|')
    L.append(f'| **strict** (no restart) | {s["n"]} | {s["mean"]:.2%} | '
             f'{s["std"]:.2%} | {s["min"]:.2%} | {s["max"]:.2%} | {s["median"]:.2%} | '
             f'**{aggregates["n_ge_99_strict"]}/{s["n"]}** |')
    L.append(f'| **as-specified** (all 5) | {a["n"]} | {a["mean"]:.2%} | '
             f'{a["std"]:.2%} | {a["min"]:.2%} | {a["max"]:.2%} | {a["median"]:.2%} | '
             f'**{aggregates["n_ge_99_all"]}/{a["n"]}** |')
    L.append('')
    if not aggregates['restart_seeds']:
        L.append('All 5 seeds completed without Phase-1 auto-restart → strict ≡ as-specified.')
    else:
        L.append(f'Restart-triggered seeds (excluded from strict): {aggregates["restart_seeds"]}')
    L.append('')

    L.append('## Wallclock per seed\n')
    L.append(f'| Seed | wallclock (s) |')
    L.append(f'|---|---|')
    for i, seed in enumerate(seeds):
        L.append(f'| {seed} | {elapsed[i]:.0f} |')
    L.append('')
    w = aggregates['wallclock']
    L.append(f'**Wallclock aggregate**: mean {w["mean"]:.0f}s ± {w["std"]:.0f}s, '
             f'range [{w["min"]:.0f}, {w["max"]:.0f}]s')
    L.append('')

    L.append('## Per-step degradation curve (5-seed mean)\n')
    L.append('| Steps | mean | std | min | max |')
    L.append('|---|---|---|---|---|')
    for st in TEST_STEPS:
        d = aggregates['per_step_5seed_mean'][str(st)]
        L.append(f'| {st} | {d["mean"]:.2%} | {d["std"]:.2%} | {d["min"]:.2%} | {d["max"]:.2%} |')
    L.append('')

    # Comparison to other 3 configs (load from existing files)
    L.append('## 4-cell (depth × expansion) grid comparison\n')
    other_paths = {
        '4×4d (d=280) baseline': 'multi_seed_results/resmlp_backbone_ablation/raw.json',
        '4×2d (d=383)': 'multi_seed_results/resmlp_grid_sweep/4blocks_x2d/config_raw.json',
        '8×2d (d=276)': 'multi_seed_results/resmlp_grid_sweep/8x2d_5seed.json',
    }
    L.append('| Config | step_params | 500-step per seed (sorted by seed) | mean ± std | ≥99% |')
    L.append('|---|---|---|---|---|')
    for cname, cpath in other_paths.items():
        if not os.path.exists(cpath):
            L.append(f'| {cname} | (file not found) | — | — | — |')
            continue
        with open(cpath) as f:
            cd = json.load(f)
        c_seeds = cd.get('seeds_completed', cd.get('seeds', []))
        c_500 = cd.get('results', {}).get('500', [])
        c_step_p = cd.get('step_params', cd.get('param_count', {}).get('step'))
        c_step_str = f'{c_step_p:,}' if isinstance(c_step_p, int) else '?'
        c_per_seed = ', '.join(f'{v:.2%}' for v in c_500)
        c_mean = float(np.mean(c_500)) if c_500 else 0
        c_std = float(np.std(c_500, ddof=1)) if len(c_500) > 1 else 0
        c_n99 = sum(1 for v in c_500 if v >= 0.99)
        L.append(f'| {cname} | {c_step_str} | '
                 f'{c_per_seed} | {c_mean:.2%} ± {c_std:.2%} | {c_n99}/{len(c_500)} |')
    n_per_seed = ', '.join(f'{v:.2%}' for v in accs500)
    L.append(f'| **8×4d (d=198) — THIS RUN** | {out["step_params"]:,} | {n_per_seed} | '
             f'{a["mean"]:.2%} ± {a["std"]:.2%} | {aggregates["n_ge_99_all"]}/5 |')
    L.append('')

    L.append('## Summary line\n')
    L.append(f'8×4d (d=198, 2,780,707 params) at 500-step: **{a["mean"]:.2%} ± {a["std"]:.2%}** '
             f'over 5 seeds, **{aggregates["n_ge_99_all"]}/5** ≥99%, range '
             f'[{a["min"]:.2%}, {a["max"]:.2%}], all {len(seeds)} seeds compute-matched '
             f'(restarts: {sum(rests)}).')
    L.append('')

    md_path = os.path.join(OUT_DIR, '8x4d_5seed.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L))
    print(f'  Saved: {md_path}')

    # Console summary
    print(f'\n{"="*72}')
    print(f'  8×4d FINAL DISTRIBUTION (500-step)')
    print(f'{"="*72}')
    print(f'  {"seed":>5} {"500-step":>10} {"restart?":>9} {"wallclock":>11}')
    for sd, ac, r, e in zip(seeds, accs500, rests, elapsed):
        print(f'  {sd:>5} {ac*100:>9.2f}% {r:>9} {e:>10.0f}s')
    s_agg = aggregates['500_strict']
    a_agg = aggregates['500_as_spec']
    print(f'\n  strict (n={s_agg["n"]}): {s_agg["mean"]*100:.2f}% ± '
          f'{s_agg["std"]*100:.2f}%, '
          f'{aggregates["n_ge_99_strict"]}/{s_agg["n"]} ≥99%')
    print(f'  as-spec (n={a_agg["n"]}): {a_agg["mean"]*100:.2f}% ± {a_agg["std"]*100:.2f}%, '
          f'{aggregates["n_ge_99_all"]}/{a_agg["n"]} ≥99%')


def main():
    out = run_5_seeds()
    report(out)


if __name__ == '__main__':
    main()
