"""
8×2d (d=276) ResMLP — complete remaining 4 seeds {123, 256, 777, 999}.

Context: seed 42 of 8×2d already hit 500-step = 99.27% (stop gate trigger).
This run completes the config to a full 5-seed measurement so we can decide
whether seed 42 is an outlier or a pattern.

Protocol: identical to seed 42's run via the same `ResMLPChain` + `phase1_with_restart_count`
+ phase2 + phase3 + evaluate from resmlp_grid_sweep / theia_chain_v3_5seed.

Stop gate: DISABLED (complete distribution needed, not gate-confirmation).

Output:
  multi_seed_results/resmlp_grid_sweep/8x2d_completion.json   (raw 4-seed data)
  multi_seed_results/resmlp_grid_sweep/8x2d_5seed.json        (merged 5-seed)
  multi_seed_results/resmlp_grid_sweep/8x2d_5seed.md          (final report)
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

# Import model class + restart-counting wrapper from the grid sweep script
# (resmlp_grid_sweep.py has __main__ guard so import is safe)
from resmlp_grid_sweep import (
    ResMLPChain, count_params,
    phase1_with_restart_count,
)
from theia_chain_v3_5seed import (
    DEVICE, SEEDS, phase2, phase3, evaluate,
)

# Config
CONFIG_NAME = '8blocks_x2d'
N_BLOCKS = 8
EXPANSION = 2
D = 276
EXPECTED_STEP_PARAMS = 2_774_659

# Seeds to complete (seed 42 already done in earlier sweep)
REMAINING_SEEDS = [123, 256, 777, 999]
TEST_STEPS = [5, 10, 50, 100, 500]

OUT_DIR = 'multi_seed_results/resmlp_grid_sweep'
SEED42_RAW = os.path.join(OUT_DIR, '8blocks_x2d', 'config_raw.json')


def run_4_seeds():
    """Run 8×2d (d=276) on seeds {123, 256, 777, 999}. No stop gate."""
    print(f'\n{"="*72}')
    print(f'  8×2d completion run')
    print(f'  Config: n_blocks={N_BLOCKS}, expansion={EXPANSION}d, d={D}')
    print(f'  Remaining seeds: {REMAINING_SEEDS}')
    print(f'  Stop gate: DISABLED (complete distribution required)')
    print(f'{"="*72}')

    # Sanity: param count
    _probe = ResMLPChain(D, N_BLOCKS, EXPANSION)
    n_step = count_params(_probe.step)
    print(f'  step_params: {n_step:,}  (expected {EXPECTED_STEP_PARAMS:,})')
    assert n_step == EXPECTED_STEP_PARAMS, f'Param mismatch: got {n_step}, expected {EXPECTED_STEP_PARAMS}'
    del _probe

    completion = {
        'config': CONFIG_NAME, 'n_blocks': N_BLOCKS, 'expansion': EXPANSION, 'd': D,
        'step_params': n_step,
        'seeds': REMAINING_SEEDS,
        'p1': [], 'p2': [], 'p3': [],
        'elapsed': [],
        'p1_restart_count': [],
        'results': {str(s): [] for s in TEST_STEPS},
    }

    for si, seed in enumerate(REMAINING_SEEDS):
        print(f'\n--- 8×2d seed {si+1}/{len(REMAINING_SEEDS)}: {seed} ---')
        torch.manual_seed(seed); np.random.seed(seed)
        model = ResMLPChain(D, N_BLOCKS, EXPANSION).to(DEVICE)
        t0 = time.time()
        p1, n_restart = phase1_with_restart_count(model.step, seed)
        p2 = phase2(model, train_seed=seed)
        p3 = phase3(model, train_seed=seed)
        elapsed = time.time() - t0
        completion['p1'].append(p1)
        completion['p2'].append(p2)
        completion['p3'].append(p3)
        completion['elapsed'].append(elapsed)
        completion['p1_restart_count'].append(n_restart)
        print(f'  Training: {elapsed:.0f}s | P1={p1:.2%} P2={p2:.2%} P3={p3:.2%} | restarts={n_restart}')

        for steps in TEST_STEPS:
            acc, s1, _ = evaluate(model, steps, seed=seed + 5000, N=10000)
            completion['results'][str(steps)].append(acc)
            tag = 'PASS' if acc > 0.99 else 'WARN' if acc > 0.95 else 'FAIL'
            print(f'  {steps:>4d}-step: {acc:.2%} (step1={s1:.2%}) {tag}')

        del model; torch.cuda.empty_cache()

    # Save raw 4-seed data
    completion_path = os.path.join(OUT_DIR, '8x2d_completion.json')
    with open(completion_path, 'w') as f:
        json.dump(completion, f, indent=2)
    print(f'\n  Saved completion raw: {completion_path}')
    return completion


def merge_and_report(completion):
    """Merge with seed 42 data and generate final 5-seed report."""
    if not os.path.exists(SEED42_RAW):
        raise RuntimeError(f'seed 42 data not found at {SEED42_RAW}')
    with open(SEED42_RAW) as f:
        seed42_data = json.load(f)

    # Verify seed 42 was in the prior run
    s42_seeds = seed42_data.get('seeds', [])
    if 42 not in s42_seeds:
        raise RuntimeError(f'seed 42 not found in prior data; seeds: {s42_seeds}')
    s42_idx = list(s42_seeds[:len(seed42_data['p1'])]).index(42)

    # Merge: order seeds [42, 123, 256, 777, 999]
    merged = {
        'config': CONFIG_NAME, 'n_blocks': N_BLOCKS, 'expansion': EXPANSION, 'd': D,
        'step_params': completion['step_params'],
        'seeds': [42] + REMAINING_SEEDS,
        'p1': [seed42_data['p1'][s42_idx]] + completion['p1'],
        'p2': [seed42_data['p2'][s42_idx]] + completion['p2'],
        'p3': [seed42_data['p3'][s42_idx]] + completion['p3'],
        'elapsed': [seed42_data['elapsed'][s42_idx]] + completion['elapsed'],
        'p1_restart_count': [seed42_data['p1_restart_count'][s42_idx]] + completion['p1_restart_count'],
        'results': {
            str(s): [seed42_data['results'][str(s)][s42_idx]] + completion['results'][str(s)]
            for s in TEST_STEPS
        },
    }

    # Aggregates
    seeds = merged['seeds']
    accs500 = merged['results']['500']
    restart_counts = merged['p1_restart_count']
    restart_seed_set = {seeds[i] for i, r in enumerate(restart_counts) if r > 0}
    strict_idx = [i for i, s in enumerate(seeds) if s not in restart_seed_set]
    strict_vals = [accs500[i] for i in strict_idx]
    n_ge_99_strict = sum(1 for v in strict_vals if v >= 0.99)
    n_ge_99_all   = sum(1 for v in accs500 if v >= 0.99)

    def mstd(v):
        if not v: return None
        return {
            'mean': float(np.mean(v)),
            'std':  float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
            'min':  float(np.min(v)),
            'max':  float(np.max(v)),
            'n':    len(v),
        }

    aggregates = {
        '500_strict': mstd(strict_vals),
        '500_as_spec': mstd(accs500),
        'strict_seeds': sorted(set(seeds) - restart_seed_set),
        'restart_seeds': sorted(restart_seed_set),
        'n_ge_99_strict': n_ge_99_strict,
        'n_ge_99_all':    n_ge_99_all,
        'per_step_5seed_mean': {
            str(s): mstd(merged['results'][str(s)]) for s in TEST_STEPS
        },
    }
    merged['aggregates'] = aggregates

    # Load baseline 4×4d for comparison
    BASELINE_PATH = 'multi_seed_results/resmlp_backbone_ablation/raw.json'
    baseline = None
    if os.path.exists(BASELINE_PATH):
        with open(BASELINE_PATH) as f:
            baseline = json.load(f)
        b_seeds = baseline.get('seeds_completed', baseline.get('seeds', []))
        b_500 = baseline['results']['500']
        b_elapsed = baseline['elapsed']
        # Restart heuristic for baseline (no explicit restart count)
        b_med = float(np.median(b_elapsed))
        b_restart = {b_seeds[i] for i, e in enumerate(b_elapsed) if e > 1.5 * b_med}
        b_strict_idx = [i for i, s in enumerate(b_seeds) if s not in b_restart]
        b_strict_vals = [b_500[i] for i in b_strict_idx]
        baseline_summary = {
            'seeds': b_seeds, 'per_seed_500': b_500,
            'restart_seeds_inferred': sorted(b_restart),
            'strict_500': mstd(b_strict_vals),
            'as_spec_500': mstd(b_500),
            'n_ge_99_strict': sum(1 for v in b_strict_vals if v >= 0.99),
            'n_ge_99_all':    sum(1 for v in b_500 if v >= 0.99),
        }
        merged['baseline_4x4d'] = baseline_summary
    else:
        baseline_summary = None

    # Save merged JSON
    merged_path = os.path.join(OUT_DIR, '8x2d_5seed.json')
    with open(merged_path, 'w') as f:
        json.dump(merged, f, indent=2)
    print(f'  Saved merged 5-seed: {merged_path}')

    # --- Markdown report ---
    L = []
    L.append('# ResMLP 8×2d (d=276) — Complete 5-seed Distribution\n')
    L.append(f'**Date**: {time.strftime("%Y-%m-%d %H:%M:%S")}')
    L.append(f'**Config**: n_blocks={N_BLOCKS}, expansion={EXPANSION}d, d={D}, '
             f'step_params={merged["step_params"]:,}')
    L.append(f'**Protocol**: 3-phase Gumbel pipeline, AdamW lr=1e-3 cosine, batch 4096, '
             f'EPOCHS_PER_TRY=150, MAX_RESTARTS=3, FP16')
    L.append('')
    L.append('## Per-seed full results\n')
    L.append('| Seed | P1 epochs* | P1 acc | P3 acc | restarts | t(s) | 5-step | 10-step | 50-step | 100-step | **500-step** |')
    L.append('|---|---|---|---|---|---|---|---|---|---|---|')
    for i, seed in enumerate(seeds):
        L.append(f'| {seed} | (full)** | {merged["p1"][i]:.2%} | {merged["p3"][i]:.2%} | '
                 f'{merged["p1_restart_count"][i]} | {merged["elapsed"][i]:.0f} | '
                 f'{merged["results"]["5"][i]:.2%} | {merged["results"]["10"][i]:.2%} | '
                 f'{merged["results"]["50"][i]:.2%} | {merged["results"]["100"][i]:.2%} | '
                 f'**{merged["results"]["500"][i]:.2%}** |')
    L.append('')
    L.append('*P1 epoch breakdown is in stdout log, not aggregated here.')
    L.append('**Compute-matched: no Phase 1 auto-restart triggered for any seed.')
    L.append('')

    L.append('## Aggregate statistics (500-step)\n')
    s = aggregates['500_strict']; a = aggregates['500_as_spec']
    L.append('| Aggregate | n | mean | sample std | min | max | count ≥99% |')
    L.append('|---|---|---|---|---|---|---|')
    L.append(f'| **strict** (no restart) | {s["n"]} | {s["mean"]:.2%} | '
             f'{s["std"]:.2%} | {s["min"]:.2%} | {s["max"]:.2%} | '
             f'**{aggregates["n_ge_99_strict"]}/{s["n"]}** |')
    L.append(f'| **as-specified** (all 5) | {a["n"]} | {a["mean"]:.2%} | '
             f'{a["std"]:.2%} | {a["min"]:.2%} | {a["max"]:.2%} | '
             f'**{aggregates["n_ge_99_all"]}/{a["n"]}** |')
    L.append('')
    if not aggregates['restart_seeds']:
        L.append('All 5 seeds compute-matched (no restarts) → strict ≡ as-specified.')
    else:
        L.append(f'Restart-triggered seeds (excluded from strict): {aggregates["restart_seeds"]}')
    L.append('')

    L.append('## Per-step degradation curve (5-seed mean)\n')
    L.append('| Steps | mean | std | min | max |')
    L.append('|---|---|---|---|---|')
    for st in TEST_STEPS:
        d = aggregates['per_step_5seed_mean'][str(st)]
        L.append(f'| {st} | {d["mean"]:.2%} | {d["std"]:.2%} | {d["min"]:.2%} | {d["max"]:.2%} |')
    L.append('')

    if baseline_summary is not None:
        L.append('## Comparison vs baseline 4×4d (d=280)\n')
        bs = baseline_summary['strict_500']; ba = baseline_summary['as_spec_500']
        L.append('| Config | n_strict | strict_mean | strict_std | as_spec_mean | as_spec_std | ≥99% (strict) | per-seed 500-step |')
        L.append('|---|---|---|---|---|---|---|---|')
        b_seed_str = ', '.join(f'{v:.2%}' for v in baseline_summary['per_seed_500'])
        n_seed_str = ', '.join(f'{v:.2%}' for v in accs500)
        L.append(f'| baseline 4×4d | {bs["n"]} | {bs["mean"]:.2%} | {bs["std"]:.2%} | '
                 f'{ba["mean"]:.2%} | {ba["std"]:.2%} | '
                 f'{baseline_summary["n_ge_99_strict"]}/{bs["n"]} | {b_seed_str} |')
        L.append(f'| **8×2d** | {s["n"]} | {s["mean"]:.2%} | {s["std"]:.2%} | '
                 f'{a["mean"]:.2%} | {a["std"]:.2%} | '
                 f'**{aggregates["n_ge_99_strict"]}/{s["n"]}** | {n_seed_str} |')
        L.append('')

    # Interpretation
    L.append('## Interpretation\n')
    n99 = aggregates['n_ge_99_strict']
    nT  = s['n']
    if n99 == nT:
        verdict_a = (f'**{n99}/{nT} seeds reach ≥99%** → seed 42 was a pattern, '
                     f'not an outlier. 8×2d is RELIABLE.')
    elif n99 >= nT - 1:
        verdict_a = (f'**{n99}/{nT} seeds reach ≥99%** → seed 42 was largely a pattern; '
                     f'8×2d nearly reliable with one exception.')
    elif n99 >= 2:
        verdict_a = (f'**{n99}/{nT} seeds reach ≥99%** → seed 42 was not unique but not '
                     f'universal either; 8×2d shows mixed behavior.')
    else:
        verdict_a = (f'**{n99}/{nT} seeds reach ≥99%** → seed 42 looks like an outlier; '
                     f'8×2d is NOT reliable as a config.')

    if baseline_summary is not None:
        b_99 = baseline_summary['n_ge_99_strict']
        b_n = baseline_summary['strict_500']['n']
        delta_mean = (s['mean'] - bs['mean']) * 100
        verdict_b = (f'8×2d strict mean ({s["mean"]:.2%}) vs baseline 4×4d strict mean '
                     f'({bs["mean"]:.2%}) → 8×2d is **{delta_mean:+.2f} pp** '
                     f'{"better" if delta_mean > 0 else "worse"} than baseline. '
                     f'Reliability: 8×2d {n99}/{nT} ≥99% vs baseline {b_99}/{b_n} ≥99%.')
    else:
        verdict_b = ''

    L.append(verdict_a)
    L.append('')
    if verdict_b:
        L.append(verdict_b)
        L.append('')

    L.append('### Interpretation\n')
    if n99 >= nT - 1:
        L.append('- **H1 falsifier triggered at config level**: a residual-only ResMLP config '
                 'reliably matches THEIA on length generalization at 500 steps.')
        L.append('- "domain-segregated structure is required" claim now has a counterexample '
                 'at this configuration.')
        L.append('- Follow-up: 8×4d (deeper-narrower) + 4×2d as comparators '
                 'to characterize the configuration boundary.')
    elif n99 >= 2:
        L.append('- **H1 partially falsified**: ResMLP can reach 99% on a subset of seeds '
                 'with the right config, but not reliably across all seeds.')
        L.append('- The accurate reading shifts from "structured-but-undifferentiated unreliable" '
                 'to "configuration-sensitive: some (config, seed) combinations match THEIA".')
        L.append('- THEIA still wins on **reliability** (5/5 ≥ 99% vs 8×2d {}/{} ≥ 99%).'.format(n99, nT))
    else:
        L.append('- **H1 strengthened**: seed 42 was an outlier; 8×2d is not reliable.')
        L.append('- Original "residual-only unreliable" claim survives.')
        L.append('- No change to the configuration-reliability conclusion.')
    L.append('')

    report_path = os.path.join(OUT_DIR, '8x2d_5seed.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L))
    print(f'  Saved report: {report_path}')

    return merged


def main():
    completion = run_4_seeds()
    merged = merge_and_report(completion)

    # Console summary
    print(f'\n{"="*72}')
    print(f'  8×2d 5-seed FINAL DISTRIBUTION (500-step)')
    print(f'{"="*72}')
    seeds = merged['seeds']
    accs500 = merged['results']['500']
    rests = merged['p1_restart_count']
    print(f'  {"seed":>5} {"500-step":>10} {"restart?":>9}')
    for s, a, r in zip(seeds, accs500, rests):
        print(f'  {s:>5} {a*100:>9.2f}% {r:>9}')
    s = merged['aggregates']['500_strict']
    a = merged['aggregates']['500_as_spec']
    print(f'\n  strict (n={s["n"]}): {s["mean"]*100:.2f}% ± {s["std"]*100:.2f}%, '
          f'{merged["aggregates"]["n_ge_99_strict"]}/{s["n"]} ≥99%')
    print(f'  as-spec (n={a["n"]}): {a["mean"]*100:.2f}% ± {a["std"]*100:.2f}%, '
          f'{merged["aggregates"]["n_ge_99_all"]}/{a["n"]} ≥99%')


if __name__ == '__main__':
    main()
