"""
Per-boundary Bayes-optimal ceiling for Has-Unknown (HU) probe accuracy (§5.3 / Table 7).

Each boundary's hidden state sees a subset of the 4 atomic Unknown flags:

  Arith boundary  → visible: {a_unk, b_unk}              (2 flags)
  Order boundary  → visible: {a_unk, b_unk, d_unk}       (3 flags)
  Set boundary    → visible: {a_unk, b_unk, s_unk}       (3 flags)
  Logic boundary  → visible: {a_unk, b_unk, d_unk, s_unk} (all 4)

(Order does NOT see s_unk because s_unk is injected at the Set engine.
 Set does NOT see d_unk because d_unk is injected at the Order engine.)

Bayes-optimal HU classifier given visible flags V: predict HU=1 iff any flag
in V is 1 (always correct, since HU = OR of all flags); else predict HU=0
(correct iff all hidden flags are also 0).
Accuracy(Bayes) = P(any V=1) + P(all V=0) · P(all hidden flags = 0)

Computed (i) closed-form (exact Bernoulli(0.15)) and (ii) empirically
(data_seed=999, N=50000), then compared against the paper probe numbers and
the w=512 stratified deep-probe run.
"""
import os, sys, json
sys.path.insert(0, '.')
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import torch
import numpy as np
torch.cuda.is_available = lambda: False

from _theia_model_def import P_UNKNOWN

DATA_SEED = 999
N = 50_000


def gen_flags(N, seed):
    torch.manual_seed(seed)
    return {
        'a_unk': (torch.rand(N) < P_UNKNOWN).numpy().astype(int),
        'b_unk': (torch.rand(N) < P_UNKNOWN).numpy().astype(int),
        'd_unk': (torch.rand(N) < P_UNKNOWN).numpy().astype(int),
        's_unk': (torch.rand(N) < P_UNKNOWN).numpy().astype(int),
    }


def bayes_ceiling_empirical(visible_flags, all_flags, N):
    """
    Empirical Bayes ceiling: classifier sees only visible_flags.
    Predict HU=1 iff any visible flag = 1, else predict HU=0.
    Compute accuracy on the actual joint distribution.
    """
    # HU truth = OR of all flags
    hu_true = np.zeros(N, dtype=int)
    for f in all_flags.values():
        hu_true = hu_true | f

    # Prediction: 1 iff any visible flag = 1
    hu_pred = np.zeros(N, dtype=int)
    for f in visible_flags:
        hu_pred = hu_pred | all_flags[f]

    accuracy = (hu_pred == hu_true).mean()
    return float(accuracy)


def bayes_ceiling_closed_form(k, total=4, p_unk=P_UNKNOWN):
    """
    Closed-form Bayes ceiling: visible flags k of total, each Bernoulli(p_unk).
    """
    return (1 - (1 - p_unk)**k) + (1 - p_unk)**total


def main():
    print('=' * 78)
    print(f'  Per-boundary Bayes-optimal HU ceiling')
    print(f'  (data_seed={DATA_SEED}, N={N}, P_UNKNOWN={P_UNKNOWN})')
    print('=' * 78)

    flags = gen_flags(N, DATA_SEED)
    print(f'\nEmpirical per-flag fractions (N={N}):')
    for f in ['a_unk', 'b_unk', 'd_unk', 's_unk']:
        print(f'  P({f}=1) = {flags[f].mean():.4f}')

    boundaries = {
        'Arith': ['a_unk', 'b_unk'],
        'Order': ['a_unk', 'b_unk', 'd_unk'],
        'Set':   ['a_unk', 'b_unk', 's_unk'],
        'Logic': ['a_unk', 'b_unk', 'd_unk', 's_unk'],
    }

    # Empirical paper / w=512 numbers (per-seed mean ± std, %)
    paper_HU = {
        'Arith': (80.2267, 0.0000),  # mechanistic_probing_v2.json, identical across 5 seeds
        'Order': (91.0600, 0.0000),
        'Set':   (90.7533, 0.0000),
        'Logic': (99.6533, 0.3294),
    }
    w512_d2_HU = {
        'Arith': (80.0280, 0.0000),
        'Order': (91.0892, 0.0000),
        'Set':   (90.7672, 0.0085),
        'Logic': (99.7596, 0.2167),
    }

    # Recompute paper / w512 from JSON for accuracy
    seeds = [42, 123, 256, 777, 999]

    def path_paper(s): return f'multi_seed_results/theia/seed_{s}/mechanistic_probing_v2.json' if s == 42 else f'multi_seed_results/theia_v2/seed_{s}/mechanistic_probing_v2.json'

    paper_recompute = {b: [] for b in boundaries}
    for s in seeds:
        j = json.load(open(path_paper(s)))
        for b in boundaries:
            v = j['probes'][b.lower()]['has_unknown_acc']
            paper_recompute[b].append(v)

    deep = json.load(open('deep_nonlinear_probe_results.json'))
    w512_d2_recompute = {}
    for b in boundaries:
        k = f'{b.lower()}__has_unknown__d2'
        w512_d2_recompute[b] = deep['aggregate']['per_cell_agg'][k]['per_seed']

    print()
    print('=' * 78)
    print(f'  PER-BOUNDARY CEILING + EMPIRICAL PROBE COMPARISON')
    print('=' * 78)
    print(f'  {"boundary":>8} | {"k":>3} | {"closed-form":>12} | {"empirical":>11} | {"paper probe":>14} | {"w=512 d=2":>14}')
    print(f'  {"":>8} | {"":>3} | {"ceiling":>12} | {"ceiling":>11} | {"(2-h 256)":>14} | {"(val-best)":>14}')
    print('  ' + '-' * 90)
    out = {
        'protocol': {
            'data_seed': DATA_SEED, 'N': N, 'P_UNKNOWN': P_UNKNOWN,
            'paper_source': 'mechanistic_probing_v2.json',
            'w512_source': 'deep_nonlinear_probe_results.json',
        },
        'per_flag_emp': {f: float(flags[f].mean()) for f in flags},
        'boundaries': {},
    }
    for b, vis in boundaries.items():
        k = len(vis)
        closed = bayes_ceiling_closed_form(k)
        emp = bayes_ceiling_empirical(vis, flags, N)
        paper_v = paper_recompute[b]
        paper_m = float(np.mean(paper_v))
        paper_s = float(np.std(paper_v, ddof=1))
        w512_v = w512_d2_recompute[b]
        w512_m = float(np.mean(w512_v))
        w512_s = float(np.std(w512_v, ddof=1))
        print(f'  {b:>8} | {k:>3} | {closed*100:>10.4f}% | {emp*100:>9.4f}% | {paper_m*100:>5.4f} ± {paper_s*100:>4.4f}% | {w512_m*100:>5.4f} ± {w512_s*100:>4.4f}%')
        out['boundaries'][b] = {
            'visible_flags': vis,
            'k': k,
            'closed_form_ceiling': float(closed),
            'empirical_ceiling':   float(emp),
            'paper_5seed_mean':    paper_m,
            'paper_5seed_std':     paper_s,
            'paper_per_seed':      [float(x) for x in paper_v],
            'w512_d2_5seed_mean':  w512_m,
            'w512_d2_5seed_std':   w512_s,
            'w512_d2_per_seed':    [float(x) for x in w512_v],
            'gap_paper_minus_emp_ceiling_pp':  float((paper_m - emp) * 100),
            'gap_w512_minus_emp_ceiling_pp':   float((w512_m - emp) * 100),
        }

    print()
    print('  Gap analysis (probe accuracy minus empirical Bayes ceiling, pp):')
    print(f'  {"boundary":>8} | {"paper gap":>11} | {"w=512 d=2 gap":>14}')
    print('  ' + '-' * 50)
    for b in boundaries:
        gp = out['boundaries'][b]['gap_paper_minus_emp_ceiling_pp']
        gw = out['boundaries'][b]['gap_w512_minus_emp_ceiling_pp']
        print(f'  {b:>8} | {gp:+8.4f}pp  | {gw:+8.4f}pp')

    json.dump(out, open('per_boundary_HU_bayes_ceiling.json', 'w'), indent=2)
    print(f'\nSaved per_boundary_HU_bayes_ceiling.json')


if __name__ == '__main__':
    main()
