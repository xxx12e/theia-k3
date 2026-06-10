"""
Merge tuned_theia_pilot_results.json (seed 42) +
      tuned_theia_5seed_results.json (seeds 123/256/777/999, written by --seeds run)
into  tuned_theia_5seed_FINAL.json with all 5 seeds.

Run AFTER the --seeds 123 256 777 999 run completes.
"""
import json
import os
import numpy as np
from collections import OrderedDict


def main():
    pilot_path  = 'tuned_theia_pilot_results.json'
    extend_path = 'tuned_theia_5seed_results.json'   # written by --seeds run
    out_path    = 'tuned_theia_5seed_FINAL.json'

    if not os.path.exists(pilot_path):
        raise SystemExit(f'missing {pilot_path}')
    if not os.path.exists(extend_path):
        raise SystemExit(f'missing {extend_path}')

    pilot  = json.load(open(pilot_path))
    extend = json.load(open(extend_path))

    # Sanity: same recipe
    assert pilot['recipe'] == extend['recipe'], 'recipe mismatch between pilot and extend'

    # Merge per_seed lists in canonical seed order [42, 123, 256, 777, 999]
    seed_to_record = {}
    for r in pilot['per_seed']:
        seed_to_record[r['seed']] = r
    for r in extend['per_seed']:
        seed_to_record[r['seed']] = r
    canonical_seeds = [42, 123, 256, 777, 999]
    merged_per_seed = [seed_to_record[s] for s in canonical_seeds if s in seed_to_record]
    if len(merged_per_seed) != 5:
        present = [s for s in canonical_seeds if s in seed_to_record]
        raise SystemExit(f'expected 5 seeds, got {len(merged_per_seed)}: {present}')

    # Compute aggregate
    times_12_12 = [r['first_12_12_wall_min'] for r in merged_per_seed
                   if r['first_12_12_wall_min'] is not None]
    epochs_12_12 = [r['first_12_12_epoch'] for r in merged_per_seed
                    if r['first_12_12_epoch'] is not None]
    final_acc = [r['final_overall_acc'] for r in merged_per_seed]
    kleene_passed = [r['final_kleene_passed'] for r in merged_per_seed]
    total_wall = [r['total_wall_min'] for r in merged_per_seed]

    aggregate = {
        'n_seeds': 5,
        'seeds': canonical_seeds,
        'first_12_12_wall_min': {
            'per_seed': times_12_12,
            'mean':  float(np.mean(times_12_12)),
            'std':   float(np.std(times_12_12, ddof=1)),
            'min':   float(np.min(times_12_12)),
            'max':   float(np.max(times_12_12)),
            'median': float(np.median(times_12_12)),
            'n_converged': len(times_12_12),
        },
        'first_12_12_epoch': {
            'per_seed': epochs_12_12,
            'mean':  float(np.mean(epochs_12_12)),
            'std':   float(np.std(epochs_12_12, ddof=1)),
        },
        'final_overall_acc': {
            'per_seed': final_acc,
            'mean':  float(np.mean(final_acc)),
            'std':   float(np.std(final_acc, ddof=1)),
            'min':   float(np.min(final_acc)),
        },
        'final_kleene_passed': {
            'per_seed': kleene_passed,
            'all_12_12': all(k == 12 for k in kleene_passed),
            'min': int(np.min(kleene_passed)),
        },
        'total_wall_min': {
            'per_seed': total_wall,
            'mean': float(np.mean(total_wall)),
            'sum':  float(np.sum(total_wall)),
        },
    }

    out = {
        'recipe': pilot['recipe'],
        'seeds_run': canonical_seeds,
        'merged_from': {
            'pilot':  pilot_path,
            'extend': extend_path,
        },
        'per_seed': merged_per_seed,
        'aggregate': aggregate,
    }
    json.dump(out, open(out_path, 'w'), indent=2)
    print(f'Saved {out_path}')

    print()
    print('=' * 78)
    print('  TUNED THEIA — 5-seed aggregate')
    print('=' * 78)
    print(f'  Recipe: lr={pilot["recipe"]["lr"]} betas={pilot["recipe"]["betas"]} '
          f'wd={pilot["recipe"]["wd"]} warmup={pilot["recipe"]["warmup_epochs"]}ep '
          f'bs={pilot["recipe"]["batch"]}')
    print()
    a = aggregate['first_12_12_wall_min']
    print(f'  first_12_12_wall_min:  {a["mean"]:.2f} ± {a["std"]:.2f} min '
          f'(min {a["min"]:.2f}, max {a["max"]:.2f}, n_converged {a["n_converged"]}/5)')
    print(f'  per-seed:              {[round(x, 2) for x in a["per_seed"]]}')
    e = aggregate['first_12_12_epoch']
    print(f'  first_12_12_epoch:     {e["mean"]:.1f} ± {e["std"]:.1f}  '
          f'(per-seed {e["per_seed"]})')
    f = aggregate['final_overall_acc']
    print(f'  final_overall_acc:     {f["mean"]*100:.4f} ± {f["std"]*100:.4f}%  '
          f'(min {f["min"]*100:.4f}%)')
    k = aggregate['final_kleene_passed']
    print(f'  final_kleene_passed:   per-seed {k["per_seed"]}, all 12/12 = {k["all_12_12"]}')
    t = aggregate['total_wall_min']
    print(f'  total_wall_min:        sum {t["sum"]:.1f} min ({t["sum"]/60:.2f} hr), '
          f'mean per seed {t["mean"]:.1f} min')
    print()

    # Compute new vs-Transformer ratio (against existing tuned-TF baseline)
    # Existing tuned-TF (TF8L tuned): pull from multi_seed_results
    tf_path = 'multi_seed_results/transformer_chain_ablation/raw.json'
    if os.path.exists(tf_path):
        tf = json.load(open(tf_path))
        # tf['elapsed'] is per-seed wall time in SECONDS for tuned TF8L
        tf_times_s = tf.get('elapsed', [])
        if tf_times_s:
            tf_times_min = [t / 60.0 for t in tf_times_s]
            tf_mean = float(np.mean(tf_times_min))
            tf_std  = float(np.std(tf_times_min, ddof=1))
            ratio_mean = tf_mean / a['mean']
            # Pairwise per-seed ratios (assuming same seed order)
            tf_seeds = tf.get('seeds', [])
            paired = []
            if tf_seeds and tf_seeds == a['per_seed'].__class__.__name__ and False:
                pass
            # Simple ordered match
            if tf_seeds == [42, 123, 256, 777, 999]:
                # a['per_seed'] is already in canonical order [42, 123, 256, 777, 999]
                pairs = list(zip(tf_seeds, tf_times_min, a['per_seed']))
                paired_ratios = [tf_t / th_t for _, tf_t, th_t in pairs]
            else:
                paired_ratios = []
            print(f'  Tuned TF8L baseline (5 seeds): {tf_mean:.2f} ± {tf_std:.2f} min '
                  f'(per-seed s: {[round(t, 1) for t in tf_times_s]})')
            print(f'  RATIO  (tuned TF / tuned THEIA) = {ratio_mean:.2f}× '
                  f'(was 3.13× under old "tuned" comparison)')
            if paired_ratios:
                print(f'  Per-seed pairwise ratios: '
                      f'{[round(r, 2) for r in paired_ratios]} '
                      f'(mean {np.mean(paired_ratios):.2f}× ± {np.std(paired_ratios, ddof=1):.2f}×)')
            print(f'  → if ratio > 3.13×, tuned THEIA is FASTER than the old story')
            print(f'  → if ratio < 3.13×, tuned THEIA is SLOWER (paper revision needed)')


if __name__ == '__main__':
    main()
