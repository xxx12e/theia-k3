"""
Activation patching — 5-seed wrapper.
Loads each THEIA checkpoint and runs the full patching protocol from
activation_patching.py with identical matched-pair construction (data seed 12345).

Output: activation_patching_5seed.json + console summary table.
"""
import json, os, sys, types, time

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import torch

# Import only the functions; activation_patching.main() has its own argparse.
from activation_patching import (
    _forward_with_hidden, _build_model,
    run_patching, classify,
    ARITH_ADD, REL_GTE, LOGIC_OR, NUM_RANGE, SET_DIM,
    PROTO_FALSE, PROTO_TRUE, PROTO_UNKNOWN,
)

# Seed-42 checkpoint path (LEGACY, load-bearing): two valid checkpoints exist,
# both pass 12/12 Kleene:
#   theia/seed_42/checkpoint.pth    (MD5 eab083db..., pre-retrain, 200-epoch run)
#   theia_v2/seed_42/checkpoint.pth (MD5 e874cbea..., 2026-04-15 retrain; used for
#                                    Table 1 first-12/12 wall-clock of 7.31 min)
# The paper's 4898/4898 patching aggregate was produced on the pre-retrain
# checkpoint; the retrain gives 4904/4904 (100% flip, +6 eligible pairs).
# The pre-retrain path is kept so this script reproduces the paper's exact
# aggregate; swap to theia_v2 for the latest model.
CHECKPOINTS = [
    (42,  'multi_seed_results/theia/seed_42/checkpoint.pth'),
    (123, 'multi_seed_results/theia_v2/seed_123/checkpoint.pth'),
    (256, 'multi_seed_results/theia_v2/seed_256/checkpoint.pth'),
    (777, 'multi_seed_results/theia_v2/seed_777/checkpoint.pth'),
    (999, 'multi_seed_results/theia_v2/seed_999/checkpoint.pth'),
]
N_PAIRS = 1000
DATA_SEED = 12345  # fixed across all seeds for comparability
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_model(ckpt_path):
    model = _build_model(DEVICE)
    state = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    if isinstance(state, dict) and 'model_state_dict' in state:
        model.load_state_dict(state['model_state_dict'])
    elif isinstance(state, dict) and 'state_dict' in state:
        model.load_state_dict(state['state_dict'])
    else:
        model.load_state_dict(state)
    model.forward_with_hidden = types.MethodType(_forward_with_hidden, model)
    return model


def main():
    results = {}
    print(f"{'='*70}")
    print(f"  Activation Patching — 5-Seed Sweep")
    print(f"  N_PAIRS={N_PAIRS}, DATA_SEED={DATA_SEED}, DEVICE={DEVICE}")
    print(f"{'='*70}")

    for seed, ckpt_path in CHECKPOINTS:
        print(f"\n--- Seed {seed} ---")
        if not os.path.exists(ckpt_path):
            print(f"  SKIP: {ckpt_path} not found")
            continue
        model = load_model(ckpt_path)
        stats = run_patching(model, N_PAIRS, DEVICE, DATA_SEED, debug=False)

        n = stats['both_baseline_correct']
        entry = {
            'baseline_t': stats['baseline_T_correct'],
            'baseline_u': stats['baseline_U_correct'],
            'eligible': n,
            'flip_T_to_U': stats['patch_flipped_to_U'],
            'flip_T_to_F': stats['patch_other_F'],
            'stay_T': stats['patch_stayed_T'],
            'set_vec_identical': stats['set_vec_identical'],
            'flip_rate': stats['patch_flipped_to_U'] / n if n > 0 else 0.0,
        }
        results[str(seed)] = entry
        print(f"  eligible={n}, flip={entry['flip_T_to_U']}/{n}, "
              f"rate={entry['flip_rate']:.4f}")
        del model
        torch.cuda.empty_cache()

    # Aggregate
    rates = [v['flip_rate'] for v in results.values()]
    eligible_total = sum(v['eligible'] for v in results.values())
    flips_total = sum(v['flip_T_to_U'] for v in results.values())
    import numpy as np
    aggregate = {
        'flip_rate_mean': float(np.mean(rates)),
        'flip_rate_std': float(np.std(rates)),
        'n_seeds': len(rates),
        'n_eligible_total': eligible_total,
        'n_flips_total': flips_total,
        'aggregate_flip_rate': flips_total / eligible_total if eligible_total > 0 else 0.0,
    }

    output = {'per_seed': results, 'aggregate': aggregate}
    out_path = 'activation_patching_5seed.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)

    # Print table
    print(f"\n{'='*70}")
    print(f"  RESULTS")
    print(f"{'='*70}")
    print(f"{'Seed':>6} | {'Base-T':>7} | {'Base-U':>7} | {'Eligible':>8} | "
          f"{'Flip->U':>7} | {'Stay-T':>6} | {'->F':>4} | {'Rate':>6}")
    print('-' * 70)
    for seed_str, v in results.items():
        print(f"{seed_str:>6} | {v['baseline_t']:>7} | {v['baseline_u']:>7} | "
              f"{v['eligible']:>8} | {v['flip_T_to_U']:>7} | {v['stay_T']:>6} | "
              f"{v['flip_T_to_F']:>4} | {v['flip_rate']:>6.4f}")
    print('-' * 70)
    print(f"{'AGG':>6} | {'':>7} | {'':>7} | {eligible_total:>8} | "
          f"{flips_total:>7} | {'':>6} | {'':>4} | "
          f"{aggregate['flip_rate_mean']:.4f}+/-{aggregate['flip_rate_std']:.4f}")
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
