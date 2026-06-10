"""
Mod-3 chain state-distribution statistics (paper sec:chain).

Computes ground-truth state frequencies at chain depths {10, 100, 500} for
5 data seeds x 10,000 test chains. No model inference — this is a property
of gen_chain_data's label distribution. Backs the paper claim that the state
distribution remains approximately uniform at all chain depths.
"""
import numpy as np
import torch
from theia_chain_v3_5seed import gen_chain_data

SEEDS = [42, 123, 256, 777, 999]
N_CHAINS = 10000
NUM_STEPS = 500
DEPTHS = [10, 100, 500]

# labels[:, t] is the cumulative mod-3 state at step t; shape (N_CHAINS, NUM_STEPS)

per_seed_freq = {d: [] for d in DEPTHS}

for seed in SEEDS:
    # seed+5000 is the eval seed offset used in theia_chain_v3_5seed evaluate()
    _, labels, _ = gen_chain_data(N_CHAINS, NUM_STEPS, seed=seed + 5000)
    labels_np = labels.numpy() if isinstance(labels, torch.Tensor) else labels

    for d in DEPTHS:
        # state at step d-1 (0-indexed)
        states_at_d = labels_np[:, d - 1]
        counts = np.bincount(states_at_d, minlength=3)
        freq = counts / counts.sum()  # shape (3,)
        per_seed_freq[d].append(freq)

# Aggregate across 5 seeds
print(f"{'Depth':>6s}  {'State 0':>14s}  {'State 1':>14s}  {'State 2':>14s}  {'MaxDev':>8s}")
print("-" * 70)

results = {}
for d in DEPTHS:
    arr = np.stack(per_seed_freq[d])  # (5, 3)
    mean = arr.mean(axis=0)  # (3,)
    std = arr.std(axis=0)    # (3,)
    max_dev_pp = max(abs(m - 1.0 / 3.0) * 100 for m in mean)
    results[d] = {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "max_dev_pp": max_dev_pp,
    }
    print(f"{d:>6d}  "
          f"{mean[0]*100:5.2f}% +/- {std[0]*100:4.2f}%  "
          f"{mean[1]*100:5.2f}% +/- {std[1]*100:4.2f}%  "
          f"{mean[2]*100:5.2f}% +/- {std[2]*100:4.2f}%  "
          f"{max_dev_pp:5.2f}pp")

print()
print("Paper-ready sentence (fill in if Task 5A applies):")
for d in DEPTHS:
    m = results[d]["mean"]
    print(f"  step {d}: {{{m[0]*100:.1f}%, {m[1]*100:.1f}%, {m[2]*100:.1f}%}}")
max_dev_all = max(results[d]["max_dev_pp"] for d in DEPTHS)
print(f"  max deviation from uniform (across all depths): {max_dev_all:.2f} pp")

import json
with open("mod3_state_distribution.json", "w") as f:
    json.dump({
        "seeds": SEEDS,
        "n_chains": N_CHAINS,
        "depths": DEPTHS,
        "results": {str(d): v for d, v in results.items()},
    }, f, indent=2)
print()
print("Saved: mod3_state_distribution.json")
