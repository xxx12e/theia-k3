#!/usr/bin/env python
"""
TF8L probing 5-seed aggregator (Transformer layer-wise probing; §5.3).

Loads tf8l_layer_probing.json from all 5 seeds,
computes mean +/- std for each layer's SVM accuracy and F-T distance,
emits paper-ready text snippets.

Output: multi_seed_results/reports/tf8l_probing_5seed.md
"""
import os, json
from datetime import datetime

ROOT = r'multi_seed_results'
REPORT_DIR = os.path.join(ROOT, 'reports')
os.makedirs(REPORT_DIR, exist_ok=True)

SEEDS = [42, 123, 256, 777, 999]

def mean_std(xs):
    if not xs: return (None, None)
    m = sum(xs) / len(xs)
    if len(xs) == 1: return (m, 0.0)
    v = sum((x - m) ** 2 for x in xs) / len(xs)
    return (m, v ** 0.5)

# Load all 5 seed JSONs
data = {}
for seed in SEEDS:
    path = os.path.join(ROOT, 'tf8l', f'seed_{seed}', 'tf8l_layer_probing.json')
    if os.path.exists(path):
        with open(path) as f:
            data[seed] = json.load(f)
    else:
        print(f"MISSING: {path}")

print(f"Loaded {len(data)}/5 TF8L probing seeds: {sorted(data.keys())}")

if len(data) == 0:
    print("ERROR: No data. Did you run probe_tf8l_layers.py on the seeds?")
    exit(1)

# Aggregate per-layer
n_layers = 9  # 0 = input, 1-8 = transformer layers
per_layer_stats = {}
for layer_idx in range(n_layers):
    svm_vals = []
    ft_vals = []
    for seed, d in data.items():
        layer_data = d.get('per_layer', {}).get(str(layer_idx))
        if layer_data:
            svm_vals.append(layer_data.get('svm_acc'))
            ft_vals.append(layer_data.get('ft_distance'))
    per_layer_stats[layer_idx] = {
        'svm_mean_std': mean_std([v for v in svm_vals if v is not None]),
        'ft_mean_std':  mean_std([v for v in ft_vals if v is not None]),
    }

# Aggregate separation ratio
sep_ratios = [d.get('separation_ratio_last_over_input') for d in data.values()]
sep_ratios = [v for v in sep_ratios if v is not None]
sep_mean, sep_std = mean_std(sep_ratios)

# Aggregate layer 1 SVM and final SVM
layer1_svms = [d.get('svm_at_layer_1') for d in data.values()]
layer1_svms = [v for v in layer1_svms if v is not None]
layer1_mean, layer1_std = mean_std(layer1_svms)

final_svms = [d.get('svm_at_final_layer') for d in data.values()]
final_svms = [v for v in final_svms if v is not None]
final_mean, final_std = mean_std(final_svms)

# Build report
lines = []
L = lines.append

L("# TF8L Layer-wise Probing — 5-Seed Aggregate")
L("")
L(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
L("")
L(f"Loaded {len(data)}/5 TF8L checkpoints (seeds: {sorted(data.keys())})")
L("")
L("---")
L("")

L("## Per-layer stats (mean +/- std across seeds)")
L("")
L("| Layer | SVM Accuracy | F-T Distance |")
L("|---|---|---|")
for layer_idx in range(n_layers):
    s = per_layer_stats[layer_idx]
    sm, ss = s['svm_mean_std']
    fm, fs = s['ft_mean_std']
    label = "input" if layer_idx == 0 else f"layer {layer_idx}"
    svm_str = f"{sm:.3f} +/- {ss:.3f}" if sm is not None else "--"
    ft_str = f"{fm:.3f} +/- {fs:.3f}" if fm is not None else "--"
    L(f"| {label} | {svm_str} | {ft_str} |")
L("")

L("## Headline numbers")
L("")
L(f"- **Layer-1 SVM accuracy**: {layer1_mean:.3f} +/- {layer1_std:.3f}")
L(f"- **Final-layer SVM accuracy**: {final_mean:.3f} +/- {final_std:.3f}")
L(f"- **Separation ratio (last / input)**: {sep_mean:.1f} +/- {sep_std:.1f}")
L("")

L("---")
L("")
L("## Paper-ready text snippets")
L("")
L("### §5.3 (drop-in replacement, NOW WITH 5-SEED MEAN +/- STD)")
L("")
L("> Linear classification probes applied to the intermediate layers of the parameter-matched ")
L(f"> 8-layer Transformer used in section 4.3 already achieve {layer1_mean:.1%} +/- {layer1_std:.1%} accuracy at the first ")
L(f"> encoder layer (well above the 33% chance baseline; mean +/- std across {len(data)} seeds), and the F-T centroid ")
L(f"> distance grows by a factor of {sep_mean:.0f} +/- {sep_std:.0f}x from the input embeddings to the final layer. THEIA's modular ")
L(f"> pipeline, in contrast, maintains near-chance probe accuracy through the arithmetic stage and grows ")
L(f"> the F-T separation by 1898x — over an order of magnitude larger than the Transformer baseline at ")
L(f"> matched final correctness.")
L("")

L("### Table 4 footnote (lineage)")
L("")
L(f"> *TF8L probing values are mean +/- std over the same {len(data)} seeds used for the THEIA Kleene comparison ")
L(f"> in Table 2 (seeds {{42, 123, 256, 777, 999}}); 50K samples per seed.*")
L("")

L("---")
L("")
L("## Per-seed raw")
L("")
L("| Seed | Layer 1 SVM | Final SVM | Separation Ratio |")
L("|---|---|---|---|")
for seed in sorted(data.keys()):
    d = data[seed]
    l1 = d.get('svm_at_layer_1', 0)
    fl = d.get('svm_at_final_layer', 0)
    sr = d.get('separation_ratio_last_over_input', 0)
    L(f"| {seed} | {l1:.3f} | {fl:.3f} | {sr:.1f}x |")

report_path = os.path.join(REPORT_DIR, 'tf8l_probing_5seed.md')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))

print(f"\nReport: {report_path}")
print()
print('\n'.join(lines))
