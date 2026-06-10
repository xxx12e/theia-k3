#!/usr/bin/env python
"""
Aggregate TF8L early-stopping runs (matched stopping criterion) and compute
the fair speedup vs THEIA for the paper's matched-protocol comparison.

Output: multi_seed_results/reports/tf8l_earlystop_fair.md
"""
import os, json
from datetime import datetime

ROOT = r'multi_seed_results'
TF_ROOT = os.path.join(ROOT, 'tf8l_earlystop')
THEIA_ROOT_42 = os.path.join(ROOT, 'theia', 'seed_42')
THEIA_ROOT_V2 = os.path.join(ROOT, 'theia_v2')
REPORT_DIR = os.path.join(ROOT, 'reports')
os.makedirs(REPORT_DIR, exist_ok=True)

SEEDS = [42, 123, 256, 777, 999]

def mean_std(xs):
    xs = [x for x in xs if x is not None]
    if not xs: return (None, None)
    m = sum(xs) / len(xs)
    if len(xs) == 1: return (m, 0.0)
    v = sum((x - m) ** 2 for x in xs) / len(xs)
    return (m, v ** 0.5)

def load_summaries(root_fn):
    out = {}
    for s in SEEDS:
        path = root_fn(s)
        if os.path.exists(path):
            with open(path) as f:
                out[s] = json.load(f)
    return out

tf_data = load_summaries(lambda s: os.path.join(TF_ROOT, f'seed_{s}', 'summary.json'))
theia_data = {}
theia_data[42] = None
p42 = os.path.join(THEIA_ROOT_42, 'summary.json')
if os.path.exists(p42):
    with open(p42) as f: theia_data[42] = json.load(f)
for s in [123, 256, 777, 999]:
    p = os.path.join(THEIA_ROOT_V2, f'seed_{s}', 'summary.json')
    if os.path.exists(p):
        with open(p) as f: theia_data[s] = json.load(f)

print(f"TF8L earlystop: {len(tf_data)}/5")
print(f"THEIA: {sum(1 for v in theia_data.values() if v)}/5")

# Convergence times
tf_times = [d.get('total_time_sec')/60 for s, d in tf_data.items()
            if d and d.get('converged')]
tf_epochs = [d.get('converge_epoch') for s, d in tf_data.items()
             if d and d.get('converged')]
theia_times = [d.get('total_time_sec')/60 for s, d in theia_data.items()
               if d and d.get('total_time_sec')]

tf_m, tf_s = mean_std(tf_times)
theia_m, theia_s = mean_std(theia_times)

lines = []
L = lines.append
L("# TF8L Fair Early-Stopping Comparison — 5-Seed Aggregate")
L("")
L(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
L("")
L("**Critical fair comparison**: Both THEIA and TF8L trained under the same early-stopping ")
L("criterion: `overall > 99.9% AND 12/12 Kleene rules > 99% on two consecutive checkpoints`.")
L("")
L("Previously, TF8L trained for a fixed 150 epochs without Kleene-based stopping, which was ")
L("asymmetric with THEIA and flagged as a methodological problem. This report closes that gap.")
L("")
L("---")
L("")
L("## Head-to-head")
L("")
L("| Model | Convergence time (min) | Convergence epoch |")
L("|---|---|---|")
if theia_m is not None:
    L(f"| THEIA (modular) | {theia_m:.1f} ± {theia_s:.1f} | (varies) |")
if tf_m is not None:
    tfep_m, tfep_s = mean_std(tf_epochs)
    L(f"| TF8L (early-stop) | {tf_m:.1f} ± {tf_s:.1f} | {tfep_m:.0f} ± {tfep_s:.0f} |")
L("")

if theia_m and tf_m:
    speedup = tf_m / theia_m
    L(f"**Fair speedup: {speedup:.1f}×**")
    L("")
    old_speedup = 60.9 / 9.2
    L(f"- Previous (unfair, TF fixed 150 epochs): 60.9/9.2 = {old_speedup:.1f}×")
    L(f"- New (fair, both early-stop): {tf_m:.1f}/{theia_m:.1f} = {speedup:.1f}×")
    L("")
    if speedup >= 5.0:
        L("**VERDICT: 6.6× claim HOLDS under fair comparison.** Paper can keep headline number or ")
        L(f"update to the more precise fair value ({speedup:.1f}×).")
    elif speedup >= 3.0:
        L(f"**VERDICT: Speedup reduces to {speedup:.1f}× under fair comparison.** Paper should update ")
        L("Table 1 and all recap mentions.")
    else:
        L(f"**VERDICT: Speedup COLLAPSES to {speedup:.1f}× under fair comparison.** Major §5.3 rewrite needed.")
L("")
L("## Per-seed detail")
L("")
L("| Seed | THEIA time (min) | TF8L early-stop time (min) | TF8L epoch | Speedup |")
L("|---|---|---|---|---|")
for s in SEEDS:
    t_time = theia_data.get(s, {}).get('total_time_sec') if theia_data.get(s) else None
    f_data = tf_data.get(s)
    t_str = f"{t_time/60:.1f}" if t_time else "--"
    if f_data and f_data.get('converged'):
        f_time = f_data.get('total_time_sec') / 60
        f_ep = f_data.get('converge_epoch')
        sp = f"{(f_time*60)/t_time:.1f}×" if t_time else "--"
        L(f"| {s} | {t_str} | {f_time:.1f} | {f_ep} | {sp} |")
    elif f_data:
        L(f"| {s} | {t_str} | NO-CONV | {f_data.get('converge_epoch')} | -- |")
    else:
        L(f"| {s} | {t_str} | MISSING | -- | -- |")

out_path = os.path.join(REPORT_DIR, 'tf8l_earlystop_fair.md')
with open(out_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
print(f"\nReport: {out_path}")
print()
print('\n'.join(lines))
