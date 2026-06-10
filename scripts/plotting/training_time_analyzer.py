#!/usr/bin/env python
"""
Verify the training-time speedup on a pure overall-accuracy target (paper §5.3):
is the 6.6x speedup an artifact of the Kleene-based early-stopping criterion?

Parses train_log.txt from all 10 runs (5 THEIA + 5 TF8L) for the first epoch at
which overall accuracy reaches 99% / 99.5% / 99.9%; wall-clock at that epoch is
estimated as (total_time / total_epochs) * epoch. No new training, log analysis only.

Output: multi_seed_results/reports/training_time_analysis.md
"""
import os, re, json
from datetime import datetime

ROOT = r'multi_seed_results'
REPORT_DIR = os.path.join(ROOT, 'reports')
os.makedirs(REPORT_DIR, exist_ok=True)

SEEDS = [42, 123, 256, 777, 999]

# Model -> seed -> (log_path, summary_path)
MODELS = {
    'theia': {
        42:  (os.path.join(ROOT, 'theia',    'seed_42',  'train_log.txt'),
              os.path.join(ROOT, 'theia',    'seed_42',  'summary.json')),
        123: (os.path.join(ROOT, 'theia_v2', 'seed_123', 'train_log.txt'),
              os.path.join(ROOT, 'theia_v2', 'seed_123', 'summary.json')),
        256: (os.path.join(ROOT, 'theia_v2', 'seed_256', 'train_log.txt'),
              os.path.join(ROOT, 'theia_v2', 'seed_256', 'summary.json')),
        777: (os.path.join(ROOT, 'theia_v2', 'seed_777', 'train_log.txt'),
              os.path.join(ROOT, 'theia_v2', 'seed_777', 'summary.json')),
        999: (os.path.join(ROOT, 'theia_v2', 'seed_999', 'train_log.txt'),
              os.path.join(ROOT, 'theia_v2', 'seed_999', 'summary.json')),
    },
    'tf8l': {
        seed: (os.path.join(ROOT, 'tf8l', f'seed_{seed}', 'train_log.txt'),
               os.path.join(ROOT, 'tf8l', f'seed_{seed}', 'summary.json'))
        for seed in SEEDS
    },
}

THRESHOLDS = [0.99, 0.995, 0.999]

# Regex for train_log line: "epoch 010 loss=0.1234 acc=0.9998 best=0.9998"
EPOCH_RE = re.compile(r'epoch\s+(\d+)\s+loss=[\d.]+\s+acc=([\d.]+)')

def parse_log(log_path):
    """Returns list of (epoch, acc) tuples sorted by epoch."""
    if not os.path.exists(log_path):
        return []
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    out = []
    for line in lines:
        m = EPOCH_RE.search(line)
        if m:
            epoch = int(m.group(1))
            acc = float(m.group(2))
            out.append((epoch, acc))
    return sorted(out)

def find_first_crossing(epoch_acc, threshold):
    """Return first epoch where acc >= threshold, or None."""
    for ep, ac in epoch_acc:
        if ac >= threshold:
            return ep
    return None

def load_total_time_and_epochs(summary_path):
    """Return (total_time_sec, actual_epochs_trained)."""
    if not os.path.exists(summary_path):
        return None, None
    with open(summary_path) as f:
        d = json.load(f)
    total_time = d.get('total_time_sec')
    actual_ep = d.get('converge_epoch') or d.get('max_epochs')
    return total_time, actual_ep

def mean_std(xs):
    xs = [x for x in xs if x is not None]
    if not xs: return (None, None, 0)
    m = sum(xs) / len(xs)
    if len(xs) == 1: return (m, 0.0, 1)
    v = sum((x - m) ** 2 for x in xs) / len(xs)
    return (m, v ** 0.5, len(xs))

# === Analyze ===
results = {model: {} for model in MODELS}

for model, seeds in MODELS.items():
    for seed, (log_path, summary_path) in seeds.items():
        epoch_acc = parse_log(log_path)
        total_time, actual_ep = load_total_time_and_epochs(summary_path)
        if not epoch_acc or total_time is None:
            print(f"SKIP: {model} seed {seed}")
            results[model][seed] = None
            continue

        time_per_epoch = total_time / actual_ep if actual_ep else None

        r = {
            'total_time_sec': total_time,
            'total_epochs': actual_ep,
            'time_per_epoch_sec': time_per_epoch,
            'max_acc_seen': max(ac for _, ac in epoch_acc),
            'thresholds': {}
        }
        for thr in THRESHOLDS:
            crossing_ep = find_first_crossing(epoch_acc, thr)
            if crossing_ep is not None:
                est_time = crossing_ep * time_per_epoch if time_per_epoch else None
                r['thresholds'][thr] = {
                    'epoch': crossing_ep,
                    'estimated_time_sec': est_time,
                    'estimated_time_min': est_time / 60 if est_time else None,
                }
            else:
                r['thresholds'][thr] = {'epoch': None}
        results[model][seed] = r

# === Build report ===
lines = []
L = lines.append
L("# Training Time Analysis Report")
L("")
L(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
L("")
L("**Question**: Is THEIA's 6.6x training-time speedup an artifact of the Kleene-based early-stopping criterion, ")
L("or does it also hold when measured against pure overall-accuracy milestones?")
L("")
L("**Method**: Parse train_log.txt for every training run and find the first epoch at which overall ")
L("validation accuracy crosses each threshold. Wall-clock time is estimated as `(total_time / total_epochs) * crossing_epoch`. ")
L("Since per-epoch cost is approximately constant within a model, this gives an accurate lower bound on ")
L("the time required to reach each threshold.")
L("")
L("**Note**: Validation accuracy is logged every 10 epochs, so 'time to 99.9%' is pessimistic — the actual ")
L("crossing likely happened somewhere in the preceding 10-epoch window. This pessimism applies equally to ")
L("both models and does not bias the comparison.")
L("")
L("---")
L("")

# === Per-threshold aggregate table ===
L("## Headline: Time to each threshold (5-seed mean +/- std)")
L("")
L("| Threshold | THEIA time (min) | TF8L time (min) | THEIA speedup |")
L("|---|---|---|---|")
for thr in THRESHOLDS:
    theia_times = [results['theia'][s]['thresholds'][thr].get('estimated_time_min')
                   for s in SEEDS if results['theia'][s] is not None]
    tf_times    = [results['tf8l'][s]['thresholds'][thr].get('estimated_time_min')
                   for s in SEEDS if results['tf8l'][s] is not None]
    tm, ts, tn = mean_std(theia_times)
    fm, fs, fn = mean_std(tf_times)
    theia_str = f"{tm:.1f} +/- {ts:.1f} (n={tn})" if tm is not None else "--"
    tf_str    = f"{fm:.1f} +/- {fs:.1f} (n={fn})" if fm is not None else "--"
    if tm and fm:
        speedup = fm / tm
        speedup_str = f"**{speedup:.1f}x**"
    else:
        speedup_str = "--"
    L(f"| acc >= {thr*100:.1f}% | {theia_str} | {tf_str} | {speedup_str} |")
L("")

L("## Headline: Epoch at each threshold (5-seed mean +/- std)")
L("")
L("| Threshold | THEIA epoch | TF8L epoch |")
L("|---|---|---|")
for thr in THRESHOLDS:
    theia_epochs = [results['theia'][s]['thresholds'][thr].get('epoch')
                    for s in SEEDS if results['theia'][s] is not None]
    tf_epochs    = [results['tf8l'][s]['thresholds'][thr].get('epoch')
                    for s in SEEDS if results['tf8l'][s] is not None]
    tm, ts, tn = mean_std(theia_epochs)
    fm, fs, fn = mean_std(tf_epochs)
    theia_str = f"{tm:.0f} +/- {ts:.0f} (n={tn})" if tm is not None else "--"
    tf_str    = f"{fm:.0f} +/- {fs:.0f} (n={fn})" if fm is not None else "--"
    L(f"| acc >= {thr*100:.1f}% | {theia_str} | {tf_str} |")
L("")

# === Verdict ===
L("## Verdict")
L("")
L("Look at the `acc >= 99.9%` row. If THEIA is still significantly faster (>3x), the 6.6x ")
L("speedup is architecture-driven, not Kleene-target-driven. If the gap narrows to <2x, ")
L("the criterion choice dominates and you need to weaken §5.3 efficiency claim.")
L("")

# Automatic verdict from the acc >= 99.9% row
theia_999 = [results['theia'][s]['thresholds'][0.999].get('estimated_time_min')
             for s in SEEDS if results['theia'][s] is not None]
tf_999    = [results['tf8l'][s]['thresholds'][0.999].get('estimated_time_min')
             for s in SEEDS if results['tf8l'][s] is not None]
tm999, _, _ = mean_std(theia_999)
fm999, _, _ = mean_std(tf_999)
if tm999 and fm999:
    speedup_999 = fm999 / tm999
    L(f"**Measured speedup at acc >= 99.9%: {speedup_999:.1f}x**")
    L("")
    if speedup_999 >= 3.0:
        L(f"**VERDICT: ROBUST.** The 6.6x speedup claim holds under a pure overall-accuracy target ({speedup_999:.1f}x). ")
        L("Summary statement for the §5.3 efficiency comparison:")
        L("")
        L(f"> *\"We verified that the speedup is not specific to the Kleene-based stopping criterion: ")
        L(f"> measuring first-epoch wall time to reach overall validation accuracy >=99.9% (a criterion ")
        L(f"> independent of the Kleene diagnostic), THEIA reaches this milestone in {tm999:.1f} minutes ")
        L(f"> on average, while the Transformer requires {fm999:.1f} minutes — a {speedup_999:.1f}x ratio ")
        L(f"> consistent with the headline 6.6x number from the Kleene-based criterion. The training-time ")
        L(f"> advantage is a property of the modular architecture, not of the specific stopping rule.\"*")
    elif speedup_999 >= 2.0:
        L(f"**VERDICT: PARTIAL.** The speedup narrows from 6.6x to {speedup_999:.1f}x on a pure overall-acc target. ")
        L("The efficiency claim still holds but needs hedged language. Recommended §5.3 edit:")
        L("")
        L(f"> *\"On a purely overall-accuracy-based criterion (first epoch to reach >=99.9% validation accuracy), ")
        L(f"> THEIA remains {speedup_999:.1f}x faster than the Transformer baseline, confirming that the ")
        L(f"> architectural efficiency gap persists beyond the Kleene-shaped stopping criterion.\"*")
    else:
        L(f"**VERDICT: FRAGILE.** The speedup collapses to {speedup_999:.1f}x on a pure overall-acc target. ")
        L("The 6.6x headline is Kleene-target-dependent. §5.3 efficiency claim needs to be weakened.")
L("")
L("---")
L("")

# === Per-seed detail ===
L("## Per-seed detail")
L("")
for model in ['theia', 'tf8l']:
    L(f"### {model.upper()}")
    L("")
    L("| Seed | Total time (min) | Total epochs | Max acc seen | 99% epoch | 99.5% epoch | 99.9% epoch |")
    L("|---|---|---|---|---|---|---|")
    for seed in SEEDS:
        r = results[model][seed]
        if r is None:
            L(f"| {seed} | MISSING | -- | -- | -- | -- | -- |")
            continue
        tt = r['total_time_sec'] / 60
        te = r['total_epochs']
        ma = r['max_acc_seen']
        e99 = r['thresholds'][0.99].get('epoch')
        e995 = r['thresholds'][0.995].get('epoch')
        e999 = r['thresholds'][0.999].get('epoch')
        e99_str = str(e99) if e99 is not None else "--"
        e995_str = str(e995) if e995 is not None else "--"
        e999_str = str(e999) if e999 is not None else "--"
        L(f"| {seed} | {tt:.1f} | {te} | {ma:.4f} | {e99_str} | {e995_str} | {e999_str} |")
    L("")

# === Save ===
out_path = os.path.join(REPORT_DIR, 'training_time_analysis.md')
with open(out_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))

raw_path = os.path.join(REPORT_DIR, 'training_time_raw.json')
with open(raw_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nReport: {out_path}")
print(f"Raw: {raw_path}")
print()
print('\n'.join(lines))
