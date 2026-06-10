#!/usr/bin/env python
"""
Aggregate multi-seed THEIA / TF8L results and ablations into one report
(paper Tables 1-2, ablation summary).

Outputs:
    multi_seed_results/reports/final_report.md
    multi_seed_results/reports/raw_data.json
"""
import os, json, sys
from collections import defaultdict
from datetime import datetime

ROOT = r'multi_seed_results'
REPORT_DIR = os.path.join(ROOT, 'reports')
os.makedirs(REPORT_DIR, exist_ok=True)

RULES = [
    "F_and_U", "T_and_U", "U_and_F", "U_and_T",
    "T_or_U",  "F_or_U",  "U_or_T",  "U_or_F",
    "F_imp_U", "T_imp_U", "T_iff_U", "F_iff_U",
]
RULE_PRETTY = {
    "F_and_U": "F and U",  "T_and_U": "T and U",
    "U_and_F": "U and F",  "U_and_T": "U and T",
    "T_or_U":  "T or U",   "F_or_U":  "F or U",
    "U_or_T":  "U or T",   "U_or_F":  "U or F",
    "F_imp_U": "F -> U",   "T_imp_U": "T -> U",
    "T_iff_U": "T <-> U",  "F_iff_U": "F <-> U",
}
EXPECTED = {
    "F_and_U": "F", "T_and_U": "U", "U_and_F": "F", "U_and_T": "U",
    "T_or_U":  "T", "F_or_U":  "U", "U_or_T":  "T", "U_or_F":  "U",
    "F_imp_U": "T", "T_imp_U": "U", "T_iff_U": "U", "F_iff_U": "U",
}

def mean_std(xs):
    if not xs: return (None, None)
    m = sum(xs) / len(xs)
    v = sum((x - m) ** 2 for x in xs) / len(xs)
    return (m, v ** 0.5)

def fmt_pct(m, s):
    if m is None: return "--"
    return f"{m:6.2f} +/- {s:.2f}"

def load_jsons(pattern_dirs):
    """Load summary.json from each dir, return list of dicts."""
    out = []
    for d in pattern_dirs:
        path = os.path.join(d, 'summary.json')
        if os.path.exists(path):
            with open(path) as f:
                out.append(json.load(f))
        else:
            print(f"WARNING: missing {path}", file=sys.stderr)
    return out

def load_kleene_fixed_v2(checkpoint_dir):
    """Load kleene_fixed_v2.json (from reeval) if exists."""
    path = os.path.join(checkpoint_dir, 'kleene_fixed_v2.json')
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None

# === Collect THEIA 5-seed data ===
# Seed 42 is in theia/seed_42 with kleene_fixed_v2.json (from reeval)
# Seeds 123, 256, 777, 999 are in theia_v2/seed_*/ with summary.json
theia_data = []

# Seed 42: load kleene from reeval
seed42_dir = os.path.join(ROOT, 'theia', 'seed_42')
seed42_kleene = load_kleene_fixed_v2(seed42_dir)
seed42_summary_path = os.path.join(seed42_dir, 'summary.json')
if seed42_kleene and os.path.exists(seed42_summary_path):
    with open(seed42_summary_path) as f:
        s = json.load(f)
    s['kleene_passed'] = seed42_kleene['kleene_passed']
    s['kleene_per_rule'] = seed42_kleene['kleene_per_rule']
    s['_source'] = 'theia/seed_42 + reeval kleene_fixed_v2'
    theia_data.append(s)
else:
    print("WARNING: seed 42 data incomplete")

# Seeds 123-999: from theia_v2
for seed in [123, 256, 777, 999]:
    d = os.path.join(ROOT, 'theia_v2', f'seed_{seed}')
    path = os.path.join(d, 'summary.json')
    if os.path.exists(path):
        with open(path) as f:
            s = json.load(f)
        s['_source'] = f'theia_v2/seed_{seed}'
        theia_data.append(s)
    else:
        print(f"WARNING: missing theia_v2/seed_{seed}/summary.json")

print(f"Loaded {len(theia_data)} THEIA seeds")

# === Collect TF8L 5-seed data ===
tf8l_data = []
for seed in [42, 123, 256, 777, 999]:
    d = os.path.join(ROOT, 'tf8l', f'seed_{seed}')
    path = os.path.join(d, 'summary.json')
    if os.path.exists(path):
        with open(path) as f:
            s = json.load(f)
        s['_source'] = f'tf8l/seed_{seed}'
        tf8l_data.append(s)
    else:
        print(f"WARNING: missing tf8l/seed_{seed}/summary.json")

# Also check reeval_tf8l_v2 result for seed 42 if exists
tf8l_seed42_reeval = load_kleene_fixed_v2(os.path.join(ROOT, 'tf8l', 'seed_42'))
if tf8l_seed42_reeval and tf8l_data:
    # Use reeval kleene over training-time kleene if both exist
    for s in tf8l_data:
        if s['seed'] == 42:
            s['kleene_passed'] = tf8l_seed42_reeval['kleene_passed']
            s['kleene_per_rule'] = tf8l_seed42_reeval['kleene_per_rule']
            s['_source'] += ' + reeval kleene_fixed_v2'
            break

print(f"Loaded {len(tf8l_data)} TF8L seeds")

# === Collect ablations ===
ablations = {}
for name, subdir in [
    ('subspace', 'subspace'),
    ('nobridge', 'nobridge'),
    ('punk005',  'punk005'),
]:
    d = os.path.join(ROOT, 'ablations', subdir, 'seed_42')
    path = os.path.join(d, 'summary.json')
    if os.path.exists(path):
        with open(path) as f:
            ablations[name] = json.load(f)
    else:
        ablations[name] = None
        print(f"WARNING: missing ablations/{subdir}/seed_42/summary.json")

# === Compute stats ===
def per_rule_stats(data):
    out = {}
    for rule in RULES:
        vals = [d['kleene_per_rule'].get(rule) for d in data if rule in d.get('kleene_per_rule', {})]
        vals = [v for v in vals if v is not None]
        out[rule] = mean_std(vals)
    return out

theia_per_rule = per_rule_stats(theia_data)
tf8l_per_rule = per_rule_stats(tf8l_data)

theia_overall_acc = mean_std([d['overall_acc'] for d in theia_data if 'overall_acc' in d])
tf8l_overall_acc = mean_std([d['overall_acc'] for d in tf8l_data if 'overall_acc' in d])

theia_passed = mean_std([d['kleene_passed'] for d in theia_data if 'kleene_passed' in d])
tf8l_passed = mean_std([d['kleene_passed'] for d in tf8l_data if 'kleene_passed' in d])

theia_epochs = mean_std([d.get('converge_epoch', d.get('max_epochs')) for d in theia_data])
theia_time_min = mean_std([d['total_time_sec']/60 for d in theia_data if 'total_time_sec' in d])
tf8l_time_min = mean_std([d['total_time_sec']/60 for d in tf8l_data if 'total_time_sec' in d])

# === Generate Markdown Report ===
lines = []
L = lines.append

L(f"# THEIA Direction A — Final Data Report")
L(f"")
L(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
L(f"")
L(f"All numbers below use the **doubly-fixed Kleene diagnostic**:")
L(f"- Fix #1: vo==2 uses du[:]=True (not au[:]=True) to avoid c_unknown pollution")
L(f"- Fix #2: vo==1 uses REL_GTE (not REL_GT) to handle c=0 edge case")
L(f"")
L(f"---")
L(f"")

# === Section 1: THEIA 5-seed ===
L(f"## Section 1: THEIA 5-seed (main result)")
L(f"")
L(f"Number of seeds loaded: **{len(theia_data)}** (expected 5)")
L(f"")
if len(theia_data) >= 5:
    L(f"### Overall")
    L(f"")
    L(f"| Metric | Value |")
    L(f"|---|---|")
    L(f"| Overall accuracy (mean +/- std) | {theia_overall_acc[0]*100:.4f} +/- {theia_overall_acc[1]*100:.4f}% |")
    L(f"| Rules passed (mean +/- std) | {theia_passed[0]:.1f} +/- {theia_passed[1]:.1f} / 12 |")
    L(f"| Convergence epoch (mean +/- std) | {theia_epochs[0]:.0f} +/- {theia_epochs[1]:.0f} |")
    L(f"| Wall time min (mean +/- std) | {theia_time_min[0]:.1f} +/- {theia_time_min[1]:.1f} |")
    L(f"| Parameters | {theia_data[0].get('params', 'N/A'):,} |")
    L(f"")

L(f"### Per-rule accuracy (5 seeds)")
L(f"")
L(f"| Rule | Expected | THEIA mean +/- std (%) |")
L(f"|---|---|---|")
for rule in RULES:
    m, s = theia_per_rule[rule]
    cell = fmt_pct(m, s) if m is not None else "--"
    L(f"| {RULE_PRETTY[rule]} | {EXPECTED[rule]} | {cell} |")
L(f"")
L(f"### Per-seed raw")
L(f"")
L(f"| Seed | Overall | Passed | Source |")
L(f"|---|---|---|---|")
for d in theia_data:
    L(f"| {d['seed']} | {d['overall_acc']*100:.4f}% | {d['kleene_passed']}/12 | `{d.get('_source','?')}` |")
L(f"")
L(f"### Per-seed per-rule (full matrix)")
L(f"")
header = "| Rule |"
for d in theia_data: header += f" s{d['seed']} |"
L(header)
L("|---|" + "---|" * len(theia_data))
for rule in RULES:
    row = f"| {RULE_PRETTY[rule]} |"
    for d in theia_data:
        v = d.get('kleene_per_rule', {}).get(rule)
        row += f" {v:.2f} |" if v is not None else " -- |"
    L(row)
L(f"")
L(f"---")
L(f"")

# === Section 2: TF8L 5-seed ===
L(f"## Section 2: TF8L 5-seed")
L(f"")
L(f"Number of seeds loaded: **{len(tf8l_data)}** (expected 5)")
L(f"")
if len(tf8l_data) >= 1:
    L(f"### Overall")
    L(f"")
    L(f"| Metric | Value |")
    L(f"|---|---|")
    if tf8l_overall_acc[0] is not None:
        L(f"| Overall accuracy (mean +/- std) | {tf8l_overall_acc[0]*100:.4f} +/- {tf8l_overall_acc[1]*100:.4f}% |")
    if tf8l_passed[0] is not None:
        L(f"| Rules passed (mean +/- std) | {tf8l_passed[0]:.1f} +/- {tf8l_passed[1]:.1f} / 12 |")
    if tf8l_time_min[0] is not None:
        L(f"| Wall time min (mean +/- std) | {tf8l_time_min[0]:.1f} +/- {tf8l_time_min[1]:.1f} |")
    if tf8l_data[0].get('params'):
        L(f"| Parameters | {tf8l_data[0]['params']:,} |")
    L(f"")

if len(tf8l_data) >= 1:
    L(f"### Per-rule accuracy")
    L(f"")
    L(f"| Rule | Expected | TF8L mean +/- std (%) |")
    L(f"|---|---|---|")
    for rule in RULES:
        m, s = tf8l_per_rule[rule]
        cell = fmt_pct(m, s) if m is not None else "--"
        L(f"| {RULE_PRETTY[rule]} | {EXPECTED[rule]} | {cell} |")
    L(f"")

L(f"### Per-seed raw")
L(f"")
L(f"| Seed | Overall | Passed | Source |")
L(f"|---|---|---|---|")
for d in tf8l_data:
    L(f"| {d['seed']} | {d.get('overall_acc',0)*100:.4f}% | {d.get('kleene_passed','?')}/12 | `{d.get('_source','?')}` |")
L(f"")
L(f"---")
L(f"")

# --- TF8L parity check ---
L(f"## Section 3: TF8L parity check")
L(f"")
L(f"Parity condition: TF8L reaches >=99.9% overall accuracy AND >=11/12 Kleene rules across all 5 seeds.")
L(f"")
if len(tf8l_data) >= 5:
    all_overall_ok = all(d.get('overall_acc', 0) >= 0.999 for d in tf8l_data)
    all_kleene_ok = all(d.get('kleene_passed', 0) >= 11 for d in tf8l_data)
    if all_overall_ok and all_kleene_ok:
        L(f"**Parity condition met.** All 5 TF8L seeds reach >=99.9% overall and >=11/12 Kleene.")
        L(f"You can use the framing: 'TF8L reaches identical correctness to THEIA but with qualitatively different representational dynamics.'")
    elif all_overall_ok and not all_kleene_ok:
        L(f"**MIXED:** All 5 TF8L seeds reach >=99.9% overall but some <11/12 Kleene. Reframe needed: emphasize 'overall accuracy parity but per-rule consistency gap'.")
    else:
        L(f"**Parity condition not met.** TF8L does not reliably reach the parity threshold.")
    L(f"")
    L(f"Per-seed check:")
    L(f"")
    L(f"| Seed | Overall >= 99.9%? | Kleene >= 11/12? |")
    L(f"|---|---|---|")
    for d in tf8l_data:
        ok1 = "YES" if d.get('overall_acc',0) >= 0.999 else "NO"
        ok2 = "YES" if d.get('kleene_passed',0) >= 11 else "NO"
        L(f"| {d['seed']} | {ok1} | {ok2} |")
else:
    L(f"**INSUFFICIENT DATA**: only {len(tf8l_data)}/5 TF8L seeds available.")
L(f"")
L(f"---")
L(f"")

# === Section 4: Ablations ===
L(f"## Section 4: Ablations (1 seed each, fixed diagnostic)")
L(f"")
ablation_descriptions = {
    'subspace': 'Single MLP Logic Engine (replaces parallel C/D/I subspaces)',
    'nobridge': 'No bridge layers (bridge_ao, bridge_as removed)',
    'punk005':  'Trained at P_unk=0.05, evaluated at P_unk=0.50',
}
for name, desc in ablation_descriptions.items():
    a = ablations.get(name)
    L(f"### Ablation: {name}")
    L(f"")
    L(f"*{desc}*")
    L(f"")
    if a is None:
        L(f"**STATUS: NOT RUN**")
    else:
        L(f"| Metric | Value |")
        L(f"|---|---|")
        L(f"| Params | {a.get('params', 'N/A'):,} |")
        L(f"| Overall accuracy | {a.get('overall_acc', 0)*100:.4f}% |")
        L(f"| Kleene passed | {a.get('kleene_passed', '?')}/12 |")
        L(f"| Convergence epoch | {a.get('converge_epoch', '?')} |")
        if name == 'punk005':
            L(f"| Train P_unk | {a.get('P_unk_train', 0.05)} |")
            L(f"| Test P_unk | {a.get('P_unk_test', 0.50)} |")
            L(f"| Overall acc on P=0.50 set | {a.get('overall_acc_p050_test', 0)*100:.4f}% |")
            if 'p050_per_class' in a:
                L(f"| Per-class on P=0.50 | {a['p050_per_class']} |")
        L(f"")
        L(f"Per-rule (Kleene diagnostic, P-independent):")
        L(f"")
        L(f"| Rule | Expected | Acc |")
        L(f"|---|---|---|")
        for rule in RULES:
            v = a.get('kleene_per_rule', {}).get(rule)
            v_str = f"{v:.2f}%" if v is not None else "--"
            L(f"| {RULE_PRETTY[rule]} | {EXPECTED[rule]} | {v_str} |")
    L(f"")

L(f"---")
L(f"")

# === Section 5: Paper-ready snippets ===
L(f"## Section 5: Paper-ready snippets")
L(f"")
L(f"### Table 1 (THEIA-only, 5 seeds)")
L(f"")
L(f"```")
L(f"Model    | Accuracy           | Params  | Epochs (avg) | Time (min)")
if theia_overall_acc[0] is not None:
    L(f"THEIA    | {theia_overall_acc[0]*100:.2f} +/- {theia_overall_acc[1]*100:.2f}%   | {theia_data[0].get('params',0)/1e6:.2f}M   | ~{theia_epochs[0]:.0f}        | ~{theia_time_min[0]:.0f}")
L(f"```")
L(f"")
L(f"### Table 2 (THEIA Kleene, 5 seeds)")
L(f"")
L(f"```")
L(f"Expression  | Expected | THEIA accuracy (mean +/- std)")
for rule in RULES:
    m, s = theia_per_rule[rule]
    cell = f"{m:.2f} +/- {s:.2f}" if m is not None else "--"
    L(f"{RULE_PRETTY[rule]:11s} | {EXPECTED[rule]:8s} | {cell}")
if theia_passed[0] is not None:
    L(f"Passed > 99% | --       | {theia_passed[0]:.1f} +/- {theia_passed[1]:.1f} / 12")
L(f"```")
L(f"")
L(f"### Ablation summary sentence (drop into §4.5)")
L(f"")
spass = ablations.get('subspace', {}).get('kleene_passed', '?') if ablations.get('subspace') else 'NOT RUN'
nbpass = ablations.get('nobridge', {}).get('kleene_passed', '?') if ablations.get('nobridge') else 'NOT RUN'
ppass = ablations.get('punk005', {}).get('kleene_passed', '?') if ablations.get('punk005') else 'NOT RUN'
ppct = ablations.get('punk005', {}).get('overall_acc_p050_test', 0) * 100 if ablations.get('punk005') else 0
L(f"> Replacing the parallel C/D/I subspaces with a single MLP still achieves {spass}/12 Kleene rules; ")
L(f"> removing the bridge layers achieves {nbpass}/12; ")
L(f"> a model trained at P_unk=0.05 achieves {ppass}/12 Kleene rules and {ppct:.2f}% overall accuracy when ")
L(f"> evaluated at P_unk=0.50, demonstrating that the model learns Kleene rules rather than fitting the training Unknown distribution.")
L(f"")
L(f"---")
L(f"")
L(f"## Section 6: Data integrity check")
L(f"")
total_seeds_run = len(theia_data) + len(tf8l_data) + sum(1 for v in ablations.values() if v is not None)
total_seeds_expected = 5 + 5 + 3
L(f"- Seeds completed: **{total_seeds_run} / {total_seeds_expected}**")
if total_seeds_run < total_seeds_expected:
    L(f"- **MISSING**: report is incomplete; rerun this script after remaining seeds finish")
else:
    L(f"- All expected data present.")
L(f"")
all_theia_12 = all(d.get('kleene_passed', 0) == 12 for d in theia_data)
if all_theia_12 and len(theia_data) >= 5:
    L(f"- **GREEN: All 5 THEIA seeds achieved 12/12 Kleene.** Direction A 'complete Kleene' claim is supported.")
elif theia_data:
    bad = [d['seed'] for d in theia_data if d.get('kleene_passed', 0) < 12]
    L(f"- **RED: THEIA seeds with <12/12: {bad}.** Direction A 'complete Kleene' claim is NOT supported. Reframe needed.")
L(f"")

# === Save ===
report = '\n'.join(lines)
report_path = os.path.join(REPORT_DIR, 'final_report.md')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report)

raw = {
    'theia_5seed': theia_data,
    'tf8l_5seed': tf8l_data,
    'ablations': ablations,
    'stats': {
        'theia_overall_acc_mean': theia_overall_acc[0],
        'theia_overall_acc_std': theia_overall_acc[1],
        'theia_per_rule_mean_std': {k: list(v) for k,v in theia_per_rule.items()},
        'tf8l_overall_acc_mean': tf8l_overall_acc[0],
        'tf8l_overall_acc_std': tf8l_overall_acc[1],
        'tf8l_per_rule_mean_std': {k: list(v) for k,v in tf8l_per_rule.items()},
    },
}
raw_path = os.path.join(REPORT_DIR, 'raw_data.json')
with open(raw_path, 'w') as f:
    json.dump(raw, f, indent=2, default=str)

print(f"\n=== REPORT WRITTEN ===")
print(f"Markdown: {report_path}")
print(f"Raw JSON: {raw_path}")
print(f"\n{'='*60}")
print(report)
