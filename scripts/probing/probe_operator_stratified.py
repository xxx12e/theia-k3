"""
Operator-stratified probe of the Set boundary on the non-Unknown (NU) stratum (§5.3).

Tests whether the +4pp NU lift above the majority baseline observed in
probe_stratified.py is explained by val_set encoding (which the Set Engine is
designed to compute; e.g. under OR, val_set=T fixes verdict=T regardless of
val_ord, so val_set is definitionally correlated with the verdict on NU samples).

Per (seed, op in {AND, OR, IMP, IFF}):
  - Filter to NU stratum (label in {F, T}) AND that op
  - 70/30 split (per-op seed for reproducibility)
  - Train LinearSVC + StandardScaler on Set boundary hidden, score test acc
  - Compute the val_set-only ceiling: rule-based predictor using ONLY
    ground-truth val_set in {F, T, U} (train-majority verdict per val_set
    value within the (op, NU) subset, applied to test)

Decision rule:
  - empirical SVC acc <= val_set-only ceiling + 2pp -> lift explained by
    val_set encoding
  - empirical SVC acc > val_set-only ceiling + 5pp -> additional leakage
    beyond val_set
"""
import json
import os
import sys
import time
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import numpy as np
import torch
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler

from _theia_model_def import (
    IsisV9, DEVICE,
    NUM_RANGE, SET_DIM, P_UNKNOWN,
    VAL_FALSE, VAL_TRUE, VAL_UNKNOWN,
    N_RELS, N_ARITH, N_OPS,
    OP_AND, OP_OR, OP_NOT, OP_IMPLIES, OP_IFF,
    apply_logic,
)

# Same protocol constants as probe_stratified.py
CKPTS = [
    (42,  'multi_seed_results/theia/seed_42/checkpoint.pth'),
    (123, 'multi_seed_results/theia_v2/seed_123/checkpoint.pth'),
    (256, 'multi_seed_results/theia_v2/seed_256/checkpoint.pth'),
    (777, 'multi_seed_results/theia_v2/seed_777/checkpoint.pth'),
    (999, 'multi_seed_results/theia_v2/seed_999/checkpoint.pth'),
]
N_SAMPLES = 50_000
DATA_SEED = 999
TRAIN_FRAC = 0.7
SPLIT_SEED = 0

# 4 binary ops only (NOT excluded: it is unary)
OPS = [
    ('AND', OP_AND),
    ('OR',  OP_OR),
    ('IMP', OP_IMPLIES),
    ('IFF', OP_IFF),
]
TV_NAME = {0: 'F', 1: 'T', 2: 'U'}

# Decision thresholds
DELTA_OK_PP   = 2.0   # empirical - ceiling <= 2pp -> consistent with hypothesis
DELTA_FAIL_PP = 5.0   # empirical - ceiling >  5pp -> additional leakage


def forward_with_hidden(model, a, b, d, set_bits, s_unk, a_unk, b_unk, d_unk,
                        arith, rel, op):
    c_vec = model.arith_eng(a, b, a_unk, b_unk, arith)
    c_for_ord = model.bridge_ao(c_vec) + c_vec
    c_for_set = model.bridge_as(c_vec) + c_vec
    ord_vec = model.order_eng(c_for_ord, d, d_unk, rel)
    set_vec = model.set_eng(c_for_set, set_bits, s_unk)
    return {'set': set_vec}  # only the Set boundary is probed here


def gen_data_extended(seed, n=N_SAMPLES):
    """Same generation as probe_stratified.gen_data, but additionally returns
    val_ord and val_set ground truth (needed for the val_set-only ceiling)."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    a = torch.randint(1, NUM_RANGE+1, (n,), device=DEVICE)
    b = torch.randint(1, NUM_RANGE+1, (n,), device=DEVICE)
    d = torch.randint(0, NUM_RANGE+1, (n,), device=DEVICE)
    arith = torch.randint(0, N_ARITH, (n,), device=DEVICE)
    rel   = torch.randint(0, N_RELS,  (n,), device=DEVICE)
    op    = torch.randint(0, N_OPS,   (n,), device=DEVICE)
    au = torch.rand(n, device=DEVICE) < P_UNKNOWN
    bu = torch.rand(n, device=DEVICE) < P_UNKNOWN
    du = torch.rand(n, device=DEVICE) < P_UNKNOWN
    su = torch.rand(n, device=DEVICE) < P_UNKNOWN
    c = torch.zeros(n, dtype=torch.long, device=DEVICE)
    c[arith==0] = torch.clamp(a+b, 0, NUM_RANGE)[arith==0]
    c[arith==1] = torch.abs(a-b)[arith==1]
    c[arith==2] = torch.clamp(a*b, 0, NUM_RANGE)[arith==2]
    c[arith==3] = (a % torch.clamp(b, 1, NUM_RANGE))[arith==3]
    c = torch.clamp(c, 0, NUM_RANGE)
    c_unk = au | bu
    ord_unk = c_unk | du
    ord_v = torch.zeros(n, dtype=torch.long, device=DEVICE)
    rt = (((rel==0) & (c >  d)) | ((rel==1) & (c <  d)) |
          ((rel==2) & (c == d)) | ((rel==3) & (c >= d)) |
          ((rel==4) & (c <= d)) | ((rel==5) & (c != d)))
    ord_v[rt] = VAL_TRUE
    val_o = torch.where(ord_unk, torch.tensor(VAL_UNKNOWN, device=DEVICE), ord_v)
    sb = torch.randint(0, 2, (n, SET_DIM), dtype=torch.float32, device=DEVICE)
    sou = su | c_unk
    ci = c.clamp(0, SET_DIM-1)
    ins = sb[torch.arange(n, device=DEVICE), ci].bool()
    sv = torch.where(ins, torch.tensor(VAL_TRUE,  device=DEVICE),
                          torch.tensor(VAL_FALSE, device=DEVICE))
    val_s = torch.where(sou, torch.tensor(VAL_UNKNOWN, device=DEVICE), sv)
    target = apply_logic(op, val_o, val_s)
    return {
        'a_norm': a.float() / NUM_RANGE,
        'b_norm': b.float() / NUM_RANGE,
        'd_norm': d.float() / NUM_RANGE,
        'sb': sb, 's_unk': su, 'a_unk': au, 'b_unk': bu, 'd_unk': du,
        'arith': arith, 'rel': rel, 'op': op,
        'val_ord': val_o, 'val_set': val_s, 'target': target,
    }


def val_set_only_ceiling(val_set_tr, y_tr, val_set_te, y_te):
    """Rule-based predictor that uses ONLY val_set in {0,1,2} as input.
    For each val_set value, picks the majority verdict observed in train.
    Returns test accuracy.
    Also reports the per-(val_set) majority class for transparency."""
    rule = {}
    for vs in (VAL_FALSE, VAL_TRUE, VAL_UNKNOWN):
        mask = (val_set_tr == vs)
        if mask.sum() == 0:
            rule[vs] = None
            continue
        y_sub = y_tr[mask]
        n_F = int((y_sub == VAL_FALSE).sum())
        n_T = int((y_sub == VAL_TRUE).sum())
        # Tie-break: pick T (matches paper's NU majority class)
        rule[vs] = VAL_TRUE if n_T >= n_F else VAL_FALSE

    pred = np.empty_like(y_te)
    for vs, cls in rule.items():
        m = (val_set_te == vs)
        if cls is not None:
            pred[m] = cls
        elif m.any():
            # No train support for this val_set value -> fall back to global majority
            n_F = int((y_tr == VAL_FALSE).sum())
            n_T = int((y_tr == VAL_TRUE).sum())
            pred[m] = VAL_TRUE if n_T >= n_F else VAL_FALSE
    return float((pred == y_te).mean()), {TV_NAME[k]: (TV_NAME[v] if v is not None else None)
                                            for k, v in rule.items()}


def probe_svc(X_tr, y_tr, X_te, y_te):
    sc = StandardScaler()
    Xs_tr = sc.fit_transform(X_tr)
    Xs_te = sc.transform(X_te)
    dual = X_tr.shape[0] < X_tr.shape[1]
    clf = LinearSVC(C=1.0, max_iter=2000, dual=dual)
    clf.fit(Xs_tr, y_tr)
    return float(clf.score(Xs_te, y_te))


def run_seed(seed, ckpt_path):
    print(f"\n{'='*72}\n  seed {seed}  ckpt: {ckpt_path}\n{'='*72}")
    if not os.path.exists(ckpt_path):
        print(f"  [MISSING]"); return None

    model = IsisV9().to(DEVICE)
    state = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    print(f"  generating {N_SAMPLES} samples (data_seed={DATA_SEED}) ...")
    data = gen_data_extended(DATA_SEED, n=N_SAMPLES)

    print(f"  forward_with_hidden (Set boundary only) ...")
    BATCH = 8192
    set_hidden_chunks = []
    with torch.no_grad():
        for i in range(0, N_SAMPLES, BATCH):
            j = min(i + BATCH, N_SAMPLES)
            with torch.amp.autocast('cuda'):
                h = forward_with_hidden(
                    model,
                    data['a_norm'][i:j], data['b_norm'][i:j], data['d_norm'][i:j],
                    data['sb'][i:j], data['s_unk'][i:j],
                    data['a_unk'][i:j], data['b_unk'][i:j], data['d_unk'][i:j],
                    data['arith'][i:j], data['rel'][i:j], data['op'][i:j],
                )
            set_hidden_chunks.append(h['set'].float().cpu().numpy())
    set_hidden = np.concatenate(set_hidden_chunks, axis=0)

    target_np  = data['target'].cpu().numpy()
    val_set_np = data['val_set'].cpu().numpy()
    op_np      = data['op'].cpu().numpy()

    # Restrict to NU stratum (label in {F, T})
    nu_mask = target_np != VAL_UNKNOWN

    out = {'seed': seed, 'checkpoint': ckpt_path, 'per_op': {}}

    print(f"  {'op':<5} {'n_NU':>6} {'svc_acc':>9} {'vset_ceil':>10} "
          f"{'delta':>8}  rule(F/T/U)")
    for op_name, op_idx in OPS:
        m_op_nu = nu_mask & (op_np == op_idx)
        n = int(m_op_nu.sum())
        if n < 100:
            print(f"  {op_name:<5} insufficient samples ({n})")
            continue

        idx_all = np.where(m_op_nu)[0]
        # Per-(seed, op) split for reproducibility: RandomState(SPLIT_SEED + seed*100 + op_idx)
        rng = np.random.RandomState(SPLIT_SEED + seed * 100 + op_idx)
        perm = rng.permutation(len(idx_all))
        n_train = int(len(idx_all) * TRAIN_FRAC)
        tr_idx = idx_all[perm[:n_train]]
        te_idx = idx_all[perm[n_train:]]

        X_tr, X_te = set_hidden[tr_idx], set_hidden[te_idx]
        y_tr, y_te = target_np[tr_idx], target_np[te_idx]
        vs_tr, vs_te = val_set_np[tr_idx], val_set_np[te_idx]

        svc_acc = probe_svc(X_tr, y_tr, X_te, y_te)
        ceil_acc, rule = val_set_only_ceiling(vs_tr, y_tr, vs_te, y_te)
        delta_pp = (svc_acc - ceil_acc) * 100.0

        out['per_op'][op_name] = {
            'n_train': int(n_train),
            'n_test':  int(len(idx_all) - n_train),
            'svc_acc_pct':       svc_acc * 100.0,
            'val_set_ceiling_pct': ceil_acc * 100.0,
            'svc_minus_ceil_pp': delta_pp,
            'val_set_majority_rule': rule,
        }

        rule_str = " ".join(f"{k}->{v}" if v else f"{k}->-" for k, v in rule.items())
        print(f"  {op_name:<5} {n:>6} {svc_acc*100:>8.2f}% "
              f"{ceil_acc*100:>9.2f}% {delta_pp:>+7.2f}pp  [{rule_str}]")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def aggregate(per_seed):
    seeds_used = [r['seed'] for r in per_seed if r is not None]
    agg = {'n_seeds': len(seeds_used), 'seeds_used': seeds_used, 'per_op': {}}
    for op_name, _ in OPS:
        svc_vals  = [r['per_op'][op_name]['svc_acc_pct']
                     for r in per_seed if r and op_name in r['per_op']]
        ceil_vals = [r['per_op'][op_name]['val_set_ceiling_pct']
                     for r in per_seed if r and op_name in r['per_op']]
        deltas    = [r['per_op'][op_name]['svc_minus_ceil_pp']
                     for r in per_seed if r and op_name in r['per_op']]
        if not svc_vals:
            continue
        verdict = ('within-2pp' if max(deltas) <= DELTA_OK_PP else
                   'within-5pp' if max(deltas) <= DELTA_FAIL_PP else
                   'EXCEEDS-5pp')
        agg['per_op'][op_name] = {
            'svc_acc_mean':       float(np.mean(svc_vals)),
            'svc_acc_std':        float(np.std(svc_vals)),
            'val_set_ceil_mean':  float(np.mean(ceil_vals)),
            'val_set_ceil_std':   float(np.std(ceil_vals)),
            'delta_pp_mean':      float(np.mean(deltas)),
            'delta_pp_std':       float(np.std(deltas)),
            'delta_pp_max':       float(max(deltas)),
            'per_seed_svc':       svc_vals,
            'per_seed_ceil':      ceil_vals,
            'per_seed_delta_pp':  deltas,
            'verdict':            verdict,
        }
    return agg


def render_markdown(per_seed, agg):
    lines = []
    lines.append("# Operator-Stratified Probe — Set Boundary, NU Subset")
    lines.append("")
    lines.append(f"**Setup**: 5 checkpoints {agg['seeds_used']}, 50K samples each, "
                 f"`data_seed={DATA_SEED}`. Restricted to **(Set boundary, NU stratum, "
                 f"single op)** subsets. Train/test 70/30 per-op split.")
    lines.append("")
    lines.append("**Probe**: LinearSVC + StandardScaler (verbatim protocol from "
                 "`reprobe_theia_v2.py`).")
    lines.append("")
    lines.append("**Ceiling**: rule-based predictor that uses ONLY ground-truth "
                 "`val_set ∈ {F,T,U}`; for each `(op, val_set)` cell picks the "
                 "train-majority verdict; reports test accuracy. This is the "
                 "*definitional* upper bound if Set boundary perfectly encodes "
                 "`val_set` and probe optimally exploits it.")
    lines.append("")
    lines.append("**Decision rule**:")
    lines.append("- `empirical − ceiling ≤ +2pp`: +4pp is **explained by val_set**; "
                 "H2 survives with paper-language tweak")
    lines.append("- `empirical − ceiling > +5pp`: **substantive verdict leakage** "
                 "beyond val_set; H2 needs revision")
    lines.append("")
    lines.append("## Aggregate (mean ± std over 5 seeds)")
    lines.append("")
    lines.append("| Op | Empirical SVC acc | val_set-only ceiling | Δ (empirical − ceiling) | "
                 "Δ max (single seed) | Verdict |")
    lines.append("|---|---|---|---|---|---|")
    for op_name, _ in OPS:
        if op_name not in agg['per_op']:
            continue
        v = agg['per_op'][op_name]
        lines.append(f"| {op_name} | "
                     f"{v['svc_acc_mean']:.2f}% ± {v['svc_acc_std']:.2f}% | "
                     f"{v['val_set_ceil_mean']:.2f}% ± {v['val_set_ceil_std']:.2f}% | "
                     f"{v['delta_pp_mean']:+.2f} ± {v['delta_pp_std']:.2f} pp | "
                     f"{v['delta_pp_max']:+.2f} pp | "
                     f"**{v['verdict']}** |")
    lines.append("")
    lines.append("## Per-seed empirical SVC acc (Set boundary, NU, op-restricted)")
    lines.append("")
    lines.append("| Op | " + " | ".join(f"seed {s}" for s in agg['seeds_used']) + " |")
    lines.append("|" + "---|" * (1 + len(agg['seeds_used'])))
    for op_name, _ in OPS:
        if op_name not in agg['per_op']:
            continue
        per = agg['per_op'][op_name]['per_seed_svc']
        lines.append(f"| {op_name} | " + " | ".join(f"{x:.2f}%" for x in per) + " |")
    lines.append("")
    lines.append("## Per-seed val_set-only ceiling (rule-based, no probe)")
    lines.append("")
    lines.append("| Op | " + " | ".join(f"seed {s}" for s in agg['seeds_used']) + " |")
    lines.append("|" + "---|" * (1 + len(agg['seeds_used'])))
    for op_name, _ in OPS:
        if op_name not in agg['per_op']:
            continue
        per = agg['per_op'][op_name]['per_seed_ceil']
        lines.append(f"| {op_name} | " + " | ".join(f"{x:.2f}%" for x in per) + " |")
    lines.append("")
    lines.append("## Per-seed val_set→verdict majority rule learned (op × val_set)")
    lines.append("")
    lines.append("Tie-break: T (matches paper's NU majority class).")
    lines.append("")
    lines.append("| Op | " + " | ".join(f"seed {s}" for s in agg['seeds_used']) + " |")
    lines.append("|" + "---|" * (1 + len(agg['seeds_used'])))
    for op_name, _ in OPS:
        rules = []
        for r in per_seed:
            if r is None or op_name not in r['per_op']:
                rules.append("-")
            else:
                rd = r['per_op'][op_name]['val_set_majority_rule']
                rules.append("/".join(f"{k}={v if v else '-'}" for k, v in rd.items()))
        lines.append(f"| {op_name} | " + " | ".join(rules) + " |")
    lines.append("")
    # Final verdict
    verdicts = {agg['per_op'][op]['verdict'] for op in agg['per_op']}
    if verdicts == {'within-2pp'}:
        lines.append("## Conclusion")
        lines.append("")
        lines.append("**All 4 ops: empirical ≤ val_set-only ceiling + 2pp.** The +4pp "
                     "Set boundary leakage observed in the baseline stratified probe is **fully attributable** "
                     "to `val_set` encoding (which the Set Engine is *designed* to compute). "
                     "H2 (delayed verdict) survives with paper-language tweak: at the Set "
                     "boundary, what looks like verdict leakage is actually "
                     "*intermediate-variable* leakage (`val_set`), which is consistent with "
                     "the architecture's own contract.")
    elif 'EXCEEDS-5pp' in verdicts:
        lines.append("## Conclusion")
        lines.append("")
        lines.append("**At least one op: empirical > val_set-only ceiling + 5pp.** Set "
                     "boundary contains verdict information **beyond** val_set encoding. "
                     "H2 has substantive hole; further investigation needed.")
    else:
        lines.append("## Conclusion")
        lines.append("")
        lines.append("**Mixed result**: some ops within +2pp (consistent with val_set "
                     "explanation), others between +2 and +5pp (intermediate range, "
                     "neither clearly explained nor clearly violated). Inspection per op "
                     "needed below.")
    return "\n".join(lines) + "\n"


def main():
    t0 = time.time()
    per_seed = []
    for seed, ckpt in CKPTS:
        per_seed.append(run_seed(seed, ckpt))
    agg = aggregate(per_seed)

    out = {
        'protocol': {
            'n_samples': N_SAMPLES,
            'data_seed': DATA_SEED,
            'train_frac': TRAIN_FRAC,
            'split_seed_base': SPLIT_SEED,
            'note_on_split': 'per-op split via SPLIT_SEED + seed*100 + op_idx',
            'boundary': 'Set',
            'stratum': 'NU (label in {F, T})',
            'ops_tested': [op for op, _ in OPS],
            'probe': 'LinearSVC C=1.0 max_iter=2000 + StandardScaler',
            'ceiling': 'rule-based: per (op, val_set) majority verdict on train, applied on test',
            'decision_thresholds_pp': {'OK': DELTA_OK_PP, 'FAIL': DELTA_FAIL_PP},
        },
        'per_seed': [r for r in per_seed if r is not None],
        'aggregate': agg,
    }
    with open('probe_operator_stratified.json', 'w') as f:
        json.dump(out, f, indent=2)

    md = render_markdown(per_seed, agg)
    with open('probe_operator_stratified.md', 'w', encoding='utf-8') as f:
        f.write(md)

    print(f"\n{'='*72}\n  Final report ({time.time()-t0:.1f}s):\n")
    print(md)


if __name__ == '__main__':
    main()
