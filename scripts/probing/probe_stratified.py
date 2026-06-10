"""
Stratified linear probe: final-verdict accuracy by label stratum (§5.3 / Table 7).

The paper reports upstream final-verdict probe accuracy 60.9 / 67.2 / 69.7%
at Arith/Order/Set and attributes it to the 74% uncertainty-only ceiling
(0.41*1.0 + 0.59*0.559 = 0.74). If the probe were only reading Has-Unknown,
verdict accuracy on the non-Unknown subset should sit at the majority baseline
(55.9%); materially higher non-Unknown accuracy would mean upstream encodes
the T/F distinction beyond the uncertainty signal.

Protocol:
  - 5 checkpoints (seeds 42, 123, 256, 777, 999)
  - 50K samples (data_seed=999, identical to the paper mechprobe protocol)
  - 70/30 train/test split (split_seed=0, identical to paper)
  - 3 boundaries: Arith, Order, Set (Logic excluded -- that is where the
    verdict is computed)
  - LinearSVC + StandardScaler on full train, 3-class verdict target
  - Score on test, stratified by target label:
      stratum U:  target == VAL_UNKNOWN
      stratum NU: target != VAL_UNKNOWN  (= F or T)
  - Majority baseline on stratum NU: max class freq of (F vs T) in train-NU
    (paper: 0.559 for True)

Stop gate: any (boundary, NU) mean acc >= 60.0% (4+ pp above the 55.9%
baseline) weakens the delayed-verdict claim.

Outputs:
  probe_stratified.json
  probe_stratified.md
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
import torch.nn as nn
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler

from _theia_model_def import (
    IsisV9, DEVICE,
    NUM_RANGE, SET_DIM, P_UNKNOWN,
    VAL_FALSE, VAL_TRUE, VAL_UNKNOWN,
    N_RELS, N_ARITH, N_OPS,
    apply_logic,
)

# --- 5 checkpoint paths (seed 42 in legacy theia/, others in theia_v2/) ---
CKPTS = [
    (42,  'multi_seed_results/theia/seed_42/checkpoint.pth'),
    (123, 'multi_seed_results/theia_v2/seed_123/checkpoint.pth'),
    (256, 'multi_seed_results/theia_v2/seed_256/checkpoint.pth'),
    (777, 'multi_seed_results/theia_v2/seed_777/checkpoint.pth'),
    (999, 'multi_seed_results/theia_v2/seed_999/checkpoint.pth'),
]
N_SAMPLES = 50_000
DATA_SEED = 999             # matches paper Table mechprobe protocol
TRAIN_FRAC = 0.7
SPLIT_SEED = 0              # matches reprobe_theia_v2.py L290
PASS_THRESHOLD_NU = 60.0    # stop gate (percent)
BOUNDARIES = ['arith', 'order', 'set']  # Logic excluded -- trivial


def forward_with_hidden(model, a, b, d, set_bits, s_unk, a_unk, b_unk, d_unk,
                        arith, rel, op):
    """Replicates reprobe_theia_v2.py L143-150 (IsisV9.forward_with_hidden)
    by calling the engines directly via the imported model's submodules.
    Byte-identical computation to the original training/inference path."""
    c_vec = model.arith_eng(a, b, a_unk, b_unk, arith)
    c_for_ord = model.bridge_ao(c_vec) + c_vec
    c_for_set = model.bridge_as(c_vec) + c_vec
    ord_vec = model.order_eng(c_for_ord, d, d_unk, rel)
    set_vec = model.set_eng(c_for_set, set_bits, s_unk)
    logic_vec = model.logic_eng(ord_vec, set_vec, op)
    return {'arith': c_vec, 'order': ord_vec, 'set': set_vec, 'logic': logic_vec}


def gen_data(seed, n=N_SAMPLES):
    """Replicates reprobe_theia_v2.py L152-196 (gen_data) verbatim with
    the imported constants. Same RNG and target computation -> same labels
    used to build paper Table mechprobe."""
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
        'target': target,
    }


def probe_classification_with_strata(X_tr, y_tr, X_te, y_te, te_mask_U, te_mask_NU):
    """Replicates reprobe_theia_v2.py L205-211 protocol (LinearSVC +
    StandardScaler), adds stratified scoring on the test set.

    Returns dict with overall test acc, U-stratum acc, NU-stratum acc."""
    sc = StandardScaler()
    Xs_tr = sc.fit_transform(X_tr)
    Xs_te = sc.transform(X_te)
    dual = X_tr.shape[0] < X_tr.shape[1]
    clf = LinearSVC(C=1.0, max_iter=2000, dual=dual)
    clf.fit(Xs_tr, y_tr)
    preds = clf.predict(Xs_te)
    acc_overall = float((preds == y_te).mean())
    acc_U  = float((preds[te_mask_U]  == y_te[te_mask_U]).mean())  if te_mask_U.sum()  else float('nan')
    acc_NU = float((preds[te_mask_NU] == y_te[te_mask_NU]).mean()) if te_mask_NU.sum() else float('nan')
    return {
        'acc_overall': acc_overall,
        'acc_U':  acc_U,
        'acc_NU': acc_NU,
        'n_te_U':  int(te_mask_U.sum()),
        'n_te_NU': int(te_mask_NU.sum()),
    }


def majority_baseline_NU(y_tr, tr_mask_NU, y_te, te_mask_NU):
    """Majority class predictor restricted to NU stratum.
    Train: pick majority class among y_tr[tr_mask_NU] (F vs T).
    Test:  predict that class for every sample in te_mask_NU; report acc."""
    if tr_mask_NU.sum() == 0 or te_mask_NU.sum() == 0:
        return {'maj_class': None, 'maj_baseline_NU': float('nan'),
                'p_T_in_NU_train': float('nan')}
    y_tr_NU = y_tr[tr_mask_NU]
    n_T = int((y_tr_NU == VAL_TRUE).sum())
    n_F = int((y_tr_NU == VAL_FALSE).sum())
    maj = VAL_TRUE if n_T >= n_F else VAL_FALSE
    maj_acc = float((y_te[te_mask_NU] == maj).mean())
    return {
        'maj_class': 'T' if maj == VAL_TRUE else 'F',
        'maj_baseline_NU': maj_acc,
        'p_T_in_NU_train': float(n_T / (n_T + n_F)),
    }


def run_seed(seed, ckpt_path):
    print(f"\n{'='*66}\n  seed {seed}  ckpt: {ckpt_path}\n{'='*66}")
    if not os.path.exists(ckpt_path):
        print(f"  [MISSING] {ckpt_path}")
        return None

    model = IsisV9().to(DEVICE)
    state = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    params = sum(p.numel() for p in model.parameters())
    print(f"  params: {params:,}")
    assert params == 2_751_232

    print(f"  generating {N_SAMPLES} samples (data_seed={DATA_SEED}) ...")
    data = gen_data(DATA_SEED, n=N_SAMPLES)

    print(f"  forward_with_hidden ...")
    BATCH = 8192
    hidden = {b: [] for b in BOUNDARIES}
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
            for b in BOUNDARIES:
                hidden[b].append(h[b].float().cpu().numpy())
    hidden = {b: np.concatenate(v, axis=0) for b, v in hidden.items()}
    target_np = data['target'].cpu().numpy()
    print(f"  hidden shapes: {{ {', '.join(f'{b}: {hidden[b].shape}' for b in BOUNDARIES)} }}")

    # Train/test split (matches reprobe_theia_v2.py L289-291)
    n_train = int(N_SAMPLES * TRAIN_FRAC)
    perm = np.random.RandomState(SPLIT_SEED).permutation(N_SAMPLES)
    tr, te = perm[:n_train], perm[n_train:]

    y_tr, y_te = target_np[tr], target_np[te]
    tr_mask_U  = (y_tr == VAL_UNKNOWN)
    tr_mask_NU = ~tr_mask_U
    te_mask_U  = (y_te == VAL_UNKNOWN)
    te_mask_NU = ~te_mask_U

    # Sanity: stratum sizes + paper-stated baselines
    p_unk_data = float((target_np == VAL_UNKNOWN).mean())
    n_T_NU_tr = int((y_tr[tr_mask_NU] == VAL_TRUE).sum())
    n_F_NU_tr = int((y_tr[tr_mask_NU] == VAL_FALSE).sum())
    p_T_in_NU_tr = float(n_T_NU_tr / (n_T_NU_tr + n_F_NU_tr))
    print(f"  P(label=U) full = {p_unk_data:.4f}  (paper: 0.41)")
    print(f"  P(T | non-U) train = {p_T_in_NU_tr:.4f}  (paper: 0.559)")

    maj = majority_baseline_NU(y_tr, tr_mask_NU, y_te, te_mask_NU)
    print(f"  majority class on NU train = {maj['maj_class']}, "
          f"baseline acc on NU test = {maj['maj_baseline_NU']*100:.2f}%")

    out_per_seed = {
        'seed': seed,
        'checkpoint': ckpt_path,
        'params': params,
        'n_samples': N_SAMPLES,
        'data_seed': DATA_SEED,
        'split_seed': SPLIT_SEED,
        'p_unk_data': p_unk_data,
        'p_T_in_NU_train': p_T_in_NU_tr,
        'maj_baseline_NU_pct': maj['maj_baseline_NU'] * 100.0,
        'maj_class_NU': maj['maj_class'],
        'per_boundary': {},
    }

    print(f"  {'boundary':<8} {'overall':>9} {'U-strat':>9} {'NU-strat':>9} "
          f"{'NU - maj':>9}")
    for b in BOUNDARIES:
        X = hidden[b]
        X_tr, X_te = X[tr], X[te]
        t0 = time.time()
        r = probe_classification_with_strata(X_tr, y_tr, X_te, y_te,
                                             te_mask_U, te_mask_NU)
        dt = time.time() - t0
        delta_NU = (r['acc_NU'] - maj['maj_baseline_NU']) * 100.0
        out_per_seed['per_boundary'][b] = {
            'acc_overall_pct': r['acc_overall'] * 100.0,
            'acc_U_pct':       r['acc_U']  * 100.0,
            'acc_NU_pct':      r['acc_NU'] * 100.0,
            'NU_delta_vs_majority_pp': delta_NU,
            'n_te_U':  r['n_te_U'],
            'n_te_NU': r['n_te_NU'],
            'fit_seconds': dt,
        }
        print(f"  {b:<8} {r['acc_overall']*100:>8.2f}% "
              f"{r['acc_U']*100:>8.2f}% {r['acc_NU']*100:>8.2f}% "
              f"{delta_NU:>+7.2f}pp")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out_per_seed


def aggregate(per_seed):
    """Aggregate per-seed results into mean ± std per (boundary, stratum)."""
    seeds = [r for r in per_seed if r is not None]
    n = len(seeds)
    agg = {
        'n_seeds': n,
        'seeds_used': [r['seed'] for r in seeds],
        'p_unk_data_mean':       float(np.mean([r['p_unk_data'] for r in seeds])),
        'p_T_in_NU_train_mean':  float(np.mean([r['p_T_in_NU_train'] for r in seeds])),
        'maj_baseline_NU_pct_mean': float(np.mean([r['maj_baseline_NU_pct'] for r in seeds])),
        'maj_baseline_NU_pct_std':  float(np.std([r['maj_baseline_NU_pct'] for r in seeds])),
        'per_boundary': {},
    }
    for b in BOUNDARIES:
        ovs  = [r['per_boundary'][b]['acc_overall_pct'] for r in seeds]
        Us   = [r['per_boundary'][b]['acc_U_pct']      for r in seeds]
        NUs  = [r['per_boundary'][b]['acc_NU_pct']     for r in seeds]
        deltas = [r['per_boundary'][b]['NU_delta_vs_majority_pp'] for r in seeds]
        agg['per_boundary'][b] = {
            'overall_mean': float(np.mean(ovs)),  'overall_std': float(np.std(ovs)),
            'U_mean':       float(np.mean(Us)),   'U_std':       float(np.std(Us)),
            'NU_mean':      float(np.mean(NUs)),  'NU_std':      float(np.std(NUs)),
            'NU_delta_vs_majority_mean_pp': float(np.mean(deltas)),
            'NU_delta_vs_majority_std_pp':  float(np.std(deltas)),
            'per_seed_NU': [float(x) for x in NUs],
            'NU_max_across_seeds': float(max(NUs)),
        }
    return agg


def render_markdown(per_seed, agg):
    lines = []
    lines.append("# Stratified Linear Probe — Final Verdict by Label Stratum")
    lines.append("")
    lines.append(f"- 5 checkpoints (seeds {agg['seeds_used']}), {N_SAMPLES} samples each, "
                 f"data_seed={DATA_SEED}, train/test split 70/30 (split_seed={SPLIT_SEED})")
    lines.append(f"- Probe: LinearSVC + StandardScaler, 3-class verdict target, "
                 f"identical to `reprobe_theia_v2.py` protocol")
    lines.append(f"- Stratification: by **label** (target == U vs target != U)")
    lines.append("")
    lines.append("## Empirical baselines (averaged over 5 seeds)")
    lines.append("")
    lines.append(f"- P(label = U) in full data = **{agg['p_unk_data_mean']*100:.2f}%** "
                 f"(paper claim: ~41%)")
    lines.append(f"- P(label = T | label ∈ {{F,T}}) in train data = "
                 f"**{agg['p_T_in_NU_train_mean']*100:.2f}%** "
                 f"(paper claim: 55.9%)")
    lines.append(f"- Majority baseline on NU test (predict majority class always): "
                 f"**{agg['maj_baseline_NU_pct_mean']:.2f}% ± "
                 f"{agg['maj_baseline_NU_pct_std']:.2f}%**")
    lines.append("")
    lines.append("## Per-boundary linear-probe accuracy (mean ± std over 5 seeds)")
    lines.append("")
    lines.append("| Boundary | Overall | U-stratum | non-U stratum | NU − majority (pp) | NU max (single seed) |")
    lines.append("|---|---|---|---|---|---|")
    for b in BOUNDARIES:
        v = agg['per_boundary'][b]
        gate = " ⚠️" if v['NU_mean'] >= PASS_THRESHOLD_NU else ""
        lines.append(f"| {b} | {v['overall_mean']:.2f}% ± {v['overall_std']:.2f}% "
                     f"| {v['U_mean']:.2f}% ± {v['U_std']:.2f}% "
                     f"| **{v['NU_mean']:.2f}% ± {v['NU_std']:.2f}%**{gate} "
                     f"| {v['NU_delta_vs_majority_mean_pp']:+.2f} ± "
                     f"{v['NU_delta_vs_majority_std_pp']:.2f} "
                     f"| {v['NU_max_across_seeds']:.2f}% |")
    lines.append("")
    lines.append("## Per-seed non-Unknown stratum accuracy")
    lines.append("")
    lines.append("| Boundary | " + " | ".join(f"seed {s}" for s in agg['seeds_used']) + " |")
    lines.append("|" + "---|" * (1 + len(agg['seeds_used'])))
    for b in BOUNDARIES:
        per_seed_NU = agg['per_boundary'][b]['per_seed_NU']
        lines.append(f"| {b} | " + " | ".join(f"{x:.2f}%" for x in per_seed_NU) + " |")
    lines.append("")
    lines.append("## Stop gate evaluation")
    lines.append("")
    lines.append(f"Threshold: NU-stratum mean acc ≥ **{PASS_THRESHOLD_NU:.1f}%** on any "
                 f"of {{Arith, Order, Set}} → delayed-verdict claim hole.")
    triggered = []
    for b in BOUNDARIES:
        v = agg['per_boundary'][b]
        if v['NU_mean'] >= PASS_THRESHOLD_NU:
            triggered.append((b, v['NU_mean']))
    if triggered:
        lines.append("")
        lines.append("**STOP GATE TRIGGERED**:")
        for b, val in triggered:
            lines.append(f"- {b}: NU mean = {val:.2f}% ≥ {PASS_THRESHOLD_NU:.1f}%")
        lines.append("")
        lines.append("Upstream representations encode T/F distinction beyond the "
                     "uncertainty signal. Delayed-verdict claim needs revision.")
    else:
        lines.append("")
        lines.append("**No boundary triggers.** All NU-stratum mean accuracies are below "
                     f"{PASS_THRESHOLD_NU:.1f}%, consistent with the delayed-verdict claim "
                     "(upstream representations cannot decode T/F beyond the "
                     "uncertainty-only ceiling on the non-Unknown subset).")
    return "\n".join(lines) + "\n"


def main():
    t_start = time.time()
    per_seed = []
    for seed, ckpt in CKPTS:
        per_seed.append(run_seed(seed, ckpt))

    agg = aggregate(per_seed)

    out = {
        'protocol': {
            'n_samples': N_SAMPLES,
            'data_seed': DATA_SEED,
            'train_frac': TRAIN_FRAC,
            'split_seed': SPLIT_SEED,
            'pass_threshold_NU_pct': PASS_THRESHOLD_NU,
            'boundaries_tested': BOUNDARIES,
            'stratification': 'by_label_target_eq_VAL_UNKNOWN',
            'probe': 'LinearSVC C=1.0 max_iter=2000 + StandardScaler',
            'reuses': '_theia_model_def.IsisV9 (byte-identical to theia_5seed_v2.py)',
        },
        'per_seed': [r for r in per_seed if r is not None],
        'aggregate': agg,
    }
    with open('probe_stratified.json', 'w') as f:
        json.dump(out, f, indent=2)

    md = render_markdown(per_seed, agg)
    with open('probe_stratified.md', 'w', encoding='utf-8') as f:
        f.write(md)

    print(f"\n{'='*66}\nFinal report ({time.time()-t_start:.1f}s):")
    print(md)


if __name__ == '__main__':
    main()
