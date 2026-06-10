"""
Probe D' — op-identity decodability at all upstream boundaries (Arith/Order/Set; §5.3).

probe_op_decomposition.py showed the Set hidden does not encode op identity
(Probe D = 20.25% ≈ 20% chance, macro-F1 20.10%); this extends the test to
Arith and Order.

Protocol (identical to probe_op_decomposition.py's Probe D, except boundary):
  - 5 checkpoints (seeds 42, 123, 256, 777, 999)
  - 50K samples per seed, data_seed=999, train/test 70/30, split_seed=0
  - LinearSVC + StandardScaler, 5-class op target
  - Train on FULL train; score on (i) full test and (ii) NU-stratum subset
  - Macro-F1 and per-op F1 reported

For each boundary in {Arith, Order, Set}:
  Probe D':  hidden 128-dim -> logic_op (5-class, OP_AND/OR/NOT/IMP/IFF)

Stop gate: any boundary D' > 25% (full eval) -> upstream encodes some op
identity -> the "operator undecodable upstream" narrative needs revision.

Note: probe_operator_stratified.forward_with_hidden returns only the Set
boundary, so a local all-boundaries variant is defined below to extract
Arith/Order/Set hidden in one pass.
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
from sklearn.metrics import f1_score

from _theia_model_def import (
    IsisV9, DEVICE,
    NUM_RANGE, SET_DIM, P_UNKNOWN,
    VAL_FALSE, VAL_TRUE, VAL_UNKNOWN,
    N_RELS, N_ARITH, N_OPS,
    OP_AND, OP_OR, OP_NOT, OP_IMPLIES, OP_IFF,
    apply_logic,
)
from probe_operator_stratified import (
    gen_data_extended,
    CKPTS, N_SAMPLES, DATA_SEED, TRAIN_FRAC, SPLIT_SEED,
)

BOUNDARIES = ['arith', 'order', 'set']
STOP_GATE_FULL_PCT = 25.0  # stop gate: any boundary D > 25% triggers stop
CHANCE_5CLASS_PCT = 20.0


def forward_all_boundaries(model, a, b, d, set_bits, s_unk, a_unk, b_unk, d_unk,
                            arith, rel, op):
    """Returns hidden vectors at all 3 upstream boundaries.
    Replicates IsisV9.forward computation (byte-identical to model's
    forward path)."""
    c_vec = model.arith_eng(a, b, a_unk, b_unk, arith)
    c_for_ord = model.bridge_ao(c_vec) + c_vec
    c_for_set = model.bridge_as(c_vec) + c_vec
    ord_vec = model.order_eng(c_for_ord, d, d_unk, rel)
    set_vec = model.set_eng(c_for_set, set_bits, s_unk)
    return {'arith': c_vec, 'order': ord_vec, 'set': set_vec}


def fit_predict_svc(X_tr, y_tr, X_te):
    sc = StandardScaler()
    Xs_tr = sc.fit_transform(X_tr)
    Xs_te = sc.transform(X_te)
    dual = X_tr.shape[0] < X_tr.shape[1]
    clf = LinearSVC(C=1.0, max_iter=2000, dual=dual)
    clf.fit(Xs_tr, y_tr)
    return clf.predict(Xs_te)


def run_seed(seed, ckpt_path):
    print(f"\n{'='*78}\n  seed {seed}  ckpt: {ckpt_path}\n{'='*78}")
    if not os.path.exists(ckpt_path):
        print(f"  [MISSING]"); return None

    model = IsisV9().to(DEVICE)
    state = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    data = gen_data_extended(DATA_SEED, n=N_SAMPLES)

    BATCH = 8192
    hidden = {b: [] for b in BOUNDARIES}
    with torch.no_grad():
        for i in range(0, N_SAMPLES, BATCH):
            j = min(i + BATCH, N_SAMPLES)
            with torch.amp.autocast('cuda'):
                h = forward_all_boundaries(
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
    op_np     = data['op'].cpu().numpy()

    # Match probe_op_decomposition.py split exactly
    n_train = int(N_SAMPLES * TRAIN_FRAC)
    perm = np.random.RandomState(SPLIT_SEED).permutation(N_SAMPLES)
    tr, te = perm[:n_train], perm[n_train:]
    op_tr, op_te = op_np[tr], op_np[te]
    te_NU_mask = (target_np[te] != VAL_UNKNOWN)

    out = {'seed': seed, 'per_boundary': {}}

    print(f"  {'boundary':<8} {'D_full':>10} {'D_NU':>10} {'macroF1':>10} "
          f"{'F1_AND':>9} {'F1_OR':>9} {'F1_NOT':>9} {'F1_IMP':>9} {'F1_IFF':>9}")
    for b in BOUNDARIES:
        H_tr, H_te = hidden[b][tr], hidden[b][te]
        t0 = time.time()
        pred = fit_predict_svc(H_tr, op_tr, H_te)
        elapsed = time.time() - t0

        acc_full = float((pred == op_te).mean()) * 100.0
        acc_NU   = float((pred[te_NU_mask] == op_te[te_NU_mask]).mean()) * 100.0
        f1_macro = f1_score(op_te, pred, average='macro') * 100.0
        f1_per   = f1_score(op_te, pred, average=None,
                            labels=[OP_AND, OP_OR, OP_NOT, OP_IMPLIES, OP_IFF]) * 100.0

        out['per_boundary'][b] = {
            'acc_full_test_pct': acc_full,
            'acc_NU_test_pct':   acc_NU,
            'macro_f1_pct':      f1_macro,
            'f1_per_op_pct': {
                'AND': float(f1_per[0]), 'OR':  float(f1_per[1]),
                'NOT': float(f1_per[2]), 'IMP': float(f1_per[3]),
                'IFF': float(f1_per[4]),
            },
            'fit_seconds': elapsed,
            'gate_triggered': acc_full > STOP_GATE_FULL_PCT,
        }

        gate_tag = ' ⚠️ GATE' if acc_full > STOP_GATE_FULL_PCT else ''
        print(f"  {b:<8} {acc_full:>9.2f}% {acc_NU:>9.2f}% {f1_macro:>9.2f}% "
              f"{f1_per[0]:>8.2f}% {f1_per[1]:>8.2f}% {f1_per[2]:>8.2f}% "
              f"{f1_per[3]:>8.2f}% {f1_per[4]:>8.2f}%{gate_tag}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def aggregate(per_seed):
    seeds_used = [r['seed'] for r in per_seed if r is not None]
    rows = [r for r in per_seed if r is not None]
    n = len(rows)

    agg = {'n_seeds': n, 'seeds_used': seeds_used, 'per_boundary': {}}
    for b in BOUNDARIES:
        full = [r['per_boundary'][b]['acc_full_test_pct'] for r in rows]
        NU   = [r['per_boundary'][b]['acc_NU_test_pct']   for r in rows]
        F1   = [r['per_boundary'][b]['macro_f1_pct']      for r in rows]
        per_op_f1 = {
            op: [r['per_boundary'][b]['f1_per_op_pct'][op] for r in rows]
            for op in ('AND', 'OR', 'NOT', 'IMP', 'IFF')
        }
        agg['per_boundary'][b] = {
            'acc_full_mean':  float(np.mean(full)),  'acc_full_std':  float(np.std(full)),
            'acc_NU_mean':    float(np.mean(NU)),    'acc_NU_std':    float(np.std(NU)),
            'macro_f1_mean':  float(np.mean(F1)),    'macro_f1_std':  float(np.std(F1)),
            'f1_per_op_mean': {op: float(np.mean(v)) for op, v in per_op_f1.items()},
            'f1_per_op_std':  {op: float(np.std(v))  for op, v in per_op_f1.items()},
            'per_seed_acc_full': full,
            'gate_triggered_aggregate': bool(np.mean(full) > STOP_GATE_FULL_PCT),
            'gate_triggered_any_seed':  bool(max(full) > STOP_GATE_FULL_PCT),
            'delta_full_minus_chance_pp': float(np.mean(full)) - CHANCE_5CLASS_PCT,
        }
    return agg


def render_markdown(per_seed, agg):
    L = []
    L.append("# Op-Identity Decodability at All Upstream Boundaries")
    L.append("")
    L.append(f"5 checkpoints {agg['seeds_used']} × 50K samples × LinearSVC + "
             f"StandardScaler, 5-class `logic_op` target.")
    L.append("Same data + split as probe_stratified / probe_op_decomposition (`data_seed=999`, `split_seed=0`, "
             f"70/30, train on FULL train).")
    L.append("")
    L.append(f"**Stop gate**: any boundary D > **{STOP_GATE_FULL_PCT}%** "
             f"on full eval → boundary encodes op identity → paper narrative "
             f"\"operator undecodable upstream\" needs revision.")
    L.append("")
    L.append("## Aggregate D' acc per boundary (mean ± std over 5 seeds)")
    L.append("")
    L.append("| Boundary | D' acc (full test) | D' acc (NU test) | macro-F1 | "
             f"Δ vs chance ({CHANCE_5CLASS_PCT:.0f}%) | Gate |")
    L.append("|---|---|---|---|---|---|")
    for b in BOUNDARIES:
        v = agg['per_boundary'][b]
        gate = "⚠️ TRIGGER" if v['gate_triggered_aggregate'] else "OK"
        L.append(f"| {b} | "
                 f"**{v['acc_full_mean']:.2f}% ± {v['acc_full_std']:.2f}%** | "
                 f"{v['acc_NU_mean']:.2f}% ± {v['acc_NU_std']:.2f}% | "
                 f"{v['macro_f1_mean']:.2f}% ± {v['macro_f1_std']:.2f}% | "
                 f"{v['delta_full_minus_chance_pp']:+.2f} pp | {gate} |")
    L.append("")
    L.append("## Per-seed full-test acc (any single seed > 25% also triggers gate)")
    L.append("")
    L.append("| Boundary | " + " | ".join(f"seed {s}" for s in agg['seeds_used']) + " | max |")
    L.append("|" + "---|" * (2 + len(agg['seeds_used'])))
    for b in BOUNDARIES:
        per = agg['per_boundary'][b]['per_seed_acc_full']
        max_v = max(per)
        gate_max = " ⚠️" if max_v > STOP_GATE_FULL_PCT else ""
        L.append(f"| {b} | " + " | ".join(f"{x:.2f}%" for x in per) +
                 f" | {max_v:.2f}%{gate_max} |")
    L.append("")
    L.append("## F1 per op per boundary (mean across 5 seeds)")
    L.append("")
    L.append("| Boundary | F1 AND | F1 OR | F1 NOT | F1 IMP | F1 IFF |")
    L.append("|---|---|---|---|---|---|")
    for b in BOUNDARIES:
        v = agg['per_boundary'][b]['f1_per_op_mean']
        L.append(f"| {b} | {v['AND']:.2f}% | {v['OR']:.2f}% | {v['NOT']:.2f}% | "
                 f"{v['IMP']:.2f}% | {v['IFF']:.2f}% |")
    L.append("")
    L.append("## Stop-gate evaluation")
    L.append("")
    triggered = [b for b in BOUNDARIES
                 if agg['per_boundary'][b]['gate_triggered_aggregate']
                 or agg['per_boundary'][b]['gate_triggered_any_seed']]
    if triggered:
        L.append(f"**STOP GATE TRIGGERED** at: {triggered}")
        L.append("")
        for b in triggered:
            v = agg['per_boundary'][b]
            L.append(f"- **{b}**: aggregate {v['acc_full_mean']:.2f}%, "
                     f"max-seed {max(v['per_seed_acc_full']):.2f}%")
    else:
        L.append("**No boundary triggers.** All upstream boundaries' op decodability "
                 f"is at or near chance ({CHANCE_5CLASS_PCT:.0f}%) within ±5pp.")
        L.append("")
        L.append("This completes the upstream-boundary evidence: **no Arith / Order / "
                 "Set boundary encodes the logic operator**, consistent with H2 / "
                 "delayed-verdict claim. The Logic Engine is the first place op "
                 "identity becomes decodable (the mechanistic-probe table `tab:mechprobe` already shows "
                 "Logic-boundary op-decoding ≥ 85%).")
    L.append("")
    return "\n".join(L) + "\n"


def main():
    t_start = time.time()
    per_seed = []
    for seed, ckpt in CKPTS:
        per_seed.append(run_seed(seed, ckpt))
    agg = aggregate(per_seed)

    out = {
        'protocol': {
            'description': 'Probe D extended to Arith / Order / Set boundaries; '
                           'matches the probe_op_decomposition Probe D protocol exactly',
            'n_samples': N_SAMPLES, 'data_seed': DATA_SEED,
            'train_frac': TRAIN_FRAC, 'split_seed': SPLIT_SEED,
            'probe': 'LinearSVC C=1.0 + StandardScaler, 5-class op target',
            'training_data': 'full 70% train (no NU filter)',
            'stop_gate_full_pct': STOP_GATE_FULL_PCT,
            'chance_5class_pct': CHANCE_5CLASS_PCT,
        },
        'per_seed': [r for r in per_seed if r is not None],
        'aggregate': agg,
    }
    with open('probe_op_identity_all_boundaries.json', 'w') as f:
        json.dump(out, f, indent=2)

    md = render_markdown(per_seed, agg)
    with open('probe_op_identity_all_boundaries.md', 'w', encoding='utf-8') as f:
        f.write(md)

    print(f"\n{'='*78}\n  Final report ({time.time()-t_start:.1f}s):\n")
    print(md)


if __name__ == '__main__':
    main()
