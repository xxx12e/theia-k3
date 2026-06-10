"""
Op-identity decomposition of the Set-boundary NU-stratum lift (Probe A/B/C/D; §5.3).

Context (mixed ops): NU-stratum SVC acc on the Set boundary = 60.13%,
+4.09 pp above the NU verdict majority (56.04%). The lift could come from
(i) Set hidden encoding val_set, (ii) Set hidden encoding op identity, or
(iii) verdict leakage proper; four probes decompose it.

Protocol (matches probe_stratified.py exactly):
  - 5 checkpoints (seeds 42, 123, 256, 777, 999)
  - 50K samples per seed, data_seed=999, train/test 70/30, split_seed=0
  - LinearSVC + StandardScaler, 3-class verdict target
  - Train on FULL train (3-class); evaluate on NU-stratum subset of test
    (test labels in {F, T}). All 4 probes use this convention so accuracies
    are directly comparable to probe_stratified.py.

Probes:
  A: Set hidden 128-dim          -> verdict (3-class)  [baseline; expect ~60.13%]
  B: op one-hot 5-dim            -> verdict (3-class)  [op alone]
  C: [Set hidden, op one-hot] 133-dim -> verdict       [Set hidden + op]
  D: Set hidden 128-dim          -> op (5-class)       [direct test of op decodability]

Plus per-op Probe-C breakdown vs the val_set-only ceiling: Probe C has explicit
access to op identity, so "Probe C - val_set ceiling" measures Set hidden's
additional contribution beyond (op identity + val_set) within each op.
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
    forward_with_hidden, gen_data_extended,
    val_set_only_ceiling,
    CKPTS, N_SAMPLES, DATA_SEED, TRAIN_FRAC, SPLIT_SEED,
    OPS,
)

OP_DIM = N_OPS  # 5 (AND, OR, NOT, IMPLIES, IFF)
SET_DIM_HIDDEN = 128  # IsisV9 hidden dim


def onehot_op(op_int_array, n_ops=N_OPS):
    """One-hot encode an integer op array of shape (n,) into (n, n_ops)."""
    n = len(op_int_array)
    x = np.zeros((n, n_ops), dtype=np.float32)
    x[np.arange(n), op_int_array.astype(int)] = 1.0
    return x


def fit_predict_svc(X_tr, y_tr, X_te):
    """Fit LinearSVC + StandardScaler, return predictions on X_te.
    Matches the probe_stratified.py / probe_operator_stratified.py protocol verbatim."""
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

    # Match probe_stratified.py split exactly
    n_train = int(N_SAMPLES * TRAIN_FRAC)
    perm = np.random.RandomState(SPLIT_SEED).permutation(N_SAMPLES)
    tr, te = perm[:n_train], perm[n_train:]

    set_tr, set_te = set_hidden[tr], set_hidden[te]
    op_tr,  op_te  = op_np[tr],     op_np[te]
    y_tr,   y_te   = target_np[tr], target_np[te]
    vs_tr,  vs_te  = val_set_np[tr], val_set_np[te]

    op_tr_oh = onehot_op(op_tr)
    op_te_oh = onehot_op(op_te)

    # NU mask on test
    te_NU_mask = (y_te != VAL_UNKNOWN)
    n_te_NU = int(te_NU_mask.sum())
    nu_majority_T_pct = float((y_te[te_NU_mask] == VAL_TRUE).mean()) * 100.0
    nu_majority_F_pct = float((y_te[te_NU_mask] == VAL_FALSE).mean()) * 100.0
    nu_majority_pct   = max(nu_majority_T_pct, nu_majority_F_pct)
    nu_majority_class = 'T' if nu_majority_T_pct >= nu_majority_F_pct else 'F'

    print(f"  n_te_NU = {n_te_NU}, NU majority = {nu_majority_class} "
          f"({nu_majority_pct:.2f}%)")

    out = {'seed': seed, 'n_te_NU': n_te_NU,
           'nu_majority_pct': nu_majority_pct,
           'nu_majority_class': nu_majority_class}

    # ===== Probe A: Set hidden -> verdict =====
    t0 = time.time()
    pred_A = fit_predict_svc(set_tr, y_tr, set_te)
    A_acc_NU = float((pred_A[te_NU_mask] == y_te[te_NU_mask]).mean()) * 100.0
    out['A_set_hidden__verdict_NU_pct'] = A_acc_NU

    # ===== Probe B: op one-hot -> verdict =====
    pred_B = fit_predict_svc(op_tr_oh, y_tr, op_te_oh)
    B_acc_NU = float((pred_B[te_NU_mask] == y_te[te_NU_mask]).mean()) * 100.0
    out['B_op_only__verdict_NU_pct'] = B_acc_NU
    # Diagnostic: for each op, what does B predict?
    B_preds_per_op = {}
    for op_name, op_idx in OPS:
        m = (op_te == op_idx) & te_NU_mask
        if m.sum() > 0:
            B_preds_per_op[op_name] = {
                'pred_distribution': {
                    'F': float((pred_B[m] == VAL_FALSE).mean()),
                    'T': float((pred_B[m] == VAL_TRUE).mean()),
                    'U': float((pred_B[m] == VAL_UNKNOWN).mean()),
                },
                'acc_pct': float((pred_B[m] == y_te[m]).mean()) * 100.0,
                'n': int(m.sum()),
            }
    out['B_per_op'] = B_preds_per_op

    # ===== Probe C: [Set hidden, op] -> verdict =====
    X_tr_C = np.concatenate([set_tr, op_tr_oh], axis=1)
    X_te_C = np.concatenate([set_te, op_te_oh], axis=1)
    pred_C = fit_predict_svc(X_tr_C, y_tr, X_te_C)
    C_acc_NU = float((pred_C[te_NU_mask] == y_te[te_NU_mask]).mean()) * 100.0
    out['C_set_plus_op__verdict_NU_pct'] = C_acc_NU

    # ===== Probe D: Set hidden -> op =====
    pred_D = fit_predict_svc(set_tr, op_tr, set_te)
    D_acc = float((pred_D == op_te).mean()) * 100.0
    D_acc_NU = float((pred_D[te_NU_mask] == op_te[te_NU_mask]).mean()) * 100.0
    f1_macro = f1_score(op_te, pred_D, average='macro') * 100.0
    f1_per = f1_score(op_te, pred_D, average=None,
                      labels=[OP_AND, OP_OR, OP_NOT, OP_IMPLIES, OP_IFF]) * 100.0
    out['D_set_hidden__op'] = {
        'acc_full_test_pct':   D_acc,
        'acc_NU_test_pct':     D_acc_NU,
        'f1_macro_pct':        f1_macro,
        'f1_per_op_pct': {
            'AND': float(f1_per[0]), 'OR':  float(f1_per[1]),
            'NOT': float(f1_per[2]), 'IMP': float(f1_per[3]),
            'IFF': float(f1_per[4]),
        },
        'chance_5class_pct': 20.0,
    }

    # ===== Per-op Probe-C breakdown vs val_set-only ceiling =====
    out['per_op_C_vs_ceiling'] = {}
    for op_name, op_idx in OPS:
        # Restrict NU test set to this op
        m = (op_te == op_idx) & te_NU_mask
        if m.sum() < 50:
            continue
        C_acc_op = float((pred_C[m] == y_te[m]).mean()) * 100.0
        # Recompute val_set-only ceiling on this op's NU subset
        # using the SAME train/test split as Probe C (so comparable)
        m_tr = (op_tr == op_idx) & (y_tr != VAL_UNKNOWN)
        m_te = (op_te == op_idx) & (y_te != VAL_UNKNOWN)
        if m_tr.sum() < 50 or m_te.sum() < 50:
            continue
        ceil_acc, _ = val_set_only_ceiling(vs_tr[m_tr], y_tr[m_tr],
                                            vs_te[m_te], y_te[m_te])
        ceil_pct = ceil_acc * 100.0
        # Also: per-op verdict majority
        n_F_op = int((y_te[m_te] == VAL_FALSE).sum())
        n_T_op = int((y_te[m_te] == VAL_TRUE).sum())
        op_majority_pct = max(n_F_op, n_T_op) / (n_F_op + n_T_op) * 100.0

        out['per_op_C_vs_ceiling'][op_name] = {
            'C_acc_pct': C_acc_op,
            'val_set_ceiling_pct': ceil_pct,
            'op_majority_pct': op_majority_pct,
            'C_minus_ceiling_pp': C_acc_op - ceil_pct,
            'C_minus_majority_pp': C_acc_op - op_majority_pct,
            'n_te': int(m_te.sum()),
        }

    elapsed = time.time() - t0
    print(f"  Probe A (Set hid -> verdict, NU eval):    {A_acc_NU:6.2f}%  "
          f"[baseline stratified probe reported 60.13%]")
    print(f"  Probe B (op only -> verdict, NU eval):    {B_acc_NU:6.2f}%")
    print(f"  Probe C (Set+op -> verdict, NU eval):     {C_acc_NU:6.2f}%")
    print(f"  Probe D (Set hid -> op, full eval):       {D_acc:6.2f}%  "
          f"[chance 20%]")
    print(f"  Probe D (Set hid -> op, NU eval):         {D_acc_NU:6.2f}%  "
          f"[chance 20%]  macro-F1 {f1_macro:.2f}%")
    print(f"  C - B = {C_acc_NU - B_acc_NU:+.2f}pp   "
          f"A - B = {A_acc_NU - B_acc_NU:+.2f}pp   "
          f"({elapsed:.1f}s)")

    print(f"  Per-op Probe C vs val_set-only ceiling (NU subset):")
    print(f"    {'op':<5} {'n_te':>5} {'C':>8} {'ceil':>8} {'maj':>8} "
          f"{'C-ceil':>9} {'C-maj':>9}")
    for op_name, _ in OPS:
        if op_name not in out['per_op_C_vs_ceiling']: continue
        v = out['per_op_C_vs_ceiling'][op_name]
        print(f"    {op_name:<5} {v['n_te']:>5} {v['C_acc_pct']:>7.2f}% "
              f"{v['val_set_ceiling_pct']:>7.2f}% {v['op_majority_pct']:>7.2f}% "
              f"{v['C_minus_ceiling_pp']:>+8.2f}pp {v['C_minus_majority_pp']:>+8.2f}pp")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def aggregate(per_seed):
    seeds_used = [r['seed'] for r in per_seed if r is not None]
    rows = [r for r in per_seed if r is not None]
    n = len(rows)

    def mstd(key_or_fn):
        if callable(key_or_fn):
            vals = [key_or_fn(r) for r in rows]
        else:
            vals = [r[key_or_fn] for r in rows]
        return float(np.mean(vals)), float(np.std(vals))

    A_m, A_s = mstd('A_set_hidden__verdict_NU_pct')
    B_m, B_s = mstd('B_op_only__verdict_NU_pct')
    C_m, C_s = mstd('C_set_plus_op__verdict_NU_pct')
    D_m, D_s = mstd(lambda r: r['D_set_hidden__op']['acc_full_test_pct'])
    D_NU_m, D_NU_s = mstd(lambda r: r['D_set_hidden__op']['acc_NU_test_pct'])
    D_F1_m, D_F1_s = mstd(lambda r: r['D_set_hidden__op']['f1_macro_pct'])
    NUmaj_m, NUmaj_s = mstd('nu_majority_pct')

    agg = {
        'n_seeds': n,
        'seeds_used': seeds_used,
        'A_set_hidden__verdict_NU_pct':       {'mean': A_m,    'std': A_s},
        'B_op_only__verdict_NU_pct':          {'mean': B_m,    'std': B_s},
        'C_set_plus_op__verdict_NU_pct':      {'mean': C_m,    'std': C_s},
        'D_set_hidden__op_acc_full_pct':      {'mean': D_m,    'std': D_s},
        'D_set_hidden__op_acc_NU_pct':        {'mean': D_NU_m, 'std': D_NU_s},
        'D_set_hidden__op_macro_f1_pct':      {'mean': D_F1_m, 'std': D_F1_s},
        'nu_majority_pct':                    {'mean': NUmaj_m,'std': NUmaj_s},
        'delta_A_minus_B_pp':                 A_m - B_m,
        'delta_C_minus_B_pp':                 C_m - B_m,
        'delta_A_minus_majority_pp':          A_m - NUmaj_m,
        'delta_C_minus_majority_pp':          C_m - NUmaj_m,
    }

    # Per-op aggregate of C vs ceiling
    agg['per_op_C_vs_ceiling'] = {}
    for op_name, _ in OPS:
        Cs   = [r['per_op_C_vs_ceiling'][op_name]['C_acc_pct']
                for r in rows if op_name in r['per_op_C_vs_ceiling']]
        Cls  = [r['per_op_C_vs_ceiling'][op_name]['val_set_ceiling_pct']
                for r in rows if op_name in r['per_op_C_vs_ceiling']]
        Maj  = [r['per_op_C_vs_ceiling'][op_name]['op_majority_pct']
                for r in rows if op_name in r['per_op_C_vs_ceiling']]
        if not Cs:
            continue
        deltas_ceil = [c-cl for c, cl in zip(Cs, Cls)]
        deltas_maj  = [c-mj for c, mj in zip(Cs, Maj)]
        agg['per_op_C_vs_ceiling'][op_name] = {
            'C_mean': float(np.mean(Cs)),     'C_std': float(np.std(Cs)),
            'ceiling_mean': float(np.mean(Cls)), 'ceiling_std': float(np.std(Cls)),
            'majority_mean': float(np.mean(Maj)), 'majority_std': float(np.std(Maj)),
            'C_minus_ceiling_mean_pp': float(np.mean(deltas_ceil)),
            'C_minus_ceiling_std_pp':  float(np.std(deltas_ceil)),
            'C_minus_majority_mean_pp': float(np.mean(deltas_maj)),
            'C_minus_majority_std_pp':  float(np.std(deltas_maj)),
        }

    # Per-op B prediction distribution aggregate
    agg['per_op_B_pred_distribution'] = {}
    for op_name, _ in OPS:
        rows_op = [r['B_per_op'][op_name] for r in rows
                   if op_name in r.get('B_per_op', {})]
        if not rows_op:
            continue
        agg['per_op_B_pred_distribution'][op_name] = {
            'pred_F_mean': float(np.mean([r['pred_distribution']['F'] for r in rows_op])),
            'pred_T_mean': float(np.mean([r['pred_distribution']['T'] for r in rows_op])),
            'pred_U_mean': float(np.mean([r['pred_distribution']['U'] for r in rows_op])),
            'B_acc_mean':  float(np.mean([r['acc_pct'] for r in rows_op])),
        }

    # D F1 per op
    agg['D_f1_per_op_mean'] = {
        op_name: float(np.mean([r['D_set_hidden__op']['f1_per_op_pct'][op_name]
                                for r in rows]))
        for op_name in ('AND', 'OR', 'NOT', 'IMP', 'IFF')
    }

    return agg


def render_markdown(per_seed, agg):
    L = []
    L.append("# Op-identity Decomposition of Set Boundary +4pp Leakage")
    L.append("")
    L.append(f"5 checkpoints {agg['seeds_used']} × 50K samples × LinearSVC + StandardScaler.")
    L.append(f"Train on FULL 70% (3-class verdict for A/B/C, 5-class op for D); "
             f"NU-stratum eval = test samples with `target ∈ {{F,T}}`.")
    L.append("")
    L.append("## Aggregate (mean ± std over 5 seeds)")
    L.append("")
    L.append(f"- NU verdict majority on test set = **{agg['nu_majority_pct']['mean']:.2f}% ± "
             f"{agg['nu_majority_pct']['std']:.2f}%** (matches baseline 56.04%)")
    L.append("")
    L.append("| Probe | Features | Target | NU-eval acc | Δ vs majority |")
    L.append("|---|---|---|---|---|")
    L.append(f"| **A** | Set hidden 128-dim | verdict 3-class | "
             f"**{agg['A_set_hidden__verdict_NU_pct']['mean']:.2f}% ± "
             f"{agg['A_set_hidden__verdict_NU_pct']['std']:.2f}%** | "
             f"{agg['delta_A_minus_majority_pp']:+.2f} pp |")
    L.append(f"| **B** | op one-hot 5-dim | verdict 3-class | "
             f"**{agg['B_op_only__verdict_NU_pct']['mean']:.2f}% ± "
             f"{agg['B_op_only__verdict_NU_pct']['std']:.2f}%** | "
             f"{agg['B_op_only__verdict_NU_pct']['mean'] - agg['nu_majority_pct']['mean']:+.2f} pp |")
    L.append(f"| **C** | Set hidden + op (133-dim) | verdict 3-class | "
             f"**{agg['C_set_plus_op__verdict_NU_pct']['mean']:.2f}% ± "
             f"{agg['C_set_plus_op__verdict_NU_pct']['std']:.2f}%** | "
             f"{agg['delta_C_minus_majority_pp']:+.2f} pp |")
    L.append(f"| **D** | Set hidden 128-dim | op 5-class | "
             f"**{agg['D_set_hidden__op_acc_full_pct']['mean']:.2f}% ± "
             f"{agg['D_set_hidden__op_acc_full_pct']['std']:.2f}%** "
             f"(full); NU-only {agg['D_set_hidden__op_acc_NU_pct']['mean']:.2f}% | "
             f"chance 20% |")
    L.append("")
    L.append(f"**Key deltas**:")
    L.append(f"- C − B = **{agg['delta_C_minus_B_pp']:+.2f} pp** (Set hidden's contribution beyond op identity)")
    L.append(f"- A − B = **{agg['delta_A_minus_B_pp']:+.2f} pp** (Set hidden alone vs op alone)")
    L.append(f"- A − majority = **{agg['delta_A_minus_majority_pp']:+.2f} pp** (the baseline probe's reported gap)")
    L.append(f"- D macro-F1 = **{agg['D_set_hidden__op_macro_f1_pct']['mean']:.2f}% ± "
             f"{agg['D_set_hidden__op_macro_f1_pct']['std']:.2f}%** (op decodability from Set hidden)")
    L.append("")
    L.append("## D macro-F1 per op (op decodability from Set hidden)")
    L.append("")
    L.append("| Op | F1 (mean across seeds) |")
    L.append("|---|---|")
    for op_name in ('AND', 'OR', 'NOT', 'IMP', 'IFF'):
        L.append(f"| {op_name} | {agg['D_f1_per_op_mean'][op_name]:.2f}% |")
    L.append("")
    L.append("## Probe B's predictions per op (NU eval)")
    L.append("")
    L.append("Diagnostic: how does the op-only probe distribute its predictions?")
    L.append("")
    L.append("| Op | pred F | pred T | pred U | B acc on op-NU |")
    L.append("|---|---|---|---|---|")
    for op_name, _ in OPS:
        if op_name not in agg['per_op_B_pred_distribution']: continue
        v = agg['per_op_B_pred_distribution'][op_name]
        L.append(f"| {op_name} | {v['pred_F_mean']*100:.1f}% | {v['pred_T_mean']*100:.1f}% | "
                 f"{v['pred_U_mean']*100:.1f}% | {v['B_acc_mean']:.2f}% |")
    L.append("")
    L.append("## Per-op Probe C vs val_set-only ceiling (NU subset)")
    L.append("")
    L.append("Now that Probe C has explicit access to op identity, the comparison "
             "Probe C vs val_set-only ceiling is **non-degenerate**: any positive Δ "
             "indicates Set hidden carries *additional* verdict-relevant signal "
             "beyond (op identity + val_set).")
    L.append("")
    L.append("| Op | Probe C | val_set-only ceiling | op verdict majority | C − ceiling | C − majority |")
    L.append("|---|---|---|---|---|---|")
    for op_name, _ in OPS:
        if op_name not in agg['per_op_C_vs_ceiling']: continue
        v = agg['per_op_C_vs_ceiling'][op_name]
        L.append(f"| {op_name} | {v['C_mean']:.2f}% ± {v['C_std']:.2f}% | "
                 f"{v['ceiling_mean']:.2f}% ± {v['ceiling_std']:.2f}% | "
                 f"{v['majority_mean']:.2f}% ± {v['majority_std']:.2f}% | "
                 f"{v['C_minus_ceiling_mean_pp']:+.2f} ± {v['C_minus_ceiling_std_pp']:.2f} pp | "
                 f"{v['C_minus_majority_mean_pp']:+.2f} ± {v['C_minus_majority_std_pp']:.2f} pp |")
    L.append("")
    L.append("## Decision table")
    L.append("")
    A = agg['A_set_hidden__verdict_NU_pct']['mean']
    B = agg['B_op_only__verdict_NU_pct']['mean']
    C = agg['C_set_plus_op__verdict_NU_pct']['mean']
    D = agg['D_set_hidden__op_acc_full_pct']['mean']
    M = agg['nu_majority_pct']['mean']

    L.append(f"| | A | B | C | D | NU majority |")
    L.append(f"|---|---|---|---|---|---|")
    L.append(f"| Observed | {A:.2f}% | {B:.2f}% | {C:.2f}% | {D:.2f}% | {M:.2f}% |")
    L.append("")
    L.append("**Reading the data**:")
    L.append("")
    # B vs A vs majority diagnostics
    if B > M + 2:
        if abs(A - B) <= 2:
            L.append("- B ≫ majority and A ≈ B: **op identity alone** explains most of "
                     "the +4pp lift in the baseline stratified probe. Set hidden's marginal contribution "
                     "(C − B) measures any additional verdict signal beyond op identity.")
        elif A > B + 2:
            L.append(f"- B ≫ majority but A > B: op identity is part of the story but "
                     f"Set hidden adds additional information (A − B = {A-B:+.2f} pp).")
        else:
            L.append("- B ≫ majority but A < B: op identity alone is *more* useful than "
                     "Set hidden — Set hidden actively loses info or has noise.")
    else:
        L.append(f"- B ≈ majority ({B:.2f}% vs {M:.2f}%): **op identity alone does NOT "
                 f"explain the lift**; the +{A-M:.2f}pp lift in Probe A comes from Set "
                 f"hidden carrying signal that op identity does not.")

    if D > 50:
        L.append(f"- D = {D:.2f}% ≫ 20% chance: Set hidden DOES encode op identity "
                 f"(macro-F1 {agg['D_set_hidden__op_macro_f1_pct']['mean']:.2f}%).")
    elif D > 30:
        L.append(f"- D = {D:.2f}% > 20% chance: Set hidden encodes some op info but "
                 f"not perfectly.")
    else:
        L.append(f"- D = {D:.2f}% ≈ 20% chance: Set hidden does NOT encode op identity.")

    L.append(f"- C − B = {C-B:+.2f} pp: marginal value of Set hidden given op identity.")
    L.append(f"- C − majority = {C-M:+.2f} pp: total verdict-relevant signal in (Set hidden + op).")
    L.append("")
    L.append("**Per-op Probe-C vs val_set ceiling**: if C ≈ ceiling within each op, "
             "Set hidden's verdict-relevant content is ≈ val_set encoding (consistent "
             "with delayed-verdict for the val_set→verdict component). If C > ceiling, "
             "Set hidden carries verdict signal beyond what val_set provides.")
    L.append("")
    return "\n".join(L) + "\n"


def main():
    t0 = time.time()
    per_seed = []
    for seed, ckpt in CKPTS:
        per_seed.append(run_seed(seed, ckpt))
    agg = aggregate(per_seed)

    out = {
        'protocol': {
            'description': 'baseline stratified-probe protocol exactly: train on full 70% (3-class verdict), evaluate on NU stratum of test',
            'n_samples': N_SAMPLES, 'data_seed': DATA_SEED,
            'train_frac': TRAIN_FRAC, 'split_seed': SPLIT_SEED,
            'probes': {
                'A': 'Set hidden 128 -> verdict (3-class)',
                'B': 'op one-hot 5 -> verdict (3-class)',
                'C': '[Set hidden, op one-hot] 133 -> verdict (3-class)',
                'D': 'Set hidden 128 -> op (5-class)',
            },
        },
        'per_seed': [r for r in per_seed if r is not None],
        'aggregate': agg,
    }
    with open('probe_op_decomposition.json', 'w') as f:
        json.dump(out, f, indent=2)

    md = render_markdown(per_seed, agg)
    with open('probe_op_decomposition.md', 'w', encoding='utf-8') as f:
        f.write(md)

    print(f"\n{'='*78}\n  Final report ({time.time()-t0:.1f}s):\n")
    print(md)


if __name__ == '__main__':
    main()
