"""
Sanity checks for probe_operator_stratified.py's val_set-only ceiling —
verifies the small SVC-vs-ceiling deltas are not artifacts. Three controls:

(B1) Shuffled val_set ceiling
     Permute val_set within each (op, NU) subset, recompute ceiling.
     If shuffled ceiling ≈ original ceiling -> the ceiling is NOT using
         val_set (degenerate to "predict majority verdict") -> bug-ish.
     If shuffled ceiling ≈ verdict majority baseline -> the ceiling IS
         using val_set.

(B2) Random Gaussian input probe
     Replace Set hidden with N(0, I) of identical shape, train LinearSVC.
     If acc ≈ verdict majority -> pipeline is null-correct; the original
         empirical SVC (~78%) genuinely uses Set hidden.
     If acc ≈ 78% -> empirical path has a leak (label leakage,
         StandardScaler peeking, etc.) -> bug.

(B3) True val_set as single feature, trained SVC
     Use one-hot(val_set) ∈ R^3 as the only feature; train LinearSVC.
     Should match the val_set-only ceiling within ±0.5pp; a gap > 1pp
         means the ceiling definition is off.

Plus reference:
  (REF) Verdict majority baseline within (op, NU): max class fraction.

Uses the same checkpoints, data, and per-(seed, op) split as
probe_operator_stratified.py, so all numbers are directly comparable.
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
from probe_operator_stratified import (
    forward_with_hidden, gen_data_extended,
    val_set_only_ceiling, probe_svc,
    CKPTS, N_SAMPLES, DATA_SEED, TRAIN_FRAC, SPLIT_SEED,
    OPS, TV_NAME,
)

N_SHUFFLES = 5  # for B1 stability
RAND_NOISE_DIM = 128  # matches Set hidden dim


def verdict_majority_baseline(y_te):
    """REF: predict majority verdict label always."""
    n_F = int((y_te == VAL_FALSE).sum())
    n_T = int((y_te == VAL_TRUE).sum())
    n_total = n_F + n_T
    if n_total == 0:
        return float('nan'), None, float('nan')
    if n_T >= n_F:
        return float(n_T / n_total), 'T', float(n_T / n_total)
    return float(n_F / n_total), 'F', float(n_F / n_total)


def shuffled_val_set_ceiling(val_set_tr, y_tr, val_set_te, y_te, n_shuffles=N_SHUFFLES, seed=0):
    """B1: shuffle val_set within the (op, NU) subset (separately for train
    and test, breaking val_set <-> verdict correlation), recompute ceiling.
    Average over n_shuffles for stability."""
    rng = np.random.RandomState(seed)
    accs = []
    for s in range(n_shuffles):
        vs_tr_perm = rng.permutation(val_set_tr)
        vs_te_perm = rng.permutation(val_set_te)
        acc, _ = val_set_only_ceiling(vs_tr_perm, y_tr, vs_te_perm, y_te)
        accs.append(acc)
    return float(np.mean(accs)), float(np.std(accs))


def random_gaussian_svc(n_tr, n_te, y_tr, y_te, dim=RAND_NOISE_DIM, seed=0):
    """B2: replace Set hidden with N(0, I) noise of identical shape,
    train LinearSVC on the noise. Should reach verdict majority baseline."""
    rng = np.random.RandomState(seed)
    X_tr = rng.randn(n_tr, dim).astype(np.float32)
    X_te = rng.randn(n_te, dim).astype(np.float32)
    return probe_svc(X_tr, y_tr, X_te, y_te)


def val_set_feature_svc(val_set_tr, y_tr, val_set_te, y_te):
    """B3: encode val_set ∈ {0, 1, 2} as one-hot R^3 feature, train SVC.
    Should match the rule-based val_set-only ceiling within ±0.5pp."""
    def onehot(vs):
        n = len(vs)
        x = np.zeros((n, 3), dtype=np.float32)
        x[np.arange(n), vs.astype(int)] = 1.0
        return x
    X_tr = onehot(val_set_tr)
    X_te = onehot(val_set_te)
    # NOTE: StandardScaler on a one-hot can be unstable; use as-is (already 0/1)
    dual = X_tr.shape[0] < X_tr.shape[1]
    clf = LinearSVC(C=1.0, max_iter=2000, dual=dual)
    clf.fit(X_tr, y_tr)
    return float(clf.score(X_te, y_te))


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
    nu_mask = target_np != VAL_UNKNOWN

    out = {'seed': seed, 'per_op': {}}

    print(f"  {'op':<5} {'n_NU':>6} {'ref_maj':>8} {'B1_shuf':>10} "
          f"{'B2_rand':>9} {'B3_vset':>9} {'orig_ceil':>10} {'orig_svc':>9}")
    for op_name, op_idx in OPS:
        m_op_nu = nu_mask & (op_np == op_idx)
        n = int(m_op_nu.sum())
        if n < 100:
            continue

        idx_all = np.where(m_op_nu)[0]
        rng = np.random.RandomState(SPLIT_SEED + seed * 100 + op_idx)
        perm = rng.permutation(len(idx_all))
        n_train = int(len(idx_all) * TRAIN_FRAC)
        tr_idx = idx_all[perm[:n_train]]
        te_idx = idx_all[perm[n_train:]]

        X_tr, X_te = set_hidden[tr_idx], set_hidden[te_idx]
        y_tr, y_te = target_np[tr_idx], target_np[te_idx]
        vs_tr, vs_te = val_set_np[tr_idx], val_set_np[te_idx]

        # Class distribution diagnostic
        n_F_te = int((y_te == VAL_FALSE).sum())
        n_T_te = int((y_te == VAL_TRUE).sum())
        n_F_tr = int((y_tr == VAL_FALSE).sum())
        n_T_tr = int((y_tr == VAL_TRUE).sum())

        # Reference
        ref_acc, ref_class, _ = verdict_majority_baseline(y_te)

        # B1
        b1_mean, b1_std = shuffled_val_set_ceiling(
            vs_tr, y_tr, vs_te, y_te,
            seed=SPLIT_SEED + seed * 100 + op_idx + 1,
        )

        # B2
        b2_acc = random_gaussian_svc(
            len(tr_idx), len(te_idx), y_tr, y_te,
            seed=SPLIT_SEED + seed * 100 + op_idx + 2,
        )

        # B3
        b3_acc = val_set_feature_svc(vs_tr, y_tr, vs_te, y_te)

        # Reproduce probe_operator_stratified numbers (sanity)
        orig_ceil, _ = val_set_only_ceiling(vs_tr, y_tr, vs_te, y_te)
        orig_svc = probe_svc(X_tr, y_tr, X_te, y_te)

        out['per_op'][op_name] = {
            'n_train': int(n_train),
            'n_test':  int(len(idx_all) - n_train),
            'class_dist_train_pct': {'F': 100.0*n_F_tr/n_train, 'T': 100.0*n_T_tr/n_train},
            'class_dist_test_pct':  {'F': 100.0*n_F_te/(n_F_te+n_T_te), 'T': 100.0*n_T_te/(n_F_te+n_T_te)},
            'ref_majority_baseline_pct': ref_acc * 100.0,
            'ref_majority_class':        ref_class,
            'B1_shuffled_ceiling_mean_pct': b1_mean * 100.0,
            'B1_shuffled_ceiling_std_pct':  b1_std  * 100.0,
            'B2_random_gaussian_svc_pct':   b2_acc * 100.0,
            'B3_val_set_onehot_svc_pct':    b3_acc * 100.0,
            'original_ceiling_pct':         orig_ceil * 100.0,
            'original_svc_pct':             orig_svc * 100.0,
        }

        print(f"  {op_name:<5} {n:>6} {ref_acc*100:>7.2f}% "
              f"{b1_mean*100:>8.2f}±{b1_std*100:>3.1f} "
              f"{b2_acc*100:>8.2f}% {b3_acc*100:>8.2f}% "
              f"{orig_ceil*100:>9.2f}% {orig_svc*100:>8.2f}%")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def aggregate(per_seed):
    seeds_used = [r['seed'] for r in per_seed if r is not None]
    agg = {'n_seeds': len(seeds_used), 'seeds_used': seeds_used, 'per_op': {}}
    keys = ['ref_majority_baseline_pct',
            'B1_shuffled_ceiling_mean_pct',
            'B2_random_gaussian_svc_pct',
            'B3_val_set_onehot_svc_pct',
            'original_ceiling_pct',
            'original_svc_pct']
    for op_name, _ in OPS:
        rows = [r['per_op'][op_name] for r in per_seed
                if r and op_name in r['per_op']]
        if not rows:
            continue
        agg['per_op'][op_name] = {}
        for k in keys:
            vals = [r[k] for r in rows]
            agg['per_op'][op_name][k + '_mean'] = float(np.mean(vals))
            agg['per_op'][op_name][k + '_std']  = float(np.std(vals))
        cd = [r['class_dist_test_pct'] for r in rows]
        agg['per_op'][op_name]['class_dist_test_pct_mean'] = {
            k: float(np.mean([d[k] for d in cd])) for k in cd[0]
        }
    return agg


def render_markdown(per_seed, agg):
    L = []
    L.append("# Sanity Check — val_set-only Ceiling Validity")
    L.append("")
    L.append(f"5 checkpoints {agg['seeds_used']} × 4 ops on Set boundary NU subset.")
    L.append("Same data + split as `probe_operator_stratified.py` (so `original_*` columns reproduce the operator-stratified results).")
    L.append("")
    L.append("## Aggregate (mean ± std over 5 seeds, all percentages)")
    L.append("")
    L.append("| Op | NU verdict majority | (B1) shuffled val_set ceil | (B2) random N(0,I) SVC | (B3) val_set onehot SVC | original ceiling | original Set SVC |")
    L.append("|---|---|---|---|---|---|---|")
    for op_name, _ in OPS:
        if op_name not in agg['per_op']: continue
        v = agg['per_op'][op_name]
        L.append(f"| {op_name} | "
                 f"{v['ref_majority_baseline_pct_mean']:.2f} ± {v['ref_majority_baseline_pct_std']:.2f} | "
                 f"{v['B1_shuffled_ceiling_mean_pct_mean']:.2f} ± {v['B1_shuffled_ceiling_mean_pct_std']:.2f} | "
                 f"{v['B2_random_gaussian_svc_pct_mean']:.2f} ± {v['B2_random_gaussian_svc_pct_std']:.2f} | "
                 f"{v['B3_val_set_onehot_svc_pct_mean']:.2f} ± {v['B3_val_set_onehot_svc_pct_std']:.2f} | "
                 f"{v['original_ceiling_pct_mean']:.2f} ± {v['original_ceiling_pct_std']:.2f} | "
                 f"{v['original_svc_pct_mean']:.2f} ± {v['original_svc_pct_std']:.2f} |")
    L.append("")
    L.append("## Class distribution within (op, NU) test set")
    L.append("")
    L.append("| Op | F % | T % | Verdict majority class |")
    L.append("|---|---|---|---|")
    for op_name, _ in OPS:
        if op_name not in agg['per_op']: continue
        cd = agg['per_op'][op_name]['class_dist_test_pct_mean']
        majc = 'T' if cd['T'] >= cd['F'] else 'F'
        L.append(f"| {op_name} | {cd['F']:.2f} | {cd['T']:.2f} | {majc} |")
    L.append("")
    L.append("## Diagnostic checks")
    L.append("")
    bug_flags = []
    for op_name, _ in OPS:
        if op_name not in agg['per_op']: continue
        v = agg['per_op'][op_name]
        ref  = v['ref_majority_baseline_pct_mean']
        b1   = v['B1_shuffled_ceiling_mean_pct_mean']
        b2   = v['B2_random_gaussian_svc_pct_mean']
        b3   = v['B3_val_set_onehot_svc_pct_mean']
        orig = v['original_ceiling_pct_mean']
        svc  = v['original_svc_pct_mean']

        L.append(f"### {op_name}")
        L.append("")
        # B1 check
        diff_b1_ref  = b1 - ref
        diff_b1_orig = b1 - orig
        if abs(diff_b1_ref) <= 1.5:
            L.append(f"- **B1 ✓**: shuffled ceiling ({b1:.2f}%) matches verdict majority "
                     f"({ref:.2f}%) within ±1.5pp → original ceiling IS using val_set info "
                     f"(shuffling kills the signal).")
        elif abs(diff_b1_orig) <= 1.5:
            L.append(f"- **B1 ✗ BUG-LIKE**: shuffled ceiling ({b1:.2f}%) ≈ original ceiling "
                     f"({orig:.2f}%); shuffling val_set did NOT change ceiling → ceiling "
                     f"is NOT using val_set; it's degenerate to majority.")
            bug_flags.append(f'{op_name}_B1')
        else:
            L.append(f"- **B1 ?**: shuffled ceiling ({b1:.2f}%) sits between majority "
                     f"({ref:.2f}%) and original ceiling ({orig:.2f}%); intermediate signal.")

        # B2 check
        diff_b2 = b2 - ref
        if abs(diff_b2) <= 2.0:
            L.append(f"- **B2 ✓**: random N(0,I) SVC ({b2:.2f}%) ≈ verdict majority "
                     f"({ref:.2f}%) within ±2pp → empirical SVC pipeline is null-correct; "
                     f"original Set SVC ({svc:.2f}%) genuinely reads Set hidden.")
        else:
            L.append(f"- **B2 ✗ SUSPICIOUS**: random N(0,I) SVC ({b2:.2f}%) ≠ verdict "
                     f"majority ({ref:.2f}%) by {diff_b2:+.2f}pp → empirical pipeline "
                     f"may have data leakage.")
            bug_flags.append(f'{op_name}_B2')

        # B3 check
        diff_b3 = b3 - orig
        if abs(diff_b3) <= 1.0:
            L.append(f"- **B3 ✓**: val_set-onehot SVC ({b3:.2f}%) matches rule-based ceiling "
                     f"({orig:.2f}%) within ±1pp → ceiling correctly captures what a probe "
                     f"with val_set input would learn.")
        else:
            L.append(f"- **B3 ✗ INCONSISTENT**: val_set-onehot SVC ({b3:.2f}%) ≠ rule-based "
                     f"ceiling ({orig:.2f}%) by {diff_b3:+.2f}pp → ceiling definition diverges "
                     f"from probe behavior.")
            bug_flags.append(f'{op_name}_B3')
        L.append("")

    L.append("## Verdict")
    L.append("")
    if not bug_flags:
        L.append("**No bugs detected.** Ceiling computation is correct (B3 ✓), "
                 "shuffling val_set kills the ceiling signal (B1 ✓), and the empirical "
                 "SVC pipeline is null-correct (B2 ✓). The operator-stratified conclusion stands.")
    else:
        L.append(f"**Bug flags raised**: {bug_flags}")
        L.append("")
        L.append("**However, observed across-op pattern may also be substantive:**")
        L.append("")
        L.append("If for some op (e.g. AND) the ceiling **happens to equal** the verdict "
                 "majority because val_set is a pivot variable that, in the NU stratum, "
                 "perfectly predicts verdict by always firing the absorbing case "
                 "(F ∧ X = F), this is *not* a bug — it's a degenerate-ceiling property. "
                 "B1 (shuffled) is the discriminator: if shuffling DOES change the "
                 "ceiling, val_set is genuinely doing work even when its rule is "
                 "constant. Read B1 first.")
    return "\n".join(L) + "\n"


def main():
    t0 = time.time()
    per_seed = []
    for seed, ckpt in CKPTS:
        per_seed.append(run_seed(seed, ckpt))
    agg = aggregate(per_seed)

    out = {
        'protocol': {
            'controls': {
                'B1_shuffled_val_set_ceiling': f'shuffle val_set within (op, NU) subset, n_shuffles={N_SHUFFLES}',
                'B2_random_gaussian_input':    f'replace Set hidden with N(0, I) of dim {RAND_NOISE_DIM}',
                'B3_val_set_onehot_svc':       'one-hot encode val_set as 3-dim feature for SVC',
                'REF_verdict_majority':        'predict majority verdict class always',
            },
            'reuses_operator_stratified_split': True,
            'n_samples': N_SAMPLES,
            'data_seed': DATA_SEED,
        },
        'per_seed': [r for r in per_seed if r is not None],
        'aggregate': agg,
    }
    with open('probe_sanity_check.json', 'w') as f:
        json.dump(out, f, indent=2)

    md = render_markdown(per_seed, agg)
    with open('probe_sanity_check.md', 'w', encoding='utf-8') as f:
        f.write(md)

    print(f"\n{'='*78}\n  Final report ({time.time()-t0:.1f}s):\n")
    print(md)


if __name__ == '__main__':
    main()
