"""
Linear SVM probe on the four boundary hidden states (§5.3 / Table 7) — a
different probe family than the MLPs, checking that the delayed-verdict bound
is not specific to MLP probes.

Protocol — identical to deep_nonlinear_probe.py except for the probe model:
  - 5 canonical THEIA seeds: 42 (legacy v1), 123, 256, 777, 999 (v2)
  - 50,000 fresh samples (data_seed=999), extract 4 boundary states (128-d each)
  - Stratified 60/20/20 split by 3-class verdict (split_seed=0)
  - Per (seed, boundary, target): fit LinearSVC (one-vs-rest, C=1.0,
    max_iter=3000) with StandardScaler on train, report test acc
  - Single fit per cell (no epoch selection needed for SVM)

Output: svm_probe_5seed_results.json. Runs on CPU.
"""
import os, sys, json, time
sys.path.insert(0, '.')

# Force CPU for the IsisV9 forward pass (avoid GPU contention with concurrent training)
os.environ['CUDA_VISIBLE_DEVICES'] = ''

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import numpy as np
import torch
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Force torch DEVICE to CPU before importing model
import torch
torch.cuda.is_available = lambda: False  # monkey-patch to ensure CPU
from _theia_model_def import (
    IsisV9,
    NUM_RANGE, SET_DIM, P_UNKNOWN,
    VAL_FALSE, VAL_TRUE, VAL_UNKNOWN,
    N_RELS, N_ARITH, N_OPS,
    REL_GT, REL_LT, REL_EQ, REL_GTE, REL_LTE, REL_NEQ,
    ARITH_ADD, ARITH_SUB, ARITH_MUL, ARITH_MOD,
    OP_AND, OP_OR, OP_NOT, OP_IMPLIES, OP_IFF,
    apply_logic,
)
DEVICE = 'cpu'

CHECKPOINTS = [
    (42,  'multi_seed_results/theia/seed_42/checkpoint.pth'),
    (123, 'multi_seed_results/theia_v2/seed_123/checkpoint.pth'),
    (256, 'multi_seed_results/theia_v2/seed_256/checkpoint.pth'),
    (777, 'multi_seed_results/theia_v2/seed_777/checkpoint.pth'),
    (999, 'multi_seed_results/theia_v2/seed_999/checkpoint.pth'),
]
N_PER_SEED = 50_000
DATA_SEED = 999    # matches deep_nonlinear_probe.py
SPLIT_SEED = 0
BATCH = 1024


def gen_balanced(n_total, seed):
    torch.manual_seed(seed)
    chunks = []
    a = torch.randint(1, NUM_RANGE+1, (n_total,))
    b = torch.randint(1, NUM_RANGE+1, (n_total,))
    d = torch.randint(0, NUM_RANGE+1, (n_total,))
    sb = torch.randint(0, 2, (n_total, SET_DIM), dtype=torch.float32)
    au = torch.rand(n_total) < P_UNKNOWN
    bu = torch.rand(n_total) < P_UNKNOWN
    du = torch.rand(n_total) < P_UNKNOWN
    su = torch.rand(n_total) < P_UNKNOWN
    ar = torch.randint(0, N_ARITH, (n_total,))
    rl = torch.randint(0, N_RELS, (n_total,))
    op = torch.randint(0, N_OPS, (n_total,))

    c = torch.zeros(n_total, dtype=torch.long)
    c[ar==ARITH_ADD] = torch.clamp(a+b, 0, NUM_RANGE)[ar==ARITH_ADD]
    c[ar==ARITH_SUB] = torch.abs(a-b)[ar==ARITH_SUB]
    c[ar==ARITH_MUL] = torch.clamp(a*b, 0, NUM_RANGE)[ar==ARITH_MUL]
    c[ar==ARITH_MOD] = (a % torch.clamp(b, 1, NUM_RANGE))[ar==ARITH_MOD]

    c_unk = au | bu
    ord_unk = c_unk | du
    rel_true = (((rl==REL_GT) & (c >  d)) | ((rl==REL_LT) & (c <  d)) |
                ((rl==REL_EQ) & (c == d)) | ((rl==REL_GTE) & (c >= d)) |
                ((rl==REL_LTE) & (c <= d)) | ((rl==REL_NEQ) & (c != d)))
    ord_v = torch.where(rel_true, torch.tensor(VAL_TRUE),
                                  torch.tensor(VAL_FALSE))
    val_o = torch.where(ord_unk, torch.tensor(VAL_UNKNOWN), ord_v)

    sou = su | c_unk
    ci = c.clamp(0, SET_DIM-1)
    ins = sb[torch.arange(n_total), ci].bool()
    sv = torch.where(ins, torch.tensor(VAL_TRUE), torch.tensor(VAL_FALSE))
    val_s = torch.where(sou, torch.tensor(VAL_UNKNOWN), sv)

    target = apply_logic(op, val_o, val_s)

    return {
        'a':  a.float() / NUM_RANGE,
        'b':  b.float() / NUM_RANGE,
        'd':  d.float() / NUM_RANGE,
        'sb': sb,
        'a_unk': au, 'b_unk': bu, 'd_unk': du, 's_unk': su,
        'ar': ar, 'rl': rl, 'op': op,
        'tgt': target,
    }


def forward_with_hidden(model, data):
    """Replicate IsisV9.forward but capture all 4 boundary hidden vectors."""
    model.eval()
    n = len(data['tgt'])
    hidden = {'arith': [], 'order': [], 'set': [], 'logic': []}
    with torch.no_grad():
        for i in range(0, n, BATCH):
            sl = slice(i, min(i+BATCH, n))
            c_vec = model.arith_eng(data['a'][sl], data['b'][sl],
                                    data['a_unk'][sl], data['b_unk'][sl], data['ar'][sl])
            c_for_ord = model.bridge_ao(c_vec) + c_vec
            c_for_set = model.bridge_as(c_vec) + c_vec
            ord_vec = model.order_eng(c_for_ord, data['d'][sl], data['d_unk'][sl], data['rl'][sl])
            set_vec = model.set_eng(c_for_set, data['sb'][sl], data['s_unk'][sl])
            logic_vec = model.logic_eng(ord_vec, set_vec, data['op'][sl])
            hidden['arith'].append(c_vec.numpy())
            hidden['order'].append(ord_vec.numpy())
            hidden['set'].append(set_vec.numpy())
            hidden['logic'].append(logic_vec.numpy())
    return {k: np.concatenate(v, axis=0) for k, v in hidden.items()}


def stratified_split_60_20_20(strat_labels, seed=SPLIT_SEED):
    rng = np.random.RandomState(seed)
    classes = np.unique(strat_labels)
    tr_idx, val_idx, te_idx = [], [], []
    for c in classes:
        idx_c = np.where(strat_labels == c)[0].copy()
        rng.shuffle(idx_c)
        n_c = len(idx_c)
        n_tr = int(n_c * 0.6); n_val = int(n_c * 0.2)
        tr_idx.extend(idx_c[:n_tr].tolist())
        val_idx.extend(idx_c[n_tr:n_tr + n_val].tolist())
        te_idx.extend(idx_c[n_tr + n_val:].tolist())
    return np.array(tr_idx), np.array(val_idx), np.array(te_idx)


def fit_eval_svm(X_train, y_train, X_test, y_test):
    """Fit StandardScaler → LinearSVC (one-vs-rest), evaluate test acc."""
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('svc', LinearSVC(C=1.0, max_iter=3000, dual='auto', random_state=0)),
    ])
    t0 = time.time()
    pipe.fit(X_train, y_train)
    fit_t = time.time() - t0
    return float(pipe.score(X_test, y_test)), fit_t


def main():
    print('=' * 78)
    print(f'  SVM probe (LinearSVC, C=1.0, StandardScaler) — adds non-MLP probe family')
    print(f'  Protocol matches deep_nonlinear_probe.py exactly except probe model')
    print('=' * 78)

    per_seed_results = []
    for seed, ckpt in CHECKPOINTS:
        if not os.path.exists(ckpt):
            print(f'\nseed {seed}: MISSING {ckpt}')
            continue
        print(f'\n--- seed {seed}: loading {ckpt} (CPU only) ---')
        t0 = time.time()
        data = gen_balanced(N_PER_SEED, seed=DATA_SEED)
        gen_t = time.time() - t0

        model = IsisV9().to('cpu')
        state = torch.load(ckpt, map_location='cpu', weights_only=True)
        model.load_state_dict(state)
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params == 2_751_232

        t0 = time.time()
        hidden = forward_with_hidden(model, data)
        fwd_t = time.time() - t0
        print(f'  data gen={gen_t:.1f}s, forward={fwd_t:.1f}s, n={N_PER_SEED}')

        verdict = data['tgt'].numpy()
        has_unknown = (data['a_unk'] | data['b_unk'] |
                       data['d_unk'] | data['s_unk']).numpy().astype(int)
        tr_idx, val_idx, te_idx = stratified_split_60_20_20(verdict)
        # SVM has no epoch selection, so val is unused: fit on train (60%) only and
        # evaluate on test, matching deep_nonlinear_probe.py's test-at-val-best reports.
        seed_record = {'seed': seed, 'verdict': {}, 'has_unknown': {}}
        for boundary in ['arith', 'order', 'set', 'logic']:
            X = hidden[boundary]
            X_tr, X_te = X[tr_idx], X[te_idx]

            # Verdict (3-class)
            y_tr, y_te = verdict[tr_idx], verdict[te_idx]
            acc_v, t_v = fit_eval_svm(X_tr, y_tr, X_te, y_te)
            seed_record['verdict'][boundary] = {'te_acc': acc_v, 'fit_s': t_v,
                                                'n_train': len(X_tr), 'n_test': len(X_te)}
            # Has-unknown (binary)
            y_tr, y_te = has_unknown[tr_idx], has_unknown[te_idx]
            acc_h, t_h = fit_eval_svm(X_tr, y_tr, X_te, y_te)
            seed_record['has_unknown'][boundary] = {'te_acc': acc_h, 'fit_s': t_h}
            print(f'  {boundary:>5}: verdict={acc_v*100:6.4f}% (fit {t_v:.1f}s) | '
                  f'has_unk={acc_h*100:6.4f}% (fit {t_h:.1f}s)')

        per_seed_results.append(seed_record)
        del model

    # Aggregate
    boundaries = ['arith', 'order', 'set', 'logic']
    aggregate = {'verdict': {}, 'has_unknown': {}}
    for tgt in ['verdict', 'has_unknown']:
        for b in boundaries:
            vals = [r[tgt][b]['te_acc'] for r in per_seed_results]
            aggregate[tgt][b] = {
                'mean':   float(np.mean(vals)),
                'std':    float(np.std(vals, ddof=1)),
                'min':    float(np.min(vals)),
                'max':    float(np.max(vals)),
                'per_seed': vals,
            }

    out = {
        'protocol': {
            'probe': 'LinearSVC C=1.0 max_iter=3000 (StandardScaler pre-pipeline)',
            'split': '60/20/20 stratified by verdict',
            'split_seed': SPLIT_SEED,
            'data_seed': DATA_SEED,
            'n_per_seed': N_PER_SEED,
            'fit_on': 'train (60%) only — val unused for SVM, test held out',
        },
        'per_seed': per_seed_results,
        'aggregate': aggregate,
    }
    json.dump(out, open('svm_probe_5seed_results.json', 'w'), indent=2)
    print('\nSaved svm_probe_5seed_results.json')

    # Summary table
    print()
    print('=' * 78)
    print('  SVM (LinearSVC, C=1) verdict accuracy 5-seed mean ± std')
    print('=' * 78)
    print(f'  {"boundary":>10} | {"mean ± std":>20} | {"per-seed":>50}')
    print('  ' + '-' * 78)
    for b in boundaries:
        ag = aggregate['verdict'][b]
        ps = ' '.join(f'{v*100:5.2f}' for v in ag['per_seed'])
        print(f'  {b:>10} | {ag["mean"]*100:6.4f} ± {ag["std"]*100:5.4f} | {ps}')

    # Aggregate max-upstream
    upstream = ['arith', 'order', 'set']
    max_mean = 0.0
    max_cell = None
    for b in upstream:
        if aggregate['verdict'][b]['mean'] > max_mean:
            max_mean = aggregate['verdict'][b]['mean']
            max_cell = b
    print(f'\n  MAX upstream verdict (SVM, 5-seed mean): {max_cell} = {max_mean*100:.4f}%')

    if max_mean < 0.72:
        print('  PASS — H2(a) holds under SVM probe family too')


if __name__ == '__main__':
    main()
