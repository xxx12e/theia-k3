"""
Deep MLP probes at depths {2, 4, 6} on THEIA boundary hidden states (§5.3 / Table 7).

Two probe targets per (boundary, depth):
  - 3-class final verdict (paper bound: < 74% uncertainty-only ceiling)
  - binary Has-Unknown (~80% at Arith — ceiling check)

Protocol changes vs nonlinear_probe.py:
  - stratified 60/20/20 train/val/test split (not 70/30 best-test)
  - val-best epoch selection (not test-best)
  - 5 canonical checkpoints loaded in single invocation
  - Single JSON output with all (seed, boundary, depth, target) cells

Probe architecture (depth-D):
  Linear(128 -> 512) -> GELU -> Dropout(0.1)
  [Linear(512 -> 512) -> GELU -> Dropout(0.1)] × (D-1)
  Linear(512 -> n_classes)

Training: AdamW lr=1e-3 wd=0.01, cosine over 40 epochs, batch 2048.

Usage:
    python deep_nonlinear_probe.py            # all seeds, all depths
    python deep_nonlinear_probe.py --seeds 42 # single seed (debug)

Output:
    deep_nonlinear_probe_results.json
    deep_nonlinear_probe_report.md
"""
import argparse
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
import torch.nn.functional as F

from _theia_model_def import (
    IsisV9, DEVICE,
    NUM_RANGE, SET_DIM, P_UNKNOWN,
    VAL_FALSE, VAL_TRUE, VAL_UNKNOWN,
    N_RELS, N_ARITH, N_OPS,
    apply_logic,
)


# ---------- Config ----------
CHECKPOINTS = [
    (42,  'multi_seed_results/theia/seed_42/checkpoint.pth'),
    (123, 'multi_seed_results/theia_v2/seed_123/checkpoint.pth'),
    (256, 'multi_seed_results/theia_v2/seed_256/checkpoint.pth'),
    (777, 'multi_seed_results/theia_v2/seed_777/checkpoint.pth'),
    (999, 'multi_seed_results/theia_v2/seed_999/checkpoint.pth'),
]
N_SAMPLES   = 50_000
DATA_SEED   = 999
PROBE_HIDDEN = 512   # paper's shallow probe used width 256; deep probes use 512
PROBE_LR    = 1e-3
PROBE_EPOCHS = 40
BATCH       = 2048
SPLIT_FRACS = (0.6, 0.2, 0.2)   # train / val / test (tightened from 70/30)
SPLIT_SEED  = 0
DEPTHS      = [2, 4, 6]
BOUNDARIES  = ['arith', 'order', 'set', 'logic']
TARGETS     = ['verdict', 'has_unknown']  # 3-class, binary

# Falsification thresholds
H2_FALSIFY_VERDICT = 0.74        # paper's stated bound; > 78% triggers H2(a) full retract
H2_DEGRADE_VERDICT = 0.72        # 72-78% range: degrade claim


def gen_data(seed, n):
    """Same generator as nonlinear_probe.py / reprobe_theia_v2.py."""
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
    rt = (((rel==0)&(c >  d)) | ((rel==1)&(c <  d)) | ((rel==2)&(c == d)) |
          ((rel==3)&(c >= d)) | ((rel==4)&(c <= d)) | ((rel==5)&(c != d)))
    ord_v[rt] = 1
    val_o = torch.where(ord_unk, torch.tensor(2, device=DEVICE), ord_v)
    sb = torch.randint(0, 2, (n, SET_DIM), dtype=torch.float32, device=DEVICE)
    sou = su | c_unk
    ci = c.clamp(0, SET_DIM-1)
    ins = sb[torch.arange(n, device=DEVICE), ci].bool()
    sv = torch.where(ins, torch.tensor(1, device=DEVICE), torch.tensor(0, device=DEVICE))
    val_s = torch.where(sou, torch.tensor(2, device=DEVICE), sv)
    target = apply_logic(op, val_o, val_s)
    has_unk = (au | bu | du | su).long()
    return {
        'a_norm': a.float()/NUM_RANGE, 'b_norm': b.float()/NUM_RANGE, 'd_norm': d.float()/NUM_RANGE,
        'sb': sb, 's_unk': su, 'a_unk': au, 'b_unk': bu, 'd_unk': du,
        'arith': arith, 'rel': rel, 'op': op,
        'target': target, 'has_unk': has_unk,
    }


def forward_with_hidden(model, a, b, d, set_bits, s_unk, a_unk, b_unk, d_unk,
                        arith, rel, op):
    c_vec = model.arith_eng(a, b, a_unk, b_unk, arith)
    c_for_ord = model.bridge_ao(c_vec) + c_vec
    c_for_set = model.bridge_as(c_vec) + c_vec
    ord_vec = model.order_eng(c_for_ord, d, d_unk, rel)
    set_vec = model.set_eng(c_for_set, set_bits, s_unk)
    logic_vec = model.logic_eng(ord_vec, set_vec, op)
    return {'arith': c_vec, 'order': ord_vec, 'set': set_vec, 'logic': logic_vec}


# ---------- Probe ----------
class DeepMLPProbe(nn.Module):
    """D-hidden-layer MLP: 128 -> 512 -> 512 -> ... -> n_classes"""
    def __init__(self, d_in=128, d_hidden=PROBE_HIDDEN, n_classes=3, depth=2):
        super().__init__()
        assert depth >= 1
        layers = [nn.Linear(d_in, d_hidden), nn.GELU(), nn.Dropout(0.1)]
        for _ in range(depth - 1):
            layers += [nn.Linear(d_hidden, d_hidden), nn.GELU(), nn.Dropout(0.1)]
        layers += [nn.Linear(d_hidden, n_classes)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_probe_val_best(X_tr, y_tr, X_val, y_val, X_te, y_te,
                          depth, n_classes, epochs=PROBE_EPOCHS,
                          lr=PROBE_LR, batch=BATCH):
    """Train MLP probe with val-best epoch selection (no double-dipping).

    Each epoch: train on tr, eval on val and test (test acc recorded but never
    used for selection). best_epoch = argmax val_acc; report te_acc[best_epoch]
    only — not max-over-epochs test acc, which would be test-best double-dipping.
    Returns: te_acc_at_val_best, val_best_acc, val_best_epoch."""
    X_tr_t  = torch.from_numpy(X_tr).float().to(DEVICE)
    y_tr_t  = torch.from_numpy(y_tr).long().to(DEVICE)
    X_val_t = torch.from_numpy(X_val).float().to(DEVICE)
    y_val_t = torch.from_numpy(y_val).long().to(DEVICE)
    X_te_t  = torch.from_numpy(X_te).float().to(DEVICE)
    y_te_t  = torch.from_numpy(y_te).long().to(DEVICE)

    probe = DeepMLPProbe(d_in=X_tr.shape[1], d_hidden=PROBE_HIDDEN,
                         n_classes=n_classes, depth=depth).to(DEVICE)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    n_tr = X_tr_t.shape[0]
    best_val_acc = -1.0
    best_te_acc_at_best_val = 0.0
    best_epoch = -1
    for ep in range(epochs):
        probe.train()
        perm = torch.randperm(n_tr, device=DEVICE)
        for i in range(0, n_tr, batch):
            idx = perm[i:i + batch]
            logits = probe(X_tr_t[idx])
            loss = F.cross_entropy(logits, y_tr_t[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        probe.eval()
        with torch.no_grad():
            val_acc = (probe(X_val_t).argmax(-1) == y_val_t).float().mean().item()
            te_acc  = (probe(X_te_t).argmax(-1)  == y_te_t).float().mean().item()
        # Val-best selection
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_te_acc_at_best_val = te_acc
            best_epoch = ep
    return {
        'val_best_epoch': best_epoch,
        'val_best_acc': best_val_acc,
        'te_acc_at_val_best': best_te_acc_at_best_val,
    }


def stratified_split_60_20_20(strat_labels, seed=SPLIT_SEED):
    """Stratified 60/20/20 split by `strat_labels` (3-class verdict — the harder target).
    Returns indices into the original array. The same split is shared by both
    targets (verdict and has_unknown); per-class proportional 60/20/20 preserves
    class distribution across splits."""
    rng = np.random.RandomState(seed)
    classes = np.unique(strat_labels)
    tr_idx, val_idx, te_idx = [], [], []
    for c in classes:
        idx_c = np.where(strat_labels == c)[0].copy()
        rng.shuffle(idx_c)
        n_c = len(idx_c)
        n_tr = int(n_c * SPLIT_FRACS[0])
        n_val = int(n_c * SPLIT_FRACS[1])
        tr_idx.extend(idx_c[:n_tr])
        val_idx.extend(idx_c[n_tr:n_tr + n_val])
        te_idx.extend(idx_c[n_tr + n_val:])
    return np.array(tr_idx), np.array(val_idx), np.array(te_idx)


def run_seed(seed, ckpt_path):
    print(f'\n{"="*72}\n  seed {seed}: {ckpt_path}\n{"="*72}')
    if not os.path.exists(ckpt_path):
        print(f'  MISSING'); return None

    model = IsisV9().to(DEVICE)
    state = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state); model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params == 2_751_232, f'param count {n_params}'

    print(f'  Generating {N_SAMPLES} samples (data_seed={DATA_SEED}) ...')
    data = gen_data(DATA_SEED, N_SAMPLES)

    # Forward with hidden extraction (single pass)
    print(f'  Extracting hidden states (4 boundaries) ...')
    BSZ = 8192
    hidden = {b: [] for b in BOUNDARIES}
    with torch.no_grad():
        for i in range(0, N_SAMPLES, BSZ):
            j = min(i + BSZ, N_SAMPLES)
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

    target = data['target'].cpu().numpy()
    has_unk = data['has_unk'].cpu().numpy()

    # Stratified 60/20/20 split by 3-class verdict (harder target); same split
    # shared across (verdict, has_unknown) targets — single extraction.
    tr_idx, val_idx, te_idx = stratified_split_60_20_20(target, seed=SPLIT_SEED)
    # Verify class proportion preserved
    class_dist = lambda idx: tuple((target[idx] == c).mean().round(3) for c in (0, 1, 2))
    has_unk_dist = lambda idx: float((has_unk[idx] == 1).mean().round(3))
    print(f'  Stratified split: train={len(tr_idx)} val={len(val_idx)} test={len(te_idx)}')
    print(f'    Verdict (F,T,U) distribution: train={class_dist(tr_idx)} val={class_dist(val_idx)} test={class_dist(te_idx)}')
    print(f'    Has-Unknown=1 fraction:        train={has_unk_dist(tr_idx)} val={has_unk_dist(val_idx)} test={has_unk_dist(te_idx)}')

    out = {
        'seed': seed,
        'checkpoint': ckpt_path,
        'protocol': {
            'split': '60/20/20',
            'split_seed': SPLIT_SEED,
            'val_best_epoch_selection': True,
            'probe_hidden': PROBE_HIDDEN,
            'probe_epochs': PROBE_EPOCHS,
            'probe_lr': PROBE_LR,
            'depths': DEPTHS,
            'targets': TARGETS,
        },
        'per_cell': {},
    }

    for b in BOUNDARIES:
        for tgt_name in TARGETS:
            y = target if tgt_name == 'verdict' else has_unk
            n_cls = 3 if tgt_name == 'verdict' else 2
            for depth in DEPTHS:
                t0 = time.time()
                X = hidden[b]
                r = train_probe_val_best(
                    X[tr_idx],  y[tr_idx],
                    X[val_idx], y[val_idx],
                    X[te_idx],  y[te_idx],
                    depth=depth, n_classes=n_cls,
                )
                dt = time.time() - t0
                key = f'{b}__{tgt_name}__d{depth}'
                out['per_cell'][key] = {
                    'boundary': b, 'target': tgt_name, 'depth': depth,
                    'n_classes': n_cls,
                    'val_best_epoch': r['val_best_epoch'],
                    'val_best_acc': r['val_best_acc'],
                    'te_acc_at_val_best': r['te_acc_at_val_best'],
                    'fit_seconds': dt,
                }
                print(f'  {b:5s} {tgt_name:11s} d={depth}: val_best={r["val_best_acc"]:.4f} '
                      f'te@best={r["te_acc_at_val_best"]:.4f}  ({dt:.1f}s, ep {r["val_best_epoch"]})')

    del model; torch.cuda.empty_cache()
    return out


def aggregate(per_seed):
    rows = [r for r in per_seed if r is not None]
    if not rows: return {}
    seeds_used = [r['seed'] for r in rows]
    agg = {'n_seeds': len(seeds_used), 'seeds_used': seeds_used, 'per_cell_agg': {}}
    cell_keys = list(rows[0]['per_cell'].keys())
    for key in cell_keys:
        vals = [r['per_cell'][key]['te_acc_at_val_best'] for r in rows]
        agg['per_cell_agg'][key] = {
            'mean': float(np.mean(vals)),
            'std':  float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            'min':  float(np.min(vals)),
            'max':  float(np.max(vals)),
            'per_seed': vals,
        }
    return agg


def render_markdown(agg, gate_status):
    L = []
    L.append('# Deep MLP Probe — H2 Falsification Test')
    L.append('')
    L.append(f'5 canonical seeds, 50K samples (data_seed={DATA_SEED}), 60/20/20 split, val-best epoch.')
    L.append(f'Probe: Linear(128->{PROBE_HIDDEN}) → GELU → Dropout → [hidden×(D-1)] → Linear(out)')
    L.append(f'Training: AdamW lr={PROBE_LR}, cosine, {PROBE_EPOCHS} epochs, batch {BATCH}.')
    L.append('')
    L.append('## Verdict probe (3-class) — H2(a) verdict-side bound')
    L.append('')
    L.append('| Boundary | depth=2 | depth=4 | depth=6 | Gate (max upstream) |')
    L.append('|---|---|---|---|---|')
    for b in BOUNDARIES:
        row = f'| {b} |'
        for d in DEPTHS:
            k = f'{b}__verdict__d{d}'
            v = agg['per_cell_agg'].get(k)
            if v is None:
                row += ' - |'
            else:
                row += f' {v["mean"]*100:.2f} ± {v["std"]*100:.2f}% |'
        # gate marker
        gate_marker = ''
        if b != 'logic':
            for d in DEPTHS:
                k = f'{b}__verdict__d{d}'
                v = agg['per_cell_agg'].get(k)
                if v and v['mean'] >= 0.78:
                    gate_marker = '⚠️ EXCEED 78%'
                elif v and v['mean'] >= 0.72:
                    gate_marker = max(gate_marker, '⚠️ 72-78%')
        row += f' {gate_marker} |'
        L.append(row)
    L.append('')
    L.append('## Has-Unknown probe (binary) — Q13 ceiling check')
    L.append('')
    L.append('| Boundary | depth=2 | depth=4 | depth=6 |')
    L.append('|---|---|---|---|')
    for b in BOUNDARIES:
        row = f'| {b} |'
        for d in DEPTHS:
            k = f'{b}__has_unknown__d{d}'
            v = agg['per_cell_agg'].get(k)
            if v is None:
                row += ' - |'
            else:
                row += f' {v["mean"]*100:.2f} ± {v["std"]*100:.2f}% |'
        L.append(row)
    L.append('')
    L.append('## Gate decision')
    L.append('')
    L.append(f'**Status: {gate_status}**')
    return '\n'.join(L) + '\n'


def gate_decide(agg):
    """Gate thresholds:
      Set boundary @ depth=4 verdict acc:
        < 72%   → keep H2(a)
        72-78%  → degrade H2(a) to "linear + shallow MLP bound"
        ≥ 78%   → retract H2(a), STOP
      Apply same to Arith, Order; take strictest."""
    upstream = ['arith', 'order', 'set']
    worst_status = 'PASS — H2(a) STANDS'
    triggered_boundary = None
    for b in upstream:
        k = f'{b}__verdict__d4'
        v = agg['per_cell_agg'].get(k)
        if v is None: continue
        m = v['mean']
        if m >= 0.78:
            return f'STOP — H2(a) RETRACTED ({b} d=4 verdict = {m*100:.2f}% ≥ 78%)', b
        if m >= 0.72:
            worst_status = f'DEGRADE — H2(a) → "linear+shallow-MLP bound" ({b} d=4 = {m*100:.2f}% in 72-78%)'
            triggered_boundary = b
    return worst_status, triggered_boundary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', type=int, nargs='+', default=None,
                   help='Restrict to specific seeds (debug)')
    args = p.parse_args()

    seeds_to_run = CHECKPOINTS
    if args.seeds:
        seeds_to_run = [(s, p) for s, p in CHECKPOINTS if s in args.seeds]

    per_seed = []
    for seed, ckpt in seeds_to_run:
        per_seed.append(run_seed(seed, ckpt))

    agg = aggregate(per_seed)
    status, trig_b = gate_decide(agg)

    out = {
        'protocol': {
            'n_samples': N_SAMPLES, 'data_seed': DATA_SEED,
            'split': '60/20/20', 'split_seed': SPLIT_SEED,
            'probe_hidden': PROBE_HIDDEN, 'probe_epochs': PROBE_EPOCHS,
            'probe_lr': PROBE_LR, 'batch': BATCH,
            'depths': DEPTHS, 'targets': TARGETS,
            'val_best_epoch_selection': True,
        },
        'per_seed': [r for r in per_seed if r is not None],
        'aggregate': agg,
        'gate_status': status,
        'gate_triggered_boundary': trig_b,
    }
    with open('deep_nonlinear_probe_results.json', 'w') as f:
        json.dump(out, f, indent=2)

    md = render_markdown(agg, status)
    with open('deep_nonlinear_probe_report.md', 'w', encoding='utf-8') as f:
        f.write(md)

    print(f'\n{"="*72}\n  GATE STATUS\n{"="*72}\n  {status}')
    print(f'\nSaved: deep_nonlinear_probe_results.json + .md')


if __name__ == '__main__':
    main()
