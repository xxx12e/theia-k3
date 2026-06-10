#!/usr/bin/env python
"""
Clean-split MLP and linear-SVM probes with val-best selection (§5.3 / Table 7).
Unlike the legacy 70/30 protocol (test-best-epoch), this adds a val split for
epoch/C selection and evaluates the test set once.

Protocols:
  reproduce_70_30 — replicate the legacy 70/30 MLP (test-best-epoch) and SVM
                    (C=1.0); used only for sanity checks against
                    nonlinear_probe.json / mechanistic_probing_v2.json.
  clean_split_v1  — 60/20/20 split; MLP val-best-epoch, SVM C swept over
                    {0.1, 1.0, 10.0} on val; test evaluated once.

Usage:
  python probe_cleansplit.py --checkpoint <path> [--protocol clean_split_v1]
  python probe_cleansplit.py --sanity        # seed 42 reproduce + compare
  python probe_cleansplit.py --aggregate     # write clean_split_v1 report

Outputs <ckpt_dir>/probe_cleansplit.json (never overwrites the legacy JSONs)
and, on --aggregate, multi_seed_results/reports/probe_cleansplit_report.md.
"""
import argparse, json, os, time, hashlib, sys
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler

# --- Constants (verbatim from reprobe_theia_v2.py / nonlinear_probe.py) ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
D_MODEL = 128
NUM_RANGE = 20
SET_DIM = 21
P_UNKNOWN = 0.15
VAL_FALSE, VAL_TRUE, VAL_UNKNOWN = 0, 1, 2
N_VALS = 3
N_RELS, N_ARITH, N_OPS = 6, 4, 5
OP_AND, OP_OR, OP_NOT, OP_IMPLIES, OP_IFF = 0, 1, 2, 3, 4

AND_T = torch.tensor([[0,0,0],[0,1,2],[0,2,2]], dtype=torch.long, device=DEVICE)
OR_T  = torch.tensor([[0,1,2],[1,1,1],[2,1,2]], dtype=torch.long, device=DEVICE)
IMP_T = torch.tensor([[1,1,1],[0,1,2],[2,1,2]], dtype=torch.long, device=DEVICE)
IFF_T = torch.tensor([[1,0,2],[0,1,2],[2,2,2]], dtype=torch.long, device=DEVICE)
NOT_T = torch.tensor([1,0,2], dtype=torch.long, device=DEVICE)

ROOT = r'multi_seed_results'
REPORT_DIR = os.path.join(ROOT, 'reports')

SEED_PATHS = {
    42:  os.path.join(ROOT, 'theia',    'seed_42',  'checkpoint.pth'),
    123: os.path.join(ROOT, 'theia_v2', 'seed_123', 'checkpoint.pth'),
    256: os.path.join(ROOT, 'theia_v2', 'seed_256', 'checkpoint.pth'),
    777: os.path.join(ROOT, 'theia_v2', 'seed_777', 'checkpoint.pth'),
    999: os.path.join(ROOT, 'theia_v2', 'seed_999', 'checkpoint.pth'),
}

PROTOCOL_VERSION = "clean_split_v1"
H2A_CEILING = 0.74  # uncertainty-only ceiling; Arith SVM/MLP and Order/Set MLP must stay below

BOUNDARIES = ['arith', 'order', 'set', 'logic']
SVM_C_GRID = [0.1, 1.0, 10.0]

# --- Model (verbatim from reprobe_theia_v2.py) ---
def apply_logic(op, va, vb):
    r = torch.zeros_like(op)
    m = op == OP_AND;     r[m] = AND_T[va[m], vb[m]]
    m = op == OP_OR;      r[m] = OR_T[va[m], vb[m]]
    m = op == OP_IMPLIES; r[m] = IMP_T[va[m], vb[m]]
    m = op == OP_IFF;     r[m] = IFF_T[va[m], vb[m]]
    m = op == OP_NOT;     r[m] = NOT_T[va[m]]
    return r


class NumEnc(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(1, D_MODEL//2), nn.GELU(), nn.Linear(D_MODEL//2, D_MODEL))
        self.unknown_vec = nn.Parameter(torch.randn(D_MODEL))
    def forward(self, x, is_unknown):
        v = self.f(x.unsqueeze(-1))
        unk = self.unknown_vec.unsqueeze(0).expand_as(v)
        return torch.where(is_unknown.unsqueeze(-1), unk, v)


class SetEnc(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(SET_DIM, D_MODEL//2), nn.GELU(), nn.Linear(D_MODEL//2, D_MODEL))
        self.unknown_vec = nn.Parameter(torch.randn(D_MODEL))
    def forward(self, bits, is_unknown):
        v = self.f(bits)
        unk = self.unknown_vec.unsqueeze(0).expand_as(v)
        return torch.where(is_unknown.unsqueeze(-1), unk, v)


def make_mlp(in_d, out_d, dropout=0.0):
    layers = [nn.Linear(in_d, in_d*2), nn.GELU(), nn.LayerNorm(in_d*2), nn.Linear(in_d*2, out_d)]
    if dropout > 0:
        layers.insert(3, nn.Dropout(dropout))
    return nn.Sequential(*layers)


class ArithEngine(nn.Module):
    def __init__(self):
        super().__init__()
        self.ne = NumEnc()
        self.ae = nn.Embedding(N_ARITH, D_MODEL)
        self.net = make_mlp(D_MODEL*3, D_MODEL)
    def forward(self, a, b, a_unk, b_unk, arith):
        return self.net(torch.cat([self.ne(a, a_unk), self.ne(b, b_unk), self.ae(arith)], dim=-1))


class OrderEngine(nn.Module):
    def __init__(self):
        super().__init__()
        self.ne = NumEnc(); self.re = nn.Embedding(N_RELS, D_MODEL)
        mk = lambda: nn.Sequential(nn.Linear(D_MODEL*3, D_MODEL*2), nn.GELU(),
                                   nn.LayerNorm(D_MODEL*2), nn.Linear(D_MODEL*2, D_MODEL))
        self.G = mk(); self.L = mk(); self.E = mk()
        self.Gg = nn.Sequential(nn.Linear(D_MODEL*2, D_MODEL), nn.GELU(), nn.LayerNorm(D_MODEL))
        self.Lg = nn.Sequential(nn.Linear(D_MODEL*2, D_MODEL), nn.GELU(), nn.LayerNorm(D_MODEL))
        self.Eg = nn.Sequential(nn.Linear(D_MODEL*3, D_MODEL), nn.GELU(), nn.LayerNorm(D_MODEL))
        self.out = make_mlp(D_MODEL*4, D_MODEL)
    def forward(self, c_vec, d, d_unk, rel):
        vd = self.ne(d, d_unk); vr = self.re(rel)
        x = torch.cat([c_vec, vd, vr], dim=-1)
        g = self.G(x); l = self.L(x); e = self.E(x)
        g = self.Gg(torch.cat([g, e], dim=-1))
        l = self.Lg(torch.cat([l, e], dim=-1))
        e = self.Eg(torch.cat([e, g, l], dim=-1))
        return self.out(torch.cat([g, l, e, c_vec], dim=-1))


class SetEngine(nn.Module):
    def __init__(self):
        super().__init__()
        self.se = SetEnc()
        self.net = make_mlp(D_MODEL*2, D_MODEL)
    def forward(self, c_vec, set_bits, s_unk):
        return self.net(torch.cat([c_vec, self.se(set_bits, s_unk)], dim=-1))


class LogicEngine(nn.Module):
    def __init__(self):
        super().__init__()
        self.oe = nn.Embedding(N_OPS, D_MODEL)
        mk = lambda: nn.Sequential(nn.Linear(D_MODEL*3, D_MODEL*2), nn.GELU(),
                                   nn.LayerNorm(D_MODEL*2), nn.Linear(D_MODEL*2, D_MODEL))
        self.C = mk(); self.D = mk(); self.I = mk()
        self.Cg = nn.Sequential(nn.Linear(D_MODEL*2, D_MODEL), nn.GELU(), nn.LayerNorm(D_MODEL))
        self.Dg = nn.Sequential(nn.Linear(D_MODEL*2, D_MODEL), nn.GELU(), nn.LayerNorm(D_MODEL))
        self.Ig = nn.Sequential(nn.Linear(D_MODEL*3, D_MODEL), nn.GELU(), nn.LayerNorm(D_MODEL))
        self.out = make_mlp(D_MODEL*3, D_MODEL)
    def forward(self, v_ord, v_set, op):
        vo = self.oe(op)
        x = torch.cat([v_ord, v_set, vo], dim=-1)
        c = self.C(x); d = self.D(x); i = self.I(x)
        c = self.Cg(torch.cat([c, i], dim=-1))
        d = self.Dg(torch.cat([d, i], dim=-1))
        i = self.Ig(torch.cat([i, c, d], dim=-1))
        return self.out(torch.cat([c, d, i], dim=-1))


class IsisV9(nn.Module):
    def __init__(self):
        super().__init__()
        self.arith_eng = ArithEngine()
        self.order_eng = OrderEngine()
        self.set_eng   = SetEngine()
        self.logic_eng = LogicEngine()
        self.bridge_ao = nn.Sequential(nn.Linear(D_MODEL, D_MODEL), nn.GELU(), nn.LayerNorm(D_MODEL))
        self.bridge_as = nn.Sequential(nn.Linear(D_MODEL, D_MODEL), nn.GELU(), nn.LayerNorm(D_MODEL))
        self.out_head  = nn.Sequential(nn.Linear(D_MODEL, D_MODEL), nn.GELU(), nn.Dropout(0.1), nn.LayerNorm(D_MODEL))
        self.sv = nn.Embedding(N_VALS, D_MODEL)
        nn.init.orthogonal_(self.sv.weight)
    def forward_with_hidden(self, a, b, d, set_bits, s_unk, a_unk, b_unk, d_unk, arith, rel, op):
        c_vec    = self.arith_eng(a, b, a_unk, b_unk, arith)
        c_for_ord = self.bridge_ao(c_vec) + c_vec
        c_for_set = self.bridge_as(c_vec) + c_vec
        ord_vec  = self.order_eng(c_for_ord, d, d_unk, rel)
        set_vec  = self.set_eng(c_for_set, set_bits, s_unk)
        logic_vec = self.logic_eng(ord_vec, set_vec, op)
        return {'arith': c_vec, 'order': ord_vec, 'set': set_vec, 'logic': logic_vec}


# --- Data gen (verbatim from nonlinear_probe.py) ---
def gen_data(seed, n):
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
    c[arith == 0] = torch.clamp(a+b, 0, NUM_RANGE)[arith == 0]
    c[arith == 1] = torch.abs(a-b)[arith == 1]
    c[arith == 2] = torch.clamp(a*b, 0, NUM_RANGE)[arith == 2]
    c[arith == 3] = (a % torch.clamp(b, 1, NUM_RANGE))[arith == 3]
    c = torch.clamp(c, 0, NUM_RANGE)
    c_unk   = au | bu
    ord_unk = c_unk | du
    ord_v = torch.zeros(n, dtype=torch.long, device=DEVICE)
    rt = (((rel==0)&(c>d)) | ((rel==1)&(c<d)) | ((rel==2)&(c==d)) |
          ((rel==3)&(c>=d)) | ((rel==4)&(c<=d)) | ((rel==5)&(c!=d)))
    ord_v[rt] = 1
    val_o = torch.where(ord_unk, torch.tensor(2, device=DEVICE), ord_v)
    sb = torch.randint(0, 2, (n, SET_DIM), dtype=torch.float32, device=DEVICE)
    sou = su | c_unk
    ci = c.clamp(0, SET_DIM-1)
    ins = sb[torch.arange(n, device=DEVICE), ci].bool()
    sv = torch.where(ins, torch.tensor(1, device=DEVICE), torch.tensor(0, device=DEVICE))
    val_s = torch.where(sou, torch.tensor(2, device=DEVICE), sv)
    target = apply_logic(op, val_o, val_s)
    return {
        'a_norm': a.float()/NUM_RANGE, 'b_norm': b.float()/NUM_RANGE, 'd_norm': d.float()/NUM_RANGE,
        'sb': sb, 's_unk': su, 'a_unk': au, 'b_unk': bu, 'd_unk': du,
        'arith': arith, 'rel': rel, 'op': op, 'target': target,
    }


# --- MLP probe ---
class MLPProbe(nn.Module):
    """
    depth=2 is verbatim from nonlinear_probe.py (input -> hidden -> hidden -> out,
    = 2 GELU activations before the output Linear). depth=D generalizes to
      Linear(d_in, d_hidden) -> GELU -> Dropout(0.1)
      [Linear(d_hidden, d_hidden) -> GELU -> Dropout(0.1)] * (depth - 1)
      Linear(d_hidden, n_classes)
    matching the layout of deep_nonlinear_probe.py.
    """
    def __init__(self, d_in=D_MODEL, d_hidden=256, n_classes=3, depth=2):
        super().__init__()
        assert depth >= 1
        layers = [nn.Linear(d_in, d_hidden), nn.GELU(), nn.Dropout(0.1)]
        for _ in range(depth - 1):
            layers += [nn.Linear(d_hidden, d_hidden), nn.GELU(), nn.Dropout(0.1)]
        layers += [nn.Linear(d_hidden, n_classes)]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)


def train_mlp_repro(X_tr, y_tr, X_te, y_te, epochs=40, lr=1e-3, d_hidden=256, depth=2):
    """Test-best-epoch selection (matches nonlinear_probe.py exactly at depth=2)."""
    X_tr_t = torch.from_numpy(X_tr).float().to(DEVICE)
    y_tr_t = torch.from_numpy(y_tr).long().to(DEVICE)
    X_te_t = torch.from_numpy(X_te).float().to(DEVICE)
    y_te_t = torch.from_numpy(y_te).long().to(DEVICE)
    probe = MLPProbe(d_in=X_tr.shape[1], d_hidden=d_hidden, depth=depth).to(DEVICE)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    BATCH = 2048
    n_tr = X_tr_t.shape[0]
    best_te = 0.0
    best_epoch = -1
    for ep in range(epochs):
        probe.train()
        perm = torch.randperm(n_tr, device=DEVICE)
        for i in range(0, n_tr, BATCH):
            idx = perm[i:i+BATCH]
            logits = probe(X_tr_t[idx])
            loss = F.cross_entropy(logits, y_tr_t[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        probe.eval()
        with torch.no_grad():
            pred = probe(X_te_t).argmax(dim=-1)
            te = (pred == y_te_t).float().mean().item()
        if te > best_te:
            best_te = te
            best_epoch = ep
    return {'test_acc': best_te, 'best_epoch_repro': best_epoch}


@torch.no_grad()
def _batched_acc(probe, X_t, y_t, batch=8192):
    probe.eval()
    hits = 0
    for i in range(0, X_t.shape[0], batch):
        pred = probe(X_t[i:i+batch]).argmax(dim=-1)
        hits += (pred == y_t[i:i+batch]).sum().item()
    return hits / X_t.shape[0]


def train_mlp_clean(X_tr, y_tr, X_val, y_val, X_te, y_te, epochs=40, lr=1e-3, d_hidden=256, depth=2):
    """Val-best-epoch selection. Test evaluated ONCE at val-best weights."""
    X_tr_t  = torch.from_numpy(X_tr).float().to(DEVICE)
    y_tr_t  = torch.from_numpy(y_tr).long().to(DEVICE)
    X_val_t = torch.from_numpy(X_val).float().to(DEVICE)
    y_val_t = torch.from_numpy(y_val).long().to(DEVICE)
    X_te_t  = torch.from_numpy(X_te).float().to(DEVICE)
    y_te_t  = torch.from_numpy(y_te).long().to(DEVICE)

    probe = MLPProbe(d_in=X_tr.shape[1], d_hidden=d_hidden, depth=depth).to(DEVICE)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    BATCH = 2048
    n_tr = X_tr_t.shape[0]

    best_val = -1.0
    best_epoch = -1
    best_state = None
    best_train_acc = None

    for ep in range(epochs):
        probe.train()
        perm = torch.randperm(n_tr, device=DEVICE)
        for i in range(0, n_tr, BATCH):
            idx = perm[i:i+BATCH]
            logits = probe(X_tr_t[idx])
            loss = F.cross_entropy(logits, y_tr_t[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        val_acc = _batched_acc(probe, X_val_t, y_val_t)
        if val_acc > best_val:
            best_val = val_acc
            best_epoch = ep
            best_train_acc = _batched_acc(probe, X_tr_t, y_tr_t)
            best_state = {k: v.detach().clone() for k, v in probe.state_dict().items()}

    # Restore val-best weights, evaluate test ONCE
    probe.load_state_dict(best_state)
    test_acc = _batched_acc(probe, X_te_t, y_te_t)
    return {
        'test_acc': test_acc,
        'val_best_epoch': best_epoch,
        'val_acc_at_best_epoch': best_val,
        'train_acc_at_best_epoch': best_train_acc,
    }


# --- SVM probe ---
def svm_repro(X_tr, y_tr, X_te, y_te, C=1.0):
    """Matches reprobe_theia_v2.probe_classification exactly (C=1.0 fixed)."""
    sc = StandardScaler()
    Xs_tr = sc.fit_transform(X_tr); Xs_te = sc.transform(X_te)
    dual = X_tr.shape[0] < X_tr.shape[1]
    clf = LinearSVC(C=C, max_iter=2000, dual=dual)
    clf.fit(Xs_tr, y_tr)
    return {'test_acc': float(clf.score(Xs_te, y_te)), 'C': C}


def svm_clean(X_tr, y_tr, X_val, y_val, X_te, y_te, C_grid=SVM_C_GRID):
    """Sweep C on val. Test evaluated ONCE at val-best C."""
    sc = StandardScaler()
    Xs_tr = sc.fit_transform(X_tr)
    Xs_val = sc.transform(X_val)
    Xs_te = sc.transform(X_te)
    dual = X_tr.shape[0] < X_tr.shape[1]
    val_scores = {}
    best_C = None
    best_val = -1.0
    best_clf = None
    for C in C_grid:
        clf = LinearSVC(C=C, max_iter=2000, dual=dual)
        clf.fit(Xs_tr, y_tr)
        v = float(clf.score(Xs_val, y_val))
        val_scores[str(C)] = v
        if v > best_val:
            best_val = v
            best_C = C
            best_clf = clf
    test_acc = float(best_clf.score(Xs_te, y_te))
    return {
        'test_acc': test_acc,
        'val_best_C': best_C,
        'val_acc_per_C': val_scores,
        'val_acc_at_best_C': best_val,
    }


# --- Run single checkpoint ---
def md5_file(path, block=1 << 20):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(block), b''):
            h.update(chunk)
    return h.hexdigest()


def extract_hidden(model, data, n_samples):
    BATCH = 8192
    hidden = {b: [] for b in BOUNDARIES}
    with torch.no_grad():
        for i in range(0, n_samples, BATCH):
            j = min(i + BATCH, n_samples)
            with torch.amp.autocast('cuda'):
                h = model.forward_with_hidden(
                    data['a_norm'][i:j], data['b_norm'][i:j], data['d_norm'][i:j],
                    data['sb'][i:j], data['s_unk'][i:j],
                    data['a_unk'][i:j], data['b_unk'][i:j], data['d_unk'][i:j],
                    data['arith'][i:j], data['rel'][i:j], data['op'][i:j],
                )
            for k in hidden:
                hidden[k].append(h[k].float().cpu().numpy())
    return {k: np.concatenate(v, axis=0) for k, v in hidden.items()}


def _arr_md5(arr):
    """md5 of a numpy array's bytes — fingerprint for determinism checks."""
    return hashlib.md5(np.ascontiguousarray(arr).tobytes()).hexdigest()


def run_single(checkpoint_path, protocol, depths=(2,), hidden=256, split_mode='random',
               n_samples=50000, data_seed=999, split_seed=0, dry_run=False):
    depths = tuple(depths)
    multi_depth = (depths != (2,)) or (hidden != 256)

    print(f"\n{'='*72}")
    print(f"  probe_cleansplit.py  protocol={protocol}  depths={list(depths)}  "
          f"hidden={hidden}  split_mode={split_mode}" + ("  [DRY-RUN]" if dry_run else ""))
    print(f"  checkpoint: {checkpoint_path}")
    print(f"{'='*72}")
    t_start = time.time()

    if split_mode == 'stratified':
        raise NotImplementedError(
            "stratified split not supported in probe_cleansplit.py; see deep_nonlinear_probe.py")
    if split_mode != 'random':
        raise ValueError(f"unknown split_mode: {split_mode}")

    model = IsisV9().to(DEVICE)
    state = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    ckpt_md5 = md5_file(checkpoint_path)
    print(f"  md5: {ckpt_md5}")

    # Gen data + hidden (ONCE, reused across all depths and boundaries)
    print(f"  Generating {n_samples} samples (seed {data_seed}) ...")
    data = gen_data(data_seed, n_samples)
    hidden_states = extract_hidden(model, data, n_samples)
    target = data['target'].cpu().numpy()
    shapes_str = {k: tuple(v.shape) for k, v in hidden_states.items()}
    print(f"  Hidden shapes: {shapes_str}")

    # Split
    perm = np.random.RandomState(split_seed).permutation(n_samples)
    if protocol == 'reproduce_70_30':
        n_tr = int(n_samples * 0.7)
        tr, te = perm[:n_tr], perm[n_tr:]
        val = None
        split_str = "70_30"
        selection = {'mlp': 'test_best_epoch', 'svm': 'fixed_C=1.0'}
    elif protocol == 'clean_split_v1':
        n_tr  = int(n_samples * 0.6)
        n_val = int(n_samples * 0.2)
        tr  = perm[:n_tr]
        val = perm[n_tr:n_tr+n_val]
        te  = perm[n_tr+n_val:]
        split_str = "60_20_20"
        selection = {'mlp': 'val_best_epoch', 'svm': 'val_best_C'}
    else:
        raise ValueError(f"Unknown protocol: {protocol}")
    print(f"  Split ({split_str}, {split_mode}): tr={len(tr)}, "
          f"val={0 if val is None else len(val)}, te={len(te)}")

    # Dry-run: dump determinism fingerprint for first cell and exit
    if dry_run:
        print(f"\n{'='*72}")
        print(f"  DRY-RUN FINGERPRINT — first cell = (boundary=arith, seed={os.path.basename(os.path.dirname(checkpoint_path))})")
        print(f"{'='*72}")
        # Data determinism
        data_concat = np.concatenate([
            data['a_norm'].cpu().numpy().reshape(-1, 1),
            data['b_norm'].cpu().numpy().reshape(-1, 1),
            data['d_norm'].cpu().numpy().reshape(-1, 1),
            data['arith'].cpu().numpy().reshape(-1, 1).astype(np.float32),
            data['rel'].cpu().numpy().reshape(-1, 1).astype(np.float32),
            data['op'].cpu().numpy().reshape(-1, 1).astype(np.float32),
            data['a_unk'].cpu().numpy().reshape(-1, 1).astype(np.float32),
            data['b_unk'].cpu().numpy().reshape(-1, 1).astype(np.float32),
            data['d_unk'].cpu().numpy().reshape(-1, 1).astype(np.float32),
            data['s_unk'].cpu().numpy().reshape(-1, 1).astype(np.float32),
            target.reshape(-1, 1).astype(np.float32),
        ], axis=1)
        print(f"  data md5 (all 50k × 11 fields): {_arr_md5(data_concat)}")
        print(f"  data md5 (first 10 rows only):  {_arr_md5(data_concat[:10])}")
        print()
        print(f"  First 10 raw samples (a*20, b*20, d*20, arith, rel, op, au, bu, du, su, target):")
        for i in range(10):
            a_i = int(data['a_norm'][i].item() * NUM_RANGE + 0.5)
            b_i = int(data['b_norm'][i].item() * NUM_RANGE + 0.5)
            d_i = int(data['d_norm'][i].item() * NUM_RANGE + 0.5)
            ar = int(data['arith'][i].item())
            re = int(data['rel'][i].item())
            op = int(data['op'][i].item())
            au = int(data['a_unk'][i].item())
            bu = int(data['b_unk'][i].item())
            du = int(data['d_unk'][i].item())
            su = int(data['s_unk'][i].item())
            tg = int(target[i])
            print(f"    [{i:2d}] a={a_i:2d} b={b_i:2d} d={d_i:2d} "
                  f"arith={ar} rel={re} op={op}  "
                  f"au={au} bu={bu} du={du} su={su}  target={tg}")
        print()
        # Hidden state determinism (arith boundary, first 10 train samples)
        X_arith = hidden_states['arith']
        first10_tr_idx = tr[:10]
        print(f"  First 10 train indices (permutation[0:10]): {first10_tr_idx.tolist()}")
        print(f"  First 10 val indices:   {(val[:10].tolist() if val is not None else '(no val in 70_30)')}")
        print(f"  First 10 test indices:  {te[:10].tolist()}")
        print()
        print(f"  hidden['arith'] full md5 (50000 × 128): {_arr_md5(X_arith)}")
        print(f"  hidden['arith'][first_10_tr] md5:       {_arr_md5(X_arith[first10_tr_idx])}")
        print(f"  hidden['arith'][tr[0]] vec[:8] (f32):   "
              f"{[round(float(v), 6) for v in X_arith[first10_tr_idx[0], :8]]}")
        print()
        # MLP arch preview
        from io import StringIO
        for D in depths:
            probe = MLPProbe(d_in=D_MODEL, d_hidden=hidden, depth=D)
            n_params = sum(p.numel() for p in probe.parameters())
            print(f"  MLPProbe(depth={D}, hidden={hidden}): params = {n_params:,}")
        print()
        print(f"  [DRY-RUN] No training performed. Exit.")
        return None

    # Run probes per boundary × depth
    boundary_results = {}
    for b in BOUNDARIES:
        X = hidden_states[b]
        print(f"\n  [{b}]")

        # ---- MLP per depth ----
        mlp_by_depth = {}
        for D in depths:
            t0 = time.time()
            if protocol == 'reproduce_70_30':
                mlp_res = train_mlp_repro(X[tr], target[tr], X[te], target[te],
                                          d_hidden=hidden, depth=D)
            else:
                mlp_res = train_mlp_clean(X[tr], target[tr], X[val], target[val],
                                          X[te], target[te], d_hidden=hidden, depth=D)
            mlp_res['elapsed_s'] = round(time.time() - t0, 2)
            print(f"    MLP d={D} test_acc = {mlp_res['test_acc']:.4f}  "
                  f"({mlp_res['elapsed_s']:.1f}s)")
            mlp_by_depth[f'd{D}'] = mlp_res

        # ---- SVM (depth-independent; only run in single-depth-2 mode to avoid duplication) ----
        svm_res = None
        if not multi_depth:
            t0 = time.time()
            if protocol == 'reproduce_70_30':
                svm_res = svm_repro(X[tr], target[tr], X[te], target[te])
            else:
                svm_res = svm_clean(X[tr], target[tr], X[val], target[val], X[te], target[te])
            svm_res['elapsed_s'] = round(time.time() - t0, 2)
            print(f"    SVM test_acc = {svm_res['test_acc']:.4f}  ({svm_res['elapsed_s']:.1f}s)")

        if multi_depth:
            boundary_results[b] = {'mlp': mlp_by_depth}
        else:
            # Backward-compatible flat schema for the single-d2 default
            boundary_results[b] = {'mlp': mlp_by_depth['d2'], 'svm': svm_res}

    total_s = round(time.time() - t_start, 2)
    out = {
        'protocol_version': protocol,
        'split': split_str,
        'split_mode': split_mode,
        'selection': selection,
        'checkpoint': checkpoint_path.replace('\\', '/'),
        'checkpoint_md5': ckpt_md5,
        'data_seed': data_seed,
        'split_seed': split_seed,
        'n_samples': n_samples,
        'depths': list(depths),
        'hidden': hidden,
        'mlp_arch_template': (f'Linear({D_MODEL},{hidden})-GELU-Dropout(0.1)'
                              f'-[Linear({hidden},{hidden})-GELU-Dropout(0.1)]x(depth-1)'
                              f'-Linear({hidden},3)'),
        'mlp_hparams': {'lr': 1e-3, 'epochs': 40, 'batch': 2048, 'weight_decay': 0.01,
                        'optimizer': 'AdamW', 'schedule': 'CosineAnnealingLR'},
        'svm_grid': (SVM_C_GRID if protocol == 'clean_split_v1' else [1.0]) if not multi_depth else None,
        'boundaries': boundary_results,
        'total_elapsed_s': total_s,
    }
    seed_dir = os.path.dirname(checkpoint_path)
    if multi_depth:
        depths_tag = '-'.join(str(d) for d in depths)
        out_name = f'probe_cleansplit_deepmlp_h{hidden}_d{depths_tag}.json'
    else:
        out_name = 'probe_cleansplit.json'
    out_path = os.path.join(seed_dir, out_name)
    if os.path.exists(out_path) and not multi_depth:
        existing = json.load(open(out_path))
        if existing.get('protocol_version') != protocol:
            suffix = f".{protocol}"
            out_path = out_path.replace('.json', f'{suffix}.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved: {out_path}  (total {total_s:.1f}s)")
    return out


# --- Sanity check ---
def sanity_check():
    """Run reproduce_70_30 on seed 42, compare to nonlinear_probe.json and
    mechanistic_probing_v2.json. PASS = all cells match within ±0.001."""
    ckpt = SEED_PATHS[42]
    print(f"\n[sanity] Running reproduce_70_30 on seed 42 ...")
    new = run_single(ckpt, protocol='reproduce_70_30')

    ckpt_dir = os.path.dirname(ckpt)
    nl_path  = os.path.join(ckpt_dir, 'nonlinear_probe.json')
    mp_path  = os.path.join(ckpt_dir, 'mechanistic_probing_v2.json')
    ref_nl = json.load(open(nl_path))
    ref_mp = json.load(open(mp_path))

    print(f"\n{'='*72}")
    print(f"  SANITY CHECK  (tolerance ±0.001)")
    print(f"{'='*72}")
    print(f"  {'boundary':<8} {'probe':<4} {'new':>10} {'ref':>10} {'diff':>10} {'status':>6}")
    all_pass = True
    for b in BOUNDARIES:
        # MLP
        new_mlp = new['boundaries'][b]['mlp']['test_acc']
        ref_mlp = ref_nl['mlp_probe_acc'][b]
        diff = new_mlp - ref_mlp
        ok = abs(diff) <= 0.001
        all_pass = all_pass and ok
        print(f"  {b:<8} {'MLP':<4} {new_mlp:>10.4f} {ref_mlp:>10.4f} {diff:>+10.4f} {'PASS' if ok else 'FAIL':>6}")
        # SVM
        new_svm = new['boundaries'][b]['svm']['test_acc']
        ref_svm = ref_mp['probes'][b]['final_verdict_acc']
        diff = new_svm - ref_svm
        ok = abs(diff) <= 0.001
        all_pass = all_pass and ok
        print(f"  {b:<8} {'SVM':<4} {new_svm:>10.4f} {ref_svm:>10.4f} {diff:>+10.4f} {'PASS' if ok else 'FAIL':>6}")

    print(f"\n  Overall: {'PASS — safe to run clean_split_v1' if all_pass else 'FAIL — investigate before proceeding'}")
    return all_pass


# --- Aggregate ---
def _mean_std(xs):
    xs = [x for x in xs if x is not None]
    if not xs: return (None, None)
    m = sum(xs) / len(xs)
    if len(xs) == 1: return (m, 0.0)
    v = sum((x - m) ** 2 for x in xs) / len(xs)
    return (m, v ** 0.5)


def aggregate(deep_file_tag='h256_d4-6'):
    # Load d=2 clean_split_v1 outputs (flat schema with 'mlp' + 'svm')
    new_data = {}
    for seed, ckpt in SEED_PATHS.items():
        seed_dir = os.path.dirname(ckpt)
        candidates = [
            os.path.join(seed_dir, 'probe_cleansplit.json'),
            os.path.join(seed_dir, f'probe_cleansplit.{PROTOCOL_VERSION}.json'),
        ]
        found = False
        for j in candidates:
            if not os.path.exists(j):
                continue
            d = json.load(open(j))
            if d.get('protocol_version') == PROTOCOL_VERSION:
                new_data[seed] = d
                found = True
                break
        if not found:
            print(f"  [warn] seed {seed}: no d=2 {PROTOCOL_VERSION} JSON in {seed_dir}")

    # Load deep MLP outputs (nested schema with 'mlp': {'d4': ..., 'd6': ...})
    deep_data = {}
    if deep_file_tag:
        for seed, ckpt in SEED_PATHS.items():
            seed_dir = os.path.dirname(ckpt)
            deep_path = os.path.join(seed_dir, f'probe_cleansplit_deepmlp_{deep_file_tag}.json')
            if os.path.exists(deep_path):
                deep_data[seed] = json.load(open(deep_path))

    # Load old references
    old_nl = {}
    old_svm = {}
    for seed, ckpt in SEED_PATHS.items():
        nlp = os.path.join(os.path.dirname(ckpt), 'nonlinear_probe.json')
        mpp = os.path.join(os.path.dirname(ckpt), 'mechanistic_probing_v2.json')
        if os.path.exists(nlp):
            old_nl[seed] = json.load(open(nlp))
        if os.path.exists(mpp):
            old_svm[seed] = json.load(open(mpp))

    print(f"  Loaded {len(new_data)}/5 clean_split_v1, "
          f"{len(old_nl)}/5 old MLP, {len(old_svm)}/5 old SVM")

    # Aggregate
    agg = {}  # agg[probe][boundary] = (mean, std)
    for probe in ('mlp', 'svm'):
        agg[probe] = {}
        for b in BOUNDARIES:
            vals = [d['boundaries'][b][probe]['test_acc'] for d in new_data.values()]
            agg[probe][b] = _mean_std(vals)

    old_mlp_agg = {b: _mean_std([d['mlp_probe_acc'][b] for d in old_nl.values()]) for b in BOUNDARIES}
    old_svm_agg = {b: _mean_std([d['probes'][b]['final_verdict_acc'] for d in old_svm.values()]) for b in BOUNDARIES}

    # Report
    from datetime import datetime
    L = []
    A = L.append
    A("# Clean-Split Probe Report (60/20/20, val-best selection)")
    A("")
    A(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    A(f"Protocol: `{PROTOCOL_VERSION}` — 60/20/20 split, MLP=val-best-epoch, SVM=val-best-C in {{0.1, 1.0, 10.0}}.")
    A(f"Reference baselines: `nonlinear_probe.json` (70/30, test-best-epoch MLP), "
      f"`mechanistic_probing_v2.json` (70/30, C=1.0 SVM).")
    A("")
    A("## H2(a) ceiling check (claim-critical, first line by design)")
    A("")
    a_svm_m, _ = agg['svm']['arith']
    a_svm_old, _ = old_svm_agg['arith']
    A(f"**Arith Linear SVM new vs old**: {a_svm_m:.3f} (clean-split) vs {a_svm_old:.3f} (70/30 old). "
      f"Uncertainty-only ceiling: {H2A_CEILING:.2f}. "
      f"Status: {'**ALERT — exceeds ceiling, H2(a) claim at risk**' if a_svm_m >= H2A_CEILING else 'below ceiling ✓'}")
    A("")
    A("### 4-cell safety check (all must be < 0.74)")
    A("")
    A("| Cell | Value | Ceiling | Status |")
    A("|---|---|---|---|")
    cells = [
        ('Arith Linear SVM', agg['svm']['arith'][0]),
        ('Arith MLP',        agg['mlp']['arith'][0]),
        ('Order MLP',        agg['mlp']['order'][0]),
        ('Set MLP',          agg['mlp']['set'][0]),
    ]
    any_alert = False
    for name, v in cells:
        ok = v < H2A_CEILING
        if not ok: any_alert = True
        A(f"| {name} | {v:.4f} | {H2A_CEILING} | {'✓' if ok else '**ALERT**'} |")
    A("")
    if any_alert:
        A("**⚠️  AT LEAST ONE CELL EXCEEDS 74% — STOP and review H2(a) before submitting.**")
    else:
        A("All 4 cells below 74% ceiling — H2(a) claim intact under clean split.")
    A("")

    # --- Depth sweep — paper-matched width (hidden=256) ---
    if deep_data:
        # Collect per-seed per-boundary per-depth test accs
        # d=2 from new_data (flat schema under ['mlp']['test_acc'])
        # d>2 from deep_data (nested schema under ['mlp'][f'd{D}']['test_acc'])
        available_deep_depths = set()
        for d in deep_data.values():
            for b_key in BOUNDARIES:
                available_deep_depths.update(d['boundaries'][b_key]['mlp'].keys())
        deep_depths_sorted = sorted(
            [int(k[1:]) for k in available_deep_depths if k.startswith('d')])
        all_depths = [2] + deep_depths_sorted  # e.g. [2, 4, 6]

        # Aggregate per (boundary, depth)
        depth_agg = {}
        for b in BOUNDARIES:
            depth_agg[b] = {}
            # d=2 from new_data
            d2_vals = [new_data[s]['boundaries'][b]['mlp']['test_acc']
                       for s in new_data if s in new_data]
            depth_agg[b][2] = _mean_std(d2_vals)
            # d>2 from deep_data
            for D in deep_depths_sorted:
                vals = []
                for s, dd in deep_data.items():
                    cell = dd['boundaries'][b]['mlp'].get(f'd{D}')
                    if cell is not None:
                        vals.append(cell['test_acc'])
                depth_agg[b][D] = _mean_std(vals)

        # Header
        A("## Depth sweep — paper-matched width (hidden=256, random 60/20/20, val-best-epoch)")
        A("")
        sample_deep = next(iter(deep_data.values())) if deep_data else None
        A(f"Seeds: {sorted(deep_data.keys())}. "
          f"Arch: `{sample_deep['mlp_arch_template'] if sample_deep else 'N/A'}`. "
          f"Hparams: AdamW lr=1e-3 wd=0.01, cosine, 40 epochs, batch 2048.")
        A("")
        A("### Depth sweep main table (verdict, 3-class, 5-seed mean ± std, %)")
        A("")
        header = "| Boundary |" + "|".join(f" d={D} @ w=256 " for D in all_depths) + "|"
        sep    = "|---" + "|---" * len(all_depths) + "|"
        A(header)
        A(sep)
        for b in BOUNDARIES:
            row = f"| {b} |"
            for D in all_depths:
                m, s = depth_agg[b][D]
                if m is None:
                    row += " -- |"
                else:
                    row += f" {m*100:.3f} ± {s*100:.3f} |"
            A(row)
        A("")

        # Max upstream across all depths
        A("### Gate-A margin across depth sweep")
        A("")
        upstream_maxes = []
        for b in ['arith', 'order', 'set']:
            for D in all_depths:
                m, _ = depth_agg[b][D]
                if m is not None:
                    upstream_maxes.append((m, b, D))
        if upstream_maxes:
            mx_acc, mx_b, mx_D = max(upstream_maxes, key=lambda x: x[0])
            margin_72 = (0.72 - mx_acc) * 100
            margin_74 = (H2A_CEILING - mx_acc) * 100
            A(f"- **Max upstream (Arith/Order/Set) across d ∈ {{{','.join(str(D) for D in all_depths)}}}**: "
              f"**{mx_acc*100:.3f}%** at {mx_b} @ d={mx_D}")
            A(f"- Margin to 72% threshold: **{margin_72:.2f}pp** "
              f"({'PASS' if mx_acc < 0.72 else 'FAIL'})")
            A(f"- Margin to 74% ceiling:   **{margin_74:.2f}pp** "
              f"({'PASS' if mx_acc < H2A_CEILING else 'FAIL'})")
            A("")

        # Depth-only delta (d=max − d=2 at same width)
        A("### Depth-only delta (d=max − d=2, same width=256)")
        A("")
        A("| Boundary | d=2 | d=max | Δ |")
        A("|---|---|---|---|")
        d_max = max(all_depths)
        for b in BOUNDARIES:
            m2, _ = depth_agg[b][2]
            mm, _ = depth_agg[b][d_max]
            if m2 is not None and mm is not None:
                A(f"| {b} | {m2*100:.3f} | {mm*100:.3f} | {(mm-m2)*100:+.3f}pp |")
        A("")

        # Cross-reference to deep probe (width=512)
        A("### Cross-reference to prior deep probe run (hidden=512, stratified 60/20/20)")
        A("")
        A("Source: `deep_nonlinear_probe_results.json` (2026-04-19 15:29). "
          "Width=512 mean values shown in parentheses.")
        A("")
        A("| Boundary | d=2 @ w=256 | d=2 @ w=512 | d=4 @ w=256 | d=4 @ w=512 | d=6 @ w=256 | d=6 @ w=512 |")
        A("|---|---|---|---|---|---|---|")
        # Load deep probe reference
        deep_ref_path = r'deep_nonlinear_probe_results.json'
        w512 = {}  # w512[b][D] = mean
        if os.path.exists(deep_ref_path):
            ref = json.load(open(deep_ref_path))
            for b in BOUNDARIES:
                w512[b] = {}
                for D in [2, 4, 6]:
                    vals = []
                    for se in ref['per_seed']:
                        k = f"{b}__verdict__d{D}"
                        if k in se['per_cell']:
                            vals.append(se['per_cell'][k]['te_acc_at_val_best'])
                    if vals:
                        w512[b][D] = sum(vals) / len(vals)
        for b in BOUNDARIES:
            cells = []
            for D in [2, 4, 6]:
                m256, _ = depth_agg[b].get(D, (None, None))
                m512 = w512.get(b, {}).get(D)
                cells.append(f"{m256*100:.3f}" if m256 is not None else "--")
                cells.append(f"{m512*100:.3f}" if m512 is not None else "--")
            A(f"| {b} | " + " | ".join(cells) + " |")
        A("")

    # Head-to-head
    A("## Head-to-head comparison (final verdict, 3-class, mean ± std over 5 seeds)")
    A("")
    A("### MLP (2-layer, 256 hidden)")
    A("")
    A("| Boundary | 70/30 old (test-best-ep) | 60/20/20 new (val-best-ep) | Δ (new − old) |")
    A("|---|---|---|---|")
    for b in BOUNDARIES:
        om, os_ = old_mlp_agg[b]
        nm, ns = agg['mlp'][b]
        if om is None or nm is None:
            A(f"| {b} | -- | -- | -- |")
        else:
            A(f"| {b} | {om:.4f} ± {os_:.4f} | {nm:.4f} ± {ns:.4f} | {(nm-om)*100:+.2f}pp |")
    A("")
    A("### Linear SVM")
    A("")
    A("| Boundary | 70/30 old (C=1.0) | 60/20/20 new (val-best-C) | Δ (new − old) |")
    A("|---|---|---|---|")
    for b in BOUNDARIES:
        om, os_ = old_svm_agg[b]
        nm, ns = agg['svm'][b]
        if om is None or nm is None:
            A(f"| {b} | -- | -- | -- |")
        else:
            A(f"| {b} | {om:.4f} ± {os_:.4f} | {nm:.4f} ± {ns:.4f} | {(nm-om)*100:+.2f}pp |")
    A("")

    # Checkpoint table
    A("## Checkpoints (seed → path → md5)")
    A("")
    A("| Seed | Path | MD5 |")
    A("|---|---|---|")
    for seed, d in sorted(new_data.items()):
        A(f"| {seed} | `{d['checkpoint']}` | `{d['checkpoint_md5']}` |")
    A("")

    # Per-seed detail
    A("## Per-seed detail")
    A("")
    A("### Test acc (clean_split_v1)")
    A("")
    A("| Seed | Probe | Arith | Order | Set | Logic |")
    A("|---|---|---|---|---|---|")
    for seed in sorted(new_data):
        d = new_data[seed]
        for probe in ('mlp', 'svm'):
            row = [f"{d['boundaries'][b][probe]['test_acc']:.4f}" for b in BOUNDARIES]
            A(f"| {seed} | {probe.upper()} | " + " | ".join(row) + " |")
    A("")
    A("### MLP val-best-epoch + train acc at best (diagnostic)")
    A("")
    A("| Seed | Boundary | val_best_ep | val@best | train@best | test |")
    A("|---|---|---|---|---|---|")
    for seed in sorted(new_data):
        d = new_data[seed]
        for b in BOUNDARIES:
            r = d['boundaries'][b]['mlp']
            A(f"| {seed} | {b} | {r.get('val_best_epoch')} | "
              f"{r.get('val_acc_at_best_epoch'):.4f} | "
              f"{r.get('train_acc_at_best_epoch'):.4f} | "
              f"{r['test_acc']:.4f} |")
    A("")
    A("### SVM val-best-C (diagnostic)")
    A("")
    A("| Seed | Boundary | best_C | val@0.1 | val@1.0 | val@10.0 | test |")
    A("|---|---|---|---|---|---|---|")
    for seed in sorted(new_data):
        d = new_data[seed]
        for b in BOUNDARIES:
            r = d['boundaries'][b]['svm']
            vpc = r.get('val_acc_per_C', {})
            A(f"| {seed} | {b} | {r.get('val_best_C')} | "
              f"{vpc.get('0.1', 0):.4f} | {vpc.get('1.0', 0):.4f} | {vpc.get('10.0', 0):.4f} | "
              f"{r['test_acc']:.4f} |")
    A("")

    os.makedirs(REPORT_DIR, exist_ok=True)
    out_path = os.path.join(REPORT_DIR, 'probe_cleansplit_report.md')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L))
    print(f"\n  Report: {out_path}")
    # Print the leading alert lines to stdout for visibility.
    # Replace unicode that Windows GBK stdout can't encode.
    preview = '\n'.join(L[:40]).encode('ascii', 'replace').decode('ascii')
    print()
    print(preview)


# --- CLI ---
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=str, default=None)
    p.add_argument('--protocol', type=str, default='clean_split_v1',
                   choices=['reproduce_70_30', 'clean_split_v1'])
    p.add_argument('--n-samples', type=int, default=50000)
    p.add_argument('--data-seed', type=int, default=999)
    p.add_argument('--split-seed', type=int, default=0)
    p.add_argument('--depths', type=int, nargs='+', default=[2],
                   help='MLP probe depths to run (default: [2], matching paper).')
    p.add_argument('--hidden', type=int, default=256,
                   help='MLP probe hidden width (default: 256, matching paper).')
    p.add_argument('--split', type=str, default='random', choices=['random', 'stratified'],
                   help='Split method. Only "random" supported here; stratified is in deep_nonlinear_probe.py.')
    p.add_argument('--seeds', type=int, nargs='+', default=None,
                   help='Subset of seeds to run. Default: all 5 (42, 123, 256, 777, 999).')
    p.add_argument('--dry-run', action='store_true',
                   help='Do data gen + hidden extraction only; print fingerprint; no training.')
    p.add_argument('--sanity', action='store_true',
                   help='Run reproduce_70_30 on seed 42 and compare to existing JSONs (tol ±0.001)')
    p.add_argument('--aggregate', action='store_true',
                   help='Aggregate clean_split_v1 outputs across 5 seeds and write report')
    p.add_argument('--all-seeds', action='store_true',
                   help='(Legacy) same as --seeds 42 123 256 777 999')
    args = p.parse_args()

    if args.sanity:
        ok = sanity_check()
        sys.exit(0 if ok else 1)
    if args.aggregate:
        aggregate()
        return

    # Resolve seed list
    if args.seeds is not None:
        seeds_to_run = [s for s in args.seeds if s in SEED_PATHS]
        missing = set(args.seeds) - set(seeds_to_run)
        if missing:
            print(f"[warn] unknown seeds ignored: {missing}")
    elif args.all_seeds or args.checkpoint is None:
        seeds_to_run = list(SEED_PATHS.keys())
    else:
        seeds_to_run = None  # single --checkpoint path below

    common = dict(
        protocol=args.protocol,
        depths=args.depths,
        hidden=args.hidden,
        split_mode=args.split,
        n_samples=args.n_samples,
        data_seed=args.data_seed,
        split_seed=args.split_seed,
        dry_run=args.dry_run,
    )

    if args.checkpoint:
        run_single(args.checkpoint, **common)
        return

    if seeds_to_run:
        for seed in seeds_to_run:
            ckpt = SEED_PATHS[seed]
            print(f"\n### SEED {seed} ###")
            run_single(ckpt, **common)
            if args.dry_run:
                print(f"\n[DRY-RUN] Stopped after seed {seed}. Other seeds skipped to keep output short.")
                break
        return

    p.print_help()


if __name__ == '__main__':
    main()
