#!/usr/bin/env python
"""
Nonlinear (MLP) probe on THEIA boundary hidden states (§5.3 / Table 7).

2-hidden-layer MLP (256 hidden, GELU, Dropout 0.1) per boundary, predicting the
3-class final verdict; compared against the linear SVM probe from
reprobe_theia_v2. Legacy protocol: 50K samples (data_seed=999), 70/30 split,
test-best-epoch selection — superseded by probe_cleansplit.py's clean split.
Decision: upstream MLP accuracy near the linear level (~60-70%) supports a
representational delayed verdict; >95% would mean the information is present
but nonlinearly encoded.

Usage:
    python nonlinear_probe.py --checkpoint <path>
    python nonlinear_probe.py --aggregate
"""
import argparse, json, os, time
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np

p = argparse.ArgumentParser()
p.add_argument('--checkpoint', type=str, default=None)
p.add_argument('--n-samples', type=int, default=50000)
p.add_argument('--data-seed', type=int, default=999)
p.add_argument('--probe-epochs', type=int, default=40)
p.add_argument('--probe-lr', type=float, default=1e-3)
p.add_argument('--probe-hidden', type=int, default=256)
p.add_argument('--aggregate', action='store_true')
args = p.parse_args()

ROOT = r'multi_seed_results'
REPORT_DIR = os.path.join(ROOT, 'reports')
os.makedirs(REPORT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
D_MODEL=128; NUM_RANGE=20; SET_DIM=21; P_UNKNOWN=0.15
VAL_FALSE=0; VAL_TRUE=1; VAL_UNKNOWN=2; N_VALS=3
N_RELS=6; N_ARITH=4; N_OPS=5
ARITH_ADD=0; ARITH_SUB=1; ARITH_MUL=2; ARITH_MOD=3
OP_AND=0; OP_OR=1; OP_NOT=2; OP_IMPLIES=3; OP_IFF=4

AND_T=torch.tensor([[0,0,0],[0,1,2],[0,2,2]],dtype=torch.long,device=DEVICE)
OR_T =torch.tensor([[0,1,2],[1,1,1],[2,1,2]],dtype=torch.long,device=DEVICE)
IMP_T=torch.tensor([[1,1,1],[0,1,2],[2,1,2]],dtype=torch.long,device=DEVICE)
IFF_T=torch.tensor([[1,0,2],[0,1,2],[2,2,2]],dtype=torch.long,device=DEVICE)
NOT_T=torch.tensor([1,0,2],dtype=torch.long,device=DEVICE)

def apply_logic(op, va, vb):
    r = torch.zeros_like(op)
    m=op==OP_AND;     r[m]=AND_T[va[m],vb[m]]
    m=op==OP_OR;      r[m]=OR_T[va[m],vb[m]]
    m=op==OP_IMPLIES; r[m]=IMP_T[va[m],vb[m]]
    m=op==OP_IFF;     r[m]=IFF_T[va[m],vb[m]]
    m=op==OP_NOT;     r[m]=NOT_T[va[m]]
    return r

# === Model (verbatim from reprobe_theia_v2) ===
class NumEnc(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(1,D_MODEL//2),nn.GELU(),nn.Linear(D_MODEL//2,D_MODEL))
        self.unknown_vec = nn.Parameter(torch.randn(D_MODEL))
    def forward(self, x, is_unknown):
        v = self.f(x.unsqueeze(-1))
        unk = self.unknown_vec.unsqueeze(0).expand_as(v)
        return torch.where(is_unknown.unsqueeze(-1), unk, v)

class SetEnc(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(SET_DIM,D_MODEL//2),nn.GELU(),nn.Linear(D_MODEL//2,D_MODEL))
        self.unknown_vec = nn.Parameter(torch.randn(D_MODEL))
    def forward(self, bits, is_unknown):
        v = self.f(bits)
        unk = self.unknown_vec.unsqueeze(0).expand_as(v)
        return torch.where(is_unknown.unsqueeze(-1), unk, v)

def make_mlp(in_d, out_d, dropout=0.0):
    layers = [nn.Linear(in_d,in_d*2),nn.GELU(),nn.LayerNorm(in_d*2),nn.Linear(in_d*2,out_d)]
    if dropout > 0: layers.insert(3, nn.Dropout(dropout))
    return nn.Sequential(*layers)

class ArithEngine(nn.Module):
    def __init__(self):
        super().__init__()
        self.ne = NumEnc(); self.ae = nn.Embedding(N_ARITH, D_MODEL)
        self.net = make_mlp(D_MODEL*3, D_MODEL)
    def forward(self, a, b, a_unk, b_unk, arith):
        return self.net(torch.cat([self.ne(a,a_unk), self.ne(b,b_unk), self.ae(arith)], dim=-1))

class OrderEngine(nn.Module):
    def __init__(self):
        super().__init__()
        self.ne = NumEnc(); self.re = nn.Embedding(N_RELS, D_MODEL)
        mk = lambda: nn.Sequential(nn.Linear(D_MODEL*3,D_MODEL*2),nn.GELU(),
                                   nn.LayerNorm(D_MODEL*2),nn.Linear(D_MODEL*2,D_MODEL))
        self.G=mk(); self.L=mk(); self.E=mk()
        self.Gg=nn.Sequential(nn.Linear(D_MODEL*2,D_MODEL),nn.GELU(),nn.LayerNorm(D_MODEL))
        self.Lg=nn.Sequential(nn.Linear(D_MODEL*2,D_MODEL),nn.GELU(),nn.LayerNorm(D_MODEL))
        self.Eg=nn.Sequential(nn.Linear(D_MODEL*3,D_MODEL),nn.GELU(),nn.LayerNorm(D_MODEL))
        self.out = make_mlp(D_MODEL*4, D_MODEL)
    def forward(self, c_vec, d, d_unk, rel):
        vd = self.ne(d, d_unk); vr = self.re(rel)
        x = torch.cat([c_vec,vd,vr],dim=-1)
        g=self.G(x); l=self.L(x); e=self.E(x)
        g=self.Gg(torch.cat([g,e],dim=-1))
        l=self.Lg(torch.cat([l,e],dim=-1))
        e=self.Eg(torch.cat([e,g,l],dim=-1))
        return self.out(torch.cat([g,l,e,c_vec], dim=-1))

class SetEngine(nn.Module):
    def __init__(self):
        super().__init__()
        self.se = SetEnc(); self.net = make_mlp(D_MODEL*2, D_MODEL)
    def forward(self, c_vec, set_bits, s_unk):
        return self.net(torch.cat([c_vec, self.se(set_bits,s_unk)], dim=-1))

class LogicEngine(nn.Module):
    def __init__(self):
        super().__init__()
        self.oe = nn.Embedding(N_OPS, D_MODEL)
        mk = lambda: nn.Sequential(nn.Linear(D_MODEL*3,D_MODEL*2),nn.GELU(),
                                   nn.LayerNorm(D_MODEL*2),nn.Linear(D_MODEL*2,D_MODEL))
        self.C=mk(); self.D=mk(); self.I=mk()
        self.Cg=nn.Sequential(nn.Linear(D_MODEL*2,D_MODEL),nn.GELU(),nn.LayerNorm(D_MODEL))
        self.Dg=nn.Sequential(nn.Linear(D_MODEL*2,D_MODEL),nn.GELU(),nn.LayerNorm(D_MODEL))
        self.Ig=nn.Sequential(nn.Linear(D_MODEL*3,D_MODEL),nn.GELU(),nn.LayerNorm(D_MODEL))
        self.out = make_mlp(D_MODEL*3, D_MODEL)
    def forward(self, v_ord, v_set, op):
        vo = self.oe(op)
        x = torch.cat([v_ord,v_set,vo], dim=-1)
        c=self.C(x); d=self.D(x); i=self.I(x)
        c=self.Cg(torch.cat([c,i],dim=-1))
        d=self.Dg(torch.cat([d,i],dim=-1))
        i=self.Ig(torch.cat([i,c,d],dim=-1))
        return self.out(torch.cat([c,d,i], dim=-1))

class IsisV9(nn.Module):
    def __init__(self):
        super().__init__()
        self.arith_eng = ArithEngine()
        self.order_eng = OrderEngine()
        self.set_eng = SetEngine()
        self.logic_eng = LogicEngine()
        self.bridge_ao = nn.Sequential(nn.Linear(D_MODEL,D_MODEL),nn.GELU(),nn.LayerNorm(D_MODEL))
        self.bridge_as = nn.Sequential(nn.Linear(D_MODEL,D_MODEL),nn.GELU(),nn.LayerNorm(D_MODEL))
        self.out_head = nn.Sequential(nn.Linear(D_MODEL,D_MODEL),nn.GELU(),nn.Dropout(0.1),nn.LayerNorm(D_MODEL))
        self.sv = nn.Embedding(N_VALS, D_MODEL)
        nn.init.orthogonal_(self.sv.weight)
    def forward_with_hidden(self, a,b,d,set_bits,s_unk,a_unk,b_unk,d_unk,arith,rel,op):
        c_vec = self.arith_eng(a,b,a_unk,b_unk,arith)
        c_for_ord = self.bridge_ao(c_vec) + c_vec
        c_for_set = self.bridge_as(c_vec) + c_vec
        ord_vec = self.order_eng(c_for_ord, d, d_unk, rel)
        set_vec = self.set_eng(c_for_set, set_bits, s_unk)
        logic_vec = self.logic_eng(ord_vec, set_vec, op)
        return {'arith': c_vec, 'order': ord_vec, 'set': set_vec, 'logic': logic_vec}

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
    c[arith==0] = torch.clamp(a+b,0,NUM_RANGE)[arith==0]
    c[arith==1] = torch.abs(a-b)[arith==1]
    c[arith==2] = torch.clamp(a*b,0,NUM_RANGE)[arith==2]
    c[arith==3] = (a % torch.clamp(b,1,NUM_RANGE))[arith==3]
    c = torch.clamp(c,0,NUM_RANGE)
    c_unk = au | bu
    ord_unk = c_unk | du
    ord_v = torch.zeros(n, dtype=torch.long, device=DEVICE)
    rt = (((rel==0)&(c>d))|((rel==1)&(c<d))|((rel==2)&(c==d))|
          ((rel==3)&(c>=d))|((rel==4)&(c<=d))|((rel==5)&(c!=d)))
    ord_v[rt] = 1
    val_o = torch.where(ord_unk, torch.tensor(2,device=DEVICE), ord_v)
    sb = torch.randint(0, 2, (n,SET_DIM), dtype=torch.float32, device=DEVICE)
    sou = su | c_unk
    ci = c.clamp(0, SET_DIM-1)
    ins = sb[torch.arange(n,device=DEVICE), ci].bool()
    sv = torch.where(ins, torch.tensor(1,device=DEVICE), torch.tensor(0,device=DEVICE))
    val_s = torch.where(sou, torch.tensor(2,device=DEVICE), sv)
    target = apply_logic(op, val_o, val_s)
    return {
        'a_norm': a.float()/NUM_RANGE, 'b_norm': b.float()/NUM_RANGE, 'd_norm': d.float()/NUM_RANGE,
        'sb': sb, 's_unk': su, 'a_unk': au, 'b_unk': bu, 'd_unk': du,
        'arith': arith, 'rel': rel, 'op': op,
        'target': target,
    }

# === MLP Probe ===
class MLPProbe(nn.Module):
    def __init__(self, d_in=D_MODEL, d_hidden=256, n_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_hidden, n_classes),
        )
    def forward(self, x):
        return self.net(x)

def train_mlp_probe(X_tr, y_tr, X_te, y_te, epochs, lr, d_hidden):
    X_tr_t = torch.from_numpy(X_tr).float().to(DEVICE)
    y_tr_t = torch.from_numpy(y_tr).long().to(DEVICE)
    X_te_t = torch.from_numpy(X_te).float().to(DEVICE)
    y_te_t = torch.from_numpy(y_te).long().to(DEVICE)

    probe = MLPProbe(d_in=X_tr.shape[1], d_hidden=d_hidden).to(DEVICE)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    BATCH = 2048
    n_tr = X_tr_t.shape[0]
    best_te_acc = 0.0
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
            logits_te = probe(X_te_t)
            pred_te = logits_te.argmax(dim=-1)
            te_acc = (pred_te == y_te_t).float().mean().item()
        if te_acc > best_te_acc:
            best_te_acc = te_acc
    return best_te_acc

def run_single(checkpoint_path):
    print(f"Loading: {checkpoint_path}")
    model = IsisV9().to(DEVICE)
    state = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    print(f"Generating {args.n_samples} samples (seed {args.data_seed})...")
    data = gen_data(args.data_seed, args.n_samples)

    print("Forward pass with hidden state extraction...")
    BATCH = 8192
    hidden = {'arith': [], 'order': [], 'set': [], 'logic': []}
    with torch.no_grad():
        for i in range(0, args.n_samples, BATCH):
            j = min(i + BATCH, args.n_samples)
            with torch.amp.autocast('cuda'):
                h = model.forward_with_hidden(
                    data['a_norm'][i:j], data['b_norm'][i:j], data['d_norm'][i:j],
                    data['sb'][i:j], data['s_unk'][i:j],
                    data['a_unk'][i:j], data['b_unk'][i:j], data['d_unk'][i:j],
                    data['arith'][i:j], data['rel'][i:j], data['op'][i:j],
                )
            for k in hidden:
                hidden[k].append(h[k].float().cpu().numpy())
    hidden = {k: np.concatenate(v, axis=0) for k, v in hidden.items()}
    target = data['target'].cpu().numpy()

    n = args.n_samples
    n_train = int(n * 0.7)
    perm = np.random.RandomState(0).permutation(n)
    tr, te = perm[:n_train], perm[n_train:]

    print(f"\nRunning MLP probes (hidden={args.probe_hidden}, epochs={args.probe_epochs})...")
    results = {}
    for b in ['arith', 'order', 'set', 'logic']:
        t0 = time.time()
        X = hidden[b]
        acc = train_mlp_probe(X[tr], target[tr], X[te], target[te],
                              args.probe_epochs, args.probe_lr, args.probe_hidden)
        elapsed = time.time() - t0
        results[b] = acc
        print(f"  {b:<8}: MLP probe acc = {acc:.4f}  ({elapsed:.1f}s)")

    out = {
        'checkpoint': checkpoint_path,
        'n_samples': args.n_samples,
        'probe_hidden': args.probe_hidden,
        'probe_epochs': args.probe_epochs,
        'mlp_probe_acc': results,
    }
    out_path = os.path.join(os.path.dirname(checkpoint_path), 'nonlinear_probe.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"Saved: {out_path}")
    return out

def mean_std(xs):
    xs = [x for x in xs if x is not None]
    if not xs: return (None, None)
    m = sum(xs) / len(xs)
    if len(xs) == 1: return (m, 0.0)
    v = sum((x - m) ** 2 for x in xs) / len(xs)
    return (m, v ** 0.5)

def aggregate():
    paths = {
        42:  os.path.join(ROOT, 'theia',    'seed_42',  'nonlinear_probe.json'),
        123: os.path.join(ROOT, 'theia_v2', 'seed_123', 'nonlinear_probe.json'),
        256: os.path.join(ROOT, 'theia_v2', 'seed_256', 'nonlinear_probe.json'),
        777: os.path.join(ROOT, 'theia_v2', 'seed_777', 'nonlinear_probe.json'),
        999: os.path.join(ROOT, 'theia_v2', 'seed_999', 'nonlinear_probe.json'),
    }
    # Linear baselines from reprobe_theia_v2
    linear_paths = {
        42:  os.path.join(ROOT, 'theia',    'seed_42',  'mechanistic_probing_v2.json'),
        123: os.path.join(ROOT, 'theia_v2', 'seed_123', 'mechanistic_probing_v2.json'),
        256: os.path.join(ROOT, 'theia_v2', 'seed_256', 'mechanistic_probing_v2.json'),
        777: os.path.join(ROOT, 'theia_v2', 'seed_777', 'mechanistic_probing_v2.json'),
        999: os.path.join(ROOT, 'theia_v2', 'seed_999', 'mechanistic_probing_v2.json'),
    }
    nl_data = {}
    for s, p in paths.items():
        if os.path.exists(p):
            with open(p) as f: nl_data[s] = json.load(f)
    lin_data = {}
    for s, p in linear_paths.items():
        if os.path.exists(p):
            with open(p) as f: lin_data[s] = json.load(f)

    print(f"Loaded {len(nl_data)}/5 nonlinear, {len(lin_data)}/5 linear")

    boundaries = ['arith', 'order', 'set', 'logic']
    nl_agg = {}
    lin_agg = {}
    for b in boundaries:
        nl_vals = [d.get('mlp_probe_acc', {}).get(b) for d in nl_data.values()]
        nl_agg[b] = mean_std(nl_vals)
        lin_vals = [d.get('probes', {}).get(b, {}).get('final_verdict_acc') for d in lin_data.values()]
        lin_agg[b] = mean_std(lin_vals)

    from datetime import datetime
    lines = []
    L = lines.append
    L("# Nonlinear (MLP) Probe vs Linear SVM Probe — 5-Seed Aggregate")
    L("")
    L(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    L("")
    L("**Purpose**: Determine whether THEIA's delayed verdict is REAL (information not encoded ")
    L("in upstream representations) or LINEAR-ONLY (information nonlinearly encoded but linear probes miss it).")
    L("")
    L("**MLP probe**: Linear(256) -> GELU -> Linear(256) -> GELU -> Linear(3), AdamW lr=1e-3, 40 epochs.")
    L("")
    L("---")
    L("")
    L("## Head-to-head comparison (final verdict, 3-class)")
    L("")
    L("| Boundary | Linear SVM (old) | Nonlinear MLP (new) | Gap |")
    L("|---|---|---|---|")
    for b in boundaries:
        lm, ls = lin_agg[b]
        nm, ns = nl_agg[b]
        if lm is not None and nm is not None:
            gap = nm - lm
            L(f"| {b} | {lm:.3f} ± {ls:.3f} | {nm:.3f} ± {ns:.3f} | +{gap*100:.1f}pp |")
        else:
            L(f"| {b} | -- | -- | -- |")
    L("")

    L("## Verdict")
    L("")
    arith_nl = nl_agg['arith'][0]
    order_nl = nl_agg['order'][0]
    set_nl = nl_agg['set'][0]
    if arith_nl is not None:
        max_upstream = max(arith_nl, order_nl, set_nl)
        L(f"**Max upstream (Arith/Order/Set) MLP probe accuracy: {max_upstream:.3f}**")
        L("")
        if max_upstream < 0.85:
            L("**DELAYED VERDICT IS REAL.** MLP probe cannot extract verdict from upstream representations ")
            L("even with nonlinear capacity. The information is not present until the Logic Engine.")
            L("")
            L("§4.4 delayed verdict claim stands as-written. Optionally add a note:")
            L("> *'Nonlinear (2-layer MLP, 256 hidden) probes on upstream domain outputs reach at most ")
            L(f"> {max_upstream:.1%} final-verdict accuracy, confirming that delayed verdict reflects the ")
            L("> absence of downstream-task information rather than linear probe insufficiency.'*")
        elif max_upstream < 0.95:
            L("**DELAYED VERDICT IS PARTIALLY LINEAR.** MLP extracts more information than SVM but still ")
            L("significantly below ceiling. The claim needs softening.")
            L("")
            L("Recommend §4.4 hedge:")
            L("> *'While linear probes are near-chance on upstream representations, nonlinear (2-layer MLP) ")
            L(f"> probes reach {max_upstream:.1%} — indicating partial but incomplete encoding of downstream ")
            L("> task information at upstream boundaries. The delayed verdict pattern is therefore primarily ")
            L("> (though not exclusively) a linear phenomenon.'*")
        else:
            L("**DELAYED VERDICT IS A LINEAR ARTIFACT.** MLP extracts near-perfect verdict from upstream ")
            L("representations. The information IS present, just nonlinearly encoded. §4.4 must be rewritten.")
            L("")
            L("Critical: the strong version of the delayed verdict claim does NOT hold. Rewrite to:")
            L("> *'Linear probes on upstream representations show low final-verdict accuracy, but nonlinear ")
            L(f"> (MLP) probes achieve {max_upstream:.1%}, indicating that the information required for the ")
            L("> final verdict IS present in upstream representations — just not in a linearly decodable form. ")
            L("> THEIA's delayed verdict is therefore a property of LINEAR decodability, not of representational ")
            L("> absence. This is still a non-trivial finding: the network learns to delay linear commitment, ")
            L("> which may be a useful signature for interpretability, but it does not imply that upstream ")
            L("> engines are 'unaware' of the final answer.'*")
    L("")
    L("---")
    L("")
    L("## Raw per-seed")
    L("")
    for s, d in nl_data.items():
        L(f"- Seed {s}: {d.get('mlp_probe_acc')}")

    out_path = os.path.join(REPORT_DIR, 'nonlinear_probe_report.md')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"\nReport: {out_path}")
    print()
    print('\n'.join(lines))

if args.aggregate:
    aggregate()
elif args.checkpoint:
    run_single(args.checkpoint)
else:
    print("Provide --checkpoint <path> or --aggregate")
