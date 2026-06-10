#!/usr/bin/env python
"""
Complete Kleene K3 diagnostic: all 39 rules (paper App. A/C). Covers the 12
paper-tested Unknown-containing binary rules (reproducibility check) plus 27
new ones: 16 definite-definite binary, 4 U-U, 4 U-D, and 3 NOT rules.

Determinism note (2026-04-20): the per-rule RNG seed is a deterministic
function of (op, va, vb); the previous implementation used
`hash(rule_key) % 2**31`, which depends on PYTHONHASHSEED and varies across
Python processes. Specific per-rule 10K-sample accuracies may differ from the
original paper-time draws by up to ~0.2pp, but the qualitative claim is
robust: 195/195 rule-seed cells pass >99% (grand mean ~99.88%). The original
run's worst cell was F∨F=F @ seed 999 = 99.15%; under deterministic seeds the
worst cell may shift to a different rule/seed combination, but the 195/195
pass/fail verdict is preserved.

Construction protocol (doubly fixed per Appendix A): a_unk = b_unk = False
(no c_unknown pollution); val_ord T: REL_GTE (3) with d = max(0, c-1);
F: REL_LT (1) with d = c (c < c is always False); U: d_unk = True.
val_set T: S[c] = 1; F: S[c] = 0; U: s_unk = True.

Usage:
    python kleene_full_diagnostic.py --checkpoint <path>
    python kleene_full_diagnostic.py --aggregate
"""
import argparse, json, os
import torch, torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datetime import datetime

p = argparse.ArgumentParser()
p.add_argument('--checkpoint', type=str, default=None)
p.add_argument('--n-per-rule', type=int, default=10000)
p.add_argument('--aggregate', action='store_true')
args = p.parse_args()

ROOT = r'multi_seed_results'
REPORT_DIR = os.path.join(ROOT, 'reports')
os.makedirs(REPORT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
D_MODEL=128; NUM_RANGE=20; SET_DIM=21
N_VALS=3; N_RELS=6; N_ARITH=4; N_OPS=5
VAL_FALSE=0; VAL_TRUE=1; VAL_UNKNOWN=2
OP_AND=0; OP_OR=1; OP_NOT=2; OP_IMPLIES=3; OP_IFF=4
REL_GT=0; REL_LT=1; REL_EQ=2; REL_GTE=3; REL_LTE=4; REL_NEQ=5

# Kleene K3 truth tables (strong semantics)
KLEENE = {
    OP_AND: {
        (VAL_FALSE,VAL_FALSE):VAL_FALSE, (VAL_FALSE,VAL_TRUE):VAL_FALSE,  (VAL_FALSE,VAL_UNKNOWN):VAL_FALSE,
        (VAL_TRUE, VAL_FALSE):VAL_FALSE, (VAL_TRUE, VAL_TRUE):VAL_TRUE,   (VAL_TRUE, VAL_UNKNOWN):VAL_UNKNOWN,
        (VAL_UNKNOWN,VAL_FALSE):VAL_FALSE,(VAL_UNKNOWN,VAL_TRUE):VAL_UNKNOWN,(VAL_UNKNOWN,VAL_UNKNOWN):VAL_UNKNOWN,
    },
    OP_OR: {
        (VAL_FALSE,VAL_FALSE):VAL_FALSE, (VAL_FALSE,VAL_TRUE):VAL_TRUE,  (VAL_FALSE,VAL_UNKNOWN):VAL_UNKNOWN,
        (VAL_TRUE, VAL_FALSE):VAL_TRUE,  (VAL_TRUE, VAL_TRUE):VAL_TRUE,  (VAL_TRUE, VAL_UNKNOWN):VAL_TRUE,
        (VAL_UNKNOWN,VAL_FALSE):VAL_UNKNOWN,(VAL_UNKNOWN,VAL_TRUE):VAL_TRUE,(VAL_UNKNOWN,VAL_UNKNOWN):VAL_UNKNOWN,
    },
    OP_IMPLIES: {
        (VAL_FALSE,VAL_FALSE):VAL_TRUE,  (VAL_FALSE,VAL_TRUE):VAL_TRUE,  (VAL_FALSE,VAL_UNKNOWN):VAL_TRUE,
        (VAL_TRUE, VAL_FALSE):VAL_FALSE, (VAL_TRUE, VAL_TRUE):VAL_TRUE,  (VAL_TRUE, VAL_UNKNOWN):VAL_UNKNOWN,
        (VAL_UNKNOWN,VAL_FALSE):VAL_UNKNOWN,(VAL_UNKNOWN,VAL_TRUE):VAL_TRUE,(VAL_UNKNOWN,VAL_UNKNOWN):VAL_UNKNOWN,
    },
    OP_IFF: {
        (VAL_FALSE,VAL_FALSE):VAL_TRUE,  (VAL_FALSE,VAL_TRUE):VAL_FALSE, (VAL_FALSE,VAL_UNKNOWN):VAL_UNKNOWN,
        (VAL_TRUE, VAL_FALSE):VAL_FALSE, (VAL_TRUE, VAL_TRUE):VAL_TRUE,  (VAL_TRUE, VAL_UNKNOWN):VAL_UNKNOWN,
        (VAL_UNKNOWN,VAL_FALSE):VAL_UNKNOWN,(VAL_UNKNOWN,VAL_TRUE):VAL_UNKNOWN,(VAL_UNKNOWN,VAL_UNKNOWN):VAL_UNKNOWN,
    },
}
KLEENE_NOT = {VAL_FALSE:VAL_TRUE, VAL_TRUE:VAL_FALSE, VAL_UNKNOWN:VAL_UNKNOWN}

def val_name(v):
    return {VAL_FALSE:'F', VAL_TRUE:'T', VAL_UNKNOWN:'U'}[v]

def op_name(o):
    return {OP_AND:'∧', OP_OR:'∨', OP_IMPLIES:'→', OP_IFF:'↔', OP_NOT:'¬'}[o]

# Rules already tested in paper (from Table 2)
PAPER_TESTED = {
    (OP_AND, VAL_FALSE, VAL_UNKNOWN), (OP_AND, VAL_TRUE, VAL_UNKNOWN),
    (OP_AND, VAL_UNKNOWN, VAL_FALSE), (OP_AND, VAL_UNKNOWN, VAL_TRUE),
    (OP_OR,  VAL_TRUE, VAL_UNKNOWN),  (OP_OR,  VAL_FALSE, VAL_UNKNOWN),
    (OP_OR,  VAL_UNKNOWN, VAL_TRUE),  (OP_OR,  VAL_UNKNOWN, VAL_FALSE),
    (OP_IMPLIES, VAL_FALSE, VAL_UNKNOWN), (OP_IMPLIES, VAL_TRUE, VAL_UNKNOWN),
    (OP_IFF, VAL_TRUE, VAL_UNKNOWN),  (OP_IFF, VAL_FALSE, VAL_UNKNOWN),
}

# --- model (verbatim) ---
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
        mk = lambda: nn.Sequential(nn.Linear(D_MODEL*3,D_MODEL*2),nn.GELU(),nn.LayerNorm(D_MODEL*2),nn.Linear(D_MODEL*2,D_MODEL))
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
        mk = lambda: nn.Sequential(nn.Linear(D_MODEL*3,D_MODEL*2),nn.GELU(),nn.LayerNorm(D_MODEL*2),nn.Linear(D_MODEL*2,D_MODEL))
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
    def forward(self, a,b,d,set_bits,s_unk,a_unk,b_unk,d_unk,arith,rel,op):
        c_vec = self.arith_eng(a,b,a_unk,b_unk,arith)
        c_for_ord = self.bridge_ao(c_vec) + c_vec
        c_for_set = self.bridge_as(c_vec) + c_vec
        ord_vec = self.order_eng(c_for_ord, d, d_unk, rel)
        set_vec = self.set_eng(c_for_set, set_bits, s_unk)
        logic_vec = self.logic_eng(ord_vec, set_vec, op)
        out = self.out_head(logic_vec)
        prototypes = F.normalize(self.sv.weight, dim=-1)
        out_norm = F.normalize(out, dim=-1)
        logits = out_norm @ prototypes.t()
        return logits

def construct_rule_samples(n, target_val_ord, target_val_set, target_op, seed=0):
    """Build a batch of n samples where the inputs to the Logic Engine are
    guaranteed to have val_ord = target_val_ord, val_set = target_val_set, op = target_op.
    Uses doubly-fixed protocol (Appendix A).
    """
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    # Random a, b, arith -> random c
    a = torch.randint(1, NUM_RANGE+1, (n,), device=DEVICE, generator=g)
    b = torch.randint(1, NUM_RANGE+1, (n,), device=DEVICE, generator=g)
    arith = torch.randint(0, N_ARITH, (n,), device=DEVICE, generator=g)

    # Compute c the same way the data pipeline does
    c = torch.zeros(n, dtype=torch.long, device=DEVICE)
    c[arith==0] = torch.clamp(a+b,0,NUM_RANGE)[arith==0]
    c[arith==1] = torch.abs(a-b)[arith==1]  # NOTE: ARITH_SUB=|a-b| (absolute diff), not a-b; matches theia_5seed_v2.build_dataset
    c[arith==2] = torch.clamp(a*b,0,NUM_RANGE)[arith==2]
    c[arith==3] = (a % torch.clamp(b,1,NUM_RANGE))[arith==3]
    c = torch.clamp(c, 0, NUM_RANGE)

    # No Unknown pollution on a, b
    a_unk = torch.zeros(n, dtype=torch.bool, device=DEVICE)
    b_unk = torch.zeros(n, dtype=torch.bool, device=DEVICE)

    # --- construct val_ord ---
    d = torch.zeros(n, dtype=torch.long, device=DEVICE)
    rel = torch.zeros(n, dtype=torch.long, device=DEVICE)
    d_unk = torch.zeros(n, dtype=torch.bool, device=DEVICE)

    if target_val_ord == VAL_TRUE:
        # rel = GTE (3), d = max(0, c-1): c >= d is always True
        rel[:] = REL_GTE
        d = torch.clamp(c - 1, min=0)
    elif target_val_ord == VAL_FALSE:
        # rel = LT (1), d = c: c < c is always False
        rel[:] = REL_LT
        d = c.clone()
    elif target_val_ord == VAL_UNKNOWN:
        d_unk[:] = True
        rel[:] = REL_GTE  # rel value doesn't matter
        d = torch.clamp(c - 1, min=0)
    else:
        raise ValueError(f"bad val_ord {target_val_ord}")

    # --- construct val_set ---
    sb = torch.zeros((n, SET_DIM), dtype=torch.float32, device=DEVICE)
    s_unk = torch.zeros(n, dtype=torch.bool, device=DEVICE)

    # Random fill first (baseline noise)
    sb = (torch.rand((n, SET_DIM), device=DEVICE, generator=g) > 0.5).float()
    ci = c.clamp(0, SET_DIM-1)

    if target_val_set == VAL_TRUE:
        sb[torch.arange(n, device=DEVICE), ci] = 1.0
    elif target_val_set == VAL_FALSE:
        sb[torch.arange(n, device=DEVICE), ci] = 0.0
    elif target_val_set == VAL_UNKNOWN:
        s_unk[:] = True
    else:
        raise ValueError(f"bad val_set {target_val_set}")

    op = torch.full((n,), target_op, dtype=torch.long, device=DEVICE)

    return {
        'a_norm': a.float()/NUM_RANGE,
        'b_norm': b.float()/NUM_RANGE,
        'd_norm': d.float()/NUM_RANGE,
        'sb': sb, 's_unk': s_unk,
        'a_unk': a_unk, 'b_unk': b_unk, 'd_unk': d_unk,
        'arith': arith, 'rel': rel, 'op': op,
    }

def eval_rule(model, target_val_ord, target_val_set, target_op, expected, n, seed):
    data = construct_rule_samples(n, target_val_ord, target_val_set, target_op, seed=seed)
    with torch.no_grad():
        # Eval in FP32 (autocast removed 2026-04-20; specific per-rule accuracies
        # may differ from pre-fix runs by up to ~0.1pp; 195/195 verified on 5-seed rerun)
        logits = model(
            data['a_norm'], data['b_norm'], data['d_norm'],
            data['sb'], data['s_unk'],
            data['a_unk'], data['b_unk'], data['d_unk'],
            data['arith'], data['rel'], data['op'],
        )
        pred = logits.argmax(dim=-1)
        acc = (pred == expected).float().mean().item()
    return acc

def run_single(checkpoint_path):
    print(f"Loading: {checkpoint_path}")
    model = IsisV9().to(DEVICE)
    state = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    results = {}

    # --- binary ops: all 36 rules ---
    # Per-rule seed = op*9 + va*3 + vb: identical draws across runs and Python
    # versions (previously hash(rule_key), PYTHONHASHSEED-dependent). 10K samples
    # at >99% accuracy has standard error ~0.1pp, so results are robust to seed choice.
    print("\n=== Binary ops (36 rules) ===")
    for op in [OP_AND, OP_OR, OP_IMPLIES, OP_IFF]:
        for va in [VAL_FALSE, VAL_TRUE, VAL_UNKNOWN]:
            for vb in [VAL_FALSE, VAL_TRUE, VAL_UNKNOWN]:
                expected = KLEENE[op][(va, vb)]
                rule_key = f"{val_name(va)}{op_name(op)}{val_name(vb)}={val_name(expected)}"
                rule_seed = op * 9 + va * 3 + vb  # deterministic, range [0, 35]
                acc = eval_rule(model, va, vb, op, expected, args.n_per_rule, seed=rule_seed)
                is_paper = (op, va, vb) in PAPER_TESTED
                tag = " [PAPER]" if is_paper else " [NEW]"
                results[rule_key] = {'acc': acc, 'op': op, 'va': va, 'vb': vb,
                                     'expected': expected, 'paper_tested': is_paper}
                marker = "✓" if acc > 0.99 else "✗"
                print(f"  {marker} {rule_key:<20} acc = {acc:.4f}{tag}")

    # --- NOT: 3 rules ---
    print("\n=== NOT (3 rules) ===")
    for va in [VAL_FALSE, VAL_TRUE, VAL_UNKNOWN]:
        # vb irrelevant for NOT; set it to T arbitrarily
        vb = VAL_TRUE
        expected = KLEENE_NOT[va]
        rule_key = f"¬{val_name(va)}={val_name(expected)}"
        rule_seed = 100 + va  # deterministic, range [100, 102]; disjoint from binary-op seeds
        acc = eval_rule(model, va, vb, OP_NOT, expected, args.n_per_rule, seed=rule_seed)
        results[rule_key] = {'acc': acc, 'op': OP_NOT, 'va': va, 'vb': vb,
                             'expected': expected, 'paper_tested': False}
        marker = "✓" if acc > 0.99 else "✗"
        print(f"  {marker} {rule_key:<20} acc = {acc:.4f} [NEW]")

    # --- summary ---
    all_accs = [r['acc'] for r in results.values()]
    pass_count = sum(1 for a in all_accs if a > 0.99)
    new_rules = {k: v for k, v in results.items() if not v['paper_tested']}
    new_pass = sum(1 for r in new_rules.values() if r['acc'] > 0.99)

    print(f"\nOverall: {pass_count}/{len(all_accs)} rules pass > 99%")
    print(f"New rules (not in paper): {new_pass}/{len(new_rules)} pass > 99%")
    print(f"Minimum accuracy: {min(all_accs):.4f}")
    min_rule = min(results.items(), key=lambda x: x[1]['acc'])
    print(f"Worst rule: {min_rule[0]} = {min_rule[1]['acc']:.4f}")

    out = {
        'checkpoint': checkpoint_path,
        'n_per_rule': args.n_per_rule,
        'results': results,
        'summary': {
            'total_pass': pass_count,
            'total_rules': len(all_accs),
            'new_pass': new_pass,
            'new_rules': len(new_rules),
            'min_acc': min(all_accs),
            'min_rule': min_rule[0],
        }
    }
    out_path = os.path.join(os.path.dirname(checkpoint_path), 'kleene_full_diagnostic.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")
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
        42:  os.path.join(ROOT, 'theia',    'seed_42',  'kleene_full_diagnostic.json'),
        123: os.path.join(ROOT, 'theia_v2', 'seed_123', 'kleene_full_diagnostic.json'),
        256: os.path.join(ROOT, 'theia_v2', 'seed_256', 'kleene_full_diagnostic.json'),
        777: os.path.join(ROOT, 'theia_v2', 'seed_777', 'kleene_full_diagnostic.json'),
        999: os.path.join(ROOT, 'theia_v2', 'seed_999', 'kleene_full_diagnostic.json'),
    }
    data = {}
    for s, p in paths.items():
        if os.path.exists(p):
            with open(p) as f: data[s] = json.load(f)
    print(f"Loaded {len(data)}/5 Kleene full diagnostic seeds")

    if not data:
        print("No data")
        return

    # Aggregate per-rule
    all_rules = list(next(iter(data.values()))['results'].keys())
    rule_stats = {}
    for rk in all_rules:
        accs = [d['results'][rk]['acc'] for d in data.values()]
        m, s = mean_std(accs)
        rule_stats[rk] = {
            'mean': m,
            'std': s,
            'min': min(accs),
            'paper_tested': data[next(iter(data.keys()))]['results'][rk]['paper_tested'],
        }

    # Build report
    lines = []
    L = lines.append
    L("# Complete Kleene K3 Diagnostic — All 39 Rules, 5-Seed Aggregate")
    L("")
    L(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    L(f"")
    L(f"Loaded {len(data)}/5 checkpoints: {sorted(data.keys())}")
    L("")
    L("Closes the 'complete Kleene' claim gap: paper's Table 2 tested only 12 Unknown-containing ")
    L("binary rules. This diagnostic tests all 39 Kleene K3 rules (36 binary + 3 NOT), with 10K ")
    L("samples per rule per seed. Construction uses the doubly-fixed protocol of Appendix A.")
    L("")
    L("---")
    L("")

    # Split by paper-tested vs new
    paper_rules = [k for k in all_rules if rule_stats[k]['paper_tested']]
    new_rules = [k for k in all_rules if not rule_stats[k]['paper_tested']]

    # Paper-tested rules: should reproduce original Table 2
    L("## Paper-tested rules (reproducibility check, 12 rules)")
    L("")
    L("| Rule | Mean ± Std (%) | Min (%) | Pass (>99%) |")
    L("|---|---|---|---|")
    for rk in paper_rules:
        s = rule_stats[rk]
        passed = "✓" if s['min'] > 0.99 else "✗"
        L(f"| {rk} | {s['mean']*100:.2f} ± {s['std']*100:.2f} | {s['min']*100:.2f} | {passed} |")
    paper_mean = np.mean([rule_stats[k]['mean'] for k in paper_rules])
    paper_min = min([rule_stats[k]['min'] for k in paper_rules])
    L(f"")
    L(f"**Paper-tested grand mean**: {paper_mean*100:.2f}%")
    L(f"**Paper-tested minimum**: {paper_min*100:.2f}%")
    L("")

    L("## NEW rules (27 rules — never reported before)")
    L("")
    L("### Definite-definite rules (16 rules: classical logic sanity check)")
    L("")
    L("| Rule | Mean ± Std (%) | Min (%) | Pass (>99%) |")
    L("|---|---|---|---|")
    definite_rules = [k for k in new_rules if 'U' not in k.split('=')[0]]
    for rk in definite_rules:
        s = rule_stats[rk]
        passed = "✓" if s['min'] > 0.99 else "✗"
        L(f"| {rk} | {s['mean']*100:.2f} ± {s['std']*100:.2f} | {s['min']*100:.2f} | {passed} |")
    def_mean = np.mean([rule_stats[k]['mean'] for k in definite_rules]) if definite_rules else 0
    def_min = min([rule_stats[k]['min'] for k in definite_rules]) if definite_rules else 0
    L(f"")
    L(f"**Definite-definite grand mean**: {def_mean*100:.2f}%")
    L(f"**Definite-definite minimum**: {def_min*100:.2f}%")
    L("")

    L("### Other new rules (Unknown-containing but not in original Table 2)")
    L("")
    L("| Rule | Mean ± Std (%) | Min (%) | Pass (>99%) |")
    L("|---|---|---|---|")
    other_new = [k for k in new_rules if 'U' in k.split('=')[0]]
    for rk in other_new:
        s = rule_stats[rk]
        passed = "✓" if s['min'] > 0.99 else "✗"
        L(f"| {rk} | {s['mean']*100:.2f} ± {s['std']*100:.2f} | {s['min']*100:.2f} | {passed} |")
    other_mean = np.mean([rule_stats[k]['mean'] for k in other_new]) if other_new else 0
    other_min = min([rule_stats[k]['min'] for k in other_new]) if other_new else 0
    L(f"")
    L(f"**Other new grand mean**: {other_mean*100:.2f}%")
    L(f"**Other new minimum**: {other_min*100:.2f}%")
    L("")

    L("---")
    L("")
    L("## Overall Verdict")
    L("")
    all_min = min(rule_stats[k]['min'] for k in all_rules)
    n_pass = sum(1 for k in all_rules if rule_stats[k]['min'] > 0.99)
    new_pass = sum(1 for k in new_rules if rule_stats[k]['min'] > 0.99)
    L(f"- **Total rules**: 39 (12 paper-tested + 27 new)")
    L(f"- **Rules passing >99% on all 5 seeds**: {n_pass}/39")
    L(f"- **New rules passing >99% on all 5 seeds**: {new_pass}/27")
    L(f"- **Minimum accuracy across all rules and seeds**: {all_min*100:.2f}%")
    L("")
    if new_pass == 27 and n_pass == 39:
        L("**VERDICT: COMPLETE KLEENE CLAIM VALID.** All 39 rules pass on all 5 seeds, ")
        L("verifying the 'complete Kleene K3' claim at the per-rule level.")
        L("")
        L("Per-rule results for all 39 rules are tabulated above.")
    elif new_pass >= 24:
        L(f"**VERDICT: MOSTLY COMPLETE.** {new_pass}/27 new rules pass. The 'complete ")
        L("Kleene' claim holds only with the per-rule breakdown qualifying the weak rules. Failing rules ")
        L("should be checked for common structure.")
    else:
        L(f"**VERDICT: NOT SUPPORTED AT FULL-TABLE LEVEL.** Only {new_pass}/27 new rules pass. The ")
        L("'complete Kleene K3' description is not supported by this run; a scoped description is accurate.")
    L("")

    out_path = os.path.join(REPORT_DIR, 'kleene_full_diagnostic.md')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"Report: {out_path}")
    print()
    print('\n'.join(lines))

if args.aggregate:
    aggregate()
elif args.checkpoint:
    run_single(args.checkpoint)
else:
    print("Provide --checkpoint <path> or --aggregate")
