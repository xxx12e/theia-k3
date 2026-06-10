#!/usr/bin/env python
"""
Eval-only model definition module for THEIA (IsisV9), imported by the probe
and diagnostic scripts. Extracted from theia_5seed_v2.py (original md5:
73db4e0b0563240665b92d4fdd0513d4); training loop, CLI parsing, and top-level
side effects removed, so importing has no side effects.

Kept (pure definitions, byte-identical to source unless marked): constants
(DEVICE, D_MODEL, BATCH, NUM_RANGE, SET_DIM, P_UNKNOWN; VAL_*/N_*/REL_*/
ARITH_*/OP_* codes), Kleene K3 truth tables (AND_T/OR_T/IMP_T/IFF_T/NOT_T),
apply_logic, NumEnc/SetEnc/make_mlp, ArithEngine/OrderEngine/SetEngine/
LogicEngine, IsisV9 (main THEIA model), build_dataset(seed) (2M-sample
4-domain generator), KLEENE_TESTS (12-rule targeted diagnostic),
make_kleene_test, run_kleene (doubly-fixed diagnostic, returns
(per_rule, passed)).

Removed (all training-side effects): argparse CLI parsing (theia_5seed_v2.py
L18-23), SEED/MAX_EPOCHS/OUT_DIR/log_f setup (L25-30), torch.manual_seed at
import time (L32-36), top-level print/log calls (L61-62), top-level
build_dataset(SEED) call (L299), top-level model instantiation (L306),
optimizer/scheduler/scaler/training loop (L312-376), final eval + checkpoint
save + summary write (L378-418).

Byte-identity to theia_5seed_v2.py can be verified by diffing the source line
ranges given in the block markers below. A no-op `log` stub is provided at
module scope so that build_dataset's internal log(...) call
(theia_5seed_v2.py L210) resolves without the original logging setup.
"""
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

# No-op log stub so build_dataset's log(...) call resolves without error
def log(_msg):
    pass

# -------- byte-identical to theia_5seed_v2.py L38-50 --------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
D_MODEL=128; BATCH=4096; NUM_RANGE=20; SET_DIM=21; P_UNKNOWN=0.15
VAL_FALSE=0; VAL_TRUE=1; VAL_UNKNOWN=2; N_VALS=3
N_RELS=6; N_ARITH=4; N_OPS=5
REL_GT=0; REL_LT=1; REL_EQ=2; REL_GTE=3; REL_LTE=4; REL_NEQ=5
ARITH_ADD=0; ARITH_SUB=1; ARITH_MUL=2; ARITH_MOD=3
OP_AND=0; OP_OR=1; OP_NOT=2; OP_IMPLIES=3; OP_IFF=4

AND_T=torch.tensor([[0,0,0],[0,1,2],[0,2,2]],dtype=torch.long,device=DEVICE)
OR_T =torch.tensor([[0,1,2],[1,1,1],[2,1,2]],dtype=torch.long,device=DEVICE)
IMP_T=torch.tensor([[1,1,1],[0,1,2],[2,1,2]],dtype=torch.long,device=DEVICE)
IFF_T=torch.tensor([[1,0,2],[0,1,2],[2,2,2]],dtype=torch.long,device=DEVICE)
NOT_T=torch.tensor([1,0,2],dtype=torch.long,device=DEVICE)
# -------- end L38-50 --------

# -------- byte-identical to theia_5seed_v2.py L52-59 --------
def apply_logic(op, va, vb):
    r = torch.zeros_like(op)
    m=op==OP_AND;     r[m]=AND_T[va[m],vb[m]]
    m=op==OP_OR;      r[m]=OR_T[va[m],vb[m]]
    m=op==OP_IMPLIES; r[m]=IMP_T[va[m],vb[m]]
    m=op==OP_IFF;     r[m]=IFF_T[va[m],vb[m]]
    m=op==OP_NOT;     r[m]=NOT_T[va[m]]
    return r
# -------- end L52-59 --------

# -------- byte-identical to theia_5seed_v2.py L65-170 (Model classes) --------
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
        self.ne = NumEnc()
        self.ae = nn.Embedding(N_ARITH, D_MODEL)
        self.net = make_mlp(D_MODEL*3, D_MODEL)
    def forward(self, a, b, a_unk, b_unk, arith):
        return self.net(torch.cat([self.ne(a,a_unk), self.ne(b,b_unk), self.ae(arith)], dim=-1))

class OrderEngine(nn.Module):
    def __init__(self):
        super().__init__()
        self.ne = NumEnc()
        self.re = nn.Embedding(N_RELS, D_MODEL)
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
        self.se = SetEnc()
        self.net = make_mlp(D_MODEL*2, D_MODEL)
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
    def forward(self, a,b,d,set_bits,s_unk,a_unk,b_unk,d_unk,arith,rel,op):
        c_vec = self.arith_eng(a,b,a_unk,b_unk,arith)
        c_for_ord = self.bridge_ao(c_vec) + c_vec
        c_for_set = self.bridge_as(c_vec) + c_vec
        ord_vec = self.order_eng(c_for_ord,d,d_unk,rel)
        set_vec = self.set_eng(c_for_set,set_bits,s_unk)
        return self.out_head(self.logic_eng(ord_vec,set_vec,op))
    def classify(self, v):
        sn = F.normalize(self.sv.weight, dim=-1)
        vn = F.normalize(v, dim=-1)
        return (vn @ sn.T).argmax(dim=-1)
# -------- end L65-170 --------

# -------- byte-identical to theia_5seed_v2.py L172-215 --------
def build_dataset(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    N = 2_000_000
    t0 = time.time()
    a_val = torch.randint(1, NUM_RANGE+1, (N,), device=DEVICE)
    b_val = torch.randint(1, NUM_RANGE+1, (N,), device=DEVICE)
    d_val = torch.randint(0, NUM_RANGE+1, (N,), device=DEVICE)
    arith = torch.randint(0, N_ARITH, (N,), device=DEVICE)
    rel   = torch.randint(0, N_RELS,  (N,), device=DEVICE)
    op    = torch.randint(0, N_OPS,   (N,), device=DEVICE)
    a_unknown = torch.rand(N, device=DEVICE) < P_UNKNOWN
    b_unknown = torch.rand(N, device=DEVICE) < P_UNKNOWN
    d_unknown = torch.rand(N, device=DEVICE) < P_UNKNOWN
    s_unknown = torch.rand(N, device=DEVICE) < P_UNKNOWN
    c_val = torch.zeros(N, dtype=torch.long, device=DEVICE)
    c_val[arith==ARITH_ADD] = torch.clamp(a_val+b_val,0,NUM_RANGE)[arith==ARITH_ADD]
    c_val[arith==ARITH_SUB] = torch.abs(a_val-b_val)[arith==ARITH_SUB]
    c_val[arith==ARITH_MUL] = torch.clamp(a_val*b_val,0,NUM_RANGE)[arith==ARITH_MUL]
    c_val[arith==ARITH_MOD] = (a_val % torch.clamp(b_val,1,NUM_RANGE))[arith==ARITH_MOD]
    c_val = torch.clamp(c_val,0,NUM_RANGE)
    c_unknown = a_unknown | b_unknown
    ord_unknown = c_unknown | d_unknown
    ord_val = torch.zeros(N, dtype=torch.long, device=DEVICE)
    rel_true = (((rel==0)&(c_val>d_val))|((rel==1)&(c_val<d_val))|
                ((rel==2)&(c_val==d_val))|((rel==3)&(c_val>=d_val))|
                ((rel==4)&(c_val<=d_val))|((rel==5)&(c_val!=d_val)))
    ord_val[rel_true] = VAL_TRUE
    val_ord = torch.where(ord_unknown, torch.tensor(VAL_UNKNOWN,device=DEVICE), ord_val)
    set_bits = torch.randint(0, 2, (N,SET_DIM), dtype=torch.float32, device=DEVICE)
    set_op_unknown = s_unknown | c_unknown
    c_idx = c_val.clamp(0, SET_DIM-1)
    in_set = set_bits[torch.arange(N,device=DEVICE), c_idx].bool()
    set_val = torch.where(in_set, torch.tensor(VAL_TRUE,device=DEVICE),
                                  torch.tensor(VAL_FALSE,device=DEVICE))
    val_set = torch.where(set_op_unknown, torch.tensor(VAL_UNKNOWN,device=DEVICE), set_val)
    target = apply_logic(op, val_ord, val_set)
    log(f"data N={N} | {time.time()-t0:.1f}s")
    perm = torch.randperm(N, device=DEVICE)
    return (a_val[perm].float()/NUM_RANGE, b_val[perm].float()/NUM_RANGE,
            d_val[perm].float()/NUM_RANGE, set_bits[perm], s_unknown[perm],
            a_unknown[perm], b_unknown[perm], d_unknown[perm],
            arith[perm], rel[perm], op[perm], target[perm], N)
# -------- end L172-215 --------

# -------- byte-identical to theia_5seed_v2.py L218-225 --------
KLEENE_TESTS = [
    ("F_and_U", 0, 2, OP_AND,     0), ("T_and_U", 1, 2, OP_AND,     2),
    ("U_and_F", 2, 0, OP_AND,     0), ("U_and_T", 2, 1, OP_AND,     2),
    ("T_or_U",  1, 2, OP_OR,      1), ("F_or_U",  0, 2, OP_OR,      2),
    ("U_or_T",  2, 1, OP_OR,      1), ("U_or_F",  2, 0, OP_OR,      2),
    ("F_imp_U", 0, 2, OP_IMPLIES, 1), ("T_imp_U", 1, 2, OP_IMPLIES, 2),
    ("T_iff_U", 1, 2, OP_IFF,     2), ("F_iff_U", 0, 2, OP_IFF,     2),
]
# -------- end L218-225 --------

# -------- byte-identical to theia_5seed_v2.py L227-271 --------
def make_kleene_test(vo, vs, logic_op, n=10000):
    """Doubly-fixed: du (not au) for vo==2, REL_GTE (not REL_GT) for vo==1."""
    a = torch.randint(1, NUM_RANGE+1, (n,), device=DEVICE)
    b = torch.randint(1, NUM_RANGE+1, (n,), device=DEVICE)
    ar = torch.randint(0, N_ARITH, (n,), device=DEVICE)
    c = torch.zeros(n, dtype=torch.long, device=DEVICE)
    c[ar==0] = torch.clamp(a+b,0,NUM_RANGE)[ar==0]
    c[ar==1] = torch.abs(a-b)[ar==1]
    c[ar==2] = torch.clamp(a*b,0,NUM_RANGE)[ar==2]
    c[ar==3] = (a % torch.clamp(b,1,NUM_RANGE))[ar==3]
    c = torch.clamp(c,0,NUM_RANGE)

    au = torch.zeros(n, dtype=torch.bool, device=DEVICE)
    bu = torch.zeros(n, dtype=torch.bool, device=DEVICE)
    du = torch.zeros(n, dtype=torch.bool, device=DEVICE)
    su = torch.zeros(n, dtype=torch.bool, device=DEVICE)

    # Fix #1
    if vo == 2: du[:] = True
    if vs == 2: su[:] = True

    if vo == 1:
        # Fix #2
        d = torch.clamp(c-1, min=0)
        rl = torch.full((n,), REL_GTE, dtype=torch.long, device=DEVICE)
    elif vo == 0:
        d = torch.clamp(c+1, max=NUM_RANGE)
        rl = torch.full((n,), REL_GT, dtype=torch.long, device=DEVICE)
    else:
        d = torch.zeros(n, dtype=torch.long, device=DEVICE)
        rl = torch.zeros(n, dtype=torch.long, device=DEVICE)

    sb = torch.randint(0, 2, (n,SET_DIM), dtype=torch.float32, device=DEVICE)
    ci = c.clamp(0, SET_DIM-1)
    if vs == 1:
        sb[torch.arange(n,device=DEVICE), ci] = 1.0
    elif vs == 0:
        sb[torch.arange(n,device=DEVICE), ci] = 0.0

    op_t = torch.full((n,), logic_op, dtype=torch.long, device=DEVICE)
    target = apply_logic(op_t,
        torch.full((n,), vo, dtype=torch.long, device=DEVICE),
        torch.full((n,), vs, dtype=torch.long, device=DEVICE))
    return (a.float()/NUM_RANGE, b.float()/NUM_RANGE, d.float()/NUM_RANGE,
            sb, su, au, bu, du, ar, rl, op_t, target)
# -------- end L227-271 --------

# -------- byte-identical to theia_5seed_v2.py L273-296 --------
def run_kleene(model):
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    model.eval()
    results = {}
    passed = 0
    try:
        with torch.no_grad():
            for name, vo, vs, lop, exp in KLEENE_TESTS:
                data = make_kleene_test(vo, vs, lop)
                with torch.amp.autocast('cuda'):
                    out = model(*data[:-1])
                    preds = model.classify(out)
                acc = (preds==data[-1]).float().mean().item() * 100
                results[name] = acc
                if acc > 99: passed += 1
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
    return results, passed
# -------- end L273-296 --------
