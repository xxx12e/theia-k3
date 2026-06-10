#!/usr/bin/env python
"""
THEIA 4-domain wrapper v2 (paper §4.2, Table 1) — both Kleene diagnostic bugs fixed:
  Fix #1 (du vs au): vo==2 uses du[:]=True to avoid c_unknown pollution
  Fix #2 (REL_GTE):  vo==1 uses REL_GTE with d=max(0,c-1) to avoid the c=0 edge case

Usage:   python theia_5seed_v2.py --seed 123
Outputs: multi_seed_results\\theia_v2\\seed_{N}\\
"""
import argparse, os, json, time
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
import numpy as np
from tqdm import tqdm

p = argparse.ArgumentParser()
p.add_argument('--seed', type=int, required=True)
p.add_argument('--max-epochs', type=int, default=200)
p.add_argument('--output-root', type=str,
               default=r'multi_seed_results\theia_v2')
args = p.parse_args()

SEED = args.seed
MAX_EPOCHS = args.max_epochs
OUT_DIR = os.path.join(args.output_root, f'seed_{SEED}')
os.makedirs(OUT_DIR, exist_ok=True)
log_f = open(os.path.join(OUT_DIR, 'train_log.txt'), 'w', encoding='utf-8')
def log(m): log_f.write(m+'\n'); log_f.flush()

torch.manual_seed(SEED); np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
D_MODEL=128; BATCH=4096; NUM_RANGE=20; SET_DIM=21; P_UNKNOWN=0.15
VAL_FALSE=0; VAL_TRUE=1; VAL_UNKNOWN=2; N_VALS=3
N_RELS=6; N_ARITH=4; N_OPS=5
REL_GT=0; REL_LT=1; REL_EQ=2; REL_GTE=3; REL_LTE=4; REL_NEQ=5
ARITH_ADD=0; ARITH_SUB=1; ARITH_MUL=2; ARITH_MOD=3  # NOTE: ARITH_SUB is |a-b| (absolute difference), not a-b. See build_dataset / make_kleene_test.
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

print(f"THEIA v2 | seed={SEED} | device={DEVICE}")
log(f"THEIA v2 | seed={SEED}")

# --- Model (isis_v9 verbatim) ---
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
    c_val[arith==ARITH_SUB] = torch.abs(a_val-b_val)[arith==ARITH_SUB]  # |a-b|, not a-b; see ARITH_SUB note above
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

# --- Kleene diagnostic (doubly fixed) ---
KLEENE_TESTS = [
    ("F_and_U", 0, 2, OP_AND,     0), ("T_and_U", 1, 2, OP_AND,     2),
    ("U_and_F", 2, 0, OP_AND,     0), ("U_and_T", 2, 1, OP_AND,     2),
    ("T_or_U",  1, 2, OP_OR,      1), ("F_or_U",  0, 2, OP_OR,      2),
    ("U_or_T",  2, 1, OP_OR,      1), ("U_or_F",  2, 0, OP_OR,      2),
    ("F_imp_U", 0, 2, OP_IMPLIES, 1), ("T_imp_U", 1, 2, OP_IMPLIES, 2),
    ("T_iff_U", 1, 2, OP_IFF,     2), ("F_iff_U", 0, 2, OP_IFF,     2),
]

def make_kleene_test(vo, vs, logic_op, n=10000):
    """Doubly-fixed: du (not au) for vo==2, REL_GTE (not REL_GT) for vo==1."""
    a = torch.randint(1, NUM_RANGE+1, (n,), device=DEVICE)
    b = torch.randint(1, NUM_RANGE+1, (n,), device=DEVICE)
    ar = torch.randint(0, N_ARITH, (n,), device=DEVICE)
    c = torch.zeros(n, dtype=torch.long, device=DEVICE)
    c[ar==0] = torch.clamp(a+b,0,NUM_RANGE)[ar==0]
    c[ar==1] = torch.abs(a-b)[ar==1]  # |a-b|, matches build_dataset
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
                # Eval in FP32 (autocast removed 2026-04-20; training kept AMP)
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

# --- Main ---
AF,BF,DF,SB,S_UNK,A_UNK,B_UNK,D_UNK,AR,RL,OP,TARGET,N = build_dataset(SEED)
split = int(N*0.8)

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

model = IsisV9().to(DEVICE)
params = sum(p.numel() for p in model.parameters())
print(f"params: {params:,}")
log(f"params: {params:,}")

class_weights = torch.tensor([1.0, 1.0, 2.0], device=DEVICE)
opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS, eta_min=1e-5)
scaler = torch.amp.GradScaler('cuda')

def get_batch(idx):
    return (AF[idx],BF[idx],DF[idx],SB[idx],S_UNK[idx],
            A_UNK[idx],B_UNK[idx],D_UNK[idx],AR[idx],RL[idx],OP[idx])

best_acc = 0.0
converge_epoch = MAX_EPOCHS
kleene_streak = 0
last_kleene_passed = 0
final_kleene_results = {}
t_start = time.time()

pbar = tqdm(range(1, MAX_EPOCHS+1), desc=f'THEIA v2 s{SEED}', ncols=110)
for epoch in pbar:
    model.train()
    perm = torch.randperm(split, device=DEVICE)
    tl = 0.0; nb = 0
    for i in range(0, split, BATCH):
        idx = perm[i:i+BATCH]
        with torch.amp.autocast('cuda'):
            out = model(*get_batch(idx))
            logits = out @ model.sv.weight.T
            loss = F.cross_entropy(logits, TARGET[idx], weight=class_weights)
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update()
        tl += loss.item(); nb += 1
    sched.step()
    avg_loss = tl / nb

    # Overall-eval cadence unified 10->5 ep (2026-04-20) to match the Kleene cadence
    # below; reruns of paper Table 1 THEIA wall-clock may early-stop a few epochs sooner.
    if epoch % 5 == 0:
        model.eval()
        with torch.no_grad():
            tidx = torch.arange(split, N, device=DEVICE)
            preds = []
            for j in range(0, len(tidx), BATCH*4):
                bi = tidx[j:j+BATCH*4]
                # Eval in FP32 (autocast removed 2026-04-20)
                preds.append(model.classify(model(*get_batch(bi))))
            acc = (torch.cat(preds)==TARGET[split:]).float().mean().item()
            if acc > best_acc: best_acc = acc
        log(f"epoch {epoch:3d} loss={avg_loss:.4f} acc={acc:.4f} best={best_acc:.4f}")

    # Kleene cadence unified to 5 ep (2026-04-20) across theia_5seed_v2 /
    # tf8l_5seed_earlystop_v2 / tf8l_5seed_tuned. Paper Table 1 THEIA wall-clock
    # (7.93 ± 1.40 min) used 20-ep cadence; 5-ep reruns may early-stop up to ~30 epochs
    # sooner ("2 consecutive checks" = 10 ep of stability instead of 40; first-12/12 unchanged).
    if epoch % 5 == 0 and best_acc > 0.99:
        kleene_results, kleene_passed = run_kleene(model)
        last_kleene_passed = kleene_passed
        final_kleene_results = kleene_results
        log(f"kleene@{epoch}: {kleene_passed}/12")
        if kleene_passed == 12 and best_acc > 0.999:
            kleene_streak += 1
            if kleene_streak >= 2:
                converge_epoch = epoch
                log(f"early stop @ {epoch}: 12/12 x2")
                break
        else:
            kleene_streak = 0

    pbar.set_postfix(loss=f'{avg_loss:.3f}', best=f'{best_acc:.3f}',
                     kl=f'{last_kleene_passed}/12')
pbar.close()

print("final kleene...")
final_kleene_results, kleene_passed = run_kleene(model)

model.eval()
with torch.no_grad():
    tidx = torch.arange(split, N, device=DEVICE)
    preds_all = []
    for j in range(0, len(tidx), BATCH*4):
        bi = tidx[j:j+BATCH*4]
        # Eval in FP32 (autocast removed 2026-04-20)
        preds_all.append(model.classify(model(*get_batch(bi))))
    preds_all = torch.cat(preds_all)
    labels = TARGET[split:]
    per_class = {}
    for vid, vn in [(0,'False'),(1,'True'),(2,'Unknown')]:
        m = labels == vid
        if m.sum() > 0:
            per_class[vn] = (preds_all[m]==labels[m]).float().mean().item()

torch.save(model.state_dict(), os.path.join(OUT_DIR, 'checkpoint.pth'))
summary = {
    'seed': SEED, 'model': 'theia_v2', 'params': params,
    'overall_acc': best_acc, 'converge_epoch': converge_epoch, 'max_epochs': MAX_EPOCHS,
    'kleene_passed': kleene_passed,
    'kleene_per_rule': final_kleene_results,
    'per_class_acc': per_class,
    'total_time_sec': time.time() - t_start,
    'diagnostic_version': 'fixed_du_and_relgte',
}
with open(os.path.join(OUT_DIR, 'summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\n{'='*60}")
print(f"THEIA v2 seed={SEED} | acc={best_acc:.4f} | {kleene_passed}/12 | "
      f"{(time.time()-t_start)/60:.1f}m")
for k, v in final_kleene_results.items():
    print(f"  {k:10s} {v:6.2f}% {'PASS' if v > 99 else 'FAIL'}")
print(f"per-class: {per_class}")
print(f"output: {OUT_DIR}")
log(f"FINAL acc={best_acc:.4f} kleene={kleene_passed}/12 epoch={converge_epoch}")
log_f.close()
