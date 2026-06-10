#!/usr/bin/env python
"""
Eval-only model definition module for N1.a BigTransformer checkpoint
evaluation. Extracted from tf8l_5seed_tuned.py (original md5:
44458444fbbe5007024a976b86346ec7); training logic, CLI parsing, and top-level
side effects removed, so importing has no side effects. Model class, truth
tables, and apply_logic are byte-identical to the original (see the marked
blocks below).

Kept (pure definitions, byte-identical to source): DEVICE and constants
(D_MODEL, BATCH, NUM_RANGE, SET_DIM, P_UNKNOWN, N_*, OP_*), Kleene K3 truth
tables (AND_T/OR_T/IMP_T/IFF_T/NOT_T), apply_logic, gen_data(seed) (kept as
reference; not called at import), BigTransformer, and the 12-rule sanity
harness (KLEENE_TESTS, make_kleene_test, run_kleene).

Removed (all training-side effects): argparse CLI parsing, SEED/MAX_EPOCHS/
OUT_DIR/log_f setup, torch.manual_seed at import time, top-level print/log
calls, top-level gen_data(SEED) call and model instantiation, optimizer/
scheduler/scaler/training loop, final eval + summary write + decision report.
"""
import torch
import torch.nn as nn

# -------- byte-identical to tf8l_5seed_tuned.py: DEVICE, constants, truth tables, apply_logic --------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
D_MODEL=192; BATCH=2048; NUM_RANGE=20; SET_DIM=21; P_UNKNOWN=0.15
N_VALS=3; N_RELS=6; N_ARITH=4; N_OPS=5
OP_AND=0; OP_OR=1; OP_NOT=2; OP_IMPLIES=3; OP_IFF=4

AND_T=torch.tensor([[0,0,0],[0,1,2],[0,2,2]],dtype=torch.long,device=DEVICE)
OR_T =torch.tensor([[0,1,2],[1,1,1],[2,1,2]],dtype=torch.long,device=DEVICE)
IMP_T=torch.tensor([[1,1,1],[0,1,2],[2,1,2]],dtype=torch.long,device=DEVICE)
IFF_T=torch.tensor([[1,0,2],[0,1,2],[2,2,2]],dtype=torch.long,device=DEVICE)
NOT_T=torch.tensor([1,0,2],dtype=torch.long,device=DEVICE)

def apply_logic(op,va,vb):
    r=torch.zeros_like(op)
    m=op==OP_AND;     r[m]=AND_T[va[m],vb[m]]
    m=op==OP_OR;      r[m]=OR_T[va[m],vb[m]]
    m=op==OP_IMPLIES; r[m]=IMP_T[va[m],vb[m]]
    m=op==OP_IFF;     r[m]=IFF_T[va[m],vb[m]]
    m=op==OP_NOT;     r[m]=NOT_T[va[m]]
    return r
# -------- end byte-identical block --------

# -------- byte-identical to tf8l_5seed_tuned.py: gen_data --------
def gen_data(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    n=2_000_000
    a=torch.randint(1,NUM_RANGE+1,(n,),device=DEVICE)
    b=torch.randint(1,NUM_RANGE+1,(n,),device=DEVICE)
    d=torch.randint(0,NUM_RANGE+1,(n,),device=DEVICE)
    ar=torch.randint(0,N_ARITH,(n,),device=DEVICE)
    rl=torch.randint(0,N_RELS,(n,),device=DEVICE)
    op=torch.randint(0,N_OPS,(n,),device=DEVICE)
    au=torch.rand(n,device=DEVICE)<P_UNKNOWN
    bu=torch.rand(n,device=DEVICE)<P_UNKNOWN
    du=torch.rand(n,device=DEVICE)<P_UNKNOWN
    su=torch.rand(n,device=DEVICE)<P_UNKNOWN
    c=torch.zeros(n,dtype=torch.long,device=DEVICE)
    c[ar==0]=torch.clamp(a+b,0,NUM_RANGE)[ar==0]
    c[ar==1]=torch.abs(a-b)[ar==1]  # NOTE: SUB is |a-b|, not a-b — matches theia_5seed_v2 data generator
    c[ar==2]=torch.clamp(a*b,0,NUM_RANGE)[ar==2]
    c[ar==3]=(a%torch.clamp(b,1,NUM_RANGE))[ar==3]
    c=torch.clamp(c,0,NUM_RANGE); cu=au|bu; ou=cu|du
    ov=torch.zeros(n,dtype=torch.long,device=DEVICE)
    rt=(((rl==0)&(c>d))|((rl==1)&(c<d))|((rl==2)&(c==d))|
        ((rl==3)&(c>=d))|((rl==4)&(c<=d))|((rl==5)&(c!=d)))
    ov[rt]=1
    vo=torch.where(ou,torch.tensor(2,device=DEVICE),ov)
    sb=torch.randint(0,2,(n,SET_DIM),dtype=torch.float32,device=DEVICE)
    sou=su|cu; ci=c.clamp(0,SET_DIM-1)
    ins=sb[torch.arange(n,device=DEVICE),ci].bool()
    sv=torch.where(ins,torch.tensor(1,device=DEVICE),torch.tensor(0,device=DEVICE))
    vs=torch.where(sou,torch.tensor(2,device=DEVICE),sv)
    tgt=apply_logic(op,vo,vs)
    return (a.float()/NUM_RANGE, b.float()/NUM_RANGE, d.float()/NUM_RANGE,
            sb, su, au, bu, du, ar, rl, op, tgt)
# -------- end byte-identical block --------

# -------- byte-identical to tf8l_5seed_tuned.py: BigTransformer --------
class BigTransformer(nn.Module):
    def __init__(self, d=D_MODEL, nhead=8, nlayers=8):
        super().__init__()
        self.num_enc=nn.Sequential(nn.Linear(1,d//2),nn.GELU(),nn.Linear(d//2,d))
        self.set_enc=nn.Sequential(nn.Linear(SET_DIM,d//2),nn.GELU(),nn.Linear(d//2,d))
        self.arith_emb=nn.Embedding(N_ARITH,d)
        self.rel_emb=nn.Embedding(N_RELS,d)
        self.op_emb=nn.Embedding(N_OPS,d)
        self.unk_emb=nn.Embedding(2,d)
        self.type_emb=nn.Embedding(11,d)
        enc=nn.TransformerEncoderLayer(d_model=d,nhead=nhead,dim_feedforward=d*4,
                                       dropout=0.1,activation='gelu',batch_first=True)
        self.transformer=nn.TransformerEncoder(enc,num_layers=nlayers)
        self.head=nn.Sequential(nn.Linear(d,d),nn.GELU(),nn.LayerNorm(d),nn.Linear(d,N_VALS))
    def forward(self,a,b,d,sb,su,au,bu,du,ar,rl,op):
        B=a.shape[0]
        toks=torch.stack([
            self.num_enc(a.unsqueeze(-1)),self.num_enc(b.unsqueeze(-1)),self.num_enc(d.unsqueeze(-1)),
            self.arith_emb(ar),self.rel_emb(rl),self.op_emb(op),
            self.unk_emb(au.long()),self.unk_emb(bu.long()),self.unk_emb(du.long()),
            self.unk_emb(su.long()),self.set_enc(sb)
        ],dim=1)
        tids=torch.arange(11,device=DEVICE).unsqueeze(0).expand(B,-1)
        toks=toks+self.type_emb(tids)
        out=self.transformer(toks)
        return self.head(out.mean(dim=1))
# -------- end byte-identical block --------

# -------- byte-identical to tf8l_5seed_tuned.py: KLEENE_TESTS + make_kleene_test --------
KLEENE_TESTS=[
    ("F_and_U",0,2,OP_AND,0),("T_and_U",1,2,OP_AND,2),
    ("U_and_F",2,0,OP_AND,0),("U_and_T",2,1,OP_AND,2),
    ("T_or_U",1,2,OP_OR,1),("F_or_U",0,2,OP_OR,2),
    ("U_or_T",2,1,OP_OR,1),("U_or_F",2,0,OP_OR,2),
    ("F_imp_U",0,2,OP_IMPLIES,1),("T_imp_U",1,2,OP_IMPLIES,2),
    ("T_iff_U",1,2,OP_IFF,2),("F_iff_U",0,2,OP_IFF,2),
]
def make_kleene_test(vo,vs,logic_op,n=10000):
    a=torch.randint(1,NUM_RANGE+1,(n,),device=DEVICE)
    b=torch.randint(1,NUM_RANGE+1,(n,),device=DEVICE)
    ar=torch.randint(0,N_ARITH,(n,),device=DEVICE)
    c=torch.zeros(n,dtype=torch.long,device=DEVICE)
    c[ar==0]=torch.clamp(a+b,0,NUM_RANGE)[ar==0]
    c[ar==1]=torch.abs(a-b)[ar==1]  # NOTE: SUB is |a-b|, not a-b — matches theia_5seed_v2 data generator
    c[ar==2]=torch.clamp(a*b,0,NUM_RANGE)[ar==2]
    c[ar==3]=(a%torch.clamp(b,1,NUM_RANGE))[ar==3]
    c=torch.clamp(c,0,NUM_RANGE)
    au=torch.zeros(n,dtype=torch.bool,device=DEVICE)
    bu=torch.zeros(n,dtype=torch.bool,device=DEVICE)
    du=torch.zeros(n,dtype=torch.bool,device=DEVICE)
    su=torch.zeros(n,dtype=torch.bool,device=DEVICE)
    if vo==2: du[:]=True
    if vs==2: su[:]=True
    if vo==1: d=torch.clamp(c-1,min=0); rl=torch.full((n,),3,dtype=torch.long,device=DEVICE)
    elif vo==0: d=torch.clamp(c+1,max=NUM_RANGE); rl=torch.zeros(n,dtype=torch.long,device=DEVICE)
    else: d=torch.randint(0,NUM_RANGE+1,(n,),device=DEVICE); rl=torch.zeros(n,dtype=torch.long,device=DEVICE)
    sb=torch.randint(0,2,(n,SET_DIM),dtype=torch.float32,device=DEVICE)
    if vs==1: ci=c.clamp(0,SET_DIM-1); sb[torch.arange(n,device=DEVICE),ci]=1.0
    elif vs==0: ci=c.clamp(0,SET_DIM-1); sb[torch.arange(n,device=DEVICE),ci]=0.0
    op_t=torch.full((n,),logic_op,dtype=torch.long,device=DEVICE)
    tgt=apply_logic(op_t,
        torch.full((n,),vo,dtype=torch.long,device=DEVICE),
        torch.full((n,),vs,dtype=torch.long,device=DEVICE))
    return (a.float()/NUM_RANGE,b.float()/NUM_RANGE,d.float()/NUM_RANGE,
            sb,su,au,bu,du,ar,rl,op_t,tgt)
# -------- end byte-identical block --------

# -------- byte-identical to tf8l_5seed_tuned.py: run_kleene --------
def run_kleene(model):
    cpu_state=torch.get_rng_state()
    cuda_state=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    torch.manual_seed(42)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)
    model.eval()
    results={}; passed=0
    try:
        with torch.no_grad():
            for name,vo,vs,lop,exp in KLEENE_TESTS:
                data=make_kleene_test(vo,vs,lop)
                # Eval in FP32 (autocast removed 2026-04-20)
                preds=model(*data[:-1]).argmax(dim=-1)
                acc=(preds==data[-1]).float().mean().item()*100
                results[name]=acc
                if acc>99: passed+=1
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_state is not None: torch.cuda.set_rng_state_all(cuda_state)
    return results, passed
# -------- end byte-identical block --------
