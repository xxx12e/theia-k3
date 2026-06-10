#!/usr/bin/env python
"""
Transformer 8L8H baseline with THEIA-identical early stopping (paper §4.2, Table 1).

tf8l_5seed.py with three changes: output dir tf8l_earlystop; Kleene diagnostic
every 10 epochs instead of 20 (later unified to 5 — see cadence note in the
training loop); early stop on "overall > 99.9% AND kleene == 12/12 twice
consecutive" instead of "stable at any kp/12". All training hyperparameters
are identical to tf8l_5seed.py.

Usage: python tf8l_5seed_earlystop_v2.py --seed 42
"""
import argparse, os, json, time
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
import numpy as np
from tqdm import tqdm

p = argparse.ArgumentParser()
p.add_argument('--seed', type=int, required=True)
p.add_argument('--max-epochs', type=int, default=150)
p.add_argument('--output-root', type=str,
               default=r'multi_seed_results\tf8l_earlystop')
args = p.parse_args()

SEED = args.seed
MAX_EPOCHS = args.max_epochs
OUT_DIR = os.path.join(args.output_root, f'seed_{SEED}')
os.makedirs(OUT_DIR, exist_ok=True)
log_f = open(os.path.join(OUT_DIR, 'train_log.txt'), 'w', encoding='utf-8')
def log(m):
    log_f.write(m+'\n')
    log_f.flush()
    os.fsync(log_f.fileno())

torch.manual_seed(SEED); np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

print(f"TF8L EARLYSTOP | seed={SEED} | device={DEVICE}")
log(f"TF8L EARLYSTOP | seed={SEED}")

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

AF,BF,DF,SB,S_UNK,A_UNK,B_UNK,D_UNK,AR,RL,OP,TGT = gen_data(SEED)
N=AF.shape[0]; split=int(N*0.8)

torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

model=BigTransformer().to(DEVICE)
params=sum(p.numel() for p in model.parameters())
print(f"params: {params:,}")
log(f"params: {params:,}")

cw=torch.tensor([1.0,1.0,2.0],device=DEVICE)
opt=optim.AdamW(model.parameters(),lr=5e-4,weight_decay=0.01)
sched=optim.lr_scheduler.CosineAnnealingLR(opt,T_max=MAX_EPOCHS,eta_min=1e-5)
scaler=torch.amp.GradScaler('cuda')

def gb(idx):
    return AF[idx],BF[idx],DF[idx],SB[idx],S_UNK[idx],A_UNK[idx],B_UNK[idx],D_UNK[idx],AR[idx],RL[idx],OP[idx]

best_acc=0.0; converge_epoch=MAX_EPOCHS; consecutive_pass=0
final_kleene_results={}; t0=time.time(); converged=False

pbar=tqdm(range(1,MAX_EPOCHS+1),desc=f'TF8L-ES s{SEED}',ncols=110)
for epoch in pbar:
    model.train()
    perm=torch.randperm(split,device=DEVICE); tl=0.0; nb=0
    for i in range(0,split,BATCH):
        idx=perm[i:i+BATCH]
        with torch.amp.autocast('cuda'):
            loss=F.cross_entropy(model(*gb(idx)),TGT[idx],weight=cw)
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        scaler.step(opt); scaler.update()
        tl+=loss.item(); nb+=1
    sched.step()

    # Eval cadence unified 10->5 ep (2026-04-20) across theia_5seed_v2 /
    # tf8l_5seed_earlystop_v2 / tf8l_5seed_tuned. Paper Table 1 matched-protocol TF
    # wall-clock (51.5 ± 11.0 min) used 10-ep cadence; 5-ep reruns may early-stop
    # up to ~5 epochs sooner.
    if epoch%5==0:
        model.eval()
        with torch.no_grad():
            tidx=torch.arange(split,N,device=DEVICE); preds=[]
            for j in range(0,len(tidx),BATCH*4):
                bi=tidx[j:j+BATCH*4]
                # Eval in FP32 (autocast removed 2026-04-20)
                preds.append(model(*gb(bi)).argmax(dim=-1))
            acc=(torch.cat(preds)==TGT[split:]).float().mean().item()
            if acc>best_acc: best_acc=acc
        elapsed=time.time()-t0
        log(f"epoch {epoch:3d} loss={tl/nb:.4f} acc={acc:.4f} best={best_acc:.4f} time={elapsed/60:.1f}min")

        # Kleene check from epoch 30 once overall acc > 99.5%; only the outer
        # cadence changed (10→5 ep).
        if epoch>=30 and acc>0.995:
            kl,kp=run_kleene(model)
            final_kleene_results=kl
            log(f"  kleene@{epoch}: {kp}/12")
            if acc>0.999 and kp==12:
                consecutive_pass+=1
                log(f"  consecutive_pass: {consecutive_pass}/2")
                if consecutive_pass>=2:
                    converge_epoch=epoch
                    converged=True
                    elapsed=time.time()-t0
                    log(f"CONVERGED @ epoch {epoch}, time={elapsed/60:.1f}min")
                    break
            else:
                consecutive_pass=0

    pbar.set_postfix(loss=f'{tl/nb:.3f}',best=f'{best_acc:.3f}',pass_=f'{consecutive_pass}/2')
pbar.close()

print("final kleene...")
final_kleene_results,kleene_passed=run_kleene(model)

model.eval()
with torch.no_grad():
    tidx=torch.arange(split,N,device=DEVICE); preds_all=[]
    for j in range(0,len(tidx),BATCH*4):
        bi=tidx[j:j+BATCH*4]
        # Eval in FP32 (autocast removed 2026-04-20)
        preds_all.append(model(*gb(bi)).argmax(dim=-1))
    preds_all=torch.cat(preds_all); labels=TGT[split:]
    per_class={}
    for vid,vn in [(0,'False'),(1,'True'),(2,'Unknown')]:
        m=labels==vid
        if m.sum()>0:
            per_class[vn]=(preds_all[m]==labels[m]).float().mean().item()

torch.save(model.state_dict(), os.path.join(OUT_DIR,'checkpoint.pth'))
total_time=time.time()-t0
summary={
    'seed':SEED,'model':'tf8l_earlystop','params':params,
    'overall_acc':best_acc,'converge_epoch':converge_epoch,'max_epochs':MAX_EPOCHS,
    'converged':converged,
    'kleene_passed':kleene_passed,'kleene_per_rule':final_kleene_results,
    'per_class_acc':per_class,'total_time_sec':total_time,
}
with open(os.path.join(OUT_DIR,'summary.json'),'w') as f:
    json.dump(summary,f,indent=2)

print(f"\n{'='*60}")
print(f"TF8L-ES seed={SEED} | acc={best_acc:.4f} | {kleene_passed}/12 | converged={converged} @ {converge_epoch} | {total_time/60:.1f}m")
for k,v in final_kleene_results.items():
    print(f"  {k:10s} {v:6.2f}% {'PASS' if v>99 else 'FAIL'}")
log(f"FINAL acc={best_acc:.4f} kleene={kleene_passed}/12 converged={converged} epoch={converge_epoch} time={total_time/60:.1f}min")
log_f.close()
