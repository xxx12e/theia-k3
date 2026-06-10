#!/usr/bin/env python
"""
Tuned-Transformer controlled experiment (paper §4.2 footnote 5; App. G recipe).

Minimal diff vs tf8l_5seed.py: same BigTransformer architecture, gen_data,
class weights, grad clipping, determinism, seed, and Kleene diagnostic.
Only the recipe changes:
  1. lr:     5e-4 -> 1e-4
  2. betas:  default (0.9, 0.999) -> (0.9, 0.98)
  3. warmup: none -> linear 5 epochs
  4. Kleene check cadence: every 20 ep (after acc>0.98) -> every 5 ep (after acc>0.95)
  5. t_first_999 and t_first_12_12 recorded separately

Reports the first time overall val_acc >= 99.9% (paper FN5), the first time
Kleene passes 12/12 (paper §4.2 main result), and ratios vs THEIA 5.7 min
(FN5) and 9.2 min (Kleene-aware).

Usage:
    python tf8l_5seed_tuned.py --seed 42
"""
import argparse, os, json, time, math
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
import numpy as np
from tqdm import tqdm

p = argparse.ArgumentParser()
p.add_argument('--seed', type=int, default=42)
p.add_argument('--max-epochs', type=int, default=150)
p.add_argument('--output-root', type=str,
               default=r'multi_seed_results\tf8l_tuned')
args = p.parse_args()

SEED = args.seed
MAX_EPOCHS = args.max_epochs
OUT_DIR = os.path.join(args.output_root, f'seed_{SEED}')
os.makedirs(OUT_DIR, exist_ok=True)
log_f = open(os.path.join(OUT_DIR, 'train_log.txt'), 'w', encoding='utf-8')
def log(m): log_f.write(m+'\n'); log_f.flush()

# --- identical to tf8l_5seed.py from here ---
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

print(f"TF8L-TUNED | seed={SEED} | device={DEVICE}")
log(f"TF8L-TUNED | seed={SEED}")

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

# --- DIFF #1 + #2: tuned optimizer ---
PEAK_LR = 1e-4
BETAS = (0.9, 0.98)
WEIGHT_DECAY = 0.01
WARMUP_EPOCHS = 5

opt=optim.AdamW(model.parameters(), lr=PEAK_LR, betas=BETAS, weight_decay=WEIGHT_DECAY)

# --- DIFF #3: linear warmup + cosine decay ---
def lr_lambda(epoch):
    if epoch < WARMUP_EPOCHS:
        return (epoch + 1) / WARMUP_EPOCHS
    progress = (epoch - WARMUP_EPOCHS) / max(1, MAX_EPOCHS - WARMUP_EPOCHS)
    return 0.5 * (1.0 + math.cos(math.pi * progress))
sched = optim.lr_scheduler.LambdaLR(opt, lr_lambda)

scaler=torch.amp.GradScaler('cuda')

def gb(idx):
    return AF[idx],BF[idx],DF[idx],SB[idx],S_UNK[idx],A_UNK[idx],B_UNK[idx],D_UNK[idx],AR[idx],RL[idx],OP[idx]

# --- DIFF #4: track t_first_999 / t_first_12_12 for the paper claim ---
best_acc=0.0
converge_epoch=MAX_EPOCHS
stable_count=0
last_kleene_passed=0
prev_kleene_passed=-1
final_kleene_results={}

t0=time.time()
first_999_epoch = None
first_999_time_min = None
first_999_val_acc = None

first_12_12_epoch = None
first_12_12_time_min = None
first_12_12_kleene = None

stable_12_12_count = 0
STOP_AT = None  # set once we have 12/12 + overall>=99.9% for 2 consecutive checks

# Per-epoch validation accuracy (cheap enough: ~50ms on val split)
def eval_overall():
    model.eval()
    with torch.no_grad():
        tidx = torch.arange(split, N, device=DEVICE)
        preds = []
        for j in range(0, len(tidx), BATCH*4):
            bi = tidx[j:j+BATCH*4]
            # Eval in FP32 (autocast removed 2026-04-20)
            preds.append(model(*gb(bi)).argmax(dim=-1))
        return (torch.cat(preds) == TGT[split:]).float().mean().item()

pbar=tqdm(range(1,MAX_EPOCHS+1),desc=f'TF8L-TUNED s{SEED}',ncols=110,leave=False)
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

    # Per-epoch overall val accuracy (needed to record t_first_999 precisely)
    acc = eval_overall()
    if acc > best_acc: best_acc = acc
    elapsed_min = (time.time() - t0) / 60.0
    log(f"epoch {epoch:3d} loss={tl/nb:.4f} acc={acc:.4f} best={best_acc:.4f} lr={opt.param_groups[0]['lr']:.2e} t={elapsed_min:.1f}m")

    if first_999_epoch is None and acc >= 0.999:
        first_999_epoch = epoch
        first_999_time_min = elapsed_min
        first_999_val_acc = acc
        torch.save(model.state_dict(), os.path.join(OUT_DIR, 'ckpt_first_999.pth'))
        log(f"  *** first val_acc >= 0.999 at epoch {epoch} ({elapsed_min:.1f} min) ***")

    # --- DIFF #5: Kleene cadence every 5 ep after acc > 0.95 ---
    if epoch % 5 == 0 and best_acc > 0.95:
        kl, kp = run_kleene(model)
        last_kleene_passed = kp
        final_kleene_results = kl
        log(f"kleene@{epoch}: {kp}/12")

        if kp == 12 and first_12_12_epoch is None:
            first_12_12_epoch = epoch
            first_12_12_time_min = elapsed_min
            first_12_12_kleene = dict(kl)
            torch.save(model.state_dict(), os.path.join(OUT_DIR, 'ckpt_first_12_12.pth'))
            log(f"  *** first 12/12 Kleene at epoch {epoch} ({elapsed_min:.1f} min) ***")

        # Paper convergence criterion: overall > 99.9% AND 12/12 on two consecutive checks
        if acc > 0.999 and kp == 12 and prev_kleene_passed == 12:
            stable_12_12_count += 1
            if stable_12_12_count >= 1:
                converge_epoch = epoch
                log(f"  paper-criterion converged @ {epoch}: 2 consecutive 12/12 + acc>99.9%")
                STOP_AT = epoch
                break
        else:
            stable_12_12_count = 0
        prev_kleene_passed = kp

    pbar.set_postfix(loss=f'{tl/nb:.3f}', best=f'{best_acc:.4f}',
                     kl=f'{last_kleene_passed}/12', t=f'{elapsed_min:.1f}m')
pbar.close()

print("\nfinal kleene diagnostic...")
final_kleene_results, kleene_passed = run_kleene(model)

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

torch.save(model.state_dict(), os.path.join(OUT_DIR,'ckpt_final.pth'))

total_time = (time.time() - t0) / 60.0

summary={
    'seed': SEED,
    'model': 'tf8l_tuned',
    'params': params,
    'tuned_package': {
        'peak_lr': PEAK_LR,
        'betas': list(BETAS),
        'warmup_epochs': WARMUP_EPOCHS,
        'weight_decay': WEIGHT_DECAY,
        'grad_clip': 1.0,
        'batch_size': BATCH,
        'class_weights_FTU': [1.0, 1.0, 2.0],
    },
    'max_epochs': MAX_EPOCHS,
    'overall_best_acc': best_acc,
    'first_999_epoch': first_999_epoch,
    'first_999_time_min': first_999_time_min,
    'first_999_val_acc': first_999_val_acc,
    'first_12_12_epoch': first_12_12_epoch,
    'first_12_12_time_min': first_12_12_time_min,
    'first_12_12_kleene': first_12_12_kleene,
    'final_kleene_passed': kleene_passed,
    'final_kleene_per_rule': final_kleene_results,
    'per_class_acc': per_class,
    'converge_epoch_paper_criterion': converge_epoch,
    'total_time_min': total_time,
}
with open(os.path.join(OUT_DIR,'summary.json'),'w') as f:
    json.dump(summary,f,indent=2)

# --- decision report ---
THEIA_OVERALL_999_MIN = 5.7   # paper FN5
THEIA_KLEENE_MIN = 9.2         # paper §4.2 main
TF8L_UNTUNED_OVERALL_999_MIN = 39.5  # paper FN5
TF8L_UNTUNED_KLEENE_MIN = 45.0       # paper §4.2 main (1-seed-42 value: 61.6)

print("\n" + "="*64)
print("  TF8L TUNED - DEFENSE EXPERIMENT RESULT")
print("="*64)
print(f"  Seed:          {SEED}")
print(f"  Params:        {params:,}")
print(f"  Total runtime: {total_time:.1f} min")
print(f"  Best overall:  {best_acc:.4f}")
print(f"  Final Kleene:  {kleene_passed}/12")
print()

if first_999_epoch is not None:
    r_fn5 = first_999_time_min / THEIA_OVERALL_999_MIN
    print(f"  Overall-99.9% milestone (matches paper FN5):")
    print(f"    tuned TF8L:    {first_999_time_min:.1f} min  @ epoch {first_999_epoch}")
    print(f"    THEIA ref:     {THEIA_OVERALL_999_MIN:.1f} min")
    print(f"    TF8L untuned:  {TF8L_UNTUNED_OVERALL_999_MIN:.1f} min")
    print(f"    ratio tuned vs THEIA: {r_fn5:.2f}x  (untuned was 6.9x)")
else:
    r_fn5 = None
    print(f"  Overall-99.9% milestone: NEVER REACHED in {MAX_EPOCHS} epochs")

print()
if first_12_12_epoch is not None:
    r_kl = first_12_12_time_min / THEIA_KLEENE_MIN
    print(f"  Kleene 12/12 milestone (matches paper §4.2 main result):")
    print(f"    tuned TF8L:    {first_12_12_time_min:.1f} min  @ epoch {first_12_12_epoch}")
    print(f"    THEIA ref:     {THEIA_KLEENE_MIN:.1f} min")
    print(f"    TF8L untuned:  {TF8L_UNTUNED_KLEENE_MIN:.1f} min (avg across 4 seeds)")
    print(f"    ratio tuned vs THEIA: {r_kl:.2f}x  (untuned was 4.9x)")
else:
    r_kl = None
    print(f"  Kleene 12/12 milestone: NEVER REACHED in {MAX_EPOCHS} epochs")
    print(f"    -> Tuning did not help the model pass 12/12; the 4.9x ratio holds trivially.")

print()
print("  VERDICT (ratio comparison):")
# Use Kleene-aware ratio as primary (matches paper 4.9x headline)
primary = r_kl if r_kl is not None else None
if primary is None:
    print("    HOLDS TRIVIALLY - tuned model did not reach paper Kleene criterion.")
    print("    The 4.9x ratio stands; note that")
    print("    'tuning did not help the baseline reach 12/12 within budget.'")
elif primary >= 4.0:
    print(f"    HOLDS - tuned ratio {primary:.2f}x is close to original 4.9x.")
    print("    The 4.9x ratio is consistent with the")
    print(f"    tuned ratio ({primary:.2f}x).")
elif primary >= 2.0:
    print(f"    SUBSTANTIALLY NARROWS - tuned ratio {primary:.2f}x.")
    print(f"    The applicable ratio is '~{primary:.1f}x under tuned baseline' rather than 4.9x.")
    print("    The matched-protocol 4.9x figure does not transfer to the tuned setting.")
else:
    print(f"    NEGATIVE - tuned ratio {primary:.2f}x too small.")
    print("    The speedup does not survive tuning; the reliability")
    print("    advantage (5/5 vs 4/5 seeds) is the remaining distinguishing property.")

print()
print(f"  Checkpoints saved:")
print(f"    {os.path.join(OUT_DIR, 'ckpt_first_999.pth')}")
if first_12_12_epoch is not None:
    print(f"    {os.path.join(OUT_DIR, 'ckpt_first_12_12.pth')}")
print(f"    {os.path.join(OUT_DIR, 'ckpt_final.pth')}")
print(f"  Summary: {os.path.join(OUT_DIR, 'summary.json')}")
print("="*64)

log(f"FINAL acc={best_acc:.4f} kleene={kleene_passed}/12 "
    f"first_999={first_999_time_min} first_12_12={first_12_12_time_min}")
log_f.close()
