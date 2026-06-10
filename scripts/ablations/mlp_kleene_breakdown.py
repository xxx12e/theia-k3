"""
Flat MLP Kleene per-rule breakdown: the flat MLP reaches 99.1% overall — where do
its ~3600 errors fall across the 12 Kleene rules? Trains the MLP to convergence,
then runs the 12-rule Kleene diagnostic.
"""
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
import sys

DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
D_MODEL=128;BATCH=4096;EPOCHS=200;NUM_RANGE=20;SET_DIM=21;P_UNKNOWN=0.15
VAL_FALSE=0;VAL_TRUE=1;VAL_UNKNOWN=2;N_VALS=3
N_RELS=6;N_ARITH=4;N_OPS=5
OP_AND=0;OP_OR=1;OP_NOT=2;OP_IMPLIES=3;OP_IFF=4

AND_T=torch.tensor([[0,0,0],[0,1,2],[0,2,2]],dtype=torch.long).to(DEVICE)
OR_T=torch.tensor([[0,1,2],[1,1,1],[2,1,2]],dtype=torch.long).to(DEVICE)
IMP_T=torch.tensor([[1,1,1],[0,1,2],[2,1,2]],dtype=torch.long).to(DEVICE)
IFF_T=torch.tensor([[1,0,2],[0,1,2],[2,2,2]],dtype=torch.long).to(DEVICE)
NOT_T=torch.tensor([1,0,2],dtype=torch.long).to(DEVICE)

def apply_logic(op,va,vb):
    r=torch.zeros_like(op)
    m=op==OP_AND;r[m]=AND_T[va[m],vb[m]]
    m=op==OP_OR;r[m]=OR_T[va[m],vb[m]]
    m=op==OP_IMPLIES;r[m]=IMP_T[va[m],vb[m]]
    m=op==OP_IFF;r[m]=IFF_T[va[m],vb[m]]
    m=op==OP_NOT;r[m]=NOT_T[va[m]]
    return r

print(f"{'='*60}")
print(f"  Flat MLP Kleene Per-Rule Breakdown")
print(f"  设备: {DEVICE}")
print(f"{'='*60}")

def gen_data(n, seed=42):
    torch.manual_seed(seed)
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
    c[ar==1]=torch.abs(a-b)[ar==1]
    c[ar==2]=torch.clamp(a*b,0,NUM_RANGE)[ar==2]
    c[ar==3]=(a%torch.clamp(b,1,NUM_RANGE))[ar==3]
    c=torch.clamp(c,0,NUM_RANGE);cu=au|bu;ou=cu|du
    ov=torch.zeros(n,dtype=torch.long,device=DEVICE)
    rt=((rl==0)&(c>d))|((rl==1)&(c<d))|((rl==2)&(c==d))|((rl==3)&(c>=d))|((rl==4)&(c<=d))|((rl==5)&(c!=d))
    ov[rt]=1
    vo=torch.where(ou,torch.tensor(2,device=DEVICE),ov)
    sb=torch.randint(0,2,(n,SET_DIM),dtype=torch.float32,device=DEVICE)
    sou=su|cu;ci=c.clamp(0,SET_DIM-1)
    ins=sb[torch.arange(n,device=DEVICE),ci].bool()
    sv=torch.where(ins,torch.tensor(1,device=DEVICE),torch.tensor(0,device=DEVICE))
    vs=torch.where(sou,torch.tensor(2,device=DEVICE),sv)
    tgt=apply_logic(op,vo,vs)
    return a.float()/NUM_RANGE,b.float()/NUM_RANGE,d.float()/NUM_RANGE,sb,su,au,bu,du,ar,rl,op,tgt

class FlatMLP(nn.Module):
    def __init__(self):
        super().__init__()
        input_dim = 3 + SET_DIM + 4 + N_ARITH + N_RELS + N_OPS  # 43
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512), nn.GELU(), nn.LayerNorm(512),
            nn.Linear(512, 1024), nn.GELU(), nn.LayerNorm(1024),
            nn.Linear(1024, 1024), nn.GELU(), nn.LayerNorm(1024), nn.Dropout(0.1),
            nn.Linear(1024, 512), nn.GELU(), nn.LayerNorm(512),
            nn.Linear(512, 256), nn.GELU(), nn.LayerNorm(256),
            nn.Linear(256, 128), nn.GELU(), nn.LayerNorm(128),
        )
        self.head = nn.Linear(128, N_VALS)
    def forward(self, a, b, d, sb, su, au, bu, du, ar, rl, op):
        nums = torch.stack([a, b, d], dim=-1)
        unks = torch.stack([su.float(), au.float(), bu.float(), du.float()], dim=-1)
        arith_oh = F.one_hot(ar, N_ARITH).float()
        rel_oh = F.one_hot(rl, N_RELS).float()
        op_oh = F.one_hot(op, N_OPS).float()
        x = torch.cat([nums, sb, unks, arith_oh, rel_oh, op_oh], dim=-1)
        return self.head(self.net(x))

# --- Training ---
N=2_000_000
af,bf,df,sb,su,au,bu,du,ar,rl,op,tgt=gen_data(N)
split=int(N*0.8);cw=torch.tensor([1.0,1.0,2.0],device=DEVICE)

model=FlatMLP().to(DEVICE)
params=sum(p.numel() for p in model.parameters())
print(f"  参数量: {params:,}")

opt=optim.AdamW(model.parameters(),lr=1e-3,weight_decay=0.01)
sched=optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS,eta_min=1e-5)
scaler=torch.amp.GradScaler('cuda') if DEVICE.type=='cuda' else None

def get_batch(idx):
    return af[idx],bf[idx],df[idx],sb[idx],su[idx],au[idx],bu[idx],du[idx],ar[idx],rl[idx],op[idx]

best=0.0;total_batches=split//BATCH
for epoch in range(1,EPOCHS+1):
    model.train();perm=torch.randperm(split,device=DEVICE);tl=0.0;nb=0
    for i in range(0,split,BATCH):
        idx=perm[i:i+BATCH]
        with torch.amp.autocast('cuda'):
            loss=F.cross_entropy(model(*get_batch(idx)),tgt[idx],weight=cw)
        opt.zero_grad()
        if scaler:
            scaler.scale(loss).backward();scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            scaler.step(opt);scaler.update()
        else:
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
        tl+=loss.item();nb+=1
        if nb%50==0:
            sys.stdout.write(f"\r  Epoch {epoch:3d} [{nb/total_batches*100:5.1f}%]")
            sys.stdout.flush()
    sched.step()
    if epoch%20==0:
        model.eval()
        with torch.no_grad():
            tidx=torch.arange(split,N,device=DEVICE);preds=[]
            for j in range(0,len(tidx),BATCH*4):
                bi=tidx[j:j+BATCH*4]
                with torch.amp.autocast('cuda'):
                    preds.append(model(*get_batch(bi)).argmax(dim=-1))
            acc=(torch.cat(preds)==tgt[split:]).float().mean().item()
            if acc>best:best=acc
        print(f"\r  Epoch {epoch:3d} | Acc:{acc:.1%} | Best:{best:.1%}         ")
        if best>=1.0-1e-5:break

print(f"\n  训练完成: {best:.1%}")
del af,bf,df,sb,su,au,bu,du,ar,rl,op,tgt
torch.cuda.empty_cache()

# --- Kleene Diagnostic ---
print(f"\n--- Kleene Diagnostic (Flat MLP) ---")

def make_test(vo,vs,logic_op,n=10000):
    a=torch.randint(1,NUM_RANGE+1,(n,),device=DEVICE)
    b=torch.randint(1,NUM_RANGE+1,(n,),device=DEVICE)
    ar=torch.randint(0,N_ARITH,(n,),device=DEVICE)
    c=torch.zeros(n,dtype=torch.long,device=DEVICE)
    c[ar==0]=torch.clamp(a+b,0,NUM_RANGE)[ar==0]
    c[ar==1]=torch.abs(a-b)[ar==1]
    c[ar==2]=torch.clamp(a*b,0,NUM_RANGE)[ar==2]
    c[ar==3]=(a%torch.clamp(b,1,NUM_RANGE))[ar==3]
    c=torch.clamp(c,0,NUM_RANGE)
    au=torch.zeros(n,dtype=torch.bool,device=DEVICE);bu=torch.zeros(n,dtype=torch.bool,device=DEVICE)
    du=torch.zeros(n,dtype=torch.bool,device=DEVICE);su=torch.zeros(n,dtype=torch.bool,device=DEVICE)
    if vo==2:au[:]=True
    if vs==2:su[:]=True
    if vo==1:d=torch.clamp(c-1,min=0);rl=torch.zeros(n,dtype=torch.long,device=DEVICE)
    elif vo==0:d=torch.clamp(c+1,max=NUM_RANGE);rl=torch.zeros(n,dtype=torch.long,device=DEVICE)
    else:d=torch.randint(0,NUM_RANGE+1,(n,),device=DEVICE);rl=torch.zeros(n,dtype=torch.long,device=DEVICE)
    sb=torch.randint(0,2,(n,SET_DIM),dtype=torch.float32,device=DEVICE)
    if vs==1:ci=c.clamp(0,SET_DIM-1);sb[torch.arange(n,device=DEVICE),ci]=1.0
    elif vs==0:ci=c.clamp(0,SET_DIM-1);sb[torch.arange(n,device=DEVICE),ci]=0.0
    op_t=torch.full((n,),logic_op,dtype=torch.long,device=DEVICE)
    target=apply_logic(op_t,torch.full((n,),vo,dtype=torch.long,device=DEVICE),torch.full((n,),vs,dtype=torch.long,device=DEVICE))
    return (a.float()/NUM_RANGE,b.float()/NUM_RANGE,d.float()/NUM_RANGE,sb,su,au,bu,du,ar,rl,op_t,target)

tests=[
    ("F∧U→F",0,2,OP_AND,0),("T∧U→U",1,2,OP_AND,2),
    ("U∧F→F",2,0,OP_AND,0),("U∧T→U",2,1,OP_AND,2),
    ("T∨U→T",1,2,OP_OR,1),("F∨U→U",0,2,OP_OR,2),
    ("U∨T→T",2,1,OP_OR,1),("U∨F→U",2,0,OP_OR,2),
    ("F→U→T",0,2,OP_IMPLIES,1),("T→U→U",1,2,OP_IMPLIES,2),
    ("T↔U→U",1,2,OP_IFF,2),("F↔U→U",0,2,OP_IFF,2),
]

val_names={0:'F',1:'T',2:'U'}

print(f"\n  {'Test':>10} | {'Expected':>8} | {'THEIA':>6} | {'TF 8L':>6} | {'MLP':>6}")
print(f"  {'-'*50}")

tf8l=[100.0,94.1,0.0,100.0,93.5,100.0,0.0,100.0,100.0,94.4,100.0,100.0]
model.eval()
mlp_passed=0
for i,(name,vo,vs,lop,exp) in enumerate(tests):
    data=make_test(vo,vs,lop)
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            preds=model(*data[:-1]).argmax(dim=-1)
    acc=(preds==data[-1]).float().mean().item()*100
    if acc>99:mlp_passed+=1
    m="✅" if acc>99 else "❌"
    print(f"  {name:>10} | {val_names[exp]:>8} | {'100%':>6} | {tf8l[i]:5.1f}% | {m}{acc:5.1f}%")

print(f"\n  Passed: THEIA 12/12 | TF 8L 7/12 | MLP {mlp_passed}/12")
print(f"\n{'='*60}")
if mlp_passed < 12:
    print(f"  MLP也在某些规则上系统性失败")
    print(f"  → monolithic architecture的共同缺陷，不仅是Transformer")
else:
    print(f"  MLP 12/12全过但overall只有99.1%")
    print(f"  → MLP的错误是distributed的，不集中在特定规则")
