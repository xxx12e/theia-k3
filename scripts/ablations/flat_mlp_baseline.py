"""
Flat MLP baseline (backbone-ablation appendix): same four-domain task, but all
inputs are concatenated and fed to one monolithic MLP, parameter-matched to
THEIA V9's 2,751,232 params.
"""

import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
import time, sys

DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
D_MODEL=128; BATCH=4096; EPOCHS=200; LR=1e-3; NUM_RANGE=20; SET_DIM=21; P_UNKNOWN=0.15
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
print(f"  THEIA Flat MLP Baseline")
print(f"  Same task: single flat MLP vs modular architecture")
print(f"  Device: {DEVICE}")
print(f"{'='*60}")

# --- Data generation (identical to V9) ---
def build_dataset():
    N=2_000_000
    print(f"Generating {N//10000}x10k samples...")
    a_val=torch.randint(1,NUM_RANGE+1,(N,),device=DEVICE)
    b_val=torch.randint(1,NUM_RANGE+1,(N,),device=DEVICE)
    d_val=torch.randint(0,NUM_RANGE+1,(N,),device=DEVICE)
    arith=torch.randint(0,N_ARITH,(N,),device=DEVICE)
    rel=torch.randint(0,N_RELS,(N,),device=DEVICE)
    op=torch.randint(0,N_OPS,(N,),device=DEVICE)
    a_unknown=torch.rand(N,device=DEVICE)<P_UNKNOWN
    b_unknown=torch.rand(N,device=DEVICE)<P_UNKNOWN
    d_unknown=torch.rand(N,device=DEVICE)<P_UNKNOWN
    s_unknown=torch.rand(N,device=DEVICE)<P_UNKNOWN
    c_val=torch.zeros(N,dtype=torch.long,device=DEVICE)
    c_val[arith==0]=torch.clamp(a_val+b_val,0,NUM_RANGE)[arith==0]
    c_val[arith==1]=torch.abs(a_val-b_val)[arith==1]
    c_val[arith==2]=torch.clamp(a_val*b_val,0,NUM_RANGE)[arith==2]
    c_val[arith==3]=(a_val%torch.clamp(b_val,1,NUM_RANGE))[arith==3]
    c_val=torch.clamp(c_val,0,NUM_RANGE)
    c_unknown=a_unknown|b_unknown
    ord_unknown=c_unknown|d_unknown
    ord_val=torch.zeros(N,dtype=torch.long,device=DEVICE)
    rel_true=((rel==0)&(c_val>d_val))|((rel==1)&(c_val<d_val))|((rel==2)&(c_val==d_val))|((rel==3)&(c_val>=d_val))|((rel==4)&(c_val<=d_val))|((rel==5)&(c_val!=d_val))
    ord_val[rel_true]=1
    val_ord=torch.where(ord_unknown,torch.tensor(2,device=DEVICE),ord_val)
    set_bits=torch.randint(0,2,(N,SET_DIM),dtype=torch.float32,device=DEVICE)
    set_op_unknown=s_unknown|c_unknown
    c_idx=c_val.clamp(0,SET_DIM-1)
    in_set=set_bits[torch.arange(N,device=DEVICE),c_idx].bool()
    set_val=torch.where(in_set,torch.tensor(1,device=DEVICE),torch.tensor(0,device=DEVICE))
    val_set=torch.where(set_op_unknown,torch.tensor(2,device=DEVICE),set_val)
    target=apply_logic(op,val_ord,val_set)
    perm=torch.randperm(N,device=DEVICE)
    return (a_val[perm].float()/NUM_RANGE, b_val[perm].float()/NUM_RANGE,
            d_val[perm].float()/NUM_RANGE, set_bits[perm], s_unknown[perm],
            a_unknown[perm], b_unknown[perm], d_unknown[perm],
            arith[perm], rel[perm], op[perm], target[perm], N)

AF,BF,DF,SB,S_UNK,A_UNK,B_UNK,D_UNK,AR,RL,OP,TARGET,N=build_dataset()
split=int(N*0.8)

# --- Flat MLP (all inputs concatenated into one vector) ---
class FlatMLP(nn.Module):
    """
    Input: a(1) + b(1) + d(1) + set_bits(21) + s_unk(1) + a_unk(1) + b_unk(1) + d_unk(1)
         + arith_onehot(4) + rel_onehot(6) + op_onehot(5) = 43 dims.

    Parameter-matched against V9's 2.75M:
    43→512 (22K) + 512→1024 (525K) + 1024→1024 (1.05M) + 1024→512 (525K) + 512→256 (131K) + 256→128 (33K) + 128→3 (384)
    ≈ 2.29M + LayerNorm/biases ≈ ~2.5M (close to V9's 2.75M).
    """
    def __init__(self):
        super().__init__()
        input_dim = 3 + SET_DIM + 4 + N_ARITH + N_RELS + N_OPS  # 3+21+4+4+6+5=43
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512), nn.GELU(), nn.LayerNorm(512),
            nn.Linear(512, 1024), nn.GELU(), nn.LayerNorm(1024),
            nn.Linear(1024, 1024), nn.GELU(), nn.LayerNorm(1024), nn.Dropout(0.1),
            nn.Linear(1024, 512), nn.GELU(), nn.LayerNorm(512),
            nn.Linear(512, 256), nn.GELU(), nn.LayerNorm(256),
            nn.Linear(256, 128), nn.GELU(), nn.LayerNorm(128),
        )
        self.head = nn.Linear(128, N_VALS)
    
    def forward(self, a, b, d, set_bits, s_unk, a_unk, b_unk, d_unk, arith, rel, op):
        nums = torch.stack([a, b, d], dim=-1)  # [B, 3]
        unks = torch.stack([s_unk.float(), a_unk.float(), b_unk.float(), d_unk.float()], dim=-1)  # [B, 4]
        arith_oh = F.one_hot(arith, N_ARITH).float()  # [B, 4]
        rel_oh = F.one_hot(rel, N_RELS).float()        # [B, 6]
        op_oh = F.one_hot(op, N_OPS).float()            # [B, 5]
        x = torch.cat([nums, set_bits, unks, arith_oh, rel_oh, op_oh], dim=-1)  # [B, 43]
        return self.head(self.net(x))  # [B, 3] logits

def get_batch(idx):
    return AF[idx],BF[idx],DF[idx],SB[idx],S_UNK[idx],A_UNK[idx],B_UNK[idx],D_UNK[idx],AR[idx],RL[idx],OP[idx]

# --- Training ---
model = FlatMLP().to(DEVICE)
params = sum(p.numel() for p in model.parameters())
print(f"  Flat MLP parameters: {params:,}")
print(f"  THEIA V9 parameters: 2,751,232")
print(f"  Ratio: {params/2751232:.1%}")

opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-5)
scaler = torch.amp.GradScaler('cuda') if DEVICE.type=='cuda' else None
cw = torch.tensor([1.0, 1.0, 2.0], device=DEVICE)

best_acc = 0.0
print(f"\n  Training start | Epochs: {EPOCHS}")

for epoch in range(1, EPOCHS+1):
    model.train()
    perm = torch.randperm(split, device=DEVICE)
    tl=0.0; nb=0
    
    for i in range(0, split, BATCH):
        idx = perm[i:i+BATCH]
        with torch.amp.autocast('cuda' if DEVICE.type=='cuda' else 'cpu'):
            logits = model(*get_batch(idx))
            loss = F.cross_entropy(logits, TARGET[idx], weight=cw)
        opt.zero_grad()
        if scaler:
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
        else:
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        tl += loss.item(); nb += 1
    
    sched.step()
    
    if epoch % 20 == 0:
        model.eval()
        with torch.no_grad():
            tidx = torch.arange(split, N, device=DEVICE); preds=[]
            for i in range(0, len(tidx), BATCH*4):
                bi = tidx[i:i+BATCH*4]
                with torch.amp.autocast('cuda' if DEVICE.type=='cuda' else 'cpu'):
                    preds.append(model(*get_batch(bi)).argmax(dim=-1))
            preds = torch.cat(preds); labels = TARGET[split:N]
            acc = (preds==labels).float().mean().item()
            if acc > best_acc: best_acc = acc
        
        f_m=labels==0; t_m=labels==1; u_m=labels==2
        f_a=(preds[f_m]==0).float().mean().item()*100 if f_m.sum()>0 else 0
        t_a=(preds[t_m]==1).float().mean().item()*100 if t_m.sum()>0 else 0
        u_a=(preds[u_m]==2).float().mean().item()*100 if u_m.sum()>0 else 0
        
        print(f"  Epoch {epoch:3d} | Loss:{tl/nb:.4f} | Acc:{acc:.1%} | Best:{best_acc:.1%} | "
              f"F:{f_a:.1f}% T:{t_a:.1f}% U:{u_a:.1f}%")
        
        if best_acc >= 1.0 - 1e-6:
            print(f"  Flat MLP reached 100% at epoch {epoch}")
            break

# --- Results ---
print(f"\n{'='*60}")
print(f"  Results comparison")
print(f"{'='*60}")
print(f"  THEIA V9 (modular):  100.0% | 2,751,232 params | ~100 epochs")
print(f"  Flat MLP (baseline): {best_acc:.1%} | {params:,} params | {EPOCHS} epochs")
print(f"{'='*60}")

if best_acc >= 0.999:
    print("  Conclusion: flat MLP also reaches 100%")
    print("  → modular architecture is not necessary for accuracy")
    print("  → modularity may still offer other advantages (interpretability, extensibility)")
    print("  → modular design enables interpretability analysis")
else:
    print(f"  Conclusion: flat MLP plateaus at {best_acc:.1%}; modular architecture is necessary")
    print(f"  → Gap: {(1.0-best_acc)*100:.1f}%")
    print("  → modular architecture is essential for perfect reasoning on this task")
    print("  → strengthens the architecture-contribution analysis")
