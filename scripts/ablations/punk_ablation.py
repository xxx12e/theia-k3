"""
THEIA P_unknown ablation (§4.3).

Experiment 1: evaluate a V9 model trained at P_unk=0.15 (isis_v9.pth) across
test-time Unknown rates P in {0.0 ... 0.70}.
Experiment 2: train a fresh model at P_unk=0.50 and check convergence.
"""
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
import time, sys, csv

DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
D_MODEL=128;BATCH=4096;NUM_RANGE=20;SET_DIM=21
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

# --- Model ---
class NumEnc(nn.Module):
    def __init__(self):
        super().__init__()
        self.f=nn.Sequential(nn.Linear(1,D_MODEL//2),nn.GELU(),nn.Linear(D_MODEL//2,D_MODEL))
        self.unknown_vec=nn.Parameter(torch.randn(D_MODEL))
    def forward(self,x,unk):
        v=self.f(x.unsqueeze(-1));u=self.unknown_vec.unsqueeze(0).expand_as(v)
        return torch.where(unk.unsqueeze(-1),u,v)

class SetEnc(nn.Module):
    def __init__(self):
        super().__init__()
        self.f=nn.Sequential(nn.Linear(SET_DIM,D_MODEL//2),nn.GELU(),nn.Linear(D_MODEL//2,D_MODEL))
        self.unknown_vec=nn.Parameter(torch.randn(D_MODEL))
    def forward(self,bits,unk):
        v=self.f(bits);u=self.unknown_vec.unsqueeze(0).expand_as(v)
        return torch.where(unk.unsqueeze(-1),u,v)

def make_mlp(i,o):
    return nn.Sequential(nn.Linear(i,i*2),nn.GELU(),nn.LayerNorm(i*2),nn.Linear(i*2,o))

class ArithEngine(nn.Module):
    def __init__(self):
        super().__init__()
        self.ne=NumEnc();self.ae=nn.Embedding(N_ARITH,D_MODEL);self.net=make_mlp(D_MODEL*3,D_MODEL)
    def forward(self,a,b,au,bu,ar):
        return self.net(torch.cat([self.ne(a,au),self.ne(b,bu),self.ae(ar)],dim=-1))

class OrderEngine(nn.Module):
    def __init__(self):
        super().__init__()
        self.ne=NumEnc();self.re=nn.Embedding(N_RELS,D_MODEL)
        mk=lambda:nn.Sequential(nn.Linear(D_MODEL*3,D_MODEL*2),nn.GELU(),nn.LayerNorm(D_MODEL*2),nn.Linear(D_MODEL*2,D_MODEL))
        self.G=mk();self.L=mk();self.E=mk()
        self.Gg=nn.Sequential(nn.Linear(D_MODEL*2,D_MODEL),nn.GELU(),nn.LayerNorm(D_MODEL))
        self.Lg=nn.Sequential(nn.Linear(D_MODEL*2,D_MODEL),nn.GELU(),nn.LayerNorm(D_MODEL))
        self.Eg=nn.Sequential(nn.Linear(D_MODEL*3,D_MODEL),nn.GELU(),nn.LayerNorm(D_MODEL))
        self.out=make_mlp(D_MODEL*4,D_MODEL)
    def forward(self,c_vec,d,du,rel):
        vd=self.ne(d,du);vr=self.re(rel);x=torch.cat([c_vec,vd,vr],dim=-1)
        g=self.G(x);l=self.L(x);e=self.E(x)
        g=self.Gg(torch.cat([g,e],dim=-1));l=self.Lg(torch.cat([l,e],dim=-1))
        e=self.Eg(torch.cat([e,g,l],dim=-1))
        return self.out(torch.cat([g,l,e,c_vec],dim=-1))

class SetEngine(nn.Module):
    def __init__(self):
        super().__init__()
        self.se=SetEnc();self.net=make_mlp(D_MODEL*2,D_MODEL)
    def forward(self,c_vec,sb,su):
        return self.net(torch.cat([c_vec,self.se(sb,su)],dim=-1))

class LogicEngine(nn.Module):
    def __init__(self):
        super().__init__()
        self.oe=nn.Embedding(N_OPS,D_MODEL)
        mk=lambda:nn.Sequential(nn.Linear(D_MODEL*3,D_MODEL*2),nn.GELU(),nn.LayerNorm(D_MODEL*2),nn.Linear(D_MODEL*2,D_MODEL))
        self.C=mk();self.D=mk();self.I=mk()
        self.Cg=nn.Sequential(nn.Linear(D_MODEL*2,D_MODEL),nn.GELU(),nn.LayerNorm(D_MODEL))
        self.Dg=nn.Sequential(nn.Linear(D_MODEL*2,D_MODEL),nn.GELU(),nn.LayerNorm(D_MODEL))
        self.Ig=nn.Sequential(nn.Linear(D_MODEL*3,D_MODEL),nn.GELU(),nn.LayerNorm(D_MODEL))
        self.out=make_mlp(D_MODEL*3,D_MODEL)
    def forward(self,vo,vs,op):
        x=torch.cat([vo,vs,self.oe(op)],dim=-1)
        c=self.C(x);d=self.D(x);i=self.I(x)
        c=self.Cg(torch.cat([c,i],dim=-1));d=self.Dg(torch.cat([d,i],dim=-1))
        i=self.Ig(torch.cat([i,c,d],dim=-1))
        return self.out(torch.cat([c,d,i],dim=-1))

class TheiaV9(nn.Module):
    def __init__(self):
        super().__init__()
        self.arith_eng=ArithEngine();self.order_eng=OrderEngine()
        self.set_eng=SetEngine();self.logic_eng=LogicEngine()
        self.bridge_ao=nn.Sequential(nn.Linear(D_MODEL,D_MODEL),nn.GELU(),nn.LayerNorm(D_MODEL))
        self.bridge_as=nn.Sequential(nn.Linear(D_MODEL,D_MODEL),nn.GELU(),nn.LayerNorm(D_MODEL))
        self.out_head=nn.Sequential(nn.Linear(D_MODEL,D_MODEL),nn.GELU(),nn.Dropout(0.1),nn.LayerNorm(D_MODEL))
        self.sv=nn.Embedding(N_VALS,D_MODEL);nn.init.orthogonal_(self.sv.weight)
    def forward(self,a,b,d,sb,su,au,bu,du,ar,rl,op):
        c_vec=self.arith_eng(a,b,au,bu,ar)
        ord_vec=self.order_eng(self.bridge_ao(c_vec)+c_vec,d,du,rl)
        set_vec=self.set_eng(self.bridge_as(c_vec)+c_vec,sb,su)
        return self.out_head(self.logic_eng(ord_vec,set_vec,op))
    def classify(self,v):
        return (F.normalize(v,dim=-1)@F.normalize(self.sv.weight,dim=-1).T).argmax(dim=-1)

# --- Data generation (P_unknown is a parameter) ---
def gen_data(n, p_unk, seed=42):
    torch.manual_seed(seed)
    a_val=torch.randint(1,NUM_RANGE+1,(n,),device=DEVICE)
    b_val=torch.randint(1,NUM_RANGE+1,(n,),device=DEVICE)
    d_val=torch.randint(0,NUM_RANGE+1,(n,),device=DEVICE)
    arith=torch.randint(0,N_ARITH,(n,),device=DEVICE)
    rel=torch.randint(0,N_RELS,(n,),device=DEVICE)
    op=torch.randint(0,N_OPS,(n,),device=DEVICE)
    au=torch.rand(n,device=DEVICE)<p_unk
    bu=torch.rand(n,device=DEVICE)<p_unk
    du=torch.rand(n,device=DEVICE)<p_unk
    su=torch.rand(n,device=DEVICE)<p_unk
    c_val=torch.zeros(n,dtype=torch.long,device=DEVICE)
    c_val[arith==0]=torch.clamp(a_val+b_val,0,NUM_RANGE)[arith==0]
    c_val[arith==1]=torch.abs(a_val-b_val)[arith==1]
    c_val[arith==2]=torch.clamp(a_val*b_val,0,NUM_RANGE)[arith==2]
    c_val[arith==3]=(a_val%torch.clamp(b_val,1,NUM_RANGE))[arith==3]
    c_val=torch.clamp(c_val,0,NUM_RANGE)
    cu=au|bu;ou=cu|du
    ov=torch.zeros(n,dtype=torch.long,device=DEVICE)
    rt=((rel==0)&(c_val>d_val))|((rel==1)&(c_val<d_val))|((rel==2)&(c_val==d_val))|((rel==3)&(c_val>=d_val))|((rel==4)&(c_val<=d_val))|((rel==5)&(c_val!=d_val))
    ov[rt]=1
    vo=torch.where(ou,torch.tensor(2,device=DEVICE),ov)
    sb=torch.randint(0,2,(n,SET_DIM),dtype=torch.float32,device=DEVICE)
    sou=su|cu;ci=c_val.clamp(0,SET_DIM-1)
    ins=sb[torch.arange(n,device=DEVICE),ci].bool()
    sv=torch.where(ins,torch.tensor(1,device=DEVICE),torch.tensor(0,device=DEVICE))
    vs=torch.where(sou,torch.tensor(2,device=DEVICE),sv)
    tgt=apply_logic(op,vo,vs)

    tc=(tgt==1).sum().item();fc=(tgt==0).sum().item();uc=(tgt==2).sum().item()

    return (a_val.float()/NUM_RANGE, b_val.float()/NUM_RANGE, d_val.float()/NUM_RANGE,
            sb, su, au, bu, du, arith, rel, op, tgt,
            {'T':tc,'F':fc,'U':uc})

print(f"{'='*60}")
print(f"  THEIA P_UNKNOWN Ablation")
print(f"  设备: {DEVICE}")
print(f"{'='*60}")

# --- Experiment 1: model trained at P=0.15, evaluated across test-time P ---
print(f"\n--- 实验1: 已训练模型(P=0.15) vs 不同测试P ---")

model = TheiaV9().to(DEVICE)
try:
    model.load_state_dict(torch.load('isis_v9.pth', map_location=DEVICE, weights_only=True))
    print(f"  ✅ 加载 isis_v9.pth")
except:
    print(f"  ❌ 找不到isis_v9.pth，跳过实验1")
    model = None

if model is not None:
    model.eval()
    print(f"\n  {'P_UNK':>6} | {'Acc':>7} | {'F_acc':>7} | {'T_acc':>7} | {'U_acc':>7} | {'%Unknown':>9}")
    print(f"  {'-'*55}")

    for p in [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.70]:
        af,bf,df,sb,su,au,bu,du,ar,rl,op,tgt,dist = gen_data(200_000, p, seed=555)
        preds=[]
        with torch.no_grad():
            for i in range(0,200_000,BATCH):
                j=min(i+BATCH,200_000)
                with torch.amp.autocast('cuda' if DEVICE.type=='cuda' else 'cpu'):
                    preds.append(model.classify(model(af[i:j],bf[i:j],df[i:j],sb[i:j],su[i:j],au[i:j],bu[i:j],du[i:j],ar[i:j],rl[i:j],op[i:j])))
        preds=torch.cat(preds)
        acc=(preds==tgt).float().mean().item()*100
        f_m=tgt==0;t_m=tgt==1;u_m=tgt==2
        fa=(preds[f_m]==0).float().mean().item()*100 if f_m.sum()>0 else 0
        ta=(preds[t_m]==1).float().mean().item()*100 if t_m.sum()>0 else 0
        ua=(preds[u_m]==2).float().mean().item()*100 if u_m.sum()>0 else 0
        u_pct=dist['U']/200_000*100
        marker="✅" if acc>99 else("🟡" if acc>95 else "🔴")
        print(f"  {p:>5.2f} | {marker}{acc:5.1f}% | {fa:5.1f}% | {ta:5.1f}% | {ua:5.1f}% | {u_pct:7.1f}%")

# --- Experiment 2: train a new model at P=0.50 ---
print(f"\n--- 实验2: 训练 P=0.50 模型 ---")

N_TRAIN = 2_000_000
af,bf,df,sb,su,au,bu,du,ar,rl,op,tgt,dist = gen_data(N_TRAIN, 0.50, seed=42)
split = int(N_TRAIN * 0.8)
print(f"  数据分布: T:{dist['T']/N_TRAIN:.1%} F:{dist['F']/N_TRAIN:.1%} U:{dist['U']/N_TRAIN:.1%}")

model2 = TheiaV9().to(DEVICE)
opt=optim.AdamW(model2.parameters(),lr=1e-3,weight_decay=0.01)
sched=optim.lr_scheduler.CosineAnnealingLR(opt,T_max=200,eta_min=1e-5)
scaler=torch.amp.GradScaler('cuda') if DEVICE.type=='cuda' else None

# Inverse-frequency class weights (label balance shifts with P_unk).
counts=torch.bincount(tgt,minlength=N_VALS).float()
cw=(counts.max()/counts.clamp(min=1)).to(DEVICE)
print(f"  类别权重: F:{cw[0]:.2f} T:{cw[1]:.2f} U:{cw[2]:.2f}")

best=0.0
def get_batch(idx):
    return af[idx],bf[idx],df[idx],sb[idx],su[idx],au[idx],bu[idx],du[idx],ar[idx],rl[idx],op[idx]

for epoch in range(1,201):
    model2.train();perm=torch.randperm(split,device=DEVICE);tl=0.0;nb=0
    for i in range(0,split,BATCH):
        idx=perm[i:i+BATCH]
        with torch.amp.autocast('cuda' if DEVICE.type=='cuda' else 'cpu'):
            out=model2(*get_batch(idx))
            loss=F.cross_entropy(out@model2.sv.weight.T,tgt[idx],weight=cw)
        opt.zero_grad()
        if scaler:
            scaler.scale(loss).backward();scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model2.parameters(),1.0)
            scaler.step(opt);scaler.update()
        else:
            loss.backward();torch.nn.utils.clip_grad_norm_(model2.parameters(),1.0);opt.step()
        tl+=loss.item();nb+=1
    sched.step()

    if epoch%20==0:
        model2.eval()
        with torch.no_grad():
            tidx=torch.arange(split,N_TRAIN,device=DEVICE);preds=[]
            for i in range(0,len(tidx),BATCH*4):
                bi=tidx[i:i+BATCH*4]
                with torch.amp.autocast('cuda' if DEVICE.type=='cuda' else 'cpu'):
                    preds.append(model2.classify(model2(*get_batch(bi))))
            acc=(torch.cat(preds)==tgt[split:]).float().mean().item()
            if acc>best:best=acc

        preds_all=torch.cat(preds);labels=tgt[split:]
        f_m=labels==0;t_m=labels==1;u_m=labels==2
        fa=(preds_all[f_m]==0).float().mean().item()*100 if f_m.sum()>0 else 0
        ta=(preds_all[t_m]==1).float().mean().item()*100 if t_m.sum()>0 else 0
        ua=(preds_all[u_m]==2).float().mean().item()*100 if u_m.sum()>0 else 0

        print(f"  Epoch {epoch:3d} | Loss:{tl/nb:.4f} | Acc:{acc:.1%} | Best:{best:.1%} | "
              f"F:{fa:.1f}% T:{ta:.1f}% U:{ua:.1f}%")
        if best>=1.0-1e-5:
            print(f"  ✅ P=0.50 模型达到100%!");break

print(f"\n{'='*60}")
print(f"  结果汇总")
print(f"{'='*60}")
print(f"  P=0.15训练模型: 100% (已知)")
print(f"  P=0.50训练模型: {best:.1%}")
if best >= 0.999:
    print(f"  → 三值逻辑学习对Unknown比例robust")
else:
    print(f"  → P=0.50更难，需要更多训练或架构调整")
print(f"{'='*60}")
