"""
Ablation: message-passing rounds vs accuracy.

Loads the already-trained model (isis_v10_ood_best.pth, trained with
MSG_PASSES=10) and overrides the message-pass count at inference time to find
the minimum number of rounds that preserves 100% accuracy.
"""
import torch, torch.nn as nn, torch.nn.functional as F, time

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
D_MODEL=128; BATCH=4096; NUM_RANGE=100
VAL_FALSE=0;VAL_TRUE=1;VAL_UNKNOWN=2;N_VALS=3;N_RELS=6
REL_GT=0;REL_LT=1;REL_EQ=2;REL_GTE=3;REL_LTE=4;REL_NEQ=5

class NumEnc(nn.Module):
    def __init__(self):
        super().__init__()
        self.f=nn.Sequential(nn.Linear(1,D_MODEL//2),nn.GELU(),nn.Linear(D_MODEL//2,D_MODEL))
    def forward(self,x): return self.f(x.unsqueeze(-1))

class DeepOrderEngineOOD(nn.Module):
    def __init__(self, msg_passes=10):
        super().__init__()
        self.msg_passes=msg_passes; self.num_enc=NumEnc()
        self.rel_emb=nn.Embedding(N_RELS,D_MODEL)
        self.msg_mlp=nn.Sequential(nn.Linear(D_MODEL*3,D_MODEL*2),nn.GELU(),nn.LayerNorm(D_MODEL*2),nn.Linear(D_MODEL*2,D_MODEL))
        self.update_mlp=nn.Sequential(nn.Linear(D_MODEL*2,D_MODEL),nn.GELU(),nn.LayerNorm(D_MODEL))
        self.norm=nn.LayerNorm(D_MODEL)
        self.readout=nn.Sequential(nn.Linear(D_MODEL*3,D_MODEL*2),nn.GELU(),nn.Dropout(0.1),nn.LayerNorm(D_MODEL*2),nn.Linear(D_MODEL*2,D_MODEL))
        self.sv=nn.Embedding(N_VALS,D_MODEL); nn.init.orthogonal_(self.sv.weight)
    def forward(self, nodes_f, edges_idx, query_rel, override_passes=None):
        """override_passes: force a different message-pass count at inference."""
        passes = override_passes if override_passes is not None else self.msg_passes
        B,n=nodes_f.shape; H=self.num_enc(nodes_f); E=self.rel_emb(edges_idx)
        for _ in range(passes):
            lm=self.msg_mlp(torch.cat([H[:,1:],E,H[:,:-1]],dim=-1))
            rm=self.msg_mlp(torch.cat([H[:,:-1],E,H[:,1:]],dim=-1))
            tm=torch.zeros_like(H); tm[:,1:]+=lm; tm[:,:-1]+=rm
            H=self.norm(H+self.update_mlp(torch.cat([H,tm],dim=-1)))
        return self.readout(torch.cat([H[:,0],H[:,-1],self.rel_emb(query_rel)],dim=-1))
    def classify(self,v): return (F.normalize(v,dim=-1)@F.normalize(self.sv.weight,dim=-1).T).argmax(dim=-1)

def gen(n,hops):
    nodes=torch.zeros((n,hops+1),dtype=torch.long,device=DEVICE)
    edges=torch.randint(0,N_RELS,(n,hops),device=DEVICE)
    nodes[:,0]=torch.randint(0,NUM_RANGE,(n,),device=DEVICE)
    for h in range(hops):
        c=nodes[:,h]; r=edges[:,h]
        r[(c==0)&(r==REL_GT)]=REL_GTE; r[(c==NUM_RANGE-1)&(r==REL_LT)]=REL_LTE
        mn=torch.zeros_like(c); mx=torch.full_like(c,NUM_RANGE-1)
        m=r==REL_GT;mx[m]=torch.clamp(c[m]-1,min=0)
        m=r==REL_LT;mn[m]=torch.clamp(c[m]+1,max=NUM_RANGE-1)
        m=r==REL_GTE;mx[m]=c[m]; m=r==REL_LTE;mn[m]=c[m]
        m=r==REL_EQ;mn[m]=c[m];mx[m]=c[m]
        cd=mn+(torch.rand(n,device=DEVICE)*(mx-mn+1)).long()
        mq=r==REL_NEQ
        if mq.any():
            c2=torch.randint(0,NUM_RANGE,(mq.sum().item(),),device=DEVICE)
            c2[c2==c[mq]]=(c2[c2==c[mq]]+1)%NUM_RANGE; cd[mq]=c2
        nodes[:,h+1]=cd; edges[:,h]=r
    q=torch.randint(0,N_RELS,(n,),device=DEVICE); hd=nodes[:,0]; tl=nodes[:,-1]
    tgt=torch.where(((q==0)&(hd>tl))|((q==1)&(hd<tl))|((q==2)&(hd==tl))|((q==3)&(hd>=tl))|((q==4)&(hd<=tl))|((q==5)&(hd!=tl)),
        torch.tensor(1,device=DEVICE),torch.tensor(0,device=DEVICE))
    return nodes.float()/NUM_RANGE, edges, q, tgt

model=DeepOrderEngineOOD().to(DEVICE)
model.load_state_dict(torch.load('isis_v10_ood_best.pth',map_location=DEVICE,weights_only=True))
model.eval()
print(f"✅ 模型已加载 | 设备: {DEVICE}")

# --- Ablation 1: fixed 5-hop data, vary message-pass count ---
print(f"\n{'='*60}")
print(f"  消融1: 消息传递轮数 vs 准确率（5跳数据）")
print(f"  模型训练时 MSG_PASSES=10，推理时强制改")
print(f"{'='*60}")

nd,ed,qr,tg = gen(100_000, 5)
for passes in [1, 2, 3, 4, 5, 6, 7, 8, 10, 15, 20]:
    preds=[]
    with torch.no_grad():
        for i in range(0,100_000,BATCH):
            j=min(i+BATCH,100_000)
            with torch.amp.autocast('cuda'):
                preds.append(model.classify(model(nd[i:j],ed[i:j],qr[i:j],override_passes=passes)))
    acc=(torch.cat(preds)==tg).float().mean().item()*100
    marker="✅" if acc>99 else("🟡" if acc>90 else "🔴")
    print(f"  {marker} {passes:2d}轮 | 准确率: {acc:6.2f}%")
del nd,ed,qr,tg

# --- Ablation 2: hop count x message-pass count cross table ---
print(f"\n{'='*60}")
print(f"  消融2: 跳数 × 消息传递轮数 交叉表")
print(f"{'='*60}")

print(f"  {'跳数':>6}", end="")
for p in [3, 5, 7, 10, 15]:
    print(f" | {p}轮", end="")
print()
print(f"  {'-'*45}")

for hops in [5, 10, 20, 50]:
    nd,ed,qr,tg = gen(50_000, hops)
    print(f"  {hops:4d}跳", end="")
    for passes in [3, 5, 7, 10, 15]:
        preds=[]
        bs = min(BATCH, max(256, 100000//hops))
        with torch.no_grad():
            for i in range(0,50_000,bs):
                j=min(i+bs,50_000)
                with torch.amp.autocast('cuda'):
                    preds.append(model.classify(model(nd[i:j],ed[i:j],qr[i:j],override_passes=passes)))
        acc=(torch.cat(preds)==tg).float().mean().item()*100
        marker="✅" if acc>99 else("🟡" if acc>90 else "🔴")
        print(f" |{marker}{acc:5.1f}%", end="")
    print()
    del nd,ed,qr,tg
    if DEVICE.type=='cuda': torch.cuda.empty_cache()

print(f"\n{'='*60}")
print("  这张交叉表直接放论文")
print("  它回答：'模型需要多少轮思考才能推理N跳链？'")
print(f"{'='*60}")
