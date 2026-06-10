"""
THEIA Multi-Hop v4: modular-arithmetic chain reasoning (the paper's mod-5 on
graphs negative result). Each edge applies +k mod 5; the label asks whether the
cumulative sum mod 5 equals a query value. There are no absorbing states and
the label is not inferable from any local prefix — message passing must
aggregate information globally along the chain. True occurs with probability
1/P (20% at P=5).
"""
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
import time, sys

DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
D_MODEL=128;BATCH=2048;EPOCHS=300;MSG_PASSES=10
N_VALS=2  # T/F
P_MOD=5   # modulus; 5 operations +0,+1,+2,+3,+4

print(f"{'='*60}")
print(f"  THEIA Multi-Hop v4: 模运算链推理")
print(f"  mod {P_MOD} | 设备: {DEVICE}")
print(f"{'='*60}")

def gen_modular_chain(n, hops, seed=42):
    torch.manual_seed(seed)
    # each edge is an operation +k mod P
    edges = torch.randint(0, P_MOD, (n, hops), device=DEVICE)

    # cumulative sum along the chain, mod P
    acc = edges.sum(dim=1) % P_MOD  # [n]

    query = torch.randint(0, P_MOD, (n,), device=DEVICE)

    # label: cumulative value == query?
    labels = (acc == query).long()

    # nodes carry only positional encoding (no value information)
    positions = torch.arange(hops+1, device=DEVICE).float() / max(hops, 1)
    positions = positions.unsqueeze(0).expand(n, -1)
    
    return positions, edges, query, labels, acc

# --- Label distribution analysis ---
print(f"\n  --- 数据分布分析 ---")
for hops in [5, 10, 20, 50]:
    _, te, _, tl, ta = gen_modular_chain(100_000, hops, seed=123)
    t_pct = (tl==1).sum().item()/100_000*100
    
    # distribution of cumulative values
    dist = torch.bincount(ta, minlength=P_MOD).float() / 100_000 * 100
    dist_str = " ".join([f"{i}:{dist[i]:.1f}%" for i in range(P_MOD)])

    # count of nonzero edges (+0 is the identity, leaves state unchanged)
    non_zero = (te != 0).float().sum(dim=1).mean().item()
    
    print(f"  {hops:3d}跳 | T:{t_pct:.1f}% | 非零边:{non_zero:.1f}/{hops} | 累积分布: {dist_str}")

# --- Model ---
class ModularChainReasoner(nn.Module):
    def __init__(self, msg_passes=MSG_PASSES):
        super().__init__()
        self.msg_passes=msg_passes
        self.pos_enc=nn.Sequential(nn.Linear(1,D_MODEL//2),nn.GELU(),nn.Linear(D_MODEL//2,D_MODEL))
        self.op_emb=nn.Embedding(P_MOD, D_MODEL)      # edge operation
        self.query_emb=nn.Embedding(P_MOD, D_MODEL)    # query value
        self.msg_mlp=nn.Sequential(nn.Linear(D_MODEL*3,D_MODEL*2),nn.GELU(),nn.LayerNorm(D_MODEL*2),nn.Linear(D_MODEL*2,D_MODEL))
        self.update_mlp=nn.Sequential(nn.Linear(D_MODEL*2,D_MODEL),nn.GELU(),nn.LayerNorm(D_MODEL))
        self.norm=nn.LayerNorm(D_MODEL)
        self.readout=nn.Sequential(nn.Linear(D_MODEL*3,D_MODEL*2),nn.GELU(),nn.Dropout(0.1),nn.LayerNorm(D_MODEL*2),nn.Linear(D_MODEL*2,D_MODEL))
        self.head=nn.Linear(D_MODEL, N_VALS)

    def forward(self, positions, edges_idx, query_val):
        B,n=positions.shape
        H=self.pos_enc(positions.unsqueeze(-1))
        E=self.op_emb(edges_idx)
        for _ in range(self.msg_passes):
            lm=self.msg_mlp(torch.cat([H[:,1:],E,H[:,:-1]],dim=-1))
            rm=self.msg_mlp(torch.cat([H[:,:-1],E,H[:,1:]],dim=-1))
            tm=torch.zeros_like(H);tm[:,1:]+=lm;tm[:,:-1]+=rm
            H=self.norm(H+self.update_mlp(torch.cat([H,tm],dim=-1)))
        return self.head(self.readout(torch.cat([H[:,0],H[:,-1],self.query_emb(query_val)],dim=-1)))

# --- Training ---
TRAIN_HOPS = 5
N_DATA = 1_000_000
print(f"\n  训练 {TRAIN_HOPS}跳 mod-{P_MOD} 链推理...")

positions, edges, query, labels, _ = gen_modular_chain(N_DATA, TRAIN_HOPS, seed=42)
split = int(N_DATA * 0.8)

counts=torch.bincount(labels, minlength=N_VALS).float()
weights=(counts.max()/counts.clamp(min=1)).to(DEVICE)
print(f"  T/F: T:{(labels==1).sum().item()/N_DATA:.1%} F:{(labels==0).sum().item()/N_DATA:.1%}")
print(f"  权重: F:{weights[0]:.2f} T:{weights[1]:.2f}")

model=ModularChainReasoner().to(DEVICE)
params=sum(p.numel() for p in model.parameters())
print(f"  参数量: {params:,}")

opt=optim.AdamW(model.parameters(),lr=1e-3,weight_decay=0.01)
sched=optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS,eta_min=1e-5)
scaler=torch.amp.GradScaler('cuda') if DEVICE.type=='cuda' else None

best=0.0
for epoch in range(1,EPOCHS+1):
    model.train();perm=torch.randperm(split,device=DEVICE);tl=0.0;nb=0
    for i in range(0,split,BATCH):
        idx=perm[i:i+BATCH]
        with torch.amp.autocast('cuda' if DEVICE.type=='cuda' else 'cpu'):
            loss=F.cross_entropy(model(positions[idx],edges[idx],query[idx]),labels[idx],weight=weights)
        opt.zero_grad()
        if scaler:
            scaler.scale(loss).backward();scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            scaler.step(opt);scaler.update()
        else:
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
        tl+=loss.item();nb+=1
    sched.step()

    if epoch%10==0:
        model.eval()
        with torch.no_grad():
            tidx=torch.arange(split,N_DATA,device=DEVICE);preds=[]
            for j in range(0,len(tidx),BATCH*4):
                bi=tidx[j:j+BATCH*4]
                with torch.amp.autocast('cuda' if DEVICE.type=='cuda' else 'cpu'):
                    preds.append(model(positions[bi],edges[bi],query[bi]).argmax(dim=-1))
            acc=(torch.cat(preds)==labels[split:]).float().mean().item()
            if acc>best:
                best=acc
                torch.save(model.state_dict(),'theia_chain_v4_best.pth')
        print(f"  Epoch {epoch:3d}/{EPOCHS} | Loss:{tl/nb:.4f} | Acc:{acc:.1%} | Best:{best:.1%}")
        if best>=1.0-1e-5:
            print(f"  ✅ 100%! Epoch {epoch}");break

del positions,edges,query,labels
if DEVICE.type=='cuda':torch.cuda.empty_cache()

# --- OOD eval + chain-structure validation ---
if best > 0.85:
    print(f"\n--- OOD泛化 ---")
    model.load_state_dict(torch.load('theia_chain_v4_best.pth',map_location=DEVICE,weights_only=True))
    model.eval()
    for hops in [3, 5, 10, 15, 20, 30, 50]:
        n_test=min(100_000,500_000//max(hops,1))
        tp,te,tq,tl,_=gen_modular_chain(n_test,hops,seed=999)
        preds=[]
        bs=min(BATCH,max(256,200000//max(hops,1)))
        with torch.no_grad():
            for i in range(0,n_test,bs):
                j=min(i+bs,n_test)
                with torch.amp.autocast('cuda' if DEVICE.type=='cuda' else 'cpu'):
                    preds.append(model(tp[i:j],te[i:j],tq[i:j]).argmax(dim=-1))
        acc=(torch.cat(preds)==tl).float().mean().item()*100
        marker="✅" if acc>95 else("🟡" if acc>80 else "🔴")
        print(f"  {marker} {hops:3d}跳 | Acc:{acc:6.2f}%")
        del tp,te,tq,tl
        if DEVICE.type=='cuda':torch.cuda.empty_cache()

    # Chain-structure validation (20 hops): shuffle middle edges across samples
    # and recompute labels; similar accuracy means the model composes the mod
    # operation rather than relying on positional cues.
    print(f"\n--- 链结构验证（20跳）---")
    tp,te,tq,tl,_=gen_modular_chain(100_000,20,seed=888)

    preds=[]
    with torch.no_grad():
        for i in range(0,100_000,BATCH):
            j=min(i+BATCH,100_000)
            with torch.amp.autocast('cuda'):
                preds.append(model(tp[i:j],te[i:j],tq[i:j]).argmax(dim=-1))
    normal=(torch.cat(preds)==tl).float().mean().item()*100
    
    # shuffle middle edges across samples
    se=te.clone()
    mid=se[:,1:-1]
    for i in range(mid.shape[1]):
        mid[:,i]=mid[torch.randperm(100_000,device=DEVICE),i]
    se[:,1:-1]=mid
    # recompute labels after shuffling
    new_acc = se.sum(dim=1) % P_MOD
    new_labels = (tq == new_acc).long()
    
    preds2=[]
    with torch.no_grad():
        for i in range(0,100_000,BATCH):
            j=min(i+BATCH,100_000)
            with torch.amp.autocast('cuda'):
                preds2.append(model(tp[i:j],se[i:j],tq[i:j]).argmax(dim=-1))
    shuffled=(torch.cat(preds2)==new_labels).float().mean().item()*100
    
    print(f"  正常:      {normal:.2f}%")
    print(f"  打乱中间边: {shuffled:.2f}% (用新label)")
    if abs(normal - shuffled) < 5:
        print(f"  ✅ 打乱后准确率相似 → 模型学会了mod运算的组合")
    else:
        print(f"  ⚠️ 差距{abs(normal-shuffled):.1f}% → 模型可能依赖位置信息")

else:
    print(f"\n  训练未达85%，跳过OOD测试")
    print(f"  最佳: {best:.1%}")
    print(f"  这个任务可能需要更多message passing轮数")
    print(f"  尝试: 增加MSG_PASSES到15或20")

print(f"\n{'='*60}")
print(f"  结论: 5跳训练最佳 {best:.1%}")
print(f"{'='*60}")
