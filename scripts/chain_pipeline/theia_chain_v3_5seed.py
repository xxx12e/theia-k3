"""
THEIA chain pipeline, 5-seed validation (paper Sec. 4.4, mod-3 sequential composition).

Runs the full three-phase pipeline once per seed and reports mean±std
accuracy at 5/10/50/100/500 steps.

Usage: python theia_chain_v3_5seed.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
import time, numpy as np
from tqdm import tqdm

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DIM = 128
NUM_RANGE = 20
LOGIC_OPS = 5
SEEDS = [42, 123, 256, 777, 999]

# --- model (same as v3) ---

class NumEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, DIM), nn.GELU(), nn.Linear(DIM, DIM))
        self.unk = nn.Parameter(torch.randn(DIM) * 0.02)
    def forward(self, x, is_unk):
        out = self.net(x.unsqueeze(-1))
        mask = is_unk.unsqueeze(-1).float()
        return out * (1 - mask) + self.unk * mask

class ArithEngine(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc_a = NumEncoder()
        self.enc_b = NumEncoder()
        self.op_emb = nn.Embedding(4, DIM)
        self.fuse = nn.Sequential(nn.Linear(DIM*3, DIM*2), nn.GELU(), nn.LayerNorm(DIM*2),
                                  nn.Linear(DIM*2, DIM), nn.GELU(), nn.LayerNorm(DIM))
    def forward(self, a, b, op, au, bu):
        return self.fuse(torch.cat([self.enc_a(a, au), self.enc_b(b, bu), self.op_emb(op)], -1))

class SubspaceEngine(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.g = nn.Sequential(nn.Linear(in_dim, DIM), nn.GELU(), nn.LayerNorm(DIM))
        self.l = nn.Sequential(nn.Linear(in_dim, DIM), nn.GELU(), nn.LayerNorm(DIM))
        self.e = nn.Sequential(nn.Linear(in_dim, DIM), nn.GELU(), nn.LayerNorm(DIM))
        self.fg = nn.Sequential(nn.Linear(DIM*2, DIM), nn.GELU())
        self.fl = nn.Sequential(nn.Linear(DIM*2, DIM), nn.GELU())
        self.fe = nn.Sequential(nn.Linear(DIM*3, DIM), nn.GELU())
        self.out = nn.Sequential(nn.Linear(DIM*4, DIM*2), nn.GELU(), nn.LayerNorm(DIM*2),
                                 nn.Linear(DIM*2, DIM), nn.GELU(), nn.LayerNorm(DIM))
    def forward(self, x, ctx):
        inp = torch.cat([x, ctx], -1)
        g, l, e = self.g(inp), self.l(inp), self.e(inp)
        gp = self.fg(torch.cat([g, e], -1))
        lp = self.fl(torch.cat([l, e], -1))
        ep = self.fe(torch.cat([e, gp, lp], -1))
        return self.out(torch.cat([gp, lp, ep, x], -1))

class OrderEngine(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_enc = NumEncoder()
        self.rel_emb = nn.Embedding(6, DIM)
        self.core = SubspaceEngine(DIM * 3)
    def forward(self, c, d, rel, du):
        return self.core(c, torch.cat([self.d_enc(d, du), self.rel_emb(rel)], -1))

class SetEngine(nn.Module):
    def __init__(self):
        super().__init__()
        self.set_enc = nn.Sequential(nn.Linear(21, DIM), nn.GELU(), nn.Linear(DIM, DIM))
        self.unk = nn.Parameter(torch.randn(DIM) * 0.02)
        self.core = SubspaceEngine(DIM * 2)
    def forward(self, c, s, su):
        sv = self.set_enc(s)
        mask = su.unsqueeze(-1).float()
        return self.core(c, sv * (1 - mask) + self.unk * mask)

class Bridge(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(DIM, DIM), nn.GELU(), nn.LayerNorm(DIM))
    def forward(self, x): return self.net(x) + x

class OutHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(DIM, DIM), nn.GELU(), nn.Linear(DIM, DIM))
        proto = torch.zeros(3, DIM)
        proto[0, :DIM//3] = 1.0; proto[1, DIM//3:2*DIM//3] = 1.0; proto[2, 2*DIM//3:] = 1.0
        self.proto = nn.Parameter(proto)
    def forward(self, x):
        x = F.normalize(self.proj(x), dim=-1)
        return torch.matmul(x, F.normalize(self.proto, dim=-1).T) * 10.0

class THEIAStep(nn.Module):
    def __init__(self):
        super().__init__()
        self.arith = ArithEngine()
        self.bridge_ao = Bridge()
        self.bridge_as = Bridge()
        self.order = OrderEngine()
        self.set_eng = SetEngine()
        self.logic = SubspaceEngine(DIM * 3)
        self.logic_op_emb = nn.Embedding(LOGIC_OPS, DIM)
        self.out = OutHead()
    def forward(self, a, b, op, d, rel, s, logic_op, au, bu, du, su):
        c = self.arith(a, b, op, au, bu)
        v_ord = self.order(self.bridge_ao(c), d, rel, du)
        v_set = self.set_eng(self.bridge_as(c), s, su)
        lo = self.logic(v_ord, torch.cat([v_set, self.logic_op_emb(logic_op)], -1))
        return self.out(lo)
    def forward_flat(self, x):
        return self(x[:,0], x[:,1], x[:,2].long(), x[:,3], x[:,4].long(), x[:,5:26],
                    x[:,26].long(), x[:,27].bool(), x[:,28].bool(), x[:,29].bool(), x[:,30].bool())

class TransitionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(6, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 3))
    def forward(self, prev_oh, local_oh):
        return self.net(torch.cat([prev_oh, local_oh], -1))

class THEIAChainV3(nn.Module):
    def __init__(self):
        super().__init__()
        self.step = THEIAStep()
        self.transition = TransitionNet()
    def gumbel_st(self, logits, tau=0.5):
        soft = F.gumbel_softmax(logits, tau=tau, hard=False)
        hard = torch.zeros_like(soft).scatter_(-1, soft.argmax(-1, keepdim=True), 1.0)
        return hard - soft.detach() + soft

# --- data generation ---

def gen_single_step_data(N, seed):
    rng = np.random.RandomState(seed)
    FEAT = 31
    inputs = np.zeros((N, FEAT), dtype=np.float32)
    labels = np.zeros(N, dtype=np.int64)
    a = rng.randint(0, NUM_RANGE, N); b = rng.randint(0, NUM_RANGE, N)
    op = rng.randint(0, 4, N); d = rng.randint(0, NUM_RANGE+1, N)
    rel = rng.randint(0, 6, N); s = rng.randint(0, 2, (N, 21)).astype(np.float32)
    logic_op = rng.randint(0, LOGIC_OPS, N)
    au = (rng.random(N) < 0.15).astype(np.float32); bu = (rng.random(N) < 0.15).astype(np.float32)
    du = (rng.random(N) < 0.15).astype(np.float32); su = (rng.random(N) < 0.15).astype(np.float32)
    inputs[:,0]=a; inputs[:,1]=b; inputs[:,2]=op; inputs[:,3]=d; inputs[:,4]=rel
    inputs[:,5:26]=s; inputs[:,26]=logic_op; inputs[:,27]=au; inputs[:,28]=bu; inputs[:,29]=du; inputs[:,30]=su

    any_unk = (au > 0) | (bu > 0)
    arith_val = np.where(op==0,a+b,np.where(op==1,a-b,np.where(op==2,a*b,a//np.where(b==0,1,b))))
    order_tv = np.ones(N, dtype=np.int64); ok = ~any_unk & ~(du > 0)
    ob = np.zeros(N, dtype=bool)
    for r, fn in enumerate([np.greater,np.less,np.greater_equal,np.less_equal,np.equal,np.not_equal]):
        ob = np.where(rel==r, fn(arith_val,d), ob)
    order_tv = np.where(ok & ob, 2, np.where(ok & ~ob, 0, order_tv))
    set_tv = np.ones(N, dtype=np.int64); sk = ~any_unk & ~(su > 0)
    s_idx = np.clip(arith_val % 21, 0, 20)
    in_set = s[np.arange(N), s_idx].astype(bool)
    set_tv = np.where(sk & in_set, 2, np.where(sk & ~in_set, 0, set_tv))
    for op_i in range(LOGIC_OPS):
        m = logic_op == op_i
        if not m.any(): continue
        atv, btv = order_tv[m], set_tv[m]; r = np.ones_like(atv)
        if op_i==0: r=np.where((atv==0)|(btv==0),0,r); r=np.where((atv==2)&(btv==2),2,r)
        elif op_i==1: r=np.where((atv==2)|(btv==2),2,r); r=np.where((atv==0)&(btv==0),0,r)
        elif op_i==2: r=np.where(atv==0,2,r); r=np.where((atv==2)&(btv==0),0,r); r=np.where((atv==2)&(btv==2),2,r)
        elif op_i==3: r=np.where((atv!=1)&(btv!=1)&(atv==btv),2,r); r=np.where((atv!=1)&(btv!=1)&(atv!=btv),0,r)
        elif op_i==4: r=np.where((atv!=1)&(btv!=1)&(atv==btv),0,r); r=np.where((atv!=1)&(btv!=1)&(atv!=btv),2,r)
        labels[m] = r
    return torch.from_numpy(inputs), torch.from_numpy(labels)

def gen_chain_data(N, num_steps, seed):
    rng = np.random.RandomState(seed)
    all_inputs = np.zeros((N, num_steps, 31), dtype=np.float32)
    labels = np.zeros((N, num_steps), dtype=np.int64)
    local_v = np.zeros((N, num_steps), dtype=np.int64)
    for t in range(num_steps):
        a=rng.randint(0,NUM_RANGE,N); b=rng.randint(0,NUM_RANGE,N); op=rng.randint(0,4,N)
        d=rng.randint(0,NUM_RANGE+1,N); rel=rng.randint(0,6,N)
        s=rng.randint(0,2,(N,21)).astype(np.float32); logic_op=rng.randint(0,LOGIC_OPS,N)
        au=(rng.random(N)<0.15).astype(np.float32); bu=(rng.random(N)<0.15).astype(np.float32)
        du=(rng.random(N)<0.15).astype(np.float32); su=(rng.random(N)<0.15).astype(np.float32)
        all_inputs[:,t,0]=a; all_inputs[:,t,1]=b; all_inputs[:,t,2]=op; all_inputs[:,t,3]=d
        all_inputs[:,t,4]=rel; all_inputs[:,t,5:26]=s; all_inputs[:,t,26]=logic_op
        all_inputs[:,t,27]=au; all_inputs[:,t,28]=bu; all_inputs[:,t,29]=du; all_inputs[:,t,30]=su
        any_unk=(au>0)|(bu>0)
        arith_val=np.where(op==0,a+b,np.where(op==1,a-b,np.where(op==2,a*b,a//np.where(b==0,1,b))))
        order_tv=np.ones(N,dtype=np.int64); ok=~any_unk&~(du>0); ob=np.zeros(N,dtype=bool)
        for r,fn in enumerate([np.greater,np.less,np.greater_equal,np.less_equal,np.equal,np.not_equal]):
            ob=np.where(rel==r,fn(arith_val,d),ob)
        order_tv=np.where(ok&ob,2,np.where(ok&~ob,0,order_tv))
        set_tv=np.ones(N,dtype=np.int64); sk=~any_unk&~(su>0)
        s_idx=np.clip(arith_val%21,0,20); in_set=s[np.arange(N),s_idx].astype(bool)
        set_tv=np.where(sk&in_set,2,np.where(sk&~in_set,0,set_tv))
        lt=np.ones(N,dtype=np.int64)
        for op_i in range(LOGIC_OPS):
            m=logic_op==op_i
            if not m.any(): continue
            atv,btv=order_tv[m],set_tv[m]; r=np.ones_like(atv)
            if op_i==0: r=np.where((atv==0)|(btv==0),0,r); r=np.where((atv==2)&(btv==2),2,r)
            elif op_i==1: r=np.where((atv==2)|(btv==2),2,r); r=np.where((atv==0)&(btv==0),0,r)
            elif op_i==2: r=np.where(atv==0,2,r); r=np.where((atv==2)&(btv==0),0,r); r=np.where((atv==2)&(btv==2),2,r)
            elif op_i==3: r=np.where((atv!=1)&(btv!=1)&(atv==btv),2,r); r=np.where((atv!=1)&(btv!=1)&(atv!=btv),0,r)
            elif op_i==4: r=np.where((atv!=1)&(btv!=1)&(atv==btv),0,r); r=np.where((atv!=1)&(btv!=1)&(atv!=btv),2,r)
            lt[m]=r
        local_v[:,t]=lt
        labels[:,t] = lt if t==0 else (labels[:,t-1]+lt)%3
    return torch.from_numpy(all_inputs), torch.from_numpy(labels), torch.from_numpy(local_v)

# --- training phases ---

def phase1(step_model, train_seed, bs=4096):
    """Phase 1 with plateau detection: if stuck below 90% after 40 epochs, restart."""
    N = 2_000_000
    inputs, labels = gen_single_step_data(N, seed=train_seed)
    inp_g, lbl_g = inputs.to(DEVICE), labels.to(DEVICE)
    # Class weights: w_U = 2.0 (chain encoding F=0, U=1, T=2)
    weights = torch.tensor([1.0, 2.0, 1.0], device=DEVICE)

    MAX_RESTARTS = 3
    EPOCHS_PER_TRY = 150
    best_overall = 0

    for restart in range(MAX_RESTARTS + 1):
        if restart > 0:
            print(f"    ⟳ Restart {restart}/{MAX_RESTARTS}")
            if best_overall < 0.85:
                # Full reinit: Linear, Embedding, and learned Parameters
                for name, m in step_model.named_modules():
                    if isinstance(m, (nn.Linear, nn.Embedding)):
                        m.reset_parameters()
                for name, p in step_model.named_parameters():
                    if 'unk' in name or 'proto' in name:
                        nn.init.normal_(p.data, 0, 0.02)
                print(f"    ⟳ Full model reinit (Linear + Embedding + unk/proto)")

        opt = torch.optim.AdamW(step_model.parameters(), lr=1e-3, weight_decay=0.01)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS_PER_TRY)
        scaler = GradScaler("cuda")
        best_this_try = 0
        last_improve_ep = 0

        for ep in range(1, EPOCHS_PER_TRY + 1):
            step_model.train()
            perm = torch.randperm(N, device=DEVICE)
            correct = 0

            for i in range(0, N, bs):
                idx = perm[i:i+bs]
                with autocast('cuda'):
                    logits = step_model.forward_flat(inp_g[idx])
                    loss = F.cross_entropy(logits, lbl_g[idx], weight=weights)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(step_model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                correct += (logits.detach().argmax(-1) == lbl_g[idx]).sum().item()

            sched.step()
            acc = correct / N

            if acc > best_this_try + 0.001:
                best_this_try = acc
                last_improve_ep = ep
            best_overall = max(best_overall, acc)

            if best_overall >= 0.999:
                print(f"    P1 ✅ converged ep {ep} (restart {restart}): {best_overall:.2%}")
                return best_overall

            if ep % 20 == 0:
                print(f"    P1 ep {ep}: {acc:.1%} (best={best_overall:.1%}, r={restart})")

            # Coarse plateau: after 40 epochs still below 90%
            if ep >= 40 and best_this_try < 0.90:
                print(f"    P1 plateau at ep {ep}: {best_this_try:.1%} < 90%, restarting...")
                break

            # Fine plateau: no improvement for 30 epochs after reaching 90%
            if ep - last_improve_ep >= 30 and best_this_try >= 0.90 and best_this_try < 0.999:
                print(f"    P1 fine plateau: no improvement for 30ep, best={best_this_try:.1%}")
                break

    print(f"    P1 finished after {MAX_RESTARTS+1} tries: {best_overall:.2%}")
    return best_overall

def phase2(model, train_seed, epochs=50, bs=4096):
    for p in model.step.parameters(): p.requires_grad = False
    N, STEPS = 500_000, 5
    _, chain_lbl, local_v = gen_chain_data(N, STEPS, seed=train_seed+1000)
    lbl_g, local_g = chain_lbl.to(DEVICE), local_v.to(DEVICE)
    opt = torch.optim.AdamW(model.transition.parameters(), lr=1e-3, weight_decay=0.01)
    best = 0
    for ep in range(1, epochs+1):
        model.transition.train(); perm = torch.randperm(N, device=DEVICE)
        correct = 0
        for i in range(0, N, bs):
            idx = perm[i:i+bs]; bl, bv = lbl_g[idx], local_g[idx]
            loss = 0
            for t in range(1, STEPS):
                prev_oh = F.one_hot(bl[:, t-1], 3).float()
                local_oh = F.one_hot(bv[:, t], 3).float()
                loss = loss + F.cross_entropy(model.transition(prev_oh, local_oh), bl[:, t])
            loss = loss / (STEPS - 1)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.transition.parameters(), 1.0); opt.step()
            with torch.no_grad():
                prev_oh = F.one_hot(bl[:,-2], 3).float()
                local_oh = F.one_hot(bv[:,-1], 3).float()
                correct += (model.transition(prev_oh, local_oh).argmax(-1) == bl[:,-1]).sum().item()
        acc = correct / N; best = max(best, acc)
        if best >= 0.999:
            print(f"    P2 converged ep {ep}: {best:.2%}")
            break
    for p in model.step.parameters(): p.requires_grad = True
    return best

def phase3(model, train_seed, epochs=50, bs=4096):
    N, STEPS = 500_000, 5
    inputs, chain_lbl, _ = gen_chain_data(N, STEPS, seed=train_seed+2000)
    inp_g, lbl_g = inputs.to(DEVICE), chain_lbl.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    scaler = GradScaler("cuda")
    best = 0
    for ep in range(1, epochs+1):
        model.train(); perm = torch.randperm(N, device=DEVICE)
        # Gumbel-ST temperature schedule: tau = max(0.1, 0.5 - 0.01*ep)
        correct = 0; tau = max(0.1, 0.5 - ep * 0.01)
        for i in range(0, N, bs):
            idx = perm[i:i+bs]; bi, bl = inp_g[idx], lbl_g[idx]
            with autocast('cuda'):
                all_lg = []; prev_oh = None
                for t in range(STEPS):
                    local_lg = model.step.forward_flat(bi[:, t])
                    if t == 0:
                        lg = local_lg; prev_oh = model.gumbel_st(local_lg, tau)
                    else:
                        local_oh = model.gumbel_st(local_lg, tau)
                        lg = model.transition(prev_oh, local_oh)
                        prev_oh = model.gumbel_st(lg, tau)
                    all_lg.append(lg)
                loss = sum((1+t*0.5)*F.cross_entropy(all_lg[t], bl[:,t]) for t in range(STEPS)) / STEPS
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            correct += (all_lg[-1].detach().argmax(-1) == bl[:,-1]).sum().item()
        sched.step(); acc = correct / N; best = max(best, acc)
        if ep % 10 == 0:
            print(f"    P3 ep {ep}: {acc:.1%} (τ={tau:.2f})")
    return best

# --- evaluation ---

@torch.no_grad()
def evaluate(model, num_steps, seed, N=10000, bs=4096):
    model.eval()
    inputs, labels, _ = gen_chain_data(N, num_steps, seed=seed)
    step_correct = torch.zeros(num_steps)
    cls_correct = torch.zeros(3); cls_total = torch.zeros(3)

    for i in tqdm(range(0, N, bs), desc=f"  {num_steps}-step", leave=False, ncols=80):
        bi = inputs[i:i+bs].to(DEVICE); bl = labels[i:i+bs]
        prev_oh = None; preds = []
        for t in range(num_steps):
            local_lg = model.step.forward_flat(bi[:, t])
            if t == 0:
                pred = local_lg.argmax(-1); prev_oh = F.one_hot(pred, 3).float()
            else:
                local_oh = F.one_hot(local_lg.argmax(-1), 3).float()
                chain_lg = model.transition(prev_oh, local_oh)
                pred = chain_lg.argmax(-1); prev_oh = F.one_hot(pred, 3).float()
            preds.append(pred.cpu())

        for t in range(num_steps):
            step_correct[t] += (preds[t] == bl[:, t]).sum()
        fp, fl = preds[-1], bl[:, -1]
        for c in range(3):
            m = fl == c; cls_total[c] += m.sum(); cls_correct[c] += (fp[m] == c).sum()

    final_acc = (step_correct[-1] / N).item()
    step1_acc = (step_correct[0] / N).item()
    per_cls = {c: (cls_correct[c]/cls_total[c]).item() if cls_total[c]>0 else 0 for c in range(3)}
    return final_acc, step1_acc, per_cls

# --- main ---

if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True

    print(f"\n{'='*60}")
    print(f"  THEIA Chain v3 — 5-Seed Validation")
    print(f"  Seeds: {SEEDS}")
    print(f"  Device: {DEVICE}")
    print(f"{'='*60}")

    TEST_STEPS = [5, 10, 50, 100, 500]
    results = {s: [] for s in TEST_STEPS}  # {steps: [acc_per_seed]}

    for si, seed in enumerate(SEEDS):
        print(f"\n{'='*60}")
        print(f"  Seed {si+1}/5: {seed}")
        print(f"{'='*60}")

        torch.manual_seed(seed)
        np.random.seed(seed)

        model = THEIAChainV3().to(DEVICE)

        t0 = time.time()
        p1 = phase1(model.step, train_seed=seed)
        p2 = phase2(model, train_seed=seed)
        p3 = phase3(model, train_seed=seed)
        elapsed = time.time() - t0
        print(f"  Training: {elapsed:.0f}s | P1={p1:.1%} P2={p2:.1%} P3={p3:.1%}")

        for steps in TEST_STEPS:
            acc, s1, cls = evaluate(model, steps, seed=seed+5000, N=10000)
            results[steps].append(acc)
            tag = "✅" if acc > 0.99 else "🟡" if acc > 0.95 else "❌"
            print(f"  {steps:>4d}-step: {acc:.2%} (step1={s1:.2%}) {tag}")

        del model
        torch.cuda.empty_cache()

    # --- summary ---
    print(f"\n{'='*60}")
    print(f"  5-SEED SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Steps':>6s}  {'Mean':>8s}  {'Std':>8s}  {'Min':>8s}  {'Max':>8s}")
    print(f"  {'-'*42}")
    for steps in TEST_STEPS:
        vals = results[steps]
        mean = np.mean(vals)
        std = np.std(vals)
        mn, mx = np.min(vals), np.max(vals)
        tag = "✅" if mean > 0.99 else "🟡" if mean > 0.95 else "❌"
        print(f"  {steps:>6d}  {mean:>7.2%}  {std:>7.2%}  {mn:>7.2%}  {mx:>7.2%}  {tag}")

    print(f"\n  For paper: N=5, train 5-step → test 500-step")
    vals500 = results[500]
    print(f"  500-step accuracy: {np.mean(vals500):.2%} ± {np.std(vals500):.2%}")
    print(f"\n{'='*60}")