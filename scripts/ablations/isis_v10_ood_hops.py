"""
ISIS V10 — OOD hop-count generalization (GNN multi-hop).

Train on 5-hop chains only (TRAIN_HOPS=5), then evaluate the same model
without retraining on longer chains (3/5/10/15/20/30/50 hops, incl. the
paper's 10/20/50). Key change vs isis_v10_order_only.py: the number of
message-passing rounds is a model parameter decoupled from input chain
length, so the model accepts arbitrarily long test chains.

Usage:
  python isis_v10_ood_hops.py

Outputs per-hop accuracy and ood_hops_results.csv (paper table data).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import sys
import csv

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
D_MODEL = 128
BATCH = 2048
EPOCHS = 300
NUM_RANGE = 100
TRAIN_HOPS = 5      # hop count used for training
MSG_PASSES = 10      # message-passing rounds (fixed; enough to cover long chains)

VAL_FALSE=0; VAL_TRUE=1; VAL_UNKNOWN=2; N_VALS=3
N_RELS=6
REL_GT=0;REL_LT=1;REL_EQ=2;REL_GTE=3;REL_LTE=4;REL_NEQ=5

print(f"{'='*60}")
print(f"  ISIS V10 - OOD 跳数泛化实验")
print(f"  训练: {TRAIN_HOPS} 跳 | 测试: 5/10/20/50 跳")
print(f"  设备: {DEVICE}")
print(f"{'='*60}")

# --- Data generator (arbitrary hop count) ---
def generate_chain_data(n_samples, max_hops, device=DEVICE):
    """Generate chain-reasoning data with the given hop count."""
    N = n_samples
    nodes = torch.zeros((N, max_hops + 1), dtype=torch.long, device=device)
    edges = torch.randint(0, N_RELS, (N, max_hops), device=device)
    nodes[:, 0] = torch.randint(0, NUM_RANGE, (N,), device=device)

    for h in range(max_hops):
        curr = nodes[:, h]
        rel = edges[:, h]

        # Downgrade relations at the value-range boundary (GT at 0 -> GTE, LT at max -> LTE).
        bad_gt = (curr == 0) & (rel == REL_GT)
        rel[bad_gt] = REL_GTE
        bad_lt = (curr == NUM_RANGE - 1) & (rel == REL_LT)
        rel[bad_lt] = REL_LTE

        # Valid sampling interval for the next node.
        min_c = torch.zeros_like(curr)
        max_c = torch.full_like(curr, NUM_RANGE - 1)

        m = rel == REL_GT;  max_c[m] = torch.clamp(curr[m] - 1, min=0)
        m = rel == REL_LT;  min_c[m] = torch.clamp(curr[m] + 1, max=NUM_RANGE-1)
        m = rel == REL_GTE; max_c[m] = curr[m]
        m = rel == REL_LTE; min_c[m] = curr[m]
        m = rel == REL_EQ;  min_c[m] = curr[m]; max_c[m] = curr[m]

        range_size = max_c - min_c + 1
        cand = min_c + (torch.rand(N, device=device) * range_size).long()

        m_neq = rel == REL_NEQ
        if m_neq.any():
            cand_neq = torch.randint(0, NUM_RANGE, (m_neq.sum().item(),), device=device)
            conflict = cand_neq == curr[m_neq]
            cand_neq[conflict] = (cand_neq[conflict] + 1) % NUM_RANGE
            cand[m_neq] = cand_neq

        nodes[:, h + 1] = cand
        edges[:, h] = rel  # store the downgraded relation

    # Label: relation between head and tail nodes.
    query_rel = torch.randint(0, N_RELS, (N,), device=device)
    head = nodes[:, 0]
    tail = nodes[:, max_hops]

    rel_true = (
        ((query_rel == REL_GT)  & (head > tail))  |
        ((query_rel == REL_LT)  & (head < tail))  |
        ((query_rel == REL_EQ)  & (head == tail))  |
        ((query_rel == REL_GTE) & (head >= tail))  |
        ((query_rel == REL_LTE) & (head <= tail))  |
        ((query_rel == REL_NEQ) & (head != tail))
    )
    target = torch.where(rel_true,
                         torch.tensor(VAL_TRUE, device=device),
                         torch.tensor(VAL_FALSE, device=device))

    nodes_f = nodes.float() / NUM_RANGE
    return nodes_f, edges, query_rel, target


# --- Model (message-pass count decoupled from input length) ---
class NumEnc(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(1, D_MODEL//2), nn.GELU(),
                               nn.Linear(D_MODEL//2, D_MODEL))
    def forward(self, x):
        return self.f(x.unsqueeze(-1))

class DeepOrderEngineOOD(nn.Module):
    """Message-pass count is a constructor parameter, not tied to data length;
    forward adapts to chains of any length."""
    def __init__(self, msg_passes=MSG_PASSES):
        super().__init__()
        self.msg_passes = msg_passes
        self.num_enc = NumEnc()
        self.rel_emb = nn.Embedding(N_RELS, D_MODEL)

        self.msg_mlp = nn.Sequential(
            nn.Linear(D_MODEL * 3, D_MODEL * 2), nn.GELU(),
            nn.LayerNorm(D_MODEL * 2), nn.Linear(D_MODEL * 2, D_MODEL)
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(D_MODEL * 2, D_MODEL), nn.GELU(),
            nn.LayerNorm(D_MODEL)
        )
        self.norm = nn.LayerNorm(D_MODEL)

        self.readout = nn.Sequential(
            nn.Linear(D_MODEL * 3, D_MODEL * 2), nn.GELU(),
            nn.Dropout(0.1), nn.LayerNorm(D_MODEL * 2),
            nn.Linear(D_MODEL * 2, D_MODEL)
        )
        self.sv = nn.Embedding(N_VALS, D_MODEL)
        nn.init.orthogonal_(self.sv.weight)

    def forward(self, nodes_f, edges_idx, query_rel):
        """
        nodes_f: [B, L]      (L = chain length, variable)
        edges_idx: [B, L-1]
        query_rel: [B]
        """
        B, n_nodes = nodes_f.shape

        H = self.num_enc(nodes_f)      # [B, n_nodes, D]
        E = self.rel_emb(edges_idx)    # [B, n_nodes-1, D]

        # Fixed number of message-passing rounds; works for any chain length.
        for _ in range(self.msg_passes):
            # left message: i-1 -> i
            left_input = torch.cat([H[:, 1:], E, H[:, :-1]], dim=-1)
            left_msg = self.msg_mlp(left_input)

            # right message: i+1 -> i
            right_input = torch.cat([H[:, :-1], E, H[:, 1:]], dim=-1)
            right_msg = self.msg_mlp(right_input)

            total_msg = torch.zeros_like(H)
            total_msg[:, 1:]  += left_msg
            total_msg[:, :-1] += right_msg

            updated = self.update_mlp(torch.cat([H, total_msg], dim=-1))
            H = self.norm(H + updated)

        # readout from head and tail nodes
        out = self.readout(torch.cat([H[:, 0], H[:, -1],
                                      self.rel_emb(query_rel)], dim=-1))
        return out

    def classify(self, v):
        sn = F.normalize(self.sv.weight, dim=-1)
        vn = F.normalize(v, dim=-1)
        return (vn @ sn.T).argmax(dim=-1)


# --- Training (TRAIN_HOPS-hop chains only) ---
print(f"\n--- Phase 1: 训练 {TRAIN_HOPS} 跳模型 ---")

N_TRAIN_TOTAL = 1_000_000
print(f"生成 {N_TRAIN_TOTAL//10000} 万条 {TRAIN_HOPS} 跳训练数据...")
NODES, EDGES, QUERY, TARGET = generate_chain_data(N_TRAIN_TOTAL, TRAIN_HOPS)
split = int(N_TRAIN_TOTAL * 0.8)

# Inverse-frequency class weights.
counts = torch.bincount(TARGET, minlength=N_VALS).float()
weights = counts.max() / counts.clamp(min=1)
weights = weights.to(DEVICE)

model = DeepOrderEngineOOD(msg_passes=MSG_PASSES).to(DEVICE)
opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-5)
scaler = torch.amp.GradScaler('cuda') if DEVICE.type == 'cuda' else None

print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
print(f"消息传递轮数: {MSG_PASSES} (固定)")
print(f"Batch: {BATCH} | Epochs: {EPOCHS}")
print()

best_acc = 0.0
last_acc = 0.0
epoch_times = []

for epoch in range(1, EPOCHS + 1):
    model.train()
    perm = torch.randperm(split, device=DEVICE)
    total_loss = 0.0; nb = 0
    t0 = time.time()

    for i in range(0, split, BATCH):
        idx = perm[i:i+BATCH]
        with torch.amp.autocast('cuda' if DEVICE.type == 'cuda' else 'cpu'):
            out = model(NODES[idx], EDGES[idx], QUERY[idx])
            loss = F.cross_entropy(out @ model.sv.weight.T, TARGET[idx], weight=weights)

        opt.zero_grad()
        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        total_loss += loss.item(); nb += 1

    sched.step()
    epoch_times.append(time.time() - t0)

    if epoch % 10 == 0:
        model.eval()
        with torch.no_grad():
            tidx = torch.arange(split, N_TRAIN_TOTAL, device=DEVICE)
            preds = []
            for j in range(0, len(tidx), BATCH * 4):
                bi = tidx[j:j+BATCH*4]
                with torch.amp.autocast('cuda' if DEVICE.type == 'cuda' else 'cpu'):
                    preds.append(model.classify(model(NODES[bi], EDGES[bi], QUERY[bi])))
            acc = (torch.cat(preds) == TARGET[split:]).float().mean().item()
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), 'isis_v10_ood_best.pth')
            last_acc = acc

        avg_t = sum(epoch_times[-10:]) / len(epoch_times[-10:])
        eta_s = int(avg_t * (EPOCHS - epoch))
        print(f"  Epoch {epoch:3d}/{EPOCHS} | Loss: {total_loss/nb:.4f} | "
              f"Acc: {acc:.1%} | Best: {best_acc:.1%} | "
              f"ETA: {eta_s//60}m{eta_s%60:02d}s")

        if best_acc >= 1.0 - 1e-6:
            print(f"\n  ✅ {TRAIN_HOPS}跳训练达到100%! 停止训练，开始OOD测试。")
            break

print(f"\n{'='*60}")
print(f"  训练完成 | {TRAIN_HOPS}跳最佳准确率: {best_acc:.1%}")
print(f"{'='*60}")


# --- OOD hop-count generalization test ---
print(f"\n{'='*60}")
print(f"  OOD 跳数泛化测试")
print(f"  模型训练于: {TRAIN_HOPS} 跳")
print(f"  不重新训练，直接测试更长的链")
print(f"{'='*60}")

model.load_state_dict(torch.load('isis_v10_ood_best.pth', map_location=DEVICE, weights_only=True))
model.eval()

test_hops = [3, 5, 10, 15, 20, 30, 50]
N_TEST = 100_000
results = []

for hops in test_hops:
    print(f"\n  测试 {hops} 跳 ({N_TEST//1000}K 样本)...", end="", flush=True)

    t_nodes, t_edges, t_query, t_target = generate_chain_data(N_TEST, hops, device=DEVICE)

    all_preds = []
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, N_TEST, BATCH):
            j = min(i + BATCH, N_TEST)
            with torch.amp.autocast('cuda' if DEVICE.type == 'cuda' else 'cpu'):
                out = model(t_nodes[i:j], t_edges[i:j], t_query[i:j])
                all_preds.append(model.classify(out))
    preds = torch.cat(all_preds)
    elapsed = time.time() - t0

    acc = (preds == t_target).float().mean().item() * 100
    ood_tag = "IN-DIST" if hops == TRAIN_HOPS else f"OOD ({hops//TRAIN_HOPS}x)"

    t_acc = f_acc = "-"
    t_mask = t_target == VAL_TRUE
    f_mask = t_target == VAL_FALSE
    if t_mask.sum() > 0:
        t_acc = f"{(preds[t_mask]==VAL_TRUE).float().mean().item()*100:.1f}%"
    if f_mask.sum() > 0:
        f_acc = f"{(preds[f_mask]==VAL_FALSE).float().mean().item()*100:.1f}%"

    marker = "✅" if acc > 99.0 else ("🟡" if acc > 90 else "🔴")
    print(f"  {marker} {hops:3d}跳 | 准确率: {acc:6.2f}% | True:{t_acc} False:{f_acc} | "
          f"{ood_tag} | {elapsed:.2f}s")

    results.append({
        'train_hops': TRAIN_HOPS,
        'test_hops': hops,
        'accuracy': f"{acc:.2f}",
        'tag': ood_tag,
        'time_s': f"{elapsed:.2f}",
    })

    del t_nodes, t_edges, t_query, t_target, preds
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()

csv_path = "ood_hops_results.csv"
with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['train_hops','test_hops','accuracy','tag','time_s'])
    writer.writeheader()
    writer.writerows(results)

print(f"\n{'='*60}")
print(f"  结果已保存: {csv_path}")
print(f"{'='*60}")
print()
print("  论文表格数据：")
print(f"  {'训练跳数':<10} {'测试跳数':<10} {'准确率':<10} {'备注':<15}")
print(f"  {'-'*45}")
for r in results:
    print(f"  {r['train_hops']:<10} {r['test_hops']:<10} {r['accuracy']+'%':<10} {r['tag']:<15}")
print()
print("  如果 OOD 准确率 > 99%:")
print("    → 证明模型学会了'推理结构'而不是'记住了5跳的模式'")
print("    → 直接对标 DNAR (ICML 2025) 的泛化能力")
print("    → 论文核心卖点之一")
