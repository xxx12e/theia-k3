"""
Decisive-edge depth distribution: in a transitive chain, at which step does
the first non-'=' edge appear? If ~95% of cases are decided within the first
5 steps, "50-hop generalization" is not a true 10x extrapolation. Pure
statistics, no model involved.
"""
import torch

N=100_000; DEVICE='cpu'  # pure statistics; no GPU needed
N_RELS=3; REL_GT=0; REL_LT=1; REL_EQ=2

def gen_chain(n, hops, seed=42):
    torch.manual_seed(seed)
    edges = torch.zeros(n, hops, dtype=torch.long)
    acc = torch.randint(0, N_RELS, (n,))  # first edge
    edges[:, 0] = acc
    for h in range(1, hops):
        new_rel = torch.zeros(n, dtype=torch.long)
        m_gt = acc == REL_GT
        if m_gt.any():
            choices = torch.randint(0, 2, (m_gt.sum().item(),))
            new_rel[m_gt] = torch.where(choices == 0, torch.tensor(REL_GT), torch.tensor(REL_EQ))
        m_lt = acc == REL_LT
        if m_lt.any():
            choices = torch.randint(0, 2, (m_lt.sum().item(),))
            new_rel[m_lt] = torch.where(choices == 0, torch.tensor(REL_LT), torch.tensor(REL_EQ))
        m_eq = acc == REL_EQ
        if m_eq.any():
            new_rel[m_eq] = torch.randint(0, N_RELS, (m_eq.sum().item(),))
        edges[:, h] = new_rel
        # update acc
        TRANS = torch.tensor([[REL_GT,-1,REL_GT],[-1,REL_LT,REL_LT],[REL_GT,REL_LT,REL_EQ]])
        acc = TRANS[acc, new_rel]
    return edges

print(f"{'='*50}")
print(f"  决定性Edge深度分布分析")
print(f"{'='*50}")

for hops in [5, 10, 20, 50]:
    edges = gen_chain(N, hops, seed=123)
    
    # Position of the first non-'=' edge per sample.
    is_non_eq = (edges != REL_EQ)  # [N, hops]
    first_decisive = torch.full((N,), hops, dtype=torch.long)  # default=hops if all =
    for h in range(hops):
        still_undecided = first_decisive == hops
        newly_decided = still_undecided & is_non_eq[:, h]
        first_decisive[newly_decided] = h
    
    # After that step the accumulated relation is no longer EQ and later edges
    # cannot change its direction, so first_decisive is the effective reasoning depth.
    mean_depth = first_decisive.float().mean().item()
    median_depth = first_decisive.float().median().item()
    within_5 = (first_decisive <= 4).float().mean().item() * 100
    within_10 = (first_decisive <= 9).float().mean().item() * 100
    all_eq = (first_decisive == hops).float().mean().item() * 100
    
    print(f"\n  {hops}跳链:")
    print(f"    第一个决定性edge平均位置: {mean_depth:.1f} (中位数: {median_depth:.0f})")
    print(f"    在前5步内确定: {within_5:.1f}%")
    print(f"    在前10步内确定: {within_10:.1f}%")
    print(f"    全部是=（未确定）: {all_eq:.1f}%")

print(f"\n  论文结论:")
print(f"  如果50跳链95%+在前5步确定 → '50-hop generalization'只是'5-hop propagation'")
print(f"  如果分布更均匀 → generalization claim有一定支撑")
