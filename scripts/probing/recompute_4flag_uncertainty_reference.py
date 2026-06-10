"""
Recompute the 4-flag uncertainty-only reference points from the actual joint
distribution (§5.3). Generates data_seed=999, N=50000 data via
_theia_model_def's gen_data formula (byte-identical reproduction; no model
loaded, pure data generation).

Two reference points:
  (a) Has-Unknown oracle: classifier knows HU ∈ {0, 1} only
      = P(HU=1)·max_v P(VU=v|HU=1) + P(HU=0)·max_v P(VU=v|HU=0)
  (b) VU-U oracle: classifier knows whether VU=U or not
      = P(VU=U)·1.0 + P(VU≠U)·max_v∈{T,F} P(VU=v|VU≠U)

Plus per-boundary margins to the canonical MLP d=6 numbers (w=256, random
60/20/20, val-best-epoch). Output: 4flag_uncertainty_reference.json + console.
"""
import os, sys, json
sys.path.insert(0, '.')
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import torch
import numpy as np
torch.cuda.is_available = lambda: False

from _theia_model_def import (
    P_UNKNOWN, NUM_RANGE, SET_DIM,
    VAL_FALSE, VAL_TRUE, VAL_UNKNOWN,
    N_RELS, N_ARITH, N_OPS,
    REL_GT, REL_LT, REL_EQ, REL_GTE, REL_LTE, REL_NEQ,
    ARITH_ADD, ARITH_SUB, ARITH_MUL, ARITH_MOD,
    apply_logic,
)

DATA_SEED = 999
N = 50_000


def gen_data(N, seed):
    """Byte-identical to _theia_model_def's gen_data, returns dict."""
    torch.manual_seed(seed)
    a = torch.randint(1, NUM_RANGE+1, (N,))
    b = torch.randint(1, NUM_RANGE+1, (N,))
    d = torch.randint(0, NUM_RANGE+1, (N,))
    arith = torch.randint(0, N_ARITH, (N,))
    rel   = torch.randint(0, N_RELS,  (N,))
    op    = torch.randint(0, N_OPS,   (N,))
    a_unk = torch.rand(N) < P_UNKNOWN
    b_unk = torch.rand(N) < P_UNKNOWN
    d_unk = torch.rand(N) < P_UNKNOWN
    s_unk = torch.rand(N) < P_UNKNOWN

    c = torch.zeros(N, dtype=torch.long)
    c[arith==ARITH_ADD] = torch.clamp(a+b, 0, NUM_RANGE)[arith==ARITH_ADD]
    c[arith==ARITH_SUB] = torch.abs(a-b)[arith==ARITH_SUB]
    c[arith==ARITH_MUL] = torch.clamp(a*b, 0, NUM_RANGE)[arith==ARITH_MUL]
    c[arith==ARITH_MOD] = (a % torch.clamp(b, 1, NUM_RANGE))[arith==ARITH_MOD]

    c_unk = a_unk | b_unk
    ord_unk = c_unk | d_unk
    rel_true = (((rel==REL_GT) & (c >  d)) | ((rel==REL_LT) & (c <  d)) |
                ((rel==REL_EQ) & (c == d)) | ((rel==REL_GTE) & (c >= d)) |
                ((rel==REL_LTE) & (c <= d)) | ((rel==REL_NEQ) & (c != d)))
    ord_v = torch.where(rel_true, torch.tensor(VAL_TRUE), torch.tensor(VAL_FALSE))
    val_o = torch.where(ord_unk, torch.tensor(VAL_UNKNOWN), ord_v)

    sb = torch.randint(0, 2, (N, SET_DIM), dtype=torch.float32)
    sou = s_unk | c_unk
    ci = c.clamp(0, SET_DIM-1)
    ins = sb[torch.arange(N), ci].bool()
    sv = torch.where(ins, torch.tensor(VAL_TRUE), torch.tensor(VAL_FALSE))
    val_s = torch.where(sou, torch.tensor(VAL_UNKNOWN), sv)

    target = apply_logic(op, val_o, val_s)
    return {
        'a_unk': a_unk, 'b_unk': b_unk, 'd_unk': d_unk, 's_unk': s_unk,
        'verdict': target,
    }


def main():
    print('=' * 78)
    print(f'  4-flag Uncertainty-Only Reference Recomputation')
    print(f'  data_seed={DATA_SEED}, N={N}, P_UNKNOWN={P_UNKNOWN}')
    print('=' * 78)

    d = gen_data(N, DATA_SEED)
    a_unk = d['a_unk'].numpy().astype(int)
    b_unk = d['b_unk'].numpy().astype(int)
    d_unk = d['d_unk'].numpy().astype(int)
    s_unk = d['s_unk'].numpy().astype(int)
    verdict = d['verdict'].numpy()

    # Has-Unknown = OR of all 4 atomic flags
    HU = (a_unk | b_unk | d_unk | s_unk).astype(int)

    n = len(verdict)
    print(f'\nMarginal distributions:')
    print(f'  P(HU=0) = {(HU==0).mean():.4f}    P(HU=1) = {(HU==1).mean():.4f}')
    print(f'  P(VU=F) = {(verdict==VAL_FALSE).mean():.4f}')
    print(f'  P(VU=T) = {(verdict==VAL_TRUE).mean():.4f}')
    print(f'  P(VU=U) = {(verdict==VAL_UNKNOWN).mean():.4f}')

    print(f'\nJoint distribution P(VU, HU):')
    print(f'  {"":>10} {"HU=0":>10} {"HU=1":>10} {"row total":>12}')
    joint = {}
    for vu_name, vu_val in [('VU=F', VAL_FALSE), ('VU=T', VAL_TRUE), ('VU=U', VAL_UNKNOWN)]:
        cell = {}
        for hu_val in [0, 1]:
            mask = (verdict == vu_val) & (HU == hu_val)
            p = mask.mean()
            cell[f'HU={hu_val}'] = float(p)
        row_tot = (verdict == vu_val).mean()
        joint[vu_name] = cell
        joint[vu_name]['row_total'] = float(row_tot)
        print(f'  {vu_name:>10} {cell["HU=0"]:>10.4f} {cell["HU=1"]:>10.4f} {row_tot:>12.4f}')
    col0_tot = (HU == 0).mean()
    col1_tot = (HU == 1).mean()
    print(f'  {"col total":>10} {col0_tot:>10.4f} {col1_tot:>10.4f} {1.0:>12.4f}')

    # --- (a) Has-Unknown oracle ---
    print(f'\n' + '=' * 78)
    print(f'  (a) HAS-UNKNOWN ORACLE')
    print(f'  = P(HU=1) * max_v P(VU=v|HU=1) + P(HU=0) * max_v P(VU=v|HU=0)')
    print('=' * 78)

    # P(VU=v | HU=h) = P(VU=v, HU=h) / P(HU=h)
    p_HU0 = (HU == 0).mean()
    p_HU1 = (HU == 1).mean()

    cond_VU_given_HU0 = {}
    cond_VU_given_HU1 = {}
    for vu_name, vu_val in [('F', VAL_FALSE), ('T', VAL_TRUE), ('U', VAL_UNKNOWN)]:
        cond_VU_given_HU0[vu_name] = float(((verdict == vu_val) & (HU == 0)).mean() / p_HU0)
        cond_VU_given_HU1[vu_name] = float(((verdict == vu_val) & (HU == 1)).mean() / p_HU1)

    print(f'  P(VU=v | HU=0):  F={cond_VU_given_HU0["F"]:.4f}  T={cond_VU_given_HU0["T"]:.4f}  U={cond_VU_given_HU0["U"]:.4f}')
    print(f'  P(VU=v | HU=1):  F={cond_VU_given_HU1["F"]:.4f}  T={cond_VU_given_HU1["T"]:.4f}  U={cond_VU_given_HU1["U"]:.4f}')

    max_HU0 = max(cond_VU_given_HU0.values())
    max_HU1 = max(cond_VU_given_HU1.values())
    argmax_HU0 = max(cond_VU_given_HU0, key=cond_VU_given_HU0.get)
    argmax_HU1 = max(cond_VU_given_HU1, key=cond_VU_given_HU1.get)

    has_unk_oracle = p_HU0 * max_HU0 + p_HU1 * max_HU1
    print(f'  argmax under HU=0: VU={argmax_HU0} ({max_HU0*100:.4f}%)')
    print(f'  argmax under HU=1: VU={argmax_HU1} ({max_HU1*100:.4f}%)')
    print(f'  HAS-UNKNOWN ORACLE = {p_HU0:.4f} * {max_HU0:.4f} + {p_HU1:.4f} * {max_HU1:.4f}')
    print(f'                     = {p_HU0 * max_HU0:.6f} + {p_HU1 * max_HU1:.6f}')
    print(f'                     = {has_unk_oracle*100:.4f}%')

    # --- (b) VU-U oracle ---
    print(f'\n' + '=' * 78)
    print(f'  (b) VU-U ORACLE (paper "uncertainty-only reference" equivalent)')
    print(f'  = P(VU=U) * 1.0 + P(VU\u2260U) * max_v\u2208{{F,T}} P(VU=v|VU\u2260U)')
    print('=' * 78)

    p_U   = (verdict == VAL_UNKNOWN).mean()
    p_neU = 1 - p_U
    p_F_given_neU = float(((verdict == VAL_FALSE) & (verdict != VAL_UNKNOWN)).mean() / p_neU)
    p_T_given_neU = float(((verdict == VAL_TRUE)  & (verdict != VAL_UNKNOWN)).mean() / p_neU)

    print(f'  P(VU=U)    = {p_U:.4f}')
    print(f'  P(VU\u2260U)   = {p_neU:.4f}')
    print(f'  P(VU=F | VU\u2260U) = {p_F_given_neU:.4f}')
    print(f'  P(VU=T | VU\u2260U) = {p_T_given_neU:.4f}')
    max_neU = max(p_F_given_neU, p_T_given_neU)
    argmax_neU = 'F' if p_F_given_neU >= p_T_given_neU else 'T'

    vu_u_oracle = p_U * 1.0 + p_neU * max_neU
    print(f'  argmax under VU\u2260U: VU={argmax_neU} ({max_neU*100:.4f}%)')
    print(f'  VU-U ORACLE = {p_U:.4f} * 1.0 + {p_neU:.4f} * {max_neU:.4f}')
    print(f'              = {p_U:.6f} + {p_neU * max_neU:.6f}')
    print(f'              = {vu_u_oracle*100:.4f}%')

    # --- Per-boundary margin analysis ---
    print(f'\n' + '=' * 78)
    print(f'  PER-BOUNDARY MARGIN ANALYSIS')
    print(f'  (canonical w=256 random-split val-best-epoch MLP d=6 numbers)')
    print('=' * 78)

    # Canonical MLP d=6 numbers from probe_cleansplit_audit.md §5.2 (5-seed mean ± std, %)
    canonical_mlp_d6 = {
        'arith': (61.272, 0.045),
        'order': (67.166, 0.060),
        'set':   (69.916, 0.118),
        'logic': (99.916, 0.040),
    }

    print(f'  {"boundary":>10} | {"MLP d=6 5-seed mean":>22} | {"vs HU oracle":>18} | {"vs VU-U oracle":>20}')
    print('  ' + '-' * 90)
    for b, (m, s) in canonical_mlp_d6.items():
        marg_hu = m - has_unk_oracle * 100
        marg_vu = m - vu_u_oracle * 100
        print(f'  {b:>10} | {m:6.4f} \u00b1 {s:5.4f}      | {marg_hu:+7.4f}pp        | {marg_vu:+7.4f}pp')

    print(f'\n  HAS-UNKNOWN oracle = {has_unk_oracle*100:.4f}%')
    print(f'  VU-U oracle        = {vu_u_oracle*100:.4f}%')

    out = {
        'data_seed': DATA_SEED,
        'n': N,
        'P_UNKNOWN': P_UNKNOWN,
        'n_atomic_flags': 4,
        'flag_names': ['a_unknown', 'b_unknown', 'd_unknown', 's_unknown'],
        'marginal': {
            'P(HU=0)': float(p_HU0),
            'P(HU=1)': float(p_HU1),
            'P(VU=F)': float((verdict == VAL_FALSE).mean()),
            'P(VU=T)': float((verdict == VAL_TRUE).mean()),
            'P(VU=U)': float((verdict == VAL_UNKNOWN).mean()),
        },
        'joint_VU_HU': joint,
        'conditionals': {
            'P(VU=v|HU=0)': cond_VU_given_HU0,
            'P(VU=v|HU=1)': cond_VU_given_HU1,
            'P(VU=F|VU!=U)': p_F_given_neU,
            'P(VU=T|VU!=U)': p_T_given_neU,
        },
        'reference_points': {
            'has_unknown_oracle': {
                'value': float(has_unk_oracle),
                'value_pct': float(has_unk_oracle * 100),
                'argmax_HU0': argmax_HU0,
                'argmax_HU1': argmax_HU1,
                'formula': 'P(HU=0)*max_v P(VU=v|HU=0) + P(HU=1)*max_v P(VU=v|HU=1)',
            },
            'vu_u_oracle': {
                'value': float(vu_u_oracle),
                'value_pct': float(vu_u_oracle * 100),
                'argmax_neU': argmax_neU,
                'formula': 'P(VU=U)*1.0 + P(VU!=U)*max_v∈{F,T} P(VU=v|VU!=U)',
                'note': 'paper "74% uncertainty-only reference" equivalent under 4-flag generator',
            },
        },
        'per_boundary_margin': {
            b: {
                'mlp_d6_5seed_mean_pct': m,
                'mlp_d6_5seed_std_pct':  s,
                'margin_to_has_unknown_oracle_pp': float(m - has_unk_oracle * 100),
                'margin_to_vu_u_oracle_pp':        float(m - vu_u_oracle * 100),
            }
            for b, (m, s) in canonical_mlp_d6.items()
        },
        'source': 'probe_cleansplit w=256 random-split val-best-epoch audit',
        'protocol': 'canonical: w=256, random 60/20/20, val-best-epoch, MLP d=6',
    }
    json.dump(out, open('4flag_uncertainty_reference.json', 'w'), indent=2)
    print(f'\nSaved: 4flag_uncertainty_reference.json')


if __name__ == '__main__':
    main()
