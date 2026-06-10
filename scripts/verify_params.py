#!/usr/bin/env python
"""
CPU-only parameter-count verification for the THEIA release.

Verifies the parameter counts asserted in the paper (Appendix B):

  4-domain THEIA (IsisV9)         total == 2,751,232
    ArithEngine                   ==   404,736
    OrderEngine                   == 1,179,648
    SetEngine                     ==   208,128
    LogicEngine                   ==   908,032
    Bridges (bridge_ao+bridge_as) ==    33,536
    OutHead                       ==    16,768
    Prototypes (sv)               ==       384
  Chain THEIAStep                 == 1,508,096
  Chain TransitionNet (MLP)       ==     4,803
  BigTransformer (matched TF)     ==  3,640,000-ish (paper: "3.64M"; exact count printed)

Runs entirely on CPU: CUDA_VISIBLE_DEVICES is cleared BEFORE torch is imported,
so this script never touches the GPU and is safe to run next to live GPU jobs.

Usage (from repo root):
    python scripts/verify_params.py
"""
import os
import sys

# Force CPU before torch import — do not touch the GPU.
# '-1' (no valid device) rather than '' because some PyTorch builds ignore the
# empty string; belt-and-braces assert below.
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, 'theia'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'scripts', 'chain_pipeline'))

import torch
import torch.nn as nn

assert not torch.cuda.is_available(), "CUDA visible despite CUDA_VISIBLE_DEVICES='' — aborting"

from _theia_model_def import IsisV9                       # 4-domain model (eval-only def)
from theia_chain_v3_5seed import THEIAStep, TransitionNet  # chain pipeline (import-safe: __main__ guard)


def n_params(m):
    return sum(p.numel() for p in m.parameters())


# --- BigTransformer (matched-protocol baseline) ---
# Verbatim copy of the model definition from
# scripts/train_4domain/tf8l_5seed_earlystop_v2.py (that script trains at import
# time, so the class is replicated here for CPU-only instantiation).
D_MODEL = 192
SET_DIM = 21
N_VALS = 3
N_RELS = 6
N_ARITH = 4
N_OPS = 5


class BigTransformer(nn.Module):
    def __init__(self, d=D_MODEL, nhead=8, nlayers=8):
        super().__init__()
        self.num_enc = nn.Sequential(nn.Linear(1, d//2), nn.GELU(), nn.Linear(d//2, d))
        self.set_enc = nn.Sequential(nn.Linear(SET_DIM, d//2), nn.GELU(), nn.Linear(d//2, d))
        self.arith_emb = nn.Embedding(N_ARITH, d)
        self.rel_emb = nn.Embedding(N_RELS, d)
        self.op_emb = nn.Embedding(N_OPS, d)
        self.unk_emb = nn.Embedding(2, d)
        self.type_emb = nn.Embedding(11, d)
        enc = nn.TransformerEncoderLayer(d_model=d, nhead=nhead, dim_feedforward=d*4,
                                         dropout=0.1, activation='gelu', batch_first=True)
        self.transformer = nn.TransformerEncoder(enc, num_layers=nlayers)
        self.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.LayerNorm(d), nn.Linear(d, N_VALS))


EXPECTED = {
    'IsisV9.total':      2_751_232,
    'IsisV9.arith_eng':    404_736,
    'IsisV9.order_eng':  1_179_648,
    'IsisV9.set_eng':      208_128,
    'IsisV9.logic_eng':    908_032,
    'IsisV9.bridges':       33_536,
    'IsisV9.out_head':      16_768,
    'IsisV9.sv':               384,
    'THEIAStep.total':   1_508_096,
    'TransitionNet.total':   4_803,
}


def main():
    torch.manual_seed(0)
    model = IsisV9()
    step = THEIAStep()
    trans = TransitionNet()
    big_tf = BigTransformer()

    actual = {
        'IsisV9.total':       n_params(model),
        'IsisV9.arith_eng':   n_params(model.arith_eng),
        'IsisV9.order_eng':   n_params(model.order_eng),
        'IsisV9.set_eng':     n_params(model.set_eng),
        'IsisV9.logic_eng':   n_params(model.logic_eng),
        'IsisV9.bridges':     n_params(model.bridge_ao) + n_params(model.bridge_as),
        'IsisV9.out_head':    n_params(model.out_head),
        'IsisV9.sv':          n_params(model.sv),
        'THEIAStep.total':    n_params(step),
        'TransitionNet.total': n_params(trans),
    }

    print(f"{'module':<24} {'expected':>12} {'actual':>12}  verdict")
    print('-' * 62)
    failures = 0
    for k, exp in EXPECTED.items():
        act = actual[k]
        ok = act == exp
        failures += (not ok)
        verdict = 'OK' if ok else '*** MISMATCH ***'
        print(f"{k:<24} {exp:>12,} {act:>12,}  {verdict}")

    # THEIAStep per-module breakdown (informative)
    print()
    print('THEIAStep breakdown (informative):')
    for name, sub in step.named_children():
        print(f"  step.{name:<14} {n_params(sub):>12,}")

    # Sanity: IsisV9 module sum == total
    parts = sum(actual[k] for k in actual if k.startswith('IsisV9.') and k != 'IsisV9.total')
    print()
    print(f"IsisV9 module sum:        {parts:>12,}  ({'OK' if parts == actual['IsisV9.total'] else '*** does not equal total ***'})")

    # BigTransformer (paper: 3.64M, matched-protocol baseline)
    btf = n_params(big_tf)
    print(f"BigTransformer total:     {btf:>12,}  (paper states 3.64M; informative)")

    print()
    if failures:
        print(f"RESULT: {failures} MISMATCH(ES) — see rows marked above")
        sys.exit(1)
    print('RESULT: all asserted parameter counts match.')


if __name__ == '__main__':
    main()
