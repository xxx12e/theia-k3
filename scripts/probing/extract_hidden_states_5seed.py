"""
Extract hidden states at the four boundary outputs (arith/order/set/logic)
from the 5 canonical THEIA checkpoints (seed 42 in the legacy
multi_seed_results/theia/ path, seeds 123/256/777/999 in
multi_seed_results/theia_v2/). One balanced 5000-sample dataset (data_seed=42)
is shared so all checkpoints see identical inputs. Supports the probing
analyses in §5.3.

Output:
  hidden_states_5seed.pt   dict[seed] -> dict{arith,order,set,logic} tensors (5000, 128)
  hidden_labels_5seed.pt   tensor (5000,)  — same for all 5 seeds (shared inputs)
  hidden_meta_5seed.pt     dict with per-input unknown flags + paths used
"""
import sys
import os
import time

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import torch
import torch.nn as nn
import torch.nn.functional as F

from _theia_model_def import (
    IsisV9, DEVICE,
    NUM_RANGE, SET_DIM, P_UNKNOWN,
    VAL_FALSE, VAL_TRUE, VAL_UNKNOWN,
    N_RELS, N_ARITH, N_OPS,
    REL_GT, REL_LT, REL_EQ, REL_GTE, REL_LTE, REL_NEQ,
    ARITH_ADD, ARITH_SUB, ARITH_MUL, ARITH_MOD,
    OP_AND, OP_OR, OP_NOT, OP_IMPLIES, OP_IFF,
    apply_logic,
)

CHECKPOINTS = [
    (42,  'multi_seed_results/theia/seed_42/checkpoint.pth'),
    (123, 'multi_seed_results/theia_v2/seed_123/checkpoint.pth'),
    (256, 'multi_seed_results/theia_v2/seed_256/checkpoint.pth'),
    (777, 'multi_seed_results/theia_v2/seed_777/checkpoint.pth'),
    (999, 'multi_seed_results/theia_v2/seed_999/checkpoint.pth'),
]
N_EXTRACT = 5000     # ~1667 per class (balanced)
DATA_SEED = 42       # input shared across all 5 checkpoints
BATCH = 1024
OUT_DIR = '.'


def generate_balanced(n_each, seed=DATA_SEED):
    """Generate balanced 3-class data (~n_each per class) with an explicit
    data_seed for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print(f'Generating balanced data ({n_each} per class, data_seed={seed}) ...')
    t0 = time.time()
    buckets = {0: [], 1: [], 2: []}
    chunk = 500_000

    while min(sum(len(x['tgt']) for x in v) for v in buckets.values()) < n_each:
        a = torch.randint(1, NUM_RANGE+1, (chunk,), device=DEVICE)
        b = torch.randint(1, NUM_RANGE+1, (chunk,), device=DEVICE)
        d = torch.randint(0, NUM_RANGE+1, (chunk,), device=DEVICE)
        ar = torch.randint(0, N_ARITH, (chunk,), device=DEVICE)
        rl = torch.randint(0, N_RELS,  (chunk,), device=DEVICE)
        op = torch.randint(0, N_OPS,   (chunk,), device=DEVICE)
        sb = torch.randint(0, 2, (chunk, SET_DIM), dtype=torch.float32, device=DEVICE)
        au = torch.rand(chunk, device=DEVICE) < P_UNKNOWN
        bu = torch.rand(chunk, device=DEVICE) < P_UNKNOWN
        du = torch.rand(chunk, device=DEVICE) < P_UNKNOWN
        su = torch.rand(chunk, device=DEVICE) < P_UNKNOWN

        c = torch.zeros(chunk, dtype=torch.long, device=DEVICE)
        c[ar==ARITH_ADD] = torch.clamp(a+b, 0, NUM_RANGE)[ar==ARITH_ADD]
        c[ar==ARITH_SUB] = torch.abs(a-b)[ar==ARITH_SUB]   # v1 symmetric (matches theia_5seed_v2.py)
        c[ar==ARITH_MUL] = torch.clamp(a*b, 0, NUM_RANGE)[ar==ARITH_MUL]
        c[ar==ARITH_MOD] = (a % torch.clamp(b, 1, NUM_RANGE))[ar==ARITH_MOD]
        c = torch.clamp(c, 0, NUM_RANGE)

        c_unk = au | bu
        ord_unk = c_unk | du
        rel_true = (((rl==REL_GT) & (c >  d)) | ((rl==REL_LT) & (c <  d)) |
                    ((rl==REL_EQ) & (c == d)) | ((rl==REL_GTE) & (c >= d)) |
                    ((rl==REL_LTE) & (c <= d)) | ((rl==REL_NEQ) & (c != d)))
        ord_v = torch.where(rel_true, torch.tensor(VAL_TRUE, device=DEVICE),
                                       torch.tensor(VAL_FALSE, device=DEVICE))
        val_o = torch.where(ord_unk, torch.tensor(VAL_UNKNOWN, device=DEVICE), ord_v)

        sou = su | c_unk
        ci = c.clamp(0, SET_DIM-1)
        ins = sb[torch.arange(chunk, device=DEVICE), ci].bool()
        sv = torch.where(ins, torch.tensor(VAL_TRUE, device=DEVICE),
                              torch.tensor(VAL_FALSE, device=DEVICE))
        val_s = torch.where(sou, torch.tensor(VAL_UNKNOWN, device=DEVICE), sv)

        target = apply_logic(op, val_o, val_s)

        for sid in range(3):
            current = sum(len(x['tgt']) for x in buckets[sid])
            need = n_each - current
            if need <= 0: continue
            idx = (target == sid).nonzero(as_tuple=True)[0][:need]
            if len(idx) == 0: continue
            buckets[sid].append({
                'a':  a[idx].float() / NUM_RANGE,
                'b':  b[idx].float() / NUM_RANGE,
                'd':  d[idx].float() / NUM_RANGE,
                'sb': sb[idx],
                's_unk': su[idx], 'a_unk': au[idx],
                'b_unk': bu[idx], 'd_unk': du[idx],
                'ar': ar[idx], 'rl': rl[idx], 'op': op[idx],
                'tgt': target[idx],
            })
        sizes = {k: sum(len(x['tgt']) for x in v) for k, v in buckets.items()}
        print(f'  F:{sizes[0]} T:{sizes[1]} U:{sizes[2]} / {n_each}')

    result = {}
    for k in buckets[0][0].keys():
        parts = [x[k] for sid in range(3) for x in buckets[sid]]
        result[k] = torch.cat(parts)[:n_each * 3]

    perm = torch.randperm(len(result['tgt']), device=DEVICE)
    for k in result:
        result[k] = result[k][perm]

    print(f'Done {time.time()-t0:.1f}s. N = {len(result["tgt"])} '
          f'(F:{(result["tgt"]==0).sum().item()} '
          f'T:{(result["tgt"]==1).sum().item()} '
          f'U:{(result["tgt"]==2).sum().item()})')
    return result


def forward_with_hidden(model, a, b, d, set_bits, s_unk, a_unk, b_unk, d_unk,
                        arith, rel, op):
    """Replicate IsisV9.forward but return all 4 boundary hidden vectors."""
    c_vec = model.arith_eng(a, b, a_unk, b_unk, arith)
    c_for_ord = model.bridge_ao(c_vec) + c_vec
    c_for_set = model.bridge_as(c_vec) + c_vec
    ord_vec = model.order_eng(c_for_ord, d, d_unk, rel)
    set_vec = model.set_eng(c_for_set, set_bits, s_unk)
    logic_vec = model.logic_eng(ord_vec, set_vec, op)
    return {'arith': c_vec, 'order': ord_vec, 'set': set_vec, 'logic': logic_vec}


def main():
    # Generate shared input data
    data = generate_balanced(N_EXTRACT // 3)
    n_total = len(data['tgt'])

    # Per-seed hidden extraction
    per_seed_hidden = {}    # seed -> {arith, order, set, logic}
    per_seed_pred_acc = {}
    missing_paths = []

    for seed, ckpt_path in CHECKPOINTS:
        print(f'\nseed {seed}: loading {ckpt_path} ...')
        if not os.path.exists(ckpt_path):
            print(f'  MISSING')
            missing_paths.append((seed, ckpt_path))
            continue

        model = IsisV9().to(DEVICE)
        state = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state)
        model.eval()
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params == 2_751_232, f'param count mismatch: {n_params}'

        all_h = {'arith': [], 'order': [], 'set': [], 'logic': []}
        all_preds = []
        with torch.no_grad():
            for i in range(0, n_total, BATCH):
                sl = slice(i, min(i + BATCH, n_total))
                h = forward_with_hidden(
                    model,
                    data['a'][sl], data['b'][sl], data['d'][sl],
                    data['sb'][sl], data['s_unk'][sl],
                    data['a_unk'][sl], data['b_unk'][sl], data['d_unk'][sl],
                    data['ar'][sl], data['rl'][sl], data['op'][sl],
                )
                out = model.out_head(h['logic'])
                pred = model.classify(out)
                for k in all_h:
                    all_h[k].append(h[k].cpu())
                all_preds.append(pred.cpu())

        per_seed_hidden[seed] = {k: torch.cat(v) for k, v in all_h.items()}
        preds = torch.cat(all_preds)
        acc = (preds == data['tgt'].cpu()).float().mean().item()
        per_seed_pred_acc[seed] = acc
        print(f'  acc on extract set: {acc:.4f}')
        del model
        torch.cuda.empty_cache()

    # Save
    labels = data['tgt'].cpu()
    meta = {
        'a_unk': data['a_unk'].cpu(),
        'b_unk': data['b_unk'].cpu(),
        'd_unk': data['d_unk'].cpu(),
        's_unk': data['s_unk'].cpu(),
        'data_seed': DATA_SEED,
        'n_extract': N_EXTRACT,
        'checkpoint_paths': dict(CHECKPOINTS),
        'per_seed_pred_acc': per_seed_pred_acc,
        'missing_paths': missing_paths,
    }

    torch.save(per_seed_hidden, os.path.join(OUT_DIR, 'hidden_states_5seed.pt'))
    torch.save(labels,          os.path.join(OUT_DIR, 'hidden_labels_5seed.pt'))
    torch.save(meta,            os.path.join(OUT_DIR, 'hidden_meta_5seed.pt'))

    print(f'\nSaved to {OUT_DIR}/')
    print(f'  hidden_states_5seed.pt  (5 seeds × 4 boundaries × {n_total} × 128)')
    print(f'  hidden_labels_5seed.pt  ({n_total},)')
    print(f'  hidden_meta_5seed.pt    (incl. per-seed acc + provenance)')
    print(f'\nPer-seed acc summary:')
    for s, a in per_seed_pred_acc.items():
        print(f'  seed {s}: {a*100:.2f}%')


if __name__ == '__main__':
    main()
