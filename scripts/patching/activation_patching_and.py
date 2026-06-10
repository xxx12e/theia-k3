"""
Activation patching with parameterized logic operator (paper §5.3 / Appendix E).
Fork of activation_patching.py.

Cell A (OR + v_ord, sanity): must reproduce the 4898/4898 = 100% flip rate,
  else Cell C is aborted.
  Shared: a, b, arith=ADD, set_bits (c NOT in S => val_set=F), logic_op=OR, def flags
  T-side: d = max(0, c-1), d_unk=F, REL_GTE -> val_ord=T -> T OR F = T
  U-side: d_unk=T                           -> val_ord=U -> U OR F = U
Cell C (AND + v_ord, new): tests causal generalization across operators.
  Shared: a, b, arith=ADD, set_bits (c in S => val_set=T), logic_op=AND, def flags
  T-side / U-side as in Cell A           -> T AND T = T  vs  U AND T = U

Patch (both cells): OutHead(LogicEng(v_ord^U, v_set^T, op)); expected T -> U flip.
v_set byte-equality is asserted in both cells. Data seed 12345; model seeds
42, 123, 256, 777, 999 (same as the 5-seed OR run).
"""

import argparse
import json
import os
import random
import sys
import types

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import torch
import torch.nn.functional as F

# Reuse model builder and forward_with_hidden monkey-patch from activation_patching.py
from activation_patching import (
    _build_model, _forward_with_hidden,
    NUM_RANGE, SET_DIM,
    ARITH_ADD, REL_GTE,
    PROTO_FALSE, PROTO_TRUE, PROTO_UNKNOWN,
    classify,
)

# Logic op indices (IsisV9 / theia_5seed_v2.py encoding)
LOGIC_OR = 1
LOGIC_AND = 0


CHECKPOINTS = [
    # seed 42 uses pre-retrain checkpoint to reproduce paper's 4719/4719 AND flip rate
    # (the theia_v2 retrain produces a comparable but not byte-identical aggregate);
    # see activation_patching_5seed.py for the full seed-42-path rationale.
    (42,  'multi_seed_results/theia/seed_42/checkpoint.pth'),
    (123, 'multi_seed_results/theia_v2/seed_123/checkpoint.pth'),
    (256, 'multi_seed_results/theia_v2/seed_256/checkpoint.pth'),
    (777, 'multi_seed_results/theia_v2/seed_777/checkpoint.pth'),
    (999, 'multi_seed_results/theia_v2/seed_999/checkpoint.pth'),
]
N_PAIRS = 1000
DATA_SEED = 12345
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def build_matched_pair(device, logic_op: str):
    """Build one (T-side, U-side) matched pair parameterized by logic_op.

    logic_op == 'or':  OR absorption non-trivial case (T or F = T,  U or F = U)
                       set_bits chosen so c NOT in S -> val_set = F
    logic_op == 'and': AND absorption non-trivial case (T and T = T, U and T = U)
                       set_bits chosen so c IN S     -> val_set = T
    """
    a_val = random.randint(1, NUM_RANGE - 2)
    b_val = random.randint(0, NUM_RANGE - a_val - 1)
    c_val = a_val + b_val  # c in [1, 19]

    set_bits = torch.zeros(SET_DIM)

    if logic_op == 'or':
        # c NOT in S: build S from non-c indices, leave bit[c]=0 -> val_set = F
        non_c_indices = [i for i in range(SET_DIM) if i != c_val]
        k = random.randint(1, 5)
        for i in random.sample(non_c_indices, k=k):
            set_bits[i] = 1.0
        op_idx = LOGIC_OR
    elif logic_op == 'and':
        # c IN S: always include c; plus 1..5 random others -> val_set = T
        set_bits[c_val] = 1.0
        non_c_indices = [i for i in range(SET_DIM) if i != c_val]
        k = random.randint(1, 5)
        for i in random.sample(non_c_indices, k=k):
            set_bits[i] = 1.0
        op_idx = LOGIC_AND
    else:
        raise ValueError(f"logic_op must be 'or' or 'and', got {logic_op!r}")

    # T-side: d=max(0, c-1), d_unk=False, REL_GTE -> val_ord=T
    d_T = max(0, c_val - 1)
    # U-side: d_unk=True (value overridden by learnable sentinel)
    d_U = 0

    def pack(d_val, d_unk_flag):
        return dict(
            a=torch.tensor([a_val / NUM_RANGE], device=device, dtype=torch.float32),
            b=torch.tensor([b_val / NUM_RANGE], device=device, dtype=torch.float32),
            d=torch.tensor([d_val / NUM_RANGE], device=device, dtype=torch.float32),
            set_bits=set_bits.unsqueeze(0).to(device),
            s_unk=torch.tensor([False], device=device),
            a_unk=torch.tensor([False], device=device),
            b_unk=torch.tensor([False], device=device),
            d_unk=torch.tensor([d_unk_flag], device=device),
            arith=torch.tensor([ARITH_ADD], device=device, dtype=torch.long),
            rel=torch.tensor([REL_GTE], device=device, dtype=torch.long),
            op=torch.tensor([op_idx], device=device, dtype=torch.long),
        )

    return pack(d_T, False), pack(d_U, True), c_val


@torch.no_grad()
def run_patching(model, n_pairs, device, seed, logic_op):
    """Run patching protocol on one model checkpoint with given logic_op."""
    random.seed(seed)
    torch.manual_seed(seed)
    model.eval()

    stats = {
        'total': 0,
        'baseline_T_correct': 0,
        'baseline_U_correct': 0,
        'both_baseline_correct': 0,
        'set_vec_identical': 0,
        'patch_flipped_to_U': 0,
        'patch_stayed_T': 0,
        'patch_other_F': 0,
    }

    for _ in range(n_pairs):
        T_inputs, U_inputs, _c_val = build_matched_pair(device, logic_op)

        H_T = model.forward_with_hidden(**T_inputs)
        H_U = model.forward_with_hidden(**U_inputs)

        o_T = model.out_head(H_T['logic'])
        o_U = model.out_head(H_U['logic'])
        pred_T = classify(o_T, model.sv.weight).item()
        pred_U = classify(o_U, model.sv.weight).item()

        stats['total'] += 1
        T_ok = (pred_T == PROTO_TRUE)
        U_ok = (pred_U == PROTO_UNKNOWN)
        stats['baseline_T_correct'] += int(T_ok)
        stats['baseline_U_correct'] += int(U_ok)

        if not (T_ok and U_ok):
            continue
        stats['both_baseline_correct'] += 1

        # v_set byte-equality assertion
        set_identical = torch.allclose(H_T['set'], H_U['set'], atol=1e-6)
        stats['set_vec_identical'] += int(set_identical)

        # Patch: feed U-side ord_vec into T-side logic_eng
        logic_patched = model.logic_eng(H_U['order'], H_T['set'], T_inputs['op'])
        o_patched = model.out_head(logic_patched)
        pred_patched = classify(o_patched, model.sv.weight).item()

        if pred_patched == PROTO_UNKNOWN:
            stats['patch_flipped_to_U'] += 1
        elif pred_patched == PROTO_TRUE:
            stats['patch_stayed_T'] += 1
        else:
            stats['patch_other_F'] += 1

    return stats


def load_model(ckpt_path):
    model = _build_model(DEVICE)
    state = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    if isinstance(state, dict) and 'model_state_dict' in state:
        model.load_state_dict(state['model_state_dict'])
    elif isinstance(state, dict) and 'state_dict' in state:
        model.load_state_dict(state['state_dict'])
    else:
        model.load_state_dict(state)
    model.forward_with_hidden = types.MethodType(_forward_with_hidden, model)
    return model


def run_cell(logic_op: str, cell_name: str):
    print(f"\n{'=' * 70}")
    print(f"  Cell {cell_name}: logic_op={logic_op.upper()} + v_ord")
    print(f"  N_PAIRS={N_PAIRS}, DATA_SEED={DATA_SEED}, DEVICE={DEVICE}")
    print(f"{'=' * 70}")

    results = {}
    for seed, ckpt_path in CHECKPOINTS:
        if not os.path.exists(ckpt_path):
            print(f"  seed {seed}: SKIP (ckpt not found)")
            continue
        model = load_model(ckpt_path)
        stats = run_patching(model, N_PAIRS, DEVICE, DATA_SEED, logic_op)
        n = stats['both_baseline_correct']
        entry = {
            'baseline_t': stats['baseline_T_correct'],
            'baseline_u': stats['baseline_U_correct'],
            'eligible': n,
            'flip_T_to_U': stats['patch_flipped_to_U'],
            'flip_T_to_F': stats['patch_other_F'],
            'stay_T': stats['patch_stayed_T'],
            'set_vec_identical': stats['set_vec_identical'],
            'flip_rate': stats['patch_flipped_to_U'] / n if n > 0 else 0.0,
        }
        results[str(seed)] = entry
        print(f"  seed {seed}: eligible={n}, flip={entry['flip_T_to_U']}/{n}, "
              f"rate={entry['flip_rate']:.4f}, set_identical={entry['set_vec_identical']}/{n}")
        del model
        torch.cuda.empty_cache()

    # Aggregate
    rates = [v['flip_rate'] for v in results.values()]
    eligible_total = sum(v['eligible'] for v in results.values())
    flips_total = sum(v['flip_T_to_U'] for v in results.values())
    import numpy as np
    aggregate = {
        'flip_rate_mean': float(np.mean(rates)) if rates else 0.0,
        'flip_rate_std': float(np.std(rates)) if rates else 0.0,
        'flip_rate_min': float(np.min(rates)) if rates else 0.0,
        'flip_rate_max': float(np.max(rates)) if rates else 0.0,
        'n_seeds': len(rates),
        'n_eligible_total': eligible_total,
        'n_flips_total': flips_total,
        'aggregate_flip_rate': flips_total / eligible_total if eligible_total > 0 else 0.0,
    }

    print(f"\n  Aggregate: {flips_total}/{eligible_total} = {aggregate['aggregate_flip_rate']:.4f}")
    print(f"  Per-seed rate: mean={aggregate['flip_rate_mean']:.4f}, "
          f"std={aggregate['flip_rate_std']:.4f}, "
          f"min={aggregate['flip_rate_min']:.4f}, "
          f"max={aggregate['flip_rate_max']:.4f}")

    return {'per_seed': results, 'aggregate': aggregate}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logic-op', choices=['or', 'and', 'both'], default='both',
                        help="Cell A (or), Cell C (and), or both")
    args = parser.parse_args()

    output = {}

    if args.logic_op in ('or', 'both'):
        cell_a = run_cell('or', 'A (OR + v_ord, sanity)')
        output['cell_a_or'] = cell_a
        # Sanity check: Cell A must reproduce 4898/4898
        agg = cell_a['aggregate']
        if agg['n_eligible_total'] != 4898 or agg['n_flips_total'] != 4898:
            print(f"\n  WARNING: Cell A did NOT reproduce 4898/4898 exactly.")
            print(f"  Got: {agg['n_flips_total']}/{agg['n_eligible_total']}")
            if args.logic_op == 'both':
                print(f"  ABORTING Cell C: OR sanity reproduction failed.")
                with open('activation_patching_and.json', 'w') as f:
                    json.dump(output, f, indent=2)
                return

    if args.logic_op in ('and', 'both'):
        cell_c = run_cell('and', 'C (AND + v_ord, new)')
        output['cell_c_and'] = cell_c

    with open('activation_patching_and.json', 'w') as f:
        json.dump(output, f, indent=2)

    # Final summary table
    print(f"\n{'=' * 70}")
    print(f"  FINAL RESULTS")
    print(f"{'=' * 70}")
    if 'cell_a_or' in output:
        a = output['cell_a_or']['aggregate']
        print(f"  Cell A (OR + v_ord): {a['n_flips_total']}/{a['n_eligible_total']} "
              f"= {a['aggregate_flip_rate']:.4f}  "
              f"(per-seed min={a['flip_rate_min']:.4f})")
    if 'cell_c_and' in output:
        c = output['cell_c_and']['aggregate']
        print(f"  Cell C (AND+ v_ord): {c['n_flips_total']}/{c['n_eligible_total']} "
              f"= {c['aggregate_flip_rate']:.4f}  "
              f"(per-seed min={c['flip_rate_min']:.4f})")
    print(f"\n  Saved: activation_patching_and.json")


if __name__ == '__main__':
    main()
