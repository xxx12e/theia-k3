"""
N1.a full 39-rule Kleene K3 diagnostic on the 3-seed tuned BigTransformer
(post-LN, 3,641,859 params, standard K3).

Checkpoints: multi_seed_results/tf8l_tuned/seed_{42,123,256}/ckpt_final.pth
Coverage: 4 binary ops (AND, OR, IMP=3, IFF=4) x 9 (val_ord, val_set)
combinations + 3 unary NOT rules = 39 (paper Table~\\ref{tab:full_kleene_new}).
OP_NOT=2 is handled separately as a unary op; XOR is not in the training op
set and is absent from the test. Encoding: F=0, T=1, U=2.

Ground truth: standard Kleene K3, imported from _tf8l_model_def via
apply_logic (AND_T/OR_T/IMP_T/IFF_T/NOT_T); never re-defined here.

Per-rule protocol: 10,000 samples per rule (paper Table 4 protocol). Binary
rules reuse make_kleene_test(vo, vs, logic_op); unary NOT rules pin vs=T,
which is label-safe because apply_logic uses NOT_T[va] only, ignoring vb.
torch.manual_seed(42) is set once at the start of each seed's diagnostic
(run_kleene's convention), so per-rule RNG progression is deterministic and
reproducible across seeds. Pass threshold: per-rule accuracy > 99.0%
(same as paper).
"""
import json
import os
import sys
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import torch

from _tf8l_model_def import (
    BigTransformer, apply_logic,
    AND_T, OR_T, IMP_T, IFF_T, NOT_T,
    OP_AND, OP_OR, OP_NOT, OP_IMPLIES, OP_IFF,
    DEVICE,
    make_kleene_test,
)

EXPECTED_PARAMS = 3_641_859
SAMPLES_PER_RULE = 10_000
PASS_THRESHOLD = 99.0

CKPTS = [
    (42,  'multi_seed_results/tf8l_tuned/seed_42/ckpt_final.pth'),
    (123, 'multi_seed_results/tf8l_tuned/seed_123/ckpt_final.pth'),
    (256, 'multi_seed_results/tf8l_tuned/seed_256/ckpt_final.pth'),
]

TV_NAME = {0: 'F', 1: 'T', 2: 'U'}

# Binary ops (display name, opcode); OP_NOT=2 is unary and excluded here.
BINARY_OPS = [
    ('and', OP_AND),
    ('or',  OP_OR),
    ('imp', OP_IMPLIES),
    ('iff', OP_IFF),
]


def enumerate_rules():
    """Yield (rule_name, vo, vs, logic_op, rule_type) for 39 Kleene rules."""
    # 36 binary rules
    for op_name, op_idx in BINARY_OPS:
        for vo in (0, 1, 2):
            for vs in (0, 1, 2):
                name = f"{TV_NAME[vo]}_{op_name}_{TV_NAME[vs]}"
                yield (name, vo, vs, op_idx, 'binary')
    # 3 unary NOT rules: val_set pinned to T (deterministic)
    for vo in (0, 1, 2):
        name = f"not_{TV_NAME[vo]}"
        yield (name, vo, 1, OP_NOT, 'unary_not')


def compute_expected(vo, vs, logic_op):
    """Compute expected target by calling imported apply_logic on a
    1-sample tensor. Never re-defines truth tables locally."""
    op_t = torch.tensor([logic_op], dtype=torch.long, device=DEVICE)
    vo_t = torch.tensor([vo],       dtype=torch.long, device=DEVICE)
    vs_t = torch.tensor([vs],       dtype=torch.long, device=DEVICE)
    return apply_logic(op_t, vo_t, vs_t).item()


def run_39_rule(model):
    """Run 39 rules against model, return per-rule dict + (passed, total)."""
    # Match run_kleene() determinism convention: reset seed at start
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    model.eval()
    per_rule = {}
    passed = 0
    total = 0

    try:
        with torch.no_grad():
            for name, vo, vs, op_idx, rtype in enumerate_rules():
                expected = compute_expected(vo, vs, op_idx)
                data = make_kleene_test(vo, vs, op_idx, n=SAMPLES_PER_RULE)
                # data = (a, b, d, sb, su, au, bu, du, ar, rl, op_t, tgt)
                # Eval in FP32 (autocast removed 2026-04-20)
                logits = model(*data[:-1])
                preds = logits.argmax(dim=-1)
                acc = (preds == data[-1]).float().mean().item() * 100.0

                # Sanity: make_kleene_test must produce tgt == expected for all samples
                tgt_unique = torch.unique(data[-1]).tolist()
                assert tgt_unique == [expected], (
                    f"tgt inconsistency in {name}: unique tgt = {tgt_unique}, "
                    f"expected single value {expected}")

                per_rule[name] = {
                    'vo': TV_NAME[vo],
                    'vs': TV_NAME[vs] if rtype == 'binary' else TV_NAME[vs] + '(dontcare)',
                    'op': op_idx,
                    'type': rtype,
                    'expected': TV_NAME[expected],
                    'acc': acc,
                    'pass': acc > PASS_THRESHOLD,
                }
                total += 1
                if acc > PASS_THRESHOLD:
                    passed += 1
                tag = 'PASS' if acc > PASS_THRESHOLD else 'FAIL'
                print(f"  {name:12s}  exp={TV_NAME[expected]}  "
                      f"acc={acc:6.2f}%  {tag}")
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)

    return per_rule, passed, total


def run_seed(seed, ckpt_path):
    print(f"\n{'='*72}")
    print(f"  Seed {seed}  -- 39-rule standard K3 Kleene diagnostic")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"{'='*72}")

    model = BigTransformer().to(DEVICE)
    params = sum(p.numel() for p in model.parameters())
    assert params == EXPECTED_PARAMS, (
        f"Architecture mismatch for seed {seed}: "
        f"got {params}, expected {EXPECTED_PARAMS}")
    print(f"  BigTransformer params: {params:,}  (match)")

    state = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    print(f"  state_dict loaded OK")
    print()

    per_rule, passed, total = run_39_rule(model)

    # Summary stats
    accs = [r['acc'] for r in per_rule.values()]
    worst_rule_name = min(per_rule.keys(), key=lambda k: per_rule[k]['acc'])
    worst_acc = per_rule[worst_rule_name]['acc']
    grand_mean = sum(accs) / len(accs)

    out = {
        'seed': seed,
        'architecture': f"post-LN BigTransformer {params}",
        'K3_semantics': "standard strong Kleene",
        'apply_logic_source': "_tf8l_model_def (from tf8l_5seed_tuned.py)",
        'op_set_tested': ["AND", "OR", "IMP", "IFF", "NOT"],
        'n_rules': total,
        'samples_per_rule': SAMPLES_PER_RULE,
        'per_rule': per_rule,
        'worst_rule': worst_rule_name,
        'worst_acc': worst_acc,
        'grand_mean': grand_mean,
        'passed': passed,
        'all_pass_99': passed == total,
    }

    out_path = os.path.join(os.path.dirname(ckpt_path),
                            'kleene_diagnostic_N1a_full.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)

    print(f"\n  Passed: {passed}/{total}")
    print(f"  Worst rule: {worst_rule_name} @ {worst_acc:.2f}%")
    print(f"  Grand mean: {grand_mean:.4f}%")
    print(f"  Saved: {out_path}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return out


def main():
    all_results = {}
    for seed, ckpt_path in CKPTS:
        result = run_seed(seed, ckpt_path)
        all_results[str(seed)] = result

    # Aggregate across seeds
    rule_names = [name for name, _, _, _, _ in enumerate_rules()]
    aggregate_per_rule = {}
    for name in rule_names:
        seed_accs = [all_results[str(s)]['per_rule'][name]['acc']
                     for s, _ in CKPTS]
        mean = sum(seed_accs) / len(seed_accs)
        var = sum((a - mean) ** 2 for a in seed_accs) / len(seed_accs)
        std = var ** 0.5
        worst = min(seed_accs)
        all_pass = all(a > PASS_THRESHOLD for a in seed_accs)
        aggregate_per_rule[name] = {
            'mean': mean,
            'std': std,
            'min': worst,
            'all_pass_99': all_pass,
            'per_seed': {str(CKPTS[i][0]): seed_accs[i] for i in range(len(CKPTS))},
        }

    total_combinations = sum(r['n_rules'] for r in all_results.values())
    total_passed = sum(r['passed'] for r in all_results.values())

    aggregate = {
        'seeds': [s for s, _ in CKPTS],
        'architecture': "post-LN BigTransformer 3641859",
        'K3_semantics': "standard strong Kleene",
        'apply_logic_source': "_tf8l_model_def (from tf8l_5seed_tuned.py)",
        'op_set_tested': ["AND", "OR", "IMP", "IFF", "NOT"],
        'n_rules_per_seed': 39,
        'samples_per_rule': SAMPLES_PER_RULE,
        'total_combinations': total_combinations,
        'total_passed': total_passed,
        'all_39x3_pass_99': total_passed == total_combinations,
        'per_rule_aggregate': aggregate_per_rule,
        'per_seed_summary': {
            str(s): {
                'passed': all_results[str(s)]['passed'],
                'worst_rule': all_results[str(s)]['worst_rule'],
                'worst_acc': all_results[str(s)]['worst_acc'],
                'grand_mean': all_results[str(s)]['grand_mean'],
            }
            for s, _ in CKPTS
        },
    }

    agg_path = 'multi_seed_results/tf8l_tuned/kleene_diagnostic_N1a_aggregate.json'
    with open(agg_path, 'w') as f:
        json.dump(aggregate, f, indent=2)

    # Final report table
    print(f"\n{'='*72}")
    print(f"  N1.a FINAL SUMMARY (39-rule standard K3 on 3 seeds)")
    print(f"{'='*72}")
    for seed, _ in CKPTS:
        r = all_results[str(seed)]
        print(f"  seed {seed}:  {r['passed']:2d}/{r['n_rules']}  "
              f"worst: {r['worst_rule']:14s} @ {r['worst_acc']:6.2f}%  "
              f"grand mean {r['grand_mean']:.4f}%")
    print()
    print(f"  Aggregate:  {total_passed}/{total_combinations} combinations > {PASS_THRESHOLD}%")
    print(f"  all_39x3_pass_99 = {aggregate['all_39x3_pass_99']}")
    print(f"  Saved: {agg_path}")


if __name__ == '__main__':
    main()
