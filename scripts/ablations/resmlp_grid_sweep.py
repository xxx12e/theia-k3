"""
ResMLP backbone grid sweep — 3 new configs vs baseline.

Baseline (already run, in resmlp_backbone_ablation.py):
  4 blocks x 4d expansion, d=280, total step ~2,848,547

This sweep adds 3 new configs to the grid:
  C1: 4 blocks x 2d expansion, d=383, total ~2,776,047
  C2: 8 blocks x 2d expansion, d=276, total ~2,774,659
  C3: 8 blocks x 4d expansion, d=197, total ~2,754,419

All other variables identical to baseline (seeds, optimizer, schedule, batch
size, FP16, 3-phase pipeline, encoder, transition MLP, evaluation).

Stages:
  --stage sanity : param check + Phase 1 first 10 epochs on seed 42 per config
  --stage full   : 3 configs x 5 seeds x 3 phases serial run (~4-5 hours)

Stop gates:
  - Sanity: acc < 50% at epoch 10 -> architecture issue, abort
  - Full:   any config 4-seed strict mean 500-step >= 95% OR any compute-matched
            seed 500-step >= 99% -> halt and report
"""
import argparse
import io
import json
import os
import sys
import time
from contextlib import redirect_stdout

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler

from theia_chain_v3_5seed import (
    DEVICE, DIM, NUM_RANGE, LOGIC_OPS, SEEDS,
    NumEncoder, TransitionNet,
    gen_single_step_data, gen_chain_data,
    phase1, phase2, phase3, evaluate,
)


# --- Restart tracker ---
class _Tee(io.StringIO):
    """stdout tee: writes to underlying stream AND captures to buffer."""
    def __init__(self, stream):
        super().__init__()
        self.stream = stream
    def write(self, s):
        self.stream.write(s)
        return super().write(s)
    def flush(self):
        self.stream.flush()
        super().flush()


def phase1_with_restart_count(step_model, seed):
    """Run phase1 and capture stdout to count auto-restart triggers.
    Returns (best_overall_acc, n_restart). Live output still flows to stdout."""
    tee = _Tee(sys.stdout)
    with redirect_stdout(tee):
        result = phase1(step_model, train_seed=seed)
    output = tee.getvalue()
    n_restart = output.count('Restart')  # matches marker-prefixed and plain 'Restart' lines
    return result, n_restart

# --- Config ---
DIM_NUM = 128
DIM_EMB = 128
DIM_SET = 128
FEAT_DIM = DIM_NUM * 3 + DIM_SET + DIM_EMB * 3 + 4   # = 900

THEIA_REF_PARAMS = 2_751_232
PARAM_LOWER = int(THEIA_REF_PARAMS * 0.95)   # 2,613,670
PARAM_UPPER = int(THEIA_REF_PARAMS * 1.05)   # 2,888,793

CONFIGS = [
    # (name,         n_blocks, expansion, d)
    ('4blocks_x2d',         4, 2, 383),
    ('8blocks_x2d',         8, 2, 276),
    ('8blocks_x4d',         8, 4, 197),
]

SANITY_EPOCHS = 10
SANITY_GATE = 0.50
STOP_GATE_500_STEP_SEED = 0.99
STOP_GATE_500_STEP_AGG = 0.95   # 4-seed strict mean

OUT_DIR = 'multi_seed_results/resmlp_grid_sweep'


# --- Model ---
class SetEncoder(nn.Module):
    """Byte-identical to matched_mlp / resmlp baseline SetEncoder."""
    def __init__(self):
        super().__init__()
        self.set_enc = nn.Sequential(
            nn.Linear(21, DIM_SET), nn.GELU(), nn.Linear(DIM_SET, DIM_SET))
        self.unk = nn.Parameter(torch.randn(DIM_SET) * 0.02)
    def forward(self, s, su):
        sv = self.set_enc(s)
        mask = su.unsqueeze(-1).float()
        return sv * (1 - mask) + self.unk * mask


class ResBlock(nn.Module):
    """[Linear(d -> E*d) -> GELU -> LayerNorm(E*d) -> Linear(E*d -> d)] + skip
    (bottleneck-style residual block, post-LN)."""
    def __init__(self, d, expansion):
        super().__init__()
        self.fc1 = nn.Linear(d, d * expansion)
        self.act = nn.GELU()
        self.ln  = nn.LayerNorm(d * expansion)
        self.fc2 = nn.Linear(d * expansion, d)
    def forward(self, x):
        return x + self.fc2(self.ln(self.act(self.fc1(x))))


class ResMLPStepComputer(nn.Module):
    """ResMLP step computer. Encoder block byte-identical to matched_mlp."""
    def __init__(self, d, n_blocks, expansion):
        super().__init__()
        self.enc_a = NumEncoder()
        self.enc_b = NumEncoder()
        self.enc_d = NumEncoder()
        self.op_emb = nn.Embedding(4, DIM_EMB)
        self.rel_emb = nn.Embedding(6, DIM_EMB)
        self.logic_op_emb = nn.Embedding(LOGIC_OPS, DIM_EMB)
        self.set_encoder = SetEncoder()
        self.input_proj = nn.Linear(FEAT_DIM, d)
        self.blocks = nn.Sequential(*[ResBlock(d, expansion) for _ in range(n_blocks)])
        self.output_proj = nn.Linear(d, 3)

    def forward(self, a, b, op, d, rel, s, logic_op, au, bu, du, su):
        ea = self.enc_a(a, au); eb = self.enc_b(b, bu); ed = self.enc_d(d, du)
        es = self.set_encoder(s, su)
        eo = self.op_emb(op); er = self.rel_emb(rel); el = self.logic_op_emb(logic_op)
        unk = torch.stack([au.float(), bu.float(), du.float(), su.float()], dim=-1)
        feat = torch.cat([ea, eb, ed, es, eo, er, el, unk], dim=-1)
        return self.output_proj(self.blocks(self.input_proj(feat)))

    def forward_flat(self, x):
        return self(
            x[:, 0], x[:, 1], x[:, 2].long(), x[:, 3], x[:, 4].long(),
            x[:, 5:26], x[:, 26].long(),
            x[:, 27].bool(), x[:, 28].bool(), x[:, 29].bool(), x[:, 30].bool(),
        )


class ResMLPChain(nn.Module):
    def __init__(self, d, n_blocks, expansion):
        super().__init__()
        self.step = ResMLPStepComputer(d, n_blocks, expansion)
        self.transition = TransitionNet()
    def gumbel_st(self, logits, tau=0.5):
        soft = F.gumbel_softmax(logits, tau=tau, hard=False)
        hard = torch.zeros_like(soft).scatter_(-1, soft.argmax(-1, keepdim=True), 1.0)
        return hard - soft.detach() + soft


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# --- Sanity ---
def phase1_sanity(step_model, seed, max_epochs=SANITY_EPOCHS, bs=4096):
    """Mirror of phase1 inner loop; runs `max_epochs`. Returns per-epoch accs."""
    N = 2_000_000
    inputs, labels = gen_single_step_data(N, seed=seed)
    inp_g, lbl_g = inputs.to(DEVICE), labels.to(DEVICE)
    weights = torch.tensor([1.0, 2.0, 1.0], device=DEVICE)
    opt = torch.optim.AdamW(step_model.parameters(), lr=1e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max_epochs)
    scaler = GradScaler("cuda")
    accs = []
    for ep in range(1, max_epochs + 1):
        step_model.train()
        perm = torch.randperm(N, device=DEVICE)
        correct = 0
        for i in range(0, N, bs):
            idx = perm[i:i + bs]
            with autocast('cuda'):
                logits = step_model.forward_flat(inp_g[idx])
                loss = F.cross_entropy(logits, lbl_g[idx], weight=weights)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(step_model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            correct += (logits.detach().argmax(-1) == lbl_g[idx]).sum().item()
        sched.step()
        acc = correct / N
        accs.append(acc)
        print(f'    ep {ep:2d}: acc={acc:.4f}')
    del inp_g, lbl_g, inputs, labels
    torch.cuda.empty_cache()
    return accs


def estimate_p1_converge_epoch(accs, target=0.999):
    """Crude log-linear extrapolation on last 5 epochs' (1 - acc).
    Returns estimated epoch number to reach `target`, or None if not extrapolable."""
    if accs[-1] >= target:
        return len(accs)
    err = [max(1.0 - a, 1e-6) for a in accs]
    n_use = min(5, len(accs))
    xs = np.array(range(len(accs) - n_use, len(accs)), dtype=float)
    ys = np.log(np.array(err[-n_use:]))
    if len(xs) < 2:
        return None
    slope, intercept = np.polyfit(xs, ys, 1)
    if slope >= 0:
        return None  # not improving
    target_log = float(np.log(1 - target))
    ep_target = (target_log - intercept) / slope
    return max(int(round(ep_target)), len(accs) + 1)


def run_sanity():
    print(f'\n{"="*72}\n  RESMLP GRID SWEEP — Stage SANITY\n{"="*72}')
    print(f'  Param range: [{PARAM_LOWER:,}, {PARAM_UPPER:,}]  (THEIA 2,751,232 ± 5%)')
    print(f'  Sanity protocol: seed 42, Phase 1 first {SANITY_EPOCHS} epochs')
    print(f'  Gate: acc >= {SANITY_GATE} at epoch {SANITY_EPOCHS}')
    results = []
    for name, n_blocks, expansion, d in CONFIGS:
        print(f'\n{"="*72}')
        print(f'  CONFIG: {name}  (n_blocks={n_blocks}, expansion={expansion}d, d={d})')
        print(f'{"="*72}')
        torch.manual_seed(42); np.random.seed(42)
        model = ResMLPChain(d, n_blocks, expansion).to(DEVICE)
        step_p = count_params(model.step)
        total_p = count_params(model)
        print(f'  Step params:  {step_p:,}')
        print(f'  Total params: {total_p:,}')
        in_range = PARAM_LOWER <= step_p <= PARAM_UPPER
        print(f'  In range [{PARAM_LOWER:,}, {PARAM_UPPER:,}]: {in_range}')
        if not in_range:
            print(f'  Param OUT OF RANGE -- skipping sanity')
            results.append({'name': name, 'n_blocks': n_blocks, 'expansion': expansion,
                            'd': d, 'step_params': step_p, 'in_range': False,
                            'sanity_accs': None, 'gate_pass': False})
            del model; torch.cuda.empty_cache(); continue

        print(f'  Phase 1 first {SANITY_EPOCHS} epochs:')
        t0 = time.time()
        accs = phase1_sanity(model.step, seed=42, max_epochs=SANITY_EPOCHS)
        dt = time.time() - t0
        est = estimate_p1_converge_epoch(accs)
        gate_pass = accs[-1] >= SANITY_GATE
        print(f'  ep10 acc: {accs[-1]:.4f}  ({"PASS" if gate_pass else "FAIL"})')
        print(f'  Wall: {dt:.0f}s ({dt/SANITY_EPOCHS:.1f}s/ep)')
        print(f'  Estimated P1 converge epoch (≥99.9%): ~{est}')
        results.append({
            'name': name, 'n_blocks': n_blocks, 'expansion': expansion, 'd': d,
            'step_params': step_p, 'total_params': total_p, 'in_range': True,
            'sanity_accs': accs, 'final_acc': accs[-1],
            'wall_seconds': dt, 'estimated_p1_converge_epoch': est,
            'gate_pass': gate_pass,
        })
        del model; torch.cuda.empty_cache()

    # Summary
    print(f'\n{"="*72}\n  SANITY SUMMARY\n{"="*72}')
    print(f'  {"config":<14} {"d":>4} {"step_params":>13} {"10ep_acc":>9} '
          f'{"est_P1":>7} {"gate":>6}')
    for r in results:
        est = r.get('estimated_p1_converge_epoch')
        est_s = f'~{est}' if est else '-'
        if r.get('sanity_accs') is None:
            print(f'  {r["name"]:<14} {r["d"]:>4} {r["step_params"]:>13,} '
                  f'{"-":>8}   {"-":>6}   {"SKIP":>6}')
        else:
            print(f'  {r["name"]:<14} {r["d"]:>4} {r["step_params"]:>13,} '
                  f'{r["final_acc"]:>8.4f}   {est_s:>5}   '
                  f'{"PASS" if r["gate_pass"] else "FAIL":>6}')

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, 'sanity.json'), 'w') as f:
        json.dump(results, f, indent=2)

    all_pass = all(r['gate_pass'] for r in results)
    if all_pass:
        print(f'\n  ALL 3 CONFIGS PASSED SANITY.')
        print(f'  Ready for: python resmlp_grid_sweep.py --stage full')
    else:
        print(f'\n  At least one sanity FAILED. Stopping.')


def run_full():
    print(f'\n{"="*72}\n  RESMLP GRID SWEEP — Stage FULL\n{"="*72}')
    os.makedirs(OUT_DIR, exist_ok=True)
    TEST_STEPS = [5, 10, 50, 100, 500]

    all_results = {}
    early_abort = False
    early_abort_reason = None

    for name, n_blocks, expansion, d in CONFIGS:
        print(f'\n{"="*72}')
        print(f'  CONFIG: {name}  (n_blocks={n_blocks}, expansion={expansion}d, d={d})')
        print(f'{"="*72}')

        config_dir = os.path.join(OUT_DIR, name)
        os.makedirs(config_dir, exist_ok=True)

        config_results = {
            'config': name, 'n_blocks': n_blocks, 'expansion': expansion, 'd': d,
            'seeds': SEEDS,
            'p1': [], 'p2': [], 'p3': [],
            'elapsed': [],
            'p1_restart_count': [],
            'results': {str(s): [] for s in TEST_STEPS},
        }

        for si, seed in enumerate(SEEDS):
            print(f'\n--- {name} seed {si+1}/{len(SEEDS)}: {seed} ---')
            torch.manual_seed(seed); np.random.seed(seed)
            model = ResMLPChain(d, n_blocks, expansion).to(DEVICE)
            t0 = time.time()
            p1, n_restart = phase1_with_restart_count(model.step, seed)
            p2 = phase2(model, train_seed=seed)
            p3 = phase3(model, train_seed=seed)
            elapsed = time.time() - t0
            config_results['p1'].append(p1)
            config_results['p2'].append(p2)
            config_results['p3'].append(p3)
            config_results['elapsed'].append(elapsed)
            config_results['p1_restart_count'].append(n_restart)
            print(f'  Training: {elapsed:.0f}s | P1={p1:.2%} P2={p2:.2%} P3={p3:.2%} '
                  f'| restarts={n_restart}')

            for steps in TEST_STEPS:
                acc, s1, _ = evaluate(model, steps, seed=seed + 5000, N=10000)
                config_results['results'][str(steps)].append(acc)
                tag = 'PASS' if acc > 0.99 else 'WARN' if acc > 0.95 else 'FAIL'
                print(f'  {steps:>4d}-step: {acc:.2%} (step1={s1:.2%}) {tag}')

            # Stop gate per-seed
            last_500 = config_results['results']['500'][-1]
            if last_500 >= STOP_GATE_500_STEP_SEED:
                print(f'\n  STOP GATE: {name} seed {seed} 500-step = '
                      f'{last_500:.2%} >= {STOP_GATE_500_STEP_SEED:.0%}')
                early_abort = True
                early_abort_reason = (f'{name} seed {seed} 500-step = '
                                      f'{last_500:.2%}')
                del model; torch.cuda.empty_cache()
                break

            del model; torch.cuda.empty_cache()

        # Per-config aggregate
        if config_results['results']['500']:
            vals = config_results['results']['500']
            config_results['agg_500'] = {
                'mean': float(np.mean(vals)),
                'std': float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                'min': float(np.min(vals)),
                'max': float(np.max(vals)),
                'n': len(vals),
            }

        all_results[name] = config_results

        # Save per-config
        with open(os.path.join(config_dir, 'config_raw.json'), 'w') as f:
            json.dump(config_results, f, indent=2)

        if early_abort:
            break

        # Stop gate per-config aggregate
        if 'agg_500' in config_results and config_results['agg_500']['mean'] >= STOP_GATE_500_STEP_AGG:
            print(f'\n  STOP GATE: {name} 5-seed mean 500-step = '
                  f'{config_results["agg_500"]["mean"]:.2%} >= {STOP_GATE_500_STEP_AGG:.0%}')
            early_abort = True
            early_abort_reason = (f'{name} aggregate mean 500-step = '
                                  f'{config_results["agg_500"]["mean"]:.2%}')
            break

    # --- Post-processing: 4-seed strict + 5-seed as-spec aggregates,
    #     plus pooled / best-worst / degradation-curve extras ---
    BASELINE_PATH = 'multi_seed_results/resmlp_backbone_ablation/raw.json'
    baseline = None
    if os.path.exists(BASELINE_PATH):
        with open(BASELINE_PATH) as f:
            baseline = json.load(f)
        print(f'\n  Loaded baseline: {BASELINE_PATH}')
    else:
        print(f'\n  WARNING: baseline JSON not found at {BASELINE_PATH}; '
              f'cross-config pooled aggregate will exclude baseline.')

    def restart_seeds_from_elapsed(elapsed_list, seeds_list, threshold=1.5):
        """Heuristic for baseline (no explicit restart count): seeds whose
        elapsed > threshold * median are flagged as restarted."""
        if not elapsed_list: return set()
        med = float(np.median(elapsed_list))
        return {seeds_list[i] for i, e in enumerate(elapsed_list) if e > threshold * med}

    # Build per-config strict / as-spec aggregates
    config_aggs = {}  # config_name -> dict
    # New configs (have explicit restart count)
    for name, _, _, _ in CONFIGS:
        if name not in all_results: continue
        cr = all_results[name]
        seeds = cr['seeds'][:len(cr['p1'])]
        accs500 = cr['results']['500']
        if not accs500: continue
        restart_counts = cr.get('p1_restart_count', [0]*len(seeds))
        restart_seed_set = {seeds[i] for i, r in enumerate(restart_counts) if r > 0}
        strict_idx = [i for i, s in enumerate(seeds) if s not in restart_seed_set]
        strict_vals = [accs500[i] for i in strict_idx]
        as_spec_vals = list(accs500)
        # Per-step mean for degradation curve
        degradation = {}
        for steps in [5, 10, 50, 100, 500]:
            sv = cr['results'].get(str(steps), [])
            if sv:
                degradation[str(steps)] = {
                    'mean': float(np.mean(sv)),
                    'std': float(np.std(sv, ddof=1)) if len(sv) > 1 else 0.0,
                    'n': len(sv),
                }
        config_aggs[name] = {
            'seeds': seeds,
            'restart_seeds': sorted(restart_seed_set),
            'strict_seeds': [seeds[i] for i in strict_idx],
            '500_strict_mean': float(np.mean(strict_vals)) if strict_vals else None,
            '500_strict_std':  float(np.std(strict_vals, ddof=1)) if len(strict_vals) > 1 else 0.0,
            '500_strict_n': len(strict_vals),
            '500_as_spec_mean': float(np.mean(as_spec_vals)),
            '500_as_spec_std':  float(np.std(as_spec_vals, ddof=1)) if len(as_spec_vals) > 1 else 0.0,
            '500_as_spec_n': len(as_spec_vals),
            'per_step_degradation': degradation,
            'per_seed_500': dict(zip([str(s) for s in seeds], accs500)),
        }

    # Baseline (use elapsed-time heuristic, no explicit restart count)
    if baseline is not None:
        b_seeds = baseline.get('seeds_completed', baseline.get('seeds', []))
        b_500 = baseline['results']['500']
        b_elapsed = baseline['elapsed']
        b_restart_seed_set = restart_seeds_from_elapsed(b_elapsed, b_seeds)
        b_strict_idx = [i for i, s in enumerate(b_seeds) if s not in b_restart_seed_set]
        b_strict_vals = [b_500[i] for i in b_strict_idx]
        b_degradation = {}
        for steps in [5, 10, 50, 100, 500]:
            sv = baseline['results'].get(str(steps), [])
            if sv:
                b_degradation[str(steps)] = {
                    'mean': float(np.mean(sv)),
                    'std': float(np.std(sv, ddof=1)) if len(sv) > 1 else 0.0,
                    'n': len(sv),
                }
        config_aggs['baseline_4blocks_x4d'] = {
            'seeds': b_seeds,
            'restart_seeds': sorted(b_restart_seed_set),
            'strict_seeds': [b_seeds[i] for i in b_strict_idx],
            '500_strict_mean': float(np.mean(b_strict_vals)) if b_strict_vals else None,
            '500_strict_std':  float(np.std(b_strict_vals, ddof=1)) if len(b_strict_vals) > 1 else 0.0,
            '500_strict_n': len(b_strict_vals),
            '500_as_spec_mean': float(np.mean(b_500)),
            '500_as_spec_std':  float(np.std(b_500, ddof=1)) if len(b_500) > 1 else 0.0,
            '500_as_spec_n': len(b_500),
            'per_step_degradation': b_degradation,
            'per_seed_500': dict(zip([str(s) for s in b_seeds], b_500)),
            'detection_method': 'elapsed_time_heuristic_1.5x_median',
        }

    # Extra 1: cross-config pooled strict aggregate
    pooled_500_strict = []
    for cname, cdata in config_aggs.items():
        for sd, val in cdata['per_seed_500'].items():
            if int(sd) not in cdata['restart_seeds']:
                pooled_500_strict.append(val)
    cross_pooled = None
    if pooled_500_strict:
        cross_pooled = {
            'mean': float(np.mean(pooled_500_strict)),
            'std': float(np.std(pooled_500_strict, ddof=1)) if len(pooled_500_strict) > 1 else 0.0,
            'min': float(np.min(pooled_500_strict)),
            'max': float(np.max(pooled_500_strict)),
            'n': len(pooled_500_strict),
            'note': 'pooled across 4 configs (3 new + baseline) with strict (non-restart) seeds only',
        }

    # Extra 2: per-seed best/worst across configs
    per_seed_best_worst = {}
    all_seeds = set()
    for cdata in config_aggs.values():
        all_seeds.update(int(s) for s in cdata['per_seed_500'].keys())
    for seed in sorted(all_seeds):
        row = []
        for cname, cdata in config_aggs.items():
            v = cdata['per_seed_500'].get(str(seed))
            if v is not None:
                row.append((cname, v))
        if not row: continue
        best = max(row, key=lambda x: x[1])
        worst = min(row, key=lambda x: x[1])
        per_seed_best_worst[str(seed)] = {
            'best_config': best[0], 'best_acc': best[1],
            'worst_config': worst[0], 'worst_acc': worst[1],
            'spread_pp': (best[1] - worst[1]) * 100.0,
            'all_configs': dict(row),
        }

    # Aggregate save
    final = {
        'configs': CONFIGS,
        'theia_ref_params': THEIA_REF_PARAMS,
        'param_range': [PARAM_LOWER, PARAM_UPPER],
        'stop_gate_seed_500': STOP_GATE_500_STEP_SEED,
        'stop_gate_agg_500': STOP_GATE_500_STEP_AGG,
        'early_abort': early_abort,
        'early_abort_reason': early_abort_reason,
        'per_config': all_results,
        # extra aggregates
        'baseline_path': BASELINE_PATH,
        'baseline_loaded': baseline is not None,
        'config_aggregates': config_aggs,
        'extra_cross_config_pooled_strict': cross_pooled,
        'extra_per_seed_best_worst': per_seed_best_worst,
    }
    with open(os.path.join(OUT_DIR, 'sweep_raw.json'), 'w') as f:
        json.dump(final, f, indent=2)

    # Markdown report
    lines = []
    lines.append('# ResMLP Backbone Grid Sweep\n')
    lines.append(f'**Date**: {time.strftime("%Y-%m-%d %H:%M:%S")}')
    lines.append(f'**Configs (besides baseline 4×4d d=280)**:\n')
    lines.append('| Config | n_blocks | expansion | d | step_params |')
    lines.append('|---|---|---|---|---|')
    for name, n, e, d in CONFIGS:
        if name in all_results:
            cr = all_results[name]
            ref_step_p = count_params(ResMLPStepComputer(d, n, e))
            lines.append(f'| {name} | {n} | {e}d | {d} | {ref_step_p:,} |')
    lines.append('')

    if early_abort:
        lines.append(f'## STOP GATE TRIGGERED\n\nReason: {early_abort_reason}\n')

    lines.append('## Per-config 500-step results\n')
    lines.append('| Config | n | mean | std | min | max | per-seed |')
    lines.append('|---|---|---|---|---|---|---|')
    for name, _, _, _ in CONFIGS:
        if name not in all_results: continue
        cr = all_results[name]
        if 'agg_500' not in cr: continue
        agg = cr['agg_500']
        per = ', '.join(f'{v:.2%}' for v in cr['results']['500'])
        lines.append(f'| {name} | {agg["n"]} | {agg["mean"]:.2%} | {agg["std"]:.2%} | '
                     f'{agg["min"]:.2%} | {agg["max"]:.2%} | {per} |')
    lines.append('')

    lines.append('## Phase metrics per (config, seed)\n')
    lines.append('| Config | Seed | P1 | P2 | P3 | t(s) | restarts | 500-step |')
    lines.append('|---|---|---|---|---|---|---|---|')
    for name, _, _, _ in CONFIGS:
        if name not in all_results: continue
        cr = all_results[name]
        rests = cr.get('p1_restart_count', [0]*len(cr['p1']))
        for i, seed in enumerate(cr['seeds'][:len(cr['p1'])]):
            five = cr['results']['500'][i] if i < len(cr['results']['500']) else None
            five_s = f'{five:.2%}' if five is not None else '-'
            r = rests[i] if i < len(rests) else '-'
            lines.append(f'| {name} | {seed} | {cr["p1"][i]:.2%} | '
                         f'{cr["p2"][i]:.2%} | {cr["p3"][i]:.2%} | '
                         f'{cr["elapsed"][i]:.0f} | {r} | {five_s} |')
    lines.append('')

    # --- Extra report sections ---
    lines.append('## Per-config 500-step: 4-seed strict vs 5-seed as-specified\n')
    lines.append('"strict" = excludes seeds that triggered Phase-1 auto-restart.\n')
    lines.append('| Config | n_strict | strict_mean | strict_std | as_spec_mean | as_spec_std | restart_seeds |')
    lines.append('|---|---|---|---|---|---|---|')
    for cname, cdata in config_aggs.items():
        rs = ', '.join(str(s) for s in cdata['restart_seeds']) or '(none)'
        sm = cdata['500_strict_mean']
        sm_s = f'{sm:.2%}' if sm is not None else '-'
        lines.append(f'| {cname} | {cdata["500_strict_n"]} | {sm_s} | '
                     f'{cdata["500_strict_std"]:.2%} | {cdata["500_as_spec_mean"]:.2%} | '
                     f'{cdata["500_as_spec_std"]:.2%} | {rs} |')
    lines.append('')

    if cross_pooled is not None:
        lines.append('## Extra 1: Cross-config pooled strict aggregate\n')
        lines.append(f'Pool: 4 configs × strict (non-restart) seeds = '
                     f'**{cross_pooled["n"]} samples**\n')
        lines.append('| Aggregate | Value |')
        lines.append('|---|---|')
        lines.append(f'| Pooled mean | **{cross_pooled["mean"]:.2%}** |')
        lines.append(f'| Pooled std (sample) | {cross_pooled["std"]:.2%} |')
        lines.append(f'| Pooled min | {cross_pooled["min"]:.2%} |')
        lines.append(f'| Pooled max | {cross_pooled["max"]:.2%} |')
        lines.append(f'| n (strict pool) | {cross_pooled["n"]} |')
        lines.append('')

    if per_seed_best_worst:
        lines.append('## Extra 2: Configuration-level best / worst per seed\n')
        lines.append('| Seed | Best config | Best acc | Worst config | Worst acc | Spread (pp) |')
        lines.append('|---|---|---|---|---|---|')
        for seed in sorted(per_seed_best_worst.keys(), key=int):
            d = per_seed_best_worst[seed]
            lines.append(f'| {seed} | {d["best_config"]} | {d["best_acc"]:.2%} | '
                         f'{d["worst_config"]} | {d["worst_acc"]:.2%} | '
                         f'{d["spread_pp"]:+.2f} |')
        lines.append('')

    lines.append('## Extra 3: Per-config 5→500 step degradation curve (5-seed mean)\n')
    lines.append('| Config | 5-step | 10-step | 50-step | 100-step | 500-step |')
    lines.append('|---|---|---|---|---|---|')
    for cname, cdata in config_aggs.items():
        deg = cdata.get('per_step_degradation', {})
        row = [f'| {cname} |']
        for sk in ['5', '10', '50', '100', '500']:
            if sk in deg:
                row.append(f' {deg[sk]["mean"]:.2%} ± {deg[sk]["std"]:.2%} |')
            else:
                row.append(' - |')
        lines.append(''.join(row))
    lines.append('')

    with open(os.path.join(OUT_DIR, 'report.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f'\n  Report:  {os.path.join(OUT_DIR, "report.md")}')
    print(f'  Raw:     {os.path.join(OUT_DIR, "sweep_raw.json")}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=['sanity', 'full'], default='sanity')
    args = p.parse_args()
    if args.stage == 'sanity':
        run_sanity()
    else:
        run_full()


if __name__ == '__main__':
    main()
