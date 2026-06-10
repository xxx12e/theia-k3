"""
ResMLP backbone ablation (chain section / backbone appendix): tests whether a
structured-but-non-domain-separated backbone (residual MLP with
expansion-contraction blocks) can sustain 500-step chain accuracy under the
identical Gumbel-ST three-phase pipeline, separating "any structure" from
"domain-segregated structure".

Architecture: identical input encoders + 4 residual blocks + output projection;
each block is x + Linear(d -> 4d) -> GELU -> LayerNorm(4d) -> Linear(4d -> d),
d chosen for total params ~2.75M (matches the parameter-matched flat MLP).
Pipeline identical to matched_mlp_backbone_ablation.py (same Phase 1/2/3
trainers, data, seeds, evaluator).

Stop gates:
  (a) param count outside [2,613,670, 2,888,793] (= 2.75M ± 5%) -> abort
  (b) sanity (seed 42, Phase 1 first 10 epochs) acc < 50% -> abort
  (c) any seed's Phase 3 500-step accuracy > 99% -> halt remaining seeds and
      report immediately

Outputs:
  multi_seed_results/resmlp_backbone_ablation/report.md
  multi_seed_results/resmlp_backbone_ablation/raw.json
"""
import os
import sys
import time
import json

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

# Reuse THEIA's chain pipeline verbatim (matches matched_mlp_backbone_ablation.py)
from theia_chain_v3_5seed import (
    DEVICE, DIM, NUM_RANGE, LOGIC_OPS, SEEDS,
    NumEncoder, TransitionNet,
    gen_single_step_data, gen_chain_data,
    phase1, phase2, phase3, evaluate,
)

# --- Config ---
DIM_NUM = 128
DIM_EMB = 128
DIM_SET = 128
D_HIDDEN = 280       # ResMLP hidden dim. Solved from 32d^2 + (input_dim + output_dim)*d ~= 2.75M
                     # for input_dim=900, output_dim=3 -> d~=280; total step ~= 2.85M (within ±5%)
N_BLOCKS = 4
EXPANSION = 4

THEIA_REF_PARAMS = 2_751_232
PARAM_LOWER = int(THEIA_REF_PARAMS * 0.95)
PARAM_UPPER = int(THEIA_REF_PARAMS * 1.05)

SANITY_EPOCHS = 10
SANITY_GATE = 0.50      # acc must reach >= 50% by epoch 10
STOP_GATE_500_STEP = 0.99  # if any seed's 500-step > 99%, stop and report


# --- Model ---
class SetEncoder(nn.Module):
    """Mirror of matched_mlp_backbone_ablation.SetEncoder (byte-identical)."""
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
    """[Linear(d -> 4d) -> GELU -> LayerNorm(4d) -> Linear(4d -> d)] + skip"""
    def __init__(self, d, expansion=EXPANSION):
        super().__init__()
        self.fc1 = nn.Linear(d, d * expansion)
        self.act = nn.GELU()
        self.ln  = nn.LayerNorm(d * expansion)
        self.fc2 = nn.Linear(d * expansion, d)

    def forward(self, x):
        return x + self.fc2(self.ln(self.act(self.fc1(x))))


class ResMLPStepComputer(nn.Module):
    """ResMLP step computer.

    Encoder block (kept identical to matched_mlp_backbone_ablation.py so the
    ablation isolates the *reasoning* stack):
      NumEncoder x 3 (a, b, d) + SetEncoder + 3 nn.Embedding (op, rel, logic_op)
      + 4 raw unk flag bits.

    Reasoning block (the ablation target):
      Linear(900 -> d) -> [ResBlock x N_BLOCKS] -> Linear(d -> 3)
    """
    def __init__(self, d=D_HIDDEN, n_blocks=N_BLOCKS):
        super().__init__()
        # Encoders (identical to matched_mlp)
        self.enc_a = NumEncoder()
        self.enc_b = NumEncoder()
        self.enc_d = NumEncoder()
        self.op_emb = nn.Embedding(4, DIM_EMB)
        self.rel_emb = nn.Embedding(6, DIM_EMB)
        self.logic_op_emb = nn.Embedding(LOGIC_OPS, DIM_EMB)
        self.set_encoder = SetEncoder()

        feat_dim = DIM_NUM * 3 + DIM_SET + DIM_EMB * 3 + 4   # = 900
        self.input_proj = nn.Linear(feat_dim, d)
        self.blocks = nn.Sequential(*[ResBlock(d) for _ in range(n_blocks)])
        self.output_proj = nn.Linear(d, 3)

    def forward(self, a, b, op, d, rel, s, logic_op, au, bu, du, su):
        ea = self.enc_a(a, au); eb = self.enc_b(b, bu); ed = self.enc_d(d, du)
        es = self.set_encoder(s, su)
        eo = self.op_emb(op); er = self.rel_emb(rel); el = self.logic_op_emb(logic_op)
        unk_flags = torch.stack([au.float(), bu.float(), du.float(), su.float()], dim=-1)
        feat = torch.cat([ea, eb, ed, es, eo, er, el, unk_flags], dim=-1)
        x = self.input_proj(feat)
        x = self.blocks(x)
        return self.output_proj(x)

    def forward_flat(self, x):
        return self(
            x[:, 0], x[:, 1], x[:, 2].long(), x[:, 3], x[:, 4].long(),
            x[:, 5:26], x[:, 26].long(),
            x[:, 27].bool(), x[:, 28].bool(), x[:, 29].bool(), x[:, 30].bool(),
        )


class ResMLPChain(nn.Module):
    def __init__(self):
        super().__init__()
        self.step = ResMLPStepComputer()
        self.transition = TransitionNet()

    def gumbel_st(self, logits, tau=0.5):
        soft = F.gumbel_softmax(logits, tau=tau, hard=False)
        hard = torch.zeros_like(soft).scatter_(-1, soft.argmax(-1, keepdim=True), 1.0)
        return hard - soft.detach() + soft


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# --- Sanity: 10 epochs of phase1 inner-loop (no plateau detection / restarts) ---
def phase1_sanity(step_model, train_seed, max_epochs=SANITY_EPOCHS, bs=4096):
    """Mirror of phase1's inner training loop (theia_chain_v3_5seed.py L253-269);
    runs `max_epochs` and reports epoch-wise acc."""
    N = 2_000_000
    inputs, labels = gen_single_step_data(N, seed=train_seed)
    inp_g, lbl_g = inputs.to(DEVICE), labels.to(DEVICE)
    weights = torch.tensor([1.0, 2.0, 1.0], device=DEVICE)

    opt = torch.optim.AdamW(step_model.parameters(), lr=1e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max_epochs)
    scaler = GradScaler("cuda")

    final_acc = 0.0
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
        print(f"    sanity ep {ep:2d}: acc={acc:.4f}")
        final_acc = acc

    # Free training data immediately
    del inp_g, lbl_g, inputs, labels
    torch.cuda.empty_cache()
    return final_acc


# --- Main ---
def main():
    torch.backends.cudnn.benchmark = True
    OUT_DIR = "multi_seed_results/resmlp_backbone_ablation"
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"\n{'='*72}")
    print(f"  ResMLP Backbone Ablation (5-seed)")
    print(f"{'='*72}")
    print(f"  d={D_HIDDEN}, n_blocks={N_BLOCKS}, expansion={EXPANSION}")
    print(f"  Block: x + Linear(d->{EXPANSION}d) -> GELU -> LayerNorm({EXPANSION}d) -> Linear({EXPANSION}d->d)")
    print(f"  Seeds: {SEEDS}")
    print(f"  Device: {DEVICE}")

    # ---- Param verification ----
    _probe = ResMLPChain()
    n_step  = count_params(_probe.step)
    n_total = count_params(_probe)
    print(f"\n  ResMLPStepComputer params: {n_step:,}")
    print(f"  ResMLPChain total params:  {n_total:,}")
    print(f"  THEIA step ref:            {THEIA_REF_PARAMS:,}")
    print(f"  Ratio:                     {n_step / THEIA_REF_PARAMS:.4f}x")
    print(f"  Acceptable [-5%,+5%]:      [{PARAM_LOWER:,}, {PARAM_UPPER:,}]")
    if not (PARAM_LOWER <= n_step <= PARAM_UPPER):
        print(f"  PARAM COUNT OUT OF RANGE -- abort.")
        sys.exit(1)
    print(f"  PARAM COUNT IN RANGE")
    del _probe
    torch.cuda.empty_cache()

    # ---- Sanity check on seed 42 ----
    print(f"\n{'='*72}")
    print(f"  Sanity: seed {SEEDS[0]}, Phase 1 inner-loop {SANITY_EPOCHS} epochs")
    print(f"{'='*72}")
    torch.manual_seed(SEEDS[0]); np.random.seed(SEEDS[0])
    sanity_model = ResMLPChain().to(DEVICE)
    t_san = time.time()
    sanity_acc = phase1_sanity(sanity_model.step, train_seed=SEEDS[0])
    print(f"  Sanity {SANITY_EPOCHS}-ep acc: {sanity_acc:.4f}   (gate: >= {SANITY_GATE:.2f})")
    sanity_dt = time.time() - t_san
    if sanity_acc < SANITY_GATE:
        print(f"\n  SANITY GATE FAILED (acc {sanity_acc:.4f} < {SANITY_GATE:.2f})")
        print(f"  Architecture has problem. Aborting.")
        sys.exit(1)
    print(f"  SANITY PASSED ({sanity_dt:.0f}s)")
    del sanity_model
    torch.cuda.empty_cache()

    # ---- Full 5-seed pipeline (re-init seed 42 to keep 5-seed treatment symmetric) ----
    TEST_STEPS = [5, 10, 50, 100, 500]
    results = {s: [] for s in TEST_STEPS}
    p1_log, p2_log, p3_log = [], [], []
    elapsed_log = []
    early_abort = False
    early_abort_reason = None
    seeds_completed = []

    for si, seed in enumerate(SEEDS):
        print(f"\n{'='*72}")
        print(f"  Seed {si+1}/{len(SEEDS)}: {seed}")
        print(f"{'='*72}")

        torch.manual_seed(seed); np.random.seed(seed)
        model = ResMLPChain().to(DEVICE)
        t_seed = time.time()

        p1 = phase1(model.step, train_seed=seed)
        p1_log.append(p1)
        print(f"  Phase 1 done: {p1:.2%}")

        p2 = phase2(model, train_seed=seed)
        p2_log.append(p2)
        print(f"  Phase 2 done: {p2:.2%}")

        p3 = phase3(model, train_seed=seed)
        p3_log.append(p3)
        print(f"  Phase 3 done: {p3:.2%}")

        elapsed = time.time() - t_seed
        elapsed_log.append(elapsed)
        print(f"  Training time: {elapsed:.0f}s | P1={p1:.2%} P2={p2:.2%} P3={p3:.2%}")

        for steps in TEST_STEPS:
            acc, s1, cls = evaluate(model, steps, seed=seed + 5000, N=10000)
            results[steps].append(acc)
            tag = "PASS" if acc > 0.99 else "WARN" if acc > 0.95 else "FAIL"
            print(f"  {steps:>4d}-step: {acc:.2%} (step1={s1:.2%}) {tag}")

        seeds_completed.append(seed)

        # Stop gate: any seed's Phase 3 500-step accuracy > 99% halts remaining seeds.
        if results[500][-1] > STOP_GATE_500_STEP:
            print(f"\n  STOP GATE TRIGGERED")
            print(f"     seed {seed}: 500-step = {results[500][-1]:.2%} > {STOP_GATE_500_STEP:.0%}")
            print(f"     ResMLP SURVIVES compositional generalization at 500 steps.")
            print(f"     This means structured-but-non-domain-separated backbone ALSO works")
            print(f"     -> falsifies H1 ('domain segregation is the credit-bearing factor') at this config.")
            print(f"     Halting remaining seeds; reporting immediately.")
            early_abort = True
            early_abort_reason = (f"seed {seed} 500-step = {results[500][-1]:.2%} > "
                                  f"{STOP_GATE_500_STEP:.0%}")
            del model; torch.cuda.empty_cache()
            break

        del model; torch.cuda.empty_cache()

    # ---- Report ----
    lines = []
    lines.append("# Stage 2: ResMLP Backbone Ablation\n")
    lines.append(f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Architecture**: 4 residual blocks, each `x + Linear(d -> {EXPANSION}d) "
                 f"-> GELU -> LayerNorm -> Linear({EXPANSION}d -> d)`")
    lines.append(f"**d** = {D_HIDDEN}")
    lines.append(f"**Seeds attempted**: {len(seeds_completed)} / {len(SEEDS)}")
    lines.append(f"**Early abort**: {early_abort}")
    if early_abort:
        lines.append(f"**Abort reason**: {early_abort_reason}")
    lines.append("")
    lines.append("## Param count\n")
    lines.append(f"- ResMLPStepComputer: **{n_step:,}**")
    lines.append(f"- THEIA step reference: **{THEIA_REF_PARAMS:,}**")
    lines.append(f"- Ratio: **{n_step / THEIA_REF_PARAMS:.4f}x**")
    lines.append(f"- Acceptable range (±5%): [{PARAM_LOWER:,}, {PARAM_UPPER:,}] ✓\n")
    lines.append(f"## Sanity check\n")
    lines.append(f"- Seed {SEEDS[0]}, Phase-1 first {SANITY_EPOCHS} epochs")
    lines.append(f"- Final acc: **{sanity_acc:.4f}** (gate: >= {SANITY_GATE:.2f}) ✓")
    lines.append(f"- Wall time: {sanity_dt:.0f}s")
    lines.append("")
    lines.append("## Phase results per seed\n")
    lines.append("| seed | P1 | P2 | P3 | time(s) |")
    lines.append("|---|---|---|---|---|")
    for i, seed in enumerate(seeds_completed):
        lines.append(f"| {seed} | {p1_log[i]:.2%} | {p2_log[i]:.2%} | {p3_log[i]:.2%} | "
                     f"{elapsed_log[i]:.0f} |")
    lines.append("")

    if results[500]:
        lines.append(f"## Length generalization (mean ± std over completed seeds)\n")
        lines.append("| Steps | Mean | Std | Min | Max | Per-seed |")
        lines.append("|---|---|---|---|---|---|")
        for steps in TEST_STEPS:
            vals = results[steps]
            if not vals: continue
            mean, std = float(np.mean(vals)), float(np.std(vals))
            mn, mx = float(np.min(vals)), float(np.max(vals))
            per_seed_str = ", ".join(f"{v:.2%}" for v in vals)
            lines.append(f"| {steps} | {mean:.2%} | {std:.2%} | {mn:.2%} | {mx:.2%} | {per_seed_str} |")
        lines.append("")

        lines.append("## Comparison to paper Table 9 / parameter-matched flat MLP\n")
        lines.append(f"- Paper's parameter-matched flat MLP (2.75M, HIDDEN=1243, 3-layer):")
        lines.append(f"  collapses to chance by 50 steps (per `matched_mlp_backbone_ablation.py` runs)")
        lines.append(f"- ResMLP (this experiment, 4 residual blocks, ~2.85M):")
        if results[500]:
            mean500 = float(np.mean(results[500]))
            if mean500 >= 0.99:
                lines.append(f"  **also >= 99% at 500 steps** ({mean500:.2%}) -> falsifies H1 at this config")
                lines.append(f"  (structured-but-non-domain-separated backbone ALSO survives)")
            elif mean500 >= 0.95:
                lines.append(f"  **{mean500:.2%}** at 500 steps - intermediate; may need further analysis")
            elif mean500 >= 0.50:
                lines.append(f"  **{mean500:.2%}** at 500 steps - degraded but not chance; "
                             f"structure helps but doesn't fully rescue")
            else:
                lines.append(f"  **{mean500:.2%}** at 500 steps - effectively chance (chance = 33%)")
                lines.append(f"  -> H1 STRENGTHENED: residual structure alone insufficient; "
                             f"domain segregation is required")

    with open(os.path.join(OUT_DIR, 'report.md'), 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))

    raw = {
        'config': {
            'd': D_HIDDEN, 'n_blocks': N_BLOCKS, 'expansion': EXPANSION,
            'feat_dim': DIM_NUM*3 + DIM_SET + DIM_EMB*3 + 4,
        },
        'param_count': {
            'step': n_step, 'total': n_total,
            'theia_ref': THEIA_REF_PARAMS,
            'ratio': n_step / THEIA_REF_PARAMS,
            'acceptable_range': [PARAM_LOWER, PARAM_UPPER],
        },
        'sanity': {'acc': sanity_acc, 'epochs': SANITY_EPOCHS,
                   'gate': SANITY_GATE, 'wall_seconds': sanity_dt},
        'seeds': SEEDS,
        'seeds_completed': seeds_completed,
        'p1': p1_log, 'p2': p2_log, 'p3': p3_log,
        'elapsed': elapsed_log,
        'results': {str(k): v for k, v in results.items()},
        'early_abort': early_abort,
        'early_abort_reason': early_abort_reason,
        'stop_gate_500_step': STOP_GATE_500_STEP,
    }
    with open(os.path.join(OUT_DIR, 'raw.json'), 'w') as f:
        json.dump(raw, f, indent=2)

    print(f"\n{'='*72}")
    print(f"  Report:  {os.path.join(OUT_DIR, 'report.md')}")
    print(f"  Raw:     {os.path.join(OUT_DIR, 'raw.json')}")
    print(f"{'='*72}")


if __name__ == '__main__':
    main()
