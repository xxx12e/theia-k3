"""
Experiment 2: Transformer step computer on mod-3 chain task.

Replaces THEIA's 4-engine step computer with the 8L8H Transformer
from tf8l_tuned_seed42.py, while keeping TransitionNet + Gumbel-ST +
3-phase training identical.

Question: Does the Transformer's monolithic self-attention also support
length generalization when equipped with the same discretization
bottleneck, or does it collapse?

Architecture:
  - TF8L tokenizer (8 tokens: a, b, op, d, rel, set, logic_op, unk_flags)
  - 8-layer Transformer encoder (d=192, 8 heads, pre-LN, ~3.5M params)
  - Linear head -> 3 logits (F/U/T)
  - Same TransitionNet, Gumbel-ST, phase1/2/3, data gen as THEIA chain

Phase 3 uses THEIA's lr (1e-3) first. If collapse, a second run at
Transformer-tuned lr (1e-4) is attempted.
"""

import os, sys, time, json

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from theia_chain_v3_5seed import (
    DEVICE, DIM, NUM_RANGE, LOGIC_OPS, SEEDS,
    TransitionNet,
    gen_single_step_data, gen_chain_data,
    phase1, phase2, phase3, evaluate,
)

# --- Transformer step computer (matches tf8l_tuned_seed42.py arch) ---
D_MODEL = 192; N_HEADS = 8; N_LAYERS = 8; DIM_FF = 768
NUM_VAL = NUM_RANGE + 2  # 0..20 + unk sentinel


class TransformerStepComputer(nn.Module):
    """8L8H Transformer operating on the same 31-dim flat input as THEIAStep."""
    def __init__(self):
        super().__init__()
        self.emb_a = nn.Embedding(NUM_VAL, D_MODEL)
        self.emb_b = nn.Embedding(NUM_VAL, D_MODEL)
        self.emb_d = nn.Embedding(NUM_VAL, D_MODEL)
        self.emb_op = nn.Embedding(4, D_MODEL)
        self.emb_rel = nn.Embedding(6, D_MODEL)
        self.emb_logic_op = nn.Embedding(LOGIC_OPS, D_MODEL)
        self.set_proj = nn.Linear(21, D_MODEL)
        self.unk_proj = nn.Linear(4, D_MODEL)
        self.pos_emb = nn.Parameter(torch.randn(1, 8, D_MODEL) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=N_HEADS, dim_feedforward=DIM_FF,
            dropout=0.0, batch_first=True, activation='gelu', norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=N_LAYERS)
        self.final_norm = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, 3)

    def _tokenize(self, x):
        """x: (B, 31) flat layout from gen_single_step_data."""
        a = x[:, 0].long().clamp(0, NUM_RANGE)
        b = x[:, 1].long().clamp(0, NUM_RANGE)
        d = x[:, 3].long().clamp(0, NUM_RANGE)
        op = x[:, 2].long()
        rel = x[:, 4].long()
        s = x[:, 5:26].float()
        logic_op = x[:, 26].long()
        unk = x[:, 27:31].float()
        UNK_IDX = NUM_RANGE + 1
        a = torch.where(x[:, 27].bool(), torch.full_like(a, UNK_IDX), a)
        b = torch.where(x[:, 28].bool(), torch.full_like(b, UNK_IDX), b)
        d = torch.where(x[:, 29].bool(), torch.full_like(d, UNK_IDX), d)
        tokens = torch.stack([
            self.emb_a(a), self.emb_b(b), self.emb_op(op), self.emb_d(d),
            self.emb_rel(rel), self.set_proj(s), self.emb_logic_op(logic_op),
            self.unk_proj(unk),
        ], dim=1)
        return tokens + self.pos_emb

    def forward_flat(self, x):
        """Required by phase1/phase3/evaluate."""
        h = self.encoder(self._tokenize(x))
        h = self.final_norm(h.mean(dim=1))
        return self.head(h)


class TransformerChain(nn.Module):
    def __init__(self):
        super().__init__()
        self.step = TransformerStepComputer()
        self.transition = TransitionNet()

    def gumbel_st(self, logits, tau=0.5):
        soft = F.gumbel_softmax(logits, tau=tau, hard=False)
        hard = torch.zeros_like(soft).scatter_(-1, soft.argmax(-1, keepdim=True), 1.0)
        return hard - soft.detach() + soft


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def main():
    torch.backends.cudnn.benchmark = True
    OUT_DIR = "multi_seed_results/transformer_chain_ablation"
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"\n{'='*64}")
    print(f"  Experiment 2: Transformer Step Computer on Mod-3 Chain")
    print(f"  Seeds: {SEEDS}")
    print(f"  Device: {DEVICE}")
    print(f"{'='*64}")

    _probe = TransformerChain()
    n_step = count_params(_probe.step)
    n_total = count_params(_probe)
    print(f"  TransformerStepComputer params: {n_step:,}")
    print(f"  TransformerChain total params:  {n_total:,}")
    print(f"  THEIA step params:              2,751,232")
    del _probe

    TEST_STEPS = [5, 10, 50, 100, 500]
    results = {s: [] for s in TEST_STEPS}
    p1_log, p2_log, p3_log = [], [], []
    elapsed_log = []
    early_abort = False

    for si, seed in enumerate(SEEDS):
        print(f"\n{'='*64}")
        print(f"  Seed {si+1}/{len(SEEDS)}: {seed}")
        print(f"{'='*64}")

        torch.manual_seed(seed)
        np.random.seed(seed)

        model = TransformerChain().to(DEVICE)
        t0 = time.time()
        p1 = phase1(model.step, train_seed=seed)
        p1_log.append(p1)

        if si == 0 and p1 < 0.95:
            elapsed = time.time() - t0
            elapsed_log.append(elapsed)
            print(f"\n  EARLY ABORT: seed 0 Phase 1 = {p1:.2%} < 95%.")
            early_abort = True
            del model; torch.cuda.empty_cache()
            break

        p2 = phase2(model, train_seed=seed)
        p2_log.append(p2)
        p3 = phase3(model, train_seed=seed)
        p3_log.append(p3)

        elapsed = time.time() - t0
        elapsed_log.append(elapsed)
        print(f"  Training: {elapsed:.0f}s | P1={p1:.2%} P2={p2:.2%} P3={p3:.2%}")

        for steps in TEST_STEPS:
            acc, s1, cls = evaluate(model, steps, seed=seed + 5000, N=10000)
            results[steps].append(acc)
            tag = "PASS" if acc > 0.99 else "WARN" if acc > 0.95 else "FAIL"
            print(f"  {steps:>4d}-step: {acc:.2%} (step1={s1:.2%}) {tag}")

        del model; torch.cuda.empty_cache()

    # --- Report ---
    lines = []
    lines.append("# Experiment 2: Transformer Step Computer on Mod-3 Chain\n")
    lines.append(f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Seeds run**: {len(p1_log)} / {len(SEEDS)}")
    lines.append(f"**Early abort**: {early_abort}\n")
    lines.append(f"## Param count\n")
    lines.append(f"- TransformerStepComputer: **{n_step:,}**")
    lines.append(f"- THEIA step: **2,751,232**\n")

    lines.append("## Phase results per seed\n")
    lines.append("| seed | P1 | P2 | P3 | time(s) |")
    lines.append("|---|---|---|---|---|")
    for i, seed in enumerate(SEEDS[:len(p1_log)]):
        p1v = f"{p1_log[i]:.2%}" if i < len(p1_log) else "-"
        p2v = f"{p2_log[i]:.2%}" if i < len(p2_log) else "-"
        p3v = f"{p3_log[i]:.2%}" if i < len(p3_log) else "-"
        tv = f"{elapsed_log[i]:.0f}" if i < len(elapsed_log) else "-"
        lines.append(f"| {seed} | {p1v} | {p2v} | {p3v} | {tv} |")

    if not early_abort and len(results[500]) == len(SEEDS):
        lines.append("\n## Length generalization (mean +/- std over 5 seeds)\n")
        lines.append("| Steps | Mean | Std | Min | Max |")
        lines.append("|---|---|---|---|---|")
        for steps in TEST_STEPS:
            vals = results[steps]
            mean, std = float(np.mean(vals)), float(np.std(vals))
            mn, mx = float(np.min(vals)), float(np.max(vals))
            lines.append(f"| {steps} | {mean:.2%} | {std:.2%} | {mn:.2%} | {mx:.2%} |")

        mean500 = float(np.mean(results[500]))
        lines.append("\n## Verdict\n")
        all_p1_ok = all(p >= 0.999 for p in p1_log)
        all_500_ok = all(v >= 0.99 for v in results[500])
        if all_p1_ok and all_500_ok:
            lines.append("**Transformer ALSO generalizes**: Same Gumbel-ST mechanism works. "
                         "Step computer architecture is NOT the differentiator for length gen.")
        elif all_p1_ok:
            lines.append("**Transformer FAILS on long chains**: P1 OK but 500-step collapsed. "
                         "This means: (a) Gumbel-ST alone is insufficient, or "
                         "(b) Transformer's representation is incompatible with discrete state "
                         "transition. Either way, modularity matters for compositionality.")
        else:
            lines.append("**Transformer P1 FAILED**: Cannot learn local Kleene task in chain "
                         "context with phase1 training recipe.")

    report_path = os.path.join(OUT_DIR, "report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    raw = {
        "seeds": SEEDS,
        "p1": p1_log, "p2": p2_log, "p3": p3_log,
        "elapsed": elapsed_log,
        "results": {str(k): v for k, v in results.items()},
        "early_abort": early_abort,
        "n_params_step": n_step,
    }
    with open(os.path.join(OUT_DIR, "raw.json"), "w") as f:
        json.dump(raw, f, indent=2)

    print(f"\n  Report: {report_path}")


if __name__ == "__main__":
    main()
