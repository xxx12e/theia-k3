"""
Parameter-matched flat-MLP backbone ablation (backbone-ablation appendix).

Identical to simple_mlp_backbone_ablation.py except HIDDEN=1243 instead of 512,
giving total params ~2,751,806 (vs THEIA 2,751,232, diff < 0.03%). Rules out
the capacity confound: if this also collapses on long chains, the modularity
claim is not about parameter count.
"""

import os, sys, time, json

# Windows GBK fix
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
    NumEncoder, TransitionNet,
    gen_single_step_data, gen_chain_data,
    phase1, phase2, phase3, evaluate,
)

# --- Parameter-matched MLP step computer (HIDDEN=1243 -> ~2.75M) ---
DIM_NUM = 128; DIM_EMB = 128; DIM_SET = 128
HIDDEN = 1243   # Chosen to match THEIA's 2,751,232 params
N_LAYERS = 3


class SetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.set_enc = nn.Sequential(
            nn.Linear(21, DIM_SET), nn.GELU(), nn.Linear(DIM_SET, DIM_SET)
        )
        self.unk = nn.Parameter(torch.randn(DIM_SET) * 0.02)

    def forward(self, s, su):
        sv = self.set_enc(s)
        mask = su.unsqueeze(-1).float()
        return sv * (1 - mask) + self.unk * mask


class MatchedMLPStepComputer(nn.Module):
    """Flat-MLP with HIDDEN=1243 to match THEIA's 2.75M param count."""
    def __init__(self):
        super().__init__()
        self.enc_a = NumEncoder()
        self.enc_b = NumEncoder()
        self.enc_d = NumEncoder()
        self.op_emb = nn.Embedding(4, DIM_EMB)
        self.rel_emb = nn.Embedding(6, DIM_EMB)
        self.logic_op_emb = nn.Embedding(LOGIC_OPS, DIM_EMB)
        self.set_encoder = SetEncoder()

        feat_dim = DIM_NUM * 3 + DIM_SET + DIM_EMB * 3 + 4  # = 900
        layers = []
        in_dim = feat_dim
        for _ in range(N_LAYERS - 1):
            layers += [nn.Linear(in_dim, HIDDEN), nn.GELU(), nn.LayerNorm(HIDDEN)]
            in_dim = HIDDEN
        layers += [nn.Linear(HIDDEN, 3)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, a, b, op, d, rel, s, logic_op, au, bu, du, su):
        ea = self.enc_a(a, au); eb = self.enc_b(b, bu); ed = self.enc_d(d, du)
        es = self.set_encoder(s, su)
        eo = self.op_emb(op); er = self.rel_emb(rel); el = self.logic_op_emb(logic_op)
        unk_flags = torch.stack([au.float(), bu.float(), du.float(), su.float()], dim=-1)
        feat = torch.cat([ea, eb, ed, es, eo, er, el, unk_flags], dim=-1)
        return self.mlp(feat)

    def forward_flat(self, x):
        return self(
            x[:, 0], x[:, 1], x[:, 2].long(), x[:, 3], x[:, 4].long(),
            x[:, 5:26], x[:, 26].long(),
            x[:, 27].bool(), x[:, 28].bool(), x[:, 29].bool(), x[:, 30].bool(),
        )


class MatchedMLPChain(nn.Module):
    def __init__(self):
        super().__init__()
        self.step = MatchedMLPStepComputer()
        self.transition = TransitionNet()

    def gumbel_st(self, logits, tau=0.5):
        soft = F.gumbel_softmax(logits, tau=tau, hard=False)
        hard = torch.zeros_like(soft).scatter_(-1, soft.argmax(-1, keepdim=True), 1.0)
        return hard - soft.detach() + soft


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def main():
    torch.backends.cudnn.benchmark = True
    OUT_DIR = "multi_seed_results/matched_mlp_backbone_ablation"
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"\n{'='*64}")
    print(f"  Experiment 1: Parameter-Matched MLP Backbone (HIDDEN={HIDDEN})")
    print(f"  Seeds: {SEEDS}")
    print(f"  Device: {DEVICE}")
    print(f"{'='*64}")

    _probe = MatchedMLPChain()
    n_step = count_params(_probe.step)
    n_total = count_params(_probe)
    print(f"  MatchedMLPStepComputer params: {n_step:,}")
    print(f"  MatchedMLPChain total params:  {n_total:,}")
    print(f"  THEIA step params:             2,751,232")
    print(f"  Param ratio:                   {n_step / 2_751_232:.4f}x")
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

        model = MatchedMLPChain().to(DEVICE)
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
    lines.append("# Experiment 1: Parameter-Matched MLP Backbone Ablation\n")
    lines.append(f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**HIDDEN**: {HIDDEN}")
    lines.append(f"**Seeds run**: {len(p1_log)} / {len(SEEDS)}")
    lines.append(f"**Early abort**: {early_abort}\n")
    lines.append(f"## Param count\n")
    lines.append(f"- MatchedMLPStepComputer: **{n_step:,}**")
    lines.append(f"- THEIA step: **2,751,232**")
    lines.append(f"- Ratio: **{n_step / 2_751_232:.4f}x**\n")
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
        all_p1_ok = all(p >= 0.999 for p in p1_log)
        all_500_ok = all(v >= 0.99 for v in results[500])
        lines.append("\n## Verdict\n")
        if all_p1_ok and all_500_ok:
            lines.append("**POSITIVE**: Parameter matching did NOT rescue flat MLP.")
            lines.append("Capacity confound eliminated. Modularity claim strengthened.")
        elif all_p1_ok:
            lines.append("**PARTIAL**: P1 OK but 500-step collapsed. Same pattern as 0.8M version.")
        else:
            lines.append("**NEGATIVE**: P1 failed. Flat MLP cannot learn local task even at 2.75M.")

    report_path = os.path.join(OUT_DIR, "report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    raw = {
        "seeds": SEEDS, "hidden": HIDDEN,
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
