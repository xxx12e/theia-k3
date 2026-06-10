"""
Backbone ablation for THEIA §4.5 sequential composition: replace the 4-engine
modular reasoning stack with a single flat MLP (~466K params vs THEIA step
~2.75M, ~6x smaller). Input encoders, the Phase 1/2/3 training pipeline, data
generation, and the 5 seeds {42, 123, 256, 777, 999} are imported verbatim
from theia_chain_v3_5seed (run from the repo root alongside that file).

Usage: python simple_mlp_backbone_ablation.py
Output: multi_seed_results/option_a_backbone_ablation/report.md
"""

import os
import sys
import time
import json

# Windows cmd defaults to GBK and crashes on the U+27F3 character that
# theia_chain_v3_5seed.phase1() prints during plateau restarts; force UTF-8
# stdout/stderr before importing anything that might print it.
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Everything training-related is imported from the paper's §4.5 script,
# guaranteeing zero divergence in data, training, and eval.
from theia_chain_v3_5seed import (
    DEVICE, DIM, NUM_RANGE, LOGIC_OPS, SEEDS,
    NumEncoder,           # THEIA's exact number encoder
    TransitionNet,        # THEIA's transition net
    gen_single_step_data, # Phase 1 data
    gen_chain_data,       # Phase 2/3 data + eval data
    phase1,               # Phase 1 trainer (with plateau restart)
    phase2,               # Phase 2 trainer (transition net teacher forcing)
    phase3,               # Phase 3 trainer (end-to-end Gumbel-ST)
    evaluate,             # Multi-step eval
)

# --- Simple MLP step computer ---
# Keep THEIA's input encoders byte-for-byte (so the ablation isolates the
# *reasoning* stack, not the *encoding* stack), then replace the 4-engine
# reasoning pipeline with a single flat MLP.
#
# Encoders kept (matching theia_chain_v3_5seed.py):
#   - NumEncoder for a, b, d  (Linear(1,128) -> GELU -> Linear(128,128))
#     each with its own learned `unk` sentinel parameter
#   - nn.Embedding(4, DIM) for op
#   - nn.Embedding(6, DIM) for rel
#   - nn.Embedding(LOGIC_OPS, DIM) for logic_op
#   - set_enc Linear(21, DIM) -> GELU -> Linear(DIM, DIM) with `unk` sentinel
#
# What is REMOVED (the actual ablation target):
#   - ArithEngine.fuse                 (3*DIM -> DIM cross-modal fusion)
#   - bridge_ao, bridge_as             (residual bridges)
#   - OrderEngine + SubspaceEngine     (G/L/E subspace decomposition)
#   - SetEngine + SubspaceEngine       (G/L/E subspace decomposition)
#   - LogicEngine SubspaceEngine       (combining order + set verdicts)
#   - OutHead prototype cosine head    (cosine to orthogonal F/T/U prototypes)
#
# What REPLACES it:
#   - Concat all encoded features -> 3-layer MLP (hidden 512) -> 3 logits
#
# Param count: ~466K (vs THEIA step ~2.75M, ~6x smaller).

DIM_NUM = 128   # match NumEncoder output
DIM_EMB = 128   # match THEIA's op/rel/logic_op embedding dim
DIM_SET = 128   # match THEIA's set_enc output
HIDDEN = 512
N_LAYERS = 3


class SetEncoder(nn.Module):
    """Mirror of theia_chain_v3_5seed.SetEngine's set_enc + unk sentinel."""
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


class SimpleMLPStepComputer(nn.Module):
    """
    Flat-MLP replacement for THEIA's 4-engine reasoning stack.

    Forward signature MUST match theia_chain_v3_5seed.THEIAStep so that
    phase1/phase2/phase3/evaluate (which call .forward_flat) work without
    modification.
    """
    def __init__(self):
        super().__init__()
        # ---- Input encoders (identical to THEIA) ----
        self.enc_a = NumEncoder()
        self.enc_b = NumEncoder()
        self.enc_d = NumEncoder()
        self.op_emb = nn.Embedding(4, DIM_EMB)
        self.rel_emb = nn.Embedding(6, DIM_EMB)
        self.logic_op_emb = nn.Embedding(LOGIC_OPS, DIM_EMB)
        self.set_encoder = SetEncoder()

        # ---- Flat reasoning MLP (the ablation) ----
        # Concat: ea(128) + eb(128) + ed(128) + e_set(128)
        #       + op_emb(128) + rel_emb(128) + logic_op_emb(128)
        #       + 4 unk binary flags = 7*128 + 4 = 900
        feat_dim = DIM_NUM * 3 + DIM_SET + DIM_EMB * 3 + 4
        layers = []
        in_dim = feat_dim
        for _ in range(N_LAYERS - 1):
            layers += [nn.Linear(in_dim, HIDDEN), nn.GELU(), nn.LayerNorm(HIDDEN)]
            in_dim = HIDDEN
        layers += [nn.Linear(HIDDEN, 3)]   # 3 logits: F=0, U=1, T=2
        self.mlp = nn.Sequential(*layers)

    def forward(self, a, b, op, d, rel, s, logic_op, au, bu, du, su):
        ea = self.enc_a(a, au)
        eb = self.enc_b(b, bu)
        ed = self.enc_d(d, du)
        es = self.set_encoder(s, su)
        eo = self.op_emb(op)
        er = self.rel_emb(rel)
        el = self.logic_op_emb(logic_op)
        unk_flags = torch.stack([au.float(), bu.float(), du.float(), su.float()], dim=-1)
        feat = torch.cat([ea, eb, ed, es, eo, er, el, unk_flags], dim=-1)
        return self.mlp(feat)

    def forward_flat(self, x):
        # Same flat layout as THEIAStep.forward_flat — required for phase1/3/eval
        return self(
            x[:, 0], x[:, 1], x[:, 2].long(), x[:, 3], x[:, 4].long(),
            x[:, 5:26], x[:, 26].long(),
            x[:, 27].bool(), x[:, 28].bool(), x[:, 29].bool(), x[:, 30].bool(),
        )


class SimpleMLPChain(nn.Module):
    """
    Chain wrapper that mimics THEIAChainV3's interface (.step, .transition,
    .gumbel_st) so that imported phase2/phase3/evaluate work unmodified.
    """
    def __init__(self):
        super().__init__()
        self.step = SimpleMLPStepComputer()
        self.transition = TransitionNet()

    def gumbel_st(self, logits, tau=0.5):
        soft = F.gumbel_softmax(logits, tau=tau, hard=False)
        hard = torch.zeros_like(soft).scatter_(-1, soft.argmax(-1, keepdim=True), 1.0)
        return hard - soft.detach() + soft


# --- Main (identical loop structure to theia_chain_v3_5seed.__main__) ---

def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def main():
    torch.backends.cudnn.benchmark = True

    OUT_DIR = "multi_seed_results/option_a_backbone_ablation"
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"\n{'='*64}")
    print(f"  Option A — SimpleMLP Backbone Ablation (5-seed)")
    print(f"  Seeds: {SEEDS}")
    print(f"  Device: {DEVICE}")
    print(f"{'='*64}")

    # Sanity: print param count of one fresh model
    _probe = SimpleMLPChain()
    n_step = count_params(_probe.step)
    n_total = count_params(_probe)
    print(f"  SimpleMLPStepComputer params: {n_step:,}")
    print(f"  SimpleMLPChain total params:  {n_total:,}")
    print(f"  (THEIA step ~2.75M, ratio ~{2_750_000 / n_step:.1f}x smaller)")
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

        model = SimpleMLPChain().to(DEVICE)

        t0 = time.time()
        p1 = phase1(model.step, train_seed=seed)
        p1_log.append(p1)

        # Early abort: if seed 0 fails Phase 1 below 95%, the ablation has
        # already answered negatively; skip the remaining seeds.
        if si == 0 and p1 < 0.95:
            elapsed = time.time() - t0
            elapsed_log.append(elapsed)
            print(f"\n  ⚠️  EARLY ABORT: seed 0 Phase 1 = {p1:.2%} < 95%.")
            print(f"  Conclusion: backbone IS necessary for local 4-domain task.")
            print(f"  Skipping remaining {len(SEEDS) - 1} seeds.")
            early_abort = True
            del model
            torch.cuda.empty_cache()
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
            tag = "✅" if acc > 0.99 else "🟡" if acc > 0.95 else "❌"
            print(f"  {steps:>4d}-step: {acc:.2%} (step1={s1:.2%}) {tag}")

        del model
        torch.cuda.empty_cache()

    # --- Summary + report ---
    print(f"\n{'='*64}")
    print(f"  Option A SUMMARY")
    print(f"{'='*64}")

    summary_lines = []
    summary_lines.append("# Option A — SimpleMLP Backbone Ablation Report\n")
    summary_lines.append(f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    summary_lines.append(f"**Seeds run**: {len(p1_log)} / {len(SEEDS)}")
    summary_lines.append(f"**Early abort**: {early_abort}\n")
    summary_lines.append("## Architecture\n")
    summary_lines.append(f"- SimpleMLPStepComputer params: **{n_step:,}**")
    summary_lines.append(f"- THEIA step params (reference): ~2,750,000")
    summary_lines.append(f"- Ratio: ~{2_750_000 / n_step:.1f}x smaller\n")
    summary_lines.append("**What was kept (identical to THEIA)**:")
    summary_lines.append("- NumEncoder for a, b, d (separate, with unk sentinels)")
    summary_lines.append("- nn.Embedding for op / rel / logic_op (DIM=128)")
    summary_lines.append("- set_enc Linear(21,128) -> GELU -> Linear(128,128) with unk sentinel")
    summary_lines.append("- Phase 1/2/3 training pipeline (imported verbatim)")
    summary_lines.append("- Data generation (imported verbatim)")
    summary_lines.append("- Class weights [1.0, 2.0, 1.0]\n")
    summary_lines.append("**What was ablated (replaced with flat MLP)**:")
    summary_lines.append("- ArithEngine.fuse (3*DIM cross-modal fusion)")
    summary_lines.append("- bridge_ao, bridge_as residual bridges")
    summary_lines.append("- OrderEngine + SubspaceEngine (G/L/E decomposition)")
    summary_lines.append("- SetEngine + SubspaceEngine (G/L/E decomposition)")
    summary_lines.append("- LogicEngine SubspaceEngine")
    summary_lines.append("- OutHead prototype cosine head\n")
    summary_lines.append("Replaced by: 3-layer MLP, hidden=512, GELU+LayerNorm, "
                         "linear head to 3 logits.\n")

    summary_lines.append("## Phase results per seed\n")
    summary_lines.append("| seed | P1 | P2 | P3 | time(s) |")
    summary_lines.append("|---|---|---|---|---|")
    for i, seed in enumerate(SEEDS[:len(p1_log)]):
        p1v = f"{p1_log[i]:.2%}" if i < len(p1_log) else "-"
        p2v = f"{p2_log[i]:.2%}" if i < len(p2_log) else "-"
        p3v = f"{p3_log[i]:.2%}" if i < len(p3_log) else "-"
        tv  = f"{elapsed_log[i]:.0f}" if i < len(elapsed_log) else "-"
        summary_lines.append(f"| {seed} | {p1v} | {p2v} | {p3v} | {tv} |")
    summary_lines.append("")

    if not early_abort and len(results[500]) == len(SEEDS):
        summary_lines.append("## Length generalization (mean ± std over 5 seeds)\n")
        summary_lines.append("| Steps | Mean | Std | Min | Max |")
        summary_lines.append("|---|---|---|---|---|")
        print(f"  {'Steps':>6s}  {'Mean':>8s}  {'Std':>8s}  {'Min':>8s}  {'Max':>8s}")
        for steps in TEST_STEPS:
            vals = results[steps]
            mean, std = float(np.mean(vals)), float(np.std(vals))
            mn, mx = float(np.min(vals)), float(np.max(vals))
            tag = "✅" if mean > 0.99 else "🟡" if mean > 0.95 else "❌"
            print(f"  {steps:>6d}  {mean:>7.2%}  {std:>7.2%}  {mn:>7.2%}  {mx:>7.2%}  {tag}")
            summary_lines.append(
                f"| {steps} | {mean:.2%} | {std:.2%} | {mn:.2%} | {mx:.2%} |"
            )
        summary_lines.append("")

        # Verdict
        mean500 = float(np.mean(results[500]))
        all_p1_ok = all(p >= 0.999 for p in p1_log)
        all_500_ok = all(v >= 0.99 for v in results[500])
        summary_lines.append("## Verdict\n")
        if all_p1_ok and all_500_ok:
            summary_lines.append("**POSITIVE**: All 5 seeds Phase 1 ≥99.9% AND 500-step ≥99%.")
            summary_lines.append("")
            summary_lines.append("Recommended Appendix C framing:")
            summary_lines.append("")
            summary_lines.append("> Replacing THEIA's full 2.75M-parameter four-engine pipeline "
                                 "with a single 3-layer flat MLP (~466K params, ~6x smaller) at "
                                 "the step computer position — while keeping the input encoders "
                                 "(NumEncoder, op/rel/logic_op embeddings, set_enc), the "
                                 "transition network, the Gumbel-ST discretization, and the "
                                 "three-phase training protocol of §4.5 unchanged — still yields "
                                 f"all 5 seeds at ≥99% accuracy on 500-step chains "
                                 f"(mean {mean500:.2%}). This demonstrates that THEIA's "
                                 "modular four-engine factorization is **not** the source of "
                                 "length generalization; the credit-bearing mechanism is the "
                                 "Gumbel-ST + discretized-state composition rule of §4.5. The "
                                 "modular factorization is justified separately by §4.4 "
                                 "interpretability probing, on different evidence.")
        elif all_p1_ok:
            summary_lines.append("**PARTIAL**: All seeds Phase 1 ≥99.9% but 500-step <99% on "
                                 "some seeds. Backbone factorization affects length generalization "
                                 "*quality* but not its existence.")
        else:
            summary_lines.append("**NEGATIVE**: Some seeds failed Phase 1 ≥99.9%. The simple "
                                 "MLP cannot reliably learn the 4-domain Kleene local task. "
                                 "Backbone factorization is necessary. Update §4.6 limitations.")
    elif early_abort:
        summary_lines.append("## Verdict\n")
        summary_lines.append("**NEGATIVE (early abort)**: Seed 0 Phase 1 < 95%. The simple "
                             "MLP cannot learn the 4-domain Kleene local task even after "
                             "plateau-restart. Backbone factorization is necessary for the "
                             "local step computer. Update §4.6 limitations to reflect this "
                             "honest finding.")

    report_path = os.path.join(OUT_DIR, "report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

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

    print(f"\n  Report written to: {report_path}")
    print(f"{'='*64}\n")


if __name__ == "__main__":
    main()
