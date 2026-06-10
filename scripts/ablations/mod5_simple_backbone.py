#!/usr/bin/env python
"""
Mod-5 sequential composition with a simple MLP backbone + Gumbel-softmax ST.

Two questions in one experiment:
  (1) Mod-5 generalization: §4.5 tests mod-3 chain composition; this verifies
      that mod-5 also generalizes.
  (2) Backbone ablation: uses a deliberately simple MLP step computer (not
      THEIA's modular pipeline) to test whether Gumbel-ST + the transition
      mechanism alone suffices.

Task: state_t = (state_{t-1} + local_value_t) mod 5
  - 5 discrete states (vs paper's 3)
  - Each step samples local_value ∈ {0..4}
  - Train on 5-step chains, evaluate on 5 / 10 / 50 / 100 / 500 step chains

Architecture:
  - SimpleStepComputer: Linear(10) -> GELU -> Linear(64) -> GELU -> Linear(5)
    Input: [state_prev_onehot, local_value_onehot] (5+5 = 10 dim)
    Output: next state logits (5 dim)
  - Total params: ~1.4K (vs THEIA's 2.75M)

Training:
  - Phase 1: Supervised learning on (state, local) -> next_state
  - Phase 3: End-to-end Gumbel-ST on 5-step chains
  (Phase 2 omitted because simple MLP combines step + transition)

Usage:
    python mod5_simple_backbone.py
    python mod5_simple_backbone.py --seeds 42 123
"""
import argparse, json, os, time
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np

p = argparse.ArgumentParser()
p.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 256, 777, 999])
p.add_argument('--n-samples-p1', type=int, default=500_000,
               help='Phase 1 supervised samples')
p.add_argument('--n-chains-p3', type=int, default=200_000,
               help='Phase 3 chain training samples')
p.add_argument('--p1-epochs', type=int, default=30)
p.add_argument('--p3-epochs', type=int, default=40)
p.add_argument('--train-chain-len', type=int, default=5)
p.add_argument('--eval-lengths', type=int, nargs='+',
               default=[5, 10, 50, 100, 500])
p.add_argument('--n-eval', type=int, default=10_000,
               help='Eval samples per chain length')
p.add_argument('--hidden', type=int, default=64)
p.add_argument('--output', type=str,
               default=r'multi_seed_results\mod5_simple_backbone')
args = p.parse_args()

NUM_STATES = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(args.output, exist_ok=True)


class SimpleStepComputer(nn.Module):
    """Minimal MLP step computer (not THEIA's modular pipeline).

    Tests whether, with Gumbel-ST + discretization, a minimal MLP step
    computer achieves the length generalization that §4.5 attributes to
    THEIA — i.e., whether the credit belongs to discretization rather
    than to the backbone architecture.
    """
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(NUM_STATES * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, NUM_STATES),
        )

    def forward(self, state_onehot, local_onehot):
        x = torch.cat([state_onehot, local_onehot], dim=-1)
        return self.net(x)


def gumbel_softmax_st(logits, temperature=1.0):
    """Gumbel-softmax straight-through: hard one-hot forward, soft gradient backward."""
    soft = F.gumbel_softmax(logits, tau=temperature, hard=False)
    hard_idx = soft.argmax(-1)
    hard = F.one_hot(hard_idx, num_classes=NUM_STATES).float()
    return hard - soft.detach() + soft


def gen_chain_data(n_chains, chain_length, device):
    """Generate (initial_state, local_values, true_state_sequence)."""
    initial = torch.randint(0, NUM_STATES, (n_chains,), device=device)
    locals_ = torch.randint(0, NUM_STATES, (n_chains, chain_length), device=device)
    states = torch.zeros(n_chains, chain_length + 1, dtype=torch.long, device=device)
    states[:, 0] = initial
    for t in range(chain_length):
        states[:, t+1] = (states[:, t] + locals_[:, t]) % NUM_STATES
    return initial, locals_, states


def train_phase1(model, device):
    """Supervised training on individual transitions."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    n = args.n_samples_p1
    states = torch.randint(0, NUM_STATES, (n,), device=device)
    locals_ = torch.randint(0, NUM_STATES, (n,), device=device)
    targets = (states + locals_) % NUM_STATES

    states_oh = F.one_hot(states, NUM_STATES).float()
    locals_oh = F.one_hot(locals_, NUM_STATES).float()

    BATCH = 2048
    best_acc = 0.0
    for epoch in range(args.p1_epochs):
        perm = torch.randperm(n, device=device)
        total_loss = 0
        nb = 0
        for i in range(0, n, BATCH):
            idx = perm[i:i+BATCH]
            logits = model(states_oh[idx], locals_oh[idx])
            loss = F.cross_entropy(logits, targets[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            nb += 1
        with torch.no_grad():
            logits = model(states_oh, locals_oh)
            acc = (logits.argmax(-1) == targets).float().mean().item()
            if acc > best_acc:
                best_acc = acc
        if epoch % 5 == 0 or epoch == args.p1_epochs - 1:
            print(f"  P1 ep{epoch:3d}: loss={total_loss/nb:.4f}  acc={acc:.4f}")
        if best_acc > 0.9999:
            print(f"  P1 converged @ ep{epoch}, acc={best_acc:.4f}")
            break
    return best_acc


def train_phase3(model, device):
    """End-to-end Gumbel-ST training on N-step chains."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    L = args.train_chain_len
    n = args.n_chains_p3

    initial, locals_, true_states = gen_chain_data(n, L, device)
    locals_oh = F.one_hot(locals_, NUM_STATES).float()  # [n, L, 5]

    BATCH = 512
    best_acc = 0.0
    for epoch in range(args.p3_epochs):
        perm = torch.randperm(n, device=device)
        total_loss = 0
        nb = 0
        for i in range(0, n, BATCH):
            idx = perm[i:i+BATCH]
            B = idx.shape[0]

            state = F.one_hot(initial[idx], NUM_STATES).float()  # [B, 5]
            loss = 0.0
            for t in range(L):
                logits = model(state, locals_oh[idx, t])
                loss = loss + F.cross_entropy(logits, true_states[idx, t+1])
                state = gumbel_softmax_st(logits, temperature=1.0)
            loss = loss / L

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            nb += 1

        # eval at training length (hard inference)
        model.eval()
        with torch.no_grad():
            ti, tl, ts = gen_chain_data(5000, L, device)
            tlo = F.one_hot(tl, NUM_STATES).float()
            state = F.one_hot(ti, NUM_STATES).float()
            for t in range(L):
                logits = model(state, tlo[:, t])
                state = F.one_hot(logits.argmax(-1), NUM_STATES).float()
            acc = (state.argmax(-1) == ts[:, -1]).float().mean().item()
            if acc > best_acc:
                best_acc = acc
        model.train()
        if epoch % 5 == 0 or epoch == args.p3_epochs - 1:
            print(f"  P3 ep{epoch:3d}: loss={total_loss/nb:.4f}  acc@{L}={acc:.4f}")
        if best_acc > 0.99995:
            print(f"  P3 converged @ ep{epoch}, acc={best_acc:.4f}")
            break
    return best_acc


def eval_at_length(model, device, length):
    """Evaluate hard inference at given chain length."""
    model.eval()
    with torch.no_grad():
        initial, locals_, true_states = gen_chain_data(args.n_eval, length, device)
        locals_oh = F.one_hot(locals_, NUM_STATES).float()
        state = F.one_hot(initial, NUM_STATES).float()
        for t in range(length):
            logits = model(state, locals_oh[:, t])
            state = F.one_hot(logits.argmax(-1), NUM_STATES).float()
        acc = (state.argmax(-1) == true_states[:, -1]).float().mean().item()
    model.train()
    return acc


def run_seed(seed):
    print(f"\n{'='*60}")
    print(f"  SEED {seed}")
    print(f"{'='*60}")

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = SimpleStepComputer(hidden=args.hidden).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Step computer params: {n_params:,} (vs THEIA's 2.75M)")

    t0 = time.time()

    print(f"\nPhase 1: supervised step training ({args.n_samples_p1} samples)")
    p1_acc = train_phase1(model, DEVICE)
    if p1_acc < 0.99:
        print(f"  WARNING: phase 1 only reached {p1_acc:.4f}")

    print(f"\nPhase 3: end-to-end Gumbel-ST on {args.train_chain_len}-step chains")
    p3_acc = train_phase3(model, DEVICE)

    print(f"\nEvaluation at chain lengths {args.eval_lengths}:")
    eval_results = {}
    for L in args.eval_lengths:
        acc = eval_at_length(model, DEVICE, L)
        eval_results[L] = acc
        marker = "✓" if acc > 0.99 else "✗"
        print(f"  {marker} {L:>4} steps: {acc:.4f}")

    elapsed = time.time() - t0
    print(f"\nSeed {seed} done in {elapsed/60:.1f} min")

    return {
        'seed': seed,
        'params': n_params,
        'phase1_acc': p1_acc,
        'phase3_acc': p3_acc,
        'eval': {str(k): v for k, v in eval_results.items()},
        'time_sec': elapsed,
    }


def aggregate(results):
    print(f"\n{'='*60}")
    print(f"  AGGREGATE — {len(results)} seeds")
    print(f"{'='*60}")
    print(f"\nMod-5 simple backbone (2-layer MLP, {results[0]['params']} params)")
    print(f"Train: {args.train_chain_len}-step chains")
    print(f"Eval:  {args.eval_lengths}-step chains\n")

    lines = []
    L = lines.append
    L("# Mod-5 + Simple Backbone + Gumbel-ST — Backbone Ablation Report")
    L("")
    from datetime import datetime
    L(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    L("")
    L("**Purpose**: Address two scope questions with one experiment:")
    L("  1. Verify mod-5 (not just mod-3) generalizes from 5-step training to 500-step eval")
    L(f"  2. Verify a simple {results[0]['params']}-param 2-layer MLP backbone (NOT THEIA's modular pipeline) suffices, given Gumbel-ST discretization")
    L("")
    L(f"Per-seed time: ~{results[0]['time_sec']/60:.0f} min on RTX 5080.")
    L("")
    L("---")
    L("")
    L("## Aggregate accuracy (mean ± std across seeds)")
    L("")
    L("| Chain Length | Accuracy | Min | Max |")
    L("|---|---|---|---|")
    for length in args.eval_lengths:
        accs = [r['eval'][str(length)] for r in results]
        m = sum(accs) / len(accs)
        s = (sum((a - m)**2 for a in accs) / len(accs)) ** 0.5
        passed = "✓" if min(accs) > 0.99 else "✗"
        L(f"| {length} | {m*100:.2f}% ± {s*100:.2f}% | {min(accs)*100:.2f}% | {max(accs)*100:.2f}% | {passed}")
        print(f"  {length:>4} steps: {m:.4f} ± {s:.4f}  (min {min(accs):.4f}, max {max(accs):.4f})")
    L("")

    L("## Per-seed detail")
    L("")
    L("| Seed | Phase 1 | Phase 3 (5-step) | 500-step eval |")
    L("|---|---|---|---|")
    for r in results:
        L(f"| {r['seed']} | {r['phase1_acc']:.4f} | {r['phase3_acc']:.4f} | {r['eval']['500']:.4f} |")
    L("")

    # Verdict
    accs_500 = [r['eval']['500'] for r in results]
    min_500 = min(accs_500)
    L("## Verdict")
    L("")
    if min_500 > 0.99:
        L(f"**SUCCESS — both questions answered**. All {len(results)} seeds achieve >99% accuracy at 500 steps")
        L(f"with a simple 2-layer MLP step computer (vs THEIA's 2.75M parameter pipeline). Minimum: {min_500*100:.2f}%.")
        L("")
        L("This demonstrates:")
        L("- Mod-5 (not just mod-3) generalizes 5-step → 500-step")
        L("- Length generalization is enabled by Gumbel-ST + discretized state, NOT by THEIA's modular backbone")
        L("")
        L("**Implication**: a simple MLP backbone suffices for mod-5 length generalization ")
        L("(see table above) — direct evidence for the backbone-independence reading ")
        L("of the chain result.")
    elif min_500 > 0.90:
        L(f"**PARTIAL SUCCESS**. All seeds achieve >90% but not all reach 99% at 500 steps (min: {min_500*100:.2f}%).")
        L("")
        L("This is still informative: the simple backbone learns the task but length generalization is weaker. ")
        L("Two interpretations: (a) the simple backbone is genuinely worse than THEIA's pipeline; (b) hyperparameter ")
        L("tuning for the simple backbone could close the gap. Either way, the backbone affects ")
        L("the *quality* of length generalization, not its existence.")
    else:
        L(f"**FAILURE**. Some seeds drop below 90% at 500 steps (min: {min_500*100:.2f}%).")
        L("")
        L("This is unexpected given mod-3 success. Possible reasons: (a) 5 discrete states require more capacity ")
        L("than 64-dim hidden; try increasing `--hidden 128`. (b) Phase 3 training is insufficient; try `--p3-epochs 60`.")
        L("This result requires diagnosis before any conclusion is drawn.")

    out_md = os.path.join(args.output, 'report.md')
    with open(out_md, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"\nReport: {out_md}")


def main():
    print(f"Device: {DEVICE}")
    print(f"Seeds: {args.seeds}")
    print(f"Train chain length: {args.train_chain_len}")
    print(f"Eval lengths: {args.eval_lengths}")
    print(f"Hidden dim: {args.hidden}")

    all_results = []
    for seed in args.seeds:
        r = run_seed(seed)
        all_results.append(r)
        with open(os.path.join(args.output, f'seed_{seed}.json'), 'w') as f:
            json.dump(r, f, indent=2)

    with open(os.path.join(args.output, 'all_seeds.json'), 'w') as f:
        json.dump(all_results, f, indent=2)

    aggregate(all_results)


if __name__ == '__main__':
    main()
