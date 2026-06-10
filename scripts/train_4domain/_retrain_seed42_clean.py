"""
Seed-42 clean retrain for the THEIA Table 1 wall-clock (paper §4.2).

The paper's seed-42 wall-clock came from a run under the earlier pre-fix Kleene
harness; its 12/12 was obtained by re-evaluating the trained checkpoint under the
doubly-fixed harness, so that wall-clock is not a Kleene-aware early-stop time.
This script retrains seed 42 from scratch under the doubly-fixed harness, with
architecture, data gen, optimizer, scheduler, class weights, grad clip, and
determinism flags identical to the paper runs for seeds {123, 256, 777, 999},
and the same early-stop protocol: overall eval every 10 epochs; Kleene every
20 epochs once best_acc > 0.99; converge on 12/12 AND best_acc > 0.999 on two
consecutive checks. (This is the paper-run cadence, predating the 2026-04-20
5-ep unification in theia_5seed_v2.py.) The resulting wall-clock is directly
comparable to the other four seeds. Tracks first_12_12_{epoch,wall_clock_min}
(first check where 12/12 + >99.9% held) and converge_{epoch,wall_clock_min}
(2-consecutive streak satisfied — the "converge" definition the other seeds use).

Outputs (cwd): seed42_retrain_clean.json, seed42_retrain_checkpoint.pth,
seed42_retrain_log.txt
"""
import json
import os
import sys
import time

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from _theia_model_def import (
    IsisV9, build_dataset, run_kleene,
    DEVICE, BATCH,
)

SEED = 42
MAX_EPOCHS = 200
EARLY_STOP_OVERALL = 0.999
EARLY_STOP_PER_RULE = 99.0  # percent, matching run_kleene's output units
CONSECUTIVE_REQUIRED = 2

OUT_JSON = 'seed42_retrain_clean.json'
OUT_CKPT = 'seed42_retrain_checkpoint.pth'
OUT_LOG  = 'seed42_retrain_log.txt'

log_f = open(OUT_LOG, 'w', encoding='utf-8')
def log(msg):
    log_f.write(msg + '\n')
    log_f.flush()

# ---- determinism setup (identical to theia_5seed_v2.py) ----
torch.manual_seed(SEED); np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

print(f"THEIA v2 clean retrain | seed={SEED} | device={DEVICE}")
log(f"THEIA v2 clean retrain | seed={SEED}")

# ---- build dataset (excluded from wall-clock, as in the original runs) ----
AF, BF, DF, SB, S_UNK, A_UNK, B_UNK, D_UNK, AR, RL, OP, TARGET, N = build_dataset(SEED)
split = int(N * 0.8)

# ---- re-seed before model init (identical to theia_5seed_v2.py) ----
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

model = IsisV9().to(DEVICE)
params = sum(p.numel() for p in model.parameters())
print(f"params: {params:,}")
log(f"params: {params:,}")
assert params == 2_751_232, f"IsisV9 param count mismatch: got {params}"

# ---- optimizer + scheduler (identical to theia_5seed_v2.py) ----
class_weights = torch.tensor([1.0, 1.0, 2.0], device=DEVICE)
opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS, eta_min=1e-5)
scaler = torch.amp.GradScaler('cuda')


def get_batch(idx):
    return (AF[idx], BF[idx], DF[idx], SB[idx], S_UNK[idx],
            A_UNK[idx], B_UNK[idx], D_UNK[idx],
            AR[idx], RL[idx], OP[idx])


def eval_overall():
    """Overall val accuracy on the held-out 20%."""
    model.eval()
    with torch.no_grad():
        tidx = torch.arange(split, N, device=DEVICE)
        preds = []
        for j in range(0, len(tidx), BATCH * 4):
            bi = tidx[j:j + BATCH * 4]
            with torch.amp.autocast('cuda'):
                preds.append(model.classify(model(*get_batch(bi))))
        return (torch.cat(preds) == TARGET[split:]).float().mean().item()


# ---- training loop ----
best_acc = 0.0
converge_epoch = MAX_EPOCHS
converge_wall_clock_min = None
kleene_streak = 0
last_kleene_passed = 0
final_kleene_results = {}

first_12_12_epoch = None
first_12_12_wall_clock_min = None
first_12_12_kleene = None

t_start = time.time()

pbar = tqdm(range(1, MAX_EPOCHS + 1),
            desc=f'THEIA v2 clean s{SEED}', ncols=110)
for epoch in pbar:
    model.train()
    perm = torch.randperm(split, device=DEVICE)
    tl = 0.0
    nb = 0
    for i in range(0, split, BATCH):
        idx = perm[i:i + BATCH]
        with torch.amp.autocast('cuda'):
            out = model(*get_batch(idx))
            logits = out @ model.sv.weight.T
            loss = F.cross_entropy(logits, TARGET[idx], weight=class_weights)
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        tl += loss.item()
        nb += 1
    sched.step()
    avg_loss = tl / nb

    # ---- overall eval every 10 epochs (paper Table 1 cadence) ----
    if epoch % 10 == 0:
        acc = eval_overall()
        if acc > best_acc:
            best_acc = acc
        log(f"epoch {epoch:3d} loss={avg_loss:.4f} acc={acc:.4f} "
            f"best={best_acc:.4f}")

    # ---- Kleene check every 20 epochs after best_acc > 0.99
    #      (paper Table 1 cadence) ----
    if epoch % 20 == 0 and best_acc > 0.99:
        kleene_results, kleene_passed = run_kleene(model)
        last_kleene_passed = kleene_passed
        final_kleene_results = kleene_results
        wall_clock_min = (time.time() - t_start) / 60.0
        log(f"kleene@{epoch}: {kleene_passed}/12  wall={wall_clock_min:.2f}m")

        # Track first hit of 12/12 + >99.9%
        if (kleene_passed == 12 and best_acc > EARLY_STOP_OVERALL
                and first_12_12_epoch is None):
            first_12_12_epoch = epoch
            first_12_12_wall_clock_min = wall_clock_min
            first_12_12_kleene = dict(kleene_results)
            log(f"  *** first 12/12 + >99.9% at epoch {epoch} "
                f"({wall_clock_min:.2f} min) ***")

        # Early stop on 2 consecutive Kleene checks
        if kleene_passed == 12 and best_acc > EARLY_STOP_OVERALL:
            kleene_streak += 1
            if kleene_streak >= CONSECUTIVE_REQUIRED:
                converge_epoch = epoch
                converge_wall_clock_min = wall_clock_min
                log(f"early stop @ {epoch}: 12/12 x{CONSECUTIVE_REQUIRED} "
                    f"(wall={wall_clock_min:.2f}m)")
                break
        else:
            kleene_streak = 0

    pbar.set_postfix(loss=f'{avg_loss:.3f}', best=f'{best_acc:.3f}',
                     kl=f'{last_kleene_passed}/12')
pbar.close()

total_time_min = (time.time() - t_start) / 60.0

# ---- final Kleene + per-class ----
print("final kleene...")
final_kleene_results, final_kleene_passed = run_kleene(model)

model.eval()
with torch.no_grad():
    tidx = torch.arange(split, N, device=DEVICE)
    preds_all = []
    for j in range(0, len(tidx), BATCH * 4):
        bi = tidx[j:j + BATCH * 4]
        with torch.amp.autocast('cuda'):
            preds_all.append(model.classify(model(*get_batch(bi))))
    preds_all = torch.cat(preds_all)
    labels = TARGET[split:]
    per_class = {}
    for vid, vn in [(0, 'False'), (1, 'True'), (2, 'Unknown')]:
        m = labels == vid
        if m.sum() > 0:
            per_class[vn] = (preds_all[m] == labels[m]).float().mean().item()

torch.save(model.state_dict(), OUT_CKPT)

summary = {
    'seed': SEED,
    'model': 'theia_v2_clean_retrain',
    'params': params,
    'max_epochs': MAX_EPOCHS,
    'diagnostic_version': 'fixed_du_and_relgte',
    'cadence': 'overall_eval_every_10ep, kleene_every_20ep_after_99pct',
    'early_stop_criterion': {
        'overall_acc_threshold': EARLY_STOP_OVERALL,
        'per_rule_threshold_pct': EARLY_STOP_PER_RULE,
        'consecutive_required': CONSECUTIVE_REQUIRED,
    },
    'first_12_12_epoch': first_12_12_epoch,
    'first_12_12_wall_clock_min': first_12_12_wall_clock_min,
    'first_12_12_kleene': first_12_12_kleene,
    'converge_epoch': converge_epoch,
    'converge_wall_clock_min': converge_wall_clock_min,
    'final_overall_acc': best_acc,
    'final_kleene_passed': final_kleene_passed,
    'final_kleene_per_rule': final_kleene_results,
    'per_class_acc': per_class,
    'total_time_min': total_time_min,
    'paper_5seed_mean_min': 9.2,
    'paper_5seed_std_min': 3.5,
}
with open(OUT_JSON, 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\n{'='*60}")
print(f"THEIA v2 clean retrain | seed={SEED}")
print(f"{'='*60}")
print(f"  params:                  {params:,}")
print(f"  final overall acc:       {best_acc:.4f}")
print(f"  final Kleene:            {final_kleene_passed}/12")
for k, v in final_kleene_results.items():
    print(f"    {k:10s} {v:6.2f}% {'PASS' if v > 99 else 'FAIL'}")
print(f"  per-class:               {per_class}")
print()
print(f"  first 12/12 + >99.9%:")
print(f"    epoch:                 {first_12_12_epoch}")
print(f"    wall-clock:            {first_12_12_wall_clock_min}")
print(f"  converge (2 consecutive):")
print(f"    epoch:                 {converge_epoch}")
print(f"    wall-clock:            {converge_wall_clock_min}")
print(f"  total train time:        {total_time_min:.2f} min")
print()
print(f"  vs paper 5-seed mean:    9.2 +/- 3.5 min")
print(f"  output: {OUT_JSON}, {OUT_CKPT}, {OUT_LOG}")

log(f"FINAL acc={best_acc:.4f} kleene={final_kleene_passed}/12 "
    f"first_12_12={first_12_12_wall_clock_min} "
    f"converge={converge_wall_clock_min}")
log_f.close()
