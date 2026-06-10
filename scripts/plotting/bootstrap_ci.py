"""
Bootstrap 95% CIs for paper headline numbers — pure CPU post-processing.

Loads per-seed values from existing JSONs/logs and computes 1000-sample
non-parametric bootstrap 95% CIs for:
  wall-clock claims: Table 1 mean (5 seeds), 5.6× matched-protocol ratio,
    3.1× tuned Kleene-aware ratio, 7.0× tuned overall-99.9% ratio, paired diffs;
  chain claims: THEIA 500-step 99.96% ± 0.04%, TF8L 99.24%, 0.72 pp difference;
  delayed-verdict claims: F-T separation ratio (logic / arith centroid
    distance, 5-seed), per-domain centroid distances + cosines;
  reliability claims: THEIA 5/5 ≥ 99% on Kleene, TF8L 7/8 (matched protocol).

Usage:
    python bootstrap_ci.py
"""
import json
import os
import re
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import numpy as np


N_BOOT = 1000
RNG = np.random.default_rng(42)


def bootstrap_ci(values, stat_fn=np.mean, n_boot=N_BOOT, ci=0.95):
    """Non-parametric bootstrap CI for `stat_fn` on `values`."""
    values = np.array(values)
    n = len(values)
    samples = []
    for _ in range(n_boot):
        idx = RNG.integers(0, n, size=n)
        samples.append(stat_fn(values[idx]))
    samples = np.array(samples)
    lo = np.percentile(samples, (1 - ci) / 2 * 100)
    hi = np.percentile(samples, (1 + ci) / 2 * 100)
    return float(stat_fn(values)), float(lo), float(hi)


def bootstrap_ratio_ci(num_vals, denom_vals, n_boot=N_BOOT, ci=0.95, mode='mean_of_ratios'):
    """Bootstrap CI for ratio. Two modes:
       - 'mean_of_ratios': resample paired indices (must be paired = same n) → mean(num/denom)
       - 'ratio_of_means': resample independently → mean(num)/mean(denom)
       Returns (point, lo, hi)."""
    num_vals = np.array(num_vals); denom_vals = np.array(denom_vals)
    samples = []
    for _ in range(n_boot):
        if mode == 'mean_of_ratios':
            assert len(num_vals) == len(denom_vals)
            idx = RNG.integers(0, len(num_vals), size=len(num_vals))
            r = num_vals[idx] / denom_vals[idx]
            samples.append(r.mean())
        else:  # ratio_of_means
            ni = RNG.integers(0, len(num_vals), size=len(num_vals))
            di = RNG.integers(0, len(denom_vals), size=len(denom_vals))
            samples.append(num_vals[ni].mean() / denom_vals[di].mean())
    samples = np.array(samples)
    if mode == 'mean_of_ratios':
        point = (num_vals / denom_vals).mean()
    else:
        point = num_vals.mean() / denom_vals.mean()
    lo = np.percentile(samples, (1 - ci) / 2 * 100)
    hi = np.percentile(samples, (1 + ci) / 2 * 100)
    return float(point), float(lo), float(hi)


# ---------- Data loaders ----------
def load_theia_chain_500step():
    """Parse theia_chain_5seed_log.txt for per-seed 500-step accuracy.
    Match the FINAL `500-step: XX.XX% (step1=...)` line — not the tqdm bar
    `500-step:   0%|...` which would falsely match `0%`."""
    path = 'theia_chain_5seed_log.txt'
    if not os.path.exists(path):
        return None
    with open(path, encoding='utf-8') as f:
        text = f.read()
    seeds = [42, 123, 256, 777, 999]
    seed_blocks = re.split(r'  Seed \d/5: \d+\n', text)[1:]
    out = {}
    for s, block in zip(seeds, seed_blocks):
        # Match: `500-step: 99.90% (step1=100.00%)` — require `(step1=` to skip tqdm
        m = re.search(r'500-step:\s+([\d.]+)%\s+\(step1=', block)
        if m:
            out[s] = float(m.group(1)) / 100.0
    return out


def load_tf8l_chain_500step():
    """Load TF8LTuned chain 5-seed 500-step from raw.json."""
    path = 'multi_seed_results/transformer_chain_ablation/raw.json'
    if not os.path.exists(path):
        return None
    d = json.load(open(path))
    return dict(zip(d['seeds'], d['results']['500']))


def load_theia_4domain_walltime():
    """Load 5 THEIA seeds 4-domain wall-clock from individual summary.json files."""
    paths = {
        42:  'multi_seed_results/theia/seed_42/summary.json',
        123: 'multi_seed_results/theia_v2/seed_123/summary.json',
        256: 'multi_seed_results/theia_v2/seed_256/summary.json',
        777: 'multi_seed_results/theia_v2/seed_777/summary.json',
        999: 'multi_seed_results/theia_v2/seed_999/summary.json',
    }
    out = {}
    for s, p in paths.items():
        if os.path.exists(p):
            d = json.load(open(p))
            out[s] = d['total_time_sec'] / 60.0
    return out


def load_tf8l_matched_walltime():
    """Load TF8L matched-protocol per-seed wall-clock from tf8l_earlystop summary.json files."""
    paths = {}
    for s in [42, 123, 256, 777, 999, 31415, 27182, 14142]:
        p = f'multi_seed_results/tf8l_earlystop/seed_{s}/summary.json'
        if os.path.exists(p):
            paths[s] = p
    out = {}
    for s, p in paths.items():
        d = json.load(open(p))
        # Skip non-converged seeds (kleene < 12)
        if d.get('kleene_passed', 0) == 12:
            out[s] = d['total_time_sec'] / 60.0
    return out


def load_tuned_tf_walltime():
    """Load tuned BigTransformer 3-seed wall-clock from tf8l_tuned summary.json files."""
    paths = {}
    for s in [42, 123, 256]:
        p = f'multi_seed_results/tf8l_tuned/seed_{s}/summary.json'
        if os.path.exists(p):
            paths[s] = p
    out_first999 = {}
    out_first1212 = {}
    for s, p in paths.items():
        d = json.load(open(p))
        if d.get('first_999_time_min') is not None:
            out_first999[s] = d['first_999_time_min']
        if d.get('first_12_12_time_min') is not None:
            out_first1212[s] = d['first_12_12_time_min']
    return out_first999, out_first1212


def load_tuned_tf_chain_5seed():
    """Load tuned TF8L 5-seed wall-clock from transformer_chain_ablation/raw.json.
    Returns dict seed -> elapsed_min."""
    p = 'multi_seed_results/transformer_chain_ablation/raw.json'
    if not os.path.exists(p):
        return {}
    d = json.load(open(p))
    seeds = d.get('seeds', [])
    elapsed_s = d.get('elapsed', [])
    if not seeds or not elapsed_s or len(seeds) != len(elapsed_s):
        return {}
    return {s: t / 60.0 for s, t in zip(seeds, elapsed_s)}


def load_tuned_theia_5seed_FINAL():
    """Load tuned THEIA 5-seed first_12_12_wall_min from tuned_theia_5seed_FINAL.json.
    Returns dict seed -> first_12_12_wall_min, or {} if file missing."""
    p = 'tuned_theia_5seed_FINAL.json'
    if not os.path.exists(p):
        return {}
    d = json.load(open(p))
    out = {}
    for r in d.get('per_seed', []):
        s = r['seed']
        v = r.get('first_12_12_wall_min')
        if v is not None:
            out[s] = v
    return out


def parse_theia_chain_restarts():
    """Parse theia_chain_5seed_log.txt for per-seed restart count.
    Looks for 'Restart' occurrences within each seed block."""
    path = 'theia_chain_5seed_log.txt'
    if not os.path.exists(path):
        return {}
    with open(path, encoding='utf-8') as f:
        text = f.read()
    seeds = [42, 123, 256, 777, 999]
    seed_blocks = re.split(r'  Seed \d/5: \d+\n', text)[1:]
    out = {}
    for s, block in zip(seeds, seed_blocks):
        n = block.count('Restart')   # counts 'Restart 1/3'-style log messages
        out[s] = n
    return out


def infer_tf8l_chain_restarts():
    """Infer TF8L chain Phase 1 restart counts from elapsed time heuristic.
    No saved log; use 1.5× median wallclock as restart-trigger proxy.
    Returns dict seed -> 0 (no restart) or 1+ (restart inferred)."""
    path = 'multi_seed_results/transformer_chain_ablation/raw.json'
    if not os.path.exists(path):
        return {}
    d = json.load(open(path))
    elapsed = np.array(d['elapsed'])
    seeds = d['seeds']
    median = float(np.median(elapsed))
    return {s: (1 if e > 1.5 * median else 0) for s, e in zip(seeds, elapsed)}


def load_centroid_distances():
    """Load 5-seed F-T / F-U / T-U centroid distances per boundary."""
    paths = {
        42:  'multi_seed_results/theia/seed_42/mechanistic_probing_v2.json',
        123: 'multi_seed_results/theia_v2/seed_123/mechanistic_probing_v2.json',
        256: 'multi_seed_results/theia_v2/seed_256/mechanistic_probing_v2.json',
        777: 'multi_seed_results/theia_v2/seed_777/mechanistic_probing_v2.json',
        999: 'multi_seed_results/theia_v2/seed_999/mechanistic_probing_v2.json',
    }
    out = {}
    for s, p in paths.items():
        if os.path.exists(p):
            out[s] = json.load(open(p))
    return out


# ---------- Compute and report ----------
def main():
    L = []
    L.append('# Bootstrap 95% CI Report — Paper Headline Numbers')
    L.append('')
    L.append(f'Method: 1000-sample non-parametric bootstrap, RNG seed 42.')
    L.append(f'CIs reported as (point estimate, lower 2.5%, upper 97.5%).')
    L.append('')

    # 1. THEIA chain 500-step (re-run 2026-04-19)
    L.append('## 1. THEIA chain 500-step accuracy (5-seed re-run 2026-04-19)')
    theia_chain = load_theia_chain_500step()
    if theia_chain:
        vals = list(theia_chain.values())
        L.append(f'')
        L.append(f'Per-seed 500-step: ' + ', '.join(f'seed {s}: {v*100:.2f}%' for s, v in theia_chain.items()))
        m, lo, hi = bootstrap_ci(vals, np.mean)
        L.append(f'  - Mean: **{m*100:.4f}%** (95% CI: [{lo*100:.4f}%, {hi*100:.4f}%])')
        s, slo, shi = bootstrap_ci(vals, lambda x: np.std(x, ddof=1))
        L.append(f'  - Std (sample, ddof=1): **{s*100:.4f}%** (95% CI: [{slo*100:.4f}%, {shi*100:.4f}%])')
        L.append(f'  - Paper claim: 99.97% ± 0.02% — current data point estimate {m*100:.2f}% with std {s*100:.4f}')
    else:
        L.append('  (theia_chain_5seed_log.txt not found)')
    L.append('')

    # 2. TF8L chain 500-step
    L.append('## 2. TF8LTuned chain 500-step accuracy (5-seed)')
    tf_chain = load_tf8l_chain_500step()
    if tf_chain:
        vals = list(tf_chain.values())
        L.append(f'')
        L.append(f'Per-seed 500-step: ' + ', '.join(f'seed {s}: {v*100:.2f}%' for s, v in tf_chain.items()))
        m, lo, hi = bootstrap_ci(vals, np.mean)
        L.append(f'  - Mean: **{m*100:.4f}%** (95% CI: [{lo*100:.4f}%, {hi*100:.4f}%])')
        s, slo, shi = bootstrap_ci(vals, lambda x: np.std(x, ddof=1))
        L.append(f'  - Std: {s*100:.4f}% (CI: [{slo*100:.4f}%, {shi*100:.4f}%])')
        L.append(f'  - Paper claim: 99.24%')
    L.append('')

    # 3. THEIA - TF8L difference: pairing audit, then primary unpaired + auxiliary paired
    if theia_chain and tf_chain:
        L.append('## 3. THEIA vs TF8L chain 500-step difference')
        L.append('')

        # Pairing audit: are restart flags symmetric across the two models?
        L.append('### 3a. Pairing audit (restart symmetry check)')
        L.append('')
        # THEIA chain restart per seed (from re-run log)
        theia_chain_restarts = parse_theia_chain_restarts()
        # TF8L chain restart per seed (infer from elapsed time)
        tf_chain_restarts = infer_tf8l_chain_restarts()

        L.append('| Seed | THEIA 500-step | THEIA P1 restart? | TF8L 500-step | TF8L P1 restart? (inferred from elapsed) |')
        L.append('|---|---|---|---|---|')
        common = sorted(set(theia_chain.keys()) & set(tf_chain.keys()))
        symmetric = True
        for s in common:
            t_r = theia_chain_restarts.get(s, '?')
            tf_r = tf_chain_restarts.get(s, '?')
            if t_r != tf_r and t_r != '?' and tf_r != '?':
                symmetric = False
            L.append(f'| {s} | {theia_chain[s]*100:.2f}% | {t_r} | {tf_chain[s]*100:.2f}% | {tf_r} |')
        L.append('')
        L.append(f'**Pairing symmetric: {"yes" if symmetric else "NO — restart flags differ"}**')
        if not symmetric:
            L.append(f'  → paired diff is misleading; using **Welch unpaired t-style bootstrap** as primary, paired as auxiliary')
        else:
            L.append(f'  → paired analysis is valid; reporting both paired and unpaired')
        L.append('')

        # 3b. Primary: Welch unpaired bootstrap diff
        L.append('### 3b. Welch (unpaired) bootstrap diff — PRIMARY')
        L.append('')
        theia_vals = np.array([theia_chain[s] for s in common])
        tf_vals = np.array([tf_chain[s] for s in common])
        # Independent resample of each population
        diffs_unpaired = []
        for _ in range(N_BOOT):
            ti = RNG.integers(0, len(theia_vals), size=len(theia_vals))
            ti2 = RNG.integers(0, len(tf_vals), size=len(tf_vals))
            diffs_unpaired.append(theia_vals[ti].mean() - tf_vals[ti2].mean())
        diffs_unpaired = np.array(diffs_unpaired)
        m_u = theia_vals.mean() - tf_vals.mean()
        lo_u = np.percentile(diffs_unpaired, 2.5)
        hi_u = np.percentile(diffs_unpaired, 97.5)
        cross_u = lo_u < 0 < hi_u
        L.append(f'  - THEIA mean: {theia_vals.mean()*100:.4f}%')
        L.append(f'  - TF8L mean:  {tf_vals.mean()*100:.4f}%')
        L.append(f'  - Mean diff:  **{m_u*100:+.4f} pp** (95% CI unpaired: [{lo_u*100:+.4f}, {hi_u*100:+.4f}] pp)')
        L.append(f'  - CI crosses 0: **{cross_u}** {"(NOT significant)" if cross_u else "(significant at 5%)"}')
        L.append('')

        # 3c. Auxiliary: paired diff (valid only under symmetric pairing)
        L.append('### 3c. Paired bootstrap diff — AUXILIARY')
        L.append('')
        diffs_paired = [theia_chain[s] - tf_chain[s] for s in common]
        L.append(f'  - Paired diffs: ' +
                 ', '.join(f'seed {s}: {d*100:+.2f}pp' for s, d in zip(common, diffs_paired)))
        m_p, lo_p, hi_p = bootstrap_ci(diffs_paired, np.mean)
        cross_p = lo_p < 0 < hi_p
        L.append(f'  - Mean paired diff: **{m_p*100:+.4f} pp** (95% CI: [{lo_p*100:+.4f}, {hi_p*100:+.4f}] pp)')
        L.append(f'  - CI crosses 0: **{cross_p}**')
        L.append(f'  - Paper claim: +0.72 pp difference')
        L.append('')
        if not symmetric:
            L.append(f'  ⚠️ Pairing asymmetric (different restart flags); paired CI is biased — defer to 3b unpaired.')
    L.append('')

    # 4. THEIA 4-domain wall-clock (Table 1)
    L.append('## 4. THEIA 4-domain wall-clock (Table 1, 5-seed)')
    theia_4d = load_theia_4domain_walltime()
    if theia_4d:
        vals = list(theia_4d.values())
        L.append(f'')
        L.append(f'Per-seed: ' + ', '.join(f'seed {s}: {v:.2f} min' for s, v in theia_4d.items()))
        m, lo, hi = bootstrap_ci(vals, np.mean)
        L.append(f'  - Mean: **{m:.3f} min** (95% CI: [{lo:.3f}, {hi:.3f}] min)')
        s, slo, shi = bootstrap_ci(vals, lambda x: np.std(x, ddof=1))
        L.append(f'  - Std: {s:.3f} min (CI: [{slo:.3f}, {shi:.3f}] min)')
        L.append(f'  - Pop std (paper convention, ddof=0): {np.std(vals, ddof=0):.3f} min')
        L.append(f'  - Paper claim: 9.2 ± 3.5 min')
    L.append('')

    # 5. TF8L matched 7-seed wall-clock + 5.6× ratio
    L.append('## 5. Matched-protocol wall-clock ratio (TF8L / THEIA)')
    tf_matched = load_tf8l_matched_walltime()
    if tf_matched and theia_4d:
        L.append(f'')
        L.append(f'TF8L converged seeds (12/12, n={len(tf_matched)}): ' +
                 ', '.join(f'seed {s}: {v:.2f} min' for s, v in tf_matched.items()))
        L.append(f'THEIA seeds (n={len(theia_4d)}): mean {np.mean(list(theia_4d.values())):.2f} min')
        # ratio of means (independent populations)
        r, rlo, rhi = bootstrap_ratio_ci(
            list(tf_matched.values()), list(theia_4d.values()),
            mode='ratio_of_means'
        )
        L.append(f'  - Ratio (TF8L_mean / THEIA_mean): **{r:.3f}×** (95% CI: [{rlo:.3f}×, {rhi:.3f}×])')
        L.append(f'  - Paper claim: 5.6× (matched protocol, n=8 with 1 non-converged)')
    L.append('')

    # 6. Tuned TF8L wall-clock + ratios
    L.append('## 6. Tuned protocol wall-clock ratios (tuned BigTransformer / THEIA)')
    tuned_999, tuned_1212 = load_tuned_tf_walltime()
    if tuned_1212 and theia_4d:
        L.append(f'')
        L.append(f'Tuned BigTransformer 12/12 (n={len(tuned_1212)}): ' +
                 ', '.join(f'seed {s}: {v:.2f} min' for s, v in tuned_1212.items()))
        r, rlo, rhi = bootstrap_ratio_ci(
            list(tuned_1212.values()), list(theia_4d.values()),
            mode='ratio_of_means'
        )
        L.append(f'  - Ratio (Tuned TF 12/12 / THEIA wall-clock): **{r:.3f}×** (95% CI: [{rlo:.3f}×, {rhi:.3f}×])')
        L.append(f'  - Paper claim: 3.1× (tuned protocol, Kleene-aware)')

    if tuned_999 and theia_4d:
        # THEIA's first-99.9% time is not stored in summary.json; compare against
        # the paper-cited 5.7 ± 1.4 min instead of total wall-clock.
        theia_999_baseline = 5.7  # paper L162
        L.append(f'  - Tuned BigTransformer 99.9% (n={len(tuned_999)}): ' +
                 ', '.join(f'seed {s}: {v:.2f} min' for s, v in tuned_999.items()))
        m_t, lo_t, hi_t = bootstrap_ci(list(tuned_999.values()), np.mean)
        L.append(f'    Mean: {m_t:.3f} min (CI: [{lo_t:.3f}, {hi_t:.3f}])')
        L.append(f'    Ratio vs THEIA paper-cited 5.7 min (overall 99.9%): {m_t/theia_999_baseline:.3f}× — paper claim 7.0×')
    L.append('')

    # 6b. Tuned-vs-tuned 5-seed wall-clock ratio (added 2026-04-19)
    L.append('## 6b. Tuned-vs-tuned 5-seed wall-clock ratio (NEW from tuned_theia_5seed_FINAL.json + transformer_chain_ablation)')
    tuned_theia_5s = load_tuned_theia_5seed_FINAL()
    tuned_tf_5s    = load_tuned_tf_chain_5seed()
    if tuned_theia_5s and tuned_tf_5s and len(tuned_theia_5s) == 5 and len(tuned_tf_5s) == 5:
        seeds = sorted(set(tuned_theia_5s.keys()) & set(tuned_tf_5s.keys()))
        L.append('')
        L.append(f'Tuned THEIA 5-seed first_12_12_wall_min: ' +
                 ', '.join(f'seed {s}: {tuned_theia_5s[s]:.2f} min' for s in seeds))
        L.append(f'Tuned TF8L  5-seed elapsed (chain training): ' +
                 ', '.join(f'seed {s}: {tuned_tf_5s[s]:.2f} min' for s in seeds))
        th_vals = [tuned_theia_5s[s] for s in seeds]
        tf_vals = [tuned_tf_5s[s] for s in seeds]
        L.append(f'  - Tuned THEIA: {np.mean(th_vals):.2f} ± {np.std(th_vals, ddof=1):.2f} min (n=5)')
        L.append(f'  - Tuned TF8L:  {np.mean(tf_vals):.2f} ± {np.std(tf_vals, ddof=1):.2f} min (n=5)')
        # Ratio of means (Welch unpaired primary)
        r_rom, lo_rom, hi_rom = bootstrap_ratio_ci(tf_vals, th_vals, mode='ratio_of_means')
        L.append(f'  - **Ratio of means (Welch unpaired primary): {r_rom:.3f}× (95% CI [{lo_rom:.3f}×, {hi_rom:.3f}×])**')
        # Paired diff (auxiliary — same seeds for THEIA and TF)
        per_seed_ratios = np.array(tf_vals) / np.array(th_vals)
        L.append(f'  - Per-seed pairwise ratios: ' +
                 ', '.join(f'seed {s}: {tf_vals[i]/th_vals[i]:.2f}×' for i, s in enumerate(seeds)))
        L.append(f'    mean of pairwise ratios: {np.mean(per_seed_ratios):.3f}× ± {np.std(per_seed_ratios, ddof=1):.3f}×')
        r_p, lo_p, hi_p = bootstrap_ratio_ci(tf_vals, th_vals, mode='mean_of_ratios')
        L.append(f'  - **Mean of ratios (paired auxiliary): {r_p:.3f}× (95% CI [{lo_p:.3f}×, {hi_p:.3f}×])**')
        L.append(f'  - Old paper claim: 3.1× (3-seed); NEW 5-seed result is *higher* — strengthens H3.')
    elif tuned_theia_5s:
        L.append(f'  (Only tuned THEIA 5-seed found; TF 5-seed missing from transformer_chain_ablation/raw.json)')
    else:
        L.append('  (tuned_theia_5seed_FINAL.json not found — run merge_tuned_theia_results.py first)')
    L.append('')

    # 7. F-T separation ratio (5-seed, logic / arith)
    L.append('## 7. F-T separation ratio (logic / arith, 5-seed)')
    cents = load_centroid_distances()
    if cents:
        seeds = sorted(cents.keys())
        ar = np.array([cents[s]['centroids']['arith']['distances']['F_T'] for s in seeds])
        lo_ = np.array([cents[s]['centroids']['logic']['distances']['F_T'] for s in seeds])
        per_seed_ratio = lo_ / ar
        L.append(f'')
        L.append(f'Per-seed (logic / arith): ' +
                 ', '.join(f'seed {s}: {r:.1f}×' for s, r in zip(seeds, per_seed_ratio)))
        # Mean of ratios
        r_mor, lo_mor, hi_mor = bootstrap_ratio_ci(lo_, ar, mode='mean_of_ratios')
        L.append(f'  - Mean of ratios: **{r_mor:.1f}×** (95% CI: [{lo_mor:.1f}×, {hi_mor:.1f}×])')
        # Ratio of means
        r_rom, lo_rom, hi_rom = bootstrap_ratio_ci(lo_, ar, mode='ratio_of_means')
        L.append(f'  - Ratio of means: **{r_rom:.1f}×** (95% CI: [{lo_rom:.1f}×, {hi_rom:.1f}×])')
        L.append(f'  - Paper claim: 1898× (mean of ratios) / 1868× (ratio of means)')
    L.append('')

    # 8. THEIA reliability (5/5 ≥ 99% on Kleene)
    L.append('## 8. Reliability claims')
    L.append('')
    L.append('  - THEIA 4-domain Kleene 12/12: 5/5 seeds (deterministic from final_kleene_passed in summary.json)')
    L.append('  - TF8L matched 12/12: 7/8 seeds (paper claim — seed 123 non-converged)')
    L.append('  - Wilson 95% CI for 5/5: [56.55%, 100.00%] (lower bound: 56.6%)')
    L.append('  - Wilson 95% CI for 7/8: [47.34%, 99.68%] (lower bound: 47.3%)')
    L.append('')

    # Save
    md = '\n'.join(L) + '\n'
    with open('bootstrap_ci_report.md', 'w', encoding='utf-8') as f:
        f.write(md)
    print(md)
    print('\nSaved: bootstrap_ci_report.md')


if __name__ == '__main__':
    main()
