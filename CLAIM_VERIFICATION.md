# Claim Verification

Code-level verification of the architectural and protocol claims in
*"THEIA: Learning Complete Kleene Three-Valued Logic in a Pure-Neural Modular Architecture"*
(arXiv:2604.11284), Appendix B, against the scripts in this repository.

**Method.** Every claim below (V1–V12) is checked against the released source. Code is quoted
verbatim and anchored by file + symbol (class/function/constant) rather than line numbers.
Parameter counts (V12) were obtained by actually instantiating the models on CPU with
`scripts/verify_params.py`. Provenance and executable-code equivalence with the original
experiment scripts are documented in the last two sections.

---

## 1. Paper component → script map

| # | Paper component | Script(s) |
|---|---|---|
| a | 4-domain THEIA training (§4.2, 2M samples, 5 seeds) | `scripts/train_4domain/theia_5seed_v2.py` (model class `IsisV9`); `_retrain_seed42_clean.py` (seed-42 clean retrain, source of the Table 1 seed-42 wall-clock) |
| b | Transformer baseline (post-LN BigTransformer, 3.64M) | `scripts/train_4domain/tf8l_5seed.py` (accuracy row, extension seeds, probing checkpoints) + `tf8l_5seed_earlystop_v2.py` (Kleene-aware early-stop wall-clock runs) + `overnight_runner.py` (orchestrator for extension seeds {31415, 27182, 14142}). Optimizer configurations: see "Optimizer configurations" below |
| c | Kleene diagnostic harness (doubly-fixed) | 12-rule version embedded in `theia_5seed_v2.py` (`KLEENE_TESTS`, `make_kleene_test`, `run_kleene`); full 39-rule version `scripts/diagnostics/kleene_full_diagnostic.py`; tuned-Transformer 39-rule version `scripts/tuned_tf/_eval_N1a_kleene_full.py` |
| d | Data generation (NUM_RANGE=20, P_unk=0.15, Boolean-OR propagation) | embedded: `theia_5seed_v2.py:build_dataset`, `tf8l_5seed*.py:gen_data`, `theia_chain_v3_5seed.py:gen_single_step_data/gen_chain_data`; eval-only copy in `theia/_theia_model_def.py` |
| e | Probes (linear SVM, MLP, Transformer layer-wise) | `scripts/probing/`: `svm_probe_5seed.py`, `probe_cleansplit.py`, `nonlinear_probe.py` (2-hidden-layer MLP), `deep_nonlinear_probe.py` (depths {2,4,6}), `probe_tf8l_layers.py`, `probe_tuned_tf_layers.py`, `per_boundary_HU_bayes_ceiling.py`, `extract_hidden_states_5seed.py`, aggregators |
| f | Set-boundary operator decomposition (Probe A/B/C/D) | `scripts/probing/probe_op_decomposition.py` (defines A–D), plus `probe_stratified.py`, `probe_operator_stratified.py`, `probe_op_identity_all_boundaries.py`, `probe_sanity_check.py` |
| g | Activation patching (OR + AND pairs) | `scripts/patching/activation_patching.py` (+`_5seed`, `_and`) |
| h | Three-phase chain pipeline | `scripts/chain_pipeline/theia_chain_v3_5seed.py` (`THEIAStep`, `TransitionNet`, `gumbel_st`, `phase1/2/3`); `mod3_state_distribution.py` |
| i | Backbone ablations | `scripts/ablations/`: `simple_mlp_backbone_ablation.py` (flat MLP 0.80M), `matched_mlp_backbone_ablation.py` (2.75M, HIDDEN=1243) + `_6L`/`_12L` depth variants, `resmlp_backbone_ablation.py` + `resmlp_grid_sweep.py` + per-config 5-seed completions, `transformer_chain_ablation.py` (pre-LN TF8LTuned, `norm_first=True`) |
| j | Tuned Transformer recipe + control on THEIA | `scripts/tuned_tf/tf8l_5seed_tuned.py` (`PEAK_LR = 1e-4`, `BETAS = (0.9, 0.98)`, `WARMUP_EPOCHS = 5`, linear warmup + cosine); `tuned_theia_5seed.py` (same constants applied to THEIA) |
| k | §4.3 ablations | `theia_subspace_ablation.py` (single-MLP Logic Engine), `theia_nobridge_ablation.py`, `theia_punk005_ablation.py` (P_unk 0.05→0.50; legacy sweep `punk_ablation.py`), `isis_v10_ood_hops.py` (GNN multi-hop OOD) + `ablation_msg_passes.py` + `decisive_edge_depth.py`, `chain_v4_modular.py` (mod-5 on graphs, negative result), `mod5_simple_backbone.py` |
| l | Statistics / figures | `scripts/plotting/`: `bootstrap_ci.py`, `training_time_analyzer.py`, `aggregate_v2.py`, `merge_tuned_theia_results.py`, `tf8l_earlystop_aggregate.py`, `kleene_chart.py`, `isis_tsne_5seed_en.py`; `scripts/probing/cluster_analysis_5seed.py` |

No component is missing. Two structural notes: data generation has no standalone module (it is
embedded in each training script, as in the original experiments), and the 12-rule diagnostic is
embedded rather than a separate file.

---

## 2. Appendix B claims (V1–V12)

### V1 — Numerical encoder input: raw integer or x/NUM_RANGE? → **normalized x/20**

The dataset is normalized at generation time (`theia_5seed_v2.py`, `build_dataset` return):

```python
return (a_val[perm].float()/NUM_RANGE, b_val[perm].float()/NUM_RANGE,
        d_val[perm].float()/NUM_RANGE, set_bits[perm], s_unknown[perm], ...)
```

The diagnostic constructor (`make_kleene_test`) returns `a.float()/NUM_RANGE` likewise. These
normalized values flow through `get_batch` → `model(...)` → `NumEnc.forward` with no rescaling:

```python
def forward(self, x, is_unknown):
    v = self.f(x.unsqueeze(-1))
    unk = self.unknown_vec.unsqueeze(0).expand_as(v)
    return torch.where(is_unknown.unsqueeze(-1), unk, v)
```

**Verdict: confirmed — enc(x) receives x/20.** Scope note: the chain-pipeline `THEIAStep`
(`theia_chain_v3_5seed.py`) feeds **raw integers** to its `NumEncoder` (by design; a separate
model variant). V1 refers to the 4-domain model.

### V2 — Encoder ownership → **consistent**

`ArithEngine` builds a single `self.ne = NumEnc()` used for both operands
(`self.ne(a,a_unk), self.ne(b,b_unk)`) — shared encoder, shared unknown vector. `OrderEngine`
constructs its own `NumEnc()` (independent encoder and unknown vector). `SetEnc` is
`Linear(SET_DIM=21, 64) → GELU → Linear(64, 128)` with its own `unknown_vec`
(`D_MODEL=128`, so `D_MODEL//2 = 64`).

### V3 — Unknown replacement → **consistent**

`NumEnc`/`SetEnc`: `self.unknown_vec = nn.Parameter(torch.randn(D_MODEL))` (standard-normal
init) and `torch.where(is_unknown.unsqueeze(-1), unk, v)` — hard replacement gated by the flag;
no flag concatenation, no side channel (flags enter the model only through the encoders).
Chain-variant `NumEncoder` uses an equivalent mask blend with `randn*0.02` init (chain scope).

### V4 — Fusion wiring → **consistent**

`OrderEngine` (G/L/E) and `LogicEngine` (C/D/I) are isomorphic:

```python
self.Gg = nn.Sequential(nn.Linear(D_MODEL*2, D_MODEL), nn.GELU(), nn.LayerNorm(D_MODEL))
...
g = self.Gg(torch.cat([g, e], dim=-1))
l = self.Lg(torch.cat([l, e], dim=-1))
e = self.Eg(torch.cat([e, g, l], dim=-1))
```

With `D_MODEL*2 = 256` and `D_MODEL*3 = 384`: Fuse2 = Linear(256,128)→GELU→LN(128),
Fuse3 = Linear(384,128)→GELU→LN(128), and the third line uses the already-updated `g, l` —
i.e. h1′=Fuse2₁([h1;h3]), h2′=Fuse2₂([h2;h3]), h3′=Fuse3([h3;h1′;h2′]).

### V5 — Output head → **consistent**

```python
self.out_head = nn.Sequential(nn.Linear(D_MODEL, D_MODEL), nn.GELU(), nn.Dropout(0.1), nn.LayerNorm(D_MODEL))
self.sv = nn.Embedding(N_VALS, D_MODEL)
nn.init.orthogonal_(self.sv.weight)
```

Training uses dot-product logits with class-weighted CE (`logits = out @ model.sv.weight.T`;
`F.cross_entropy(logits, TARGET[idx], weight=class_weights)`); inference uses cosine argmax
(`IsisV9.classify`: normalize both, `(vn @ sn.T).argmax(dim=-1)`). Prototypes are 3×128,
orthogonally initialized.

### V6 — Class weights → **consistent**

`class_weights = torch.tensor([1.0, 1.0, 2.0], device=DEVICE)` with `VAL_FALSE=0, VAL_TRUE=1,
VAL_UNKNOWN=2` — i.e. w_F=1.0, w_T=1.0, w_U=2.0. (The chain pipeline's phase 1 uses
`[1.0, 2.0, 1.0]` under its own label encoding F=0/U=1/T=2 — also w_U=2.0.)

### V7 — Chain head → **consistent**

`theia_chain_v3_5seed.py`, `OutHead`:

```python
self.proj = nn.Sequential(nn.Linear(DIM, DIM), nn.GELU(), nn.Linear(DIM, DIM))
proto = torch.zeros(3, DIM)
proto[0, :DIM//3] = 1.0; proto[1, DIM//3:2*DIM//3] = 1.0; proto[2, 2*DIM//3:] = 1.0
...
x = F.normalize(self.proj(x), dim=-1)
return torch.matmul(x, F.normalize(self.proto, dim=-1).T) * 10.0
```

Linear→GELU→Linear, L2-normalized scaled cosine against an axis-aligned (block one-hot,
mutually orthogonal) prototype matrix, temperature (scale) 10.

### V8 — Tau schedule → **consistent**

`phase3`: `tau = max(0.1, 0.5 - ep * 0.01)`.

### V9 — Plateau restart → **consistent**

`phase1`: coarse plateau `if ep >= 40 and best_this_try < 0.90: ... break`, followed by full
reinit on restart (`m.reset_parameters()` for Linear/Embedding; `nn.init.normal_(p.data, 0, 0.02)`
for `unk`/`proto` parameters). A fine-plateau branch (no improvement for 30 epochs after
reaching 90%) also restarts, as disclosed in §4.4 of the paper.

### V10 — Data semantics → **consistent**

`build_dataset`: `a_val, b_val = torch.randint(1, NUM_RANGE+1, ...)` (U{1..20});
`d_val = torch.randint(0, NUM_RANGE+1, ...)` (U{0..20}); four independent per-input flags with
`P_UNKNOWN = 0.15`. Arithmetic and propagation:

```python
c_val[arith==ARITH_ADD] = torch.clamp(a_val+b_val, 0, NUM_RANGE)[...]
c_val[arith==ARITH_SUB] = torch.abs(a_val-b_val)[...]            # |a-b|, not a-b
c_val[arith==ARITH_MUL] = torch.clamp(a_val*b_val, 0, NUM_RANGE)[...]
c_val[arith==ARITH_MOD] = (a_val % torch.clamp(b_val, 1, NUM_RANGE))[...]
c_unknown = a_unknown | b_unknown
ord_unknown = c_unknown | d_unknown
set_op_unknown = s_unknown | c_unknown
```

`clamp(·, 0, 20)` on positive inputs equals `min(·, 20)`, so c = min(a+b,20), |a−b|,
min(a·b,20), a mod b (b ≥ 1), with Boolean-OR Unknown propagation — all as claimed.

### V11 — Diagnostic construction fixes → **consistent**

`make_kleene_test` (and `kleene_full_diagnostic.py:construct_rule_samples`):

```python
# Fix #1: Unknown is injected via d (a_unk = b_unk = False)
if vo == 2: du[:] = True
...
if vo == 1:
    # Fix #2: REL_GTE with d = max(0, c-1)
    d = torch.clamp(c-1, min=0)
    rl = torch.full((n,), REL_GTE, ...)
```

The 39-rule version additionally constructs val_ord = False via REL_LT with d = c.

### V12 — Parameter counts → **all exact**

Output of `python scripts/verify_params.py` (CPU-only; `CUDA_VISIBLE_DEVICES=-1` with a
no-GPU assertion; Python 3.12.10, torch 2.12.0.dev+cu128):

```
module                       expected       actual  verdict
--------------------------------------------------------------
IsisV9.total                2,751,232    2,751,232  OK
IsisV9.arith_eng              404,736      404,736  OK
IsisV9.order_eng            1,179,648    1,179,648  OK
IsisV9.set_eng                208,128      208,128  OK
IsisV9.logic_eng              908,032      908,032  OK
IsisV9.bridges                 33,536       33,536  OK
IsisV9.out_head                16,768       16,768  OK
IsisV9.sv                         384          384  OK
THEIAStep.total             1,508,096    1,508,096  OK
TransitionNet.total             4,803        4,803  OK

IsisV9 module sum:           2,751,232  (OK)
BigTransformer total:        3,641,859  (paper states 3.64M; informative)

RESULT: all asserted parameter counts match.
```

---

## 3. Optimizer configurations (development-default; paper App. H, Correction)

Each architecture trains at its development-default configuration. The two configurations are
**not identical**; see App. H (Correction) of the paper for the corrected description of what
earlier drafts called the "matched" protocol.

| | THEIA (`theia_5seed_v2.py`, `_retrain_seed42_clean.py`) | Transformer (`tf8l_5seed.py`, `tf8l_5seed_earlystop_v2.py` — identical to each other) |
|---|---|---|
| Optimizer | AdamW(lr=1e-3, weight_decay=0.01, betas default (0.9, 0.999)) | AdamW(lr=5e-4, weight_decay=0.01, betas default (0.9, 0.999)) |
| Schedule | CosineAnnealingLR(T_max=200, eta_min=1e-5), no warmup | CosineAnnealingLR(T_max=150, eta_min=1e-5), no warmup |
| Batch size | 4096 | 2048 |
| Epoch budget | 200 | 150 |
| Grad clip / precision | 1.0 / FP16 AMP | 1.0 / FP16 AMP |
| Class weights | [F, T, U] = [1, 1, 2] | [F, T, U] = [1, 1, 2] |

Anchors: `theia_5seed_v2.py` constants `D_MODEL=128; BATCH=4096` and the
`optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)` /
`CosineAnnealingLR(opt, T_max=MAX_EPOCHS, eta_min=1e-5)` lines;
`tf8l_5seed.py` / `tf8l_5seed_earlystop_v2.py` constants `D_MODEL=192; BATCH=2048` and
`optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)`. The docstring of
`overnight_runner.py` describes the Transformer recipe incorrectly (lr=1e-3, batch 4096); the
file is preserved verbatim as the historical orchestrator, with a correction note appended —
the script constants are authoritative.

The tuned recipe (App. G) is a third configuration: AdamW(peak lr=1e-4, betas=(0.9, 0.98),
weight_decay=0.01), 5-epoch linear warmup then cosine, batch 2048
(`tf8l_5seed_tuned.py`, `tuned_theia_5seed.py`).

---

## 4. Notes and known divergences

1. **"Byte-identical" v_set check.** The patching pairs share all Set-path inputs by
   construction, so v_set is byte-identical across sides under a deterministic forward pass.
   The runtime sanity check is the weaker `torch.allclose(H_T['set'], H_U['set'], atol=1e-6)`
   (`activation_patching.py`, `activation_patching_and.py`). Kept as-is for zero divergence
   from the original experiments.
2. **2026-04-20 supplement edits.** Released training/diagnostic scripts carry three
   post-submission edits, each marked in-file: evaluation forced to FP32 (autocast removed at
   eval sites), evaluation/diagnostic cadence unified to every 5 epochs, and deterministic
   per-rule diagnostic seeds (`op*9+va*3+vb`, replacing PYTHONHASHSEED-dependent `hash()`).
   All headline numbers were re-verified after these edits (195/195, OR 4898/4898,
   AND 4719/4719); reruns may early-stop a few epochs sooner than the paper's wall-clock.
3. **Chain variant differences (disclosed in §4.4).** The chain `THEIAStep` differs from the
   4-domain model: raw-integer numeric encoding, `Linear(1,128)→GELU→Linear(128,128)` encoder,
   `randn*0.02` unknown init, per-operand encoders, different data semantics (`a-b`, unclamped
   add/mul, integer division) and label encoding (F=0/U=1/T=2).
4. **Training/inference objective asymmetry.** Training uses unnormalized dot-product logits;
   inference uses cosine argmax. Matches the paper's wording ("equivalent at inference when
   prototypes remain orthonormal"); prototypes are trainable.
5. **Seed-42 checkpoint paths.** `activation_patching_5seed.py` (and `_and`) point seed 42 at
   the legacy pre-retrain checkpoint (`multi_seed_results/theia/seed_42/`) to reproduce the
   paper's 4898/4898 and 4719/4719 aggregates; both seed-42 checkpoints pass 39/39.
   `kleene_full_diagnostic.py --aggregate` also expects seed 42 under
   `multi_seed_results/theia/seed_42/` (legacy layout); when retraining from this repo, place
   the seed-42 results there before aggregating.
6. **Mod-5 on graphs.** `chain_v4_modular.py` is the (only) source of the negative result; the
   topology is a chain processed by message passing. No general-graph variant exists.
7. **Legacy checkpoint dependencies.** `punk_ablation.py` expects `isis_v9.pth` (historical;
   use `theia_punk005_ablation.py` for the §4.3 ablation); `ablation_msg_passes.py` expects
   `isis_v10_ood_best.pth`, produced by `isis_v10_ood_hops.py`.

---

## 5. Provenance and code equivalence

Scripts are copied from the author's experiment archive (the versions whose runs produced the
paper numbers, including the 2026-04-20 supplement edits noted above). Two transformation
passes were applied for release; **neither alters executable code semantics**, verified
mechanically by an AST-equivalence gate (parse, normalize docstrings, compare `ast.dump`) plus
full byte-compilation of every file:

**Pass 1 — portability/anonymization (executable string literals; exhaustive list).**

| File | Change |
|---|---|
| `train_4domain/theia_5seed_v2.py` | argparse `--output-root` default: absolute → `multi_seed_results\theia_v2` |
| `train_4domain/tf8l_5seed.py` | same → `multi_seed_results\tf8l` |
| `train_4domain/tf8l_5seed_earlystop_v2.py` | same → `multi_seed_results\tf8l_earlystop` |
| `train_4domain/mlp_5seed.py` | same → `multi_seed_results\mlp` |
| `tuned_tf/tf8l_5seed_tuned.py` | same → `multi_seed_results\tf8l_tuned` |
| `tuned_tf/tuned_theia_5seed.py` | gate-label return strings `'STOP_FOR_KK*'` → `'STOP_FOR_REVIEW*'` (written to JSON/stdout only; no control flow reads them) |
| `diagnostics/kleene_full_diagnostic.py` | `ROOT` constant: absolute → `multi_seed_results` |
| `ablations/theia_{subspace,nobridge,punk005}_ablation.py` | argparse `--output-root` default → `multi_seed_results\ablations\...` |
| `ablations/mod5_simple_backbone.py` | argparse `--output` default → `multi_seed_results\mod5_simple_backbone` |
| `ablations/ablation_msg_passes.py` | `torch.load` path → `isis_v10_ood_best.pth` (relative) |
| `ablations/chain_v4_modular.py` | `torch.save`/`torch.load` path → `theia_chain_v4_best.pth` (relative) |
| `ablations/punk_ablation.py` | `torch.load` path → `isis_v9.pth` (relative) |
| `plotting/{aggregate_v2,tf8l_earlystop_aggregate,training_time_analyzer}.py` | `ROOT` constant → `multi_seed_results` |
| `plotting/isis_tsne_5seed_en.py` | `SAVE_DIR` → `visualizations`; `DATA_DIR` → `.` |
| `plotting/kleene_chart.py` | `plt.savefig` path → `visualizations/kleene_comparison_en.png` |
| `probing/{cluster_analysis_5seed,nonlinear_probe,probe_cleansplit,probing_aggregate,tf8l_probing_aggregate}.py` | `ROOT` constant → `multi_seed_results` |
| `probing/probe_cleansplit.py` | `deep_ref_path` → `deep_nonlinear_probe_results.json` (relative) |
| `probing/extract_hidden_states_5seed.py` | `OUT_DIR` → `.` |

**Pass 2 — comment-only cleanup.** Comments and docstrings were condensed to standard
engineering style; no string literal, identifier, value, or statement was touched. Verified by
the AST gate above against the pre-cleanup tree.

`scripts/verify_params.py` is new for this release (verification utility, not part of any
experiment).
