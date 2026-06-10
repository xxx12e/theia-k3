"""
Layer-wise probing for TUNED TF8L checkpoints (Transformer layer-wise probing; §5.3).

The tuned TF uses a different architecture (TF8LTuned: 8 tokens with discrete
embeddings + unk sentinel) than the matched TF (BigTransformer: 11 tokens with
continuous NumEnc); this script uses the TF8LTuned architecture. Data comes from
gen_single_step_data in theia_chain_v3_5seed (the data the tuned TF was trained
on); probes the input embedding + each of 8 transformer layers.

Output: per-layer SVM accuracy, F-T centroid distance, separation ratio.
"""
import json, os, sys, time
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import numpy as np
import torch
import torch.nn as nn
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler

from theia_chain_v3_5seed import gen_single_step_data, DEVICE, NUM_RANGE, LOGIC_OPS

D_MODEL = 192; N_HEADS = 8; N_LAYERS_TF = 8; DIM_FF = 768
NUM_VAL = NUM_RANGE + 2
N_SAMPLES = 50000
DATA_SEED = 999


class TF8LTuned(nn.Module):
    """Matches tf8l_tuned_seed42.py architecture exactly."""
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
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=N_LAYERS_TF)
        self.final_norm = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, 3)

    def _tokenize(self, x):
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

    def forward_with_layer_outputs(self, x):
        """Returns dict: {0: input_emb_meanpool, 1..8: per-layer meanpool}"""
        toks = self._tokenize(x)
        outputs = {0: toks.mean(dim=1).float().cpu().numpy()}
        h = toks
        for i, layer in enumerate(self.encoder.layers):
            h = layer(h)
            outputs[i + 1] = h.mean(dim=1).float().cpu().numpy()
        return outputs


def probe_classification(X_tr, y_tr, X_te, y_te):
    sc = StandardScaler()
    Xs_tr = sc.fit_transform(X_tr)
    Xs_te = sc.transform(X_te)
    dual = X_tr.shape[0] < X_tr.shape[1]
    clf = LinearSVC(C=1.0, max_iter=2000, dual=dual)
    clf.fit(Xs_tr, y_tr)
    return clf.score(Xs_te, y_te)


def compute_ft_distance(X, labels):
    # Chain script: False=0, Unknown=1, True=2
    f_mask = labels == 0
    t_mask = labels == 2
    if f_mask.sum() == 0 or t_mask.sum() == 0:
        return None
    return float(np.linalg.norm(X[f_mask].mean(axis=0) - X[t_mask].mean(axis=0)))


# Seed 42 tuned ckpt uses BigTransformer (matched arch, tuned hyperparams).
# Seeds 123/256 tuned ckpts use TF8LTuned (different arch).
# Only probe seeds 123/256 here (same architecture, comparable).
# Seed 42 already has probing data from probe_tf8l_layers.py (matched arch).
CHECKPOINTS = [
    (123, 'overnight_results/tuned_seed123/best_model.pth'),
    (256, 'overnight_results/tuned_seed256/best_model.pth'),
]


def main():
    print(f"Generating {N_SAMPLES} samples (seed {DATA_SEED})...")
    inp, lbl = gen_single_step_data(N_SAMPLES, seed=DATA_SEED)
    inp_g = inp.to(DEVICE)
    target = lbl.numpy()
    n_train = int(N_SAMPLES * 0.7)
    perm = np.random.RandomState(0).permutation(N_SAMPLES)
    tr, te = perm[:n_train], perm[n_train:]

    all_results = {}

    for seed, ckpt_path in CHECKPOINTS:
        print(f"\n{'='*60}")
        print(f"  Tuned TF8L seed {seed}")
        print(f"  Checkpoint: {ckpt_path}")
        print(f"{'='*60}")

        if not os.path.exists(ckpt_path):
            print(f"  SKIP: not found")
            continue

        model = TF8LTuned().to(DEVICE)
        state = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state)
        model.eval()

        # Extract per-layer hidden states
        n_layers = 9  # 0=input, 1-8=layers
        layer_hidden = {i: [] for i in range(n_layers)}
        BATCH = 4096
        with torch.no_grad():
            for i in range(0, N_SAMPLES, BATCH):
                j = min(i + BATCH, N_SAMPLES)
                with torch.amp.autocast('cuda'):
                    outs = model.forward_with_layer_outputs(inp_g[i:j])
                for k, v in outs.items():
                    layer_hidden[k].append(v)
        layer_hidden = {k: np.concatenate(v, axis=0) for k, v in layer_hidden.items()}

        # Probe each layer
        layer_results = {}
        for layer_idx in range(n_layers):
            X = layer_hidden[layer_idx]
            svm_acc = probe_classification(X[tr], target[tr], X[te], target[te])
            ft_dist = compute_ft_distance(X, target)
            layer_results[layer_idx] = {'svm_acc': svm_acc, 'ft_distance': ft_dist}
            label = "input" if layer_idx == 0 else f"layer {layer_idx}"
            print(f"  {label:<10} SVM={svm_acc:.3f}  F-T={ft_dist:.3f}")

        first_ft = layer_results[0]['ft_distance']
        last_ft = layer_results[8]['ft_distance']
        sep_ratio = last_ft / max(first_ft, 1e-9)
        print(f"  Separation ratio: {sep_ratio:.1f}x")

        result = {
            'seed': seed,
            'checkpoint': ckpt_path,
            'n_samples': N_SAMPLES,
            'data_seed': DATA_SEED,
            'per_layer': {str(k): v for k, v in layer_results.items()},
            'separation_ratio': sep_ratio,
            'svm_layer0': layer_results[0]['svm_acc'],
            'svm_layer8': layer_results[8]['svm_acc'],
        }
        all_results[str(seed)] = result

        # Save per-seed
        out_dir = os.path.dirname(ckpt_path)
        out_path = os.path.join(out_dir, 'tuned_tf_layer_probing.json')
        with open(out_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"  Saved: {out_path}")

        del model
        torch.cuda.empty_cache()

    # Summary comparison
    print(f"\n{'='*60}")
    print(f"  COMPARISON: Matched vs Tuned TF8L")
    print(f"{'='*60}")
    print(f"  Matched (seed 42): Layer 0 SVM=0.751, Layer 8 SVM=0.999, sep=44x")
    for seed_str, r in all_results.items():
        print(f"  Tuned  (seed {seed_str}): Layer 0 SVM={r['svm_layer0']:.3f}, "
              f"Layer 8 SVM={r['svm_layer8']:.3f}, sep={r['separation_ratio']:.1f}x")

    # Save aggregate
    with open('tuned_tf_probing_aggregate.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: tuned_tf_probing_aggregate.json")


if __name__ == '__main__':
    main()
