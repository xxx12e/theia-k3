#!/usr/bin/env python
"""
TF8L layer-wise linear probing (Transformer layer-wise probing table; §5.3).

Loads a TF8L checkpoint (same architecture as the §4.3 5-seed Kleene
comparison), mean-pools the 11-token sequence at the embedding input and at
each of the 8 TransformerEncoderLayer outputs, trains LinearSVC (3-class final
verdict) per layer, and reports per-layer SVM accuracy, F-T centroid distance,
and the separation ratio (last layer / input).

Usage:
    python probe_tf8l_layers.py --checkpoint multi_seed_results\\tf8l\\seed_42\\checkpoint.pth
"""
import argparse, json, os, time
import torch, torch.nn as nn
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler

p = argparse.ArgumentParser()
p.add_argument('--checkpoint', type=str, required=True)
p.add_argument('--n-samples', type=int, default=50000)
p.add_argument('--data-seed', type=int, default=999)
args = p.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
D_MODEL=192; NUM_RANGE=20; SET_DIM=21; P_UNKNOWN=0.15
N_VALS=3; N_RELS=6; N_ARITH=4; N_OPS=5
VAL_FALSE=0; VAL_TRUE=1; VAL_UNKNOWN=2
ARITH_ADD=0; ARITH_SUB=1; ARITH_MUL=2; ARITH_MOD=3
OP_AND=0; OP_OR=1; OP_NOT=2; OP_IMPLIES=3; OP_IFF=4

AND_T=torch.tensor([[0,0,0],[0,1,2],[0,2,2]],dtype=torch.long,device=DEVICE)
OR_T =torch.tensor([[0,1,2],[1,1,1],[2,1,2]],dtype=torch.long,device=DEVICE)
IMP_T=torch.tensor([[1,1,1],[0,1,2],[2,1,2]],dtype=torch.long,device=DEVICE)
IFF_T=torch.tensor([[1,0,2],[0,1,2],[2,2,2]],dtype=torch.long,device=DEVICE)
NOT_T=torch.tensor([1,0,2],dtype=torch.long,device=DEVICE)

def apply_logic(op, va, vb):
    r = torch.zeros_like(op)
    m=op==OP_AND;     r[m]=AND_T[va[m],vb[m]]
    m=op==OP_OR;      r[m]=OR_T[va[m],vb[m]]
    m=op==OP_IMPLIES; r[m]=IMP_T[va[m],vb[m]]
    m=op==OP_IFF;     r[m]=IFF_T[va[m],vb[m]]
    m=op==OP_NOT;     r[m]=NOT_T[va[m]]
    return r

# === Model (verbatim from tf8l_5seed.py) ===
class BigTransformer(nn.Module):
    def __init__(self, d=D_MODEL, nhead=8, nlayers=8):
        super().__init__()
        self.num_enc=nn.Sequential(nn.Linear(1,d//2),nn.GELU(),nn.Linear(d//2,d))
        self.set_enc=nn.Sequential(nn.Linear(SET_DIM,d//2),nn.GELU(),nn.Linear(d//2,d))
        self.arith_emb=nn.Embedding(N_ARITH,d)
        self.rel_emb=nn.Embedding(N_RELS,d)
        self.op_emb=nn.Embedding(N_OPS,d)
        self.unk_emb=nn.Embedding(2,d)
        self.type_emb=nn.Embedding(11,d)
        enc=nn.TransformerEncoderLayer(d_model=d,nhead=nhead,dim_feedforward=d*4,
                                       dropout=0.1,activation='gelu',batch_first=True)
        self.transformer=nn.TransformerEncoder(enc,num_layers=nlayers)
        self.head=nn.Sequential(nn.Linear(d,d),nn.GELU(),nn.LayerNorm(d),nn.Linear(d,N_VALS))

    def build_tokens(self, a,b,d,sb,su,au,bu,du,ar,rl,op):
        B=a.shape[0]
        toks=torch.stack([
            self.num_enc(a.unsqueeze(-1)),self.num_enc(b.unsqueeze(-1)),self.num_enc(d.unsqueeze(-1)),
            self.arith_emb(ar),self.rel_emb(rl),self.op_emb(op),
            self.unk_emb(au.long()),self.unk_emb(bu.long()),self.unk_emb(du.long()),
            self.unk_emb(su.long()),self.set_enc(sb)
        ],dim=1)
        tids=torch.arange(11,device=DEVICE).unsqueeze(0).expand(B,-1)
        toks=toks+self.type_emb(tids)
        return toks

    def forward_with_layer_outputs(self, a,b,d,sb,su,au,bu,du,ar,rl,op):
        """Returns dict with mean-pooled hidden state at each layer.
        Layer 0 = input embeddings (post type_emb, pre transformer)
        Layers 1-8 = output of each TransformerEncoderLayer
        """
        toks = self.build_tokens(a,b,d,sb,su,au,bu,du,ar,rl,op)
        outputs = {0: toks.mean(dim=1).float().cpu().numpy()}
        x = toks
        for i, layer in enumerate(self.transformer.layers):
            x = layer(x)
            outputs[i+1] = x.mean(dim=1).float().cpu().numpy()
        return outputs

# === Data generation (same as tf8l_5seed.py) ===
def gen_data(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    n = args.n_samples
    a = torch.randint(1, NUM_RANGE+1, (n,), device=DEVICE)
    b = torch.randint(1, NUM_RANGE+1, (n,), device=DEVICE)
    d = torch.randint(0, NUM_RANGE+1, (n,), device=DEVICE)
    ar = torch.randint(0, N_ARITH, (n,), device=DEVICE)
    rl = torch.randint(0, N_RELS,  (n,), device=DEVICE)
    op = torch.randint(0, N_OPS,   (n,), device=DEVICE)
    au = torch.rand(n, device=DEVICE) < P_UNKNOWN
    bu = torch.rand(n, device=DEVICE) < P_UNKNOWN
    du = torch.rand(n, device=DEVICE) < P_UNKNOWN
    su = torch.rand(n, device=DEVICE) < P_UNKNOWN
    c = torch.zeros(n, dtype=torch.long, device=DEVICE)
    c[ar==0] = torch.clamp(a+b,0,NUM_RANGE)[ar==0]
    c[ar==1] = torch.abs(a-b)[ar==1]
    c[ar==2] = torch.clamp(a*b,0,NUM_RANGE)[ar==2]
    c[ar==3] = (a % torch.clamp(b,1,NUM_RANGE))[ar==3]
    c = torch.clamp(c,0,NUM_RANGE)
    cu = au | bu
    ou = cu | du
    ov = torch.zeros(n, dtype=torch.long, device=DEVICE)
    rt = (((rl==0)&(c>d))|((rl==1)&(c<d))|((rl==2)&(c==d))|
          ((rl==3)&(c>=d))|((rl==4)&(c<=d))|((rl==5)&(c!=d)))
    ov[rt] = 1
    val_o = torch.where(ou, torch.tensor(2,device=DEVICE), ov)
    sb = torch.randint(0, 2, (n,SET_DIM), dtype=torch.float32, device=DEVICE)
    sou = su | cu
    ci = c.clamp(0, SET_DIM-1)
    ins = sb[torch.arange(n,device=DEVICE), ci].bool()
    sv = torch.where(ins, torch.tensor(1,device=DEVICE), torch.tensor(0,device=DEVICE))
    val_s = torch.where(sou, torch.tensor(2,device=DEVICE), sv)
    target = apply_logic(op, val_o, val_s)
    return {
        'a_norm': a.float()/NUM_RANGE, 'b_norm': b.float()/NUM_RANGE, 'd_norm': d.float()/NUM_RANGE,
        'sb': sb, 's_unk': su, 'a_unk': au, 'b_unk': bu, 'd_unk': du,
        'arith': ar, 'rel': rl, 'op': op,
        'target': target,
    }

def probe_classification(X_tr, y_tr, X_te, y_te):
    sc = StandardScaler()
    Xs_tr = sc.fit_transform(X_tr); Xs_te = sc.transform(X_te)
    dual = X_tr.shape[0] < X_tr.shape[1]
    clf = LinearSVC(C=1.0, max_iter=2000, dual=dual)
    clf.fit(Xs_tr, y_tr)
    return clf.score(Xs_te, y_te)

def compute_ft_distance(X, labels):
    f_mask = labels == VAL_FALSE
    t_mask = labels == VAL_TRUE
    if f_mask.sum() == 0 or t_mask.sum() == 0:
        return None
    f_centroid = X[f_mask].mean(axis=0)
    t_centroid = X[t_mask].mean(axis=0)
    return float(np.linalg.norm(f_centroid - t_centroid))

def main():
    print(f"Loading: {args.checkpoint}")
    print(f"Device: {DEVICE}")

    model = BigTransformer().to(DEVICE)
    state = torch.load(args.checkpoint, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    params = sum(p.numel() for p in model.parameters())
    print(f"Params: {params:,}")

    print(f"\nGenerating {args.n_samples} test samples (seed {args.data_seed})...")
    data = gen_data(args.data_seed)

    print("Forward pass with per-layer extraction...")
    BATCH = 4096
    n_layers = 9  # 0 = input embeddings, 1-8 = each transformer layer
    layer_hidden = {i: [] for i in range(n_layers)}
    with torch.no_grad():
        for i in range(0, args.n_samples, BATCH):
            j = min(i + BATCH, args.n_samples)
            with torch.amp.autocast('cuda'):
                outs = model.forward_with_layer_outputs(
                    data['a_norm'][i:j], data['b_norm'][i:j], data['d_norm'][i:j],
                    data['sb'][i:j], data['s_unk'][i:j],
                    data['a_unk'][i:j], data['b_unk'][i:j], data['d_unk'][i:j],
                    data['arith'][i:j], data['rel'][i:j], data['op'][i:j],
                )
            for k, v in outs.items():
                layer_hidden[k].append(v)
    layer_hidden = {k: np.concatenate(v, axis=0) for k, v in layer_hidden.items()}
    print(f"Per-layer shapes: layer 0 = {layer_hidden[0].shape}, layer 8 = {layer_hidden[8].shape}")

    target = data['target'].cpu().numpy()
    n = args.n_samples
    n_train = int(n * 0.7)
    perm = np.random.RandomState(0).permutation(n)
    tr, te = perm[:n_train], perm[n_train:]

    print("\nRunning per-layer probes...")
    layer_results = {}
    t0 = time.time()
    for layer_idx in range(n_layers):
        X = layer_hidden[layer_idx]
        X_tr, X_te = X[tr], X[te]
        svm_acc = probe_classification(X_tr, target[tr], X_te, target[te])
        ft_dist = compute_ft_distance(X, target)
        layer_results[layer_idx] = {'svm_acc': svm_acc, 'ft_distance': ft_dist}
        print(f"  Layer {layer_idx}: SVM acc = {svm_acc:.3f}, F-T dist = {ft_dist:.3f}")

    elapsed = time.time() - t0
    print(f"\nAll probes done in {elapsed:.1f}s")

    # Separation ratio: last layer F-T / first layer F-T
    first_ft = layer_results[0]['ft_distance']
    last_ft = layer_results[8]['ft_distance']
    sep_ratio = last_ft / max(first_ft, 1e-9)

    # === Print Table ===
    print(f"\n{'='*60}")
    print(f"  TF8L Layer-wise Probing")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"{'='*60}")
    print(f"{'Layer':<10} {'SVM acc':>12} {'F-T dist':>12}")
    print('-' * 36)
    for layer_idx in range(n_layers):
        r = layer_results[layer_idx]
        label = "input" if layer_idx == 0 else f"layer {layer_idx}"
        print(f"{label:<10} {r['svm_acc']:>12.3f} {r['ft_distance']:>12.3f}")
    print('-' * 36)
    print(f"\nSeparation ratio (layer 8 / input): {sep_ratio:.1f}x")
    print(f"  (Paper claim for TF4L baseline: 54x)")
    print(f"  (Paper claim for THEIA: 1331x)")

    print(f"\nKey numbers for §5.3 update:")
    print(f"  - Layer 1 SVM accuracy: {layer_results[1]['svm_acc']:.1%}")
    print(f"  - Final layer SVM accuracy: {layer_results[8]['svm_acc']:.1%}")
    print(f"  - F-T separation ratio: {sep_ratio:.0f}x")

    out = {
        'checkpoint': args.checkpoint,
        'architecture': 'TF8L (8 layers, 8 heads, d=192)',
        'params': params,
        'n_samples': args.n_samples,
        'data_seed': args.data_seed,
        'per_layer': {str(k): v for k, v in layer_results.items()},
        'separation_ratio_last_over_input': sep_ratio,
        'svm_at_layer_1': layer_results[1]['svm_acc'],
        'svm_at_final_layer': layer_results[8]['svm_acc'],
    }
    out_dir = os.path.dirname(args.checkpoint)
    out_path = os.path.join(out_dir, 'tf8l_layer_probing.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")

if __name__ == '__main__':
    main()
