"""
Activation patching for THEIA (paper §5.3 / Appendix E): causal test that the
Logic Engine's delayed verdict is driven by the Order Engine output (ord_vec).

Matched pair construction (T ∨ F  vs  U ∨ F):
    Shared:  a, b, arith=ADD, set_bits, s_unk=F, logic_op=OR,
             a_unk=F, b_unk=F, rel=REL_GTE
    T-side:  d = max(0, c-1), d_unk=False  →  val_ord = True     →  T ∨ F = T
    U-side:  d = 0 (overridden), d_unk=True →  val_ord = Unknown →  U ∨ F = U
set_bits is selected so c ∉ S (val_set = False on both sides), hence c_vec,
bridge_ao(c_vec), bridge_as(c_vec), and set_vec are identical by construction;
only ord_vec differs.

Patch: out_head(logic_eng(ord_vec_U, set_vec_T, OR)). A T→U prediction flip
indicates the verdict is causally driven by ord_vec rather than a shortcut.

Usage:
    python activation_patching.py --checkpoint ckpt_seed42.pth --n-pairs 1000
    python activation_patching.py --checkpoint ckpt_seed42.pth --debug
"""

import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# --- Model architecture (self-contained, matches theia_5seed.py) ---
D_MODEL = 128
N_ARITH = 4; N_RELS = 6; N_OPS = 5; N_VALS = 3; SET_DIM_M = 21

class NumEnc(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(1,D_MODEL//2),nn.GELU(),nn.Linear(D_MODEL//2,D_MODEL))
        self.unknown_vec = nn.Parameter(torch.randn(D_MODEL))
    def forward(self, x, is_unknown):
        v = self.f(x.unsqueeze(-1))
        unk = self.unknown_vec.unsqueeze(0).expand_as(v)
        return torch.where(is_unknown.unsqueeze(-1), unk, v)

class SetEnc(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(SET_DIM_M,D_MODEL//2),nn.GELU(),nn.Linear(D_MODEL//2,D_MODEL))
        self.unknown_vec = nn.Parameter(torch.randn(D_MODEL))
    def forward(self, bits, is_unknown):
        v = self.f(bits)
        unk = self.unknown_vec.unsqueeze(0).expand_as(v)
        return torch.where(is_unknown.unsqueeze(-1), unk, v)

def make_mlp(in_d, out_d, dropout=0.0):
    layers = [nn.Linear(in_d,in_d*2),nn.GELU(),nn.LayerNorm(in_d*2),nn.Linear(in_d*2,out_d)]
    if dropout > 0: layers.insert(3, nn.Dropout(dropout))
    return nn.Sequential(*layers)

class ArithEngine(nn.Module):
    def __init__(self):
        super().__init__()
        self.ne = NumEnc(); self.ae = nn.Embedding(N_ARITH, D_MODEL)
        self.net = make_mlp(D_MODEL*3, D_MODEL)
    def forward(self, a, b, a_unk, b_unk, arith):
        return self.net(torch.cat([self.ne(a,a_unk), self.ne(b,b_unk), self.ae(arith)], dim=-1))

class OrderEngine(nn.Module):
    def __init__(self):
        super().__init__()
        self.ne = NumEnc(); self.re = nn.Embedding(N_RELS, D_MODEL)
        mk = lambda: nn.Sequential(nn.Linear(D_MODEL*3,D_MODEL*2),nn.GELU(),
                                   nn.LayerNorm(D_MODEL*2),nn.Linear(D_MODEL*2,D_MODEL))
        self.G=mk(); self.L=mk(); self.E=mk()
        self.Gg=nn.Sequential(nn.Linear(D_MODEL*2,D_MODEL),nn.GELU(),nn.LayerNorm(D_MODEL))
        self.Lg=nn.Sequential(nn.Linear(D_MODEL*2,D_MODEL),nn.GELU(),nn.LayerNorm(D_MODEL))
        self.Eg=nn.Sequential(nn.Linear(D_MODEL*3,D_MODEL),nn.GELU(),nn.LayerNorm(D_MODEL))
        self.out = make_mlp(D_MODEL*4, D_MODEL)
    def forward(self, c_vec, d, d_unk, rel):
        vd = self.ne(d, d_unk); vr = self.re(rel)
        x = torch.cat([c_vec,vd,vr],dim=-1)
        g=self.G(x); l=self.L(x); e=self.E(x)
        g=self.Gg(torch.cat([g,e],dim=-1))
        l=self.Lg(torch.cat([l,e],dim=-1))
        e=self.Eg(torch.cat([e,g,l],dim=-1))
        return self.out(torch.cat([g,l,e,c_vec], dim=-1))

class SetEngine(nn.Module):
    def __init__(self):
        super().__init__()
        self.se = SetEnc(); self.net = make_mlp(D_MODEL*2, D_MODEL)
    def forward(self, c_vec, set_bits, s_unk):
        return self.net(torch.cat([c_vec, self.se(set_bits,s_unk)], dim=-1))

class LogicEngine(nn.Module):
    def __init__(self):
        super().__init__()
        self.oe = nn.Embedding(N_OPS, D_MODEL)
        mk = lambda: nn.Sequential(nn.Linear(D_MODEL*3,D_MODEL*2),nn.GELU(),
                                   nn.LayerNorm(D_MODEL*2),nn.Linear(D_MODEL*2,D_MODEL))
        self.C=mk(); self.D=mk(); self.I=mk()
        self.Cg=nn.Sequential(nn.Linear(D_MODEL*2,D_MODEL),nn.GELU(),nn.LayerNorm(D_MODEL))
        self.Dg=nn.Sequential(nn.Linear(D_MODEL*2,D_MODEL),nn.GELU(),nn.LayerNorm(D_MODEL))
        self.Ig=nn.Sequential(nn.Linear(D_MODEL*3,D_MODEL),nn.GELU(),nn.LayerNorm(D_MODEL))
        self.out = make_mlp(D_MODEL*3, D_MODEL)
    def forward(self, v_ord, v_set, op):
        vo = self.oe(op)
        x = torch.cat([v_ord,v_set,vo], dim=-1)
        c=self.C(x); d=self.D(x); i=self.I(x)
        c=self.Cg(torch.cat([c,i],dim=-1))
        d=self.Dg(torch.cat([d,i],dim=-1))
        i=self.Ig(torch.cat([i,c,d],dim=-1))
        return self.out(torch.cat([c,d,i], dim=-1))

class IsisV9(nn.Module):
    def __init__(self):
        super().__init__()
        self.arith_eng = ArithEngine()
        self.order_eng = OrderEngine()
        self.set_eng = SetEngine()
        self.logic_eng = LogicEngine()
        self.bridge_ao = nn.Sequential(nn.Linear(D_MODEL,D_MODEL),nn.GELU(),nn.LayerNorm(D_MODEL))
        self.bridge_as = nn.Sequential(nn.Linear(D_MODEL,D_MODEL),nn.GELU(),nn.LayerNorm(D_MODEL))
        self.out_head = nn.Sequential(nn.Linear(D_MODEL,D_MODEL),nn.GELU(),nn.Dropout(0.1),nn.LayerNorm(D_MODEL))
        self.sv = nn.Embedding(N_VALS, D_MODEL)
        nn.init.orthogonal_(self.sv.weight)

def _build_model(device):
    return IsisV9().to(device)

# --- Encoding constants ---
# These indices must match the data generator used during training
# (theia_5seed.py / theia_5seed_v2.py); wrong prediction baselines usually
# indicate a mismatch here.

ARITH_ADD = 0      # arith_op index for addition
REL_GTE   = 3      # rel index for >= (verified: REL_GT=0, REL_LT=1, REL_EQ=2, REL_GTE=3, REL_LTE=4, REL_NEQ=5)
LOGIC_OR  = 1      # logic_op index for OR

# Prototype order from theia_5seed.py: sv = Embedding(3, 128)
# sv.weight[0] = False, [1] = True, [2] = Unknown
PROTO_FALSE   = 0
PROTO_TRUE    = 1
PROTO_UNKNOWN = 2

NUM_RANGE = 20   # training range [0, 20], paper §4.1
SET_DIM   = 21   # set_bits dimensionality, paper §3.2


def classify(o_vec, sv_weight):
    """
    Cosine-similarity classification (matches the inference path).
    Training uses an unnormalized dot product; inference (paper §3.3)
    uses cosine similarity.
    """
    sn = F.normalize(sv_weight, dim=-1)   # (3, 128)
    vn = F.normalize(o_vec, dim=-1)       # (B, 128)
    return (vn @ sn.T).argmax(dim=-1)     # (B,)


def build_matched_pair(device):
    """
    Build one (T-side, U-side) matched pair for T∨F vs U∨F.

    Returns:
        T_inputs: dict of tensors (batch=1)
        U_inputs: dict of tensors (batch=1)
        c_val:    scalar c = a + b (for debugging/sanity checks)
    """
    # Sample a, b so that c = a + b is well-defined and stays in range
    a_val = random.randint(1, NUM_RANGE - 2)
    b_val = random.randint(0, NUM_RANGE - a_val - 1)
    c_val = a_val + b_val                         # c ∈ [1, 19]

    # Build set_bits so that bit[c] = 0  →  c ∉ S  →  val_set = False
    set_bits = torch.zeros(SET_DIM)
    # Put random non-c elements in S (at least 1, at most 5) to avoid empty set
    non_c_indices = [i for i in range(SET_DIM) if i != c_val]
    k = random.randint(1, 5)
    for i in random.sample(non_c_indices, k=k):
        set_bits[i] = 1.0

    # T-side: d = max(0, c-1), d_unk=False, rel=REL_GTE  →  c >= d always  →  val_ord = True
    d_T = max(0, c_val - 1)
    # U-side: d_unk=True (value overridden by learnable sentinel)
    d_U = 0

    def pack(d_val, d_unk_flag):
        return dict(
            a=torch.tensor([a_val / NUM_RANGE], device=device, dtype=torch.float32),
            b=torch.tensor([b_val / NUM_RANGE], device=device, dtype=torch.float32),
            d=torch.tensor([d_val / NUM_RANGE], device=device, dtype=torch.float32),
            set_bits=set_bits.unsqueeze(0).to(device),       # (1, 21)
            s_unk=torch.tensor([False], device=device),
            a_unk=torch.tensor([False], device=device),
            b_unk=torch.tensor([False], device=device),
            d_unk=torch.tensor([d_unk_flag], device=device),
            arith=torch.tensor([ARITH_ADD], device=device, dtype=torch.long),
            rel=torch.tensor([REL_GTE],   device=device, dtype=torch.long),
            op=torch.tensor([LOGIC_OR],   device=device, dtype=torch.long),
        )

    return pack(d_T, False), pack(d_U, True), c_val


@torch.no_grad()
def run_patching(model, n_pairs, device, seed, debug=False):
    random.seed(seed)
    torch.manual_seed(seed)
    model.eval()

    stats = dict(
        total=0,
        baseline_T_correct=0,
        baseline_U_correct=0,
        both_baseline_correct=0,
        set_vec_identical=0,
        patch_flipped_to_U=0,
        patch_stayed_T=0,
        patch_other_F=0,
    )

    for i in range(n_pairs):
        T_inputs, U_inputs, c_val = build_matched_pair(device)

        H_T = model.forward_with_hidden(**T_inputs)
        H_U = model.forward_with_hidden(**U_inputs)

        # Baseline predictions via the inference (cosine) path
        o_T = model.out_head(H_T['logic'])
        o_U = model.out_head(H_U['logic'])
        pred_T = classify(o_T, model.sv.weight).item()
        pred_U = classify(o_U, model.sv.weight).item()

        stats['total'] += 1
        T_ok = (pred_T == PROTO_TRUE)
        U_ok = (pred_U == PROTO_UNKNOWN)
        stats['baseline_T_correct'] += int(T_ok)
        stats['baseline_U_correct'] += int(U_ok)

        if debug and i < 5:
            print(f"[pair {i}] c={c_val}  T-pred={pred_T} (want {PROTO_TRUE})  "
                  f"U-pred={pred_U} (want {PROTO_UNKNOWN})  T_ok={T_ok}  U_ok={U_ok}")

        if not (T_ok and U_ok):
            continue
        stats['both_baseline_correct'] += 1

        # Sanity: set_vec should be identical across sides by construction
        set_identical = torch.allclose(H_T['set'], H_U['set'], atol=1e-6)
        stats['set_vec_identical'] += int(set_identical)

        # PATCH: feed ord_vec from U-side into the T-side logic_eng call
        logic_patched = model.logic_eng(H_U['order'], H_T['set'], T_inputs['op'])
        o_patched = model.out_head(logic_patched)
        pred_patched = classify(o_patched, model.sv.weight).item()

        if pred_patched == PROTO_UNKNOWN:
            stats['patch_flipped_to_U'] += 1
        elif pred_patched == PROTO_TRUE:
            stats['patch_stayed_T'] += 1
        else:
            stats['patch_other_F'] += 1

        if debug and i < 5:
            print(f"           set_vec_identical={set_identical}  "
                  f"patched_pred={pred_patched}")

    return stats


def print_results(stats):
    print("=" * 64)
    print("Activation Patching Results")
    print("=" * 64)
    n_tot = stats['total']
    print(f"Total pairs constructed:    {n_tot}")
    if n_tot == 0:
        print("No pairs — aborting.")
        return
    print(f"Baseline T-side correct:    {stats['baseline_T_correct']}/{n_tot} "
          f"({100*stats['baseline_T_correct']/n_tot:.1f}%)")
    print(f"Baseline U-side correct:    {stats['baseline_U_correct']}/{n_tot} "
          f"({100*stats['baseline_U_correct']/n_tot:.1f}%)")
    print(f"Both baselines correct:     {stats['both_baseline_correct']}/{n_tot}")

    n = stats['both_baseline_correct']
    if n == 0:
        print()
        print("ZERO usable pairs. Likely causes:")
        print("  (a) ARITH_ADD / REL_GTE / LOGIC_OR constants wrong — check data generator")
        print("  (b) Checkpoint loading failed or model is not trained")
        print("  (c) set_bits construction doesn't match training distribution")
        return

    print(f"  set_vec identical check:  {stats['set_vec_identical']}/{n} "
          f"({100*stats['set_vec_identical']/n:.2f}%)")
    print()
    print(f"PATCH (ord_vec_U → T-side logic_eng, predict on patched output):")
    flip = stats['patch_flipped_to_U']
    stay = stats['patch_stayed_T']
    other = stats['patch_other_F']
    print(f"  Flipped to U (causal):    {flip}/{n}  ({100*flip/n:.2f}%)   ← MAIN RESULT")
    print(f"  Stayed at T:              {stay}/{n}  ({100*stay/n:.2f}%)")
    print(f"  Other (F):                {other}/{n}  ({100*other/n:.2f}%)")
    print("=" * 64)
    flip_rate = 100 * flip / n
    if flip_rate >= 95:
        verdict = "STRONG causal support — delayed verdict is causally driven by ord_vec."
    elif flip_rate >= 80:
        verdict = f"PARTIAL support — {flip_rate:.1f}% flip rate; residual pathway exists."
    else:
        verdict = f"WEAK/FAILED — flip rate {flip_rate:.1f}% below 80%; Logic Engine uses other shortcuts."
    print(f"VERDICT: {verdict}")
    print("=" * 64)


def _forward_with_hidden(self, a, b, d, set_bits, s_unk, a_unk, b_unk, d_unk, arith, rel, op):
    """Monkey-patched forward that returns per-engine hidden states."""
    c_vec = self.arith_eng(a, b, a_unk, b_unk, arith)
    c_for_ord = self.bridge_ao(c_vec) + c_vec
    c_for_set = self.bridge_as(c_vec) + c_vec
    ord_vec = self.order_eng(c_for_ord, d, d_unk, rel)
    set_vec = self.set_eng(c_for_set, set_bits, s_unk)
    logic_vec = self.logic_eng(ord_vec, set_vec, op)
    return {'arith': c_vec, 'order': ord_vec, 'set': set_vec, 'logic': logic_vec}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, help='Path to trained THEIA checkpoint (.pth)')
    parser.add_argument('--n-pairs', type=int, default=1000)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--debug', action='store_true', help='Print first 5 pairs verbosely')
    args = parser.parse_args()

    model = _build_model(args.device)
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    elif isinstance(ckpt, dict) and 'state_dict' in ckpt:
        model.load_state_dict(ckpt['state_dict'])
    else:
        model.load_state_dict(ckpt)

    import types
    model.forward_with_hidden = types.MethodType(_forward_with_hidden, model)

    stats = run_patching(model, args.n_pairs, args.device, args.seed, debug=args.debug)
    print_results(stats)


if __name__ == '__main__':
    main()
