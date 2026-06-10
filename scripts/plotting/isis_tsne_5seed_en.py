"""
t-SNE and cosine-similarity figures for THEIA hidden states (paper figures).

Centroid distances and cosine similarities are computed in the original
128-dim representation space (not the 2-D t-SNE embedding, where Euclidean
distance is not meaningful) and reported as mean ± std across the 5 canonical
seeds; t-SNE is used only for the qualitative scatter, drawn from one
representative seed (default 42) with explicit disclosure in the figure title.

Inputs:  hidden_states_5seed.pt / hidden_labels_5seed.pt / hidden_meta_5seed.pt
         (run extract_hidden_states_5seed.py first)
Outputs: visualizations/tsne_4domains_5seed_en.png
         visualizations/cosine_similarity_5seed_en.png
"""
import os
import sys
import time

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
except ImportError:
    print('Missing scikit-learn'); sys.exit(1)


SAVE_DIR = 'visualizations'
DATA_DIR = '.'
REPRESENTATIVE_SEED = 42  # for t-SNE scatter (single seed for clarity)
os.makedirs(SAVE_DIR, exist_ok=True)


# ---------- Load 5-seed hidden states ----------
states_path = os.path.join(DATA_DIR, 'hidden_states_5seed.pt')
labels_path = os.path.join(DATA_DIR, 'hidden_labels_5seed.pt')
meta_path   = os.path.join(DATA_DIR, 'hidden_meta_5seed.pt')

for p in [states_path, labels_path, meta_path]:
    if not os.path.exists(p):
        print(f'Missing: {p}')
        print('Run extract_hidden_states_5seed.py first')
        sys.exit(1)

per_seed_hidden = torch.load(states_path, map_location='cpu', weights_only=False)
labels = torch.load(labels_path, map_location='cpu', weights_only=False).numpy()
meta = torch.load(meta_path, map_location='cpu', weights_only=False)

seeds = sorted(per_seed_hidden.keys())
n_total = len(labels)
print(f'Loaded {len(seeds)} seeds, {n_total} samples each')
print(f'  Labels: F={(labels==0).sum()} T={(labels==1).sum()} U={(labels==2).sum()}')
print(f'  Per-seed acc: {meta.get("per_seed_pred_acc", {})}')

domain_names = {
    'arith': 'Arithmetic (c vector)',
    'order': 'Order (c rel d vector)',
    'set':   'Set (c in S vector)',
    'logic': 'Logic (final verdict vector)',
}
boundaries = ['arith', 'order', 'set', 'logic']
colors = {0: '#FF4444', 1: '#44AA44', 2: '#4444FF'}
class_names = {0: 'False', 1: 'True', 2: 'Unknown'}


# ---------- Centroid distances in original 128-dim space, 5-seed aggregate ----------
def compute_centroids_128d(vecs, labels):
    """Returns dict cls -> centroid (128-dim) for cls ∈ {0,1,2}."""
    out = {}
    for c in [0, 1, 2]:
        mask = labels == c
        if mask.sum() > 0:
            out[c] = vecs[mask].mean(axis=0)
    return out


def euclid(c1, c2):
    return float(np.linalg.norm(c1 - c2))


def cosine(c1, c2):
    n1 = np.linalg.norm(c1) + 1e-12
    n2 = np.linalg.norm(c2) + 1e-12
    return float(np.dot(c1, c2) / (n1 * n2))


# Per-domain × per-pair statistics aggregated over 5 seeds
per_domain_stats = {}   # domain -> {pair: {'dist_mean':, 'dist_std':, 'cos_mean':, 'cos_std':}}
print('\nComputing per-seed centroid stats in 128-dim space...')
for domain in boundaries:
    pair_dists = {('F','T'): [], ('F','U'): [], ('T','U'): []}
    pair_coses = {('F','T'): [], ('F','U'): [], ('T','U'): []}
    cls_map = {0: 'F', 1: 'T', 2: 'U'}
    for seed in seeds:
        vecs = per_seed_hidden[seed][domain].float().numpy()
        cents = compute_centroids_128d(vecs, labels)
        for (a_cls, b_cls), key in [((0,1),('F','T')), ((0,2),('F','U')), ((1,2),('T','U'))]:
            if a_cls in cents and b_cls in cents:
                pair_dists[key].append(euclid(cents[a_cls], cents[b_cls]))
                pair_coses[key].append(cosine(cents[a_cls], cents[b_cls]))
    per_domain_stats[domain] = {
        'dist': {k: (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v)>1 else 0.0)
                 for k, v in pair_dists.items() if v},
        'cos':  {k: (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v)>1 else 0.0)
                 for k, v in pair_coses.items() if v},
    }
    print(f'  {domain}: F-T dist = {per_domain_stats[domain]["dist"][("F","T")][0]:.3f} ± '
          f'{per_domain_stats[domain]["dist"][("F","T")][1]:.3f}')


# ---------- t-SNE on representative seed ----------
def run_tsne(vectors_128d, perplexity=40, seed=42):
    """PCA(50) → t-SNE(2) with init='pca'. Returns (n, 2) embedding."""
    n = len(vectors_128d)
    if vectors_128d.shape[1] > 50:
        pca = PCA(n_components=min(50, n - 1))
        vectors_128d = pca.fit_transform(vectors_128d)
    tsne = TSNE(
        n_components=2,
        perplexity=min(perplexity, n // 4),
        max_iter=1000,
        random_state=seed,
        init='pca',     # pca init is more stable than 'random'
        verbose=0,
    )
    return tsne.fit_transform(vectors_128d)


print(f'\nRunning t-SNE on representative seed {REPRESENTATIVE_SEED} ...')
embeddings_rep = {}
for domain in boundaries:
    print(f'  {domain} ...', end=' ')
    t0 = time.time()
    vecs = per_seed_hidden[REPRESENTATIVE_SEED][domain].float().numpy()
    embeddings_rep[domain] = run_tsne(vecs)
    print(f'{time.time()-t0:.1f}s')


# ---------- Figure 1: 4-domain t-SNE 2x2 grid ----------
print('\nGenerating tsne_4domains_5seed_en.png ...')
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.patch.set_facecolor('#1a1a2e')
axes_flat = axes.flatten()

for ax, domain in zip(axes_flat, boundaries):
    emb = embeddings_rep[domain]
    ax.set_facecolor('#1a1a2e')
    for c in [0, 1, 2]:
        m = labels == c
        if m.sum() == 0: continue
        ax.scatter(emb[m, 0], emb[m, 1], c=colors[c],
                   label=class_names[c], alpha=0.6, s=8, linewidths=0)
    ax.set_title(domain_names[domain], color='white', fontsize=11, pad=4)

    # Centroid distances: 128-dim space, 5-seed mean±std (not t-SNE space)
    stats = per_domain_stats[domain]['dist']
    txt = ' | '.join(
        f'{a}-{b}: {stats[(a,b)][0]:.2f}±{stats[(a,b)][1]:.2f}'
        for (a, b) in [('F','T'), ('F','U'), ('T','U')]
        if (a, b) in stats
    )
    ax.text(0.02, 0.02, f'128-dim centroid distance (5-seed mean±std):\n{txt}',
            transform=ax.transAxes, color='#aaaacc', fontsize=8, va='bottom',
            bbox=dict(facecolor='#1a1a2e', alpha=0.6, edgecolor='#444466', pad=3))

    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('#444466')
    ax.legend(fontsize=8, labelcolor='white', facecolor='#2a2a4e', framealpha=0.3)

# Single-seed t-SNE disclosure in suptitle
plt.suptitle(
    f'THEIA: Hidden Representations by Processing Stage  '
    f'(t-SNE scatter from seed {REPRESENTATIVE_SEED}; centroid distances are '
    f'5-seed mean ± std in 128-dim space)',
    color='white', fontsize=12, y=1.005,
)
plt.tight_layout()
out1 = os.path.join(SAVE_DIR, 'tsne_4domains_5seed_en.png')
plt.savefig(out1, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print(f'  Saved: {out1}')


# ---------- Figure 2: cosine-similarity heatmaps ----------
print('\nGenerating cosine_similarity_5seed_en.png ...')
fig, axes = plt.subplots(1, len(boundaries), figsize=(5 * len(boundaries) + 1, 4.5))
fig.patch.set_facecolor('#1a1a2e')
if len(boundaries) == 1:
    axes = [axes]

# Display the 5-seed mean cosine-similarity matrix
ims = []
for ax, domain in zip(axes, boundaries):
    cls_map = {'F': 0, 'T': 1, 'U': 2}
    class_labels = ['F', 'T', 'U']
    K = len(class_labels)

    # Per-seed cosine matrix, then mean across seeds
    sim_per_seed = []
    for seed in seeds:
        vecs = per_seed_hidden[seed][domain].float().numpy()
        cents = compute_centroids_128d(vecs, labels)
        if not all(cls_map[c] in cents for c in class_labels):
            continue
        # Build (K, 128) centroid matrix, normalize, dot product
        cmat = np.stack([cents[cls_map[c]] for c in class_labels])
        cmat /= (np.linalg.norm(cmat, axis=1, keepdims=True) + 1e-12)
        sim_per_seed.append(cmat @ cmat.T)
    sim_mean = np.mean(np.stack(sim_per_seed), axis=0) if sim_per_seed else np.zeros((K, K))
    sim_std  = np.std(np.stack(sim_per_seed), axis=0, ddof=1) if len(sim_per_seed) > 1 else np.zeros((K, K))

    # Diverging cmap centered at 0 with symmetric range, so the "ideal"
    # value 0 (orthogonal centroids) renders near-white
    im = ax.imshow(sim_mean, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
    ims.append(im)

    ax.set_xticks(range(K))
    ax.set_yticks(range(K))
    ax.set_xticklabels(class_labels, color='white', fontsize=11)
    ax.set_yticklabels(class_labels, color='white', fontsize=11)
    ax.set_title(domain_names[domain], color='white', fontsize=10, pad=6)
    ax.set_facecolor('#1a1a2e')

    # Auto-contrast cell text; show per-cell std across seeds
    for i in range(K):
        for j in range(K):
            v = sim_mean[i, j]
            s = sim_std[i, j]
            # White text on dark cells (|v| > 0.5), black on near-white
            text_color = 'white' if abs(v) > 0.5 else 'black'
            cell_text = f'{v:+.2f}\n±{s:.2f}'
            ax.text(j, i, cell_text, ha='center', va='center',
                    color=text_color, fontsize=8.5, fontweight='bold')

# Single shared colorbar on the right
cbar = fig.colorbar(ims[-1], ax=axes, shrink=0.85, pad=0.02, orientation='vertical')
cbar.set_label('Cosine similarity (5-seed mean)', color='white', fontsize=10)
cbar.ax.tick_params(colors='white')
cbar.outline.set_edgecolor('#444466')

plt.suptitle(
    'Cosine Similarity Between Class Centroids by Domain  '
    '(0 = orthogonal = ideal class separability; '
    f'5-seed mean ± std across seeds {seeds})',
    color='white', fontsize=11, y=1.02,
)
out2 = os.path.join(SAVE_DIR, 'cosine_similarity_5seed_en.png')
plt.savefig(out2, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print(f'  Saved: {out2}')


# ---------- Console summary ----------
print('\n' + '='*72)
print('  Per-domain centroid distance + cosine (5-seed mean ± std, 128-dim)')
print('='*72)
print(f'  {"domain":<8} {"F-T dist":>16} {"F-U dist":>16} {"T-U dist":>16}')
for d in boundaries:
    s = per_domain_stats[d]['dist']
    row = ''
    for pair in [('F','T'), ('F','U'), ('T','U')]:
        m, sd = s.get(pair, (float('nan'), float('nan')))
        row += f' {m:>7.2f} ± {sd:.2f}'
    print(f'  {d:<8} {row}')
print()
print(f'  {"domain":<8} {"F-T cos":>16} {"F-U cos":>16} {"T-U cos":>16}')
for d in boundaries:
    s = per_domain_stats[d]['cos']
    row = ''
    for pair in [('F','T'), ('F','U'), ('T','U')]:
        m, sd = s.get(pair, (float('nan'), float('nan')))
        row += f' {m:>+7.3f} ± {sd:.3f}'
    print(f'  {d:<8} {row}')

print()
print(f'Outputs:')
print(f'  {out1}')
print(f'  {out2}')
