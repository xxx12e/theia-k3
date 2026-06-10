"""
Per-rule Kleene accuracy bar chart: THEIA (12/12) vs Transformer 8L8H (7/12)
vs flat MLP (6/12). Per-rule values are hardcoded from the result tables.

Output: visualizations/kleene_comparison_en.png
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

labels = ['F∧U→F','T∧U→U','U∧F→F','U∧T→U','T∨U→T','F∨U→U','U∨T→T','U∨F→U','F→U→T','T→U→U','T↔U→U','F↔U→U']
theia = [100,100,100,100,100,100,100,100,100,100,100,100]
tf8l =  [100,94.1,0,100,93.5,100,0,100,100,94.4,100,100]
mlp =   [98.9,93.3,0,100,93.3,99.3,0,100,99.5,92.7,100,100]

x = np.arange(len(labels))
w = 0.25

fig, ax = plt.subplots(figsize=(12, 4.5))

ax.bar(x - w, theia, w, label='THEIA (12/12)', color='#1D9E75')
ax.bar(x,     tf8l,  w, label='Transformer 8L8H (7/12)', color='#378ADD')
ax.bar(x + w, mlp,   w, label='Flat MLP (6/12)', color='#888780')

ax.set_ylabel('Accuracy (%)', fontsize=11)
ax.set_ylim(0, 112)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=8.5)
ax.legend(fontsize=10, loc='upper right')
ax.grid(axis='y', alpha=0.15)

for i in [2, 6]:
    ax.annotate('0%', xy=(i, 3), fontsize=9, ha='center', color='#A32D2D', fontweight='bold')

plt.tight_layout()
plt.savefig('visualizations/kleene_comparison_en.png', dpi=200, bbox_inches='tight')
print("Saved: kleene_comparison_en.png")
