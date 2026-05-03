import json
import matplotlib.pyplot as plt
import numpy as np

# Load 5k showdown
with open('data/5k_showdown_results.json', 'r') as f:
    results_5k = json.load(f)

# Load Einstein
with open('data/einstein_5k_results.json', 'r') as f:
    results_einstein = json.load(f)

steps = np.arange(0, 5001, 200)

nanogpt_val = results_5k["nanogpt"]["val"][:len(steps)]
massive_val = results_5k["massive_pure"]["val"][:len(steps)]
einstein_val = results_einstein["einstein_pure"]["val"][:len(steps)]

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

fig, ax = plt.subplots(figsize=(12, 7))

ax.plot(steps, nanogpt_val, color='#dc2626', linestyle='-', linewidth=3.5, label='NanoGPT Baseline (1.50 Plateau)')
ax.plot(steps, massive_val, color='#10b981', marker='D', markersize=4, linewidth=3.5, label='Massive Pure [Symmetric Init, No Gating] (1.59)')
ax.plot(steps, einstein_val, color='#8b5cf6', marker='*', markersize=5, linewidth=3.5, label='Einstein Pure [Fourier Init, Input Gating] (1.71)')

ax.grid(True, linestyle='--', alpha=0.3, color='gray')
ax.set_xlabel('Training Steps (TinyShakespeare)', fontsize=12)
ax.set_ylabel('Validation Cross-Entropy Loss', fontsize=12)
ax.set_title('5k-Step Architectural Convergence Showdown', fontsize=16, fontweight='bold', pad=20)
ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=False)

ax.set_ylim(1.3, 4.0)

props = dict(boxstyle='round,pad=0.5', facecolor='#f3f4f6', edgecolor='#8b5cf6', alpha=0.9)
ax.text(0.4, 0.45, "Einstein Optimization (1.71) overfit to the training set (Train: 1.43).\nHuman-engineered Fourier Initializations restricted co-adaptation!\nMassive Pure's symmetric raw initialization is mathematically superior.", 
        transform=ax.transAxes, fontsize=11,
        verticalalignment='center', horizontalalignment='center', bbox=props, color='#4c1d95')

plt.tight_layout()
plt.savefig('einstein_showdown_chart.png', dpi=300, bbox_inches='tight')
print("Chart generated: einstein_showdown_chart.png")
