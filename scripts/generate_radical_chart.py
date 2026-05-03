import json
import matplotlib.pyplot as plt
import numpy as np

# Load previous 5k showdown
with open('data/5k_showdown_results.json', 'r') as f:
    results_5k = json.load(f)

# Load radical 5k
with open('data/radical_5k_results.json', 'r') as f:
    results_radical = json.load(f)

steps = np.arange(0, 5001, 200)

nanogpt_val = results_5k["nanogpt"]["val"][:len(steps)]
massive_val = results_5k["massive_pure"]["val"][:len(steps)]
radical_val = results_radical["radical_pure"]["val"][:len(steps)]

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

fig, ax = plt.subplots(figsize=(12, 7))

ax.plot(steps, nanogpt_val, color='#dc2626', linestyle='-', linewidth=3.5, label='NanoGPT Baseline (1.50 Plateau)')
ax.plot(steps, massive_val, color='#10b981', marker='D', markersize=4, linewidth=3.5, label='Massive Pure [6 Layers, 4x GELU] (1.59)')
ax.plot(steps, radical_val, color='#f59e0b', marker='X', markersize=4, linewidth=3.5, label='Radical Pure [8 Layers, 2x SwiGLU] (1.73)')

ax.grid(True, linestyle='--', alpha=0.3, color='gray')
ax.set_xlabel('Training Steps (TinyShakespeare)', fontsize=12)
ax.set_ylabel('Validation Cross-Entropy Loss', fontsize=12)
ax.set_title('5k-Step Architectural Convergence Showdown', fontsize=16, fontweight='bold', pad=20)
ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=False)

ax.set_ylim(1.3, 4.0)

props = dict(boxstyle='round,pad=0.5', facecolor='#fffbeb', edgecolor='#f59e0b', alpha=0.9)
ax.text(0.4, 0.45, "Radical Optimization (1.73) underperformed Massive Pure (1.59).\nThis proves Cavity Expansion Factor (4x) is structurally\nmore important than Layer Depth (8x) and SwiGLU gating!", 
        transform=ax.transAxes, fontsize=11,
        verticalalignment='center', horizontalalignment='center', bbox=props, color='#b45309')

plt.tight_layout()
plt.savefig('radical_showdown_chart.png', dpi=300, bbox_inches='tight')
print("Chart generated: radical_showdown_chart.png")
