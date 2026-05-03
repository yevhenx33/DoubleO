import json
import matplotlib.pyplot as plt
import numpy as np

with open('data/5k_showdown_results.json', 'r') as f:
    results = json.load(f)

steps = np.arange(0, 5001, 200)
# ensure lengths match
nanogpt_val = results["nanogpt"]["val"][:len(steps)]
massive_val = results["massive_pure"]["val"][:len(steps)]

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

fig, ax = plt.subplots(figsize=(12, 7))

ax.plot(steps, nanogpt_val, color='#dc2626', linestyle='-', linewidth=3.5, label='NanoGPT Baseline (~834k)')
ax.plot(steps, massive_val, color='#10b981', marker='D', markersize=4, linewidth=3.5, label='DoubleO: Massive Pure (~897k)')

ax.grid(True, linestyle='--', alpha=0.3, color='gray')
ax.set_xlabel('Training Steps (TinyShakespeare)', fontsize=12)
ax.set_ylabel('Validation Cross-Entropy Loss', fontsize=12)
ax.set_title('5k-Step Convergence Showdown:\nTransformer vs Pure Dynamical Systems', fontsize=16, fontweight='bold', pad=20)
ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=False)

ax.set_ylim(1.3, 4.0)

props = dict(boxstyle='round,pad=0.5', facecolor='#f0fdf4', edgecolor='#10b981', alpha=0.9)
ax.text(0.4, 0.45, "NanoGPT plateaus at ~1.50 early on.\nMassive Pure starts slower but continuously converges,\nreaching 1.59 without ANY Attention mechanisms!", 
        transform=ax.transAxes, fontsize=11,
        verticalalignment='center', horizontalalignment='center', bbox=props, color='#065f46')

plt.tight_layout()
plt.savefig('5k_showdown_chart.png', dpi=300, bbox_inches='tight')
print("Chart generated: 5k_showdown_chart.png")
