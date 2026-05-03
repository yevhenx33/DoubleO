import json
import matplotlib.pyplot as plt
import numpy as np

with open('data/5k_showdown_results.json', 'r') as f:
    results_5k = json.load(f)

with open('data/orthogonal_5k_results.json', 'r') as f:
    results_ortho = json.load(f)

steps = np.arange(0, 5001, 200)

nano_train = results_5k["nanogpt"]["train"][:len(steps)]
nano_val = results_5k["nanogpt"]["val"][:len(steps)]

massive_train = results_5k["massive_pure"]["train"][:len(steps)]
massive_val = results_5k["massive_pure"]["val"][:len(steps)]

ortho_train = results_ortho["orthogonal_pure"]["train"][:len(steps)]
ortho_val = results_ortho["orthogonal_pure"]["val"][:len(steps)]

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
fig.suptitle('The Regularization Pareto Frontier: Overfitting vs Constraints', fontsize=18, fontweight='bold', y=1.05)

# NanoGPT
ax1.plot(steps, nano_train, color='#fca5a5', linestyle='--', linewidth=2, label='Train')
ax1.plot(steps, nano_val, color='#dc2626', linestyle='-', linewidth=3, label='Val')
ax1.fill_between(steps, nano_train, nano_val, color='#fee2e2', alpha=0.5)
ax1.set_title('NanoGPT (Transformer)\nSevere Overfit (Gap: 0.42)', fontsize=14)
ax1.set_ylabel('Cross-Entropy Loss', fontsize=12)
ax1.set_ylim(1.0, 3.0)

# Massive Pure (Scalar Decay)
ax2.plot(steps, massive_train, color='#6ee7b7', linestyle='--', linewidth=2, label='Train')
ax2.plot(steps, massive_val, color='#10b981', linestyle='-', linewidth=3, label='Val')
ax2.fill_between(steps, massive_train, massive_val, color='#d1fae5', alpha=0.5)
ax2.set_title('Massive Pure (DoubleO v7)\nOptimal Zone (Gap: 0.10)', fontsize=14)
ax2.set_xlabel('Training Steps', fontsize=12)

# Orthogonal Pure (Cayley Transform)
ax3.plot(steps, ortho_train, color='#93c5fd', linestyle='--', linewidth=2, label='Train')
ax3.plot(steps, ortho_val, color='#2563eb', linestyle='-', linewidth=3, label='Val')
ax3.fill_between(steps, ortho_train, ortho_val, color='#dbeafe', alpha=0.5)
ax3.set_title('Orthogonal Pure (DoubleO v9)\nZero Overfit / Rigid (Gap: 0.04)', fontsize=14)

for ax in [ax1, ax2, ax3]:
    ax.grid(True, linestyle='--', alpha=0.3, color='gray')
    ax.legend(loc='upper right')

plt.tight_layout()
plt.savefig('regularization_frontier_chart.png', dpi=300, bbox_inches='tight')
print("Chart generated: regularization_frontier_chart.png")
