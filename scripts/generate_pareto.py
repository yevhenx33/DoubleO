import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

# ============================================================
# Load all results
# ============================================================
with open('data/continual_learning_results.json', 'r') as f:
    nano = json.load(f)['nanogpt']['metrics']
with open('data/cheby_moe_results.json', 'r') as f:
    doov1 = json.load(f)['cheby_moe']['metrics']
with open('data/doubleo_v2_results.json', 'r') as f:
    doov2 = json.load(f)['doubleo']['metrics']
with open('data/transformer_moe_results.json', 'r') as f:
    tmoe = json.load(f)['transformer_moe']['metrics']
with open('data/hybrid_v3_results.json', 'r') as f:
    hybrid = json.load(f)['hybrid_v3']['metrics']

def get_loss(metrics_list, target_step):
    for step, loss in metrics_list:
        if step == target_step:
            return loss
    return metrics_list[-1][1]

# ============================================================
# Data: (Retention = Task A @ End P2, Plasticity = Task B @ End P2)
# Lower is better for both axes
# ============================================================
architectures = {
    "NanoGPT\n(Baseline)": {
        "retention": get_loss(nano['val_A'], 3999),
        "plasticity": get_loss(nano['val_B'], 3999),
        "color": "#888888", "marker": "s", "size": 120,
    },
    "DoubleO v1\n(EWC)": {
        "retention": get_loss(doov1['val_A'], 3999),
        "plasticity": get_loss(doov1['val_B'], 3999),
        "color": "#9467bd", "marker": "D", "size": 120,
    },
    "DoubleO v2\n(L2+Dual)": {
        "retention": get_loss(doov2['val_A'], 3999),
        "plasticity": get_loss(doov2['val_B'], 3999),
        "color": "#1f77b4", "marker": "^", "size": 140,
    },
    "Transformer\nH&S": {
        "retention": get_loss(tmoe['val_A'], 3999),
        "plasticity": get_loss(tmoe['val_B'], 3999),
        "color": "#d62728", "marker": "o", "size": 140,
    },
    "Hybrid v3\n(Pre-Agg)": {
        "retention": get_loss(hybrid['val_A'], 3999),
        "plasticity": get_loss(hybrid['val_B'], 3999),
        "color": "#ff7f0e", "marker": "*", "size": 280,
    },
}

# ============================================================
# Style
# ============================================================
plt.rcParams.update({
    "figure.facecolor": "#ffffff",
    "axes.facecolor": "#fafafa",
    "axes.edgecolor": "#333333",
    "text.color": "#1a1a1a",
    "axes.labelcolor": "#333333",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "grid.color": "#e8e8e8",
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.spines.top": False,
    "axes.spines.right": False,
})

fig, ax = plt.subplots(figsize=(10, 8))

# ============================================================
# Plot each architecture
# ============================================================
for name, d in architectures.items():
    ax.scatter(d["retention"], d["plasticity"],
               c=d["color"], marker=d["marker"], s=d["size"],
               edgecolors='white', linewidths=1.5, zorder=5)

# Labels with offset to avoid overlap
label_offsets = {
    "NanoGPT\n(Baseline)": (15, -25),
    "DoubleO v1\n(EWC)": (15, 10),
    "DoubleO v2\n(L2+Dual)": (15, -20),
    "Transformer\nH&S": (-90, 20),
    "Hybrid v3\n(Pre-Agg)": (-105, -25),
}

for name, d in architectures.items():
    ox, oy = label_offsets[name]
    ax.annotate(name, xy=(d["retention"], d["plasticity"]),
                xytext=(ox, oy), textcoords='offset points',
                fontsize=9, fontweight='bold', color=d["color"],
                arrowprops=dict(arrowstyle='->', color=d["color"], lw=1.2),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor=d["color"], alpha=0.9))

# ============================================================
# Pareto Frontier
# ============================================================
# Compute Pareto-optimal points (lower is better on both axes)
points = [(d["retention"], d["plasticity"], name) for name, d in architectures.items()]
points_sorted = sorted(points, key=lambda p: p[0])  # Sort by retention (x-axis)

pareto = []
min_plasticity = float('inf')
for ret, plas, name in points_sorted:
    if plas < min_plasticity:
        pareto.append((ret, plas, name))
        min_plasticity = plas

# Draw the Pareto frontier
if len(pareto) >= 2:
    pareto_x = [p[0] for p in pareto]
    pareto_y = [p[1] for p in pareto]
    ax.plot(pareto_x, pareto_y, color='#ff7f0e', linewidth=2.5, linestyle='-',
            alpha=0.4, zorder=2)
    # Extend frontier to axes
    ax.plot([pareto_x[0], pareto_x[0]], [pareto_y[0], 5.0], color='#ff7f0e',
            linewidth=1.5, linestyle=':', alpha=0.3, zorder=1)
    ax.plot([pareto_x[-1], 5.0], [pareto_y[-1], pareto_y[-1]], color='#ff7f0e',
            linewidth=1.5, linestyle=':', alpha=0.3, zorder=1)

# Fill the dominated region
ax.fill_between([1.0, 5.0], [5.0, 5.0], [1.0, 1.0],
                alpha=0.02, color='red', zorder=0)

# ============================================================
# Ideal point and arrow
# ============================================================
ax.scatter([1.0], [1.0], c='#4caf50', marker='P', s=200, zorder=4,
           edgecolors='white', linewidths=2)
ax.annotate("Ideal\n(1.0, 1.0)", xy=(1.0, 1.0), xytext=(25, 25),
            textcoords='offset points', fontsize=9, color='#4caf50',
            fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#4caf50', lw=1.5))

# Arrow showing improvement direction
ax.annotate('', xy=(1.2, 1.1), xytext=(2.0, 2.0),
            arrowprops=dict(arrowstyle='->', color='#2196f3', lw=2, 
                          connectionstyle='arc3,rad=0.2'))
ax.text(1.8, 2.15, "Better", fontsize=10, color='#2196f3',
        fontweight='bold', fontstyle='italic', ha='center')

# ============================================================
# Axis labels and formatting
# ============================================================
ax.set_xlabel("Retention: Task A Loss After Phase 2 (Shakespeare)\n(lower = better retention)", 
              fontsize=11, labelpad=10)
ax.set_ylabel("Plasticity: Task B Loss After Phase 2 (Math)\n(lower = better new-task learning)", 
              fontsize=11, labelpad=10)
ax.set_title("Pareto Frontier: Retention vs Plasticity\nHub-and-Spoke Continual Learning Architectures",
             fontsize=14, fontweight='bold', pad=15)

ax.set_xlim(1.0, 5.0)
ax.set_ylim(1.0, 5.0)
ax.set_aspect('equal')
ax.grid(True, linestyle='--', alpha=0.4)

# Add quadrant labels
ax.text(1.3, 4.5, "Good Retention\nPoor Plasticity", fontsize=8, color='#aaa',
        ha='center', va='center', fontstyle='italic')
ax.text(4.5, 1.3, "Poor Retention\nGood Plasticity", fontsize=8, color='#aaa',
        ha='center', va='center', fontstyle='italic')
ax.text(1.3, 1.3, "PARETO\nOPTIMUM", fontsize=9, color='#4caf50',
        ha='center', va='center', fontweight='bold', alpha=0.5)
ax.text(4.5, 4.5, "CATASTROPHIC\nFAILURE", fontsize=9, color='#d62728',
        ha='center', va='center', fontweight='bold', alpha=0.3)

plt.tight_layout()
plt.savefig("pareto_frontier.png", dpi=300, bbox_inches="tight")
print("Saved pareto_frontier.png")
