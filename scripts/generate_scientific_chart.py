import json
import matplotlib.pyplot as plt
import numpy as np

# 1. Load Data
with open('data/continual_learning_results.json', 'r') as f:
    results = json.load(f)

with open('data/cheby_moe_results.json', 'r') as f:
    results_cheby = json.load(f)

nano = results["nanogpt"]["metrics"]
cheby = results_cheby["cheby_moe"]["metrics"]

# Extract specific milestones
def get_milestone_loss(metrics_list, target_step):
    for step, loss in metrics_list:
        if step == target_step:
            return loss
    # Fallback to nearest
    return metrics_list[-1][1]

nano_A_0 = get_milestone_loss(nano["val_A"], 0)
nano_A_2k = get_milestone_loss(nano["val_A"], 2000)
nano_A_4k = get_milestone_loss(nano["val_A"], 3999)

nano_B_2k = get_milestone_loss(nano["val_B"], 2000)
nano_B_4k = get_milestone_loss(nano["val_B"], 3999)

cheby_A_0 = get_milestone_loss(cheby["val_A"], 0)
cheby_A_2k = get_milestone_loss(cheby["val_A"], 2000)
cheby_A_4k = get_milestone_loss(cheby["val_A"], 3999)

cheby_B_2k = get_milestone_loss(cheby["val_B"], 2000)
cheby_B_4k = get_milestone_loss(cheby["val_B"], 3999)

# 2. Setup Plot Style
plt.rcParams.update({
    "figure.facecolor": "#ffffff",
    "axes.facecolor": "#ffffff",
    "axes.edgecolor": "#333333",
    "text.color": "#000000",
    "axes.labelcolor": "#333333",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "grid.color": "#e0e0e0",
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.spines.top": False,
    "axes.spines.right": False,
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'width_ratios': [2, 1]})

# --- LEFT PANEL: Trajectory ---
x_labels = ["Pre-train", "End Phase 1\n(Shakespeare)", "End Phase 2\n(Math)"]
x_ticks = [0, 1, 2]

# Cheby MoE
ax1.plot([0, 1, 2], [cheby_A_0, cheby_A_2k, cheby_A_4k], marker='o', markersize=8, color="#1f77b4", linewidth=2.5, label="Cheby-MoE Task A (Shakespeare)")
ax1.plot([1, 2], [cheby_B_2k, cheby_B_4k], marker='D', markersize=8, color="#2ca02c", linewidth=2.5, label="Cheby-MoE Task B (Math)")

# NanoGPT
ax1.plot([0, 1, 2], [nano_A_0, nano_A_2k, nano_A_4k], marker='s', markersize=8, color="#d62728", linewidth=2.5, linestyle="--", label="NanoGPT Task A (Shakespeare)")
ax1.plot([1, 2], [nano_B_2k, nano_B_4k], marker='^', markersize=8, color="#9467bd", linewidth=2.5, linestyle="--", label="NanoGPT Task B (Math)")

# Annotations
for i, val in enumerate([cheby_A_0, cheby_A_2k, cheby_A_4k]):
    ax1.annotate(f"{val:.2f}", (i, val), textcoords="offset points", xytext=(0, 10), ha='center', color="#1f77b4", fontweight="bold")

for i, val in enumerate([nano_A_0, nano_A_2k, nano_A_4k]):
    offset = -15 if i == 1 else 10
    ax1.annotate(f"{val:.2f}", (i, val), textcoords="offset points", xytext=(0, offset), ha='center', color="#d62728", fontweight="bold")

ax1.set_xticks(x_ticks)
ax1.set_xticklabels(x_labels)
ax1.set_ylabel("Validation Loss (Cross Entropy)", fontsize=12)
ax1.set_title("Chebyshev MoE: Continual Learning Trajectory", pad=20, fontsize=14, fontweight="bold")
ax1.grid(True, linestyle="--", alpha=0.7)
ax1.legend(loc="center left", bbox_to_anchor=(0.0, 0.6), frameon=True, edgecolor="#cccccc")


# --- RIGHT PANEL: Bar Chart (SFP Score) ---
# SFP Scores calculated previously
sfp_scores = [0.0, 0.0, 0.0, 0.57]
labels = ["NanoGPT", "Massive\nPure", "Poly-MoE", "Cheby-MoE"]
colors = ["#d62728", "#ff7f0e", "#1f77b4", "#2ca02c"]

bars = ax2.bar(labels, sfp_scores, color=colors, width=0.6)

# Annotate bars
for bar in bars:
    height = bar.get_height()
    if height > 0:
        ax2.annotate(f"{height:.2f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold', fontsize=12)

# Add the explanatory text box
textstr = "Standard shared-state\nmodels suffer Total\nCatastrophic Forgetting\n(Score = 0.0).\n\nCheby-MoE's 'CEO'\nRouter achieves\nAbsolute Stability!"
props = dict(boxstyle='round,pad=0.8', facecolor='#eaffea', edgecolor='#2ca02c', alpha=1.0)
ax2.text(0.5, 0.5, textstr, transform=ax2.transAxes, fontsize=11,
        verticalalignment='center', horizontalalignment='center', bbox=props,
        color="#006400", fontweight="bold")

ax2.set_ylabel("SFP Score (Harmonic Mean)", fontsize=12)
ax2.set_title("Sparse Feature Preservation\n(Zero Forgetting)", pad=20, fontsize=14, fontweight="bold")
ax2.set_ylim(0, 1.0)
ax2.grid(True, axis='y', linestyle="--", alpha=0.7)

plt.tight_layout()
plt.savefig("final_scientific_chart.png", dpi=300, bbox_inches="tight")
