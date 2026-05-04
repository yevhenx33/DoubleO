import json
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ============================================================
# Load data
# ============================================================
with open('data/transformer_moe_results.json', 'r') as f:
    tmoe = json.load(f)['transformer_moe']['metrics']

with open('data/continual_learning_results.json', 'r') as f:
    cl = json.load(f)
nano = cl['nanogpt']['metrics']

with open('data/cheby_moe_results.json', 'r') as f:
    doubleo = json.load(f)['cheby_moe']['metrics']

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
    "grid.color": "#e0e0e0",
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.spines.top": False,
    "axes.spines.right": False,
})

PHASE_1_END = 2000

# ============================================================
# Figure: Two panels side-by-side
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5), sharey=False)

# --- Colors ---
C_SPOKE1 = "#1f77b4"   # Blue
C_SPOKE2 = "#2ca02c"   # Green
C_NANO_A = "#d62728"    # Red
C_NANO_B = "#9467bd"    # Purple
C_DOO_A = "#ff7f0e"     # Orange (DoubleO Task A)
C_DOO_B = "#17becf"     # Cyan   (DoubleO Task B)
C_PHASE = "#cccccc"

# ============================================================
# Panel 1: Transformer Hub-and-Spoke (Full Trajectory)
# ============================================================
steps_a = [s for s, _ in tmoe['val_A']]
vals_a = [v for _, v in tmoe['val_A']]
steps_b = [s for s, _ in tmoe['val_B']]
vals_b = [v for _, v in tmoe['val_B']]

ax1.plot(steps_a, vals_a, color=C_SPOKE1, linewidth=2.2, label="Spoke 1 — Shakespeare", zorder=3)
ax1.plot(steps_b, vals_b, color=C_SPOKE2, linewidth=2.2, label="Spoke 2 — Math", zorder=3)

# Phase boundary
ax1.axvline(x=PHASE_1_END, color=C_PHASE, linestyle="--", linewidth=1.5, zorder=1)
ax1.fill_betweenx([0, 5], 0, PHASE_1_END, alpha=0.04, color=C_SPOKE1, zorder=0)
ax1.fill_betweenx([0, 5], PHASE_1_END, 4000, alpha=0.04, color=C_SPOKE2, zorder=0)

# Annotations
ax1.annotate("Phase 1\nShakespeare Only", xy=(1000, 4.3), ha='center', fontsize=9, color="#888888", style='italic')
ax1.annotate("Phase 2\nMath (Spoke 1 Frozen)", xy=(3000, 4.3), ha='center', fontsize=9, color="#888888", style='italic')

# Final values
final_a = vals_a[-1]
final_b = vals_b[-1]
ax1.annotate(f"{final_a:.2f}", xy=(steps_a[-1], final_a), xytext=(-40, -20),
             textcoords='offset points', fontsize=10, fontweight='bold', color=C_SPOKE1,
             arrowprops=dict(arrowstyle='->', color=C_SPOKE1, lw=1.2))
ax1.annotate(f"{final_b:.2f}", xy=(steps_b[-1], final_b), xytext=(-40, 15),
             textcoords='offset points', fontsize=10, fontweight='bold', color=C_SPOKE2,
             arrowprops=dict(arrowstyle='->', color=C_SPOKE2, lw=1.2))

# Retention box
retention = 1.0 - max(0, (final_a - 1.57) / 1.57)
ax1.text(3000, 3.6, f"Retention: {retention:.1%}\nPlasticity: {final_b:.2f}",
         fontsize=10, ha='center', va='center',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#e8f5e9', edgecolor='#4caf50', alpha=0.9))

ax1.set_xlabel("Training Steps", fontsize=11)
ax1.set_ylabel("Validation Loss (CE)", fontsize=11)
ax1.set_title("Transformer Hub-and-Spoke\n(Dual Optimizer + L2 Anchor)", fontsize=13, fontweight='bold', pad=12)
ax1.set_ylim(0.8, 4.5)
ax1.set_xlim(-100, 4100)
ax1.legend(loc='upper right', frameon=True, edgecolor='#cccccc', fontsize=9)
ax1.grid(True, linestyle='--', alpha=0.5)

# ============================================================
# Panel 2: Milestone Comparison (3 architectures)
# ============================================================
def get_loss(metrics_list, target_step):
    for step, loss in metrics_list:
        if step == target_step:
            return loss
    return metrics_list[-1][1]

# Gather milestone data
archs = ["NanoGPT\n(Baseline)", "DoubleO\n(Resonant Cavity)", "Transformer\nHub-and-Spoke"]
task_a_end_p1 = [
    get_loss(nano['val_A'], 1950),
    get_loss(doubleo['val_A'], 1950),
    get_loss(tmoe['val_A'], 1950),
]
task_a_end_p2 = [
    get_loss(nano['val_A'], 3999),
    get_loss(doubleo['val_A'], 3999),
    get_loss(tmoe['val_A'], 3999),
]
task_b_end_p2 = [
    get_loss(nano['val_B'], 3999),
    get_loss(doubleo['val_B'], 3999),
    get_loss(tmoe['val_B'], 3999),
]

x = np.arange(len(archs))
width = 0.25

bars1 = ax2.bar(x - width, task_a_end_p1, width, label='Task A @ End Phase 1', color=C_SPOKE1, alpha=0.85, edgecolor='white', linewidth=0.8)
bars2 = ax2.bar(x, task_a_end_p2, width, label='Task A @ End Phase 2', color=C_NANO_A, alpha=0.85, edgecolor='white', linewidth=0.8)
bars3 = ax2.bar(x + width, task_b_end_p2, width, label='Task B @ End Phase 2', color=C_SPOKE2, alpha=0.85, edgecolor='white', linewidth=0.8)

# Value labels
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax2.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 4), textcoords='offset points', ha='center', va='bottom',
                     fontsize=8, fontweight='bold', color='#333333')

# Forgetting arrows for NanoGPT
ax2.annotate('', xy=(x[0], task_a_end_p2[0]), xytext=(x[0] - width, task_a_end_p1[0]),
             arrowprops=dict(arrowstyle='->', color='red', lw=2))
ax2.text(x[0] - 0.05, (task_a_end_p1[0] + task_a_end_p2[0]) / 2 + 0.3, "FORGOT",
         fontsize=8, color='red', fontweight='bold', ha='center', rotation=0)

ax2.set_xticks(x)
ax2.set_xticklabels(archs, fontsize=10)
ax2.set_ylabel("Validation Loss (CE)", fontsize=11)
ax2.set_title("Architecture Comparison\n(Retention vs Plasticity)", fontsize=13, fontweight='bold', pad=12)
ax2.legend(loc='upper left', frameon=True, edgecolor='#cccccc', fontsize=9)
ax2.grid(True, axis='y', linestyle='--', alpha=0.5)
ax2.set_ylim(0, max(max(task_a_end_p2), max(task_b_end_p2)) + 1.5)

plt.tight_layout(w_pad=3)
plt.savefig("transformer_moe_chart.png", dpi=300, bbox_inches="tight")
print("Saved transformer_moe_chart.png")
