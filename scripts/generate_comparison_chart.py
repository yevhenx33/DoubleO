import json
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ============================================================
# Load all datasets
# ============================================================
with open('data/transformer_moe_results.json', 'r') as f:
    tmoe = json.load(f)['transformer_moe']['metrics']

with open('data/doubleo_v2_results.json', 'r') as f:
    doov2 = json.load(f)['doubleo']['metrics']

with open('data/continual_learning_results.json', 'r') as f:
    nano = json.load(f)['nanogpt']['metrics']

with open('data/cheby_moe_results.json', 'r') as f:
    doov1 = json.load(f)['cheby_moe']['metrics']

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
# Figure: 2x2 grid
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
ax_tmoe, ax_doo, ax_bar, ax_table = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

C_S1 = "#1f77b4"   # Blue (Task A / Shakespeare)
C_S2 = "#2ca02c"   # Green (Task B / Math)
C_PHASE = "#cccccc"

# ============================================================
# Panel A: Transformer Hub-and-Spoke Trajectory
# ============================================================
steps_a = [s for s, _ in tmoe['val_A']]
vals_a = [v for _, v in tmoe['val_A']]
steps_b = [s for s, _ in tmoe['val_B']]
vals_b = [v for _, v in tmoe['val_B']]

ax_tmoe.plot(steps_a, vals_a, color=C_S1, linewidth=2.2, label="Spoke 1 — Shakespeare")
ax_tmoe.plot(steps_b, vals_b, color=C_S2, linewidth=2.2, label="Spoke 2 — Math")
ax_tmoe.axvline(x=PHASE_1_END, color=C_PHASE, linestyle="--", linewidth=1.2)
ax_tmoe.fill_betweenx([0, 5], 0, PHASE_1_END, alpha=0.04, color=C_S1)
ax_tmoe.fill_betweenx([0, 5], PHASE_1_END, 4000, alpha=0.04, color=C_S2)

ax_tmoe.annotate(f"{vals_a[-1]:.2f}", xy=(steps_a[-1], vals_a[-1]), xytext=(-45, -18),
                 textcoords='offset points', fontsize=9, fontweight='bold', color=C_S1,
                 arrowprops=dict(arrowstyle='->', color=C_S1, lw=1))
ax_tmoe.annotate(f"{vals_b[-1]:.2f}", xy=(steps_b[-1], vals_b[-1]), xytext=(-45, 12),
                 textcoords='offset points', fontsize=9, fontweight='bold', color=C_S2,
                 arrowprops=dict(arrowstyle='->', color=C_S2, lw=1))

ax_tmoe.set_xlabel("Training Steps", fontsize=10)
ax_tmoe.set_ylabel("Validation Loss (CE)", fontsize=10)
ax_tmoe.set_title("(A) Transformer Hub-and-Spoke", fontsize=12, fontweight='bold', pad=10)
ax_tmoe.set_ylim(0.8, 4.5)
ax_tmoe.legend(loc='upper right', fontsize=8, frameon=True, edgecolor='#ccc')
ax_tmoe.grid(True, linestyle='--', alpha=0.4)

# ============================================================
# Panel B: DoubleO v2 (Resonant Cavity) Trajectory
# ============================================================
steps_a2 = [s for s, _ in doov2['val_A']]
vals_a2 = [v for _, v in doov2['val_A']]
steps_b2 = [s for s, _ in doov2['val_B']]
vals_b2 = [v for _, v in doov2['val_B']]

ax_doo.plot(steps_a2, vals_a2, color=C_S1, linewidth=2.2, label="Expert 1 — Shakespeare")
ax_doo.plot(steps_b2, vals_b2, color=C_S2, linewidth=2.2, label="Expert 2 — Math")
ax_doo.axvline(x=PHASE_1_END, color=C_PHASE, linestyle="--", linewidth=1.2)
ax_doo.fill_betweenx([0, 5], 0, PHASE_1_END, alpha=0.04, color=C_S1)
ax_doo.fill_betweenx([0, 5], PHASE_1_END, 4000, alpha=0.04, color=C_S2)

ax_doo.annotate(f"{vals_a2[-1]:.2f}", xy=(steps_a2[-1], vals_a2[-1]), xytext=(-45, -18),
                textcoords='offset points', fontsize=9, fontweight='bold', color=C_S1,
                arrowprops=dict(arrowstyle='->', color=C_S1, lw=1))
ax_doo.annotate(f"{vals_b2[-1]:.2f}", xy=(steps_b2[-1], vals_b2[-1]), xytext=(-45, 12),
                textcoords='offset points', fontsize=9, fontweight='bold', color=C_S2,
                arrowprops=dict(arrowstyle='->', color=C_S2, lw=1))

ax_doo.set_xlabel("Training Steps", fontsize=10)
ax_doo.set_ylabel("Validation Loss (CE)", fontsize=10)
ax_doo.set_title("(B) DoubleO Resonant Cavity", fontsize=12, fontweight='bold', pad=10)
ax_doo.set_ylim(0.8, 4.5)
ax_doo.legend(loc='upper right', fontsize=8, frameon=True, edgecolor='#ccc')
ax_doo.grid(True, linestyle='--', alpha=0.4)

# ============================================================
# Panel C: Architecture Comparison Bar Chart
# ============================================================
def get_loss(metrics_list, target_step):
    for step, loss in metrics_list:
        if step == target_step:
            return loss
    return metrics_list[-1][1]

archs = ["NanoGPT\n(Baseline)", "DoubleO v1\n(EWC)", "DoubleO v2\n(L2+Dual Opt)", "Transformer\nH&S"]

# Task A at end of Phase 1 (best Shakespeare)
task_a_p1 = [
    get_loss(nano['val_A'], 1950),
    get_loss(doov1['val_A'], 1950),
    get_loss(doov2['val_A'], 1950),
    get_loss(tmoe['val_A'], 1950),
]
# Task A at end of Phase 2 (retention test)
task_a_p2 = [
    get_loss(nano['val_A'], 3999),
    get_loss(doov1['val_A'], 3999),
    get_loss(doov2['val_A'], 3999),
    get_loss(tmoe['val_A'], 3999),
]
# Task B at end of Phase 2 (plasticity test)
task_b_p2 = [
    get_loss(nano['val_B'], 3999),
    get_loss(doov1['val_B'], 3999),
    get_loss(doov2['val_B'], 3999),
    get_loss(tmoe['val_B'], 3999),
]

x = np.arange(len(archs))
width = 0.22

bars1 = ax_bar.bar(x - width, task_a_p1, width, label='Task A @ End P1', color=C_S1, alpha=0.85, edgecolor='white', linewidth=0.8)
bars2 = ax_bar.bar(x, task_a_p2, width, label='Task A @ End P2 (Retention)', color='#d62728', alpha=0.85, edgecolor='white', linewidth=0.8)
bars3 = ax_bar.bar(x + width, task_b_p2, width, label='Task B @ End P2 (Plasticity)', color=C_S2, alpha=0.85, edgecolor='white', linewidth=0.8)

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax_bar.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords='offset points', ha='center', va='bottom',
                        fontsize=7, fontweight='bold', color='#333')

ax_bar.set_xticks(x)
ax_bar.set_xticklabels(archs, fontsize=9)
ax_bar.set_ylabel("Validation Loss (CE)", fontsize=10)
ax_bar.set_title("(C) Architecture Comparison", fontsize=12, fontweight='bold', pad=10)
ax_bar.legend(loc='upper left', fontsize=8, frameon=True, edgecolor='#ccc')
ax_bar.grid(True, axis='y', linestyle='--', alpha=0.4)

# ============================================================
# Panel D: Summary Table
# ============================================================
ax_table.axis('off')

retention = [
    f"{(task_a_p2[i] - task_a_p1[i]):.2f}" for i in range(len(archs))
]

table_data = [
    ['NanoGPT', f'{task_a_p1[0]:.2f}', f'{task_a_p2[0]:.2f}', f'{task_b_p2[0]:.2f}', retention[0], 'NO'],
    ['DoubleO v1 (EWC)', f'{task_a_p1[1]:.2f}', f'{task_a_p2[1]:.2f}', f'{task_b_p2[1]:.2f}', retention[1], 'YES'],
    ['DoubleO v2 (L2)', f'{task_a_p1[2]:.2f}', f'{task_a_p2[2]:.2f}', f'{task_b_p2[2]:.2f}', retention[2], 'YES'],
    ['Transformer H&S', f'{task_a_p1[3]:.2f}', f'{task_a_p2[3]:.2f}', f'{task_b_p2[3]:.2f}', retention[3], 'YES'],
]
col_labels = ['Architecture', 'Task A\n(End P1)', 'Task A\n(End P2)', 'Task B\n(End P2)', 'Drift Δ', 'Zero\nForget?']

table = ax_table.table(cellText=table_data, colLabels=col_labels, loc='center',
                       cellLoc='center', colWidths=[0.22, 0.12, 0.12, 0.12, 0.12, 0.10])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.0, 1.8)

# Style header
for j in range(len(col_labels)):
    table[0, j].set_facecolor('#e8eaf6')
    table[0, j].set_text_props(fontweight='bold')

# Color code results
for i in range(1, len(table_data) + 1):
    if table_data[i-1][5] == '❌':
        table[i, 4].set_facecolor('#ffebee')
        table[i, 5].set_facecolor('#ffebee')
    else:
        table[i, 4].set_facecolor('#e8f5e9')
        table[i, 5].set_facecolor('#e8f5e9')

ax_table.set_title("(D) Summary — Retention vs Plasticity", fontsize=12, fontweight='bold', pad=10)

fig.suptitle("DoubleO: Hub-and-Spoke Continual Learning — Architecture Comparison",
             fontsize=15, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("transformer_moe_chart.png", dpi=300, bbox_inches="tight")
print("Saved transformer_moe_chart.png")
