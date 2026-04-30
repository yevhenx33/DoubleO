import matplotlib.pyplot as plt
import numpy as np
import os
import json

# Styling configuration
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['figure.facecolor'] = 'white'

COLOR_MATH_DO = '#2563EB'   # Blue
COLOR_CODE_DO = '#16A34A'   # Green
COLOR_MATH_STD = '#DC2626'  # Red
COLOR_CODE_STD = '#D97706'  # Orange
COLOR_GRID = '#E5E5E5'

phases = ['Pre-train', 'End Task A\n(Math)', 'End Task B\n(Code)']

# Proven Data from Terminal Outputs
data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'results.json')
with open(data_path, 'r') as f:
    results = json.load(f)

# Data from StandardGPT (Catastrophic Forgetting)
math_loss_std = results["StandardGPT"]["math_loss"]
code_loss_std = results["StandardGPT"]["code_loss"]

# Data from NanoDoubleO (Zero-Forgetting)
math_loss_do = results["NanoDoubleO"]["math_loss"]
code_loss_do = results["NanoDoubleO"]["code_loss"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'width_ratios': [2.5, 1]})

# ==========================================
# LEFT PLOT: Loss Trajectory
# ==========================================
ax1.plot(phases, math_loss_do, marker='o', color=COLOR_MATH_DO, linewidth=2.5, markersize=8, label='NanoDoubleO Math (Retention)')
ax1.plot(phases, code_loss_do, marker='s', color=COLOR_CODE_DO, linewidth=2.5, markersize=8, label='NanoDoubleO Code (Plasticity)')

ax1.plot(phases, math_loss_std, marker='X', color=COLOR_MATH_STD, linewidth=2, markersize=8, linestyle='--', label='StandardGPT Math (Catastrophic)')
ax1.plot(phases, code_loss_std, marker='^', color=COLOR_CODE_STD, linewidth=2, markersize=8, linestyle='--', label='StandardGPT Code')

# Annotations for NanoDoubleO Math
for i, val in enumerate(math_loss_do):
    ax1.annotate(f"{val:.4f}", (i, val), textcoords="offset points", xytext=(0, -15), ha='center', color=COLOR_MATH_DO, fontsize=10, fontweight='bold')

# Annotations for NanoDoubleO Code
for i, val in enumerate(code_loss_do):
    ax1.annotate(f"{val:.4f}", (i, val), textcoords="offset points", xytext=(0, 10), ha='center', color=COLOR_CODE_DO, fontsize=10, fontweight='bold')

# Annotations for StandardGPT Math Spikes
ax1.annotate(f"{math_loss_std[1]:.4f}", (1, math_loss_std[1]), textcoords="offset points", xytext=(0, 10), ha='center', color=COLOR_MATH_STD, fontsize=10)
ax1.annotate(f"{math_loss_std[2]:.4f}\n(Catastrophic!)", (2, math_loss_std[2]), textcoords="offset points", xytext=(0, 15), ha='center', color=COLOR_MATH_STD, fontsize=10, fontweight='bold')

# Bounding box for zero forgetting annotation
bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec=COLOR_MATH_DO, lw=1, alpha=0.9)
ax1.annotate('Zero Forgetting\nAchieved (+0.02)', xy=(2, math_loss_do[2]), xytext=(1.5, 2.50),
             arrowprops=dict(facecolor=COLOR_MATH_DO, edgecolor=COLOR_MATH_DO, arrowstyle='->', shrinkA=0, shrinkB=5),
             bbox=bbox_props, ha='center', color=COLOR_MATH_DO, fontsize=10, fontweight='bold')

ax1.set_title('NanoDoubleO vs StandardGPT: Continual Learning Trajectory', fontsize=14, pad=15, fontweight='bold')
ax1.set_ylabel('Cross Entropy Loss (Lower is Better)', fontsize=11)
ax1.grid(True, color=COLOR_GRID, linestyle='--', alpha=0.7)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.legend(loc='upper left', frameon=True, facecolor='white', edgecolor=COLOR_GRID)

# ==========================================
# RIGHT PLOT: Forgetting Comparison
# ==========================================
forgetting_std = math_loss_std[2] - math_loss_std[1]
forgetting_do = math_loss_do[2] - math_loss_do[1]

models = ['StandardGPT\n(Baseline)', 'NanoDoubleO\n(Complex Chebyshev)']
forgetting_vals = [forgetting_std, forgetting_do]
colors = [COLOR_MATH_STD, COLOR_MATH_DO]

bars = ax2.bar(models, forgetting_vals, color=colors, width=0.6)

# Annotate bars
ax2.annotate(f"+{forgetting_std:.4f}", (0, forgetting_std), textcoords="offset points", xytext=(0, 5), ha='center', color='black', fontsize=11, fontweight='bold')
ax2.annotate(f"+{forgetting_do:.4f}", (1, forgetting_do), textcoords="offset points", xytext=(0, 5), ha='center', color='black', fontsize=11, fontweight='bold')

# Arrow annotation for StandardGPT bar
ax2.annotate('Violent weight\noverwrite (GELU)', xy=(0, forgetting_std), xytext=(0, forgetting_std + 0.1),
             ha='center', color=COLOR_MATH_STD, fontsize=9, arrowprops=dict(arrowstyle='->', color=COLOR_MATH_STD))

ax2.set_title(r'Catastrophic Forgetting\n(Math Loss $\Delta$)', fontsize=14, pad=15, fontweight='bold')
ax2.set_ylabel('Degradation Penalty', fontsize=11)
ax2.set_ylim(0, forgetting_std * 1.3)
ax2.grid(True, axis='y', color=COLOR_GRID, linestyle='--', alpha=0.7)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

data_dir = os.path.dirname(data_path)
plt.savefig(os.path.join(data_dir, '..', 'comparison_chart.png'), dpi=300, bbox_inches='tight')
