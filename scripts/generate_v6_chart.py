import matplotlib.pyplot as plt
import numpy as np

# Data from Extreme Grokking Run
phases = ['Pre-train', 'End Phase 1\n(Addition)', 'End Phase 3\n(Add + Sub)']

# Accuracy data (in percentages)
add_train = [0, 100.0, 100.0]
add_test = [0, 24.0, 33.0]
sub_train = [0, 0, 100.0]
sub_test = [0, 0, 26.0]

# Setup plot style
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

fig, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [2, 1]}, figsize=(13, 6))

# -------------------------------------------------------------
# Left Panel: Continual Learning Trajectory
# -------------------------------------------------------------
ax1.plot(phases, add_train, marker='o', markersize=8, linewidth=2.5, color='#2563eb', label='Addition Train')
ax1.plot(phases, add_test, marker='D', markersize=8, linewidth=2.5, color='#16a34a', label='Addition Test (Generalization)')
ax1.plot(phases, sub_train, marker='s', markersize=8, linewidth=2.5, linestyle='--', color='#ea580c', label='Subtraction Train')
ax1.plot(phases, sub_test, marker='^', markersize=8, linewidth=2.5, linestyle='--', color='#9333ea', label='Subtraction Test')

# Add gridlines
ax1.grid(True, linestyle='--', alpha=0.3, color='gray')

# Labels and title
ax1.set_ylabel('Accuracy (%)', fontsize=12)
ax1.set_title('DoubleO v6: Multi-Digit Grokking Trajectory', fontsize=14, fontweight='bold', pad=20)
ax1.legend(loc='upper left', frameon=True, fancybox=True, shadow=False)
ax1.set_ylim(-5, 110)

# Annotations for Phase 1
ax1.text(1, 100.0 + 3, '100.0%', ha='center', va='bottom', color='#2563eb', fontweight='bold')
ax1.text(1, 24.0 - 5, '24.0%', ha='center', va='top', color='#16a34a', fontweight='bold')

# Annotations for Phase 3
ax1.text(2, 100.0 + 3, '100.0%', ha='center', va='bottom', color='#2563eb', fontweight='bold')
ax1.text(2, 33.0 + 3, '33.0%', ha='center', va='bottom', color='#16a34a', fontweight='bold')
ax1.text(2, 26.0 - 5, '26.0%', ha='center', va='top', color='#9333ea', fontweight='bold')

# -------------------------------------------------------------
# Right Panel: Retention Delta
# -------------------------------------------------------------
labels = ['Addition Retention $\\Delta$']
values = [9.0]
colors = ['#2563eb']

bars = ax2.bar(labels, values, color=colors, width=0.4)

ax2.grid(True, linestyle='--', alpha=0.3, color='gray', axis='y')
ax2.set_ylabel('Accuracy Delta (%)', fontsize=12)
ax2.set_title('Positive Transfer\n(Zero Forgetting)', fontsize=14, fontweight='bold', pad=20)

# Add value label on top of bar
for bar in bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
            f'+{height}%',
            ha='center', va='bottom', fontweight='bold', fontsize=12)

ax2.set_ylim(0, 15)

# Add a text box explaining the result
props = dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#16a34a', alpha=0.9)
ax2.text(0.5, 0.5, "Standard LLMs suffer\nCatastrophic Forgetting\n(Negative Delta).\n\nDoubleO's Auto-Router\nachieves Positive Transfer!", 
        transform=ax2.transAxes, fontsize=10,
        verticalalignment='center', horizontalalignment='center', bbox=props, color='#16a34a', fontweight='bold')

plt.tight_layout()
plt.savefig('v6_grokking_chart.png', dpi=300, bbox_inches='tight')
print("Chart generated: v6_grokking_chart.png")
