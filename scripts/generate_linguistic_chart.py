import matplotlib.pyplot as plt
import numpy as np

# Data extracted from Modal logs
steps = [0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]

# NanoGPT (~800k)
nano_train = [4.2005, 2.3230, 2.0254, 1.7204, 1.5708, 1.4779, 1.4365, 1.3413, 1.3035, 1.2988, 1.2612]
nano_val   = [3.7814, 2.3753, 2.0551, 1.8737, 1.7623, 1.6732, 1.6046, 1.5565, 1.6011, 1.5254, 1.5350]

# DoubleO v6 (~90k)
do_train = [6.4197, 3.2624, 3.1522, 3.0183, 3.0016, 2.9634, 2.9258, 2.9050, 2.8868, 2.8850, 2.8789]
do_val   = [5.9443, 3.2582, 3.1634, 3.0747, 3.0478, 3.0546, 3.0017, 2.9696, 2.9543, 2.9884, 2.9715]

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

fig, ax = plt.subplots(figsize=(10, 6))

# Plot lines
ax.plot(steps, nano_train, color='#dc2626', linestyle='--', linewidth=2, alpha=0.5, label='NanoGPT Train (~800k)')
ax.plot(steps, nano_val,   color='#dc2626', marker='o', markersize=6, linewidth=2.5, label='NanoGPT Val (1.53)')

ax.plot(steps, do_train, color='#2563eb', linestyle='--', linewidth=2, alpha=0.5, label='DoubleO Train (~90k)')
ax.plot(steps, do_val,   color='#2563eb', marker='D', markersize=6, linewidth=2.5, label='DoubleO Val (2.97)')

# Grid and labels
ax.grid(True, linestyle='--', alpha=0.3, color='gray')
ax.set_xlabel('Training Steps (TinyShakespeare)', fontsize=12)
ax.set_ylabel('Cross-Entropy Loss (Lower is Better)', fontsize=12)
ax.set_title('Linguistic Architecture Comparison\n(Transformer vs Resonant Cavity)', fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=False)

# Set limits
ax.set_ylim(1.0, 6.5)

# Annotations highlighting the size discrepancy
props = dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='gray', alpha=0.9)
ax.text(0.5, 0.6, "Note: DoubleO operates at a massive 10x parameter deficit.\nTo compete linguistically, we must scale the polynomial degree.", 
        transform=ax.transAxes, fontsize=10,
        verticalalignment='center', horizontalalignment='center', bbox=props, color='#4b5563', style='italic')

plt.tight_layout()
plt.savefig('linguistic_comparison_chart.png', dpi=300, bbox_inches='tight')
print("Chart generated: linguistic_comparison_chart.png")
