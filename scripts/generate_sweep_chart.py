import json
import matplotlib.pyplot as plt

# Load results
with open('data/sweep_results.json', 'r') as f:
    results = json.load(f)

# NanoGPT baseline data (from previous run)
nano_val = [3.7814, 2.3753, 2.0551, 1.8737, 1.7623, 1.6732, 1.6046, 1.5565, 1.6011, 1.5254, 1.5350]

steps = [0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

fig, ax = plt.subplots(figsize=(12, 7))

# Plot Baseline
ax.plot(steps, nano_val, color='#dc2626', marker='o', markersize=6, linewidth=3, label='NanoGPT Baseline (~800k)')

# Plot 4 Challengers
ax.plot(steps, results["multihead"]["val"], color='#9ca3af', marker='x', markersize=6, linewidth=2, label='DoubleO: Multi-Head (3.32)')
ax.plot(steps, results["decay"]["val"], color='#fbbf24', marker='s', markersize=6, linewidth=2, label='DoubleO: Temporal Decay (2.08)')
ax.plot(steps, results["rope"]["val"], color='#a855f7', marker='^', markersize=6, linewidth=2, label='DoubleO: RoPE Injection (2.03)')
ax.plot(steps, results["hybrid"]["val"], color='#2563eb', marker='D', markersize=6, linewidth=3, label='DoubleO: Hybrid Attention (1.91)')

ax.grid(True, linestyle='--', alpha=0.3, color='gray')
ax.set_xlabel('Training Steps (TinyShakespeare)', fontsize=12)
ax.set_ylabel('Validation Cross-Entropy Loss', fontsize=12)
ax.set_title('DoubleO v6 Positional Memory Strategies vs Transformer', fontsize=16, fontweight='bold', pad=20)
ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=False)

ax.set_ylim(1.0, 4.0)

# Annotation for Hybrid
props = dict(boxstyle='round,pad=0.5', facecolor='#eff6ff', edgecolor='#2563eb', alpha=0.9)
ax.text(0.5, 0.4, "Hybrid Attention bridges the Positional Memory gap,\npushing DoubleO aggressively toward Transformer parity!", 
        transform=ax.transAxes, fontsize=11,
        verticalalignment='center', horizontalalignment='center', bbox=props, color='#1e3a8a', fontweight='bold')

plt.tight_layout()
plt.savefig('linguistic_sweep_results.png', dpi=300, bbox_inches='tight')
print("Chart generated: linguistic_sweep_results.png")
