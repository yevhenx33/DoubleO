#!/usr/bin/env python3
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'ab_scale')

# Collect results
results = []
for fname in os.listdir(out_dir):
    if fname.endswith('.json'):
        with open(os.path.join(out_dir, fname)) as f:
            results.append(json.load(f))

# Organize data
cur_vals = [r['avg_forgetting'] for r in results if r['variant'] == 'do_current']
scl_vals = [r['avg_forgetting'] for r in results if r['variant'] == 'do_scaled']

cur_mean, cur_std = np.mean(cur_vals), np.std(cur_vals, ddof=1)
scl_mean, scl_std = np.mean(scl_vals), np.std(scl_vals, ddof=1)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('DoubleO Architecture: The Impact of Per-Task Scaling (128-dim, 4 tasks, N=3)', fontsize=14, fontweight='bold', y=1.05)

# Panel 1: Bar chart
ax = axes[0]
bars = ax.bar([0, 1], [cur_mean, scl_mean], yerr=[cur_std, scl_std], 
              color=['#DC2626', '#2563EB'], width=0.5, capsize=5, edgecolor='black')

ax.set_xticks([0, 1])
ax.set_xticklabels(['Current\n(Train Shared MLP)\n524K trainable params', 'Scaled\n(Train Scaling Only)\n4K trainable params'], fontsize=11)
ax.set_ylabel('Avg Forgetting (Δ Loss)', fontsize=11)
ax.set_title('Average Forgetting Comparison', fontsize=12, fontweight='bold')
ax.axhline(0, color='black', linewidth=0.8)
ax.grid(axis='y', alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add value labels
for bar, m, s in zip(bars, [cur_mean, scl_mean], [cur_std, scl_std]):
    ax.annotate(f'+{m:.4f}\n±{s:.4f}',
                (bar.get_x() + bar.get_width()/2, bar.get_height() + s + 0.05),
                ha='center', fontsize=10, fontweight='bold')

# Panel 2: Individual Task Breakdown (using one representative seed, e.g., seed 42)
ax = axes[1]
cur_s42 = next(r for r in results if r['variant'] == 'do_current' and r['seed'] == 42)['forgetting']
scl_s42 = next(r for r in results if r['variant'] == 'do_scaled' and r['seed'] == 42)['forgetting']

tasks = list(cur_s42.keys())
x = np.arange(len(tasks))
width = 0.35

ax.bar(x - width/2, [cur_s42[t] for t in tasks], width, label='Current', color='#DC2626', edgecolor='black', alpha=0.8)
ax.bar(x + width/2, [scl_s42[t] for t in tasks], width, label='Scaled (Zero Interference)', color='#2563EB', edgecolor='black', alpha=0.8)

ax.set_xticks(x)
ax.set_xticklabels(tasks, fontsize=11)
ax.set_ylabel('Forgetting (Δ Loss)', fontsize=11)
ax.set_title('Per-Task Forgetting (Seed 42)', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.axhline(0, color='black', linewidth=0.8)
ax.grid(axis='y', alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
chart_path = os.path.join(out_dir, 'ab_scale_results.png')
plt.savefig(chart_path, dpi=200, bbox_inches='tight')
print(f"Saved {chart_path}")
