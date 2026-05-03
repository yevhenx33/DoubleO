import json
import matplotlib.pyplot as plt

with open('data/continual_learning_results.json', 'r') as f:
    results = json.load(f)

nano = results["nanogpt"]["metrics"]
massive = results["massive_pure"]["metrics"]

nano_val_A = [x[1] for x in nano["val_A"]]
massive_val_A = [x[1] for x in massive["val_A"]]
steps_A = [x[0] for x in nano["val_A"]]

nano_val_B = [x[1] for x in nano["val_B"]]
massive_val_B = [x[1] for x in massive["val_B"]]
steps_B = [x[0] for x in nano["val_B"]]

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Continual Learning Gauntlet: Catastrophic Forgetting vs Plasticity', fontsize=18, fontweight='bold', y=1.05)

# Plot 1: Catastrophic Forgetting (Task A: Shakespeare Loss)
ax1.plot(steps_A, nano_val_A, color='#dc2626', linestyle='-', linewidth=2.5, label='NanoGPT (Transformer)')
ax1.plot(steps_A, massive_val_A, color='#10b981', linestyle='-', linewidth=2.5, label='Massive Pure (DoubleO)')
ax1.axvline(x=2000, color='gray', linestyle='--', linewidth=2, alpha=0.7)
ax1.text(2050, 6, 'Phase 2 Start\n(Switch to Math)', color='gray', fontsize=11, fontweight='bold')
ax1.set_title('Catastrophic Forgetting\n(Loss on Task A: Shakespeare)', fontsize=14)
ax1.set_xlabel('Training Steps', fontsize=12)
ax1.set_ylabel('Cross-Entropy Loss', fontsize=12)
ax1.legend(loc='upper left')
ax1.grid(True, linestyle='--', alpha=0.3, color='gray')

# Plot 2: Plasticity (Task B: Math Loss)
ax2.plot(steps_B, nano_val_B, color='#fca5a5', linestyle='-', linewidth=2.5, label='NanoGPT (Plasticity)')
ax2.plot(steps_B, massive_val_B, color='#6ee7b7', linestyle='-', linewidth=2.5, label='Massive Pure (Plasticity)')
ax2.set_title('Plasticity\n(Loss on Task B: Math)', fontsize=14)
ax2.set_xlabel('Training Steps (Phase 2 Only)', fontsize=12)
ax2.legend(loc='upper right')
ax2.grid(True, linestyle='--', alpha=0.3, color='gray')

props = dict(boxstyle='round,pad=0.5', facecolor='#fef2f2', edgecolor='#dc2626', alpha=0.9)
ax1.text(0.5, 0.4, "DEVASTATING FORGETTING\nBoth models suffer instant structural collapse!\nMassive Pure is actually slightly worse because its\nglobal continuous state $A$ is completely overwritten.", 
        transform=ax1.transAxes, fontsize=11,
        verticalalignment='center', horizontalalignment='center', bbox=props, color='#991b1b')

plt.tight_layout()
plt.savefig('continual_learning_chart.png', dpi=300, bbox_inches='tight')
print("Chart generated: continual_learning_chart.png")
