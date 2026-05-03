import json
import matplotlib.pyplot as plt

with open('data/continual_learning_results.json', 'r') as f:
    results = json.load(f)

with open('data/poly_moe_results.json', 'r') as f:
    results_poly = json.load(f)

with open('data/ewc_continual_results.json', 'r') as f:
    results_ewc = json.load(f)

nano = results["nanogpt"]["metrics"]
massive = results["massive_pure"]["metrics"]
poly = results_poly["poly_moe"]["metrics"]
ewc = results_ewc["ewc_pure"]["metrics"]

nano_val_A = [x[1] for x in nano["val_A"]]
massive_val_A = [x[1] for x in massive["val_A"]]
poly_val_A = [x[1] for x in poly["val_A"]]
ewc_val_A = [x[1] for x in ewc["val_A"]]

nano_val_B = [x[1] for x in nano["val_B"]]
massive_val_B = [x[1] for x in massive["val_B"]]
poly_val_B = [x[1] for x in poly["val_B"]]
ewc_val_B = [x[1] for x in ewc["val_B"]]

steps_A = [x[0] for x in nano["val_A"]]
steps_B = [x[0] for x in nano["val_B"]]

plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#0d1117",
    "axes.edgecolor": "#30363d",
    "text.color": "#c9d1d9",
    "axes.labelcolor": "#8b949e",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "grid.color": "#21262d",
    "font.family": "monospace",
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Task A (Catastrophic Forgetting)
ax1.plot(steps_A, nano_val_A, color="#f85149", linewidth=2.5, alpha=0.9, label="NanoGPT")
ax1.plot(steps_A, massive_val_A, color="#2ea043", linewidth=2.5, alpha=0.9, label="Massive Pure")
ax1.plot(steps_A, poly_val_A, color="#58a6ff", linewidth=2.5, alpha=0.9, linestyle="--", label="Poly-MoE")
ax1.plot(steps_A, ewc_val_A, color="#d2a8ff", linewidth=2.5, alpha=0.9, linestyle=":", label="EWC + Prototype")

ax1.axvline(x=2000, color="#8b949e", linestyle="--", alpha=0.5, label="Phase 2 Start")
ax1.set_title("Catastrophic Forgetting (Validation Loss A)", pad=20, fontsize=14, fontweight="bold", color="#ffffff")
ax1.set_xlabel("Training Steps")
ax1.set_ylabel("Cross Entropy Loss")
ax1.set_ylim(1.0, 15.0)
ax1.grid(True, linestyle="--", alpha=0.5)
ax1.legend(frameon=True, facecolor="#0d1117", edgecolor="#30363d")

# Plot 2: Task B (Plasticity)
ax2.plot(steps_B, nano_val_B, color="#f85149", linewidth=2.5, alpha=0.9, label="NanoGPT")
ax2.plot(steps_B, massive_val_B, color="#2ea043", linewidth=2.5, alpha=0.9, label="Massive Pure")
ax2.plot(steps_B, poly_val_B, color="#58a6ff", linewidth=2.5, alpha=0.9, linestyle="--", label="Poly-MoE")
ax2.plot(steps_B, ewc_val_B, color="#d2a8ff", linewidth=2.5, alpha=0.9, linestyle=":", label="EWC + Prototype")

ax2.set_title("Plasticity (Validation Loss B)", pad=20, fontsize=14, fontweight="bold", color="#ffffff")
ax2.set_xlabel("Training Steps")
ax2.set_ylabel("Cross Entropy Loss")
ax2.set_ylim(1.0, 10.0)
ax2.grid(True, linestyle="--", alpha=0.5)
ax2.legend(frameon=True, facecolor="#0d1117", edgecolor="#30363d")

plt.tight_layout()
print("Chart generated: continual_learning_chart.png")
