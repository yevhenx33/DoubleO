"""
Extended Continual Learning Gauntlet: 4 tasks, 6-layer models, periodic evaluation.

Protocol:
    Phase 0: Pre-train on all 4 tasks mixed       (3000 steps)
    Phase 1: Freeze non-MLP. Train Task 0 (Math)  (2000 steps)
    Phase 2: Train Task 1 (Code)                   (2000 steps)
    Phase 3: Train Task 2 (Logic)                  (2000 steps)
    Phase 4: Train Task 3 (Spell)                  (2000 steps)

Evaluates all 4 tasks every EVAL_INTERVAL steps.
Exports results_extended.json and comparison_chart_extended.png.

Estimated runtime: 1.5-3 hours on CPU.
"""
import os
import sys
import json
import time
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from model_extended import (
    StandardGPT, StandardGPTConfig,
    DoubleOGPT, DoubleOGPTConfig,
)

# ============================================================
# Configuration
# ============================================================
PRETRAIN_STEPS = 3000
TASK_STEPS = 2000
EVAL_INTERVAL = 100
EVAL_BATCHES = 30
BATCH_SIZE = 16
BLOCK_SIZE = 128
LR = 3e-4
NUM_TASKS = 4
TASK_NAMES = ["Math", "Code", "Logic", "Spell"]

# ============================================================
# Dataset
# ============================================================

class TextDataset(Dataset):
    def __init__(self, filepath, stoi, block_size):
        with open(filepath, 'r') as f:
            lines = [json.loads(line)['text'] for line in f]
        self.data = []
        for line in lines:
            tokens = [stoi.get(c, 0) for c in line]
            tokens.append(stoi['<|endoftext|>'])
            self.data.extend(tokens)
        self.data = torch.tensor(self.data, dtype=torch.long)
        self.block_size = block_size

    def __len__(self):
        return max(0, len(self.data) - self.block_size)

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.block_size]
        y = self.data[idx + 1:idx + self.block_size + 1]
        return x, y


def make_loaders(data_dir, stoi):
    loaders = []
    for name in ["math", "code", "logic", "spell"]:
        ds = TextDataset(os.path.join(data_dir, f"{name}.jsonl"), stoi, BLOCK_SIZE)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        loaders.append(loader)
    return loaders


def infinite_iter(loader):
    while True:
        for batch in loader:
            yield batch


# ============================================================
# Evaluation
# ============================================================

def evaluate_all(model, loaders, device):
    """Evaluate on all tasks, return list of losses."""
    model.eval()
    losses = []
    for task_idx, loader in enumerate(loaders):
        model.set_task(task_idx)
        total = 0.0
        with torch.no_grad():
            for i, (x, y) in enumerate(loader):
                if i >= EVAL_BATCHES:
                    break
                x, y = x.to(device), y.to(device)
                _, loss = model(x, y)
                total += loss.item()
        losses.append(total / EVAL_BATCHES)
    model.train()
    return losses


# ============================================================
# Training
# ============================================================

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def run_gauntlet(model_name, model, loaders, device):
    print(f"\n{'=' * 60}")
    print(f"  GAUNTLET: {model_name}")
    total, trainable = count_params(model)
    print(f"  Parameters: {total:,} total, {trainable:,} trainable")
    print(f"{'=' * 60}\n")

    # Result storage: history[phase_name] = list of (step, [loss_per_task])
    history = {}
    iters = [infinite_iter(loader) for loader in loaders]

    # ---- Phase 0: Pre-train (mixed) ----
    phase = "pretrain"
    history[phase] = []
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    model.train()

    t0 = time.time()
    for step in range(PRETRAIN_STEPS):
        task_idx = step % NUM_TASKS
        model.set_task(task_idx)
        x, y = next(iters[task_idx])
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % EVAL_INTERVAL == 0:
            losses = evaluate_all(model, loaders, device)
            history[phase].append((step, losses))
            elapsed = time.time() - t0
            eta = (elapsed / (step + 1)) * (PRETRAIN_STEPS - step - 1)
            print(f"  [{model_name}] pretrain step {step}/{PRETRAIN_STEPS}  "
                  f"losses={[f'{l:.3f}' for l in losses]}  "
                  f"elapsed={elapsed:.0f}s  eta={eta:.0f}s")

    # Final eval for pretrain
    losses = evaluate_all(model, loaders, device)
    history[phase].append((PRETRAIN_STEPS, losses))
    pretrain_time = time.time() - t0
    print(f"  [{model_name}] pretrain done in {pretrain_time:.0f}s. "
          f"Final losses: {[f'{l:.4f}' for l in losses]}")

    # ---- Phase 1: Freeze non-MLP ----
    for name, param in model.named_parameters():
        if 'mlp' not in name:
            param.requires_grad = False
    _, trainable_after = count_params(model)
    print(f"\n  Froze non-MLP params. Trainable: {trainable_after:,}")
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR
    )

    # ---- Phases 2-5: Sequential task training ----
    for task_idx in range(NUM_TASKS):
        phase = f"task_{task_idx}"
        history[phase] = []
        model.set_task(task_idx)
        task_iter = infinite_iter(loaders[task_idx])

        t_task = time.time()
        for step in range(TASK_STEPS):
            x, y = next(task_iter)
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            _, loss = model(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if step % EVAL_INTERVAL == 0:
                losses = evaluate_all(model, loaders, device)
                history[phase].append((step, losses))
                elapsed = time.time() - t_task
                eta = (elapsed / (step + 1)) * (TASK_STEPS - step - 1)
                print(f"  [{model_name}] {TASK_NAMES[task_idx]} step {step}/{TASK_STEPS}  "
                      f"losses={[f'{l:.3f}' for l in losses]}  "
                      f"elapsed={elapsed:.0f}s  eta={eta:.0f}s")

        losses = evaluate_all(model, loaders, device)
        history[phase].append((TASK_STEPS, losses))
        task_time = time.time() - t_task
        print(f"  [{model_name}] {TASK_NAMES[task_idx]} done in {task_time:.0f}s. "
              f"Final losses: {[f'{l:.4f}' for l in losses]}")

    return history


# ============================================================
# Plotting
# ============================================================

def plot_results(results, out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    COLORS = ['#2563EB', '#16A34A', '#DC2626', '#D97706']
    TASK_SHORT = TASK_NAMES

    fig, axes = plt.subplots(2, 3, figsize=(22, 12))
    fig.suptitle('Extended Continual Learning Gauntlet: NanoDoubleO vs StandardGPT',
                 fontsize=16, fontweight='bold', y=0.98)

    # ---- Top row: per-task loss curves across all phases ----
    for task_idx in range(NUM_TASKS):
        ax = axes[0, task_idx] if task_idx < 3 else axes[1, 2]

        for model_name, style in [("StandardGPT", "--"), ("NanoDoubleO", "-")]:
            hist = results[model_name]
            steps_global = []
            losses = []
            offset = 0
            phase_boundaries = []
            for phase_key in ["pretrain"] + [f"task_{i}" for i in range(NUM_TASKS)]:
                for (s, l) in hist[phase_key]:
                    steps_global.append(s + offset)
                    losses.append(l[task_idx])
                if hist[phase_key]:
                    last_step = hist[phase_key][-1][0]
                    phase_boundaries.append(offset + last_step)
                    offset += last_step

            alpha = 1.0 if style == "-" else 0.6
            label = f"{model_name}"
            ax.plot(steps_global, losses, style, color=COLORS[task_idx],
                    linewidth=2 if style == "-" else 1.5,
                    alpha=alpha, label=label)

            # Draw phase boundaries
            if style == "-":  # Only once
                for pb in phase_boundaries[:-1]:
                    ax.axvline(x=pb, color='#CCCCCC', linestyle=':', linewidth=0.8)

        ax.set_title(f'Task {task_idx}: {TASK_SHORT[task_idx]}', fontsize=13, fontweight='bold')
        ax.set_xlabel('Cumulative Steps')
        ax.set_ylabel('Cross-Entropy Loss')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # ---- Bottom-left: Forgetting bar chart ----
    ax_bar = axes[1, 0]
    # Forgetting = loss after all tasks done - loss right after that task was trained
    forgetting_std = []
    forgetting_do = []
    for task_idx in range(NUM_TASKS):
        # Loss right after training this task
        hist_std = results["StandardGPT"]
        hist_do = results["NanoDoubleO"]

        loss_after_own_std = hist_std[f"task_{task_idx}"][-1][1][task_idx]
        loss_after_own_do = hist_do[f"task_{task_idx}"][-1][1][task_idx]

        # Loss after ALL tasks are done
        loss_final_std = hist_std[f"task_{NUM_TASKS - 1}"][-1][1][task_idx]
        loss_final_do = hist_do[f"task_{NUM_TASKS - 1}"][-1][1][task_idx]

        forgetting_std.append(loss_final_std - loss_after_own_std)
        forgetting_do.append(loss_final_do - loss_after_own_do)

    x_pos = np.arange(NUM_TASKS)
    width = 0.35
    bars1 = ax_bar.bar(x_pos - width / 2, forgetting_std, width,
                       color='#DC2626', alpha=0.8, label='StandardGPT')
    bars2 = ax_bar.bar(x_pos + width / 2, forgetting_do, width,
                       color='#2563EB', alpha=0.8, label='NanoDoubleO')

    for bar, val in zip(bars1, forgetting_std):
        ax_bar.annotate(f'{val:+.3f}', (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        textcoords="offset points", xytext=(0, 5), ha='center', fontsize=8,
                        fontweight='bold', color='#DC2626')
    for bar, val in zip(bars2, forgetting_do):
        ax_bar.annotate(f'{val:+.3f}', (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        textcoords="offset points", xytext=(0, 5), ha='center', fontsize=8,
                        fontweight='bold', color='#2563EB')

    ax_bar.set_title('Per-Task Forgetting Penalty (Δ Loss)', fontsize=13, fontweight='bold')
    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(TASK_SHORT)
    ax_bar.set_ylabel('Loss Degradation')
    ax_bar.legend(fontsize=9)
    ax_bar.grid(True, axis='y', alpha=0.3)
    ax_bar.spines['top'].set_visible(False)
    ax_bar.spines['right'].set_visible(False)
    ax_bar.axhline(y=0, color='black', linewidth=0.5)

    # ---- Bottom-center: Summary table ----
    ax_table = axes[1, 1]
    ax_table.axis('off')

    # Build summary data
    table_data = []
    for task_idx in range(NUM_TASKS):
        f_std = forgetting_std[task_idx]
        f_do = forgetting_do[task_idx]
        reduction = ((f_std - f_do) / abs(f_std) * 100) if abs(f_std) > 1e-6 else 0
        table_data.append([
            TASK_SHORT[task_idx],
            f'{f_std:+.4f}',
            f'{f_do:+.4f}',
            f'{reduction:.1f}%'
        ])

    # Average
    avg_std = np.mean(forgetting_std)
    avg_do = np.mean(forgetting_do)
    avg_red = ((avg_std - avg_do) / abs(avg_std) * 100) if abs(avg_std) > 1e-6 else 0
    table_data.append(['Average', f'{avg_std:+.4f}', f'{avg_do:+.4f}', f'{avg_red:.1f}%'])

    table = ax_table.table(
        cellText=table_data,
        colLabels=['Task', 'Std Forget', 'DoubleO Forget', 'Reduction'],
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    ax_table.set_title('Forgetting Summary', fontsize=13, fontweight='bold', pad=20)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = os.path.join(out_dir, 'comparison_chart_extended.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\nSaved chart to {out_path}")


# ============================================================
# Main
# ============================================================

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'extended')
    if not os.path.exists(os.path.join(data_dir, "vocab.json")):
        print("ERROR: Extended datasets not found. Run data/data_gen_extended.py first.")
        sys.exit(1)

    with open(os.path.join(data_dir, "vocab.json"), 'r') as f:
        vocab = json.load(f)
    stoi = vocab['stoi']
    vocab_size = vocab['vocab_size']
    print(f"Vocab size: {vocab_size}")

    loaders = make_loaders(data_dir, stoi)
    print(f"Datasets loaded. Samples per task: "
          f"{[len(l.dataset) for l in loaders]}")

    results = {}

    # ---- 1. StandardGPT Baseline ----
    std_config = StandardGPTConfig()
    std_config.vocab_size = vocab_size
    std_model = StandardGPT(std_config).to(device)
    t0 = time.time()
    results["StandardGPT"] = run_gauntlet("StandardGPT", std_model, loaders, device)
    std_time = time.time() - t0
    print(f"\n  StandardGPT total time: {std_time:.0f}s ({std_time/60:.1f}min)")

    # ---- 2. NanoDoubleO ----
    do_config = DoubleOGPTConfig()
    do_config.vocab_size = vocab_size
    do_model = DoubleOGPT(do_config).to(device)
    t0 = time.time()
    results["NanoDoubleO"] = run_gauntlet("NanoDoubleO", do_model, loaders, device)
    do_time = time.time() - t0
    print(f"\n  NanoDoubleO total time: {do_time:.0f}s ({do_time/60:.1f}min)")

    # ---- Save results ----
    out_path = os.path.join(data_dir, "results_extended.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {out_path}")

    # ---- Plot ----
    plot_results(results, os.path.join(os.path.dirname(__file__), '..'))

    # ---- Print summary ----
    print(f"\n{'=' * 60}")
    print("  FINAL SUMMARY")
    print(f"{'=' * 60}")
    for task_idx in range(NUM_TASKS):
        hist_std = results["StandardGPT"]
        hist_do = results["NanoDoubleO"]
        loss_after_own_std = hist_std[f"task_{task_idx}"][-1][1][task_idx]
        loss_after_own_do = hist_do[f"task_{task_idx}"][-1][1][task_idx]
        loss_final_std = hist_std[f"task_{NUM_TASKS - 1}"][-1][1][task_idx]
        loss_final_do = hist_do[f"task_{NUM_TASKS - 1}"][-1][1][task_idx]
        f_std = loss_final_std - loss_after_own_std
        f_do = loss_final_do - loss_after_own_do
        print(f"  {TASK_NAMES[task_idx]:8s}  Baseline forget: {f_std:+.4f}  "
              f"DoubleO forget: {f_do:+.4f}")
    print(f"\n  Total runtime: {(std_time + do_time)/60:.1f} minutes")


if __name__ == "__main__":
    main()
