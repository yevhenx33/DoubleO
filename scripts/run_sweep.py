#!/usr/bin/env python3
"""
Overnight Sweep: C-lite configuration.

Runs 8 experiment arms in parallel (4 workers × 2 threads each = 8 cores).

Arms:
  Size sweep (4 tasks):
    1. small_std       96-dim, 3L, StandardGPT
    2. small_do        96-dim, 3L, DoubleO
    3. med_std        128-dim, 4L, StandardGPT
    4. med_do         128-dim, 4L, DoubleO
  Task-count sweep (128-dim, DoubleO):
    5. med_do_2task   128-dim, 4L, 2 tasks
    6. med_do_3task   128-dim, 4L, 3 tasks
  Ablations (128-dim, 4 tasks):
    7. med_do_noclamp 128-dim, 4L, clamp disabled
    8. med_do_ln      128-dim, 4L, LayerNorm instead of RMSNorm

Estimated wall time: 4-6 hours on 8-core CPU.

Usage:
    python3 scripts/run_sweep.py
"""
import os
import sys
import json
import time
import multiprocessing as mp

# ---------------------------------------------------------------------------
# Arm definitions
# ---------------------------------------------------------------------------
ARMS = [
    # Size sweep
    {"name": "small_std",      "arch": "standard", "n_embd": 96,  "n_layer": 3, "n_head": 4, "block_size": 64,  "num_tasks": 4, "clamp_val": 2.0, "norm_type": "rmsnorm"},
    {"name": "small_do",       "arch": "doubleo",  "n_embd": 96,  "n_layer": 3, "n_head": 4, "block_size": 64,  "num_tasks": 4, "clamp_val": 2.0, "norm_type": "rmsnorm"},
    {"name": "med_std",        "arch": "standard", "n_embd": 128, "n_layer": 4, "n_head": 4, "block_size": 96,  "num_tasks": 4, "clamp_val": 2.0, "norm_type": "rmsnorm"},
    {"name": "med_do",         "arch": "doubleo",  "n_embd": 128, "n_layer": 4, "n_head": 4, "block_size": 96,  "num_tasks": 4, "clamp_val": 2.0, "norm_type": "rmsnorm"},
    # Task-count sweep
    {"name": "med_do_2task",   "arch": "doubleo",  "n_embd": 128, "n_layer": 4, "n_head": 4, "block_size": 96,  "num_tasks": 2, "clamp_val": 2.0, "norm_type": "rmsnorm"},
    {"name": "med_do_3task",   "arch": "doubleo",  "n_embd": 128, "n_layer": 4, "n_head": 4, "block_size": 96,  "num_tasks": 3, "clamp_val": 2.0, "norm_type": "rmsnorm"},
    # Ablations
    {"name": "med_do_noclamp", "arch": "doubleo",  "n_embd": 128, "n_layer": 4, "n_head": 4, "block_size": 96,  "num_tasks": 4, "clamp_val": 0.0, "norm_type": "rmsnorm"},
    {"name": "med_do_ln",      "arch": "doubleo",  "n_embd": 128, "n_layer": 4, "n_head": 4, "block_size": 96,  "num_tasks": 4, "clamp_val": 2.0, "norm_type": "layernorm"},
]

# Training config
PRETRAIN_STEPS = 2000
TASK_STEPS = 1500
EVAL_INTERVAL = 100
EVAL_BATCHES = 20
BATCH_SIZE = 16
LR = 3e-4
TASK_NAMES = ["Math", "Code", "Logic", "Spell"]
MAX_WORKERS = 4
THREADS_PER_WORKER = 2

# ---------------------------------------------------------------------------
# Worker function (runs in a subprocess)
# ---------------------------------------------------------------------------
def run_arm(arm_cfg):
    """Run a single experiment arm. Called in a spawned subprocess."""
    import torch
    torch.set_num_threads(THREADS_PER_WORKER)

    src_dir = os.path.join(os.path.dirname(__file__), '..', 'src')
    sys.path.insert(0, os.path.abspath(src_dir))
    from model_extended import (
        StandardGPT, StandardGPTConfig,
        DoubleOGPT, DoubleOGPTConfig,
    )
    from torch.utils.data import Dataset, DataLoader

    name = arm_cfg["name"]
    num_tasks = arm_cfg["num_tasks"]
    block_size = arm_cfg["block_size"]
    device = 'cpu'

    log_path = os.path.join(out_dir_global, f"{name}.log")
    log_f = open(log_path, 'w')

    def log(msg):
        line = f"[{name}] {msg}"
        print(line, flush=True)
        log_f.write(line + "\n")
        log_f.flush()

    # --- Dataset ---
    class TextDataset(Dataset):
        def __init__(self, filepath, stoi, bs):
            with open(filepath, 'r') as f:
                lines = [json.loads(l)['text'] for l in f]
            self.data = []
            for line in lines:
                tokens = [stoi.get(c, 0) for c in line]
                tokens.append(stoi['<|endoftext|>'])
                self.data.extend(tokens)
            self.data = torch.tensor(self.data, dtype=torch.long)
            self.block_size = bs
        def __len__(self):
            return max(0, len(self.data) - self.block_size)
        def __getitem__(self, idx):
            x = self.data[idx:idx + self.block_size]
            y = self.data[idx + 1:idx + self.block_size + 1]
            return x, y

    def infinite_iter(loader):
        while True:
            for batch in loader:
                yield batch

    def evaluate_all(model, loaders):
        model.eval()
        losses = []
        for task_idx, loader in enumerate(loaders):
            model.set_task(task_idx)
            total = 0.0
            with torch.no_grad():
                for i, (x, y) in enumerate(loader):
                    if i >= EVAL_BATCHES: break
                    _, loss = model(x, y)
                    total += loss.item()
            losses.append(total / EVAL_BATCHES)
        model.train()
        return losses

    # --- Load data ---
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'extended')
    with open(os.path.join(data_dir, "vocab.json")) as f:
        vocab = json.load(f)
    stoi = vocab['stoi']
    vocab_size = vocab['vocab_size']

    task_files = ["math", "code", "logic", "spell"][:num_tasks]
    loaders = []
    for tf in task_files:
        ds = TextDataset(os.path.join(data_dir, f"{tf}.jsonl"), stoi, block_size)
        loaders.append(DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True))

    # --- Build model ---
    if arm_cfg["arch"] == "standard":
        cfg = StandardGPTConfig()
        cfg.vocab_size = vocab_size
        cfg.n_embd = arm_cfg["n_embd"]
        cfg.n_layer = arm_cfg["n_layer"]
        cfg.n_head = arm_cfg["n_head"]
        cfg.block_size = block_size
        model = StandardGPT(cfg)
    else:
        cfg = DoubleOGPTConfig()
        cfg.vocab_size = vocab_size
        cfg.n_embd = arm_cfg["n_embd"]
        cfg.n_layer = arm_cfg["n_layer"]
        cfg.n_head = arm_cfg["n_head"]
        cfg.block_size = block_size
        cfg.clamp_val = arm_cfg["clamp_val"]
        cfg.norm_type = arm_cfg["norm_type"]
        model = DoubleOGPT(cfg)

    total_params = sum(p.numel() for p in model.parameters())
    log(f"START | arch={arm_cfg['arch']} dim={arm_cfg['n_embd']} layers={arm_cfg['n_layer']} "
        f"tasks={num_tasks} params={total_params:,} clamp={arm_cfg['clamp_val']} norm={arm_cfg['norm_type']}")

    history = {}
    iters = [infinite_iter(ld) for ld in loaders]

    # --- Phase 0: Pretrain (mixed) ---
    phase = "pretrain"
    history[phase] = []
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    model.train()
    t0 = time.time()

    for step in range(PRETRAIN_STEPS):
        task_idx = step % num_tasks
        model.set_task(task_idx)
        x, y = next(iters[task_idx])
        optimizer.zero_grad()
        _, loss = model(x, y)
        if not torch.isfinite(loss):
            log(f"WARNING: NaN/Inf loss at pretrain step {step}, task {task_idx}")
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % EVAL_INTERVAL == 0:
            losses = evaluate_all(model, loaders)
            history[phase].append((step, losses))
            elapsed = time.time() - t0
            eta = (elapsed / (step + 1)) * (PRETRAIN_STEPS - step - 1) if step > 0 else 0
            log(f"pretrain {step}/{PRETRAIN_STEPS} losses={[f'{l:.3f}' for l in losses]} "
                f"elapsed={elapsed:.0f}s eta={eta:.0f}s")

    losses = evaluate_all(model, loaders)
    history[phase].append((PRETRAIN_STEPS, losses))
    log(f"pretrain DONE in {time.time()-t0:.0f}s final={[f'{l:.4f}' for l in losses]}")

    # --- Freeze non-MLP ---
    for pname, param in model.named_parameters():
        if 'mlp' not in pname:
            param.requires_grad = False
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

    # --- Sequential task training ---
    for task_idx in range(num_tasks):
        phase = f"task_{task_idx}"
        history[phase] = []
        model.set_task(task_idx)
        task_iter = infinite_iter(loaders[task_idx])
        t_task = time.time()

        for step in range(TASK_STEPS):
            x, y = next(task_iter)
            optimizer.zero_grad()
            _, loss = model(x, y)
            if not torch.isfinite(loss):
                log(f"WARNING: NaN/Inf at {TASK_NAMES[task_idx]} step {step}")
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if step % EVAL_INTERVAL == 0:
                losses = evaluate_all(model, loaders)
                history[phase].append((step, losses))
                elapsed = time.time() - t_task
                eta = (elapsed / (step + 1)) * (TASK_STEPS - step - 1) if step > 0 else 0
                log(f"{TASK_NAMES[task_idx]} {step}/{TASK_STEPS} "
                    f"losses={[f'{l:.3f}' for l in losses]} elapsed={elapsed:.0f}s eta={eta:.0f}s")

        losses = evaluate_all(model, loaders)
        history[phase].append((TASK_STEPS, losses))
        log(f"{TASK_NAMES[task_idx]} DONE in {time.time()-t_task:.0f}s final={[f'{l:.4f}' for l in losses]}")

    total_time = time.time() - t0
    log(f"FINISHED in {total_time:.0f}s ({total_time/60:.1f}min)")

    # --- Compute forgetting ---
    forgetting = {}
    last_phase = f"task_{num_tasks - 1}"
    for task_idx in range(num_tasks):
        after_own = history[f"task_{task_idx}"][-1][1][task_idx]
        after_all = history[last_phase][-1][1][task_idx]
        forgetting[TASK_NAMES[task_idx]] = after_all - after_own

    result = {
        "config": arm_cfg,
        "history": history,
        "forgetting": forgetting,
        "total_time_s": total_time,
        "total_params": total_params,
    }

    result_path = os.path.join(out_dir_global, f"{name}.json")
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)
    log(f"Saved results to {result_path}")
    log_f.close()
    return name, forgetting, total_time


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_sweep(out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    results = {}
    for arm in ARMS:
        path = os.path.join(out_dir, f"{arm['name']}.json")
        if os.path.exists(path):
            with open(path) as f:
                results[arm['name']] = json.load(f)

    if not results:
        print("No results to plot.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle('NanoDoubleO C-Lite Sweep: Overnight Results', fontsize=16, fontweight='bold', y=0.98)

    COLORS = {'small_std': '#DC2626', 'small_do': '#2563EB',
              'med_std': '#B91C1C', 'med_do': '#1D4ED8',
              'med_do_2task': '#7C3AED', 'med_do_3task': '#059669',
              'med_do_noclamp': '#D97706', 'med_do_ln': '#0891B2'}
    LABELS = {'small_std': 'Std 96d', 'small_do': 'DO 96d',
              'med_std': 'Std 128d', 'med_do': 'DO 128d',
              'med_do_2task': 'DO 128d 2T', 'med_do_3task': 'DO 128d 3T',
              'med_do_noclamp': 'DO NoClamp', 'med_do_ln': 'DO LayerNorm'}

    # --- Panel 1: Size comparison forgetting bars ---
    ax = axes[0, 0]
    size_arms = ['small_std', 'small_do', 'med_std', 'med_do']
    available = [a for a in size_arms if a in results]
    if available:
        x_pos = np.arange(len(available))
        vals = []
        colors = []
        for a in available:
            fgt = results[a]['forgetting']
            avg = np.mean(list(fgt.values()))
            vals.append(avg)
            colors.append(COLORS.get(a, '#888888'))
        bars = ax.bar(x_pos, vals, color=colors, width=0.6)
        for bar, val in zip(bars, vals):
            ax.annotate(f'{val:+.4f}', (bar.get_x() + bar.get_width()/2, bar.get_height()),
                        textcoords="offset points", xytext=(0, 5), ha='center', fontsize=9, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([LABELS[a] for a in available], fontsize=9)
        ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_title('Size Comparison: Avg Forgetting', fontsize=13, fontweight='bold')
    ax.set_ylabel('Avg Forgetting (Δ Loss)')
    ax.grid(True, axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # --- Panel 2: Task-count sweep ---
    ax = axes[0, 1]
    task_arms = ['med_do_2task', 'med_do_3task', 'med_do']
    available = [a for a in task_arms if a in results]
    if available:
        x_pos = np.arange(len(available))
        vals = [np.mean(list(results[a]['forgetting'].values())) for a in available]
        colors = [COLORS.get(a, '#888888') for a in available]
        labels = ['2 Tasks', '3 Tasks', '4 Tasks'][:len(available)]
        bars = ax.bar(x_pos, vals, color=colors, width=0.6)
        for bar, val in zip(bars, vals):
            ax.annotate(f'{val:+.4f}', (bar.get_x() + bar.get_width()/2, bar.get_height()),
                        textcoords="offset points", xytext=(0, 5), ha='center', fontsize=9, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, fontsize=9)
        ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_title('Task-Count Sweep (128d DoubleO)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Avg Forgetting (Δ Loss)')
    ax.grid(True, axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # --- Panel 3: Ablation comparison ---
    ax = axes[1, 0]
    ablation_arms = ['med_do', 'med_do_noclamp', 'med_do_ln']
    available = [a for a in ablation_arms if a in results]
    if available:
        x_pos = np.arange(len(available))
        vals = [np.mean(list(results[a]['forgetting'].values())) for a in available]
        colors = [COLORS.get(a, '#888888') for a in available]
        labels = [LABELS.get(a, a) for a in available]
        bars = ax.bar(x_pos, vals, color=colors, width=0.6)
        for bar, val in zip(bars, vals):
            ax.annotate(f'{val:+.4f}', (bar.get_x() + bar.get_width()/2, bar.get_height()),
                        textcoords="offset points", xytext=(0, 5), ha='center', fontsize=9, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, fontsize=9)
        ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_title('Ablation: Clamp & Norm Effect', fontsize=13, fontweight='bold')
    ax.set_ylabel('Avg Forgetting (Δ Loss)')
    ax.grid(True, axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # --- Panel 4: Summary table ---
    ax = axes[1, 1]
    ax.axis('off')
    table_data = []
    for arm in ARMS:
        n = arm['name']
        if n not in results:
            continue
        r = results[n]
        fgt = r['forgetting']
        avg_f = np.mean(list(fgt.values()))
        t_min = r['total_time_s'] / 60
        table_data.append([
            LABELS.get(n, n),
            f"{r['total_params']:,}",
            str(arm['num_tasks']),
            f"{avg_f:+.4f}",
            f"{t_min:.0f}m"
        ])
    if table_data:
        table = ax.table(
            cellText=table_data,
            colLabels=['Arm', 'Params', 'Tasks', 'Avg Forget', 'Time'],
            loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.8)
    ax.set_title('Summary', fontsize=13, fontweight='bold', pad=20)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    chart_path = os.path.join(out_dir, 'sweep_results.png')
    plt.savefig(chart_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\nSaved chart to {chart_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
out_dir_global = os.path.join(os.path.dirname(__file__), '..', 'data', 'sweep')

if __name__ == '__main__':
    mp.set_start_method('spawn')

    os.makedirs(out_dir_global, exist_ok=True)

    print(f"=" * 60)
    print(f"  NanoDoubleO C-Lite Sweep")
    print(f"  Arms: {len(ARMS)}")
    print(f"  Workers: {MAX_WORKERS} × {THREADS_PER_WORKER} threads = {MAX_WORKERS * THREADS_PER_WORKER} cores")
    print(f"  Pretrain steps: {PRETRAIN_STEPS}, Task steps: {TASK_STEPS}")
    print(f"  Output: {os.path.abspath(out_dir_global)}")
    print(f"=" * 60)

    for i, arm in enumerate(ARMS):
        print(f"  [{i+1}] {arm['name']:20s} arch={arm['arch']:8s} dim={arm['n_embd']} "
              f"layers={arm['n_layer']} tasks={arm['num_tasks']} "
              f"clamp={arm['clamp_val']} norm={arm['norm_type']}")
    print()

    t_start = time.time()

    with mp.Pool(MAX_WORKERS) as pool:
        results = pool.map(run_arm, ARMS)

    t_total = time.time() - t_start

    print(f"\n{'=' * 60}")
    print(f"  ALL ARMS COMPLETE — {t_total/60:.1f} minutes total")
    print(f"{'=' * 60}")
    for name, forgetting, arm_time in results:
        avg = sum(forgetting.values()) / len(forgetting)
        print(f"  {name:20s} avg_forget={avg:+.4f} time={arm_time/60:.1f}min")

    # Aggregate all results
    agg = {}
    for arm in ARMS:
        path = os.path.join(out_dir_global, f"{arm['name']}.json")
        if os.path.exists(path):
            with open(path) as f:
                agg[arm['name']] = json.load(f)
    with open(os.path.join(out_dir_global, "sweep_all.json"), 'w') as f:
        json.dump(agg, f, indent=2)

    print(f"\nAggregated results: {out_dir_global}/sweep_all.json")

    plot_sweep(out_dir_global)
    print("\nDone.")
