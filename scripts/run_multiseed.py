#!/usr/bin/env python3
"""
Multi-Seed Validation: 4 key arms × 5 seeds = 20 runs.

Produces mean ± std, 95% confidence intervals, and Welch's t-test
p-values for the central claim (DoubleO forgetting < Baseline forgetting).

Arms:
  1. small_std  (96d, 3L, StandardGPT)
  2. small_do   (96d, 3L, DoubleO)
  3. med_std   (128d, 4L, StandardGPT)
  4. med_do    (128d, 4L, DoubleO)

Estimated wall time: ~1.5 hours on 8-core CPU.
"""
import os, sys, json, time, math
import multiprocessing as mp

SEEDS = [42, 137, 256, 512, 1024]

CONFIGS = [
    {"name": "small_std", "arch": "standard", "n_embd": 96,  "n_layer": 3, "n_head": 4, "block_size": 64,  "num_tasks": 4, "clamp_val": 2.0, "norm_type": "rmsnorm"},
    {"name": "small_do",  "arch": "doubleo",  "n_embd": 96,  "n_layer": 3, "n_head": 4, "block_size": 64,  "num_tasks": 4, "clamp_val": 2.0, "norm_type": "rmsnorm"},
    {"name": "med_std",   "arch": "standard", "n_embd": 128, "n_layer": 4, "n_head": 4, "block_size": 96,  "num_tasks": 4, "clamp_val": 2.0, "norm_type": "rmsnorm"},
    {"name": "med_do",    "arch": "doubleo",  "n_embd": 128, "n_layer": 4, "n_head": 4, "block_size": 96,  "num_tasks": 4, "clamp_val": 2.0, "norm_type": "rmsnorm"},
]

PRETRAIN_STEPS = 2000
TASK_STEPS = 1500
EVAL_INTERVAL = 200
EVAL_BATCHES = 20
BATCH_SIZE = 16
LR = 3e-4
TASK_NAMES = ["Math", "Code", "Logic", "Spell"]
MAX_WORKERS = 4
THREADS_PER_WORKER = 2

out_dir_global = os.path.join(os.path.dirname(__file__), '..', 'data', 'multiseed')


def run_arm(args):
    """Run a single (config, seed) pair."""
    arm_cfg, seed = args
    import torch
    torch.set_num_threads(THREADS_PER_WORKER)
    torch.manual_seed(seed)

    src_dir = os.path.join(os.path.dirname(__file__), '..', 'src')
    sys.path.insert(0, os.path.abspath(src_dir))
    from model_extended import StandardGPT, StandardGPTConfig, DoubleOGPT, DoubleOGPTConfig
    from torch.utils.data import Dataset, DataLoader

    name = arm_cfg["name"]
    run_id = f"{name}_s{seed}"
    num_tasks = arm_cfg["num_tasks"]
    block_size = arm_cfg["block_size"]

    log_path = os.path.join(out_dir_global, f"{run_id}.log")
    log_f = open(log_path, 'w')
    def log(msg):
        line = f"[{run_id}] {msg}"
        print(line, flush=True)
        log_f.write(line + "\n")
        log_f.flush()

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
        for ti, loader in enumerate(loaders):
            model.set_task(ti)
            total = 0.0
            with torch.no_grad():
                for i, (x, y) in enumerate(loader):
                    if i >= EVAL_BATCHES: break
                    _, loss = model(x, y)
                    total += loss.item()
            losses.append(total / EVAL_BATCHES)
        model.train()
        return losses

    # Load data
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'extended')
    with open(os.path.join(data_dir, "vocab.json")) as f:
        vocab = json.load(f)
    stoi, vocab_size = vocab['stoi'], vocab['vocab_size']

    task_files = ["math", "code", "logic", "spell"][:num_tasks]
    gen = torch.Generator().manual_seed(seed)
    loaders = []
    for tf in task_files:
        ds = TextDataset(os.path.join(data_dir, f"{tf}.jsonl"), stoi, block_size)
        loaders.append(DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                                  drop_last=True, generator=gen))

    # Build model (seed already set above)
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
    log(f"START seed={seed} arch={arm_cfg['arch']} dim={arm_cfg['n_embd']} params={total_params:,}")

    iters = [infinite_iter(ld) for ld in loaders]
    t0 = time.time()

    # Phase 0: Pretrain
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    model.train()
    for step in range(PRETRAIN_STEPS):
        ti = step % num_tasks
        model.set_task(ti)
        x, y = next(iters[ti])
        optimizer.zero_grad()
        _, loss = model(x, y)
        if not torch.isfinite(loss):
            log(f"NaN at pretrain step {step}")
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % EVAL_INTERVAL == 0:
            ls = evaluate_all(model, loaders)
            log(f"pretrain {step}/{PRETRAIN_STEPS} losses={[f'{l:.3f}' for l in ls]}")

    log(f"pretrain DONE {time.time()-t0:.0f}s")

    # Freeze non-MLP
    for pn, p in model.named_parameters():
        if 'mlp' not in pn:
            p.requires_grad = False
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

    # Record post-pretrain losses
    post_pretrain = evaluate_all(model, loaders)

    # Sequential tasks
    post_task = {}
    for ti in range(num_tasks):
        model.set_task(ti)
        task_iter = infinite_iter(loaders[ti])
        for step in range(TASK_STEPS):
            x, y = next(task_iter)
            optimizer.zero_grad()
            _, loss = model(x, y)
            if not torch.isfinite(loss):
                log(f"NaN at {TASK_NAMES[ti]} step {step}")
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        post_task[ti] = evaluate_all(model, loaders)
        log(f"{TASK_NAMES[ti]} DONE losses={[f'{l:.4f}' for l in post_task[ti]]}")

    total_time = time.time() - t0

    # Compute forgetting per task
    forgetting = {}
    final_losses = post_task[num_tasks - 1]
    for ti in range(num_tasks):
        after_own = post_task[ti][ti]
        after_all = final_losses[ti]
        forgetting[TASK_NAMES[ti]] = after_all - after_own

    avg_forget = sum(forgetting.values()) / len(forgetting)
    log(f"FINISHED {total_time:.0f}s avg_forget={avg_forget:+.4f} per_task={forgetting}")

    result = {
        "name": name, "seed": seed, "arch": arm_cfg["arch"],
        "n_embd": arm_cfg["n_embd"], "n_layer": arm_cfg["n_layer"],
        "params": total_params, "forgetting": forgetting,
        "avg_forgetting": avg_forget, "total_time_s": total_time,
        "post_pretrain": post_pretrain,
        "final_losses": final_losses,
    }
    with open(os.path.join(out_dir_global, f"{run_id}.json"), 'w') as f:
        json.dump(result, f, indent=2)
    log_f.close()
    return result


def analyze_and_plot(out_dir):
    """Aggregate results, compute statistics, generate chart."""
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy import stats

    # Collect results
    all_results = {}
    for cfg in CONFIGS:
        all_results[cfg["name"]] = []
    for fname in sorted(os.listdir(out_dir)):
        if not fname.endswith('.json') or fname == 'multiseed_summary.json':
            continue
        with open(os.path.join(out_dir, fname)) as f:
            r = json.load(f)
        if r["name"] in all_results:
            all_results[r["name"]].append(r)

    # Compute stats
    summary = {}
    for name, runs in all_results.items():
        if not runs:
            continue
        avgs = [r["avg_forgetting"] for r in runs]
        per_task = {}
        for tn in TASK_NAMES:
            vals = [r["forgetting"].get(tn, 0) for r in runs]
            per_task[tn] = {"mean": np.mean(vals), "std": np.std(vals, ddof=1),
                           "values": vals}
        summary[name] = {
            "n": len(runs),
            "avg_forget_mean": np.mean(avgs),
            "avg_forget_std": np.std(avgs, ddof=1),
            "avg_forget_values": avgs,
            "avg_forget_ci95": 1.96 * np.std(avgs, ddof=1) / np.sqrt(len(avgs)),
            "per_task": per_task,
            "arch": runs[0]["arch"],
            "params": runs[0]["params"],
            "n_embd": runs[0]["n_embd"],
        }

    # T-tests: DoubleO vs Baseline at each scale
    tests = {}
    for scale_label, std_name, do_name in [("96d", "small_std", "small_do"),
                                            ("128d", "med_std", "med_do")]:
        if std_name in summary and do_name in summary:
            std_vals = summary[std_name]["avg_forget_values"]
            do_vals = summary[do_name]["avg_forget_values"]
            t_stat, p_val = stats.ttest_ind(std_vals, do_vals, equal_var=False)
            # Effect size (Cohen's d)
            pooled_std = np.sqrt((np.std(std_vals, ddof=1)**2 + np.std(do_vals, ddof=1)**2) / 2)
            cohens_d = (np.mean(std_vals) - np.mean(do_vals)) / pooled_std if pooled_std > 0 else float('inf')
            tests[scale_label] = {
                "t_stat": t_stat, "p_value": p_val, "cohens_d": cohens_d,
                "std_mean": np.mean(std_vals), "do_mean": np.mean(do_vals),
                "reduction_pct": (1 - np.mean(do_vals) / np.mean(std_vals)) * 100,
            }

    # Save summary
    out = {"summary": {}, "tests": tests}
    for k, v in summary.items():
        out["summary"][k] = {key: val for key, val in v.items()
                             if key != "avg_forget_values" and key != "per_task"}
        out["summary"][k]["avg_forget_values"] = v["avg_forget_values"]
    with open(os.path.join(out_dir, "multiseed_summary.json"), 'w') as f:
        json.dump(out, f, indent=2, default=float)

    # Print stats
    print(f"\n{'='*70}")
    print(f"  MULTI-SEED RESULTS ({len(SEEDS)} seeds)")
    print(f"{'='*70}")
    print(f"{'Arm':15s} {'Mean':>10s} {'Std':>8s} {'95% CI':>14s} {'N':>4s}")
    print(f"{'-'*55}")
    for name in ["small_std", "small_do", "med_std", "med_do"]:
        if name not in summary:
            continue
        s = summary[name]
        ci = s["avg_forget_ci95"]
        print(f"{name:15s} {s['avg_forget_mean']:>+10.4f} {s['avg_forget_std']:>8.4f} "
              f"[{s['avg_forget_mean']-ci:+.3f},{s['avg_forget_mean']+ci:+.3f}] {s['n']:>4d}")

    print(f"\n{'='*70}")
    print(f"  STATISTICAL TESTS (Welch's t-test, two-sided)")
    print(f"{'='*70}")
    for label, t in tests.items():
        sig = "***" if t["p_value"] < 0.001 else "**" if t["p_value"] < 0.01 else "*" if t["p_value"] < 0.05 else "ns"
        print(f"  {label}: Baseline={t['std_mean']:+.4f} vs DoubleO={t['do_mean']:+.4f} "
              f"reduction={t['reduction_pct']:.1f}% "
              f"t={t['t_stat']:.2f} p={t['p_value']:.6f} d={t['cohens_d']:.2f} {sig}")

    # --- Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'NanoDoubleO Multi-Seed Validation (N={len(SEEDS)} per arm)',
                 fontsize=15, fontweight='bold', y=1.02)

    COLORS = {'small_std': '#DC2626', 'small_do': '#2563EB',
              'med_std': '#B91C1C', 'med_do': '#1D4ED8'}
    LABELS = {'small_std': 'Std 96d', 'small_do': 'DO 96d',
              'med_std': 'Std 128d', 'med_do': 'DO 128d'}

    # Panel 1: Bar chart with error bars
    ax = axes[0]
    arms_order = ["small_std", "small_do", "med_std", "med_do"]
    available = [a for a in arms_order if a in summary]
    x_pos = np.arange(len(available))
    means = [summary[a]["avg_forget_mean"] for a in available]
    stds = [summary[a]["avg_forget_std"] for a in available]
    colors = [COLORS[a] for a in available]
    bars = ax.bar(x_pos, means, yerr=stds, color=colors, width=0.6,
                  capsize=5, edgecolor='black', linewidth=0.5)
    for bar, m, s in zip(bars, means, stds):
        ax.annotate(f'{m:+.3f}\n±{s:.3f}',
                    (bar.get_x() + bar.get_width()/2, bar.get_height() + s),
                    textcoords="offset points", xytext=(0, 8),
                    ha='center', fontsize=8, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([LABELS[a] for a in available], fontsize=10)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_title('Avg Forgetting (mean ± std)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Avg Forgetting (Δ Loss)')
    ax.grid(True, axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Panel 2: Individual seed scatter
    ax = axes[1]
    for i, name in enumerate(available):
        vals = summary[name]["avg_forget_values"]
        jitter = np.random.RandomState(0).uniform(-0.15, 0.15, len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals,
                   color=COLORS[name], s=60, alpha=0.7, edgecolors='black', linewidth=0.5,
                   zorder=3)
        ax.hlines(summary[name]["avg_forget_mean"], i-0.3, i+0.3,
                  color=COLORS[name], linewidth=2, zorder=4)
    ax.set_xticks(range(len(available)))
    ax.set_xticklabels([LABELS[a] for a in available], fontsize=10)
    ax.set_title('Individual Seed Results', fontsize=12, fontweight='bold')
    ax.set_ylabel('Avg Forgetting')
    ax.grid(True, axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Panel 3: Significance summary
    ax = axes[2]
    ax.axis('off')
    table_rows = []
    for label, t in tests.items():
        sig = "p<0.001 ***" if t["p_value"] < 0.001 else \
              f"p={t['p_value']:.4f} **" if t["p_value"] < 0.01 else \
              f"p={t['p_value']:.4f} *" if t["p_value"] < 0.05 else \
              f"p={t['p_value']:.4f} ns"
        table_rows.append([
            label,
            f"{t['std_mean']:+.3f}",
            f"{t['do_mean']:+.3f}",
            f"{t['reduction_pct']:.1f}%",
            f"{t['cohens_d']:.2f}",
            sig,
        ])
    if table_rows:
        table = ax.table(
            cellText=table_rows,
            colLabels=['Scale', 'Baseline', 'DoubleO', 'Reduction', "Cohen's d", 'Significance'],
            loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.3, 2.2)
    ax.set_title("Welch's t-test Results", fontsize=12, fontweight='bold', pad=20)

    plt.tight_layout()
    chart_path = os.path.join(out_dir, 'multiseed_results.png')
    plt.savefig(chart_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\nSaved chart: {chart_path}")


if __name__ == '__main__':
    mp.set_start_method('spawn')
    os.makedirs(out_dir_global, exist_ok=True)

    # Build all (config, seed) pairs
    jobs = []
    for cfg in CONFIGS:
        for seed in SEEDS:
            jobs.append((cfg, seed))

    print(f"{'='*60}")
    print(f"  Multi-Seed Validation")
    print(f"  Arms: {len(CONFIGS)} × {len(SEEDS)} seeds = {len(jobs)} runs")
    print(f"  Workers: {MAX_WORKERS} × {THREADS_PER_WORKER} threads")
    print(f"  Output: {os.path.abspath(out_dir_global)}")
    print(f"{'='*60}")
    for i, (cfg, seed) in enumerate(jobs):
        print(f"  [{i+1:2d}] {cfg['name']:12s} seed={seed}")
    print()

    t_start = time.time()
    with mp.Pool(MAX_WORKERS) as pool:
        results = pool.map(run_arm, jobs)
    t_total = time.time() - t_start

    print(f"\nAll {len(jobs)} runs completed in {t_total/60:.1f} minutes.")

    # Check scipy is available for analysis
    try:
        from scipy import stats
        analyze_and_plot(out_dir_global)
    except ImportError:
        print("scipy not installed — skipping statistical analysis.")
        print("Install with: pip install scipy")
        print("Then run: python -c \"exec(open('scripts/run_multiseed.py').read()); analyze_and_plot('data/multiseed')\"")
