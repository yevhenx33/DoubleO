#!/usr/bin/env python3
"""
Quick A/B test: DoubleO current vs DoubleO + per-task scaling.

6 runs total (2 variants × 3 seeds), ~20 min on 8-core CPU.

Variant A (current):  freeze non-MLP, train all MLP weights
Variant B (scaled):   freeze EVERYTHING including MLP, train only task_scale vectors
"""
import os, sys, json, time
import multiprocessing as mp

SEEDS = [42, 137, 256]
PRETRAIN_STEPS = 1000
TASK_STEPS = 800
EVAL_BATCHES = 20
BATCH_SIZE = 16
LR = 3e-4
NUM_TASKS = 4
TASK_NAMES = ["Math", "Code", "Logic", "Spell"]
MAX_WORKERS = 4
THREADS_PER_WORKER = 2

VARIANTS = [
    {"name": "do_current", "use_task_scale": False},
    {"name": "do_scaled",  "use_task_scale": True},
]

out_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'ab_scale')


def run_arm(args):
    variant, seed = args
    import torch
    torch.set_num_threads(THREADS_PER_WORKER)
    torch.manual_seed(seed)

    src_dir = os.path.join(os.path.dirname(__file__), '..', 'src')
    sys.path.insert(0, os.path.abspath(src_dir))
    from model_extended import DoubleOGPT, DoubleOGPTConfig
    from torch.utils.data import Dataset, DataLoader

    run_id = f"{variant['name']}_s{seed}"

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

    gen = torch.Generator().manual_seed(seed)
    loaders = []
    for tf in ["math", "code", "logic", "spell"]:
        ds = TextDataset(os.path.join(data_dir, f"{tf}.jsonl"), stoi, 96)
        loaders.append(DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                                  drop_last=True, generator=gen))

    # Build model
    cfg = DoubleOGPTConfig()
    cfg.vocab_size = vocab_size
    cfg.n_embd = 128
    cfg.n_layer = 4
    cfg.n_head = 4
    cfg.block_size = 96
    cfg.use_task_scale = variant['use_task_scale']
    cfg.num_tasks = NUM_TASKS
    model = DoubleOGPT(cfg)

    total_params = sum(p.numel() for p in model.parameters())
    scale_params = sum(p.numel() for n, p in model.named_parameters() if 'task_scale' in n)
    print(f"[{run_id}] START params={total_params:,} scale_params={scale_params:,}", flush=True)

    iters = [infinite_iter(ld) for ld in loaders]
    t0 = time.time()

    # Phase 0: Pretrain (all params trainable)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    model.train()
    for step in range(PRETRAIN_STEPS):
        ti = step % NUM_TASKS
        model.set_task(ti)
        x, y = next(iters[ti])
        optimizer.zero_grad()
        _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    post_pretrain = evaluate_all(model, loaders)
    print(f"[{run_id}] pretrain DONE {time.time()-t0:.0f}s losses={[f'{l:.3f}' for l in post_pretrain]}", flush=True)

    # Phase 1: Freeze — THIS IS THE KEY DIFFERENCE
    if variant['use_task_scale']:
        # SCALED variant: freeze EVERYTHING, unfreeze only task_scale
        for p in model.parameters():
            p.requires_grad = False
        for name, p in model.named_parameters():
            if 'task_scale' in name:
                p.requires_grad = True
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[{run_id}] SCALED mode: {trainable} trainable params (task_scale only)", flush=True)
    else:
        # CURRENT variant: freeze non-MLP, train all MLP weights
        for name, p in model.named_parameters():
            if 'mlp' not in name:
                p.requires_grad = False
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[{run_id}] CURRENT mode: {trainable} trainable params (all MLP)", flush=True)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

    # Phase 2: Sequential task training
    post_task = {}
    for ti in range(NUM_TASKS):
        model.set_task(ti)
        task_iter = infinite_iter(loaders[ti])
        for step in range(TASK_STEPS):
            x, y = next(task_iter)
            optimizer.zero_grad()
            _, loss = model(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        post_task[ti] = evaluate_all(model, loaders)
        print(f"[{run_id}] {TASK_NAMES[ti]} DONE losses={[f'{l:.3f}' for l in post_task[ti]]}", flush=True)

    total_time = time.time() - t0

    # Compute forgetting
    final = post_task[NUM_TASKS - 1]
    forgetting = {}
    for ti in range(NUM_TASKS):
        forgetting[TASK_NAMES[ti]] = final[ti] - post_task[ti][ti]
    avg_forget = sum(forgetting.values()) / len(forgetting)

    print(f"[{run_id}] DONE {total_time:.0f}s avg_forget={avg_forget:+.4f} {forgetting}", flush=True)

    result = {
        "variant": variant["name"], "seed": seed,
        "forgetting": forgetting, "avg_forgetting": avg_forget,
        "total_time_s": total_time, "total_params": total_params,
        "trainable_params": trainable, "scale_params": scale_params,
    }
    with open(os.path.join(out_dir, f"{run_id}.json"), 'w') as f:
        json.dump(result, f, indent=2)
    return result


if __name__ == '__main__':
    mp.set_start_method('spawn')
    os.makedirs(out_dir, exist_ok=True)

    jobs = [(v, s) for v in VARIANTS for s in SEEDS]
    print(f"A/B Test: {len(jobs)} runs ({len(VARIANTS)} variants × {len(SEEDS)} seeds)")
    print(f"Variant A: do_current (freeze non-MLP, train all MLP)")
    print(f"Variant B: do_scaled  (freeze ALL, train task_scale only)")
    print()

    t0 = time.time()
    with mp.Pool(MAX_WORKERS) as pool:
        results = pool.map(run_arm, jobs)

    # Analyze
    from collections import defaultdict
    import numpy as np

    by_variant = defaultdict(list)
    for r in results:
        by_variant[r["variant"]].append(r["avg_forgetting"])

    print(f"\n{'='*60}")
    print(f"  A/B RESULTS ({time.time()-t0:.0f}s total)")
    print(f"{'='*60}")
    for vname in ["do_current", "do_scaled"]:
        vals = by_variant[vname]
        m, s = np.mean(vals), np.std(vals, ddof=1)
        print(f"  {vname:15s}  mean={m:+.4f}  std={s:.4f}  values={[f'{v:+.4f}' for v in vals]}")
        # Also print trainable params
        tp = [r["trainable_params"] for r in results if r["variant"] == vname][0]
        sp = [r["scale_params"] for r in results if r["variant"] == vname][0]
        print(f"  {'':15s}  trainable={tp:,}  (task_scale={sp:,})")

    cur = by_variant["do_current"]
    scl = by_variant["do_scaled"]
    improvement = (1 - np.mean(scl) / np.mean(cur)) * 100
    print(f"\n  Improvement: {improvement:+.1f}% reduction in forgetting")
    print(f"  (scaled vs current: {np.mean(scl):+.4f} vs {np.mean(cur):+.4f})")
