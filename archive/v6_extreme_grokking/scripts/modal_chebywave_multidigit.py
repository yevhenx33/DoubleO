import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random
import time
import modal
import triton
import triton.language as tl

import sys
try:
    from scripts.cheby_triton import cheby_scan_triton, get_cheby_matrix
except ModuleNotFoundError:
    sys.path.append("/root")
    from scripts.cheby_triton import cheby_scan_triton, get_cheby_matrix

# ══════════════════════════════════════════════════════════════
# DATASET GENERATION
# ══════════════════════════════════════════════════════════════

def generate_all_addition_problems():
    problems = []
    for a in range(100):
        for b in range(100):
            ans = str(a + b)[::-1]
            problems.append(f"{a} + {b} = {ans}")
    return problems

def generate_all_subtraction_problems():
    problems = []
    for a in range(100):
        for b in range(100):
            if a >= b:
                ans = str(a - b)[::-1]
                problems.append(f"{a} - {b} = {ans}")
    return problems

VOCAB = ["\n", " ", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "+", "-", "*", "=", "<|endoftext|>"]
CTOI = {c: i for i, c in enumerate(VOCAB)}
ITOC = {i: c for i, c in enumerate(VOCAB)}
VOCAB_SIZE = len(VOCAB)

class ArithmeticDataset(Dataset):
    def __init__(self, problems, block_size):
        self.block_size = block_size
        self.problems = problems

    def __len__(self):
        return len(self.problems)

    def __getitem__(self, idx):
        text = self.problems[idx] + "\n"
        tokens = [CTOI[c] for c in text if c in CTOI]
        
        # Pad to block_size + 1 to accommodate x and y
        if len(tokens) < self.block_size + 1:
            pad_len = self.block_size + 1 - len(tokens)
            tokens = tokens + [CTOI["<|endoftext|>"]] * pad_len
        else:
            tokens = tokens[:self.block_size + 1]
            
        data = torch.tensor(tokens, dtype=torch.long)
        x = data[:-1]
        y = data[1:]
        return x, y

def evaluate_accuracy(model, problems, task_idx, prefix=""):
    device = next(model.parameters()).device
    model.eval()
    correct = 0
    total = len(problems)
    
    print(f"\n  [{prefix}] Accuracy:")
    for prob in problems:
        parts = prob.split("=")
        question = parts[0] + "="
        expected = parts[1].strip()
        
        tokens = [CTOI[c] for c in question if c in CTOI]
        x = torch.tensor([tokens], dtype=torch.long).to(device)
        
        with torch.no_grad():
            logits, _ = model(x)
            last_token_logits = logits[0, -1, :]
            pred_token_idx = torch.argmax(last_token_logits).item()
            pred_char = ITOC[pred_token_idx]
            
            # Auto-regressive decode (up to 4 chars for answers, e.g. 198\n)
            out_str = pred_char
            cur_x = torch.cat([x, torch.tensor([[pred_token_idx]], device=device)], dim=1)
            for _ in range(3):
                logits, _ = model(cur_x)
                next_tok = torch.argmax(logits[0, -1, :]).item()
                if ITOC[next_tok] == "\n" or ITOC[next_tok] == "<|endoftext|>": break
                out_str += ITOC[next_tok]
                cur_x = torch.cat([cur_x, torch.tensor([[next_tok]], device=device)], dim=1)
                
        out_str = out_str.strip()
        is_correct = (out_str == expected)
        if is_correct: correct += 1
        
        if total <= 20 or random.random() < 0.05:
            mark = "✓" if is_correct else "✗"
            human_expected = expected[::-1]
            human_out = out_str[::-1]
            print(f"    {mark} {question}{human_expected}  →  model: '{human_out}'")
            
    acc = 100.0 * correct / total
    print(f"  [{prefix}] Accuracy: {correct}/{total} = {acc:.1f}%")
    return acc

# ══════════════════════════════════════════════════════════════
# MODEL DEFINITION (v6 Auto-Routed)
# ══════════════════════════════════════════════════════════════

class AutoRoutedCavity(nn.Module):
    def __init__(self, max_degree, num_tasks=2):
        super().__init__()
        self.num_tasks = num_tasks
        self.gate_norm = nn.LayerNorm(max_degree)
        self.gate_proj = nn.Linear(max_degree, max_degree)
        self.iso_norm = nn.LayerNorm(max_degree)
        self.iso_resonators = nn.ModuleList([
            nn.Sequential(nn.Linear(max_degree, max_degree*2), nn.GELU(), nn.Linear(max_degree*2, max_degree))
            for _ in range(num_tasks)
        ])
        self.shared_norm = nn.LayerNorm(max_degree)
        self.shared_resonator = nn.Sequential(nn.Linear(max_degree, max_degree*2), nn.GELU(), nn.Linear(max_degree*2, max_degree))
        self.da_gate_logits = nn.Parameter(torch.full((num_tasks,), -2.0))
        self.damping_logits = nn.Parameter(torch.full((max_degree,), 2.0))
        self.router = nn.Sequential(nn.LayerNorm(max_degree), nn.Linear(max_degree, 1))

    def _bound(self, x):
        return x / x.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-5)

    def forward(self, h):
        damping = torch.sigmoid(self.damping_logits)
        h = h * damping
        gate = torch.sigmoid(self.gate_proj(self.gate_norm(h)))
        h_gated = h + h * gate
        h_norm = self._bound(self.iso_norm(h_gated))

        h_t2 = 2 * h_norm**2 - 1           # T2
        h_t3 = 4 * h_norm**3 - 3 * h_norm  # T3

        out_t2 = self.iso_resonators[0](h_t2)
        out_t3 = self.iso_resonators[1](h_t3)

        w = torch.sigmoid(self.router(h_gated))  # (B, 1)
        isolated = w * out_t2 + (1 - w) * out_t3

        shared = self.shared_resonator(self.shared_norm(h_gated))
        alpha_0 = torch.sigmoid(self.da_gate_logits[0])
        alpha_1 = torch.sigmoid(self.da_gate_logits[1])
        alpha = w * alpha_0 + (1 - w) * alpha_1

        return h + isolated + alpha * shared, w

class ChebyConfig:
    vocab_size = VOCAB_SIZE
    max_degree = 128       # Kept at 128 to fit A matrix in 228KB SRAM
    num_tasks = 2
    block_size = 64
    max_iterations = 24    # Scaled up for nonlinear thought depth
    convergence_threshold = 1e-3

class ChebyWaveModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.max_degree = config.max_degree
        self.cavity = AutoRoutedCavity(config.max_degree, config.num_tasks)
        self.max_iters = config.max_iterations
        self.conv_thresh = config.convergence_threshold
        self.ln_f = nn.LayerNorm(config.max_degree)
        self.decoders = nn.ModuleList([
            nn.Linear(config.max_degree, config.vocab_size)
            for _ in range(config.num_tasks)
        ])

    def forward(self, idx, targets=None):
        B, T = idx.size()
        
        # 1. Linear Temporal Scan (Triton)
        X = torch.zeros(B, T, self.max_degree, device=idx.device)
        freq_idx = (idx + 1).clamp(max=self.max_degree - 1)
        X.scatter_(2, freq_idx.unsqueeze(2), 1.0)
        
        A = get_cheby_matrix(self.max_degree, device=idx.device)
        H_out = cheby_scan_triton(X, A)  # [B, T, D]
        
        # 2. Parallel Resonant Cavity
        # Flatten sequence to process all time steps in parallel!
        h_proc = H_out.view(B * T, self.max_degree)
        w_final = None
        for i in range(self.max_iters):
            h_next, w = self.cavity(h_proc)
            w_final = w
            diff = (h_next - h_proc).norm(dim=-1).mean()
            h_proc = h_next
            if diff < self.conv_thresh: break

        h_normed = self.ln_f(h_proc)
        logits_0 = self.decoders[0](h_normed)  # addition decoder
        logits_1 = self.decoders[1](h_normed)  # subtraction decoder
        logits = w_final * logits_0 + (1 - w_final) * logits_1

        logits = logits.view(B, T, -1)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                targets.view(-1),
                ignore_index=CTOI["<|endoftext|>"]
            )
        return logits, loss

# ══════════════════════════════════════════════════════════════
# MODAL DEPLOYMENT & EVALUATION GAUNTLET
# ══════════════════════════════════════════════════════════════

app = modal.App("chebywave-multidigit-gauntlet")
image = modal.Image.debian_slim(python_version="3.12").pip_install("torch", "triton")

@app.function(image=image, gpu="H100", timeout=7200)
def run_chebywave_eval():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    config = ChebyConfig()
    model = ChebyWaveModel(config).to(device)
    model = torch.compile(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
    
    params = sum(p.numel() for p in model.parameters())
    print(f"  [v6 Multi-Digit Auto-Routed] Params: {params:,}")
    print(f"\n{'='*60}\nACCURACY GAUNTLET: 2-Digit Math (Auto-Routed)\n{'='*60}\n")
    
    # ── Datasets (No Leakage Splitting) ──
    all_add = generate_all_addition_problems()
    all_sub = generate_all_subtraction_problems()
    
    random.seed(42)
    random.shuffle(all_add)
    random.shuffle(all_sub)
    
    # Addition: 10,000 total. Train on 4000, Test on 1000 (disjoint)
    add_train = all_add[:4000]
    add_test  = all_add[4000:5000]
    
    # Subtraction: 5,050 total. Train on 4000, Test on 1050 (disjoint)
    sub_train = all_sub[:4000]
    sub_test  = all_sub[4000:5050]
    
    # ── Phase 1: Train Addition (Task A) ──
    print("--- Phase 1: Training on 2-Digit Addition (Task A) ---")
    dataset_a = ArithmeticDataset(add_train, block_size=config.block_size)
    loader_a = DataLoader(dataset_a, batch_size=256, shuffle=True)
    loader_a_iter = iter(loader_a)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80000, eta_min=1e-5)
    
    model.train()
    t0 = time.time()
    for step in range(80000):
        try: x, y = next(loader_a_iter)
        except StopIteration: loader_a_iter = iter(loader_a); x, y = next(loader_a_iter)
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        if step % 2000 == 0:
            print(f"  Step {step}, loss: {loss.item():.4f}, lr: {scheduler.get_last_lr()[0]:.2e}")
            
    print(f"  Phase 1 took {time.time()-t0:.1f}s\n")
    
    # ── Phase 2: Addition Eval ──
    print("--- Phase 2: 2-Digit Addition Accuracy (before Task B) ---\n")
    add_train_acc = evaluate_accuracy(model, add_train[:100], 0, "Addition TRAIN (subset)")
    add_test_acc  = evaluate_accuracy(model, add_test[:100], 0, "Addition TEST (subset)")
    
    # ── Phase 3: Train A + B (Cooperative Interleaved) ──
    print("\n--- Phase 3: Cooperative Training (A+B mixed, auto-routed) ---")
    mixed_data = add_train + sub_train
    random.shuffle(mixed_data)
    dataset_ab = ArithmeticDataset(mixed_data, block_size=config.block_size)
    loader_ab = DataLoader(dataset_ab, batch_size=256, shuffle=True)
    loader_ab_iter = iter(loader_ab)
    
    # Reset scheduler and optimizer LR for Phase 3
    for param_group in optimizer.param_groups:
        param_group['lr'] = 1e-3
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80000, eta_min=1e-5)
    
    model.train()
    t0 = time.time()
    for step in range(80000):
        try: x, y = next(loader_ab_iter)
        except StopIteration: loader_ab_iter = iter(loader_ab); x, y = next(loader_ab_iter)
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        if step % 2000 == 0:
            print(f"  Step {step}, loss: {loss.item():.4f}, lr: {scheduler.get_last_lr()[0]:.2e}")
            
    print(f"  Phase 3 took {time.time()-t0:.1f}s\n")
    
    # ── Phase 4: Final Eval ──
    print("--- Phase 4: Final Accuracy (after Task B) ---\n")
    add_train_after = evaluate_accuracy(model, add_train[:100], 0, "Addition TRAIN (retention subset)")
    add_test_after  = evaluate_accuracy(model, add_test[:100], 0, "Addition TEST (retention subset)")
    sub_train_acc   = evaluate_accuracy(model, sub_train[:100], 1, "Subtraction TRAIN (subset)")
    sub_test_acc    = evaluate_accuracy(model, sub_test[:100], 1, "Subtraction TEST (subset)")
    
    retention = add_test_after - add_test_acc
    print(f"\n{'='*60}")
    print(f"SUMMARY: 2-Digit Math ChebyWave v6 (Auto-Routed)")
    print(f"{'='*60}")
    print(f"  Addition (before):  Train={add_train_acc:.1f}%  Test={add_test_acc:.1f}%")
    print(f"  Addition (after):   Train={add_train_after:.1f}%  Test={add_test_after:.1f}%")
    print(f"  Subtraction:        Train={sub_train_acc:.1f}%  Test={sub_test_acc:.1f}%")
    print(f"  Retention Delta:    {retention:+.1f}%")

@app.local_entrypoint()
def main():
    run_chebywave_eval.remote()
