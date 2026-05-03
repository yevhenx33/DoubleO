import os
import json
import modal
import torch
import torch.nn as nn
import torch.nn.functional as F

app = modal.App("ewc-continual-learning")

image = modal.Image.debian_slim().pip_install(
    "torch", "triton", "python-dotenv", "transformers"
).add_local_dir("data", remote_path="/root/data").add_local_dir("src", remote_path="/root/src").add_local_dir("scripts", remote_path="/root/scripts")

secrets = [modal.Secret.from_dotenv()]

class DoubleOBlock(nn.Module):
    def __init__(self, dim, expansion, iterations, use_decay=True):
        super().__init__()
        self.iterations = iterations
        self.use_decay = use_decay
        
        self.W_a = nn.Parameter(torch.randn(dim, dim) * (1.0 / dim))
        if use_decay:
            self.decay = nn.Parameter(torch.tensor(0.0))
            
        self.norm_cavity = nn.RMSNorm(dim)
        self.W_in = nn.Parameter(torch.randn(dim, dim * expansion) * (1.0 / dim))
        self.W_out = nn.Parameter(torch.randn(dim * expansion, dim) * (1.0 / dim))
        self.b_in = nn.Parameter(torch.zeros(dim * expansion))
        self.b_out = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        from scripts.cheby_triton import cheby_scan_triton
            
        B, T, D = x.size()
        pad_len = (4 - T % 4) % 4
        if pad_len > 0: x = F.pad(x, (0, 0, 0, pad_len))
        
        A = self.W_a
        if self.use_decay:
            A = A * torch.sigmoid(self.decay)
            
        h_seq = cheby_scan_triton(x, A)
        if pad_len > 0: h_seq = h_seq[:, :T, :]
            
        h_norm = self.norm_cavity(h_seq)
        for _ in range(self.iterations):
            h_hidden = F.gelu(h_norm @ self.W_in + self.b_in)
            h_delta = h_hidden @ self.W_out + self.b_out
            h_norm = h_norm + h_delta
            
        return x + h_norm 

class EWCDoubleO(nn.Module):
    def __init__(self, vocab_size, config):
        super().__init__()
        dim = 128
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([
            DoubleOBlock(
                dim=dim,
                expansion=config["expansion"],
                iterations=config["iterations"],
                use_decay=True
            ) for _ in range(config["num_layers"])
        ])
        self.norm_post = nn.RMSNorm(dim)
        
        self.lm_head_A = nn.Linear(dim, vocab_size, bias=False)
        self.lm_head_B = nn.Linear(dim, vocab_size, bias=False)
        
        self.register_buffer("centroids", torch.zeros(2, dim))
        self.register_buffer("centroid_counts", torch.zeros(2))

    def forward(self, x, targets=None, task_idx=None):
        h = self.token_emb(x)
        for block in self.blocks:
            h = block(h)
        h_norm = self.norm_post(h)
        
        # Prototype Routing
        if self.training and task_idx is not None:
            with torch.no_grad():
                alpha = 0.01
                mean_h = h_norm.mean(dim=(0, 1))
                if self.centroid_counts[task_idx] == 0:
                    self.centroids[task_idx] = mean_h
                    self.centroid_counts[task_idx] = 1.0
                else:
                    self.centroids[task_idx] = (1 - alpha) * self.centroids[task_idx] + alpha * mean_h
            w = torch.ones(h_norm.size(0), h_norm.size(1), 1, device=h.device) if task_idx == 0 else torch.zeros(h_norm.size(0), h_norm.size(1), 1, device=h.device)
        else:
            c0 = self.centroids[0]
            c1 = self.centroids[1]
            d0 = ((h_norm - c0)**2).sum(dim=-1, keepdim=True)
            d1 = ((h_norm - c1)**2).sum(dim=-1, keepdim=True)
            tau = 10.0
            w_stacked = torch.softmax(torch.cat([-d0/tau, -d1/tau], dim=-1), dim=-1)
            w = w_stacked[:, :, 0:1]
            
        logits_A = self.lm_head_A(h_norm)
        logits_B = self.lm_head_B(h_norm)
        logits = w * logits_A + (1 - w) * logits_B
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
        return logits, loss

def compute_fisher(model, train_A, device, block_size=256, num_samples=1000):
    model.eval()
    fisher_dict = {}
    optimal_weights = {}
    
    # We only protect shared parameters, not lm_head_B
    for name, param in model.named_parameters():
        if param.requires_grad and "lm_head_B" not in name:
            fisher_dict[name] = torch.zeros_like(param)
            optimal_weights[name] = param.clone().detach()
            
    count = 0
    # Batch size 1 for true empirical FIM mapping
    for _ in range(num_samples):
        ix = torch.randint(len(train_A) - block_size, (1,))
        x = train_A[ix:ix+block_size].unsqueeze(0).to(device)
        y = train_A[ix+1:ix+block_size+1].unsqueeze(0).to(device)
        
        model.zero_grad()
        logits, loss = model(x, y, task_idx=0)
        
        # Scale loss by sequence length to compute gradient of sum of log-likelihoods
        loss_sum = loss * block_size
        loss_sum.backward()
        
        for name, param in model.named_parameters():
            if name in fisher_dict and param.grad is not None:
                fisher_dict[name] += param.grad.data ** 2
        count += 1
        
    for name in fisher_dict:
        fisher_dict[name] /= count
        
    return fisher_dict, optimal_weights

@app.function(image=image, secrets=secrets, gpu="H100", timeout=3600)
def run_model_eval():
    import sys
    import torch
    import json
    sys.path.append("/root")
    
    # 1. Load Datasets
    with open("/root/data/tinyshakespeare.txt", "r", encoding='utf-8') as f:
        text_A = f.read()
        
    text_B = ""
    with open("/root/data/math.jsonl", "r", encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            text_B += obj["text"] + "\n"
            
    # 2. Build Unified Vocabulary
    chars = sorted(list(set(text_A + text_B)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    
    # 3. Encode Data
    data_A = torch.tensor([stoi[c] for c in text_A], dtype=torch.long)
    data_B = torch.tensor([stoi[c] for c in text_B], dtype=torch.long)
    
    nA = int(0.9 * len(data_A))
    train_A, val_A = data_A[:nA], data_A[nA:]
    
    nB = int(0.9 * len(data_B))
    train_B, val_B = data_B[:nB], data_B[nB:]
    
    def get_batch(split, task='A', batch_size=64, block_size=256):
        if task == 'A':
            data_split = train_A if split == 'train' else val_A
        else:
            data_split = train_B if split == 'train' else val_B
            
        ix = torch.randint(len(data_split) - block_size, (batch_size,))
        x = torch.stack([data_split[i:i+block_size] for i in ix])
        y = torch.stack([data_split[i+1:i+block_size+1] for i in ix])
        return x.cuda(), y.cuda()

    # 4. Initialize Model
    config = {"num_layers": 6, "expansion": 4, "iterations": 4}
    model = EWCDoubleO(vocab_size=vocab_size, config=config).cuda()
        
    params = sum(p.numel() for p in model.parameters())
    print(f"[EWC_DOUBLEO] Initialized. Vocab: {vocab_size}, Params: {params}")
    
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
    model = torch.compile(model)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
    
    metrics = {
        "val_A": [], # Tracked across entire 4k steps
        "val_B": []  # Tracked ONLY during Phase 2
    }
    
    STEPS_PHASE_1 = 2000
    STEPS_PHASE_2 = 2000
    TOTAL_STEPS = STEPS_PHASE_1 + STEPS_PHASE_2
    
    fisher_dict = None
    optimal_weights = None
    lambda_ewc = 0.001 # Start with 0.001 to balance large scaled gradients
    
    for step in range(TOTAL_STEPS):
        model.train()
        
        # Phase Switch Logic
        if step < STEPS_PHASE_1:
            current_task = 'A'
            task_idx = 0
        else:
            if step == STEPS_PHASE_1:
                print("\n[EWC] Phase 1 Complete. Computing True Empirical Fisher Information Matrix...")
                fisher_dict, optimal_weights = compute_fisher(model, train_A, device=torch.device("cuda"))
                
                # Freeze Task A head
                for name, param in model.named_parameters():
                    if "lm_head_A" in name:
                        param.requires_grad = False
                
                print("[EWC] Fisher Matrix Computed. Starting Phase 2 (Math) with EWC penalty.\n")
                
            current_task = 'B'
            task_idx = 1
        
        x, y = get_batch('train', task=current_task)
        logits, ce_loss = model(x, y, task_idx=task_idx)
        
        loss = ce_loss
        ewc_penalty = 0.0
        if current_task == 'B' and fisher_dict is not None:
            for name, param in model.named_parameters():
                if name in fisher_dict:
                    ewc_penalty += (fisher_dict[name] * (param - optimal_weights[name]) ** 2).sum()
            
            # Print occasionally to monitor EWC magnitude vs CE
            if step == STEPS_PHASE_1 or step % 500 == 0:
                print(f"  --> [EWC Diag] CE: {ce_loss.item():.4f} | Raw Penalty: {ewc_penalty.item():.2f} | Weighted Penalty: {((lambda_ewc / 2) * ewc_penalty).item():.4f}")
                
            loss = ce_loss + (lambda_ewc / 2) * ewc_penalty
            
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # High-resolution sampling (every 50 steps)
        if step % 50 == 0 or step == (TOTAL_STEPS-1):
            model.eval()
            with torch.no_grad():
                # Always track Task A (Catastrophic Forgetting)
                vx_A, vy_A = get_batch('val', task='A')
                _, vloss_A = model(vx_A, vy_A, task_idx=None)
                metrics["val_A"].append((step, vloss_A.item()))
                
                vloss_B_val = None
                # Track Task B only in Phase 2 (Plasticity)
                if step >= STEPS_PHASE_1:
                    vx_B, vy_B = get_batch('val', task='B')
                    _, vloss_B = model(vx_B, vy_B, task_idx=None)
                    metrics["val_B"].append((step, vloss_B.item()))
                    vloss_B_val = vloss_B.item()
                    
            if step < STEPS_PHASE_1:
                print(f"[EWC] P1 (Shakespeare) Step {step:4d} | CE: {ce_loss.item():.4f} | Val A: {vloss_A.item():.4f}")
            else:
                ewc_val = ewc_penalty.item() if isinstance(ewc_penalty, torch.Tensor) else 0.0
                print(f"[EWC] P2 (Math) Step {step:4d} | CE: {ce_loss.item():.4f} | EWC: {ewc_val:.2f} | Val A (Forget): {vloss_A.item():.4f} | Val B (Plasticity): {vloss_B_val:.4f}")
            
    return {
        "name": "ewc_pure",
        "params": params,
        "metrics": metrics
    }

@app.local_entrypoint()
def main():
    print("Launching EWC Continual Learning Gauntlet on Modal...")
    res = run_model_eval.remote()
    
    final_dict = {res["name"]: res}
    
    with open("data/ewc_continual_results.json", "w") as f:
        json.dump(final_dict, f)
        
    print("Continual Gauntlet complete! Results saved to data/ewc_continual_results.json")
