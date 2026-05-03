import os
import json
import modal
import torch
import torch.nn as nn
import torch.nn.functional as F

app = modal.App("poly-moe-continual")

image = modal.Image.debian_slim().pip_install(
    "torch", "triton", "python-dotenv", "transformers"
).add_local_dir("data", remote_path="/root/data").add_local_dir("src", remote_path="/root/src")

secrets = [modal.Secret.from_dotenv()]

class PolyMoEDoubleOBlock(nn.Module):
    def __init__(self, dim, expansion, iterations, num_experts=4):
        super().__init__()
        self.iterations = iterations
        self.num_experts = num_experts
        
        # MoE Router
        self.W_router = nn.Parameter(torch.randn(dim, num_experts) * (1.0 / dim))
        
        # K Polynomial Experts
        self.W_a = nn.Parameter(torch.randn(num_experts, dim, dim) * (1.0 / dim))
        self.decays = nn.Parameter(torch.zeros(num_experts))
            
        self.norm_cavity = nn.RMSNorm(dim)
        self.W_in = nn.Parameter(torch.randn(dim, dim * expansion) * (1.0 / dim))
        self.W_out = nn.Parameter(torch.randn(dim * expansion, dim) * (1.0 / dim))
        self.b_in = nn.Parameter(torch.zeros(dim * expansion))
        self.b_out = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        from scripts.cheby_triton import cheby_scan_triton
            
        B, T, D = x.size()
        pad_len = (4 - T % 4) % 4
        if pad_len > 0: x_pad = F.pad(x, (0, 0, 0, pad_len))
        else: x_pad = x
        
        # Compute router gates based on padded x
        gates = F.softmax(x_pad @ self.W_router, dim=-1) # (B, T_pad, K)
        
        h_agg = torch.zeros_like(x_pad)
        
        # Run K concurrent polynomial scans
        for k in range(self.num_experts):
            A = self.W_a[k] * torch.sigmoid(self.decays[k])
            h_k = cheby_scan_triton(x_pad, A)
            gate_k = gates[:, :, k].unsqueeze(-1)
            h_agg = h_agg + gate_k * h_k
            
        if pad_len > 0: h_agg = h_agg[:, :T, :]
            
        h_norm = self.norm_cavity(h_agg)
        
        # Resonant Cavity Iterations
        for _ in range(self.iterations):
            h_hidden = F.gelu(h_norm @ self.W_in + self.b_in)
            h_delta = h_hidden @ self.W_out + self.b_out
            h_norm = h_norm + h_delta
            
        return x + h_norm 

class PolyMoEDoubleO(nn.Module):
    def __init__(self, vocab_size, config):
        super().__init__()
        dim = 128
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([
            PolyMoEDoubleOBlock(
                dim=dim,
                expansion=config["expansion"],
                iterations=config["iterations"],
                num_experts=config["num_experts"]
            ) for _ in range(config["num_layers"])
        ])
        self.norm_post = nn.RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.token_emb.weight = self.lm_head.weight 

    def forward(self, x, targets=None):
        x = self.token_emb(x)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.norm_post(x))
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) if targets is not None else None
        return logits, loss

@app.function(image=image, secrets=secrets, gpu="H100", timeout=3600)
def run_model_eval(model_name: str):
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
    config = {"num_layers": 4, "expansion": 4, "iterations": 4, "num_experts": 4}
    model = PolyMoEDoubleO(vocab_size=vocab_size, config=config).cuda()
        
    params = sum(p.numel() for p in model.parameters())
    print(f"[{model_name.upper()}] Initialized. Vocab: {vocab_size}, Params: {params}")
    
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
    
    for step in range(TOTAL_STEPS):
        model.train()
        
        # Phase Switch Logic
        current_task = 'A' if step < STEPS_PHASE_1 else 'B'
        
        x, y = get_batch('train', task=current_task)
        logits, loss = model(x, y)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if step % 50 == 0 or step == (TOTAL_STEPS-1):
            model.eval()
            with torch.no_grad():
                vx_A, vy_A = get_batch('val', task='A')
                _, vloss_A = model(vx_A, vy_A)
                metrics["val_A"].append((step, vloss_A.item()))
                
                vloss_B_val = None
                if step >= STEPS_PHASE_1:
                    vx_B, vy_B = get_batch('val', task='B')
                    _, vloss_B = model(vx_B, vy_B)
                    metrics["val_B"].append((step, vloss_B.item()))
                    vloss_B_val = vloss_B.item()
                    
            if step < STEPS_PHASE_1:
                print(f"[{model_name.upper()}] P1 Step {step:4d} | Train: {loss.item():.4f} | Val A: {vloss_A.item():.4f}")
            else:
                print(f"[{model_name.upper()}] P2 Step {step:4d} | Train: {loss.item():.4f} | Val A (Forget): {vloss_A.item():.4f} | Val B (Plast): {vloss_B_val:.4f}")
            
    return {
        "name": model_name,
        "params": params,
        "metrics": metrics
    }

@app.local_entrypoint()
def main():
    models = ["poly_moe"]
    
    print("Launching Poly-MoE Continual Learning Gauntlet on Modal...")
    results = list(run_model_eval.map(models))
    
    final_dict = {res["name"]: res for res in results}
    
    with open("data/poly_moe_results.json", "w") as f:
        json.dump(final_dict, f)
        
    print("Poly-MoE Gauntlet complete! Results saved to data/poly_moe_results.json")
