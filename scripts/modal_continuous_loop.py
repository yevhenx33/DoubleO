import os
import math
import json
import modal
import torch
import torch.nn as nn
import torch.nn.functional as F

app = modal.App("continuous-improvement-sweep")

image = modal.Image.debian_slim().pip_install(
    "torch", "triton", "python-dotenv", "transformers"
).add_local_dir("data", remote_path="/root/data").add_local_dir("src", remote_path="/root/src")

secrets = [modal.Secret.from_dotenv()]

# --- Layered DoubleO Architecture ---

class DoubleOBlock(nn.Module):
    def __init__(self, dim, expansion, iterations, use_attn=False, use_decay=False, block_size=256):
        super().__init__()
        self.use_attn = use_attn
        self.use_decay = use_decay
        self.iterations = iterations
        
        self.norm_attn = nn.RMSNorm(dim) if use_attn else None
        if use_attn:
            from src.nanogpt.model import CausalSelfAttention, GPTConfig
            config = GPTConfig(vocab_size=65, block_size=block_size, n_embd=dim, n_head=4, dropout=0.0)
            self.attn = CausalSelfAttention(config)
            
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
        if self.use_attn:
            x = x + self.attn(self.norm_attn(x))
            
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
            
        return x + h_norm # Residual connection!

class ScaledChebyWaveLM(nn.Module):
    def __init__(self, vocab_size, config):
        super().__init__()
        dim = 128
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([
            DoubleOBlock(
                dim=dim,
                expansion=config["expansion"],
                iterations=config["iterations"],
                use_attn=config["attn_layers"][i],
                use_decay=config["use_decay"]
            ) for i in range(config["num_layers"])
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

# --- Orchestrator ---

@app.function(image=image, secrets=secrets, gpu="H100", timeout=3600)
def run_model_eval(config):
    import sys
    import torch
    sys.path.append("/root")
    
    with open("/root/data/tinyshakespeare.txt", "r", encoding='utf-8') as f:
        text = f.read()
        
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    
    def get_batch(split, batch_size=64, block_size=256):
        data_split = train_data if split == 'train' else val_data
        ix = torch.randint(len(data_split) - block_size, (batch_size,))
        x = torch.stack([data_split[i:i+block_size] for i in ix])
        y = torch.stack([data_split[i+1:i+block_size+1] for i in ix])
        return x.cuda(), y.cuda()

    model = ScaledChebyWaveLM(vocab_size=vocab_size, config=config).cuda()
    params = sum(p.numel() for p in model.parameters())
    print(f"[{config['name'].upper()}] Initialized. Params: {params}")
    
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
    model = torch.compile(model)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
    
    train_losses = []
    val_losses = []
    
    for step in range(2000):
        model.train()
        x, y = get_batch('train')
        logits, loss = model(x, y)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if step % 200 == 0 or step == 1999:
            model.eval()
            with torch.no_grad():
                vx, vy = get_batch('val')
                _, vloss = model(vx, vy)
            print(f"[{config['name'].upper()}] Step {step:4d} | Train: {loss.item():.4f} | Val: {vloss.item():.4f}")
            train_losses.append(loss.item())
            val_losses.append(vloss.item())
            
    return {
        "name": config["name"],
        "params": params,
        "train_loss": train_losses,
        "val_loss": val_losses
    }

@app.local_entrypoint()
def main():
    # Define the Search Grid (Max 6 parallel runs)
    grid = [
        # Pure Dynamical Baseline
        {"name": "deep_pure", "num_layers": 4, "expansion": 4, "iterations": 4, "attn_layers": [False]*4, "use_decay": True},
        # Hybrid Attention 1-Layer (First Layer)
        {"name": "hybrid_first", "num_layers": 4, "expansion": 4, "iterations": 4, "attn_layers": [True, False, False, False], "use_decay": True},
        # Hybrid Attention 1-Layer (Last Layer)
        {"name": "hybrid_last", "num_layers": 4, "expansion": 4, "iterations": 4, "attn_layers": [False, False, False, True], "use_decay": True},
        # Shallow Wide Hybrid
        {"name": "shallow_hybrid", "num_layers": 2, "expansion": 8, "iterations": 8, "attn_layers": [True, True], "use_decay": True},
        # Massive Pure Dynamical (No Attention)
        {"name": "massive_pure", "num_layers": 6, "expansion": 4, "iterations": 4, "attn_layers": [False]*6, "use_decay": True},
        # NanoGPT Clone Test (All Attention, Minimal Cavity)
        {"name": "attention_heavy", "num_layers": 4, "expansion": 2, "iterations": 2, "attn_layers": [True]*4, "use_decay": True},
    ]
    
    print("Launching Continuous Iteration Loop across 6x H100s...")
    results = list(run_model_eval.map(grid))
    
    final_dict = {res["name"]: {"params": res["params"], "train": res["train_loss"], "val": res["val_loss"]} for res in results}
    
    with open("data/continuous_loop_results.json", "w") as f:
        json.dump(final_dict, f)
        
    print("Continuous loop complete! Results saved to data/continuous_loop_results.json")
    
    # Evaluate against baseline
    baseline = 1.53
    best_val = min([res["val_loss"][-1] for res in results])
    best_model = [res["name"] for res in results if res["val_loss"][-1] == best_val][0]
    
    print(f"\n--- GAUNTLET RESULTS ---")
    print(f"NanoGPT Baseline: {baseline}")
    print(f"Best Challenger: {best_model} ({best_val:.4f})")
    
    if best_val < baseline:
        print(f"🎉 BREAKTHROUGH! {best_model} beat the baseline by {(baseline - best_val):.4f}!")
    else:
        print(f"❌ NanoGPT survives. Best model missed by {(best_val - baseline):.4f}.")
