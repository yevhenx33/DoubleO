import os
import math
import json
import modal
import torch
import torch.nn as nn
import torch.nn.functional as F

app = modal.App("einstein-5k")

image = modal.Image.debian_slim().pip_install(
    "torch", "triton", "python-dotenv", "transformers"
).add_local_dir("data", remote_path="/root/data").add_local_dir("src", remote_path="/root/src")

secrets = [modal.Secret.from_dotenv()]

class EinsteinDoubleOBlock(nn.Module):
    def __init__(self, dim, expansion, iterations, init_decay_val):
        super().__init__()
        self.iterations = iterations
        
        # Data-Dependent Input Gating
        self.W_filter = nn.Parameter(torch.randn(dim, dim) * (1.0 / dim))
        
        self.W_a = nn.Parameter(torch.randn(dim, dim) * (1.0 / dim))
        
        # Fourier Initialization (Log-Scaled Decay)
        val = max(min(init_decay_val, 0.999), 0.001)
        init_x = math.log(val / (1.0 - val))
        self.decay = nn.Parameter(torch.tensor(init_x, dtype=torch.float32))
        
        self.norm_cavity = nn.RMSNorm(dim)
        
        # Standard GELU Cavity
        self.W_in = nn.Parameter(torch.randn(dim, dim * expansion) * (1.0 / dim))
        self.W_out = nn.Parameter(torch.randn(dim * expansion, dim) * (1.0 / dim))
        self.b_in = nn.Parameter(torch.zeros(dim * expansion))
        self.b_out = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        from scripts.cheby_triton import cheby_scan_triton
            
        B, T, D = x.size()
        pad_len = (4 - T % 4) % 4
        
        # Pre-Scan Filter
        gate = torch.sigmoid(x @ self.W_filter)
        x_filtered = x * gate
        
        if pad_len > 0: x_scan = F.pad(x_filtered, (0, 0, 0, pad_len))
        else: x_scan = x_filtered
        
        A = self.W_a * torch.sigmoid(self.decay)
            
        h_seq = cheby_scan_triton(x_scan, A)
        if pad_len > 0: h_seq = h_seq[:, :T, :]
            
        h_norm = self.norm_cavity(h_seq)
        
        # Cavity Iterations
        for _ in range(self.iterations):
            h_hidden = F.gelu(h_norm @ self.W_in + self.b_in)
            h_delta = h_hidden @ self.W_out + self.b_out
            h_norm = h_norm + h_delta
            
        return x + h_norm 

class EinsteinDoubleO(nn.Module):
    def __init__(self, vocab_size, config):
        super().__init__()
        dim = 128
        self.token_emb = nn.Embedding(vocab_size, dim)
        
        decays = config["decays"]
        self.blocks = nn.ModuleList([
            EinsteinDoubleOBlock(
                dim=dim,
                expansion=config["expansion"],
                iterations=config["iterations"],
                init_decay_val=decays[i]
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

@app.function(image=image, secrets=secrets, gpu="H100", timeout=3600)
def run_model_eval(model_name: str):
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

    config = {
        "num_layers": 5, 
        "expansion": 4, 
        "iterations": 4,
        "decays": [0.1, 0.3, 0.5, 0.7, 0.9]
    }
    model = EinsteinDoubleO(vocab_size=vocab_size, config=config).cuda()
        
    params = sum(p.numel() for p in model.parameters())
    print(f"[{model_name.upper()}] Initialized. Params: {params}")
    
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
    model = torch.compile(model)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
    
    train_losses = []
    val_losses = []
    
    STEPS = 5000
    
    for step in range(STEPS):
        model.train()
        x, y = get_batch('train')
        logits, loss = model(x, y)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if step % 200 == 0 or step == (STEPS-1):
            model.eval()
            with torch.no_grad():
                vx, vy = get_batch('val')
                _, vloss = model(vx, vy)
            print(f"[{model_name.upper()}] Step {step:4d} | Train: {loss.item():.4f} | Val: {vloss.item():.4f}")
            train_losses.append(loss.item())
            val_losses.append(vloss.item())
            
    return {
        "name": model_name,
        "params": params,
        "train_loss": train_losses,
        "val_loss": val_losses
    }

@app.local_entrypoint()
def main():
    models = ["einstein_pure"]
    
    print("Launching Einstein Optimization 5k-step Showdown on Modal...")
    results = list(run_model_eval.map(models))
    
    final_dict = {res["name"]: {"params": res["params"], "train": res["train_loss"], "val": res["val_loss"]} for res in results}
    
    with open("data/einstein_5k_results.json", "w") as f:
        json.dump(final_dict, f)
        
    print("Einstein 5k Showdown complete! Results saved to data/einstein_5k_results.json")
