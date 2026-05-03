import os
import math
import time
import json
import modal
import torch
import torch.nn as nn
import torch.nn.functional as F

app = modal.App("linguistic-sweep")

image = modal.Image.debian_slim().pip_install(
    "torch", "triton", "python-dotenv", "transformers"
).add_local_dir("data", remote_path="/root/data").add_local_dir("src", remote_path="/root/src")

secrets = [modal.Secret.from_dotenv()]

# --- Common Utilities ---
def precompute_freqs_cis(dim, end, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs).float()
    return torch.polar(torch.ones_like(freqs), freqs)

def apply_rotary_emb(xq, freqs_cis):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.unsqueeze(0) # add batch dim
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(2)
    return xq_out.type_as(xq)

# --- 4 Challenger Architectures ---

class MultiHeadChebyWaveLM(nn.Module):
    def __init__(self, vocab_size, dim=256, max_degree=64, num_heads=4, max_iterations=24):
        super().__init__()
        self.dim = dim
        self.max_degree = max_degree
        self.num_heads = num_heads
        self.max_iterations = max_iterations
        
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.W_a = nn.Parameter(torch.randn(max_degree, max_degree) * (1.0 / max_degree))
        
        self.W_in = nn.Parameter(torch.randn(dim, dim * 2) * (1.0 / dim))
        self.W_out = nn.Parameter(torch.randn(dim * 2, dim) * (1.0 / dim))
        self.b_in = nn.Parameter(torch.zeros(dim * 2))
        self.b_out = nn.Parameter(torch.zeros(dim))
        self.norm_pre = nn.RMSNorm(dim)
        self.norm_post = nn.RMSNorm(dim)
        
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.token_emb.weight = self.lm_head.weight 

    def forward(self, x, targets=None):
        from scripts.cheby_triton import cheby_scan_triton
        B, T = x.size()
        x_emb = self.token_emb(x) # (B, T, 256)
        
        pad_len = (4 - T % 4) % 4
        if pad_len > 0:
            x_emb = F.pad(x_emb, (0, 0, 0, pad_len))
        
        # Reshape to multi-head: (B*4, T_padded, 64)
        T_pad = x_emb.size(1)
        x_head = x_emb.view(B, T_pad, self.num_heads, self.max_degree).permute(0, 2, 1, 3).reshape(B * self.num_heads, T_pad, self.max_degree)
        
        # Parallel scan
        h_seq = cheby_scan_triton(x_head, self.W_a)
        
        # Reshape back to (B, T_padded, 256)
        h_seq = h_seq.view(B, self.num_heads, T_pad, self.max_degree).permute(0, 2, 1, 3).reshape(B, T_pad, self.dim)
        
        if pad_len > 0:
            h_seq = h_seq[:, :T, :]
            
        h_norm = self.norm_pre(h_seq)
        for _ in range(self.max_iterations):
            h_hidden = F.gelu(h_norm @ self.W_in + self.b_in)
            h_delta = h_hidden @ self.W_out + self.b_out
            h_norm = h_norm + h_delta
            
        logits = self.lm_head(self.norm_post(h_norm))
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) if targets is not None else None
        return logits, loss

class DecayedChebyWaveLM(nn.Module):
    def __init__(self, vocab_size, dim=128, max_iterations=24):
        super().__init__()
        self.dim = dim
        self.max_iterations = max_iterations
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.W_a = nn.Parameter(torch.randn(dim, dim) * (1.0 / dim))
        self.decay = nn.Parameter(torch.tensor(0.0)) # Learned EMA decay
        
        self.W_in = nn.Parameter(torch.randn(dim, dim * 2) * (1.0 / dim))
        self.W_out = nn.Parameter(torch.randn(dim * 2, dim) * (1.0 / dim))
        self.b_in = nn.Parameter(torch.zeros(dim * 2))
        self.b_out = nn.Parameter(torch.zeros(dim))
        self.norm_pre = nn.RMSNorm(dim)
        self.norm_post = nn.RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.token_emb.weight = self.lm_head.weight 

    def forward(self, x, targets=None):
        from scripts.cheby_triton import cheby_scan_triton
        B, T = x.size()
        x_emb = self.token_emb(x)
        
        pad_len = (4 - T % 4) % 4
        if pad_len > 0: x_emb = F.pad(x_emb, (0, 0, 0, pad_len))
            
        # Apply decay to transition matrix
        A_decayed = self.W_a * torch.sigmoid(self.decay)
        h_seq = cheby_scan_triton(x_emb, A_decayed)
        
        if pad_len > 0: h_seq = h_seq[:, :T, :]
            
        h_norm = self.norm_pre(h_seq)
        for _ in range(self.max_iterations):
            h_hidden = F.gelu(h_norm @ self.W_in + self.b_in)
            h_delta = h_hidden @ self.W_out + self.b_out
            h_norm = h_norm + h_delta
            
        logits = self.lm_head(self.norm_post(h_norm))
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) if targets is not None else None
        return logits, loss

class HybridChebyWaveLM(nn.Module):
    def __init__(self, vocab_size, dim=128, max_iterations=24, block_size=256):
        super().__init__()
        self.dim = dim
        self.max_iterations = max_iterations
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.W_a = nn.Parameter(torch.randn(dim, dim) * (1.0 / dim))
        
        from src.nanogpt.model import CausalSelfAttention, GPTConfig
        config = GPTConfig(vocab_size=vocab_size, block_size=block_size, n_embd=dim, n_head=4, dropout=0.0)
        self.attn = CausalSelfAttention(config)
        self.norm_attn = nn.LayerNorm(dim)
        
        self.W_in = nn.Parameter(torch.randn(dim, dim * 2) * (1.0 / dim))
        self.W_out = nn.Parameter(torch.randn(dim * 2, dim) * (1.0 / dim))
        self.b_in = nn.Parameter(torch.zeros(dim * 2))
        self.b_out = nn.Parameter(torch.zeros(dim))
        self.norm_pre = nn.RMSNorm(dim)
        self.norm_post = nn.RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.token_emb.weight = self.lm_head.weight 

    def forward(self, x, targets=None):
        from scripts.cheby_triton import cheby_scan_triton
        B, T = x.size()
        x_emb = self.token_emb(x)
        
        # Hybrid Attention injection
        x_emb = x_emb + self.attn(self.norm_attn(x_emb))
        
        pad_len = (4 - T % 4) % 4
        if pad_len > 0: x_emb = F.pad(x_emb, (0, 0, 0, pad_len))
        h_seq = cheby_scan_triton(x_emb, self.W_a)
        if pad_len > 0: h_seq = h_seq[:, :T, :]
            
        h_norm = self.norm_pre(h_seq)
        for _ in range(self.max_iterations):
            h_hidden = F.gelu(h_norm @ self.W_in + self.b_in)
            h_delta = h_hidden @ self.W_out + self.b_out
            h_norm = h_norm + h_delta
            
        logits = self.lm_head(self.norm_post(h_norm))
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) if targets is not None else None
        return logits, loss

class RoPEChebyWaveLM(nn.Module):
    def __init__(self, vocab_size, dim=128, max_iterations=24, block_size=256):
        super().__init__()
        self.dim = dim
        self.max_iterations = max_iterations
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.W_a = nn.Parameter(torch.randn(dim, dim) * (1.0 / dim))
        
        self.register_buffer("freqs_cis", precompute_freqs_cis(dim, block_size * 2))
        
        self.W_in = nn.Parameter(torch.randn(dim, dim * 2) * (1.0 / dim))
        self.W_out = nn.Parameter(torch.randn(dim * 2, dim) * (1.0 / dim))
        self.b_in = nn.Parameter(torch.zeros(dim * 2))
        self.b_out = nn.Parameter(torch.zeros(dim))
        self.norm_pre = nn.RMSNorm(dim)
        self.norm_post = nn.RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.token_emb.weight = self.lm_head.weight 

    def forward(self, x, targets=None):
        from scripts.cheby_triton import cheby_scan_triton
        B, T = x.size()
        x_emb = self.token_emb(x)
        
        # Apply RoPE
        x_emb = apply_rotary_emb(x_emb, self.freqs_cis[:T])
        
        pad_len = (4 - T % 4) % 4
        if pad_len > 0: x_emb = F.pad(x_emb, (0, 0, 0, pad_len))
        h_seq = cheby_scan_triton(x_emb, self.W_a)
        if pad_len > 0: h_seq = h_seq[:, :T, :]
            
        h_norm = self.norm_pre(h_seq)
        for _ in range(self.max_iterations):
            h_hidden = F.gelu(h_norm @ self.W_in + self.b_in)
            h_delta = h_hidden @ self.W_out + self.b_out
            h_norm = h_norm + h_delta
            
        logits = self.lm_head(self.norm_post(h_norm))
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) if targets is not None else None
        return logits, loss

# --- Orchestrator ---

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

    if model_name == "multihead":
        model = MultiHeadChebyWaveLM(vocab_size=vocab_size).cuda()
    elif model_name == "decay":
        model = DecayedChebyWaveLM(vocab_size=vocab_size).cuda()
    elif model_name == "hybrid":
        model = HybridChebyWaveLM(vocab_size=vocab_size).cuda()
    elif model_name == "rope":
        model = RoPEChebyWaveLM(vocab_size=vocab_size).cuda()
    else:
        raise ValueError(f"Unknown model: {model_name}")
        
    print(f"[{model_name.upper()}] Initialized. Params: {sum(p.numel() for p in model.parameters())}")
    
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
            print(f"[{model_name.upper()}] Step {step:4d} | Train: {loss.item():.4f} | Val: {vloss.item():.4f}")
            train_losses.append(loss.item())
            val_losses.append(vloss.item())
            
    return {
        "model": model_name,
        "train_loss": train_losses,
        "val_loss": val_losses
    }

@app.local_entrypoint()
def main():
    models = ["multihead", "decay", "hybrid", "rope"]
    print("Launching 4x parallel evaluations on Modal...")
    results = list(run_model_eval.map(models))
    
    final_dict = {res["model"]: {"train": res["train_loss"], "val": res["val_loss"]} for res in results}
    
    with open("data/sweep_results.json", "w") as f:
        json.dump(final_dict, f)
    print("Sweep complete! Results saved to data/sweep_results.json")
