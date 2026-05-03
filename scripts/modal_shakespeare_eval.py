import os
import modal

app = modal.App("shakespeare-eval")

image = modal.Image.debian_slim().pip_install(
    "torch", "triton", "python-dotenv"
).add_local_dir("data", remote_path="/root/data").add_local_dir("src", remote_path="/root/src")

secrets = [modal.Secret.from_dotenv()]

@app.function(image=image, secrets=secrets, gpu="H100", timeout=3600)
def run_eval():
    import sys
    sys.path.append("/root") 
    
    import torch
    import torch.nn as nn
    from src.nanogpt.model import GPT, GPTConfig
    from scripts.cheby_triton import cheby_scan_triton
    import time
    import json
    
    # Load Data
    with open("/root/data/tinyshakespeare.txt", "r", encoding='utf-8') as f:
        text = f.read()
        
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    
    print(f"Loaded TinyShakespeare. Vocab size: {vocab_size}, Total chars: {len(text)}")
    
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

    class ChebyWaveLM(nn.Module):
        def __init__(self, vocab_size, dim=256, max_degree=256, max_iterations=24, block_size=256):
            super().__init__()
            self.dim = dim
            self.block_size = block_size
            self.max_degree = max_degree
            self.max_iterations = max_iterations
            
            self.token_emb = nn.Embedding(vocab_size, dim)
            
            self.W_a = nn.Parameter(torch.randn(max_degree, max_degree) * (1.0 / max_degree))
            
            self.W_in = nn.Parameter(torch.randn(max_degree, max_degree * 2) * (1.0 / max_degree))
            self.W_out = nn.Parameter(torch.randn(max_degree * 2, max_degree) * (1.0 / max_degree))
            self.b_in = nn.Parameter(torch.zeros(max_degree * 2))
            self.b_out = nn.Parameter(torch.zeros(max_degree))
            self.norm_pre = nn.RMSNorm(max_degree)
            self.norm_post = nn.RMSNorm(max_degree)
            
            self.lm_head = nn.Linear(max_degree, vocab_size, bias=False)
            self.token_emb.weight = self.lm_head.weight 
            
        def forward(self, x, targets=None):
            B, T = x.size()
            x_emb = self.token_emb(x)
            
            pad_len = (4 - T % 4) % 4
            if pad_len > 0:
                x_emb = torch.nn.functional.pad(x_emb, (0, 0, 0, pad_len))
                
            h_seq = cheby_scan_triton(x_emb, self.W_a)
            
            if pad_len > 0:
                h_seq = h_seq[:, :T, :]
                
            h_norm = self.norm_pre(h_seq)
            
            for _ in range(self.max_iterations):
                h_hidden = torch.nn.functional.gelu(h_norm @ self.W_in + self.b_in)
                h_delta = h_hidden @ self.W_out + self.b_out
                h_norm = h_norm + h_delta
                
            h_out = self.norm_post(h_norm)
            logits = self.lm_head(h_out)
            
            loss = None
            if targets is not None:
                loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
                
            return logits, loss
            
    print("\n--- Initializing Models ---")
    config = GPTConfig(vocab_size=vocab_size, block_size=256, n_layer=4, n_head=4, n_embd=128)
    model_gpt = GPT(config).cuda()
    
    model_doubleo = ChebyWaveLM(vocab_size=vocab_size, dim=128, max_degree=128, max_iterations=24, block_size=256).cuda()
    print(f"[DoubleO v6] Params: {sum(p.numel() for p in model_doubleo.parameters())}")
    
    print("\nCompiling Models...")
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
    model_gpt = torch.compile(model_gpt)
    model_doubleo = torch.compile(model_doubleo)
    
    def train_model(model, name, steps=2000):
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
        
        train_losses = []
        val_losses = []
        
        t0 = time.time()
        for step in range(steps):
            model.train()
            x, y = get_batch('train')
            logits, loss = model(x, y)
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            if step % 200 == 0 or step == steps - 1:
                model.eval()
                with torch.no_grad():
                    vx, vy = get_batch('val')
                    _, vloss = model(vx, vy)
                print(f"[{name}] Step {step:4d} | Train Loss: {loss.item():.4f} | Val Loss: {vloss.item():.4f}")
                train_losses.append(loss.item())
                val_losses.append(vloss.item())
                
        t1 = time.time()
        print(f"[{name}] Finished 2000 steps in {t1-t0:.2f} seconds.")
        return train_losses, val_losses

    print("\n==============================================")
    print("Evaluating NanoGPT (Transformer Baseline)")
    print("==============================================")
    gpt_train, gpt_val = train_model(model_gpt, "NanoGPT", steps=2000)
    
    print("\n==============================================")
    print("Evaluating DoubleO v6 (Resonant Cavity)")
    print("==============================================")
    do_train, do_val = train_model(model_doubleo, "DoubleO", steps=2000)
    
    results = {
        "nanogpt_train": gpt_train,
        "nanogpt_val": gpt_val,
        "doubleo_train": do_train,
        "doubleo_val": do_val
    }
    with open("/root/data/shakespeare_results.json", "w") as f:
        json.dump(results, f)
    
    print("\nResults successfully saved to /root/data/shakespeare_results.json")

@app.local_entrypoint()
def main():
    run_eval.remote()
