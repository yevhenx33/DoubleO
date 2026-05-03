import os
import json
import modal
import torch
import torch.nn as nn
import torch.nn.functional as F

app = modal.App("cheby-moe-continual")

image = modal.Image.debian_slim().pip_install(
    "torch", "triton", "python-dotenv", "transformers"
).add_local_dir("data", remote_path="/root/data").add_local_dir("src", remote_path="/root/src").add_local_dir("scripts", remote_path="/root/scripts")

secrets = [modal.Secret.from_dotenv()]

class TemporalRouter(nn.Module):
    def __init__(self, dim, num_experts=2):
        super().__init__()
        # Use router_dim=16 to satisfy Triton's tl.dot requirement (K >= 16)
        self.router_dim = 16
        self.router_proj = nn.Linear(dim, self.router_dim)
        self.decay = nn.Parameter(torch.tensor(0.0))
        self.W_a = nn.Parameter(torch.randn(self.router_dim, self.router_dim) * (1.0 / self.router_dim))
        self.out_proj = nn.Linear(self.router_dim, num_experts)
        
    def forward(self, x):
        from scripts.cheby_triton import cheby_scan_triton
        B, T, D = x.size()
        pad_len = (4 - T % 4) % 4
        if pad_len > 0: x = F.pad(x, (0, 0, 0, pad_len))
        
        r = self.router_proj(x) # [B, T, 16]
        
        A = self.W_a * torch.sigmoid(self.decay)
        r_seq = cheby_scan_triton(r, A)
        
        if pad_len > 0: r_seq = r_seq[:, :T, :]
        
        logits = self.out_proj(r_seq)
        w = torch.softmax(logits, dim=-1)
        return w

class DoubleOBlock(nn.Module):
    def __init__(self, dim, expansion, iterations):
        super().__init__()
        self.iterations = iterations
        
        # Expert 1
        self.A_1 = nn.Parameter(torch.randn(dim, dim) * (1.0 / dim))
        self.decay_1 = nn.Parameter(torch.tensor(0.0))
        self.norm_1 = nn.RMSNorm(dim)
        self.W_in_1 = nn.Parameter(torch.randn(dim, dim * expansion) * (1.0 / dim))
        self.W_out_1 = nn.Parameter(torch.randn(dim * expansion, dim) * (1.0 / dim))
        self.b_in_1 = nn.Parameter(torch.zeros(dim * expansion))
        self.b_out_1 = nn.Parameter(torch.zeros(dim))
        
        # Expert 2
        self.A_2 = nn.Parameter(torch.randn(dim, dim) * (1.0 / dim))
        self.decay_2 = nn.Parameter(torch.tensor(0.0))
        self.norm_2 = nn.RMSNorm(dim)
        self.W_in_2 = nn.Parameter(torch.randn(dim, dim * expansion) * (1.0 / dim))
        self.W_out_2 = nn.Parameter(torch.randn(dim * expansion, dim) * (1.0 / dim))
        self.b_in_2 = nn.Parameter(torch.zeros(dim * expansion))
        self.b_out_2 = nn.Parameter(torch.zeros(dim))

    def forward(self, h1, h2):
        from scripts.cheby_triton import cheby_scan_triton
        B, T, D = h1.size()
        pad_len = (4 - T % 4) % 4
        if pad_len > 0:
            h1 = F.pad(h1, (0, 0, 0, pad_len))
            h2 = F.pad(h2, (0, 0, 0, pad_len))
            
        A_1 = self.A_1 * torch.sigmoid(self.decay_1)
        h1_seq = cheby_scan_triton(h1, A_1)
        h1_norm = self.norm_1(h1_seq)
        for _ in range(self.iterations):
            h1_norm = h1_norm + F.gelu(h1_norm @ self.W_in_1 + self.b_in_1) @ self.W_out_1 + self.b_out_1
            
        A_2 = self.A_2 * torch.sigmoid(self.decay_2)
        h2_seq = cheby_scan_triton(h2, A_2)
        h2_norm = self.norm_2(h2_seq)
        for _ in range(self.iterations):
            h2_norm = h2_norm + F.gelu(h2_norm @ self.W_in_2 + self.b_in_2) @ self.W_out_2 + self.b_out_2
            
        h1_out = h1 + h1_norm
        h2_out = h2 + h2_norm
        
        if pad_len > 0:
            h1_out = h1_out[:, :T, :]
            h2_out = h2_out[:, :T, :]
            
        return h1_out, h2_out

class DoubleO(nn.Module):
    def __init__(self, vocab_size, config):
        super().__init__()
        dim = 128
        
        # The CEO
        self.router_emb = nn.Embedding(vocab_size, dim)
        self.router = TemporalRouter(dim, num_experts=2)
        
        # The Engineers
        self.token_emb_1 = nn.Embedding(vocab_size, dim)
        self.token_emb_2 = nn.Embedding(vocab_size, dim)
        
        self.blocks = nn.ModuleList([
            DoubleOBlock(dim=dim, expansion=config["expansion"], iterations=config["iterations"])
            for _ in range(config["num_layers"])
        ])
        
        self.norm_post_1 = nn.RMSNorm(dim)
        self.norm_post_2 = nn.RMSNorm(dim)
        
        self.lm_head_1 = nn.Linear(dim, vocab_size, bias=False)
        self.lm_head_2 = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, x, targets=None):
        r_emb = self.router_emb(x)
        w = self.router(r_emb)
        
        h1 = self.token_emb_1(x)
        h2 = self.token_emb_2(x)
        
        for block in self.blocks:
            h1, h2 = block(h1, h2)
            
        logits_1 = self.lm_head_1(self.norm_post_1(h1))
        logits_2 = self.lm_head_2(self.norm_post_2(h2))
        
        # Final Resonant Consensus
        logits = w[:, :, 0:1] * logits_1 + w[:, :, 1:2] * logits_2
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
        return logits, loss, w

def compute_ceo_fisher(model, train_A, device, block_size=256, num_samples=1000):
    model.eval()
    fisher_dict = {}
    optimal_weights = {}
    
    ceo_names = ["router_emb", "router"]
    for name, param in model.named_parameters():
        if any(c in name for c in ceo_names) and param.requires_grad:
            fisher_dict[name] = torch.zeros_like(param)
            optimal_weights[name] = param.clone().detach()
            
    count = 0
    for _ in range(num_samples):
        ix = torch.randint(len(train_A) - block_size, (1,))
        x = train_A[ix:ix+block_size].unsqueeze(0).to(device)
        y = train_A[ix+1:ix+block_size+1].unsqueeze(0).to(device)
        
        model.zero_grad()
        logits, loss, _ = model(x, y)
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
    
    with open("/root/data/tinyshakespeare.txt", "r", encoding='utf-8') as f:
        text_A = f.read()
        
    text_B = ""
    with open("/root/data/math.jsonl", "r", encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            text_B += obj["text"] + "\n"
            
    chars = sorted(list(set(text_A + text_B)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    
    data_A = torch.tensor([stoi[c] for c in text_A], dtype=torch.long)
    data_B = torch.tensor([stoi[c] for c in text_B], dtype=torch.long)
    
    nA = int(0.9 * len(data_A))
    train_A, val_A = data_A[:nA], data_A[nA:]
    
    nB = int(0.9 * len(data_B))
    train_B, val_B = data_B[:nB], data_B[nB:]
    
    def get_batch(split, task='A', batch_size=64, block_size=256):
        data_split = (train_A if split == 'train' else val_A) if task == 'A' else (train_B if split == 'train' else val_B)
        ix = torch.randint(len(data_split) - block_size, (batch_size,))
        x = torch.stack([data_split[i:i+block_size] for i in ix])
        y = torch.stack([data_split[i+1:i+block_size+1] for i in ix])
        return x.cuda(), y.cuda()

    config = {"num_layers": 6, "expansion": 4, "iterations": 4}
    model = DoubleO(vocab_size=vocab_size, config=config).cuda()
        
    params = sum(p.numel() for p in model.parameters())
    print(f"[DoubleO] Initialized. Vocab: {vocab_size}, Params: {params}")
    
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
    model = torch.compile(model)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
    
    metrics = {
        "val_A": [],
        "val_B": []
    }
    
    STEPS_PHASE_1 = 2000
    STEPS_PHASE_2 = 2000
    TOTAL_STEPS = STEPS_PHASE_1 + STEPS_PHASE_2
    
    fisher_dict = None
    optimal_weights = None
    lambda_ewc = 0.01
    
    # During Phase 1, we freeze Expert 2 entirely so it stays pristine for Task B
    for name, param in model.named_parameters():
        if "_2" in name:
            param.requires_grad = False
            
    for step in range(TOTAL_STEPS):
        model.train()
        
        current_task = 'A' if step < STEPS_PHASE_1 else 'B'
        
        if step == STEPS_PHASE_1:
            print("\n[DoubleO] Phase 1 Complete. Computing FIM exclusively on CEO (Router)...")
            fisher_dict, optimal_weights = compute_ceo_fisher(model, train_A, device=torch.device("cuda"))
            
            frozen_count = 0
            for name, param in model.named_parameters():
                if "_1" in name:
                    param.requires_grad = False
                    frozen_count += param.numel()
                if "_2" in name:
                    param.requires_grad = True # Unfreeze Expert 2
            print(f"[DoubleO] Frozen Expert 1 ({frozen_count} params). Phase 2 starting.\n")
        
        x, y = get_batch('train', task=current_task)
        logits, ce_loss, w = model(x, y)
        
        # Add a tiny router auxiliary loss to encourage it to use Expert 1 in Phase 1, and Expert 2 in Phase 2
        aux_target = 0 if current_task == 'A' else 1
        w_mean = w.mean(dim=(0, 1))
        aux_loss = 0.1 * ((w_mean[aux_target] - 1.0) ** 2)
        
        loss = ce_loss + aux_loss
        ewc_penalty = 0.0
        
        if current_task == 'B' and fisher_dict is not None:
            for name, param in model.named_parameters():
                if name in fisher_dict:
                    ewc_penalty += (fisher_dict[name] * (param - optimal_weights[name]) ** 2).sum()
            
            if step % 500 == 0:
                print(f"  --> [EWC Diag] CE: {ce_loss.item():.4f} | Raw Penalty: {ewc_penalty.item():.2f} | Weighted: {((lambda_ewc / 2) * ewc_penalty).item():.4f}")
                
            loss = loss + (lambda_ewc / 2) * ewc_penalty
            
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if step % 50 == 0 or step == (TOTAL_STEPS-1):
            model.eval()
            with torch.no_grad():
                vx_A, vy_A = get_batch('val', task='A')
                _, vloss_A, _ = model(vx_A, vy_A)
                metrics["val_A"].append((step, vloss_A.item()))
                
                vloss_B_val = None
                if step >= STEPS_PHASE_1:
                    vx_B, vy_B = get_batch('val', task='B')
                    _, vloss_B, _ = model(vx_B, vy_B)
                    metrics["val_B"].append((step, vloss_B.item()))
                    vloss_B_val = vloss_B.item()
                    
            if step < STEPS_PHASE_1:
                print(f"[DoubleO] P1 (Shakespeare) Step {step:4d} | CE: {ce_loss.item():.4f} | Val A: {vloss_A.item():.4f} | Router w1={w_mean[0].item():.2f}")
            else:
                ewc_val = ewc_penalty.item() if isinstance(ewc_penalty, torch.Tensor) else 0.0
                print(f"[DoubleO] P2 (Math) Step {step:4d} | CE: {ce_loss.item():.4f} | EWC: {ewc_val:.2f} | Val A: {vloss_A.item():.4f} | Val B: {vloss_B_val:.4f} | Router w2={w_mean[1].item():.2f}")
            
    return {
        "name": "doubleo",
        "params": params,
        "metrics": metrics
    }

@app.local_entrypoint()
def main():
    print("Launching DoubleO Continual Learning Gauntlet on Modal...")
    res = run_model_eval.remote()
    
    final_dict = {res["name"]: res}
    with open("data/cheby_moe_results.json", "w") as f:
        json.dump(final_dict, f)
        
    print("Continual Gauntlet complete! Results saved to data/cheby_moe_results.json")
