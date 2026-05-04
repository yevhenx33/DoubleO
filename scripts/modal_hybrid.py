"""
DoubleO v3 — Hybrid Hub-and-Spoke with Pre-Aggregation

Architecture:
  - Router: Chebyshev Temporal Router (from DoubleO — ultra-stable polynomial routing)
  - Spokes: Transformer Blocks (from H&S — richer representational capacity)
  - Aggregation: PRE-aggregation at hidden-state level, not logits.
    • h_mix = w1 * spoke_1(x1) + w2 * spoke_2(x2)
    • logits = shared_lm_head(h_mix)
  - Training: Dual optimizer + L2 anchor + rehearsal buffer
"""
import os
import json
import modal
import torch
import torch.nn as nn
import torch.nn.functional as F

app = modal.App("hybrid-doubleo-v3")

image = modal.Image.debian_slim().pip_install(
    "torch", "triton", "python-dotenv", "transformers"
).add_local_dir("data", remote_path="/root/data").add_local_dir("src", remote_path="/root/src").add_local_dir("scripts", remote_path="/root/scripts")

secrets = [modal.Secret.from_dotenv()]

# ============================================================
# Router: Chebyshev Temporal Router (from DoubleO)
# ============================================================
class ChebyshevRouter(nn.Module):
    def __init__(self, dim, num_experts=2):
        super().__init__()
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
        
        r = self.router_proj(x)
        A = self.W_a * torch.sigmoid(self.decay)
        r_seq = cheby_scan_triton(r, A)
        
        if pad_len > 0: r_seq = r_seq[:, :T, :]
        
        logits = self.out_proj(r_seq)
        w = torch.softmax(logits, dim=-1)
        return w

# ============================================================
# Spokes: Transformer Blocks (from H&S)
# ============================================================
class TransformerSpoke(nn.Module):
    def __init__(self, config, num_layers):
        super().__init__()
        from src.nanogpt.model import Block, LayerNorm
        self.blocks = nn.ModuleList([Block(config) for _ in range(num_layers)])
        self.ln_f = LayerNorm(config.n_embd, bias=config.bias)
        
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.ln_f(x)

# ============================================================
# Hybrid Model: Chebyshev Router + Transformer Spokes + Pre-Aggregation
# ============================================================
class HybridDoubleO(nn.Module):
    def __init__(self, vocab_size, config):
        super().__init__()
        from src.nanogpt.model import GPTConfig
        dim = config.n_embd
        
        # CEO: Chebyshev Temporal Router (isolated embedding)
        self.router_emb = nn.Embedding(vocab_size, dim)
        self.router = ChebyshevRouter(dim, num_experts=2)
        
        # Spoke embeddings (fully isolated)
        self.wte_1 = nn.Embedding(vocab_size, dim)
        self.wpe_1 = nn.Embedding(config.block_size, dim)
        
        self.wte_2 = nn.Embedding(vocab_size, dim)
        self.wpe_2 = nn.Embedding(config.block_size, dim)
        
        self.drop = nn.Dropout(config.dropout)
        
        # Transformer Spokes
        spoke_config = GPTConfig(block_size=config.block_size, vocab_size=vocab_size,
                                 n_layer=config.spoke_layers, n_head=config.n_head, n_embd=dim,
                                 dropout=config.dropout, bias=config.bias)
        self.spoke_1 = TransformerSpoke(spoke_config, num_layers=config.spoke_layers)
        self.spoke_2 = TransformerSpoke(spoke_config, num_layers=config.spoke_layers)
        
        # PRE-AGGREGATION with ISOLATED HEADS
        # Instead of: logits = w1 * head_1(h1) + w2 * head_2(h2)  [post-agg, mixes distributions]
        # We do:      logits = head_1(w1 * h1) + head_2(w2 * h2)  [pre-agg, mixes features before heads]
        # Each head sees a routing-modulated feature, not the raw spoke output.
        # This forces the heads to decode from a routing-aware representation.
        self.lm_head_1 = nn.Linear(dim, vocab_size, bias=False)
        self.lm_head_2 = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        
        # CEO routing via Chebyshev
        r_emb = self.router_emb(idx)
        w = self.router(r_emb) # [B, T, 2]
        
        # Spoke processing (fully isolated paths)
        x_1 = self.drop(self.wte_1(idx) + self.wpe_1(pos))
        x_2 = self.drop(self.wte_2(idx) + self.wpe_2(pos))
        
        h1 = self.spoke_1(x_1)
        h2 = self.spoke_2(x_2)
        
        # === PRE-AGGREGATION with ISOLATED HEADS ===
        # Route-modulate hidden states BEFORE the heads
        h1_routed = w[:, :, 0:1] * h1
        h2_routed = w[:, :, 1:2] * h2
        
        logits = self.lm_head_1(h1_routed) + self.lm_head_2(h2_routed)
        
        # Direct spoke logits (for isolated training, unmodulated)
        logits_1 = self.lm_head_1(h1)
        logits_2 = self.lm_head_2(h2)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
        return logits, loss, w, logits_1, logits_2

def snapshot_ceo_weights(model):
    anchor = {}
    ceo_names = ["router_emb", "router"]
    for name, param in model.named_parameters():
        if any(c in name for c in ceo_names):
            anchor[name] = param.clone().detach()
    return anchor

@app.function(image=image, secrets=secrets, gpu="H100", timeout=3600)
def run_model_eval():
    import sys
    import torch
    import json
    sys.path.append("/root")
    from src.nanogpt.model import GPTConfig
    
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

    class CustomConfig(GPTConfig):
        spoke_layers: int = 4

    config = CustomConfig(block_size=256, vocab_size=vocab_size, n_head=4, n_embd=128, dropout=0.0, bias=False)
    model = HybridDoubleO(vocab_size=vocab_size, config=config).cuda()
        
    params = sum(p.numel() for p in model.parameters())
    print(f"[Hybrid-v3] Initialized. Vocab: {vocab_size}, Params: {params}")
    
    # --- Main optimizer (CEO + Spoke 1 + shared LM head) ---
    optimizer_main = torch.optim.AdamW(
        [p for n, p in model.named_parameters() if "_2" not in n and p.requires_grad],
        lr=1e-3, weight_decay=0.1
    )
    optimizer_spoke2 = None
    
    metrics = {
        "val_A": [],
        "val_B": []
    }
    
    STEPS_PHASE_1 = 2000
    STEPS_PHASE_2 = 2000
    TOTAL_STEPS = STEPS_PHASE_1 + STEPS_PHASE_2
    
    ceo_anchor = None
    lambda_l2 = 2.0
    rehearsal_ratio = 0.15
    
    # Phase 1: Freeze Spoke 2
    for name, param in model.named_parameters():
        if "_2" in name:
            param.requires_grad = False
            
    for step in range(TOTAL_STEPS):
        model.train()
        
        current_task = 'A' if step < STEPS_PHASE_1 else 'B'
        
        if step == STEPS_PHASE_1:
            print("\n[Hybrid-v3] Phase 1 Complete. Snapshotting CEO weights for L2 anchor...")
            ceo_anchor = snapshot_ceo_weights(model)
            
            frozen_count = 0
            for name, param in model.named_parameters():
                if "_1" in name:
                    param.requires_grad = False
                    frozen_count += param.numel()
                if "_2" in name:
                    param.requires_grad = True
            
            spoke2_params = [p for n, p in model.named_parameters() if "_2" in n and p.requires_grad]
            optimizer_spoke2 = torch.optim.AdamW(spoke2_params, lr=1e-3, weight_decay=0.1)
            
            # Rebuild main optimizer (CEO + shared LM head, no spoke_1)
            optimizer_main = torch.optim.AdamW(
                [p for n, p in model.named_parameters() if "_2" not in n and "_1" not in n and p.requires_grad],
                lr=1e-3, weight_decay=0.1
            )
            print(f"[Hybrid-v3] Frozen Spoke 1 ({frozen_count} params). Phase 2 starting.\n")
        
        # Rehearsal buffer
        if current_task == 'B' and torch.rand(1).item() < rehearsal_ratio:
            x, y = get_batch('train', task='A')
            rehearsal = True
        else:
            x, y = get_batch('train', task=current_task)
            rehearsal = False
        
        logits, ce_loss, w, logits_1, logits_2 = model(x, y)
        
        aux_target = 0 if (current_task == 'A' or rehearsal) else 1
        w_mean = w.mean(dim=(0, 1))
        aux_loss = 0.1 * ((w_mean[aux_target] - 1.0) ** 2)
        
        if current_task == 'A':
            loss = ce_loss + aux_loss
            optimizer_main.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer_main.step()
        else:
            # Phase 2: Two separate forward passes to avoid in-place grad conflicts
            
            # Pass 1: Train Spoke 2 directly on Math
            spoke2_loss = F.cross_entropy(logits_2.view(-1, logits_2.size(-1)), y.view(-1))
            optimizer_spoke2.zero_grad(set_to_none=True)
            spoke2_loss.backward()
            torch.nn.utils.clip_grad_norm_([p for n, p in model.named_parameters() if "_2" in n], 1.0)
            optimizer_spoke2.step()
            
            # Pass 2: Fresh forward for CEO + shared head
            logits_2nd, ce_loss_2nd, w_2nd, _, _ = model(x, y)
            w_mean = w_2nd.mean(dim=(0, 1))
            aux_target_2 = 0 if rehearsal else 1
            aux_loss_2 = 0.1 * ((w_mean[aux_target_2] - 1.0) ** 2)
            
            router_loss = ce_loss_2nd + aux_loss_2
            l2_penalty = 0.0
            if ceo_anchor is not None:
                for name, param in model.named_parameters():
                    if name in ceo_anchor:
                        l2_penalty += ((param - ceo_anchor[name]) ** 2).sum()
                router_loss = router_loss + lambda_l2 * l2_penalty
            
            if step % 500 == 0:
                l2_val = l2_penalty.item() if isinstance(l2_penalty, torch.Tensor) else 0.0
                print(f"  --> [Diag] Spoke2 CE: {spoke2_loss.item():.4f} | Mix CE: {ce_loss_2nd.item():.4f} | L2: {l2_val:.4f}")
            
            optimizer_main.zero_grad(set_to_none=True)
            router_loss.backward()
            torch.nn.utils.clip_grad_norm_([p for n, p in model.named_parameters() if p.requires_grad and "_2" not in n], 1.0)
            optimizer_main.step()
        
        if step % 50 == 0 or step == (TOTAL_STEPS-1):
            model.eval()
            with torch.no_grad():
                vx_A, vy_A = get_batch('val', task='A')
                _, vloss_A, _, logits_1_v, _ = model(vx_A, vy_A)
                vloss_A_s1 = F.cross_entropy(logits_1_v.view(-1, logits_1_v.size(-1)), vy_A.view(-1)).item()
                # Also measure the pre-aggregated mixture loss
                vloss_A_mix = vloss_A.item()
                metrics["val_A"].append((step, vloss_A_s1))
                
                vloss_B_s2 = None
                vloss_B_mix = None
                if step >= STEPS_PHASE_1:
                    vx_B, vy_B = get_batch('val', task='B')
                    _, vloss_B, _, _, logits_2_v = model(vx_B, vy_B)
                    vloss_B_s2 = F.cross_entropy(logits_2_v.view(-1, logits_2_v.size(-1)), vy_B.view(-1)).item()
                    vloss_B_mix = vloss_B.item()
                    metrics["val_B"].append((step, vloss_B_s2))
                    
            if step < STEPS_PHASE_1:
                print(f"[Hybrid-v3] P1 Step {step:4d} | Mix: {vloss_A_mix:.4f} | Spoke1: {vloss_A_s1:.4f} | w1={w_mean[0].item():.2f}")
            else:
                l2_val = l2_penalty.item() if isinstance(l2_penalty, torch.Tensor) else 0.0
                print(f"[Hybrid-v3] P2 Step {step:4d} | Spoke1-Shk: {vloss_A_s1:.4f} | Spoke2-Math: {vloss_B_s2:.4f} | Mix-Math: {vloss_B_mix:.4f} | L2: {l2_val:.4f} | w2={w_mean[1].item():.2f}")
            
    return {
        "name": "hybrid_v3",
        "params": params,
        "metrics": metrics
    }

@app.local_entrypoint()
def main():
    print("Launching Hybrid DoubleO v3 on Modal...")
    res = run_model_eval.remote()
    
    final_dict = {res["name"]: res}
    with open("data/hybrid_v3_results.json", "w") as f:
        json.dump(final_dict, f)
        
    print("Continual Gauntlet complete! Results saved to data/hybrid_v3_results.json")
