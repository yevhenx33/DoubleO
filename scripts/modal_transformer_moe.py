import os
import json
import modal
import torch
import torch.nn as nn
import torch.nn.functional as F

app = modal.App("transformer-moe-continual")

image = modal.Image.debian_slim().pip_install(
    "torch", "transformers"
).add_local_dir("data", remote_path="/root/data").add_local_dir("src", remote_path="/root/src")

secrets = [modal.Secret.from_dotenv()]

class LinearCEO(nn.Module):
    """Minimal linear router — no attention, just a projection.
    Tiny parameter count makes L2 anchoring trivially effective."""
    def __init__(self, dim, num_experts=2, temperature=2.0):
        super().__init__()
        self.proj = nn.Linear(dim, num_experts, bias=False)
        self.temperature = temperature
        
    def forward(self, x):
        logits = self.proj(x)
        w = torch.softmax(logits / self.temperature, dim=-1)
        return w, logits

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

class TransformerMoE(nn.Module):
    def __init__(self, vocab_size, config):
        super().__init__()
        from src.nanogpt.model import GPTConfig
        self.config = config
        
        self.wte_ceo = nn.Embedding(vocab_size, config.n_embd)
        self.wpe_ceo = nn.Embedding(config.block_size, config.n_embd)
        
        self.wte_1 = nn.Embedding(vocab_size, config.n_embd)
        self.wpe_1 = nn.Embedding(config.block_size, config.n_embd)
        
        self.wte_2 = nn.Embedding(vocab_size, config.n_embd)
        self.wpe_2 = nn.Embedding(config.block_size, config.n_embd)
        
        self.drop = nn.Dropout(config.dropout)
        
        # The Hub — minimal linear router
        self.ceo = LinearCEO(config.n_embd, num_experts=2)
        
        # The Spokes
        spoke_config = GPTConfig(block_size=config.block_size, vocab_size=vocab_size, 
                                 n_layer=config.spoke_layers, n_head=config.n_head, n_embd=config.n_embd, 
                                 dropout=config.dropout, bias=config.bias)
        
        self.spoke_1 = TransformerSpoke(spoke_config, num_layers=config.spoke_layers)
        self.spoke_2 = TransformerSpoke(spoke_config, num_layers=config.spoke_layers)
        
        self.lm_head_1 = nn.Linear(config.n_embd, vocab_size, bias=False)
        self.lm_head_2 = nn.Linear(config.n_embd, vocab_size, bias=False)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        
        x_ceo = self.drop(self.wte_ceo(idx) + self.wpe_ceo(pos))
        w, r_logits = self.ceo(x_ceo) # [B, T, 2]
        
        # Spoke processing
        x_1 = self.drop(self.wte_1(idx) + self.wpe_1(pos))
        x_2 = self.drop(self.wte_2(idx) + self.wpe_2(pos))
        
        h1 = self.spoke_1(x_1)
        h2 = self.spoke_2(x_2)
        
        logits_1 = self.lm_head_1(h1)
        logits_2 = self.lm_head_2(h2)
        
        # Final Hub-and-Spoke Consensus
        logits = w[:, :, 0:1] * logits_1 + w[:, :, 1:2] * logits_2
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
        return logits, loss, w, r_logits, logits_1, logits_2

def snapshot_ceo_weights(model):
    """Save a deep copy of all CEO parameters for L2 anchoring."""
    anchor = {}
    ceo_names = ["ceo", "wte_ceo", "wpe_ceo"]
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
    model = TransformerMoE(vocab_size=vocab_size, config=config).cuda()
        
    params = sum(p.numel() for p in model.parameters())
    print(f"[Transformer-MoE] Initialized. Vocab: {vocab_size}, Params: {params}")
    
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
    # Note: torch.compile disabled — incompatible with dynamic freeze/unfreeze
    
    # --- Main optimizer (CEO + Spoke 1) ---
    optimizer_main = torch.optim.AdamW(
        [p for n, p in model.named_parameters() if "_2" not in n and p.requires_grad],
        lr=1e-3, weight_decay=0.1
    )
    optimizer_spoke2 = None  # Created after Phase 1 unfreezes Spoke 2
    
    metrics = {
        "val_A": [],
        "val_B": []
    }
    
    STEPS_PHASE_1 = 2000
    STEPS_PHASE_2 = 2000
    TOTAL_STEPS = STEPS_PHASE_1 + STEPS_PHASE_2
    
    ceo_anchor = None
    lambda_l2 = 2.0          # Moderate L2 — allow some adaptation
    rehearsal_ratio = 0.15   # 15% rehearsal
    
    # Phase 1: Freeze Spoke 2
    for name, param in model.named_parameters():
        if "_2" in name:
            param.requires_grad = False
            
    for step in range(TOTAL_STEPS):
        model.train()
        
        current_task = 'A' if step < STEPS_PHASE_1 else 'B'
        
        if step == STEPS_PHASE_1:
            print("\n[Transformer-MoE] Phase 1 Complete. Snapshotting CEO weights for L2 anchor...")
            ceo_anchor = snapshot_ceo_weights(model)
            
            frozen_count = 0
            for name, param in model.named_parameters():
                if "_1" in name:
                    param.requires_grad = False
                    frozen_count += param.numel()
                if "_2" in name:
                    param.requires_grad = True
            
            # Create Spoke 2 optimizer now that its params require grad
            spoke2_params = [p for n, p in model.named_parameters() if "_2" in n and p.requires_grad]
            optimizer_spoke2 = torch.optim.AdamW(spoke2_params, lr=1e-3, weight_decay=0.1)
            
            # Rebuild main optimizer to exclude frozen Spoke 1 params
            optimizer_main = torch.optim.AdamW(
                [p for n, p in model.named_parameters() if "_2" not in n and "_1" not in n and p.requires_grad],
                lr=1e-3, weight_decay=0.1
            )
            print(f"[Transformer-MoE] Frozen Spoke 1 ({frozen_count} params). Phase 2 starting.\n")
        
        # --- Rehearsal buffer: mix 15% Shakespeare during Phase 2 ---
        if current_task == 'B' and torch.rand(1).item() < rehearsal_ratio:
            x, y = get_batch('train', task='A')
            rehearsal = True
        else:
            x, y = get_batch('train', task=current_task)
            rehearsal = False
        
        logits, ce_loss, w, r_logits, logits_1, logits_2 = model(x, y)
        
        # Router auxiliary: push toward the correct spoke
        if rehearsal:
            aux_target_idx = 0  # Shakespeare -> Spoke 1
        else:
            aux_target_idx = 0 if current_task == 'A' else 1
        router_target = torch.full((r_logits.size(0) * r_logits.size(1),), aux_target_idx, dtype=torch.long, device=x.device)
        aux_loss = F.cross_entropy(r_logits.view(-1, 2), router_target)
        
        w_mean = w.mean(dim=(0, 1))
        
        if current_task == 'A':
            # Phase 1: train CEO + Spoke 1 normally
            loss = ce_loss + 0.1 * aux_loss
            optimizer_main.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer_main.step()
        else:
            # Phase 2: two independent objectives
            # 1) Train Spoke 2 directly on Math (bypass router)
            spoke2_loss = F.cross_entropy(logits_2.view(-1, logits_2.size(-1)), y.view(-1))
            optimizer_spoke2.zero_grad(set_to_none=True)
            spoke2_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_([p for p in spoke2_params], 1.0)
            optimizer_spoke2.step()
            
            # 2) Train CEO router with L2 anchor + aux loss
            router_loss = 0.1 * aux_loss
            l2_penalty = 0.0
            if ceo_anchor is not None:
                for name, param in model.named_parameters():
                    if name in ceo_anchor:
                        l2_penalty += ((param - ceo_anchor[name]) ** 2).sum()
                router_loss = router_loss + lambda_l2 * l2_penalty
            
            if step % 500 == 0:
                l2_val = l2_penalty.item() if isinstance(l2_penalty, torch.Tensor) else 0.0
                print(f"  --> [Diag] Spoke2 CE: {spoke2_loss.item():.4f} | Router Aux: {aux_loss.item():.4f} | L2: {l2_val:.4f}")
            
            optimizer_main.zero_grad(set_to_none=True)
            router_loss.backward()
            torch.nn.utils.clip_grad_norm_([p for n, p in model.named_parameters() if p.requires_grad and "_2" not in n], 1.0)
            optimizer_main.step()
        
        if step % 50 == 0 or step == (TOTAL_STEPS-1):
            model.eval()
            with torch.no_grad():
                vx_A, vy_A = get_batch('val', task='A')
                _, vloss_A, _, _, logits_1_v, logits_2_v = model(vx_A, vy_A)
                # Direct spoke losses (bypass router)
                vloss_A_s1 = F.cross_entropy(logits_1_v.view(-1, logits_1_v.size(-1)), vy_A.view(-1)).item()
                vloss_A_s2 = F.cross_entropy(logits_2_v.view(-1, logits_2_v.size(-1)), vy_A.view(-1)).item()
                metrics["val_A"].append((step, vloss_A_s1))  # Track Spoke 1's Shakespeare performance
                
                vloss_B_s2 = None
                if step >= STEPS_PHASE_1:
                    vx_B, vy_B = get_batch('val', task='B')
                    _, vloss_B, _, _, logits_1_vb, logits_2_vb = model(vx_B, vy_B)
                    vloss_B_s2 = F.cross_entropy(logits_2_vb.view(-1, logits_2_vb.size(-1)), vy_B.view(-1)).item()
                    metrics["val_B"].append((step, vloss_B_s2))  # Track Spoke 2's Math performance
                    
            if step < STEPS_PHASE_1:
                print(f"[Transformer-MoE] P1 Step {step:4d} | CE: {ce_loss.item():.4f} | Spoke1-Shakespeare: {vloss_A_s1:.4f} | Router w1={w_mean[0].item():.2f}")
            else:
                l2_val = l2_penalty.item() if isinstance(l2_penalty, torch.Tensor) else 0.0
                print(f"[Transformer-MoE] P2 Step {step:4d} | Spoke1-Shk: {vloss_A_s1:.4f} | Spoke2-Math: {vloss_B_s2:.4f} | L2: {l2_val:.4f} | Router w2={w_mean[1].item():.2f}")
            
    return {
        "name": "transformer_moe",
        "params": params,
        "metrics": metrics
    }

@app.local_entrypoint()
def main():
    print("Launching Transformer Hub-and-Spoke Continual Learning on Modal...")
    res = run_model_eval.remote()
    
    final_dict = {res["name"]: res}
    with open("data/transformer_moe_results.json", "w") as f:
        json.dump(final_dict, f)
        
    print("Continual Gauntlet complete! Results saved to data/transformer_moe_results.json")
