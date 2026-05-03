import math
import torch
import torch.nn as nn
from torch.nn import functional as F

class BitLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__(in_features, out_features, bias)
        self.frozen_mask = None
        
    def lock_math_subspace(self):
        w = self.weight
        gamma = w.abs().mean()
        w_scaled = w / (gamma + 1e-5)
        w_quant = torch.clamp(torch.round(w_scaled), -1, 1)
        
        # Lock any weight that is actively 1 or -1
        self.frozen_mask = (w_quant != 0).detach()
        
        # Register a backward hook to zero out gradients where mask is True
        self.weight.register_hook(lambda grad, m=self.frozen_mask: grad.masked_fill(m, 0.0))
        
    def forward(self, x):
        w = self.weight
        
        # Quantize Weights
        gamma = w.abs().mean()
        w_scaled = w / (gamma + 1e-5)
        w_quant = torch.clamp(torch.round(w_scaled), -1, 1)
        
        # Straight-Through Estimator (STE)
        w_quant = w + (w_quant - w).detach()
        
        # Quantize Activations
        Qb = 127.0
        x_gamma = x.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-5)
        x_scaled = x * (Qb / x_gamma)
        x_quant = torch.clamp(torch.round(x_scaled), -Qb, Qb)
        x_quant = x + (x_quant - x).detach()
        
        out = F.linear(x_quant, w_quant, self.bias)
        out = out * (x_gamma / Qb) * gamma
        return out

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y

class BitNetMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Replacing Standard nn.Linear with BitLinear
        self.c_fc = BitLinear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = BitLinear(4 * config.n_embd, config.n_embd, bias=False)
        
    def forward(self, x, task_idx):
        # Ignores task routing
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = BitNetMLP(config)

    def forward(self, x, task_idx):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x), task_idx)
        return x

class BitNetGPTConfig:
    vocab_size: int = 65
    block_size: int = 32
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 64

class BitNetGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.current_task_idx = 0
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(0.1),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

    def set_task(self, idx):
        self.current_task_idx = idx

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        
        for block in self.transformer.h:
            x = block(x, self.current_task_idx)
            
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
