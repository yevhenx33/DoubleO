import math
import torch
import torch.nn as nn
from torch.nn import functional as F

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        norm = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return x / norm * self.weight

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

class ChebyshevMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc_real = nn.Linear(config.n_embd, config.n_embd * 2, bias=False)
        self.c_fc_imag = nn.Linear(config.n_embd, config.n_embd * 2, bias=False)
        self.c_proj_real = nn.Linear(config.n_embd * 2, config.n_embd, bias=False)
        self.c_proj_imag = nn.Linear(config.n_embd * 2, config.n_embd, bias=False)
        self.use_task_scale = getattr(config, 'use_task_scale', False)
        if self.use_task_scale:
            num_tasks = getattr(config, 'num_tasks', 2)
            self.task_scale = nn.ParameterList([
                nn.Parameter(torch.ones(config.n_embd * 2)) for _ in range(num_tasks)
            ])
        
    def forward(self, x, task_idx):
        z_real = self.c_fc_real(x)
        z_imag = self.c_fc_imag(x)
        
        if self.use_task_scale:
            s = self.task_scale[task_idx]
            z_real = z_real * s
            z_imag = z_imag * s
        
        if task_idx == 0:
            # Task A: T1 (Linear)
            p_r = z_real
            p_i = z_imag
        else:
            # Task B: T3 (Cubic)
            p_r = 4 * (z_real**3 - 3 * z_real * z_imag**2) - 3 * z_real
            p_i = 4 * (3 * z_real**2 * z_imag - z_imag**3) - 3 * z_imag
            
        return self.c_proj_real(p_r) + self.c_proj_imag(p_i)

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = RMSNorm(config.n_embd)
        self.mlp = ChebyshevMLP(config)

    def forward(self, x, task_idx):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x), task_idx)
        return x

class DoubleOGPTConfig:
    vocab_size: int = 65
    block_size: int = 32
    n_layer: int = 2
    n_head: int = 2
    n_embd: int = 32
    use_task_scale: bool = False
    num_tasks: int = 2

class DoubleOGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.current_task_idx = 0
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(0.1),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = RMSNorm(config.n_embd),
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
