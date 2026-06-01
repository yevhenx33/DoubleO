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
        self.use_task_lora = getattr(config, 'use_task_lora', False)
        num_tasks = getattr(config, 'num_tasks', 2)
        inner_dim = config.n_embd * 2
        
        self.inner_norm_real = nn.Tanh()
        self.inner_norm_imag = nn.Tanh()
        
        if self.use_task_scale:
            self.task_scale = nn.ParameterList([
                nn.Parameter(torch.ones(inner_dim)) for _ in range(num_tasks)
            ])
            
        if self.use_task_lora:
            import math
            self.lora_rank = getattr(config, 'lora_rank', 8)
            self.lora_alpha = getattr(config, 'lora_alpha', 16.0)
            
            # A projects from n_embd to lora_rank, B projects from lora_rank to inner_dim
            self.lora_A_real = nn.ParameterList([nn.Parameter(torch.empty(config.n_embd, self.lora_rank)) for _ in range(num_tasks)])
            self.lora_B_real = nn.ParameterList([nn.Parameter(torch.empty(self.lora_rank, inner_dim)) for _ in range(num_tasks)])
            self.lora_A_imag = nn.ParameterList([nn.Parameter(torch.empty(config.n_embd, self.lora_rank)) for _ in range(num_tasks)])
            self.lora_B_imag = nn.ParameterList([nn.Parameter(torch.empty(self.lora_rank, inner_dim)) for _ in range(num_tasks)])
            
            # Output adapters (applied after polynomial)
            # p_r/p_i are of size inner_dim, mapping back to n_embd
            self.lora_C_real = nn.ParameterList([nn.Parameter(torch.empty(inner_dim, self.lora_rank)) for _ in range(num_tasks)])
            self.lora_D_real = nn.ParameterList([nn.Parameter(torch.empty(self.lora_rank, config.n_embd)) for _ in range(num_tasks)])
            self.lora_C_imag = nn.ParameterList([nn.Parameter(torch.empty(inner_dim, self.lora_rank)) for _ in range(num_tasks)])
            self.lora_D_imag = nn.ParameterList([nn.Parameter(torch.empty(self.lora_rank, config.n_embd)) for _ in range(num_tasks)])
            
            self.task_bias_real = nn.ParameterList([nn.Parameter(torch.zeros(inner_dim)) for _ in range(num_tasks)])
            self.task_bias_imag = nn.ParameterList([nn.Parameter(torch.zeros(inner_dim)) for _ in range(num_tasks)])
            
            for i in range(num_tasks):
                nn.init.kaiming_uniform_(self.lora_A_real[i], a=math.sqrt(5))
                nn.init.zeros_(self.lora_B_real[i])
                nn.init.kaiming_uniform_(self.lora_A_imag[i], a=math.sqrt(5))
                nn.init.zeros_(self.lora_B_imag[i])
                
                nn.init.kaiming_uniform_(self.lora_C_real[i], a=math.sqrt(5))
                nn.init.zeros_(self.lora_D_real[i])
                nn.init.kaiming_uniform_(self.lora_C_imag[i], a=math.sqrt(5))
                nn.init.zeros_(self.lora_D_imag[i])
        
    def forward(self, x, task_idx):
        z_real = self.c_fc_real(x)
        z_imag = self.c_fc_imag(x)
        
        if self.use_task_scale:
            s = self.task_scale[task_idx]
            z_real = z_real * s
            z_imag = z_imag * s
            
        if self.use_task_lora:
            lora_r = (x @ self.lora_A_real[task_idx]) @ self.lora_B_real[task_idx]
            lora_i = (x @ self.lora_A_imag[task_idx]) @ self.lora_B_imag[task_idx]
            
            scaling = self.lora_alpha / self.lora_rank
            z_real = z_real + (lora_r * scaling) + self.task_bias_real[task_idx]
            z_imag = z_imag + (lora_i * scaling) + self.task_bias_imag[task_idx]
            
        z_real = self.inner_norm_real(z_real)
        z_imag = self.inner_norm_imag(z_imag)
        
        if task_idx == 0:
            # Task A: T2
            p_r = 2 * (z_real**2 - z_imag**2) - 1
            p_i = 4 * z_real * z_imag
            # Task A reads Real projection
            out = self.c_proj_real(p_r) - self.c_proj_imag(p_i)
        else:
            # Task B: T3
            p_r = 4 * (z_real**3 - 3 * z_real * z_imag**2) - 3 * z_real
            p_i = 4 * (3 * z_real**2 * z_imag - z_imag**3) - 3 * z_imag
            # Task B reads Imaginary projection
            out = self.c_proj_imag(p_r) + self.c_proj_real(p_i)
            
        if self.use_task_lora:
            scaling = self.lora_alpha / self.lora_rank
            out_lora_r_pr = (p_r @ self.lora_C_real[task_idx]) @ self.lora_D_real[task_idx]
            out_lora_i_pi = (p_i @ self.lora_C_imag[task_idx]) @ self.lora_D_imag[task_idx]
            out_lora_r_pi = (p_i @ self.lora_C_real[task_idx]) @ self.lora_D_real[task_idx]
            out_lora_i_pr = (p_r @ self.lora_C_imag[task_idx]) @ self.lora_D_imag[task_idx]
            
            if task_idx == 0:
                out = out + (out_lora_r_pr - out_lora_i_pi) * scaling
            else:
                out = out + (out_lora_i_pr + out_lora_r_pi) * scaling
            
        return out

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
    use_task_lora: bool = False
    lora_rank: int = 32
    lora_alpha: float = 32.0
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
