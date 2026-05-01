"""
Extended model definitions for the 4-task continual learning experiment.
Both StandardGPT and DoubleOGPT at a larger scale (6 layers, 256 dim, 8 heads).
"""
import math
import torch
import torch.nn as nn
from torch.nn import functional as F


# ============================================================
# Shared attention (identical in both architectures)
# ============================================================

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
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


# ============================================================
# StandardGPT (Baseline) — LayerNorm + GELU MLP
# ============================================================

class StandardMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x, task_idx):
        return self.c_proj(F.gelu(self.c_fc(x)))


class StandardBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = StandardMLP(config)

    def forward(self, x, task_idx):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x), task_idx)
        return x


class StandardGPTConfig:
    vocab_size: int = 65
    block_size: int = 128
    n_layer: int = 6
    n_head: int = 8
    n_embd: int = 256


class StandardGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.current_task_idx = 0
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(0.1),
            h=nn.ModuleList([StandardBlock(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

    def set_task(self, idx):
        self.current_task_idx = idx

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        x = self.transformer.drop(self.transformer.wte(idx) + self.transformer.wpe(pos))
        for block in self.transformer.h:
            x = block(x, self.current_task_idx)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


# ============================================================
# DoubleOGPT — RMSNorm + Complex Chebyshev MLP (4 tasks)
# ============================================================

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        norm = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return x / norm * self.weight


class ChebyshevMLP(nn.Module):
    """
    Complex-valued MLP with task-conditioned Chebyshev polynomial activations.

    Supports 4 tasks routed through T1, T2, T3, T4:
        T1(z) = z
        T2(z) = 2z^2 - 1
        T3(z) = 4z^3 - 3z
        T4(z) = 8z^4 - 8z^2 + 1

    Activations are clamped to [-clamp_val, clamp_val] before polynomial
    application to prevent numerical explosion from higher-order terms.
    """
    def __init__(self, config):
        super().__init__()
        inner_dim = config.n_embd * 2
        self.c_fc_real = nn.Linear(config.n_embd, inner_dim, bias=False)
        self.c_fc_imag = nn.Linear(config.n_embd, inner_dim, bias=False)
        self.c_proj_real = nn.Linear(inner_dim, config.n_embd, bias=False)
        self.c_proj_imag = nn.Linear(inner_dim, config.n_embd, bias=False)
        self.clamp_val = getattr(config, 'clamp_val', 2.0)
        self.use_task_scale = getattr(config, 'use_task_scale', False)
        if self.use_task_scale:
            num_tasks = getattr(config, 'num_tasks', 4)
            self.task_scale = nn.ParameterList([
                nn.Parameter(torch.ones(inner_dim)) for _ in range(num_tasks)
            ])

    def forward(self, x, task_idx):
        z_r = self.c_fc_real(x)
        z_i = self.c_fc_imag(x)
        if self.use_task_scale:
            s = self.task_scale[task_idx]
            z_r = z_r * s
            z_i = z_i * s
        if self.clamp_val > 0:
            z_r = z_r.clamp(-self.clamp_val, self.clamp_val)
            z_i = z_i.clamp(-self.clamp_val, self.clamp_val)

        if task_idx == 0:
            # T1(z) = z
            p_r, p_i = z_r, z_i

        elif task_idx == 1:
            # T2(z) = 2z^2 - 1
            # z^2 = (z_r + iz_i)^2 = (z_r^2 - z_i^2) + i(2 z_r z_i)
            p_r = 2 * (z_r ** 2 - z_i ** 2) - 1
            p_i = 4 * z_r * z_i

        elif task_idx == 2:
            # T3(z) = 4z^3 - 3z
            # z^3 = (z_r^3 - 3 z_r z_i^2) + i(3 z_r^2 z_i - z_i^3)
            p_r = 4 * (z_r ** 3 - 3 * z_r * z_i ** 2) - 3 * z_r
            p_i = 4 * (3 * z_r ** 2 * z_i - z_i ** 3) - 3 * z_i

        elif task_idx == 3:
            # T4(z) = 8z^4 - 8z^2 + 1
            # z^4 real = z_r^4 - 6 z_r^2 z_i^2 + z_i^4
            # z^4 imag = 4 z_r z_i (z_r^2 - z_i^2)
            z_r2 = z_r ** 2
            z_i2 = z_i ** 2
            z4_r = z_r2 ** 2 - 6 * z_r2 * z_i2 + z_i2 ** 2
            z4_i = 4 * z_r * z_i * (z_r2 - z_i2)
            z2_r = z_r2 - z_i2
            z2_i = 2 * z_r * z_i
            p_r = 8 * z4_r - 8 * z2_r + 1
            p_i = 8 * z4_i - 8 * z2_i
        else:
            raise ValueError(f"Unsupported task_idx={task_idx}. Max 4 tasks (0-3).")

        return self.c_proj_real(p_r) + self.c_proj_imag(p_i)


def _make_norm(config):
    norm_type = getattr(config, 'norm_type', 'rmsnorm')
    if norm_type == 'layernorm':
        return nn.LayerNorm(config.n_embd)
    return RMSNorm(config.n_embd)


class DoubleOBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = _make_norm(config)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = _make_norm(config)
        self.mlp = ChebyshevMLP(config)

    def forward(self, x, task_idx):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x), task_idx)
        return x


class DoubleOGPTConfig:
    vocab_size: int = 65
    block_size: int = 128
    n_layer: int = 6
    n_head: int = 8
    n_embd: int = 256
    norm_type: str = 'rmsnorm'   # 'rmsnorm' or 'layernorm'
    clamp_val: float = 2.0       # 0.0 = no clamping
    use_task_scale: bool = False # per-task diagonal scaling
    num_tasks: int = 4           # number of tasks (for task_scale)


class DoubleOGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.current_task_idx = 0
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(0.1),
            h=nn.ModuleList([DoubleOBlock(config) for _ in range(config.n_layer)]),
            ln_f=_make_norm(config),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

    def set_task(self, idx):
        self.current_task_idx = idx

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        x = self.transformer.drop(self.transformer.wte(idx) + self.transformer.wpe(pos))
        for block in self.transformer.h:
            x = block(x, self.current_task_idx)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
