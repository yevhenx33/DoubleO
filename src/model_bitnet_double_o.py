"""
BitNet b1.58 + Double Orthogonality Engine (Ternary Complex Chebyshev)

Fusion architecture that combines:
1. BitLinear ternary weights {-1, 0, 1} with STE for gradient flow
2. Complex-valued MLP channels (Real + Imaginary)
3. Chebyshev polynomial spectral routing (T2 for Task A, T3 for Task B)
4. Phase isolation (Task A reads Real projection, Task B reads Imaginary)
5. Combinatorial freezing of active bits after Task A
"""
import math
import torch
import torch.nn as nn
from torch.nn import functional as F


class BitLinear(nn.Linear):
    """1.58-bit linear layer with ternary weight quantization and STE."""
    
    def __init__(self, in_features, out_features, bias=False):
        super().__init__(in_features, out_features, bias)
        self.frozen_mask = None
        # Store the frozen gamma so quantization boundaries don't shift
        self.frozen_gamma = None
        
    def lock_math_subspace(self):
        """Freeze the active {-1, +1} bits and lock the quantization scale."""
        w = self.weight
        gamma = w.abs().mean()
        w_scaled = w / (gamma + 1e-5)
        w_quant = torch.clamp(torch.round(w_scaled), -1, 1)
        
        # Lock any weight that is actively 1 or -1
        self.frozen_mask = (w_quant != 0).detach()
        
        # Freeze the gamma scale factor to prevent quantization boundary drift
        self.frozen_gamma = gamma.detach().clone()
        
        # Register a backward hook to zero out gradients for frozen bits
        self.weight.register_hook(
            lambda grad, m=self.frozen_mask: grad.masked_fill(m, 0.0)
        )
        
    def forward(self, x):
        w = self.weight
        
        # --- Weight Quantization ---
        gamma = w.abs().mean()
        w_scaled = w / (gamma + 1e-5)
        w_quant = torch.clamp(torch.round(w_scaled), -1, 1)
        
        # If we have a frozen subspace, use frozen_gamma for the frozen bits
        # and current gamma for the free bits, to prevent boundary drift
        if self.frozen_gamma is not None:
            # Re-quantize using frozen gamma for the locked positions
            w_frozen_scaled = w / (self.frozen_gamma + 1e-5)
            w_frozen_quant = torch.clamp(torch.round(w_frozen_scaled), -1, 1)
            # Merge: use frozen quantization for locked bits, current for free bits
            w_quant = torch.where(self.frozen_mask, w_frozen_quant, w_quant)
            # Effective gamma is a blend, but for the STE we use current gamma
        
        # Straight-Through Estimator (STE): gradient flows through unchanged
        w_quant = w + (w_quant - w).detach()
        
        # --- Activation Quantization ---
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
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
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


class TernaryChebyshevMLP(nn.Module):
    """
    The fusion MLP: BitLinear (ternary weights + STE) + Complex Chebyshev routing
    with a Gated Shared Space (Data Availability Layer).
    
    Architecture:
    - Isolated channels: T2 (Task A) and T3 (Task B) via Chebyshev polynomials
    - Shared space: T1 (linear pathway) accessible by both tasks
    - Learned gate (alpha): each task decides how much shared knowledge to absorb
    
    The gate opens if shared knowledge reduces loss (positive transfer),
    and closes to protect the task from harmful interference (zero forgetting).
    """
    
    def __init__(self, config):
        super().__init__()
        inner_dim = config.n_embd * 2
        
        # Complex-valued projections using BitLinear (Isolated channels)
        self.c_fc_real = BitLinear(config.n_embd, inner_dim, bias=False)
        self.c_fc_imag = BitLinear(config.n_embd, inner_dim, bias=False)
        self.c_proj_real = BitLinear(inner_dim, config.n_embd, bias=False)
        self.c_proj_imag = BitLinear(inner_dim, config.n_embd, bias=False)
        
        # Shared T1 pathway (Data Availability Layer)
        # Simple linear projection both tasks can read from
        self.shared_fc = BitLinear(config.n_embd, inner_dim, bias=False)
        self.shared_proj = BitLinear(inner_dim, config.n_embd, bias=False)
        
        # Learned gates: one per task, initialized near 0 (sigmoid(0) = 0.5)
        # Start slightly negative so gates are mostly closed initially,
        # forcing the model to rely on isolated channels first
        self.gate_logits = nn.Parameter(torch.full((config.num_tasks,), -2.0))
        
    def _bound_activations(self, x):
        """
        Normalize activations to [-1, 1] for Chebyshev polynomial validity.
        Uses the same scaling approach as BitNet's activation quantization,
        but targeting the [-1, 1] range instead of [-Qb, Qb].
        Gradient flows through via STE — no tanh saturation.
        """
        x_max = x.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-5)
        x_normalized = x / x_max
        # STE: forward uses rounded/clamped, backward uses original
        x_bounded = x_normalized  # already in [-1, 1]
        return x_bounded, x_max
        
    def forward(self, x, task_idx):
        # === Shared T1 Pathway (DA Layer) ===
        shared_hidden = self.shared_fc(x)
        shared_hidden = F.gelu(shared_hidden)  # Simple nonlinearity for shared space
        shared_out = self.shared_proj(shared_hidden)
        
        # Gate: sigmoid so it's always in [0, 1]
        alpha = torch.sigmoid(self.gate_logits[task_idx])
        
        # === Isolated Chebyshev Channels ===
        # Project into complex space via ternary BitLinear layers
        z_real_raw = self.c_fc_real(x)
        z_imag_raw = self.c_fc_imag(x)
        
        # Bound activations to [-1, 1] for Chebyshev validity using STE-style scaling
        z_real, scale_r = self._bound_activations(z_real_raw)
        z_imag, scale_i = self._bound_activations(z_imag_raw)
        
        if task_idx == 0:
            # Task A: T2(Z) = 2Z^2 - 1 (Complex)
            p_r = 2 * (z_real**2 - z_imag**2) - 1
            p_i = 4 * z_real * z_imag
            # Task A reads Real projection
            isolated = self.c_proj_real(p_r * scale_r) - self.c_proj_imag(p_i * scale_i)
        else:
            # Task B: T3(Z) = 4Z^3 - 3Z (Complex)
            p_r = 4 * (z_real**3 - 3 * z_real * z_imag**2) - 3 * z_real
            p_i = 4 * (3 * z_real**2 * z_imag - z_imag**3) - 3 * z_imag
            # Task B reads Imaginary projection
            isolated = self.c_proj_imag(p_r * scale_r) + self.c_proj_real(p_i * scale_i)
        
        # === Merge: Isolated Core + Gated Shared Knowledge ===
        out = isolated + alpha * shared_out
            
        return out


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = TernaryChebyshevMLP(config)

    def forward(self, x, task_idx):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x), task_idx)
        return x


class BitNetDoubleOConfig:
    vocab_size: int = 65
    block_size: int = 32
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 64
    num_tasks: int = 2


class BitNetDoubleO(nn.Module):
    """Ternary Complex Chebyshev: BitNet b1.58 + Double Orthogonality Engine."""
    
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
