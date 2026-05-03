"""
ChebyWave v4: Resonant Cavity + DoubleO Task Isolation.

Instead of stacking N feedforward layers, this uses a SINGLE resonant cavity
iterated until the wave state converges (Deep Equilibrium approach).

Key properties:
- Adaptive computation: easy problems converge fast, hard ones iterate longer
- Infinite effective depth with fixed parameter count
- Natural noise rejection: non-resonant frequencies decay with each iteration
- DoubleO isolation: T2 cavity for Task A, T3 cavity for Task B
- Shared DA layer for selective positive transfer
"""
import torch
import torch.nn as nn
from torch.nn import functional as F


class ResonantCavity(nn.Module):
    """
    A single resonant cavity that is iterated until convergence.
    Contains task-specific polynomial routing + shared DA layer.
    """
    def __init__(self, max_degree, num_tasks=2):
        super().__init__()

        # Spectral gate: which frequencies to amplify
        self.gate_norm = nn.LayerNorm(max_degree)
        self.gate_proj = nn.Linear(max_degree, max_degree)

        # Task-specific resonators (tuned to different polynomial harmonics)
        self.iso_norm = nn.LayerNorm(max_degree)
        self.iso_resonators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(max_degree, max_degree * 2),
                nn.GELU(),
                nn.Linear(max_degree * 2, max_degree),
            )
            for _ in range(num_tasks)
        ])

        # Shared DA pathway
        self.shared_norm = nn.LayerNorm(max_degree)
        self.shared_resonator = nn.Sequential(
            nn.Linear(max_degree, max_degree * 2),
            nn.GELU(),
            nn.Linear(max_degree * 2, max_degree),
        )

        # DA gate per task
        self.da_gate_logits = nn.Parameter(torch.full((num_tasks,), -2.0))

        # Damping factor: controls how much of the previous state survives
        # Learned per-frequency, initialized to ~0.9 (most energy retained)
        self.damping_logits = nn.Parameter(torch.full((max_degree,), 2.0))

    def _bound(self, x):
        x_max = x.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-5)
        return x / x_max

    def forward(self, h, task_idx):
        # Damping: older resonances decay, preventing runaway amplification
        damping = torch.sigmoid(self.damping_logits)
        h = h * damping

        # Spectral gate
        gate = torch.sigmoid(self.gate_proj(self.gate_norm(h)))
        h_gated = h + h * gate

        # Task-specific polynomial
        h_norm = self._bound(self.iso_norm(h_gated))
        if task_idx == 0:
            h_poly = 2 * h_norm**2 - 1      # T2
        else:
            h_poly = 4 * h_norm**3 - 3 * h_norm  # T3

        isolated = self.iso_resonators[task_idx](h_poly)

        # Shared DA
        shared = self.shared_resonator(self.shared_norm(h_gated))
        alpha = torch.sigmoid(self.da_gate_logits[task_idx])

        # Residual: the wave state evolves, not resets
        h_out = h + isolated + alpha * shared
        return h_out


class ChebyWaveConfig:
    vocab_size: int = 17
    max_degree: int = 128
    n_layer: int = 1         # only 1 cavity, iterated
    num_tasks: int = 2
    block_size: int = 32
    max_iterations: int = 8  # max resonance bounces
    convergence_threshold: float = 1e-3


class ChebyWave(nn.Module):
    """
    ChebyWave v4: Resonant Cavity.

    One cavity, iterated until convergence. Adaptive computation depth.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.max_degree = config.max_degree
        self.current_task_idx = 0
        self.max_iters = config.max_iterations
        self.conv_thresh = config.convergence_threshold

        # Single resonant cavity (iterated)
        self.cavity = ResonantCavity(config.max_degree, config.num_tasks)

        # Final norm + decoder
        self.ln_f = nn.LayerNorm(config.max_degree)
        self.decoder = nn.Linear(config.max_degree, config.vocab_size)

        total = sum(p.numel() for p in self.parameters())
        print(f"  [ChebyWave v4 Resonant] Params: {total:,} | Cavity iters: {config.max_iterations} | Harmonics: {config.max_degree}")

    def chebyshev_shift(self, h):
        B, D = h.shape
        h_new = torch.zeros_like(h)
        if D > 1:
            h_new[:, 1] += h[:, 0]
        if D > 2:
            h_new[:, 0:-2] += 0.5 * h[:, 1:-1]
            h_new[:, 2:]   += 0.5 * h[:, 1:-1]
        if D > 1:
            h_new[:, -2] += 0.5 * h[:, -1]
        return h_new

    def set_task(self, idx):
        self.current_task_idx = idx

    def forward(self, idx, targets=None):
        B, T = idx.size()
        device = idx.device

        all_logits = []
        h = torch.zeros(B, self.max_degree, device=device)

        for t in range(T):
            token = idx[:, t]

            # 1. Shift wave
            h = self.chebyshev_shift(h)

            # 2. Add token frequency
            freq_idx = (token + 1).clamp(max=self.max_degree - 1)
            token_wave = torch.zeros(B, self.max_degree, device=device)
            token_wave.scatter_(1, freq_idx.unsqueeze(1), 1.0)
            h = h + token_wave

            # 3. Resonate: iterate the cavity until convergence
            h_res = h
            for i in range(self.max_iters):
                h_next = self.cavity(h_res, self.current_task_idx)
                # Check convergence (stop early if stable)
                diff = (h_next - h_res).norm(dim=-1).mean()
                h_res = h_next
                if diff < self.conv_thresh:
                    break

            # 4. Decode
            logits = self.decoder(self.ln_f(h_res))
            all_logits.append(logits.unsqueeze(1))

        logits = torch.cat(all_logits, dim=1)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss
