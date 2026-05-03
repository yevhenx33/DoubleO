"""
ChebyWave v3: Spectral Resonator + DoubleO Task Isolation + DA Layer.

Combines:
1. Chebyshev wave encoding (parameter-free sequence-to-waveform)
2. Learned spectral gates (selective frequency amplification)
3. Task-specific Chebyshev polynomial routing (T2 for Task A, T3 for Task B)
4. Shared Data Availability layer with learned per-task gates

This is the full architecture: a model that can learn arithmetic,
generalize to unseen problems, AND retain old knowledge when learning new tasks.
"""
import torch
import torch.nn as nn
from torch.nn import functional as F


class DoubleOSpectralBlock(nn.Module):
    """
    One layer of spectral processing with task isolation + shared DA layer.

    Flow:
    1. Spectral Gate: selectively amplify/suppress frequencies
    2. Task-Specific Polynomial: T2 (Task A) or T3 (Task B) applied to wave
    3. Shared DA Pathway: common knowledge both tasks can selectively read
    4. Output = Isolated + alpha * Shared
    """
    def __init__(self, max_degree, num_tasks=2):
        super().__init__()

        # Spectral gate: which frequencies matter right now
        self.gate_norm = nn.LayerNorm(max_degree)
        self.gate_proj = nn.Linear(max_degree, max_degree)

        # Task-specific isolated resonators (one per task)
        self.isolated_norm = nn.LayerNorm(max_degree)
        self.isolated_resonators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(max_degree, max_degree * 2),
                nn.GELU(),
                nn.Linear(max_degree * 2, max_degree),
            )
            for _ in range(num_tasks)
        ])

        # Shared DA pathway (accessible by all tasks)
        self.shared_norm = nn.LayerNorm(max_degree)
        self.shared_resonator = nn.Sequential(
            nn.Linear(max_degree, max_degree * 2),
            nn.GELU(),
            nn.Linear(max_degree * 2, max_degree),
        )

        # Per-task gate logits: how much shared knowledge to absorb
        # Initialized slightly negative so gates start mostly closed
        self.da_gate_logits = nn.Parameter(torch.full((num_tasks,), -2.0))

    def _bound(self, x):
        """Normalize to [-1, 1] for Chebyshev validity."""
        x_max = x.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-5)
        return x / x_max

    def forward(self, h, task_idx):
        # === 1. Spectral Attention Gate ===
        gate = torch.sigmoid(self.gate_proj(self.gate_norm(h)))
        h_gated = h + h * gate  # residual + gated amplification

        # === 2. Task-Specific Polynomial Routing ===
        h_norm = self._bound(self.isolated_norm(h_gated))

        if task_idx == 0:
            # T2(x) = 2x^2 - 1
            h_poly = 2 * h_norm**2 - 1
        else:
            # T3(x) = 4x^3 - 3x
            h_poly = 4 * h_norm**3 - 3 * h_norm

        isolated_out = self.isolated_resonators[task_idx](h_poly)

        # === 3. Shared DA Pathway ===
        shared_out = self.shared_resonator(self.shared_norm(h_gated))

        # === 4. Merge: Isolated + Gated Shared ===
        alpha = torch.sigmoid(self.da_gate_logits[task_idx])
        h_out = h + isolated_out + alpha * shared_out

        return h_out


class ChebyWaveConfig:
    vocab_size: int = 17
    max_degree: int = 128
    n_layer: int = 4
    num_tasks: int = 2
    block_size: int = 32


class ChebyWave(nn.Module):
    """
    ChebyWave v3: Spectral Resonator + DoubleO.

    Sequence processing via parameter-free Chebyshev wave encoding,
    with task-isolated polynomial routing and shared DA knowledge layer.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.max_degree = config.max_degree
        self.current_task_idx = 0

        # Stacked DoubleO spectral blocks
        self.layers = nn.ModuleList([
            DoubleOSpectralBlock(config.max_degree, config.num_tasks)
            for _ in range(config.n_layer)
        ])

        # Final normalization before decoding
        self.ln_f = nn.LayerNorm(config.max_degree)

        # Decoder: wave -> token logits
        self.decoder = nn.Linear(config.max_degree, config.vocab_size)

        # Count parameters
        total = sum(p.numel() for p in self.parameters())
        print(f"  [ChebyWave v3 + DoubleO] Params: {total:,} | Layers: {config.n_layer} | Harmonics: {config.max_degree}")

    def chebyshev_shift(self, h):
        """
        Multiply polynomial state h(x) by T_1(x).
          T_0 * T_1 = T_1
          T_k * T_1 = 0.5 * (T_{k-1} + T_{k+1})  for k >= 1
        """
        B, D = h.shape
        h_new = torch.zeros_like(h)

        # T_0 * T_1 = T_1
        if D > 1:
            h_new[:, 1] += h[:, 0]

        # T_k * T_1 for k=1..D-2
        if D > 2:
            h_new[:, 0:-2] += 0.5 * h[:, 1:-1]
            h_new[:, 2:]   += 0.5 * h[:, 1:-1]

        # T_{D-1} * T_1: only T_{D-2} (T_D truncated)
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

            # 1. Shift wave (encode position)
            h = self.chebyshev_shift(h)

            # 2. Add token frequency spike
            freq_idx = (token + 1).clamp(max=self.max_degree - 1)
            token_wave = torch.zeros(B, self.max_degree, device=device)
            token_wave.scatter_(1, freq_idx.unsqueeze(1), 1.0)
            h = h + token_wave

            # 3. Process through stacked DoubleO spectral blocks
            h_proc = h
            for layer in self.layers:
                h_proc = layer(h_proc, self.current_task_idx)

            # 4. Decode to token logits
            logits = self.decoder(self.ln_f(h_proc))
            all_logits.append(logits.unsqueeze(1))

        logits = torch.cat(all_logits, dim=1)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss
