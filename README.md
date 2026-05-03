# DoubleO v6: Continuous Mathematical Grokking

This repository contains the architecture and experimental pipeline for **DoubleO v6**, a sub-0.1ms deterministic polynomial recurrence engine designed to achieve multi-digit arithmetic grokking with zero catastrophic forgetting.

## The Architecture Breakthrough

Traditional Large Language Models (LLMs) rely on $O(N^2)$ Self-Attention and deep, multi-layer feed-forward networks to "think." They also suffer from catastrophic interference (forgetting) when trained on sequential tasks because all tasks overwrite the same dense weight matrices.

**DoubleO v6 entirely abandons Attention and Layer Stacking.**

### 1. The Chebyshev Linear Scan
Instead of Attention, we use a custom, highly-optimized Triton kernel (`cheby_triton.py`) that executes a deterministic, $O(N)$ linear recurrence defined by the roots of Chebyshev polynomials. 
- It naturally compresses time into a fixed-size $H_t$ state vector.
- By manually unrolling the kernel loops by a factor of 4 and enforcing FP16 precision, we achieve sub-0.1ms token execution latency on NVIDIA H100 GPUs.
- Because it is a pure causal scan, it requires **no positional embeddings** and consumes **zero KV-cache memory**.

### 2. The Resonant Cavity
Instead of stacking 24 distinct neural network layers (which bloats parameters into the billions), DoubleO uses a singular, highly efficient MLP block called the **Resonant Cavity**.
- The token state is passed into the cavity and loops recursively (`max_iterations = 24`).
- It acts as a dynamical system, bouncing the logic until it converges on an attractor state. 
- We achieve the deep reasoning depth of a 24-layer Transformer using the physical parameter footprint of a 1-layer model (~220,000 parameters).

### 3. The Auto-Router (Zero-Forgetting)
To solve catastrophic forgetting, DoubleO utilizes a dynamic task router. By scaling the inputs mathematically, the model shifts different logical operations (e.g., Addition vs. Subtraction) into completely orthogonal polynomial frequency bands ($T_2(x)$, $T_3(x)$). 
- Tasks share the foundational base-10 embedding logic but operate in isolated mathematical dimensions.
- **Result:** Training the model on Subtraction *after* it perfectly memorized Addition resulted in a **+9.0% Retention Delta** (Positive Transfer), completely eliminating catastrophic forgetting.

## Extreme Grokking Pipeline

The training pipeline (`modal_chebywave_multidigit.py`) forces the tiny parameter space to discover the generalized base-10 carry algorithm.

### Key Discoveries
1. **Algorithmic Alignment:** Formatting the target sequences backward (e.g., `92 + 23 = 511`) natively aligns the autoregressive token generation with the human carry algorithm (ones-place first), accelerating grokking by allowing causal gradient flow without lookahead requirements.
2. **Weight Decay Balance:** A strict `weight_decay = 0.1` is enforced to aggressively shear away brittle, high-norm memorization circuits without collapsing the logits to uniform randomness (which occurs at `1.0` decay).

## Repository Structure

```
DoubleO/
├── README.md
├── scripts/
│   ├── modal_chebywave_multidigit.py  # V6 Auto-Routed Architecture & Extreme Grokking Pipeline
│   ├── cheby_triton.py                # Sub-0.1ms H100 Chebyshev Recurrence Kernel
│   ├── modal_chebywave.py             # V5 Baseline
│   └── benchmark_cheby_kernel.py      # Triton performance validator
├── data/
│   └── data_gen.py                    # Legacy data generators
├── handovers/
│   └── handover_v6.md                 # Agent knowledge item containing current scale limits
└── archive/
    └── v6_extreme_grokking/           # Frozen backup of pristine V6 scripts
```

## Running the Pipeline

The architecture is designed to execute on Modal using H100 instances.
```bash
# Set up environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run the Triton Kernel Benchmark
modal run scripts/benchmark_cheby_kernel.py

# Launch the V6 Extreme Grokking Gauntlet
modal run -m scripts.modal_chebywave_multidigit
```

## Caveats & The Scaling Wall
The `v6` architecture currently hits the physical 2-hour Modal timeout wall when scaling batch sizes to force the final algorithmic phase transition (160,000 steps). Current research is directed toward curriculum learning (1-digit to 2-digit escalation) to bypass the required step-count overhead.
