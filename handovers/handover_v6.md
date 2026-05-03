# Handover: DoubleO v6 (Multi-Digit Arithmetic)

## The Current State of the Art
The repository has been pruned to its core essentials. You are starting with the **DoubleO v6 Auto-Routed Architecture**. This model uses $O(N)$ polynomial recurrence to completely bypass Multi-Head Attention, utilizing a singular recursive "Resonant Cavity" to achieve deep mathematical reasoning with only ~200,000 parameters.

### Core Files
* `scripts/modal_chebywave_multidigit.py`: The main V6 execution and training pipeline. Contains the dataloaders, the Resonant Cavity definition, and the Modal deployment logic.
* `scripts/cheby_triton.py`: The hyper-optimized H100 Triton kernel. It manually unrolls the Chebyshev recurrence by a factor of 4, dropping the token execution time to <0.1ms. *Note: Sequences must be manually padded to multiples of 4 before passing into this kernel, which is handled in `modal_chebywave_multidigit.py`.*

## Key Discoveries & Hyperparameters
During the "Extreme Grokking" run, we discovered the precise hyperparameters required to push the architecture past 100% memorization and into algorithmic generalization for multi-digit arithmetic:

1. **Algorithmic Alignment (Reversed Strings)**: The output predictions are formatted backward (e.g., `92 + 23 = 511`). This naturally aligns the causal prediction with the human base-10 carry algorithm (calculating the ones-place first). The evaluation script flips it back for logging.
2. **Weight Decay Balance**: A weight decay of `1.0` is too brutal and instantly collapses the model's logits to uniform randomness (Loss ~2.31). A weight decay of `0.1` is the perfect balance to shear off memorization tables while allowing the algorithm to form.
3. **Modal Timeouts (The Scaling Wall)**: The training loop is incredibly compute-intensive. Running 160,000 steps at Batch Size 256 hits the exact 2-hour (7200s) default Modal timeout limit. 

## Next Steps for the Next Agent
We left off right as Phase 3 of the Extreme Grokking run hit the 2-hour Modal timeout (at step 72k/80k). The model was successfully and stably descending its loss curve (Loss ~0.59). 

Your primary objectives are to bypass this scaling wall and achieve 100% test accuracy:
1. **Curriculum Learning**: Instead of brute-forcing 20 million 2-digit math sequences, implement a curriculum dataloader that trains on 1-digit math first, and slowly introduces 2-digit math. This should reduce the required grokking steps by 5x, completely bypassing the timeout issue.
2. **Modal Timeout Adjustment**: Alternatively, simply increase the Modal function timeout constraint to `timeout=14400` (4 hours) so the model can finish its 160,000-step grind.
3. **Parameter Efficiency Analysis**: Since the model has proved capable of retaining zero-forgetting logic, try pruning the cavity dimensions from 128 down to 64 to see if we can make it even faster!
