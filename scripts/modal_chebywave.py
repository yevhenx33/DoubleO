
"""
Modal GPU Runner: ChebyWave v3 + v4 on T4 GPU.

Runs both the DoubleO Spectral Resonator (v3, feedforward) and the
Resonant Cavity (v4, iterative) back-to-back on GPU for fast evaluation.
"""
import modal

app = modal.App("chebywave-eval")

image = modal.Image.debian_slim(python_version="3.12").pip_install("torch")


@app.function(image=image, gpu="H100", timeout=5400)
def run_chebywave_eval():
    import torch
    import torch.nn as nn
    from torch.nn import functional as F
    from torch.utils.data import Dataset, DataLoader
    import random
    import json
    import time

    device = 'cuda'
    print(f"Device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Vocabulary ──
    CHARS = list("\n 0123456789+-*=")
    STOI = {ch: i for i, ch in enumerate(CHARS)}
    STOI["<|endoftext|>"] = len(CHARS)
    ITOS = {i: ch for ch, i in STOI.items()}
    VOCAB_SIZE = len(STOI)

    def encode(text):
        return [STOI.get(c, 0) for c in text]

    def decode(tokens):
        return "".join(ITOS.get(t, "?") for t in tokens)

    # ── Data Generation ──
    def generate_addition_problems(n, max_digits=1, seed=42):
        rng = random.Random(seed)
        problems, seen = [], set()
        max_val = 10**max_digits - 1
        while len(problems) < n:
            a, b = rng.randint(0, max_val), rng.randint(0, max_val)
            if (a, b) in seen: continue
            seen.add((a, b))
            c = a + b
            problems.append({"prompt": f"{a} + {b} = ", "answer": str(c), "full": f"{a} + {b} = {c}"})
        return problems

    def generate_subtraction_problems(n, max_digits=1, seed=43):
        rng = random.Random(seed)
        problems, seen = [], set()
        max_val = 10**max_digits - 1
        while len(problems) < n:
            a = rng.randint(0, max_val)
            b = rng.randint(0, a)
            if (a, b) in seen: continue
            seen.add((a, b))
            c = a - b
            problems.append({"prompt": f"{a} - {b} = ", "answer": str(c), "full": f"{a} - {b} = {c}"})
        return problems

    # ── Dataset ──
    class ArithmeticDataset(Dataset):
        def __init__(self, problems, block_size=32):
            self.data = []
            for p in problems:
                tokens = encode(p["full"])
                tokens.append(STOI["<|endoftext|>"])
                self.data.extend(tokens)
            self.data = torch.tensor(self.data, dtype=torch.long)
            self.block_size = block_size
        def __len__(self):
            return max(1, len(self.data) - self.block_size)
        def __getitem__(self, idx):
            x = self.data[idx:idx+self.block_size]
            y = self.data[idx+1:idx+self.block_size+1]
            return x, y

    # ── Generation ──
    @torch.no_grad()
    def generate(model, prompt_tokens, max_new_tokens=10):
        model.eval()
        tokens = prompt_tokens.clone().unsqueeze(0).to(device)
        for _ in range(max_new_tokens):
            block_size = model.config.block_size
            tokens_cond = tokens[:, -block_size:]
            logits, _ = model(tokens_cond)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            tokens = torch.cat([tokens, next_token], dim=1)
            if next_token.item() == STOI["<|endoftext|>"] or next_token.item() == STOI["\n"]:
                break
        return tokens[0]

    # ── Accuracy Eval ──
    def evaluate_accuracy(model, problems, task_idx, label=""):
        model.eval()
        model.set_task(task_idx)
        correct, total = 0, len(problems)
        examples = []
        for p in problems:
            prompt_tokens = torch.tensor(encode(p["prompt"]), dtype=torch.long)
            generated = generate(model, prompt_tokens, max_new_tokens=8)
            generated_text = decode(generated.tolist())
            if "= " in generated_text:
                predicted = generated_text.split("= ", 1)[1].strip()
                predicted = predicted.replace("<|endoftext|>", "").replace("?", "").replace("\n", "").strip()
            else:
                predicted = generated_text.strip().replace("<|endoftext|>", "").replace("?", "").replace("\n", "").strip()
            is_correct = predicted == p["answer"]
            if is_correct: correct += 1
            if len(examples) < 5:
                examples.append({"prompt": p["prompt"], "expected": p["answer"], "predicted": predicted, "correct": is_correct})
        accuracy = correct / total * 100.0 if total > 0 else 0.0
        print(f"\n  [{label}] Accuracy: {correct}/{total} = {accuracy:.1f}%")
        for ex in examples:
            mark = "✓" if ex["correct"] else "✗"
            print(f"    {mark} {ex['prompt']}{ex['expected']}  →  model: '{ex['predicted']}'")
        return accuracy

    # ══════════════════════════════════════════════════════════════
    # MODEL DEFINITIONS (self-contained)
    # ══════════════════════════════════════════════════════════════

    # ── v3: DoubleO Spectral Block (feedforward) ──
    class DoubleOSpectralBlock(nn.Module):
        def __init__(self, max_degree, num_tasks=2):
            super().__init__()
            self.gate_norm = nn.LayerNorm(max_degree)
            self.gate_proj = nn.Linear(max_degree, max_degree)
            self.iso_norm = nn.LayerNorm(max_degree)
            self.iso_resonators = nn.ModuleList([
                nn.Sequential(nn.Linear(max_degree, max_degree*2), nn.GELU(), nn.Linear(max_degree*2, max_degree))
                for _ in range(num_tasks)
            ])
            self.shared_norm = nn.LayerNorm(max_degree)
            self.shared_resonator = nn.Sequential(nn.Linear(max_degree, max_degree*2), nn.GELU(), nn.Linear(max_degree*2, max_degree))
            self.da_gate_logits = nn.Parameter(torch.full((num_tasks,), -2.0))
        def _bound(self, x):
            return x / x.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-5)
        def forward(self, h, task_idx):
            gate = torch.sigmoid(self.gate_proj(self.gate_norm(h)))
            h_gated = h + h * gate
            h_norm = self._bound(self.iso_norm(h_gated))
            if task_idx == 0:
                h_poly = 2 * h_norm**2 - 1
            else:
                h_poly = 4 * h_norm**3 - 3 * h_norm
            isolated = self.iso_resonators[task_idx](h_poly)
            shared = self.shared_resonator(self.shared_norm(h_gated))
            alpha = torch.sigmoid(self.da_gate_logits[task_idx])
            return h + isolated + alpha * shared

    # ── Auto-Routed Resonant Cavity ──
    class AutoRoutedCavity(nn.Module):
        """
        Resonant cavity with learned per-token routing.
        Computes BOTH T2 and T3 paths, then blends based on
        a router that reads the wave state.
        No manual set_task() needed.
        """
        def __init__(self, max_degree, num_tasks=2):
            super().__init__()
            self.num_tasks = num_tasks

            # Spectral gate
            self.gate_norm = nn.LayerNorm(max_degree)
            self.gate_proj = nn.Linear(max_degree, max_degree)

            # Task-specific resonators
            self.iso_norm = nn.LayerNorm(max_degree)
            self.iso_resonators = nn.ModuleList([
                nn.Sequential(nn.Linear(max_degree, max_degree*2), nn.GELU(), nn.Linear(max_degree*2, max_degree))
                for _ in range(num_tasks)
            ])

            # Shared DA pathway
            self.shared_norm = nn.LayerNorm(max_degree)
            self.shared_resonator = nn.Sequential(nn.Linear(max_degree, max_degree*2), nn.GELU(), nn.Linear(max_degree*2, max_degree))
            self.da_gate_logits = nn.Parameter(torch.full((num_tasks,), -2.0))

            # Damping
            self.damping_logits = nn.Parameter(torch.full((max_degree,), 2.0))

            # === THE ROUTER ===
            # Reads wave state → outputs routing weight w ∈ [0,1]
            # w≈1 → T2 (addition), w≈0 → T3 (subtraction)
            self.router = nn.Sequential(
                nn.LayerNorm(max_degree),
                nn.Linear(max_degree, 1),
            )

        def _bound(self, x):
            return x / x.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-5)

        def forward(self, h):
            # Damping
            damping = torch.sigmoid(self.damping_logits)
            h = h * damping

            # Spectral gate
            gate = torch.sigmoid(self.gate_proj(self.gate_norm(h)))
            h_gated = h + h * gate

            # Normalize for polynomial input
            h_norm = self._bound(self.iso_norm(h_gated))

            # Compute BOTH polynomial paths
            h_t2 = 2 * h_norm**2 - 1           # T2
            h_t3 = 4 * h_norm**3 - 3 * h_norm  # T3

            out_t2 = self.iso_resonators[0](h_t2)
            out_t3 = self.iso_resonators[1](h_t3)

            # Router decides blend weight per sample
            w = torch.sigmoid(self.router(h_gated))  # (B, 1)

            # Blend isolated outputs
            isolated = w * out_t2 + (1 - w) * out_t3

            # Shared DA (blended gate)
            shared = self.shared_resonator(self.shared_norm(h_gated))
            alpha_0 = torch.sigmoid(self.da_gate_logits[0])
            alpha_1 = torch.sigmoid(self.da_gate_logits[1])
            alpha = w * alpha_0 + (1 - w) * alpha_1

            return h + isolated + alpha * shared, w

    # ── Config ──
    class ChebyConfig:
        vocab_size = VOCAB_SIZE
        max_degree = 128
        n_layer = 4
        num_tasks = 2
        block_size = 32
        max_iterations = 8
        convergence_threshold = 1e-3

    # ── Auto-Routed ChebyWave model ──
    class ChebyWaveModel(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.max_degree = config.max_degree

            self.cavity = AutoRoutedCavity(config.max_degree, config.num_tasks)
            self.max_iters = config.max_iterations
            self.conv_thresh = config.convergence_threshold

            self.ln_f = nn.LayerNorm(config.max_degree)
            # Task-specific decoders
            self.decoders = nn.ModuleList([
                nn.Linear(config.max_degree, config.vocab_size)
                for _ in range(config.num_tasks)
            ])
            total = sum(p.numel() for p in self.parameters())
            print(f"  [v6 auto-routed] Params: {total:,}")

        def chebyshev_shift(self, h):
            B, D = h.shape
            h_new = torch.zeros_like(h)
            if D > 1: h_new[:, 1] += h[:, 0]
            if D > 2:
                h_new[:, 0:-2] += 0.5 * h[:, 1:-1]
                h_new[:, 2:]   += 0.5 * h[:, 1:-1]
            if D > 1: h_new[:, -2] += 0.5 * h[:, -1]
            return h_new

        def set_task(self, idx): pass  # no-op, kept for API compat

        def forward(self, idx, targets=None):
            B, T = idx.size()
            all_logits = []
            h = torch.zeros(B, self.max_degree, device=idx.device)
            for t in range(T):
                token = idx[:, t]
                h = self.chebyshev_shift(h)
                freq_idx = (token + 1).clamp(max=self.max_degree - 1)
                token_wave = torch.zeros(B, self.max_degree, device=idx.device)
                token_wave.scatter_(1, freq_idx.unsqueeze(1), 1.0)
                h = h + token_wave

                # Resonate with auto-routing
                h_proc = h
                w_final = None
                for i in range(self.max_iters):
                    h_next, w = self.cavity(h_proc)
                    w_final = w
                    diff = (h_next - h_proc).norm(dim=-1).mean()
                    h_proc = h_next
                    if diff < self.conv_thresh: break

                # Soft-decode through blended decoders
                h_normed = self.ln_f(h_proc)
                logits_0 = self.decoders[0](h_normed)  # addition decoder
                logits_1 = self.decoders[1](h_normed)  # subtraction decoder
                logits = w_final * logits_0 + (1 - w_final) * logits_1

                all_logits.append(logits.unsqueeze(1))
            logits = torch.cat(all_logits, dim=1)
            loss = None
            if targets is not None:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss

    # ══════════════════════════════════════════════════════════════
    # GAUNTLET
    # ══════════════════════════════════════════════════════════════

    def run_gauntlet(model_name, model):
        print(f"\n{'='*60}")
        print(f"ACCURACY GAUNTLET: {model_name}")
        print(f"{'='*60}")

        add_train = generate_addition_problems(80, max_digits=1, seed=42)
        add_test  = generate_addition_problems(20, max_digits=1, seed=999)
        sub_train = generate_subtraction_problems(45, max_digits=1, seed=43)
        sub_test  = generate_subtraction_problems(10, max_digits=1, seed=998)

        add_dataset = ArithmeticDataset(add_train, block_size=model.config.block_size)
        sub_dataset = ArithmeticDataset(sub_train, block_size=model.config.block_size)
        add_loader = DataLoader(add_dataset, batch_size=64, shuffle=True)
        sub_loader = DataLoader(sub_dataset, batch_size=64, shuffle=True)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        # Phase 1: Train Addition (NO set_task — router learns from data)
        print("\n--- Phase 1: Training on Addition (Task A) ---")
        model.train()
        t0 = time.time()
        add_iter = iter(add_loader)
        for step in range(2000):
            try: x, y = next(add_iter)
            except StopIteration: add_iter = iter(add_loader); x, y = next(add_iter)
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            _, loss = model(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if step % 500 == 0:
                print(f"  Step {step}, loss: {loss.item():.4f}")
        print(f"  Phase 1 took {time.time()-t0:.1f}s")

        # Phase 2: Eval Addition (task_idx ignored — auto-routed)
        print("\n--- Phase 2: Addition Accuracy (before Task B) ---")
        add_train_acc = evaluate_accuracy(model, add_train, 0, "Addition TRAIN (all)")
        add_test_acc  = evaluate_accuracy(model, add_test, 0, "Addition TEST (all)")

        # Phase 3: Cooperative Training (mixed add+sub, auto-routed)
        print("\n--- Phase 3: Cooperative Training (A+B mixed, auto-routed) ---")
        model.train()
        t0 = time.time()
        add_iter2 = iter(add_loader)
        sub_iter = iter(sub_loader)
        for step in range(1500):
            if step % 2 == 0:
                try: x, y = next(sub_iter)
                except StopIteration: sub_iter = iter(sub_loader); x, y = next(sub_iter)
            else:
                try: x, y = next(add_iter2)
                except StopIteration: add_iter2 = iter(add_loader); x, y = next(add_iter2)
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            _, loss = model(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if step % 400 == 0:
                print(f"  Step {step}, loss: {loss.item():.4f}")
        print(f"  Phase 3 took {time.time()-t0:.1f}s")

        # Phase 4: Eval Both (auto-routed — model figures it out)
        print("\n--- Phase 4: Final Accuracy (after Task B) ---")
        add_train_after = evaluate_accuracy(model, add_train, 0, "Addition TRAIN (retention)")
        add_test_after  = evaluate_accuracy(model, add_test, 0, "Addition TEST (retention)")
        sub_train_acc   = evaluate_accuracy(model, sub_train, 0, "Subtraction TRAIN")
        sub_test_acc    = evaluate_accuracy(model, sub_test, 0, "Subtraction TEST")

        results = {
            "add_train_before": add_train_acc, "add_test_before": add_test_acc,
            "add_train_after": add_train_after, "add_test_after": add_test_after,
            "sub_train": sub_train_acc, "sub_test": sub_test_acc,
            "retention": add_test_after - add_test_acc,
        }
        print(f"\n{'='*60}")
        print(f"SUMMARY: {model_name}")
        print(f"{'='*60}")
        print(f"  Addition (before):  Train={add_train_acc:.1f}%  Test={add_test_acc:.1f}%")
        print(f"  Addition (after):   Train={add_train_after:.1f}%  Test={add_test_after:.1f}%")
        print(f"  Subtraction:        Train={sub_train_acc:.1f}%  Test={sub_test_acc:.1f}%")
        print(f"  Retention Delta:    {results['retention']:+.1f}%")
        return results

    # ── v6: Auto-Routed Resonant Cavity ──
    config = ChebyConfig()
    all_results = {}

    model_v6 = ChebyWaveModel(config).to(device)
    all_results["v6_AutoRouted"] = run_gauntlet("ChebyWave v6 (Auto-Routed Resonant Cavity)", model_v6)

    print("\n\n" + "="*60)
    print("FINAL COMPARISON")
    print("="*60)
    for name, res in all_results.items():
        print(f"\n{name}:")
        print(f"  Add before: {res['add_train_before']:.1f}% / {res['add_test_before']:.1f}%")
        print(f"  Add after:  {res['add_train_after']:.1f}% / {res['add_test_after']:.1f}%")
        print(f"  Sub:        {res['sub_train']:.1f}% / {res['sub_test']:.1f}%")
        print(f"  Retention:  {res['retention']:+.1f}%")

    return all_results


@app.local_entrypoint()
def main():
    import json
    results = run_chebywave_eval.remote()
    with open("/home/ubuntu/DoubleO/data/modal_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("\nSaved to data/modal_results.json")
