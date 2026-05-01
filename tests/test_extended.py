"""
Comprehensive unit tests for the extended experiment.
Covers: model shapes, Chebyshev math, gradient isolation,
data pipeline, training dynamics, numerical stability, freeze logic.
"""
import os, sys, json, math, unittest
import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from model_extended import (
    StandardGPT, StandardGPTConfig,
    DoubleOGPT, DoubleOGPTConfig,
    ChebyshevMLP, RMSNorm, CausalSelfAttention,
)

DEVICE = 'cpu'

def make_config(cls, vocab=54):
    c = cls()
    c.vocab_size = vocab
    return c

# ============================================================
# 1. Forward pass shapes
# ============================================================
class TestForwardShapes(unittest.TestCase):
    def _check(self, model, B=4, T=64):
        x = torch.randint(0, model.config.vocab_size, (B, T))
        y = torch.randint(0, model.config.vocab_size, (B, T))
        logits, loss = model(x, y)
        self.assertEqual(logits.shape, (B, T, model.config.vocab_size))
        self.assertIsNotNone(loss)
        self.assertEqual(loss.dim(), 0)  # scalar

    def test_standard_shapes(self):
        m = StandardGPT(make_config(StandardGPTConfig))
        self._check(m)

    def test_doubleo_shapes(self):
        m = DoubleOGPT(make_config(DoubleOGPTConfig))
        for t in range(4):
            m.set_task(t)
            self._check(m)

    def test_no_target_no_loss(self):
        m = DoubleOGPT(make_config(DoubleOGPTConfig))
        logits, loss = m(torch.randint(0, 54, (2, 32)))
        self.assertIsNone(loss)
        self.assertEqual(logits.shape, (2, 32, 54))

# ============================================================
# 2. Parameter counts & weight tying
# ============================================================
class TestParamCounts(unittest.TestCase):
    def test_weight_tying_standard(self):
        m = StandardGPT(make_config(StandardGPTConfig))
        self.assertIs(m.transformer.wte.weight, m.lm_head.weight)

    def test_weight_tying_doubleo(self):
        m = DoubleOGPT(make_config(DoubleOGPTConfig))
        self.assertIs(m.transformer.wte.weight, m.lm_head.weight)

    def test_nonzero_params(self):
        for cls, cfg_cls in [(StandardGPT, StandardGPTConfig), (DoubleOGPT, DoubleOGPTConfig)]:
            m = cls(make_config(cfg_cls))
            total = sum(p.numel() for p in m.parameters())
            self.assertGreater(total, 100_000)

# ============================================================
# 3. Chebyshev polynomial correctness (real-valued spot checks)
# ============================================================
class TestChebyshevMath(unittest.TestCase):
    """Verify T1-T4 against analytic values using real inputs (imag=0)."""

    def setUp(self):
        cfg = make_config(DoubleOGPTConfig)
        self.mlp = ChebyshevMLP(cfg)
        # Override weights to identity-like for testability
        d = cfg.n_embd
        inner = d * 2
        with torch.no_grad():
            self.mlp.c_fc_real.weight.zero_()
            self.mlp.c_fc_imag.weight.zero_()
            # Map first dim to first dim (scaled by 0.5 to stay in clamp range)
            for i in range(min(d, inner)):
                self.mlp.c_fc_real.weight[i, i] = 0.5
            # imag rail stays zero
            self.mlp.c_proj_real.weight.zero_()
            self.mlp.c_proj_imag.weight.zero_()
            for i in range(min(inner, d)):
                self.mlp.c_proj_real.weight[i, i] = 1.0

    def _get_scalar_output(self, task_idx, input_val):
        """Feed a scalar through dim-0, read dim-0 of output."""
        x = torch.zeros(1, 1, self.mlp.c_fc_real.in_features)
        x[0, 0, 0] = input_val
        out = self.mlp(x, task_idx)
        return out[0, 0, 0].item()

    def test_t1_identity(self):
        # T1(z)=z, with 0.5 scaling from fc, z=0.5*input
        for v in [0.0, 1.0, -1.0, 2.0]:
            out = self._get_scalar_output(0, v)
            expected = 0.5 * v  # fc scales by 0.5, T1 is identity
            self.assertAlmostEqual(out, expected, places=4, msg=f"T1 failed for {v}")

    def test_t2_quadratic(self):
        # T2(z) = 2z^2 - 1, z = 0.5*input
        for v in [0.0, 1.0, -1.0, 2.0]:
            z = 0.5 * v
            z = max(-2.0, min(2.0, z))  # clamp
            expected = 2 * z**2 - 1
            out = self._get_scalar_output(1, v)
            self.assertAlmostEqual(out, expected, places=3, msg=f"T2 failed for {v}")

    def test_t3_cubic(self):
        # T3(z) = 4z^3 - 3z, z = 0.5*input
        for v in [0.0, 1.0, -1.0]:
            z = 0.5 * v
            expected = 4 * z**3 - 3 * z
            out = self._get_scalar_output(2, v)
            self.assertAlmostEqual(out, expected, places=3, msg=f"T3 failed for {v}")

    def test_t4_quartic(self):
        # T4(z) = 8z^4 - 8z^2 + 1, z = 0.5*input
        for v in [0.0, 1.0, -1.0]:
            z = 0.5 * v
            expected = 8 * z**4 - 8 * z**2 + 1
            out = self._get_scalar_output(3, v)
            self.assertAlmostEqual(out, expected, places=3, msg=f"T4 failed for {v}")

    def test_invalid_task_raises(self):
        x = torch.zeros(1, 1, self.mlp.c_fc_real.in_features)
        with self.assertRaises(ValueError):
            self.mlp(x, task_idx=99)

# ============================================================
# 4. Activation clamping
# ============================================================
class TestActivationClamping(unittest.TestCase):
    def test_clamp_prevents_explosion(self):
        cfg = make_config(DoubleOGPTConfig)
        mlp = ChebyshevMLP(cfg)
        # Feed very large input
        x = torch.randn(2, 16, cfg.n_embd) * 100.0
        for t in range(4):
            out = mlp(x, t)
            self.assertTrue(torch.isfinite(out).all(),
                            f"Non-finite output for task {t} with large input")

    def test_zero_input(self):
        cfg = make_config(DoubleOGPTConfig)
        mlp = ChebyshevMLP(cfg)
        x = torch.zeros(1, 1, cfg.n_embd)
        for t in range(4):
            out = mlp(x, t)
            self.assertTrue(torch.isfinite(out).all(),
                            f"Non-finite output for task {t} with zero input")

# ============================================================
# 5. RMSNorm correctness
# ============================================================
class TestRMSNorm(unittest.TestCase):
    def test_output_scale(self):
        norm = RMSNorm(64)
        x = torch.randn(2, 8, 64) * 10
        out = norm(x)
        # RMS of output along last dim should be ~1 (due to learnable weight=1)
        rms = out.pow(2).mean(-1).sqrt()
        self.assertTrue((rms - 1.0).abs().mean() < 0.2)

    def test_preserves_direction(self):
        norm = RMSNorm(32)
        x = torch.randn(1, 1, 32)
        out = norm(x)
        # Cosine similarity should be ~1 (same direction)
        cos = nn.functional.cosine_similarity(x.view(1, -1), out.view(1, -1))
        self.assertGreater(cos.item(), 0.99)

# ============================================================
# 6. Gradient isolation between tasks
# ============================================================
class TestGradientIsolation(unittest.TestCase):
    def test_different_tasks_produce_different_gradients(self):
        """Verify that different task indices produce different gradient vectors."""
        cfg = make_config(DoubleOGPTConfig)
        model = DoubleOGPT(cfg)
        x = torch.randint(0, cfg.vocab_size, (4, 32))
        y = torch.randint(0, cfg.vocab_size, (4, 32))

        grads = []
        for t in range(4):
            model.zero_grad()
            model.set_task(t)
            _, loss = model(x, y)
            loss.backward()
            g = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
            grads.append(g.clone())

        # Each pair of tasks should have different gradients
        for i in range(4):
            for j in range(i+1, 4):
                cos = nn.functional.cosine_similarity(
                    grads[i].unsqueeze(0), grads[j].unsqueeze(0)
                ).item()
                self.assertLess(cos, 0.99,
                    f"Tasks {i} and {j} have nearly identical gradients (cos={cos:.4f})")

    def test_baseline_same_output_regardless_of_task(self):
        """StandardGPT ignores task_idx, so forward outputs must be identical."""
        cfg = make_config(StandardGPTConfig)
        model = StandardGPT(cfg)
        model.eval()
        x = torch.randint(0, cfg.vocab_size, (4, 32))

        outputs = []
        with torch.no_grad():
            for t in range(4):
                model.set_task(t)
                logits, _ = model(x)
                outputs.append(logits.clone())

        for i in range(1, 4):
            diff = (outputs[0] - outputs[i]).abs().max().item()
            self.assertLess(diff, 1e-6,
                f"Baseline task {i} output differs from task 0 (max diff={diff})")

    def test_doubleo_different_output_per_task(self):
        """DoubleOGPT must produce different outputs for different task indices."""
        cfg = make_config(DoubleOGPTConfig)
        model = DoubleOGPT(cfg)
        model.eval()
        x = torch.randint(0, cfg.vocab_size, (4, 32))

        outputs = []
        with torch.no_grad():
            for t in range(4):
                model.set_task(t)
                logits, _ = model(x)
                outputs.append(logits.clone())

        for i in range(4):
            for j in range(i + 1, 4):
                diff = (outputs[i] - outputs[j]).abs().max().item()
                self.assertGreater(diff, 1e-3,
                    f"DoubleO tasks {i} and {j} have identical outputs")

# ============================================================
# 7. Parameter freeze logic
# ============================================================
class TestFreezeLogic(unittest.TestCase):
    def _freeze_non_mlp(self, model):
        for name, param in model.named_parameters():
            if 'mlp' not in name:
                param.requires_grad = False

    def test_freeze_standard(self):
        m = StandardGPT(make_config(StandardGPTConfig))
        total_before = sum(p.numel() for p in m.parameters() if p.requires_grad)
        self._freeze_non_mlp(m)
        total_after = sum(p.numel() for p in m.parameters() if p.requires_grad)
        self.assertGreater(total_before, total_after)
        self.assertGreater(total_after, 0, "No MLP params remain trainable")
        # Only MLP params should be trainable
        for name, p in m.named_parameters():
            if p.requires_grad:
                self.assertIn('mlp', name, f"Non-MLP param {name} still trainable")

    def test_freeze_doubleo(self):
        m = DoubleOGPT(make_config(DoubleOGPTConfig))
        self._freeze_non_mlp(m)
        trainable = [(n, p) for n, p in m.named_parameters() if p.requires_grad]
        self.assertGreater(len(trainable), 0)
        for name, _ in trainable:
            self.assertIn('mlp', name)

    def test_frozen_params_dont_update(self):
        m = DoubleOGPT(make_config(DoubleOGPTConfig))
        self._freeze_non_mlp(m)
        # Snapshot frozen param
        attn_w = m.transformer.h[0].attn.c_attn.weight.clone()
        opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, m.parameters()), lr=1e-2)
        m.set_task(0)
        x = torch.randint(0, 54, (4, 32))
        y = torch.randint(0, 54, (4, 32))
        for _ in range(5):
            opt.zero_grad()
            _, loss = m(x, y)
            loss.backward()
            opt.step()
        diff = (m.transformer.h[0].attn.c_attn.weight - attn_w).abs().max().item()
        self.assertLess(diff, 1e-9, "Frozen attention weights changed during training")

# ============================================================
# 8. Training dynamics - loss decreases
# ============================================================
class TestTrainingDynamics(unittest.TestCase):
    def _train_steps(self, model, steps=50):
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        model.set_task(0)
        x = torch.randint(0, model.config.vocab_size, (8, 32))
        y = torch.randint(0, model.config.vocab_size, (8, 32))
        losses = []
        for _ in range(steps):
            opt.zero_grad()
            _, loss = model(x, y)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        return losses

    def test_standard_loss_decreases(self):
        m = StandardGPT(make_config(StandardGPTConfig))
        losses = self._train_steps(m)
        self.assertLess(losses[-1], losses[0],
            f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}")

    def test_doubleo_loss_decreases(self):
        m = DoubleOGPT(make_config(DoubleOGPTConfig))
        losses = self._train_steps(m)
        self.assertLess(losses[-1], losses[0],
            f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}")

    def test_doubleo_all_tasks_learn(self):
        """Each task polynomial should independently reduce loss."""
        m = DoubleOGPT(make_config(DoubleOGPTConfig))
        for t in range(4):
            # Reset model
            m2 = DoubleOGPT(make_config(DoubleOGPTConfig))
            m2.load_state_dict(m.state_dict())
            m2.set_task(t)
            opt = torch.optim.AdamW(m2.parameters(), lr=1e-3)
            x = torch.randint(0, 54, (8, 32))
            y = torch.randint(0, 54, (8, 32))
            first, last = None, None
            for step in range(30):
                opt.zero_grad()
                _, loss = m2(x, y)
                loss.backward()
                opt.step()
                if step == 0: first = loss.item()
                last = loss.item()
            self.assertLess(last, first,
                f"Task {t}: loss did not decrease ({first:.4f} -> {last:.4f})")

# ============================================================
# 9. Data pipeline
# ============================================================
class TestDataPipeline(unittest.TestCase):
    def setUp(self):
        self.data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'extended')

    def test_datasets_exist(self):
        for name in ["math", "code", "logic", "spell"]:
            path = os.path.join(self.data_dir, f"{name}.jsonl")
            self.assertTrue(os.path.exists(path), f"Missing {path}")

    def test_vocab_exists_and_valid(self):
        vpath = os.path.join(self.data_dir, "vocab.json")
        self.assertTrue(os.path.exists(vpath))
        with open(vpath) as f:
            v = json.load(f)
        self.assertIn('stoi', v)
        self.assertIn('vocab_size', v)
        self.assertGreater(v['vocab_size'], 20)

    def test_all_tokens_in_vocab(self):
        """Every character in every dataset must be in the vocabulary."""
        vpath = os.path.join(self.data_dir, "vocab.json")
        with open(vpath) as f:
            stoi = json.load(f)['stoi']
        for name in ["math", "code", "logic", "spell"]:
            path = os.path.join(self.data_dir, f"{name}.jsonl")
            with open(path) as f:
                for i, line in enumerate(f):
                    if i >= 100: break  # spot check
                    text = json.loads(line)['text']
                    for ch in text:
                        self.assertIn(ch, stoi,
                            f"Char '{ch}' in {name} line {i} not in vocab")

    def test_dataset_class_shapes(self):
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
        from run_extended import TextDataset
        vpath = os.path.join(self.data_dir, "vocab.json")
        with open(vpath) as f:
            stoi = json.load(f)['stoi']
        ds = TextDataset(os.path.join(self.data_dir, "math.jsonl"), stoi, 128)
        self.assertGreater(len(ds), 0)
        x, y = ds[0]
        self.assertEqual(x.shape, (128,))
        self.assertEqual(y.shape, (128,))
        self.assertEqual(x.dtype, torch.long)

# ============================================================
# 10. Numerical stability under stress
# ============================================================
class TestNumericalStability(unittest.TestCase):
    def test_100_forward_backward_no_nan(self):
        """Run 100 train steps and verify no NaN/Inf in params or grads."""
        m = DoubleOGPT(make_config(DoubleOGPTConfig))
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
        for step in range(100):
            t = step % 4
            m.set_task(t)
            x = torch.randint(0, 54, (8, 64))
            y = torch.randint(0, 54, (8, 64))
            opt.zero_grad()
            _, loss = m(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            self.assertTrue(torch.isfinite(loss),
                f"NaN/Inf loss at step {step}, task {t}")
        # Check all params finite
        for n, p in m.named_parameters():
            self.assertTrue(torch.isfinite(p).all(), f"Non-finite param: {n}")

    def test_t4_extreme_values_clamped(self):
        """T4 with large inputs should still produce finite outputs."""
        cfg = make_config(DoubleOGPTConfig)
        mlp = ChebyshevMLP(cfg)
        x = torch.ones(1, 1, cfg.n_embd) * 1000.0
        out = mlp(x, task_idx=3)
        self.assertTrue(torch.isfinite(out).all())

# ============================================================
# 11. Causal attention mask
# ============================================================
class TestCausalMask(unittest.TestCase):
    def test_mask_is_lower_triangular(self):
        cfg = make_config(DoubleOGPTConfig)
        attn = CausalSelfAttention(cfg)
        mask = attn.bias[0, 0]  # (block_size, block_size)
        for i in range(cfg.block_size):
            for j in range(cfg.block_size):
                if j <= i:
                    self.assertEqual(mask[i, j].item(), 1.0)
                else:
                    self.assertEqual(mask[i, j].item(), 0.0)

# ============================================================
# 12. Evaluation helper
# ============================================================
class TestEvaluation(unittest.TestCase):
    def test_evaluate_all_returns_correct_length(self):
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
        from run_extended import evaluate_all, TextDataset
        from torch.utils.data import DataLoader

        data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'extended')
        with open(os.path.join(data_dir, "vocab.json")) as f:
            stoi = json.load(f)['stoi']
            vocab_size = len(stoi)

        # Use small subset
        loaders = []
        for name in ["math", "code", "logic", "spell"]:
            ds = TextDataset(os.path.join(data_dir, f"{name}.jsonl"), stoi, 128)
            loaders.append(DataLoader(ds, batch_size=4, shuffle=False, drop_last=True))

        cfg = make_config(DoubleOGPTConfig, vocab=vocab_size)
        model = DoubleOGPT(cfg)
        losses = evaluate_all(model, loaders, 'cpu')
        self.assertEqual(len(losses), 4)
        for l in losses:
            self.assertGreater(l, 0)
            self.assertTrue(math.isfinite(l))


if __name__ == '__main__':
    unittest.main()
