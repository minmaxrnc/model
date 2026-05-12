# SPDX-FileCopyrightText: 2026 Alessandro Ronca
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""
Test suite for the MinMax RNC model.

Organisation
------------
TestInitialisers        — small_init, wang_init
TestFeedForwardConfig   — proj_up_dim computation, validation
TestFeedForwardModules  — shape checks, activations, factory
TestMinMaxOperator      — mathematical correctness of apply / compose
TestMinMaxScan          — parallel prefix scan vs sequential reference
TestMinMaxRNCConfig     — flat config, layer_cfg property, presets
TestMinMaxRNCForward    — shapes, state, continuity, chunking, edge cases
TestMinMaxRNCOptions    — smoke test for every config knob
TestMinMaxRNCLM         — LM wrapper shapes, weight tying, continuity
"""

import math
import unittest

import torch
import torch.nn as nn

from minmaxrnc.minmax_rnc      import MinMaxRNC, MinMaxRNCConfig
from minmaxrnc.minmax_rnc_lm   import MinMaxRNC_LM, MinMaxRNCLMConfig
from minmaxrnc.minmax_scan      import all_states
from minmaxrnc.minmax_operator  import apply as mm_apply, compose as mm_compose
from minmaxrnc.modules.initialisers import small_init_init_, wang_init_
from minmaxrnc.modules.feedforward  import (
    FeedForwardConfig, FeedForward, GatedFeedForward, create_feedforward,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

B, T, D = 2, 16, 32
SMALL_CFG = MinMaxRNCConfig(d_model=D, n_layers=2, d_state=8)


# ---------------------------------------------------------------------------
# Initialisers
# ---------------------------------------------------------------------------

class TestInitialisers(unittest.TestCase):

    def _param(self, *shape):
        return nn.Parameter(torch.empty(*shape))

    def test_small_init_std(self):
        """Empirical std should match sqrt(2 / (5 * dim)) within 5%."""
        dim = 512
        p = self._param(2000, dim)
        small_init_init_(p, dim=dim)
        expected = math.sqrt(2.0 / (5.0 * dim))
        self.assertAlmostEqual(p.std().item(), expected, delta=expected * 0.05)

    def test_wang_init_std(self):
        """Empirical std should match 2 / (N * sqrt(dim)) within 5%."""
        dim, n = 256, 8
        p = self._param(2000, dim)
        wang_init_(p, dim=dim, num_blocks=n)
        expected = 2.0 / (n * math.sqrt(dim))
        self.assertAlmostEqual(p.std().item(), expected, delta=expected * 0.05)

    def test_wang_init_depth_scaling(self):
        """Doubling num_blocks should halve the std."""
        dim = 256
        p1 = self._param(5000, dim)
        p2 = self._param(5000, dim)
        wang_init_(p1, dim=dim, num_blocks=4)
        wang_init_(p2, dim=dim, num_blocks=8)
        self.assertAlmostEqual(p1.std().item() / p2.std().item(), 2.0, delta=0.1)


# ---------------------------------------------------------------------------
# FeedForwardConfig
# ---------------------------------------------------------------------------

class TestFeedForwardConfig(unittest.TestCase):

    def _cfg(self, **kw):
        base = dict(_num_blocks=4, embedding_dim=64, proj_factor=1.3,
                    round_proj_up_to_multiple_of=2)
        base.update(kw)
        return FeedForwardConfig(**base)

    def test_proj_up_dim_rounds_up_by_default(self):
        # 64 * 1.3 = 83.2 → ceil(83.2 / 2) * 2 = 84
        self.assertEqual(self._cfg()._proj_up_dim, 84)

    def test_proj_up_dim_rounds_down(self):
        # floor(83.2 / 2) * 2 = 82
        self.assertEqual(self._cfg(round_proj_up_dim_up=False)._proj_up_dim, 82)

    def test_proj_up_dim_zero_before_embedding_set(self):
        # embedding_dim defaults to -1, so proj_up_dim cannot be computed yet
        cfg = FeedForwardConfig(_num_blocks=4)
        self.assertEqual(cfg._proj_up_dim, 0)

    def test_invalid_activation_raises(self):
        with self.assertRaises(ValueError):
            FeedForwardConfig(_num_blocks=4, embedding_dim=64, act_fn='tanh')


# ---------------------------------------------------------------------------
# FeedForward modules
# ---------------------------------------------------------------------------

class TestFeedForwardModules(unittest.TestCase):

    def _cfg(self, ffn_type='basic', d=64, **kw):
        return FeedForwardConfig(
            _num_blocks=4, embedding_dim=d, embedding_dim_out=d,
            proj_factor=1.3, ffn_type=ffn_type, **kw,
        )

    def _x(self, d=64):
        return torch.randn(2, 8, d)

    def test_basic_output_shape(self):
        out = FeedForward(self._cfg('basic'))(self._x())
        self.assertEqual(out.shape, (2, 8, 64))

    def test_gated_output_shape(self):
        out = GatedFeedForward(self._cfg('gated'))(self._x())
        self.assertEqual(out.shape, (2, 8, 64))

    def test_asymmetric_output_dim(self):
        cfg = FeedForwardConfig(_num_blocks=4, embedding_dim=64, embedding_dim_out=32,
                                proj_factor=1.3, ffn_type='basic')
        out = FeedForward(cfg)(self._x())
        self.assertEqual(out.shape, (2, 8, 32))

    def test_all_activations_run(self):
        for act in ('relu', 'relu^2', 'gelu', 'swish', 'sigmoid', 'selu'):
            with self.subTest(act=act):
                cfg = self._cfg('gated', act_fn=act)
                out = GatedFeedForward(cfg)(self._x())
                self.assertEqual(out.shape, (2, 8, 64))

    def test_scaled_init_runs(self):
        cfg = self._cfg('gated', init='scaled')
        out = GatedFeedForward(cfg)(self._x())
        self.assertEqual(out.shape, (2, 8, 64))

    def test_factory_creates_basic(self):
        self.assertIsInstance(create_feedforward(self._cfg('basic')), FeedForward)

    def test_factory_creates_gated(self):
        self.assertIsInstance(create_feedforward(self._cfg('gated')), GatedFeedForward)


# ---------------------------------------------------------------------------
# MinMax operator
# ---------------------------------------------------------------------------

class TestMinMaxOperator(unittest.TestCase):
    """
    The MinMax operator is f(x) = max(min(a, x), b).

    Clamp regions
    -------------
    x < b        →  f(x) = b          (lower clamp)
    b ≤ x ≤ a   →  f(x) = x          (identity window)
    x > a        →  f(x) = a          (upper clamp)

    Composition closure
    -------------------
    f2(f1(x)) is also a MinMax operator with
        a_c = min(a2, a1)
        b_c = max(min(a2, b1), b2)
    """

    # a = 3, b = 1 → f clips x into [1, 3]
    A = torch.tensor(3.0)
    B = torch.tensor(1.0)

    def test_apply_lower_clamp(self):
        out = mm_apply(self.A, self.B, torch.tensor(0.0))
        torch.testing.assert_close(out, self.B)

    def test_apply_upper_clamp(self):
        out = mm_apply(self.A, self.B, torch.tensor(5.0))
        torch.testing.assert_close(out, self.A)

    def test_apply_identity_window(self):
        x = torch.tensor(2.0)
        out = mm_apply(self.A, self.B, x)
        torch.testing.assert_close(out, x)

    def test_compose_matches_sequential(self):
        """Composing (a2,b2) with (a1,b1) must equal applying f1 then f2."""
        a1, b1 = torch.tensor(3.0), torch.tensor(1.0)   # f1 clips to [1, 3]
        a2, b2 = torch.tensor(2.0), torch.tensor(0.5)   # f2 clips to [0.5, 2]
        xs = torch.tensor([0.0, 0.8, 1.5, 2.5, 4.0])

        expected = mm_apply(a2, b2, mm_apply(a1, b1, xs))   # f2(f1(x))

        a_c, b_c = mm_compose(a2, b2, a1, b1)
        actual = mm_apply(a_c, b_c, xs)

        torch.testing.assert_close(actual, expected)

    def test_compose_associativity(self):
        """(f3 ∘ f2) ∘ f1  ==  f3 ∘ (f2 ∘ f1)."""
        a1, b1 = torch.tensor(5.0), torch.tensor(0.0)
        a2, b2 = torch.tensor(3.0), torch.tensor(1.0)
        a3, b3 = torch.tensor(2.5), torch.tensor(0.5)

        a_left,  b_left  = mm_compose(*mm_compose(a3, b3, a2, b2), a1, b1)
        a_right, b_right = mm_compose(a3, b3, *mm_compose(a2, b2, a1, b1))

        torch.testing.assert_close(a_left, a_right)
        torch.testing.assert_close(b_left, b_right)


# ---------------------------------------------------------------------------
# MinMax scan
# ---------------------------------------------------------------------------

class TestMinMaxScan(unittest.TestCase):

    @staticmethod
    def _sequential_all_states(a, b, x0):
        """Reference loop implementation of the MinMax recurrence."""
        B, T, D = a.shape
        x = x0
        out = [x0.unsqueeze(1)]
        for t in range(T):
            x = torch.maximum(torch.minimum(a[:, t], x), b[:, t])
            out.append(x.unsqueeze(1))
        return torch.cat(out, dim=1)   # (B, T+1, D)

    def _inputs(self, B=3, T=10, D=8):
        torch.manual_seed(0)
        a  = torch.rand(B, T, D) + 0.5    # keep a reasonably above 0
        b  = torch.rand(B, T, D) * 0.3    # keep b small and below a
        x0 = torch.rand(B, D)
        return a, b, x0

    def test_output_shape(self):
        a, b, x0 = self._inputs(B=3, T=7, D=16)
        out = all_states(a, b, x0)
        self.assertEqual(out.shape, (3, 8, 16))   # T+1 time steps

    def test_initial_state_preserved(self):
        a, b, x0 = self._inputs()
        out = all_states(a, b, x0)
        torch.testing.assert_close(out[:, 0], x0)

    def test_matches_sequential_loop(self):
        """Scan must agree exactly with the sequential recurrence.

        min/max are exact floating-point operations (no rounding), so the two
        implementations should produce bit-identical results.
        """
        a, b, x0 = self._inputs(B=4, T=12, D=16)
        parallel   = all_states(a, b, x0)
        sequential = self._sequential_all_states(a, b, x0)
        torch.testing.assert_close(parallel, sequential)


# ---------------------------------------------------------------------------
# MinMaxRNCConfig
# ---------------------------------------------------------------------------

class TestMinMaxRNCConfig(unittest.TestCase):

    def test_flat_params_stored(self):
        cfg = MinMaxRNCConfig(d_model=64, n_layers=4, d_state=16,
                              norm='rmsnorm', output_gate=True)
        self.assertEqual(cfg.d_model, 64)
        self.assertEqual(cfg.n_layers, 4)
        self.assertEqual(cfg.d_state, 16)
        self.assertEqual(cfg.norm, 'rmsnorm')
        self.assertTrue(cfg.output_gate)

    def test_layer_cfg_derives_correctly(self):
        cfg = MinMaxRNCConfig(d_model=64, n_layers=4, d_state=16)
        lc  = cfg.layer_cfg
        self.assertEqual(lc.d_model, 64)
        self.assertEqual(lc.neuron.d_model, 64)
        self.assertEqual(lc.neuron.d_state, 16)
        self.assertEqual(lc.neuron._num_blocks, 4)
        self.assertEqual(lc.feedforward._num_blocks, 4)

    def test_presets_default_dims(self):
        self.assertEqual(MinMaxRNCConfig.small().d_model, 90)
        self.assertEqual(MinMaxRNCConfig.small().n_layers, 2)
        self.assertEqual(MinMaxRNCConfig.medium().d_model,  512)
        self.assertEqual(MinMaxRNCConfig.medium().n_layers, 8)
        self.assertEqual(MinMaxRNCConfig.large().d_model, 728)
        self.assertEqual(MinMaxRNCConfig.large().n_layers, 12)

    def test_preset_d_model_override(self):
        cfg = MinMaxRNCConfig.small()
        self.assertEqual(cfg.d_model, 90)
        self.assertEqual(cfg.n_layers, 2)   # preset n_layers unchanged


# ---------------------------------------------------------------------------
# MinMaxRNC forward
# ---------------------------------------------------------------------------

class TestMinMaxRNCForward(unittest.TestCase):

    def setUp(self):
        self.model = MinMaxRNC(SMALL_CFG)
        self.model.eval()

    def test_output_shape(self):
        y = self.model(torch.randn(B, T, D), unroll_steps=1)
        self.assertEqual(y.shape, (B, T, D))

    def test_return_state_false_returns_tensor(self):
        out = self.model(torch.randn(B, T, D), unroll_steps=1, return_state=False)
        self.assertIsInstance(out, torch.Tensor)

    def test_return_state_true_returns_tuple(self):
        y, state = self.model(torch.randn(B, T, D), unroll_steps=1, return_state=True)
        self.assertEqual(y.shape, (B, T, D))
        self.assertEqual(len(state), SMALL_CFG.n_layers)

    def test_stateless_matches_stateful_output(self):
        u = torch.randn(B, T, D)
        with torch.no_grad():
            y1 = self.model(u, unroll_steps=1)
            y2, _ = self.model(u, unroll_steps=1, return_state=True)
        torch.testing.assert_close(y1, y2)

    def test_state_continuity(self):
        """Processing [A | B] at once equals processing A then B from the returned state."""
        u    = torch.randn(B, T, D)
        half = T // 2
        with torch.no_grad():
            y_full       = self.model(u, unroll_steps=1)
            y_a, state_a = self.model(u[:, :half], unroll_steps=1, return_state=True)
            y_b          = self.model(u[:, half:], unroll_steps=1, state=state_a)
        torch.testing.assert_close(y_full[:, :half], y_a)
        torch.testing.assert_close(y_full[:, half:], y_b)

    def test_unroll_steps_equivalence(self):
        """unroll_steps=1 (token-by-token) and unroll_steps=T (single chunk) give identical output."""
        u = torch.randn(B, T, D)
        m1 = MinMaxRNC(MinMaxRNCConfig(d_model=D, n_layers=2, d_state=8))
        m2 = MinMaxRNC(MinMaxRNCConfig(d_model=D, n_layers=2, d_state=8))
        m2.load_state_dict(m1.state_dict())
        m1.eval(); m2.eval()
        with torch.no_grad():
            torch.testing.assert_close(m1(u,unroll_steps=1), m2(u,unroll_steps=T))

    def test_batch_size_1(self):
        u = torch.randn(1, T, D)
        y = self.model(u, unroll_steps=1)
        self.assertEqual(y.shape, (1, T, D))

    def test_single_token(self):
        u = torch.randn(B, 1, D)
        y, state = self.model(u, unroll_steps=1, return_state=True)
        self.assertEqual(y.shape, (B, 1, D))
        self.assertEqual(len(state), SMALL_CFG.n_layers)

    def test_gradient_flow(self):
        """A backward pass must complete without error and produce non-zero gradients."""
        u = torch.randn(B, T, D)
        y = self.model(u, unroll_steps=1)
        loss = y.sum()
        loss.backward()
        grads = [p.grad for p in self.model.parameters() if p.grad is not None]
        self.assertGreater(len(grads), 0)
        self.assertTrue(any(g.abs().sum().item() > 0 for g in grads))


# ---------------------------------------------------------------------------
# MinMaxRNC config options (smoke tests)
# ---------------------------------------------------------------------------

class TestMinMaxRNCOptions(unittest.TestCase):
    """Each test exercises a single config knob end-to-end."""

    def _smoke(self, cfg):
        model = MinMaxRNC(cfg)
        model.eval()
        u = torch.randn(B, T, cfg.d_model)
        with torch.no_grad():
            y, state = model(u, unroll_steps=1, return_state=True)
        self.assertEqual(y.shape, (B, T, cfg.d_model))
        self.assertEqual(len(state), cfg.n_layers)

    def test_simple_conv(self):
        self._smoke(MinMaxRNCConfig(d_model=D, n_layers=2, d_state=8, conv_type='simple'))

    def test_simplesimple_conv(self):
        self._smoke(MinMaxRNCConfig(d_model=D, n_layers=2, d_state=8, conv_type='simplesimple'))

    def test_output_gate(self):
        self._smoke(MinMaxRNCConfig(d_model=D, n_layers=2, d_state=8, output_gate=True))

    def test_train_init(self):
        self._smoke(MinMaxRNCConfig(d_model=D, n_layers=2, d_state=8, train_init=True))

    def test_layernorm(self):
        self._smoke(MinMaxRNCConfig(d_model=D, n_layers=2, d_state=8, norm='layernorm'))

    def test_rmsnorm(self):
        self._smoke(MinMaxRNCConfig(d_model=D, n_layers=2, d_state=8, norm='rmsnorm'))

    def test_no_norm(self):
        self._smoke(MinMaxRNCConfig(d_model=D, n_layers=2, d_state=8, norm='none'))

    def test_postlayers_ffn(self):
        self._smoke(MinMaxRNCConfig(d_model=D, n_layers=2, d_state=8, use_postlayers_ffn=True))

    def test_basic_ffn(self):
        self._smoke(MinMaxRNCConfig(d_model=D, n_layers=2, d_state=8, ffn_type='basic'))

    def test_gated_ffn(self):
        self._smoke(MinMaxRNCConfig(d_model=D, n_layers=2, d_state=8, ffn_type='gated'))

    def test_prelayers_dropout(self):
        self._smoke(MinMaxRNCConfig(d_model=D, n_layers=2, d_state=8, prelayers_dropout=0.1))



# ---------------------------------------------------------------------------
# MinMaxRNC LM wrapper
# ---------------------------------------------------------------------------

class TestMinMaxRNCLM(unittest.TestCase):

    def setUp(self):
        self.vocab  = 100
        self.lm_cfg = MinMaxRNCLMConfig(backbone=SMALL_CFG)
        self.lm     = MinMaxRNC_LM(vocab_size=self.vocab, cfg=self.lm_cfg)
        self.lm.eval()

    def _tokens(self, t=T):
        return torch.randint(0, self.vocab, (B, t))

    def test_logits_shape(self):
        self.assertEqual(self.lm(self._tokens(), unroll_steps=1).shape, (B, T, self.vocab))

    def test_return_state(self):
        logits, state = self.lm(self._tokens(), unroll_steps=1, return_state=True)
        self.assertEqual(logits.shape, (B, T, self.vocab))
        self.assertEqual(len(state), SMALL_CFG.n_layers)

    def test_stateless_matches_stateful(self):
        tokens = self._tokens()
        with torch.no_grad():
            l1 = self.lm(tokens, unroll_steps=1)
            l2, _ = self.lm(tokens, unroll_steps=1, return_state=True)
        torch.testing.assert_close(l1, l2)

    def test_tied_weights(self):
        lm = MinMaxRNC_LM(self.vocab,
                          MinMaxRNCLMConfig(backbone=SMALL_CFG, tie_weights=True))
        self.assertIs(lm.token_emb.weight, lm.lm_head.weight)

    def test_untied_weights(self):
        lm = MinMaxRNC_LM(self.vocab,
                          MinMaxRNCLMConfig(backbone=SMALL_CFG, tie_weights=False))
        self.assertIsNot(lm.token_emb.weight, lm.lm_head.weight)

    def test_state_continuity(self):
        """Processing [A | B] at once must equal processing A then B from the returned state."""
        tokens = self._tokens()
        half   = T // 2
        with torch.no_grad():
            l_full       = self.lm(tokens, unroll_steps=1)
            l_a, state_a = self.lm(tokens[:, :half], unroll_steps=1, return_state=True)
            l_b          = self.lm(tokens[:, half:], unroll_steps=1, state=state_a)
        torch.testing.assert_close(l_full[:, :half], l_a)
        torch.testing.assert_close(l_full[:, half:], l_b)


if __name__ == '__main__':
    unittest.main()
