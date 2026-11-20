# SPDX-FileCopyrightText: 2026 Alessandro Ronca
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""
Tests for the matrix MinMax neurons (new in 0.4.0).

Organisation
------------
TestMinMaxMatrixOperator        — apply_matrix / compose_matrix correctness
TestMinMaxMatrixScan            — matrix parallel scan vs sequential reference
TestMatrixNeuronGrouping        — num_neurons > 1 ≡ separate neurons summed
TestMatrixNeuronLowRank         — n_h > 0 rank-1-product parametrisation of R_t
TestNeuronOutputInitScaling     — wang_init_ scaling across a layer's neurons
TestMatrixNeuronsConfig         — MinMaxRNCConfig.matrix_neurons parsing/validation/warning
TestMatrixNeuronsLayerAndModel  — layers/models mixing diagonal and matrix neurons
"""

import math
import unittest
import warnings

import torch

from minmaxrnc import MatrixNeuronGroup
from minmaxrnc.minmax_rnc      import MinMaxRNC, MinMaxRNCConfig
from minmaxrnc.minmax_rnc_lm   import MinMaxRNC_LM, MinMaxRNCLMConfig
from minmaxrnc.minmax_layer    import MinMaxLayer, MinMaxLayerConfig
from minmaxrnc.minmax_neuron   import (
    MinMaxNeuron, MinMaxNeuronConfig,
    MatrixMinMaxNeuron, MatrixMinMaxNeuronConfig,
)
from minmaxrnc.minmax_scan     import all_states_matrix
from minmaxrnc.minmax_operator import (
    apply as mm_apply,
    apply_matrix as mm_apply_matrix, compose_matrix as mm_compose_matrix,
)
from minmaxrnc.modules.basic_conv  import BasicConvConfig
from minmaxrnc.modules.feedforward import FeedForwardConfig


def _layer(d_model, d_state=0, matrix_neurons=(), n_layers=1, **kwargs):
    """A single MinMaxLayer built through the public config, as the model does."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)   # d_state=1 groups warn
        cfg = MinMaxRNCConfig(
            d_model=d_model, n_layers=n_layers, d_state=d_state,
            matrix_neurons=matrix_neurons, ffn_type='basic', **kwargs,
        )
    return MinMaxLayer(cfg.layer_cfg, first=True)


# ---------------------------------------------------------------------------
# Matrix operator
# ---------------------------------------------------------------------------

class TestMinMaxMatrixOperator(unittest.TestCase):
    """
    f(x) = max(R ⊗ x, s) with (R ⊗ x)_i = max_j min(R_ij, x_j). A diagonal R
    (off-diagonal -inf) recovers the scalar/diagonal operator.
    """

    def _random(self, B=3, D=6, seed=0):
        g = torch.Generator().manual_seed(seed)
        R = torch.randn(B, D, D, generator=g)
        s = torch.randn(B, D, generator=g)
        x = torch.randn(B, D, generator=g)
        return R, s, x

    def test_diagonal_special_case_matches_scalar_operator(self):
        D = 5
        a = torch.rand(D) + 0.5
        b = torch.rand(D) * 0.3
        x = torch.rand(D)
        R = torch.full((D, D), float('-inf'))
        R[torch.arange(D), torch.arange(D)] = a
        torch.testing.assert_close(mm_apply_matrix(R, b, x), mm_apply(a, b, x))

    def test_compose_matrix_matches_sequential(self):
        R2, s2, _ = self._random(seed=1)
        R1, s1, x = self._random(seed=2)
        expected = mm_apply_matrix(R2, s2, mm_apply_matrix(R1, s1, x))
        R_c, s_c = mm_compose_matrix(R2, s2, R1, s1)
        torch.testing.assert_close(mm_apply_matrix(R_c, s_c, x), expected, atol=1e-6, rtol=1e-6)

    def test_compose_matrix_associativity(self):
        R1, s1, _ = self._random(seed=3)
        R2, s2, _ = self._random(seed=4)
        R3, s3, _ = self._random(seed=5)
        R_left,  s_left  = mm_compose_matrix(*mm_compose_matrix(R3, s3, R2, s2), R1, s1)
        R_right, s_right = mm_compose_matrix(R3, s3, *mm_compose_matrix(R2, s2, R1, s1))
        torch.testing.assert_close(R_left, R_right, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(s_left, s_right, atol=1e-6, rtol=1e-6)


# ---------------------------------------------------------------------------
# Matrix scan
# ---------------------------------------------------------------------------

class TestMinMaxMatrixScan(unittest.TestCase):

    @staticmethod
    def _sequential(R, s, x0):
        x, out = x0, [x0.unsqueeze(1)]
        for t in range(R.shape[1]):
            x = mm_apply_matrix(R[:, t], s[:, t], x)
            out.append(x.unsqueeze(1))
        return torch.cat(out, dim=1)

    def _inputs(self, B=3, T=10, D=6):
        torch.manual_seed(0)
        return torch.randn(B, T, D, D), torch.randn(B, T, D), torch.randn(B, D)

    def test_output_shape(self):
        R, s, x0 = self._inputs(B=3, T=7, D=8)
        self.assertEqual(all_states_matrix(R, s, x0).shape, (3, 8, 8))

    def test_initial_state_preserved(self):
        R, s, x0 = self._inputs()
        torch.testing.assert_close(all_states_matrix(R, s, x0)[:, 0], x0)

    def test_matches_sequential_loop(self):
        for T in (1, 2, 5, 12, 17):
            with self.subTest(T=T):
                R, s, x0 = self._inputs(B=4, T=T, D=6)
                torch.testing.assert_close(
                    all_states_matrix(R, s, x0), self._sequential(R, s, x0), atol=1e-5, rtol=1e-5,
                )


# ---------------------------------------------------------------------------
# Grouping (num_neurons > 1)
# ---------------------------------------------------------------------------

class TestMatrixNeuronGrouping(unittest.TestCase):
    """num_neurons=K batches K neurons into one module: must equal K separate
    num_neurons=1 neurons whose outputs are summed."""

    def test_default_num_neurons_is_one(self):
        cfg = MatrixMinMaxNeuronConfig(_num_blocks=1, d_model=8, d_state=4)
        self.assertEqual(cfg.num_neurons, 1)
        self.assertEqual(MatrixMinMaxNeuron(cfg).initial_state.shape, (1, 4))

    def test_shapes(self):
        d_model, d_state, K, B, T = 10, 3, 5, 2, 6
        neuron = MatrixMinMaxNeuron(MatrixMinMaxNeuronConfig(
            _num_blocks=1, d_model=d_model, d_state=d_state, num_neurons=K,
        ))
        self.assertEqual(neuron.initial_state.shape, (K, d_state))
        out, state = neuron(torch.randn(B, T, d_model), neuron.initial_state)
        self.assertEqual(out.shape, (B, T, d_model))
        self.assertEqual(state.shape, (B, K, d_state))

    def test_batched_matches_separate_neurons_summed(self):
        torch.manual_seed(0)
        d_model, D, K, B, T = 12, 4, 3, 2, 7
        grouped = MatrixMinMaxNeuron(MatrixMinMaxNeuronConfig(
            _num_blocks=1, d_model=d_model, d_state=D, num_neurons=K,
        ))
        separates = [
            MatrixMinMaxNeuron(MatrixMinMaxNeuronConfig(_num_blocks=1, d_model=d_model, d_state=D))
            for _ in range(K)
        ]
        # Biases are zero-initialised regardless of grouping; copy weights only.
        with torch.no_grad():
            for k, n in enumerate(separates):
                grouped.s.weight[k*D:(k+1)*D].copy_(n.s.weight)
                grouped.r.weight[k*D*D:(k+1)*D*D].copy_(n.r.weight)
                grouped.o.weight[:, k*D:(k+1)*D].copy_(n.o.weight)
                grouped.o_g.weight[k*D:(k+1)*D].copy_(n.o_g.weight)

        u = torch.randn(B, T, d_model)
        g_out, g_state = grouped(u, grouped.initial_state)
        outs, states = zip(*(n(u, n.initial_state) for n in separates))
        torch.testing.assert_close(g_out, sum(outs), atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(g_state, torch.cat(states, dim=1), atol=1e-6, rtol=1e-6)


# ---------------------------------------------------------------------------
# Low-rank parametrisation (n_h > 0, experimental)
# ---------------------------------------------------------------------------

class TestMatrixNeuronLowRank(unittest.TestCase):

    def _neuron(self, **kw):
        base = dict(_num_blocks=1, d_model=8, d_state=4)
        base.update(kw)
        return MatrixMinMaxNeuron(MatrixMinMaxNeuronConfig(**base))

    def test_n_h_zero_uses_r_not_k_beta(self):
        n = self._neuron(n_h=0)
        self.assertTrue(hasattr(n, 'r'))
        self.assertFalse(hasattr(n, 'k'))
        self.assertFalse(hasattr(n, 'beta'))

    def test_n_h_positive_uses_k_beta_not_r(self):
        n = self._neuron(n_h=2)
        self.assertFalse(hasattr(n, 'r'))
        self.assertTrue(hasattr(n, 'k'))
        self.assertTrue(hasattr(n, 'beta'))

    def test_shapes(self):
        n = self._neuron(d_model=10, d_state=3, num_neurons=5, n_h=2)
        out, state = n(torch.randn(2, 6, 10), n.initial_state)
        self.assertEqual(out.shape, (2, 6, 10))
        self.assertEqual(state.shape, (2, 5, 3))

    def _check_R(self, normalize_keys):
        torch.manual_seed(0)
        d_model, D, K, n_h, B, T = 6, 3, 2, 3, 2, 4
        n = self._neuron(d_model=d_model, d_state=D, num_neurons=K, n_h=n_h,
                         normalize_keys=normalize_keys)
        u = torch.randn(B, T, d_model)
        R = n._compute_R_low_rank(u, B, T)
        k    = n.k(u).view(B, T, K, n_h, D)
        beta = n.beta(u).view(B, T, K, n_h)
        eye  = torch.eye(D)
        expected = torch.empty(B, T, K, D, D)
        for b in range(B):
            for t in range(T):
                for kk in range(K):
                    Rt = eye.clone()
                    for i in range(n_h):
                        k_i = k[b, t, kk, i]
                        if normalize_keys:
                            k_i = k_i / k_i.norm()
                        Rt = Rt @ (eye - beta[b, t, kk, i] * torch.outer(k_i, k_i))
                    expected[b, t, kk] = Rt
        torch.testing.assert_close(R, expected, atol=1e-5, rtol=1e-5)

    def test_R_matches_manual_rank1_product(self):
        self._check_R(normalize_keys=False)

    def test_R_matches_manual_rank1_product_with_normalize_keys(self):
        self._check_R(normalize_keys=True)

    def test_normalize_keys_default_false(self):
        self.assertFalse(MatrixMinMaxNeuronConfig(_num_blocks=1, d_model=8, d_state=4, n_h=2).normalize_keys)

    def test_beta_unconstrained_affine(self):
        n = self._neuron(d_model=6, d_state=3, n_h=2)
        with torch.no_grad():
            n.beta.weight.fill_(10.0)
            n.beta.bias.zero_()
        self.assertGreater(n.beta(torch.full((1, 1, 6), 10.0)).abs().max().item(), 2.0)

    def test_gradient_flow(self):
        for normalize_keys in (False, True):
            with self.subTest(normalize_keys=normalize_keys):
                n = self._neuron(n_h=2, normalize_keys=normalize_keys)
                out, _ = n(torch.randn(2, 5, 8), n.initial_state)
                out.sum().backward()
                for name, p in [('k', n.k.weight), ('beta', n.beta.weight), ('o', n.o.weight)]:
                    self.assertIsNotNone(p.grad, f"{name}.weight has no gradient")
                    self.assertGreater(p.grad.abs().sum().item(), 0, f"{name}.weight gradient is zero")

    def test_config_passes_n_h_and_normalize_keys_through(self):
        layer = _layer(10, matrix_neurons=[(3, 2, 2, True)])
        m = layer.matrix_neurons[0]
        self.assertTrue(hasattr(m, 'k'))
        self.assertEqual(m.cfg.n_h, 2)
        self.assertTrue(m.cfg.normalize_keys)
        out, _ = layer(torch.randn(2, 5, 10), layer.initial_state)
        self.assertEqual(out.shape, (2, 5, 10))

    def test_rejects_negative_n_h(self):
        with self.assertRaises(ValueError):
            MinMaxRNCConfig(d_model=8, n_layers=1, matrix_neurons=[(3, 2, -1)])
        with self.assertRaises(ValueError):
            MatrixMinMaxNeuronConfig(_num_blocks=1, d_model=8, d_state=3, n_h=-1)


# ---------------------------------------------------------------------------
# Output-projection init scaling
# ---------------------------------------------------------------------------

class TestNeuronOutputInitScaling(unittest.TestCase):
    """Each neuron's output projection is wang_init_-scaled with
    dim = d_state * _num_neurons_in_layer, keeping the layer's *total*
    residual contribution O(1) however its neurons are split."""

    def test_default_num_neurons_in_layer_is_one(self):
        self.assertEqual(MinMaxNeuronConfig(_num_blocks=1, d_model=8, d_state=4)._num_neurons_in_layer, 1)
        self.assertEqual(MatrixMinMaxNeuronConfig(_num_blocks=1, d_model=8, d_state=4)._num_neurons_in_layer, 1)

    def test_wang_init_std_scales_with_num_neurons_in_layer(self):
        torch.manual_seed(0)
        n1 = MinMaxNeuron(MinMaxNeuronConfig(_num_blocks=1, d_model=16, d_state=512, _num_neurons_in_layer=1))
        n4 = MinMaxNeuron(MinMaxNeuronConfig(_num_blocks=1, d_model=16, d_state=512, _num_neurons_in_layer=4))
        ratio = n4.o.weight.std().item() / n1.o.weight.std().item()
        self.assertAlmostEqual(ratio, 1.0 / math.sqrt(4), delta=0.05)

    def test_layer_counts_all_neurons(self):
        layer = _layer(10, d_state=4, matrix_neurons=[(3, 2), (2, 5)])
        expected = 1 + 2 + 5
        self.assertEqual(layer.neuron.cfg._num_neurons_in_layer, expected)
        for m in layer.matrix_neurons:
            self.assertEqual(m.cfg._num_neurons_in_layer, expected)

    def test_diagonal_matches_grouped_d1_matrix_neurons(self):
        """d_state=n (diagonal) and matrix_neurons=[(1, n)] are the same model:
        same seed → identical output (and identical parameter shapes)."""
        d_model, B, T = 20, 3, 5
        for n in (1, 5, 20):
            with self.subTest(n=n):
                torch.manual_seed(0)
                diag = _layer(d_model, d_state=n)
                torch.manual_seed(0)
                mat = _layer(d_model, matrix_neurons=[(1, n)])
                u = torch.randn(B, T, d_model)
                out_diag, _ = diag(u, diag.initial_state)
                out_mat, _ = mat(u, mat.initial_state)
                torch.testing.assert_close(out_diag, out_mat, atol=1e-6, rtol=1e-6)

    def test_layer_output_scale_invariant_to_neuron_split(self):
        d_model, B, T = 32, 8, 16
        torch.manual_seed(0)
        one = _layer(d_model, matrix_neurons=[(2, 16)], n_layers=4)
        torch.manual_seed(0)
        two = _layer(d_model, matrix_neurons=[(2, 8), (3, 8)], n_layers=4)
        u = torch.randn(B, T, d_model)
        std_one = (one(u, one.initial_state)[0] - u).std().item()
        std_two = (two(u, two.initial_state)[0] - u).std().item()
        self.assertAlmostEqual(std_one, std_two, delta=0.3 * max(std_one, std_two))


# ---------------------------------------------------------------------------
# MinMaxRNCConfig.matrix_neurons
# ---------------------------------------------------------------------------

class TestMatrixNeuronsConfig(unittest.TestCase):

    def test_tuple_list_and_group_inputs_normalise_identically(self):
        a = MinMaxRNCConfig(16, 1, 0, matrix_neurons=[(3, 2)])
        b = MinMaxRNCConfig(16, 1, 0, matrix_neurons=[[3, 2, 0, False]])          # YAML-style lists
        c = MinMaxRNCConfig(16, 1, 0, matrix_neurons=(MatrixNeuronGroup(3, 2),))
        self.assertEqual(a.matrix_neurons, (MatrixNeuronGroup(3, 2, 0, False),))
        self.assertEqual(a, b)
        self.assertEqual(a, c)

    def test_config_is_hashable(self):
        cfg = MinMaxRNCConfig(16, 1, 0, matrix_neurons=[[3, 2]])
        self.assertEqual(hash(cfg), hash(MinMaxRNCConfig(16, 1, 0, matrix_neurons=[(3, 2)])))

    def test_rejects_bad_entries(self):
        for bad in ([(0, 2)], [(3, 0)], [(3, 2, -1)], [()], [(1, 2, 3, False, 'extra')], [3]):
            with self.subTest(entry=bad), self.assertRaises((ValueError, TypeError)):
                MinMaxRNCConfig(16, 1, 4, matrix_neurons=bad)

    def test_rejects_no_neurons(self):
        with self.assertRaises(ValueError):
            MinMaxRNCConfig(d_model=16, n_layers=1)
        with self.assertRaises(ValueError):
            MinMaxRNCConfig(16, 1, 0)

    def test_rejects_negative_d_state(self):
        with self.assertRaises(ValueError):
            MinMaxRNCConfig(16, 1, -1)

    def test_same_d_state_with_different_n_h_allowed(self):
        model = MinMaxRNC(MinMaxRNCConfig(16, 1, 0, matrix_neurons=[(3, 2, 0), (3, 2, 2)]))
        self.assertEqual(len(model.layers[0].matrix_neurons), 2)

    def test_scalar_state_group_warns(self):
        """A d_state=1 group is a scalar-state neuron, equivalent to a
        diagonal one: warn, whatever n_h is."""
        for entry in [(1, 4), (1, 1, 2)]:
            with self.subTest(entry=entry):
                with self.assertWarns(UserWarning) as cm:
                    MinMaxRNCConfig(16, 1, 0, matrix_neurons=[entry])
                self.assertIn('d_state=1', str(cm.warning))

    def test_non_scalar_groups_do_not_warn(self):
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            MinMaxRNCConfig(16, 1, 4, matrix_neurons=[(2, 3), (4, 1, 2)])

    def test_lm_wrapper_does_not_repeat_the_warning(self):
        with self.assertWarns(UserWarning):
            backbone = MinMaxRNCConfig(16, 1, 0, matrix_neurons=[(1, 3)])
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            MinMaxRNC_LM(10, MinMaxRNCLMConfig(backbone=backbone))

    def test_num_neurons_per_layer(self):
        self.assertEqual(MinMaxRNCConfig(16, 1, 8).num_neurons_per_layer, 1)
        self.assertEqual(MinMaxRNCConfig(16, 1, 0, matrix_neurons=[(3, 16)]).num_neurons_per_layer, 16)
        self.assertEqual(MinMaxRNCConfig(16, 1, 8, matrix_neurons=[(3, 2), (2, 1)]).num_neurons_per_layer, 4)

    def test_layer_config_rejects_no_neurons(self):
        with self.assertRaises(ValueError):
            MinMaxLayerConfig(
                neuron=None,
                conv=BasicConvConfig(embedding_dim=8),
                d_model=8,
                feedforward=FeedForwardConfig(_num_blocks=1, ffn_type='basic'),
            )


# ---------------------------------------------------------------------------
# Layers and models with matrix neurons
# ---------------------------------------------------------------------------

class TestMatrixNeuronsLayerAndModel(unittest.TestCase):

    def test_mixed_layer_structure_and_state(self):
        layer = _layer(12, d_state=4, matrix_neurons=[(3, 2)])
        self.assertIsInstance(layer.neuron, MinMaxNeuron)
        self.assertEqual(len(layer.matrix_neurons), 1)
        self.assertEqual(layer.matrix_neurons[0].K, 2)
        state = layer.initial_state
        self.assertEqual(list(state.keys()), ['neuron', 'conv', 'matrix_neurons'])
        out, new_state = layer(torch.randn(2, 7, 12), state)
        self.assertEqual(out.shape, (2, 7, 12))
        self.assertEqual(new_state['neuron'].shape, (2, 4))
        self.assertEqual([s.shape for s in new_state['matrix_neurons']], [(2, 2, 3)])

    def test_matrix_only_layer(self):
        layer = _layer(10, matrix_neurons=[(4, 1), (5, 1)])
        self.assertIsNone(layer.neuron)
        self.assertNotIn('neuron', layer.initial_state)
        self.assertEqual(sorted(m.D for m in layer.matrix_neurons), [4, 5])
        out, new_state = layer(torch.randn(2, 5, 10), layer.initial_state)
        self.assertEqual(out.shape, (2, 5, 10))
        self.assertNotIn('neuron', new_state)

    def test_mixed_layer_sums_neuron_outputs(self):
        """The layer adds diagonal + matrix outputs to the residual stream."""
        torch.manual_seed(0)
        layer = _layer(8, d_state=3, matrix_neurons=[(2, 2)], norm='none').eval()
        u = torch.randn(2, 6, 8)
        out, _ = layer(u, layer.initial_state)

        conv, _ = layer.conv(u, layer.initial_state['conv'])
        ffn = layer.ffn(u + conv)
        neuron_in = u + conv + ffn
        diag_out, _ = layer.neuron(neuron_in, layer.neuron.initial_state)
        mat_out, _ = layer.matrix_neurons[0](neuron_in, layer.matrix_neurons[0].initial_state)
        torch.testing.assert_close(out, u + diag_out + mat_out, atol=1e-6, rtol=1e-6)

    def test_gradient_flow_mixed(self):
        layer = _layer(8, d_state=3, matrix_neurons=[(2, 2), (3, 1, 2)])
        out, _ = layer(torch.randn(2, 6, 8), layer.initial_state)
        out.sum().backward()
        for name, p in layer.named_parameters():
            if p.requires_grad and 'matrix_neurons' in name and name.endswith('weight'):
                self.assertIsNotNone(p.grad, f"{name} has no gradient")

    def test_chunked_forward_matches_single_pass(self):
        """The carried matrix-neuron state makes chunked unrolling exact."""
        for d_state, groups in [(0, [(3, 2)]), (4, [(3, 2), (2, 1, 2)])]:
            with self.subTest(d_state=d_state, groups=groups):
                torch.manual_seed(0)
                model = MinMaxRNC(MinMaxRNCConfig(12, 2, d_state, matrix_neurons=groups)).eval()
                u = torch.randn(2, 17, 12)
                with torch.no_grad():
                    full = model(u, 17)
                    chunked = model(u, 5)
                torch.testing.assert_close(chunked, full, atol=1e-5, rtol=1e-5)

    def test_state_continuation_across_calls(self):
        torch.manual_seed(0)
        model = MinMaxRNC(MinMaxRNCConfig(12, 2, 4, matrix_neurons=[(3, 2)])).eval()
        u = torch.randn(2, 10, 12)
        with torch.no_grad():
            full = model(u, 10)
            y1, st = model(u[:, :6], 6, return_state=True)
            y2 = model(u[:, 6:], 4, state=st)
        torch.testing.assert_close(torch.cat([y1, y2], dim=1), full, atol=1e-5, rtol=1e-5)

    def test_train_init_makes_matrix_initial_state_learnable(self):
        model = MinMaxRNC(MinMaxRNCConfig(8, 1, 0, matrix_neurons=[(3, 2)], train_init=True))
        self.assertTrue(model.layers[0].matrix_neurons[0].initial_state.requires_grad)

    def test_lm_with_matrix_neurons(self):
        model = MinMaxRNC_LM(20, MinMaxRNCLMConfig(
            backbone=MinMaxRNCConfig(16, 2, 0, matrix_neurons=[(3, 4)]),
        ))
        logits = model(torch.randint(0, 20, (2, 9)), 4)
        self.assertEqual(logits.shape, (2, 9, 20))

    def test_state_dict_round_trip(self):
        torch.manual_seed(0)
        cfg = MinMaxRNCConfig(12, 2, 4, matrix_neurons=[(3, 2), (2, 1, 2)])
        a = MinMaxRNC(cfg).eval()
        torch.manual_seed(1)
        b = MinMaxRNC(cfg).eval()
        b.load_state_dict(a.state_dict(), strict=True)
        u = torch.randn(1, 7, 12)
        with torch.no_grad():
            torch.testing.assert_close(a(u, 7), b(u, 7))
        self.assertTrue(any(k.startswith('layers.0.matrix_neurons.1.') for k in a.state_dict()))


if __name__ == '__main__':
    unittest.main()
