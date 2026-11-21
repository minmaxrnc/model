# SPDX-FileCopyrightText: 2026 Alessandro Ronca
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""
Tests for MinMax neuron arrays of any state dimension (new in 0.4.0).

Organisation
------------
TestMinMaxMatrixOperator   — apply_matrix / compose_matrix correctness
TestMinMaxMatrixScan       — matrix parallel scan vs sequential reference
TestArrayBatching          — num_neurons > 1 ≡ separate neurons summed
TestScalarStateArrays      — d_state = 1: element-wise path, ≡ 0.3.0 neuron, ≡ matrix path
TestScalarNeuronsField     — MinMaxRNCConfig.scalar_neurons ≡ its legacy name d_state
TestOutputInitScaling      — wang_init_ scaling by the layer's total state size
TestNeuronsConfig          — MinMaxRNCConfig.neurons parsing and validation
TestLayersAndModels        — layers/models mixing arrays of different state dimensions
"""

import math
import unittest
import warnings
from unittest import mock

import torch

from minmaxrnc import NeuronArray, minmax_scan
from minmaxrnc.minmax_rnc      import MinMaxRNC, MinMaxRNCConfig
from minmaxrnc.minmax_rnc_lm   import MinMaxRNC_LM, MinMaxRNCLMConfig
from minmaxrnc.minmax_layer    import MinMaxLayer, MinMaxLayerConfig
from minmaxrnc.minmax_neuron   import (
    MinMaxNeuron, MinMaxNeuronConfig,
    MinMaxNeuronArray, MinMaxNeuronArrayConfig,
)
from minmaxrnc.minmax_scan     import all_states_matrix
from minmaxrnc.minmax_operator import (
    apply as mm_apply,
    apply_matrix as mm_apply_matrix, compose_matrix as mm_compose_matrix,
)
from minmaxrnc.modules.basic_conv  import BasicConvConfig
from minmaxrnc.modules.feedforward import FeedForwardConfig


def _layer(d_model, d_state=0, neurons=(), n_layers=1, **kwargs):
    """A single MinMaxLayer built through the public config, as the model does."""
    cfg = MinMaxRNCConfig(
        d_model=d_model, n_layers=n_layers, d_state=d_state,
        neurons=neurons, ffn_type='basic', **kwargs,
    )
    return MinMaxLayer(cfg.layer_cfg, first=True)


def _array(**kw):
    base = dict(_num_blocks=1, d_model=8, d_state=4)
    base.update(kw)
    return MinMaxNeuronArray(MinMaxNeuronArrayConfig(**base))


def _aten_ops(fn, *args):
    from torch.fx.experimental.proxy_tensor import make_fx
    with torch.no_grad():
        graph = make_fx(fn)(*args)
    return [str(n.target) for n in graph.graph.nodes if n.op == 'call_function']


# ---------------------------------------------------------------------------
# Matrix operator and scan
# ---------------------------------------------------------------------------

class TestMinMaxMatrixOperator(unittest.TestCase):
    """
    f(x) = max(R ⊗ x, s) with (R ⊗ x)_i = max_j min(R_ij, x_j). A diagonal R
    (off-diagonal -inf) recovers the element-wise operator.
    """

    def _random(self, B=3, D=6, seed=0):
        g = torch.Generator().manual_seed(seed)
        return torch.randn(B, D, D, generator=g), torch.randn(B, D, generator=g), torch.randn(B, D, generator=g)

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
# Array batching (num_neurons > 1)
# ---------------------------------------------------------------------------

class TestArrayBatching(unittest.TestCase):
    """num_neurons = K batches K neurons into one module: must equal K separate
    one-neuron arrays whose outputs are summed."""

    def test_defaults(self):
        cfg = MinMaxNeuronArrayConfig(_num_blocks=1, d_model=8, d_state=4)
        self.assertEqual(cfg.num_neurons, 1)
        self.assertEqual(MinMaxNeuronArray(cfg).initial_state.shape, (4,))

    def test_shapes_use_flat_state_layout(self):
        d_model, d, K, B, T = 10, 3, 5, 2, 6
        arr = _array(d_model=d_model, d_state=d, num_neurons=K)
        self.assertEqual((arr.d, arr.K, arr.D), (d, K, K * d))
        self.assertEqual(arr.initial_state.shape, (K * d,))
        out, state = arr(torch.randn(B, T, d_model), arr.initial_state)
        self.assertEqual(out.shape, (B, T, d_model))
        self.assertEqual(state.shape, (B, K * d))

    def test_batched_matches_separate_neurons_summed(self):
        for d in (1, 4):
            with self.subTest(d_state=d):
                torch.manual_seed(0)
                d_model, K, B, T = 12, 3, 2, 7
                batched = _array(d_model=d_model, d_state=d, num_neurons=K)
                separates = [_array(d_model=d_model, d_state=d) for _ in range(K)]
                # Biases are zero-initialised regardless of batching; copy weights only.
                with torch.no_grad():
                    for k, n in enumerate(separates):
                        batched.s.weight[k*d:(k+1)*d].copy_(n.s.weight)
                        batched.r.weight[k*d*d:(k+1)*d*d].copy_(n.r.weight)
                        batched.o.weight[:, k*d:(k+1)*d].copy_(n.o.weight)
                        batched.o_g.weight[k*d:(k+1)*d].copy_(n.o_g.weight)
                u = torch.randn(B, T, d_model)
                batched_out, batched_state = batched(u, batched.initial_state)
                outs, states = zip(*(n(u, n.initial_state) for n in separates))
                torch.testing.assert_close(batched_out, sum(outs), atol=1e-6, rtol=1e-6)
                torch.testing.assert_close(batched_state, torch.cat(states, dim=1), atol=1e-6, rtol=1e-6)


# ---------------------------------------------------------------------------
# Scalar-state arrays (d_state = 1)
# ---------------------------------------------------------------------------

class TestScalarStateArrays(unittest.TestCase):

    @staticmethod
    def _spy():
        calls = []
        orig = {n: getattr(minmax_scan, n) for n in ('all_states', 'all_states_matrix')}
        def wrap(name):
            def spy(*a):
                calls.append(name)
                return orig[name](*a)
            return spy
        patches = [mock.patch.object(minmax_scan, n, wrap(n)) for n in orig]
        return calls, patches

    def _scans_used(self, array):
        calls, patches = self._spy()
        for p in patches:
            p.start()
        try:
            array(torch.randn(2, 5, array.I), array.initial_state)
        finally:
            for p in patches:
                p.stop()
        return set(calls)

    def test_d1_uses_elementwise_scan(self):
        self.assertEqual(self._scans_used(_array(d_state=1, num_neurons=6)), {'all_states'})

    def test_d_gt_1_uses_matrix_scan(self):
        self.assertEqual(self._scans_used(_array(d_state=3, num_neurons=2)), {'all_states_matrix'})

    def test_d1_elementwise_path_equals_matrix_path(self):
        """For d = 1 the element-wise scan is just a faster evaluation of the
        same recurrence: forcing the matrix path gives the same states."""
        torch.manual_seed(0)
        arr = _array(d_model=8, d_state=1, num_neurons=5)
        B, T = 2, 9
        u = torch.randn(B, T, 8)
        s, r = arr.s(u), arr.r(u)
        x0 = torch.randn(B, 5)
        torch.testing.assert_close(
            arr._scan_scalar(s, r, x0, B, T),
            arr._scan_matrix(s, r, x0, B, T),
        )

    def test_minmax_neuron_is_an_array_of_scalar_state_neurons(self):
        n = MinMaxNeuron(MinMaxNeuronConfig(_num_blocks=1, d_model=8, d_state=6))
        self.assertIsInstance(n, MinMaxNeuronArray)
        self.assertEqual((n.d, n.K, n.D), (1, 6, 6))
        self.assertEqual((n.array_cfg.d_state, n.array_cfg.num_neurons), (1, 6))
        self.assertIsInstance(n.cfg, MinMaxNeuronConfig)

    def test_minmax_neuron_equals_scalar_state_array(self):
        """MinMaxNeuron(d_state=D) and MinMaxNeuronArray(d_state=1,
        num_neurons=D): same parameters, same seeded init, same outputs, same
        computation graph."""
        torch.manual_seed(0)
        a = MinMaxNeuron(MinMaxNeuronConfig(_num_blocks=2, d_model=8, d_state=6))
        torch.manual_seed(0)
        b = _array(_num_blocks=2, d_model=8, d_state=1, num_neurons=6)
        sa, sb = a.state_dict(), b.state_dict()
        self.assertEqual(list(sa), list(sb))
        self.assertTrue(all(torch.equal(sa[k], sb[k]) for k in sa))
        u = torch.randn(2, 5, 8)
        torch.testing.assert_close(a(u, a.initial_state), b(u, b.initial_state), rtol=0, atol=0)
        self.assertEqual(
            _aten_ops(lambda x: a(x, a.initial_state), u),
            _aten_ops(lambda x: b(x, b.initial_state), u),
        )

    def test_d_state_shorthand_equals_explicit_array_in_a_model(self):
        """MinMaxRNCConfig(d_state=n) and neurons=[(1, n)] are the same model:
        same seeded weights, same outputs, same computation graph."""
        for n in (1, 5, 20):
            with self.subTest(n=n):
                torch.manual_seed(0)
                a = MinMaxRNC(MinMaxRNCConfig(20, 2, n)).eval()
                torch.manual_seed(0)
                b = MinMaxRNC(MinMaxRNCConfig(20, 2, 0, neurons=[(1, n)])).eval()
                # Same tensors in the same order; only the names differ
                # (layers.N.neuron.* vs layers.N.neuron_arrays.0.*).
                va, vb = list(a.state_dict().values()), list(b.state_dict().values())
                self.assertEqual(len(va), len(vb))
                self.assertTrue(all(torch.equal(x, y) for x, y in zip(va, vb)))
                u = torch.randn(3, 7, 20)
                with torch.no_grad():
                    torch.testing.assert_close(a(u, 4), b(u, 4), rtol=0, atol=0)
                self.assertEqual(_aten_ops(lambda x: a(x, 4), u), _aten_ops(lambda x: b(x, 4), u))


# ---------------------------------------------------------------------------
# scalar_neurons and its legacy name d_state
# ---------------------------------------------------------------------------

class TestScalarNeuronsField(unittest.TestCase):

    def test_scalar_neurons_equals_d_state(self):
        """scalar_neurons=N and d_state=N build the same model: same parameter
        names and seeded values, same outputs, same computation graph."""
        for kw in ({}, dict(neurons=[(3, 2)])):
            with self.subTest(extra=kw):
                torch.manual_seed(0)
                a = MinMaxRNC(MinMaxRNCConfig(16, 2, scalar_neurons=8, **kw)).eval()
                torch.manual_seed(0)
                b = MinMaxRNC(MinMaxRNCConfig(16, 2, d_state=8, **kw)).eval()
                sa, sb = a.state_dict(), b.state_dict()
                self.assertEqual(list(sa), list(sb))
                self.assertTrue(all(torch.equal(sa[k], sb[k]) for k in sa))
                u = torch.randn(2, 9, 16)
                with torch.no_grad():
                    torch.testing.assert_close(a(u, 4), b(u, 4), rtol=0, atol=0)
                self.assertEqual(_aten_ops(lambda x: a(x, 4), u), _aten_ops(lambda x: b(x, 4), u))

    def test_checkpoints_interchangeable(self):
        a = MinMaxRNC(MinMaxRNCConfig(16, 2, scalar_neurons=8))
        b = MinMaxRNC(MinMaxRNCConfig(16, 2, 8))
        b.load_state_dict(a.state_dict(), strict=True)
        a.load_state_dict(b.state_dict(), strict=True)

    def test_num_scalar_neurons_and_arrays(self):
        for cfg in (MinMaxRNCConfig(16, 1, scalar_neurons=8), MinMaxRNCConfig(16, 1, d_state=8)):
            with self.subTest(cfg=cfg):
                self.assertEqual(cfg.num_scalar_neurons, 8)
                self.assertEqual(cfg.neuron_arrays, (NeuronArray(1, 8),))
                self.assertIsInstance(MinMaxRNC(cfg).layers[0].neuron, MinMaxNeuron)

    def test_both_set_is_rejected(self):
        with self.assertRaises(ValueError):
            MinMaxRNCConfig(16, 1, d_state=8, scalar_neurons=8)

    def test_negative_or_missing_is_rejected(self):
        with self.assertRaises(ValueError):
            MinMaxRNCConfig(16, 1, scalar_neurons=-1)
        with self.assertRaises(ValueError):
            MinMaxRNCConfig(16, 1)                    # no neurons at all

    def test_scalar_neurons_only_with_arrays(self):
        cfg = MinMaxRNCConfig(16, 1, scalar_neurons=4, neurons=[(3, 2)])
        self.assertEqual(cfg.state_size_per_layer, 4 + 6)
        out = MinMaxRNC(cfg)(torch.randn(1, 5, 16), 5)
        self.assertEqual(out.shape, (1, 5, 16))

    def test_lm_wrapper(self):
        lm = MinMaxRNC_LM(20, MinMaxRNCLMConfig(backbone=MinMaxRNCConfig(16, 2, scalar_neurons=8)))
        self.assertEqual(lm(torch.randint(0, 20, (2, 5)), 5).shape, (2, 5, 20))


# ---------------------------------------------------------------------------
# Transition matrix R_t
# ---------------------------------------------------------------------------

class TestTransitionMatrix(unittest.TestCase):

    def test_r_projection_shape(self):
        arr = _array(d_model=10, d_state=3, num_neurons=5)
        self.assertEqual((arr.r.in_features, arr.r.out_features), (10, 5 * 3 * 3))
        for name in ('k', 'beta'):
            self.assertFalse(hasattr(arr, name))

    def test_matrix_path_matches_sequential_recurrence(self):
        """The array's states equal a step-by-step loop of
        x_{t+1} = max(R_{t+1} ⊗ x_t, s_{t+1}) with R_{t+1} = reshape(W_r u_{t+1})."""
        torch.manual_seed(0)
        d_model, d, K, B, T = 6, 3, 2, 2, 5
        arr = _array(d_model=d_model, d_state=d, num_neurons=K).eval()
        u = torch.randn(B, T, d_model)
        _, last = arr(u, arr.initial_state)
        R = arr.r(u).view(B, T, K, d, d)
        s = arr.s(u).view(B, T, K, d)
        x = torch.zeros(B, K, d)
        for t in range(T):
            x = mm_apply_matrix(R[:, t], s[:, t], x)
        torch.testing.assert_close(last, x.reshape(B, K * d), atol=1e-6, rtol=1e-6)

    def test_gradient_flow(self):
        arr = _array(d_state=3, num_neurons=2)
        out, _ = arr(torch.randn(2, 5, 8), arr.initial_state)
        out.sum().backward()
        for name, p in [('s', arr.s.weight), ('r', arr.r.weight), ('o', arr.o.weight)]:
            self.assertIsNotNone(p.grad, f"{name}.weight has no gradient")
            self.assertGreater(p.grad.abs().sum().item(), 0, f"{name}.weight gradient is zero")

    def test_n_h_and_normalize_keys_are_gone(self):
        with self.assertRaises(TypeError):
            MinMaxNeuronArrayConfig(_num_blocks=1, d_model=8, d_state=3, n_h=2)
        with self.assertRaises(TypeError):
            MinMaxRNCConfig(d_model=8, n_layers=1, neurons=[(3, 2, 0)])     # old (d, K, n_h) triples


# ---------------------------------------------------------------------------
# Output-projection init scaling
# ---------------------------------------------------------------------------

class TestOutputInitScaling(unittest.TestCase):
    """Every array's output projection is wang_init_-scaled with dim = the
    layer's total state size Σ num_neurons · d_state: the layer's summed
    contribution stays O(1) however its state is split across arrays."""

    def test_default_is_own_state_size(self):
        torch.manual_seed(0)
        own = _array(d_model=16, d_state=4, num_neurons=128)                    # D = 512
        torch.manual_seed(0)
        explicit = _array(d_model=16, d_state=4, num_neurons=128, _state_size_in_layer=512)
        self.assertTrue(torch.equal(own.o.weight, explicit.o.weight))

    def test_wang_init_std_scales_with_state_size_in_layer(self):
        torch.manual_seed(0)
        n1 = MinMaxNeuron(MinMaxNeuronConfig(_num_blocks=1, d_model=16, d_state=512))
        n4 = MinMaxNeuron(MinMaxNeuronConfig(_num_blocks=1, d_model=16, d_state=512, _state_size_in_layer=4 * 512))
        ratio = n4.o.weight.std().item() / n1.o.weight.std().item()
        self.assertAlmostEqual(ratio, 1.0 / math.sqrt(4), delta=0.05)

    def test_layer_passes_total_state_size(self):
        layer = _layer(10, d_state=4, neurons=[(3, 2), (2, 5)])
        expected = 4 + 3 * 2 + 2 * 5
        for arr in layer.neuron_array_list():
            self.assertEqual(arr.array_cfg._state_size_in_layer, expected)

    def test_layer_output_scale_invariant_to_state_split(self):
        d_model, B, T = 32, 8, 16
        torch.manual_seed(0)
        one = _layer(d_model, neurons=[(2, 16)], n_layers=4)
        torch.manual_seed(0)
        two = _layer(d_model, d_state=8, neurons=[(2, 4), (4, 4)], n_layers=4)
        u = torch.randn(B, T, d_model)
        std_one = (one(u, one.initial_state)[0] - u).std().item()
        std_two = (two(u, two.initial_state)[0] - u).std().item()
        self.assertAlmostEqual(std_one, std_two, delta=0.3 * max(std_one, std_two))


# ---------------------------------------------------------------------------
# MinMaxRNCConfig.neurons
# ---------------------------------------------------------------------------

class TestNeuronsConfig(unittest.TestCase):

    def test_tuple_list_and_array_inputs_normalise_identically(self):
        a = MinMaxRNCConfig(16, 1, 0, neurons=[(3, 2)])
        b = MinMaxRNCConfig(16, 1, 0, neurons=[[3, 2]])                    # YAML-style lists
        c = MinMaxRNCConfig(16, 1, 0, neurons=(NeuronArray(3, 2),))
        self.assertEqual(a.neurons, (NeuronArray(3, 2),))
        self.assertEqual(a, b)
        self.assertEqual(a, c)

    def test_config_is_hashable(self):
        self.assertEqual(
            hash(MinMaxRNCConfig(16, 1, 0, neurons=[[3, 2]])),
            hash(MinMaxRNCConfig(16, 1, 0, neurons=[(3, 2)])),
        )

    def test_rejects_bad_entries(self):
        for bad in ([(0, 2)], [(3, 0)], [(3, 2, 0)], [()], [3]):
            with self.subTest(entry=bad), self.assertRaises((ValueError, TypeError)):
                MinMaxRNCConfig(16, 1, 4, neurons=bad)

    def test_rejects_no_neurons(self):
        with self.assertRaises(ValueError):
            MinMaxRNCConfig(d_model=16, n_layers=1)
        with self.assertRaises(ValueError):
            MinMaxRNCConfig(16, 1, 0)

    def test_rejects_negative_d_state(self):
        with self.assertRaises(ValueError):
            MinMaxRNCConfig(16, 1, -1)

    def test_any_state_dimension_is_valid_without_warnings(self):
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            MinMaxRNC(MinMaxRNCConfig(16, 1, 4, neurons=[(1, 3), (2, 3), (4, 1)]))

    def test_same_d_state_in_several_arrays_allowed(self):
        model = MinMaxRNC(MinMaxRNCConfig(16, 1, 0, neurons=[(3, 2), (3, 4)]))
        self.assertEqual(len(model.layers[0].neuron_arrays), 2)

    def test_neuron_arrays_and_sizes(self):
        cfg = MinMaxRNCConfig(16, 1, 8, neurons=[(3, 2), (2, 1)])
        self.assertEqual(cfg.neuron_arrays, (NeuronArray(1, 8), NeuronArray(3, 2), NeuronArray(2, 1)))
        self.assertEqual(cfg.num_neurons_per_layer, 8 + 2 + 1)
        self.assertEqual(cfg.state_size_per_layer, 8 + 6 + 2)
        self.assertEqual(MinMaxRNCConfig(16, 1, 0, neurons=[(3, 16)]).neuron_arrays, (NeuronArray(3, 16),))

    def test_layer_config_rejects_no_neurons(self):
        with self.assertRaises(ValueError):
            MinMaxLayerConfig(
                neuron=None,
                conv=BasicConvConfig(embedding_dim=8),
                d_model=8,
                feedforward=FeedForwardConfig(_num_blocks=1, ffn_type='basic'),
            )


# ---------------------------------------------------------------------------
# Layers and models
# ---------------------------------------------------------------------------

class TestLayersAndModels(unittest.TestCase):

    def test_layer_structure_and_state(self):
        layer = _layer(12, d_state=4, neurons=[(3, 2)])
        self.assertIsInstance(layer.neuron, MinMaxNeuron)
        self.assertEqual(len(layer.neuron_arrays), 1)
        self.assertEqual([type(arr) for arr in layer.neuron_array_list()], [MinMaxNeuron, MinMaxNeuronArray])
        state = layer.initial_state
        self.assertEqual(list(state.keys()), ['neuron', 'conv', 'neuron_arrays'])
        out, new_state = layer(torch.randn(2, 7, 12), state)
        self.assertEqual(out.shape, (2, 7, 12))
        self.assertEqual(list(new_state.keys()), ['conv', 'neuron', 'neuron_arrays'])
        self.assertEqual(new_state['neuron'].shape, (2, 4))
        self.assertEqual([s.shape for s in new_state['neuron_arrays']], [(2, 6)])

    def test_layer_without_d_state_shorthand(self):
        layer = _layer(10, neurons=[(4, 1), (5, 1)])
        self.assertIsNone(layer.neuron)
        self.assertNotIn('neuron', layer.initial_state)
        self.assertEqual(sorted(arr.d for arr in layer.neuron_arrays), [4, 5])
        out, new_state = layer(torch.randn(2, 5, 10), layer.initial_state)
        self.assertEqual(out.shape, (2, 5, 10))
        self.assertNotIn('neuron', new_state)

    def test_layer_sums_array_outputs(self):
        torch.manual_seed(0)
        layer = _layer(8, d_state=3, neurons=[(2, 2)], norm='none').eval()
        u = torch.randn(2, 6, 8)
        out, _ = layer(u, layer.initial_state)
        conv, _ = layer.conv(u, layer.initial_state['conv'])
        neuron_in = u + conv + layer.ffn(u + conv)
        expected = u + sum(arr(neuron_in, arr.initial_state)[0] for arr in layer.neuron_array_list())
        torch.testing.assert_close(out, expected, atol=1e-6, rtol=1e-6)

    def test_gradient_flow(self):
        layer = _layer(8, d_state=3, neurons=[(2, 2), (3, 1), (1, 4)])
        out, _ = layer(torch.randn(2, 6, 8), layer.initial_state)
        out.sum().backward()
        for name, p in layer.named_parameters():
            if p.requires_grad and 'neuron' in name and name.endswith('weight'):
                self.assertIsNotNone(p.grad, f"{name} has no gradient")

    def test_chunked_forward_matches_single_pass(self):
        for d_state, arrays in [(0, [(3, 2)]), (4, [(3, 2), (2, 1), (1, 3)])]:
            with self.subTest(d_state=d_state, arrays=arrays):
                torch.manual_seed(0)
                model = MinMaxRNC(MinMaxRNCConfig(12, 2, d_state, neurons=arrays)).eval()
                u = torch.randn(2, 17, 12)
                with torch.no_grad():
                    torch.testing.assert_close(model(u, 5), model(u, 17), atol=1e-5, rtol=1e-5)

    def test_state_continuation_across_calls(self):
        torch.manual_seed(0)
        model = MinMaxRNC(MinMaxRNCConfig(12, 2, 4, neurons=[(3, 2)])).eval()
        u = torch.randn(2, 10, 12)
        with torch.no_grad():
            full = model(u, 10)
            y1, st = model(u[:, :6], 6, return_state=True)
            y2 = model(u[:, 6:], 4, state=st)
        torch.testing.assert_close(torch.cat([y1, y2], dim=1), full, atol=1e-5, rtol=1e-5)

    def test_train_init_makes_every_initial_state_learnable(self):
        model = MinMaxRNC(MinMaxRNCConfig(8, 1, 3, neurons=[(3, 2)], train_init=True))
        for arr in model.layers[0].neuron_array_list():
            self.assertTrue(arr.initial_state.requires_grad)

    def test_lm_with_neuron_arrays(self):
        model = MinMaxRNC_LM(20, MinMaxRNCLMConfig(backbone=MinMaxRNCConfig(16, 2, 0, neurons=[(3, 4)])))
        self.assertEqual(model(torch.randint(0, 20, (2, 9)), 4).shape, (2, 9, 20))

    def test_state_dict_round_trip_and_names(self):
        torch.manual_seed(0)
        cfg = MinMaxRNCConfig(12, 2, 4, neurons=[(3, 2), (2, 1)])
        a = MinMaxRNC(cfg).eval()
        torch.manual_seed(1)
        b = MinMaxRNC(cfg).eval()
        b.load_state_dict(a.state_dict(), strict=True)
        u = torch.randn(1, 7, 12)
        with torch.no_grad():
            torch.testing.assert_close(a(u, 7), b(u, 7))
        keys = list(a.state_dict())
        self.assertTrue(any(k.startswith('layers.0.neuron.') for k in keys))
        self.assertTrue(any(k.startswith('layers.0.neuron_arrays.1.') for k in keys))


if __name__ == '__main__':
    unittest.main()
