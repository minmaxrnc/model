# SPDX-FileCopyrightText: 2026 Alessandro Ronca
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""
Backward-compatibility tests: 0.4.0 with default settings must behave
exactly like 0.3.0.

The reference values in tests/fixtures/v0_3_0_reference.pt were produced by
tests/fixtures/make_v0_3_0_reference.py running against a pristine 0.3.0
checkout (commit 6ea620f). They cover several configurations and record the
seeded initial parameters, fp32 outputs and final state for a chunked forward
pass, the same under an outer CPU bf16 autocast, the forward pass's sequence
of aten operations, and the preset parameter counts.
"""

import inspect
import os
import platform
import unittest
import warnings

import torch

import minmaxrnc
from minmaxrnc.minmax_rnc     import MinMaxRNC, MinMaxRNCConfig
from minmaxrnc.minmax_rnc_lm  import MinMaxRNC_LM, MinMaxRNCLMConfig
from minmaxrnc.minmax_neuron  import MinMaxNeuron, MinMaxNeuronConfig
from minmaxrnc.minmax_layer   import MinMaxLayerConfig


FIXTURE = os.path.join(os.path.dirname(__file__), 'fixtures', 'v0_3_0_reference.pt')
REF = torch.load(FIXTURE, weights_only=False)
META = REF['meta']


def _build(kind, kwargs):
    """Rebuild a model exactly as the fixture script did (same seed, same API)."""
    torch.manual_seed(META['seed'])
    backbone = MinMaxRNCConfig(**kwargs)
    if kind == 'lm':
        return MinMaxRNC_LM(META['vocab'], MinMaxRNCLMConfig(backbone=backbone))
    return MinMaxRNC(backbone)


def trace_aten_ops(model, inp, unroll):
    """Identical copy of tests/fixtures/make_v0_3_0_reference.py's tracer."""
    from torch.fx.experimental.proxy_tensor import make_fx
    with torch.no_grad():
        graph = make_fx(lambda x: model(x, unroll, return_state=True))(inp)
    return [str(n.target) for n in graph.graph.nodes if n.op == 'call_function']


def _same_torch_minor(version_a: str, version_b: str) -> bool:
    return version_a.split('.')[:2] == version_b.split('.')[:2]


def _same_environment() -> bool:
    """True when running where the fixture was recorded: same torch version, OS
    and CPU architecture. Only there are seeded initialisation and fp32/bf16
    kernels expected to reproduce bit for bit; PyTorch does not guarantee it
    across versions or platforms."""
    return (
        torch.__version__ == META['torch_version']
        and platform.system() == META.get('platform_system')
        and platform.machine() == META.get('platform_machine')
    )


SAME_ENV = _same_environment()
ENV_NOTE = (
    f"fixture recorded with torch {META['torch_version']} on "
    f"{META.get('platform_system')}/{META.get('platform_machine')}; running torch "
    f"{torch.__version__} on {platform.system()}/{platform.machine()}"
)
# Elsewhere: tolerances for float rounding of a different BLAS / random-number
# implementation (fp32) and of different bf16 kernels (bf16, ~3 significant
# digits). The exact same-environment comparison of 0.3.0 against 0.4.0 is
# done by the validation notebook, which runs both versions side by side.
FP32_TOL = dict(atol=1e-6, rtol=1e-5) if SAME_ENV else dict(atol=1e-4, rtol=1e-4)
BF16_TOL = dict(atol=1e-6, rtol=1e-5) if SAME_ENV else dict(atol=5e-2, rtol=5e-2)


def _assert_state_equal(test, actual, expected, atol=1e-6, rtol=1e-5):
    """Compare recurrent states: same nested structure, same keys, close values."""
    test.assertEqual(type(actual), type(expected))
    if torch.is_tensor(expected):
        test.assertEqual(actual.shape, expected.shape)
        test.assertEqual(actual.dtype, expected.dtype)
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    elif isinstance(expected, dict):
        test.assertEqual(list(actual.keys()), list(expected.keys()))
        for k in expected:
            _assert_state_equal(test, actual[k], expected[k], atol, rtol)
    else:
        test.assertEqual(len(actual), len(expected))
        for a, e in zip(actual, expected):
            _assert_state_equal(test, a, e, atol, rtol)


class TestV030ReferenceBehaviour(unittest.TestCase):
    """Replays every recorded 0.3.0 configuration against the current code."""

    def _each(self):
        for name, rec in REF['configs'].items():
            with self.subTest(config=name):
                yield name, rec

    def test_seeded_init_matches_v030(self):
        """Same seed → same parameters, names and order as 0.3.0: bit for bit
        in the fixture's environment, up to float rounding elsewhere."""
        for _, rec in self._each():
            model = _build(rec['kind'], rec['kwargs'])
            sd = model.state_dict()
            self.assertEqual(list(sd.keys()), list(rec['state_dict'].keys()))
            for k, v in rec['state_dict'].items():
                self.assertEqual(sd[k].shape, v.shape, k)
                if SAME_ENV:
                    self.assertTrue(torch.equal(sd[k], v), f"parameter {k} differs from 0.3.0")
                else:
                    torch.testing.assert_close(sd[k], v, atol=1e-6, rtol=1e-5,
                                               msg=lambda m: f"parameter {k}: {m} ({ENV_NOTE})")

    def test_parameter_count_unchanged(self):
        for _, rec in self._each():
            model = _build(rec['kind'], rec['kwargs'])
            self.assertEqual(sum(p.numel() for p in model.parameters()), rec['num_params'])

    def test_v030_checkpoint_loads_strictly(self):
        """A 0.3.0 state_dict loads with strict=True into a model built with
        a different seed, with no key remapping."""
        for _, rec in self._each():
            torch.manual_seed(META['seed'] + 1)
            model = _build(rec['kind'], rec['kwargs'])
            result = model.load_state_dict(rec['state_dict'], strict=True)
            self.assertEqual(result.missing_keys, [])
            self.assertEqual(result.unexpected_keys, [])

    def _model_with_v030_weights(self, rec):
        """The recorded 0.3.0 weights, so that only the computation is compared."""
        model = _build(rec['kind'], rec['kwargs']).eval()
        model.load_state_dict(rec['state_dict'], strict=True)
        return model

    def test_fp32_outputs_and_state_match(self):
        for _, rec in self._each():
            model = self._model_with_v030_weights(rec)
            with torch.no_grad():
                y, state = model(rec['input'], META['unroll'], return_state=True)
            torch.testing.assert_close(y, rec['output'], **FP32_TOL)
            _assert_state_equal(self, state, rec['state'], **FP32_TOL)

    def test_outer_bf16_autocast_outputs_and_state_match(self):
        """Default scan_dtype='ambient' / amp_dtype=None leave the 0.3.0
        mixed-precision behaviour under a caller's autocast untouched,
        including the dtype of the carried state."""
        for _, rec in self._each():
            model = self._model_with_v030_weights(rec)
            with torch.no_grad(), torch.autocast(device_type='cpu', dtype=torch.bfloat16):
                y, state = model(rec['input'], META['unroll'], return_state=True)
            self.assertEqual(y.dtype, rec['output_bf16'].dtype)
            torch.testing.assert_close(y, rec['output_bf16'], **BF16_TOL)
            _assert_state_equal(self, state, rec['state_bf16'], **BF16_TOL)

    def test_seeded_model_reproduces_outputs(self):
        """End to end: a model built from the same seed (not loaded from the
        recorded weights) reproduces the 0.3.0 outputs."""
        for _, rec in self._each():
            model = _build(rec['kind'], rec['kwargs']).eval()
            with torch.no_grad():
                y = model(rec['input'], META['unroll'])
            torch.testing.assert_close(y, rec['output'], **FP32_TOL)

    @unittest.skipUnless(
        _same_torch_minor(torch.__version__, META['torch_version']),
        f"aten op sequences were recorded with torch {META['torch_version']}; "
        f"traces are only comparable within the same torch minor version",
    )
    def test_forward_graph_identical_to_v030(self):
        """The forward pass runs exactly the same aten operations, in the same
        order, as 0.3.0 — no extra reshapes, casts or kernels in the
        scalar-state path."""
        for _, rec in self._each():
            model = _build(rec['kind'], rec['kwargs']).eval()
            self.assertEqual(trace_aten_ops(model, rec['input'], META['unroll']), rec['aten_ops'])

    def test_preset_parameter_counts_unchanged(self):
        for preset, n in REF['presets'].items():
            with self.subTest(preset=preset):
                model = MinMaxRNC(getattr(MinMaxRNCConfig, preset)())
                self.assertEqual(sum(p.numel() for p in model.parameters()), n)


class TestV030Api(unittest.TestCase):
    """The 0.3.0 public API is still accepted, with the same meaning."""

    def test_positional_config_construction(self):
        cfg = MinMaxRNCConfig(90, 2, 40)
        self.assertEqual((cfg.d_model, cfg.n_layers, cfg.d_state), (90, 2, 40))
        self.assertEqual(cfg, MinMaxRNCConfig(d_model=90, n_layers=2, d_state=40))

    def test_v030_fields_keep_their_order(self):
        """Fields added in 0.4.0 are appended after every 0.3.0 field."""
        v030_fields = [
            'd_model', 'n_layers', 'd_state', 'norm', 'postlayers_norm',
            'ffn_type', 'ffn_proj_factor', 'ffn_act_fn', 'ffn_dropout', 'ffn_init',
            'output_gate', 'train_init', 'neuron_dropout', 's_r_init',
            'conv_type', 'conv_init_val', 'prelayers_dropout', 'use_postlayers_ffn',
        ]
        fields = list(MinMaxRNCConfig.__dataclass_fields__)
        self.assertEqual(fields[:len(v030_fields)], v030_fields)

        v030_neuron = ['_num_blocks', 'd_model', 'd_state', 'dropout',
                       'train_init', 'output_gate', 's_r_init']
        self.assertEqual(list(MinMaxNeuronConfig.__dataclass_fields__)[:len(v030_neuron)], v030_neuron)

        v030_layer = ['neuron', 'conv', 'd_model', 'first_in_dropout', 'feedforward', 'norm']
        self.assertEqual(list(MinMaxLayerConfig.__dataclass_fields__)[:len(v030_layer)], v030_layer)

    def test_new_settings_default_to_v030_behaviour(self):
        cfg = MinMaxRNCConfig(16, 1, 8)
        self.assertEqual(cfg.neurons, ())
        self.assertEqual(cfg.scan_dtype, 'ambient')
        self.assertIsNone(cfg.amp_dtype)

    def test_presets_unchanged(self):
        self.assertEqual(MinMaxRNCConfig.small(),  MinMaxRNCConfig(d_model=90,  n_layers=2,  d_state=40))
        self.assertEqual(MinMaxRNCConfig.medium(), MinMaxRNCConfig(d_model=512, n_layers=8,  d_state=512))
        self.assertEqual(MinMaxRNCConfig.large(),  MinMaxRNCConfig(d_model=728, n_layers=12, d_state=1456))

    def test_minmax_neuron_is_still_the_diagonal_neuron(self):
        neuron = MinMaxNeuron(MinMaxNeuronConfig(_num_blocks=1, d_model=8, d_state=5))
        self.assertEqual(neuron.r.out_features, 5)          # r is a vector, not a matrix
        self.assertEqual(tuple(neuron.initial_state.shape), (5,))
        y, x = neuron(torch.randn(2, 3, 8), neuron.initial_state)
        self.assertEqual(tuple(y.shape), (2, 3, 8))
        self.assertEqual(tuple(x.shape), (2, 5))

    def test_layer_attributes_and_state_structure(self):
        model = MinMaxRNC(MinMaxRNCConfig(16, 2, 8))
        for layer in model.layers:
            self.assertIsInstance(layer.neuron, MinMaxNeuron)
            self.assertEqual(len(layer.neuron_groups), 0)
        for layer_state in model.initial_state:
            self.assertEqual(list(layer_state.keys()), ['neuron', 'conv'])
        _, state = model(torch.randn(1, 4, 16), 4, return_state=True)
        for layer_state in state:
            self.assertEqual(list(layer_state.keys()), ['conv', 'neuron'])

    def test_v030_style_configs_raise_no_warnings(self):
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            for preset in ('small', 'medium', 'large'):
                getattr(MinMaxRNCConfig, preset)()
            MinMaxRNC(MinMaxRNCConfig(16, 2, 8))
            MinMaxRNC_LM(10, MinMaxRNCLMConfig(backbone=MinMaxRNCConfig(16, 2, 8)))

    def test_v030_public_names_still_exported(self):
        for name in ['MinMaxRNC', 'MinMaxRNCConfig', 'MinMaxRNC_LM', 'MinMaxRNCLMConfig', 'SRInitType']:
            self.assertIn(name, minmaxrnc.__all__)
            self.assertTrue(hasattr(minmaxrnc, name))

    def test_forward_signatures_unchanged(self):
        self.assertEqual(
            list(inspect.signature(MinMaxRNC.forward).parameters),
            ['self', 'u', 'unroll_steps', 'state', 'return_state'],
        )
        self.assertEqual(
            list(inspect.signature(MinMaxRNC_LM.forward).parameters),
            ['self', 'tokens', 'unroll_steps', 'state', 'return_state'],
        )


if __name__ == '__main__':
    unittest.main()
