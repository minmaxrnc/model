# SPDX-FileCopyrightText: 2026 Alessandro Ronca
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""
Backward-compatibility tests: 0.4.0 with default settings must behave
exactly like 0.3.0.

The reference values in tests/fixtures/v0_3_0_reference.pt were produced by
tests/fixtures/make_v0_3_0_reference.py running against a pristine 0.3.0
checkout (commit 6ea620f). They cover several configurations and record the
seeded initial parameters, fp32 outputs and final state for a chunked forward
pass, the same under an outer CPU bf16 autocast, and the preset parameter
counts.
"""

import inspect
import os
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

    def test_seeded_init_is_bitwise_identical(self):
        """Same seed → same parameters, same names, same order as 0.3.0."""
        for _, rec in self._each():
            model = _build(rec['kind'], rec['kwargs'])
            sd = model.state_dict()
            self.assertEqual(list(sd.keys()), list(rec['state_dict'].keys()))
            for k, v in rec['state_dict'].items():
                self.assertTrue(torch.equal(sd[k], v), f"parameter {k} differs from 0.3.0")

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

    def test_fp32_outputs_and_state_match(self):
        for _, rec in self._each():
            model = _build(rec['kind'], rec['kwargs']).eval()
            with torch.no_grad():
                y, state = model(rec['input'], META['unroll'], return_state=True)
            torch.testing.assert_close(y, rec['output'], atol=1e-6, rtol=1e-5)
            _assert_state_equal(self, state, rec['state'])

    def test_outer_bf16_autocast_outputs_and_state_match(self):
        """Default scan_dtype='ambient' / amp_dtype=None leave the 0.3.0
        mixed-precision behaviour under a caller's autocast untouched,
        including the dtype of the carried state."""
        for _, rec in self._each():
            model = _build(rec['kind'], rec['kwargs']).eval()
            with torch.no_grad(), torch.autocast(device_type='cpu', dtype=torch.bfloat16):
                y, state = model(rec['input'], META['unroll'], return_state=True)
            self.assertEqual(y.dtype, rec['output_bf16'].dtype)
            torch.testing.assert_close(y, rec['output_bf16'], atol=1e-6, rtol=1e-5)
            _assert_state_equal(self, state, rec['state_bf16'])

    def test_loaded_checkpoint_reproduces_outputs(self):
        """Loading the recorded weights (rather than re-seeding) also
        reproduces the 0.3.0 outputs."""
        for _, rec in self._each():
            torch.manual_seed(0)
            model = _build(rec['kind'], rec['kwargs']).eval()
            model.load_state_dict(rec['state_dict'])
            with torch.no_grad():
                y = model(rec['input'], META['unroll'])
            torch.testing.assert_close(y, rec['output'], atol=1e-6, rtol=1e-5)

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
        self.assertEqual(cfg.matrix_neurons, ())
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
            self.assertEqual(len(layer.matrix_neurons), 0)
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
