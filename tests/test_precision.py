# SPDX-FileCopyrightText: 2026 Alessandro Ronca
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""
Tests for the precision settings (new in 0.4.0):

- MinMaxRNCConfig.recurrence_dtype: None (default, = 0.3.0) | 'fp32' | 'bf16' | 'fp16'
- MinMaxRNCConfig.amp_dtype:  None (default, = 0.3.0) | 'bf16' | 'fp16'

The scan functions are wrapped with a spy recording the dtypes they receive
and whether autocast is active, for both neuron types. CPU only.
"""

import contextlib
import unittest
from unittest import mock

import torch

from minmaxrnc import minmax_scan
from minmaxrnc.minmax_rnc    import MinMaxRNC, MinMaxRNCConfig
from minmaxrnc.minmax_rnc_lm import MinMaxRNC_LM, MinMaxRNCLMConfig


DT = {'fp32': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}

# (name, kwargs) — the legacy d_state shorthand, scalar_neurons, an explicit
# scalar-state group, a group with d > 1, and a mix of all.
ARCHS = [
    ('d_state',        dict(d_state=6)),
    ('scalar_neurons', dict(scalar_neurons=6)),
    ('scalar',         dict(neurons=[(1, 6)])),
    ('matrix',         dict(neurons=[(3, 2)])),
    ('mixed',          dict(scalar_neurons=4, neurons=[(3, 2), (2, 1), (1, 3)])),
]


def _cpu_fp16_supported():
    try:
        with torch.autocast('cpu', dtype=torch.float16):
            torch.nn.functional.linear(torch.randn(2, 3), torch.randn(4, 3))
        torch.maximum(torch.randn(3).half(), torch.randn(3).half())
        return True
    except (RuntimeError, AssertionError):
        return False


FP16_OK = _cpu_fp16_supported()


def _model(recurrence_dtype=None, amp_dtype=None, arch=dict(d_state=6), seed=0):
    torch.manual_seed(seed)
    return MinMaxRNC(MinMaxRNCConfig(
        d_model=12, n_layers=2, recurrence_dtype=recurrence_dtype, amp_dtype=amp_dtype,
        ffn_dropout=0.0, **arch,
    ))


@contextlib.contextmanager
def _spy_scans():
    """Record (fn name, input dtypes, autocast enabled) for every scan call."""
    calls = []
    originals = {n: getattr(minmax_scan, n) for n in ('all_states', 'all_states_matrix')}

    def wrap(name):
        def spy(*args):
            calls.append((name, tuple(a.dtype for a in args), torch.is_autocast_enabled('cpu')))
            return originals[name](*args)
        return spy

    with mock.patch.object(minmax_scan, 'all_states', wrap('all_states')), \
         mock.patch.object(minmax_scan, 'all_states_matrix', wrap('all_states_matrix')):
        yield calls


def _outer(dtype):
    """Outer caller-side autocast (or nothing)."""
    if dtype is None:
        return contextlib.nullcontext()
    return torch.autocast('cpu', dtype=DT[dtype])


def _flat_states(state):
    out = []
    for layer in state:
        if 'neuron' in layer:
            out.append(layer['neuron'])
        out.extend(layer.get('neuron_groups', []))
    return out


class TestRecurrenceDtype(unittest.TestCase):

    def _run(self, model, outer=None, T=9, unroll=4):
        u = torch.randn(2, T, 12)
        with _spy_scans() as calls, _outer(outer):
            y, state = model(u, unroll, return_state=True)
        return y, state, calls

    def test_default_without_autocast_is_fp32(self):
        for name, arch in ARCHS:
            with self.subTest(arch=name):
                _, state, calls = self._run(_model(arch=arch))
                self.assertTrue(calls)
                for _, dtypes, ac in calls:
                    self.assertEqual(set(dtypes), {torch.float32})
                    self.assertFalse(ac)
                self.assertTrue(all(s.dtype == torch.float32 for s in _flat_states(state)))

    def test_default_under_outer_bf16_autocast_is_0_3_0_behaviour(self):
        """Projections arrive in bf16 and the fp32 x0 parameter is not cast:
        exactly the implicit mixed-precision behaviour of 0.3.0."""
        for name, arch in ARCHS:
            with self.subTest(arch=name):
                _, _, calls = self._run(_model(arch=arch), outer='bf16', T=4, unroll=4)  # one chunk
                for fn, dtypes, ac in calls:
                    self.assertTrue(ac, "the default must not disable the caller's autocast")
                    self.assertEqual(dtypes[-1], torch.float32, "x0 stays fp32 by default")
                    if fn == 'all_states':
                        self.assertEqual(dtypes[:2], (torch.bfloat16, torch.bfloat16))
                    else:
                        self.assertEqual(dtypes[1], torch.bfloat16)   # s

    def test_pinned_dtype_is_exact_whatever_the_surrounding_precision(self):
        for recurrence_dtype in ('fp32', 'bf16', 'fp16'):
            if recurrence_dtype == 'fp16' and not FP16_OK:
                continue
            for outer in (None, 'bf16'):
                for name, arch in ARCHS:
                    with self.subTest(recurrence_dtype=recurrence_dtype, outer=outer, arch=name):
                        _, state, calls = self._run(_model(recurrence_dtype, arch=arch), outer=outer)
                        self.assertTrue(calls)
                        for _, dtypes, ac in calls:
                            self.assertEqual(set(dtypes), {DT[recurrence_dtype]})
                            self.assertFalse(ac, "autocast must be disabled inside a pinned scan")
                        for s in _flat_states(state):
                            self.assertEqual(s.dtype, DT[recurrence_dtype], "carried state stays pinned")

    def test_default_and_fp32_identical_without_autocast(self):
        u = torch.randn(2, 9, 12)
        for name, arch in ARCHS:
            with self.subTest(arch=name):
                a = _model(None, arch=arch).eval()
                b = _model('fp32', arch=arch).eval()
                with torch.no_grad():
                    self.assertTrue(torch.equal(a(u, 4), b(u, 4)))

    def test_bf16_scan_close_to_fp32_and_output_dtype_restored(self):
        """Without autocast, a bf16-pinned scan still returns fp32 activations
        (cast back for the output projection), close to the fp32 result."""
        u = torch.randn(2, 9, 12)
        for name, arch in ARCHS:
            with self.subTest(arch=name):
                ref = _model('fp32', arch=arch).eval()
                low = _model('bf16', arch=arch).eval()
                with torch.no_grad():
                    y_ref, y_low = ref(u, 4), low(u, 4)
                self.assertEqual(y_low.dtype, torch.float32)
                torch.testing.assert_close(y_low, y_ref, atol=0.1, rtol=0.1)

    def test_pinned_state_continuation(self):
        """A pinned-dtype state returned by one call is accepted by the next
        and gives the same result as one call over the whole sequence."""
        u = torch.randn(2, 10, 12)
        for name, arch in ARCHS:
            with self.subTest(arch=name):
                model = _model('bf16', arch=arch).eval()
                with torch.no_grad():
                    full = model(u, 10)
                    y1, st = model(u[:, :6], 6, return_state=True)
                    y2 = model(u[:, 6:], 4, state=st)
                torch.testing.assert_close(torch.cat([y1, y2], 1), full, atol=1e-2, rtol=1e-2)

    def test_gradients_finite_in_every_mode(self):
        combos = [(s, a) for s in (None, 'fp32', 'bf16', 'fp16') for a in (None, 'bf16')]
        for recurrence_dtype, amp_dtype in combos:
            if recurrence_dtype == 'fp16' and not FP16_OK:
                continue
            for name, arch in ARCHS:
                with self.subTest(recurrence_dtype=recurrence_dtype, amp_dtype=amp_dtype, arch=name):
                    model = _model(recurrence_dtype, amp_dtype, arch=arch)
                    y = model(torch.randn(2, 9, 12), 4)
                    y.float().pow(2).mean().backward()
                    grads = [p.grad for p in model.parameters() if p.grad is not None]
                    self.assertTrue(grads)
                    self.assertTrue(all(torch.isfinite(g).all() for g in grads))


class TestAmpDtype(unittest.TestCase):

    def _scan_s_dtypes(self, model, outer=None):
        with _spy_scans() as calls, _outer(outer):
            model(torch.randn(2, 4, 12), 4)
        # s is argument 1 of both all_states(r, s, x0) and all_states_matrix(R, s, x0)
        return {dtypes[1] for _, dtypes, _ in calls}, {ac for _, _, ac in calls}

    def test_none_does_not_touch_autocast(self):
        self.assertEqual(self._scan_s_dtypes(_model(amp_dtype=None)), ({torch.float32}, {False}))
        self.assertEqual(self._scan_s_dtypes(_model(amp_dtype=None), outer='bf16'), ({torch.bfloat16}, {True}))

    def test_bf16_enables_autocast_without_outer_context(self):
        for name, arch in ARCHS:
            with self.subTest(arch=name):
                self.assertEqual(
                    self._scan_s_dtypes(_model(amp_dtype='bf16', arch=arch)),
                    ({torch.bfloat16}, {True}),
                )

    def test_model_amp_matches_outer_autocast(self):
        """amp_dtype='bf16' is the same computation as wrapping the call in
        torch.autocast('cpu', bfloat16) yourself."""
        u = torch.randn(2, 9, 12)
        for name, arch in ARCHS:
            with self.subTest(arch=name):
                inner = _model(amp_dtype='bf16', arch=arch).eval()
                outer = _model(amp_dtype=None, arch=arch).eval()
                with torch.no_grad():
                    y_inner = inner(u, 4)
                    with torch.autocast('cpu', dtype=torch.bfloat16):
                        y_outer = outer(u, 4)
                self.assertTrue(torch.equal(y_inner, y_outer))

    @unittest.skipUnless(FP16_OK, "CPU fp16 autocast not supported by this torch build")
    def test_explicit_amp_dtype_overrides_outer_autocast(self):
        self.assertEqual(
            self._scan_s_dtypes(_model(amp_dtype='fp16'), outer='bf16'),
            ({torch.float16}, {True}),
        )

    def test_autocast_scope_ends_with_forward(self):
        model = _model(amp_dtype='bf16')
        model(torch.randn(1, 3, 12), 3)
        self.assertFalse(torch.is_autocast_enabled('cpu'))

    def test_recipe_bf16_amp_with_fp32_scan(self):
        """The model_mixed recipe: projections in bf16, scan pinned to fp32."""
        with _spy_scans() as calls:
            _model('fp32', 'bf16', arch=ARCHS[-1][1])(torch.randn(2, 4, 12), 4)
        for _, dtypes, ac in calls:
            self.assertEqual(set(dtypes), {torch.float32})
            self.assertFalse(ac)

    def test_recipe_fully_bf16(self):
        """The experiments-bf16 'model/' recipe: everything, scan included, in bf16."""
        with _spy_scans() as calls:
            _model('bf16', 'bf16', arch=ARCHS[-1][1])(torch.randn(2, 4, 12), 4)
        for _, dtypes, _ in calls:
            self.assertEqual(set(dtypes), {torch.bfloat16})

    def test_lm_amp_covers_embedding_and_head(self):
        torch.manual_seed(0)
        lm = MinMaxRNC_LM(30, MinMaxRNCLMConfig(
            backbone=MinMaxRNCConfig(12, 2, 6, amp_dtype='bf16'), ffn_dropout=0.0,
        ))
        logits = lm(torch.randint(0, 30, (2, 7)), 4)
        self.assertEqual(logits.dtype, torch.bfloat16)        # head ran under autocast
        lm_ref = MinMaxRNC_LM(30, MinMaxRNCLMConfig(backbone=MinMaxRNCConfig(12, 2, 6), ffn_dropout=0.0))
        self.assertEqual(lm_ref(torch.randint(0, 30, (2, 7)), 4).dtype, torch.float32)

    def test_rejects_unknown_values(self):
        with self.assertRaises(ValueError):
            MinMaxRNCConfig(12, 1, 4, recurrence_dtype='fp64')
        with self.assertRaises(ValueError):
            MinMaxRNCConfig(12, 1, 4, recurrence_dtype='ambient')
        with self.assertRaises(TypeError):
            MinMaxRNCConfig(12, 1, 4, scan_dtype='fp32')
        with self.assertRaises(ValueError):
            MinMaxRNCConfig(12, 1, 4, amp_dtype='fp32')


if __name__ == '__main__':
    unittest.main()
