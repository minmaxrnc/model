<!--
SPDX-FileCopyrightText: 2026 Alessandro Ronca
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
-->

# MinMax RNCs — Upgrading from 0.3.0

[Index](README.md) · [Mathematical foundations](foundations.md) · [Architecture](architecture.md) · [Configuration](configuration.md) · [Numerical precision](precision.md) · [Upgrading from 0.3.0](upgrading.md)

---

0.4.0 is backward compatible with 0.3.0; no code changes are needed. This page collects everything
that relates 0.4.0 to 0.3.0; see [CHANGELOG.md](../CHANGELOG.md) for the full list of changes.

## Same API

- Every 0.3.0 class, config field and argument keeps its name, position and meaning, including
  positional [`MinMaxRNCConfig(d_model, n_layers, d_state)`](../src/minmaxrnc/minmax_rnc.py).
- The fields new in 0.4.0 (`scalar_neurons`, `neurons`, `recurrence_dtype`, `amp_dtype`) are appended
  after the 0.3.0 fields, so positional construction is unchanged, and their defaults
  (`scalar_neurons=0`, `neurons=()`, `recurrence_dtype=None`, `amp_dtype=None`) reproduce the 0.3.0
  behaviour.
- **`d_state` is the legacy name of `scalar_neurons`.** It was 0.3.0's way of setting a layer's
  neurons: `d_state=N` and `scalar_neurons=N` build exactly the same model; set at most one of
  them. The presets (`small`, `medium`, `large`) still set `d_state`, so `preset.d_state` keeps its
  0.3.0 value; `preset.num_scalar_neurons` gives the count whichever field was used.
- **[`MinMaxNeuron`](../src/minmaxrnc/minmax_neuron.py)** is still the class of 0.3.0's MinMax Neuron. With `d_state = N` it is the array
  of `N` scalar-state neurons, now a special case of [`MinMaxNeuronArray`](../src/minmaxrnc/minmax_neuron.py), which implements neurons
  of any state dimension. It is how a layer holds its `scalar_neurons` (attribute `neuron`, the
  0.3.0 name).

## Same weights

With the defaults, a model has the same parameter names, shapes and seeded initial values as in
0.3.0, so 0.3.0 checkpoints load with `load_state_dict(strict=True)` and no key remapping.

## Same outputs and computation

- Outputs and recurrent states are identical to 0.3.0, in fp32 and under a caller-side
  `torch.autocast`.
- For scalar-state neurons, every parameter, tensor and operation is exactly that of 0.3.0's MinMax
  Neuron: the forward pass runs the same sequence of PyTorch operations, so speed and memory use
  are unchanged.
- The test suite checks all of this against reference values recorded with 0.3.0
  (`tests/test_backward_compat.py`).

## Precision settings and 0.3.0

- `amp_dtype=None` (the default) leaves autocast to the caller, as in 0.3.0.
- `recurrence_dtype=None` (the default) does no casting, as in 0.3.0. Under an outer bf16 autocast
  this means the prefix composition runs in bf16 while the fp32 initial state promotes the final
  application to fp32 — the mixed, implicit precision of 0.3.0. A pinned `recurrence_dtype` makes the
  choice explicit (see [Numerical precision](precision.md)).

## Recurrent state format

Per layer it is still `{'neuron': …, 'conv': …}`; models with `neurons` arrays add a
`'neuron_arrays': [...]` entry (one `(B, num_neurons · d_state)` state per array), and models
without scalar neurons have no `'neuron'` entry.

## Small behaviour changes

Only for configs 0.3.0 rejected anyway: `d_state` now defaults to 0, so leaving it out raises a
`ValueError` (instead of a `TypeError`) unless `scalar_neurons` or `neurons` is given; invalid
`recurrence_dtype`/`amp_dtype` values raise `ValueError`.

## From the matrix-neuron experiment branch

Rename config keys `diagonal_d_state → scalar_neurons` and `d_states → neurons`, turning each
`(d_state, num_neurons, n_h)` triple into a `(d_state, num_neurons)` pair. Only `n_h = 0` is
supported (the rank-1-product parametrisation of `n_h > 0`, and with it `normalize_keys`, was
dropped).

That branch's checkpoints use different parameter names and do not load directly, but they convert
with a key renaming: `layers.i.neurons.j.*` becomes `layers.i.neuron_arrays.j.*`, and each array's
`_initial_state` is flattened from shape `(num_neurons, d_state)` to `(num_neurons · d_state,)`:

```python
import re

def convert_branch_state_dict(sd):
    out = {}
    for k, v in sd.items():
        k = re.sub(r'\.neurons\.(\d+)\.', r'.neuron_arrays.\1.', k)
        if k.endswith('._initial_state') and '.neuron_arrays.' in k and v.dim() == 2:
            v = v.reshape(-1)
        out[k] = v
    return out

model.load_state_dict(convert_branch_state_dict(old_state_dict), strict=True)
```

This was verified on S3 checkpoints of models with neuron arrays only (no scalar-state neurons):
after conversion, the logits are identical to the branch's. It has not been checked for models
that also have scalar-state neurons.
