<!--
SPDX-FileCopyrightText: 2026 Alessandro Ronca
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
-->

# MinMax RNCs — Upgrading from 0.3.0

[Index](model.md) · [Mathematical foundation](foundations.md) · [Architecture](architecture.md) · [Configuration and implementation](configuration.md) · [Upgrading from 0.3.0](upgrading.md)

0.4.0 is backward compatible with 0.3.0; no code changes are needed. This page collects everything
that relates 0.4.0 to 0.3.0; see [CHANGELOG.md](../CHANGELOG.md) for the full list of changes.

## Same API

- Every 0.3.0 class, config field and argument keeps its name, position and meaning, including
  positional `MinMaxRNCConfig(d_model, n_layers, d_state)`.
- The fields new in 0.4.0 (`scalar_neurons`, `neurons`, `recurrence_dtype`, `amp_dtype`) are appended
  after the 0.3.0 fields, so positional construction is unchanged, and their defaults
  (`scalar_neurons=0`, `neurons=()`, `recurrence_dtype=None`, `amp_dtype=None`) reproduce the 0.3.0
  behaviour.
- **`d_state` is the legacy name of `scalar_neurons`.** It was 0.3.0's way of setting a layer's
  neurons: `d_state=N` and `scalar_neurons=N` build exactly the same model; set at most one of
  them. The presets (`small`, `medium`, `large`) still set `d_state`, so `preset.d_state` keeps its
  0.3.0 value; `preset.num_scalar_neurons` gives the count whichever field was used.
- **`MinMaxNeuron`** is still the class of 0.3.0's MinMax Neuron. With `d_state = N` it is the group
  of `N` scalar-state neurons, now a special case of `MinMaxNeuronGroup`, which implements neurons
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
  choice explicit (see [Numerical precision](configuration.md#numerical-precision)).

## Recurrent state format

Per layer it is still `{'neuron': …, 'conv': …}`; models with `neurons` groups add a
`'neuron_groups': [...]` entry (one `(B, num_neurons · d_state)` state per group), and models
without scalar neurons have no `'neuron'` entry.

## Small behaviour changes

Only for configs 0.3.0 rejected anyway: `d_state` now defaults to 0, so leaving it out raises a
`ValueError` (instead of a `TypeError`) unless `scalar_neurons` or `neurons` is given; invalid
`recurrence_dtype`/`amp_dtype` values raise `ValueError`.

## From the matrix-neuron experiment branch

Rename config keys `diagonal_d_state → scalar_neurons` and `d_states → neurons`, turning each
`(d_state, num_neurons, n_h)` triple into a `(d_state, num_neurons)` pair. Only `n_h = 0` is
supported (the rank-1-product parametrisation of `n_h > 0`, and with it `normalize_keys`, was
dropped). That branch's checkpoints use different parameter names (`neurons.N.*`) and do not load
directly.
