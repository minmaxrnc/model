<!--
SPDX-FileCopyrightText: 2026 Alessandro Ronca
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
-->

# MinMax RNCs — Configuration

[Index](README.md) · [Mathematical foundations](foundations.md) · [Architecture](architecture.md) · [Configuration](configuration.md) · [Numerical precision](precision.md) · [Upgrading from 0.3.0](upgrading.md)

---

This page lists every field of [`MinMaxRNCConfig`](../src/minmaxrnc/minmax_rnc.py) and
[`MinMaxRNCLMConfig`](../src/minmaxrnc/minmax_rnc_lm.py), and the preset
configurations.

All architecture hyperparameters are specified through a single flat
`MinMaxRNCConfig` dataclass.

## MinMaxRNCConfig

| Field | Type | Default | Description |
|---|---|---|---|
| `d_model` | `int` | — | Residual-stream / embedding width |
| `n_layers` | `int` | — | Number of MinMaxLayers |
| `d_state` | `int` | `0` | Legacy name of `scalar_neurons` (see [Upgrading from 0.3.0](upgrading.md)): `d_state=N` builds the same model as `scalar_neurons=N`. Set at most one of the two |
| `norm` | `str` | `'layernorm'` | In-layer pre-norm: `'layernorm'`, `'rmsnorm'`, `'none'` |
| `postlayers_norm` | `str` | `'layernorm'` | Norm applied after the final layer |
| `ffn_type` | `str` | `'gated'` | `'gated'` (ReGLU/SwiGLU) or `'basic'` (MLP) — see [Feed-forward Variants](architecture.md#feed-forward-variants) |
| `ffn_proj_factor` | `float` | `1.3` | FFN hidden-dim expansion factor |
| `ffn_act_fn` | `str` | `'relu'` | FFN activation: `'relu'`, `'relu^2'`, `'gelu'`, `'swish'`, `'sigmoid'`, `'selu'` |
| `ffn_dropout` | `float` | `0.1` | Dropout inside the FFN |
| `ffn_init` | `str` | `'scaled'` | FFN init: `'basic'` (PyTorch default) or `'scaled'` (small\_init + wang\_init) |
| `output_gate` | `bool` | `True` | Gate the neuron output by a learned projection of the input |
| `train_init` | `bool` | `False` | Make the initial hidden state $`x_0`$ a learned parameter |
| `neuron_dropout` | `float` | `0.0` | Dropout on the neuron input |
| `s_r_init` | `str` | `'small_init'` | Init scheme for s and r projections: `'small_init'`, `'kaiming'`, or `'asymmetric'` — see [Initialisation](architecture.md#initialisation) |
| `conv_type` | `str` | `'basic'` | `'basic'` (learned affine map of the previous and current token) or `'gated'` (per-feature gate between them) — see [Convolution Variants](architecture.md#convolution-variants) |
| `conv_init_val` | `float` | `0.0` | Initial gate logit for [`GatedConv`](../src/minmaxrnc/modules/gated_conv.py) |
| `prelayers_dropout` | `float` | `0.0` | FFN dropout override for the first layer only |
| `use_postlayers_ffn` | `bool` | `False` | Add an FFN after all layers |
| `scalar_neurons` | `int` | `0` | Number of scalar-state neurons per layer (shorthand for the array [`NeuronArray(1, scalar_neurons)`](../src/minmaxrnc/minmax_rnc.py)). Each layer needs `scalar_neurons` (or `d_state`) > 0 and/or a non-empty `neurons` |
| `neurons` | `tuple[NeuronArray, ...]` | `()` | Further neuron arrays per layer, of any state dimension: `NeuronArray(d_state, num_neurons=1)` or equivalent tuples/lists. Each layer holds an ordered list of neuron arrays: the `scalar_neurons` array first, then these in order |
| `recurrence_dtype` | `str \| None` | `None` | Recurrence precision: `None` (no casting), or pinned `'fp32'`, `'bf16'`, `'fp16'` — see [Numerical precision](precision.md) |
| `amp_dtype` | `str \| None` | `None` | Model-level autocast: `None` (caller decides), `'bf16'`, `'fp16'` |


## MinMaxRNCLMConfig

| Field | Type | Default | Description |
|---|---|---|---|
| `backbone` | [`MinMaxRNCConfig`](#minmaxrncconfig) | — | Backbone configuration |
| `head_dropout` | `float` | `0.0` | Dropout before the LM head |
| `tie_weights` | `bool` | `True` | Share embedding and LM-head weights |
| `output_gate` | `bool` | `True` | Gate each neuron output by $`\sigma(W_\mathrm{g} \cdot u_t + b_\mathrm{g})`$; always replaces `backbone.output_gate` |
| `conv_type` | `str` | `'basic'` | Short-range conv variant; always replaces `backbone.conv_type` |
| `ffn_dropout` | `float` | `0.1` | Dropout inside the FFN of every layer (except the first, which uses `backbone.prelayers_dropout`); always replaces `backbone.ffn_dropout` |

## Presets

`MinMaxRNCConfig.small()`, `medium()` and `large()` return configurations of three sizes. They set
only the width, the depth and the number of scalar-state neurons per layer; every other field keeps
its default.

| Preset | `d_model` | `n_layers` (default) | Scalar-state neurons per layer |
|---|---|---|---|
| `small(n_layers=2)` | 90 | 2 | 40 |
| `medium(n_layers=8)` | 512 | 8 | 512 |
| `large(n_layers=12)` | 728 | 12 | 1456 |

- **Depth.** The `n_layers` argument changes the number of layers and nothing else.
- **Other fields.** Configurations are frozen dataclasses, so other fields are changed by copying:
  `dataclasses.replace(MinMaxRNCConfig.large(), norm='rmsnorm')`. To add neuron arrays of larger
  state dimension, replace `neurons` the same way.
- **Neuron count.** The presets set the legacy field `d_state` (see
  [Upgrading from 0.3.0](upgrading.md)); `preset.num_scalar_neurons` gives the number of
  scalar-state neurons whichever field was used. To replace the scalar-state neurons, set
  `d_state=0` in the same `replace` call as `neurons` (setting `scalar_neurons` while `d_state` is
  also set raises a `ValueError`).
- **Language model.** [`MinMaxRNC_LM`](../src/minmaxrnc/minmax_rnc_lm.py) takes `conv_type`,
  `output_gate` and `ffn_dropout` from [`MinMaxRNCLMConfig`](../src/minmaxrnc/minmax_rnc_lm.py), replacing the backbone's values (see
  [MinMaxRNCLMConfig](#minmaxrnclmconfig)); set them there.

Parameter counts (the LM with the GPT-2 vocabulary of 50,257 tokens and tied embedding and output
weights):

| Preset | Backbone | LM | Backbone, `conv_type='gated'` | LM, `conv_type='gated'` |
|---|---|---|---|---|
| `small` | ~0.1 M | ~4.7 M | ~0.1 M | ~4.6 M |
| `medium` | ~20.8 M | ~46.6 M | ~16.6 M | ~42.4 M |
| `large` | ~88.6 M | ~125.2 M | ~75.9 M | ~112.5 M |

The gated convolution replaces each layer's `Linear(2·d_model, d_model)` with a per-feature gate,
hence the smaller counts. `large` with the gated convolution is the 112M language model that
reaches GPT-2's validation loss on OpenWebText.
