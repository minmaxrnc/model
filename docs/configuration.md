<!--
SPDX-FileCopyrightText: 2026 Alessandro Ronca
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
-->

# MinMax RNCs — Configuration and Implementation Reference

[Index](model.md) · [Mathematical foundation](foundations.md) · [Architecture](architecture.md) · [Configuration and implementation](configuration.md)

Configuration fields, numerical precision, initialisation and feed-forward variants. Notation as in
[Mathematical foundation](foundations.md).

## Configuration Reference

All architecture hyperparameters are specified through a single flat
`MinMaxRNCConfig` dataclass.

### MinMaxRNCConfig

| Field | Type | Default | Description |
|---|---|---|---|
| `d_model` | int | — | Residual-stream / embedding width |
| `n_layers` | int | — | Number of MinMaxLayers |
| `d_state` | int | `0` | Legacy name of `scalar_neurons` (the 0.3.0 field, kept for backward compatibility): `d_state=N` builds the same model as `scalar_neurons=N`. Set at most one of the two |
| `norm` | str | `'layernorm'` | In-layer pre-norm: `'layernorm'`, `'rmsnorm'`, `'none'` |
| `postlayers_norm` | str | `'layernorm'` | Norm applied after the final layer |
| `ffn_type` | str | `'gated'` | `'gated'` (ReGLU/SwiGLU) or `'basic'` (MLP) |
| `ffn_proj_factor` | float | `1.3` | FFN hidden-dim expansion factor |
| `ffn_act_fn` | str | `'relu'` | FFN activation: `'relu'`, `'relu^2'`, `'gelu'`, `'swish'`, `'sigmoid'`, `'selu'` |
| `ffn_dropout` | float | `0.1` | Dropout inside the FFN |
| `ffn_init` | str | `'scaled'` | FFN init: `'basic'` (PyTorch default) or `'scaled'` (small\_init + wang\_init) |
| `output_gate` | bool | `True` | Gate the neuron output by a learned projection of the input |
| `train_init` | bool | `False` | Make the initial hidden state $`x_0`$ a learned parameter |
| `neuron_dropout` | float | `0.0` | Dropout on the neuron input |
| `s_r_init` | str | `'small_init'` | Init scheme for s and r projections: `'small_init'`, `'kaiming'`, or `'asymmetric'` |
| `conv_type` | str | `'basic'` | `'gated'` (scalar gate) or `'basic'` (linear mix) |
| `conv_init_val` | float | `0.0` | Initial gate logit for GatedConv |
| `prelayers_dropout` | float | `0.0` | FFN dropout override for the first layer only |
| `use_postlayers_ffn` | bool | `False` | Add an FFN after all layers |
| `scalar_neurons` | int | `0` | *0.4.0.* Number of scalar-state neurons per layer (shorthand for the group `NeuronGroup(1, scalar_neurons)`). Each layer needs `scalar_neurons` (or `d_state`) > 0 and/or a non-empty `neurons` |
| `neurons` | tuple | `()` | *0.4.0.* Further neuron groups per layer, of any state dimension: `NeuronGroup(d_state, num_neurons=1)` or equivalent tuples/lists |
| `scan_dtype` | str | `'ambient'` | *0.4.0.* Scan precision: `'ambient'` (no casting, 0.3.0 behaviour), or pinned `'fp32'`, `'bf16'`, `'fp16'` — see [Numerical precision](#numerical-precision) |
| `amp_dtype` | str \| None | `None` | *0.4.0.* Model-level autocast: `None` (caller decides, 0.3.0 behaviour), `'bf16'`, `'fp16'` |

Fields new in 0.4.0 are appended after the 0.3.0 fields, so positional
construction is unchanged.

### MinMaxRNCLMConfig

| Field | Type | Default | Description |
|---|---|---|---|
| `backbone` | MinMaxRNCConfig | — | Backbone configuration |
| `head_dropout` | float | `0.0` | Dropout before the LM head |
| `tie_weights` | bool | `True` | Share embedding and LM-head weights |
| `output_gate` | bool | `True` | Gate each neuron output by $`\sigma(W_g \cdot u + b_g)`$; overrides `backbone.output_gate` |
| `conv_type` | `'basic'` \| `'gated'` | `'basic'` | Short-range conv variant; overrides `backbone.conv_type` |
| `ffn_dropout` | float | `0.1` | Dropout inside the FFN of every layer (except the first, which uses `backbone.prelayers_dropout`); overrides `backbone.ffn_dropout` |

### Preset factories

```python
MinMaxRNCConfig.small()    # d_model=90,  n_layers=2,  40 scalar-state neurons
MinMaxRNCConfig.medium()   # d_model=512, n_layers=8,  512 scalar-state neurons
MinMaxRNCConfig.large()    # d_model=728, n_layers=12, 1456 scalar-state neurons
```

The presets set the scalar-state neurons through the legacy field `d_state`, so
`preset.d_state` keeps its 0.3.0 value; `preset.num_scalar_neurons` gives the
count whichever field was used.


---

## Numerical precision

*New in 0.4.0.* Two independent settings control precision; both default to
the 0.3.0 behaviour.

**`amp_dtype`** — if set, `MinMaxRNC.forward` (and, for `MinMaxRNC_LM`, the
embedding and LM head as well) runs inside
`torch.autocast(device_type, dtype=amp_dtype)`. If `None`, the model does not
touch autocast, so a context opened by the caller applies as usual. An
explicit `amp_dtype` takes precedence over an outer autocast inside the
model. Loss scaling for fp16 (`torch.amp.GradScaler`) stays with the caller.

**`scan_dtype`** — the precision of the MinMax scan in every neuron:

| Value | Scan inputs | Autocast during scan | Carried state |
|---|---|---|---|
| `'ambient'` | as they arrive (no casting) | unchanged | whatever the scan produced |
| `'fp32'` / `'bf16'` / `'fp16'` | cast to that dtype | disabled | that dtype |

With a pinned value, the scan's outputs are cast back to the ambient compute
dtype before the output gate and projection.

Why this matters: the scan is a long chain of element-wise min/max
*selections*. It is memory-bandwidth bound (no tensor-core acceleration), so
its dtype's byte width largely determines its cost; but the min/max routing
also depends on fine distinctions between values, which bf16's 8-bit mantissa
can blur. Under an outer bf16 autocast, `'ambient'` (as in 0.3.0) composes the
prefixes in bf16 while the fp32 initial state promotes the final application
to fp32 — mixed and implicit. Pinning makes the choice explicit:

| Recipe | `amp_dtype` | `scan_dtype` |
|---|---|---|
| Full precision / 0.3.0 behaviour (default) | `None` | `'ambient'` |
| bf16 AMP, full-precision scan | `'bf16'` | `'fp32'` |
| Fully bf16, scan included | `'bf16'` | `'bf16'` |
| bf16 AMP, fp16 scan (10-bit mantissa, same width; untested) | `'bf16'` | `'fp16'` |

---

## Initialisers

Both functions initialise by sampling $`p \sim \mathcal{N}(0, \sigma^2)`$; the table gives $`\sigma`$.

| Function | $`\sigma`$ | Paper |
|---|---|---|
| `small_init_init_(p, dim)` | √(2 / (5·dim)) | Nguyen & Salazar, IWSLT 2019 |
| `wang_init_(p, dim, N)` | 2 / (N·√dim) | Radford et al. 2019 (GPT-2) + Nguyen & Salazar 2019; used in Beck et al. 2024 (xLSTM) |

`small_init_` is applied to all **input-side** projections: token embedding,
LM head (when untied), BasicConv, FFN up-projection, and the neuron $`W_s`$, $`W_r`$,
and output-gate $`W_g`$ projections.  It keeps activations small at initialisation
in deep networks.

`wang_init_` is applied to the two **residual output** projections: the
neuron output $`W_o`$ and the FFN down-projection.  It additionally divides by N
(number of residual blocks) so the total variance contributed by all residual
branches to the stream stays O(1/N) regardless of depth — following the GPT-2
scaled-init strategy.

### s and r Projection Initialisation

The `s_r_init` config field selects the initialisation scheme for the $`W_s`$ and $`W_r`$ projections
($`W_r`$ produces $`R_t`$: one entry per neuron for scalar-state neurons, $`d^2`$ per neuron otherwise).

| Scheme | $`W_s`$ weights | $`W_r`$ weights | $`b_s`$ | $`b_r`$ |
|---|---|---|---|---|
| `'small_init'` | `small_init_` | `small_init_` | 0 | 0 |
| `'kaiming'` | Kaiming uniform (a=√5) | Kaiming uniform (a=√5) | 0 | 0 |
| `'asymmetric'` | Kaiming uniform (a=√5) | `small_init_` | +1.0 | 0 |

The **asymmetric** scheme addresses an initialisation dead zone, seen most easily in the scalar case: with $`x_0 = 0`$ and both projections near zero, `max(min(r, x), s)` keeps every hidden unit at 0 whenever s < 0 < r, preventing state writes early in training.  Setting $`b_s = +1.0`$ ensures $`s > 0`$ at initialisation so writes happen from the first token; keeping $`W_r`$ small prevents r from immediately over-writing the written values.

---

## Feed-forward Variants

| Class | Formula | Paper |
|---|---|---|
| `FeedForward` | $`W_2 \cdot \phi(W_1 \cdot x + b_1) + b_2`$ | Vaswani et al., NeurIPS 2017 |
| `GatedFeedForward` | $`W_2 \cdot (\phi(\mathrm{gate}) * \mathrm{value}) + b_2`$, where $`[\mathrm{gate}; \mathrm{value}] = W_1 \cdot x + b_1`$ | Dauphin et al., ICML 2017; Shazeer 2020 |

Here $`\phi`$ is the activation selected by `ffn_act_fn`, $`\cdot`$ matrix
multiplication and $`*`$ the element-wise product. The gated variant is the default (`ffn_type='gated'`).  With `act_fn='relu'`
it is **ReGLU**; with `act_fn='swish'` it is **SwiGLU**.

---

## References

- Vaswani et al. (2017). *Attention Is All You Need.* NeurIPS.
  https://arxiv.org/abs/1706.03762

- Dauphin et al. (2017). *Language Modeling with Gated Convolutional Networks.* ICML.
  https://arxiv.org/abs/1612.08083

- Radford et al. (2019). *Language Models are Unsupervised Multitask Learners.*
  OpenAI Blog.  https://openai.com/research/language-unsupervised

- Nguyen & Salazar (2019). *Transformers without Tears: Improving the
  Normalization of Self-Attention.* IWSLT.
  https://arxiv.org/abs/1910.05895

- Shazeer (2020). *GLU Variants Improve Transformer.*
  https://arxiv.org/abs/2002.05202

- Beck et al. (2024). *xLSTM: Extended Long Short-Term Memory.*
  https://arxiv.org/abs/2405.04517
