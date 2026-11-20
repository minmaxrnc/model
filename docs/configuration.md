<!--
SPDX-FileCopyrightText: 2026 Alessandro Ronca
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
-->

# MinMax RNCs — Configuration and Implementation Reference

[Index](model.md) · [Mathematical foundation](foundations.md) · [Architecture](architecture.md) · [Configuration and implementation](configuration.md)

Configuration fields, numerical precision and initialisation. Notation as in
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
| `ffn_type` | str | `'gated'` | `'gated'` (ReGLU/SwiGLU) or `'basic'` (MLP) — see [Feed-forward Variants](architecture.md#feed-forward-variants) |
| `ffn_proj_factor` | float | `1.3` | FFN hidden-dim expansion factor |
| `ffn_act_fn` | str | `'relu'` | FFN activation: `'relu'`, `'relu^2'`, `'gelu'`, `'swish'`, `'sigmoid'`, `'selu'` |
| `ffn_dropout` | float | `0.1` | Dropout inside the FFN |
| `ffn_init` | str | `'scaled'` | FFN init: `'basic'` (PyTorch default) or `'scaled'` (small\_init + wang\_init) |
| `output_gate` | bool | `True` | Gate the neuron output by a learned projection of the input |
| `train_init` | bool | `False` | Make the initial hidden state $`x_0`$ a learned parameter |
| `neuron_dropout` | float | `0.0` | Dropout on the neuron input |
| `s_r_init` | str | `'small_init'` | Init scheme for s and r projections: `'small_init'`, `'kaiming'`, or `'asymmetric'` |
| `conv_type` | str | `'basic'` | `'basic'` (learned affine map of the previous and current token) or `'gated'` (per-feature gate between them) — see [Convolution Variants](architecture.md#convolution-variants) |
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

**What the scan's precision changes.** The scan is a chain of element-wise
min/max *selections*: every state value it produces is one of its input values
($`R_t`$, $`s_t`$ or $`x_0`$), never a newly computed one. Three consequences:

1. **Forward pass.** Running the scan in a *higher* precision than its inputs
   changes nothing: under bf16 AMP, $`R_t`$ and $`s_t`$ are already bf16, and
   min/max of bf16 values is exact in any wider type, so an `'fp32'` scan gives
   bitwise the same outputs as a `'bf16'` one. The exception is a learned
   initial state (`train_init=True`): $`x_0`$ is an fp32 parameter, which a
   bf16 scan rounds. Running the scan in a *lower* precision than its inputs
   (an fp32 model with `scan_dtype='bf16'`) rounds them, and so changes the
   forward pass.
2. **Backward pass.** The gradient of a selected value is the sum of the
   gradients of every later state it was selected into — for a value held over
   $`k`$ steps, a sum of $`k`$ terms. The scan's backward pass runs in the
   scan's dtype, so with `'fp32'` these sums are accumulated in fp32 even under
   bf16 AMP (and rounded to bf16 once, at the end), whereas with `'bf16'` every
   partial sum is rounded to bf16's 8-bit significand. The difference grows
   with how long values are held.
3. **Cost.** The scan is memory-bandwidth bound (no tensor-core acceleration),
   and its prefix-composition intermediates are a large share of activation
   memory for long sequences, so its cost and memory scale with the dtype's
   byte width: about 2× for fp32 vs bf16/fp16.

Under an outer bf16 autocast, `'ambient'` (as in 0.3.0) composes the prefixes in
bf16 while the fp32 initial state promotes the final application to fp32 —
mixed and implicit. Pinning makes the choice explicit:

| Recipe | `amp_dtype` | `scan_dtype` | Forward | Scan gradient sums | Scan memory |
|---|---|---|---|---|---|
| Full precision / 0.3.0 behaviour (default) | `None` | `'ambient'` | fp32 | fp32 | 4 bytes |
| bf16 AMP, fp32 scan | `'bf16'` | `'fp32'` | = bf16 scan | fp32 | 4 bytes |
| Fully bf16, scan included | `'bf16'` | `'bf16'` | bf16 | bf16 | 2 bytes |
| fp32 model, bf16 scan | `None` | `'bf16'` | scan rounds its inputs | bf16 | 2 bytes |
| bf16 AMP, fp16 scan (untested) | `'bf16'` | `'fp16'` | = bf16 scan, within fp16's range | fp16 (finer than bf16; overflow above 65504) | 2 bytes |

The effect of the more accurate gradient sums on training has not been measured
yet; compare `'bf16'` against `'fp32'` on your task before relying on it.

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

## References

- Radford et al. (2019). *Language Models are Unsupervised Multitask Learners.*
  OpenAI Blog.  https://openai.com/research/language-unsupervised

- Nguyen & Salazar (2019). *Transformers without Tears: Improving the
  Normalization of Self-Attention.* IWSLT.
  https://arxiv.org/abs/1910.05895

- Beck et al. (2024). *xLSTM: Extended Long Short-Term Memory.*
  https://arxiv.org/abs/2405.04517
