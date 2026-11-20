<!--
SPDX-FileCopyrightText: 2026 Alessandro Ronca
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
-->

# MinMax Recurrent Neural Cascades (MinMax RNCs)


See [arxiv.org/abs/2605.06384](https://arxiv.org/abs/2605.06384) for the formal description and analyses.

## Mathematical Foundation

### The MinMax Operator

The MinMax operator is a function f : R → R parameterised by two scalars
a (upper bound) and b (lower bound):

```
f(x) = max(min(a, x), b)
```

Applied elementwise to vectors, it acts as a soft clamp:

```
x < b        →  f(x) = b         (lower clamp — signal injection)
b ≤ x ≤ a    →  f(x) = x         (identity — memory preserved unchanged)
x > a        →  f(x) = a         (upper clamp — saturation / forgetting)
```

### Composition Closure

The composition of two MinMax operators is again a MinMax operator:

```
f₂(f₁(x))  =  max(min(aₒ, x), bₒ)

  aₒ = min(a₂, a₁)
  bₒ = max(min(a₂, b₁), b₂)
```

This closure property is what makes the parallel scan possible: instead of
applying f₁, f₂, …, fₜ one by one, we can compose all prefix products
simultaneously in O(log T) depth.

### The MinMax Recurrence

A MinMax neuron with state dimension d maintains a hidden state xₜ ∈ Rᵈ,
updated by the MinMax recurrence — the affine recurrence of the (max,min)
semiring, where max plays the role of addition and min of multiplication:

```
x_{t+1} = max(Rₜ ⊗ xₜ, sₜ)
(R ⊗ x)ᵢ = maxⱼ min(Rᵢⱼ, xⱼ)       (matrix-vector product in the (max,min) semiring)
```

where Rₜ ∈ Rᵈˣᵈ and sₜ ∈ Rᵈ are computed from the current input uₜ ∈ Rᴵ.
`sₜ = Wₛ uₜ` (the lower bound — write / inject signal). Rₜ is built one of two
ways, selected by `n_h`:

```
n_h = 0 (default):  Rₜ = reshape(Wᵣ uₜ, (d, d))    (d² outputs, reshaped to a matrix)

n_h > 0:            Rₜ = ∏_{i=1}^{n_h} (I - βᵢ(uₜ) kᵢ(uₜ) kᵢ(uₜ)ᵀ)
                     kᵢ = reshape(Wₖ uₜ, (n_h, d)),  βᵢ = Wᵦ uₜ
```

**Experimental.** The `n_h > 0` form is a product of rank-1 corrections to the identity, in the
style of DeltaProduct's (Siems et al., NeurIPS 2025) state-transition matrix
— but unlike DeltaProduct, βᵢ is a plain unconstrained affine function of uₜ
(no sigmoid bounding into `[0, 2]`) and kᵢ is not unit-normalised, so each
factor is a general rank-1 perturbation of the identity rather than strictly
a Householder reflection. DeltaProduct needs bounded eigenvalues to keep its
*linear* recurrence numerically stable over long sequences; the (max,min)
recurrence here is idempotent — every entry of a composed Rₜ (or of xₜ
itself) is *selected from*, never accumulated from, the individual per-step
R/s values — so there is no analogous blow-up/vanishing-gradient mode to
guard against, and no bounding is needed. Either way, `Rₜ` is an ordinary
real d×d matrix once built, so it feeds into the recurrence/scan below
identically; params scale as O(d²) for `n_h = 0` vs. O(n_h · d)
for `n_h > 0` (typically far fewer once `n_h < d`).

`kᵢ`'s norm is left unconstrained by default: it drifts with `uₜ` and is
entangled with `βᵢ`, so both jointly determine each factor's magnitude,
which can make it harder for the network to keep `Rₜ`'s entries in a useful
range for the (max,min) comparisons. Setting `normalize_keys=True`
L2-normalises each `kᵢ` along its `d` axis before forming `kᵢ kᵢᵀ`, as
DeltaProduct does, so `βᵢ` alone controls the reflection's magnitude — at
the cost of `k`'s norm no longer carrying information and of coupling
gradients across all `d` entries of `kᵢ` (they jointly determine one shared
norm). Only affects neuron groups with `n_h > 0`.

**Scalar-state neurons (d = 1).** With d = 1, Rₜ = rₜ is a scalar and the
recurrence becomes

```
x_{t+1} = max(min(rₜ, xₜ), sₜ)
rₜ = Wᵣ uₜ      (upper bound — gate / forget signal)
sₜ = Wₛ uₜ      (lower bound — write / inject signal)
```

This is the neuron of 0.3.0: `MinMaxRNCConfig.d_state = D` gives every layer
D scalar-state neurons, i.e. a state xₜ ∈ Rᴰ updated element-wise. Scalar-state
neurons are evaluated with a dedicated element-wise scan, so their cost per
step is O(1) per neuron. More generally, a d × d matrix whose off-diagonal
entries are -∞ decouples the coordinates into d scalar-state neurons.

**Closure under composition.** The (max,min)-semiring matrix product ⊗ is
associative and distributes over element-wise max, so the composition of two
MinMax maps is again a MinMax map — exactly as in the scalar case above:

```
f₂(f₁(x)) = max(R ⊗ x, s)
  R = R₂ ⊗ R₁,  (R₂ ⊗ R₁)ᵢₖ = maxⱼ min(R₂ᵢⱼ, R₁ⱼₖ)
  s = max(R₂ ⊗ s₁, s₂)
```

**Cost.** Per neuron, parameters scale as O(d²) (O(n_h · d) for `n_h > 0`) and
each composition step of the scan as O(d³). A neuron with state dimension d
can realise permutation semiautomata of degree d, so d should be as large as
the task needs but typically small; layers scale by having more neurons
(`num_neurons`) and more neuron groups rather than larger d.

**Output.** Each neuron's state is projected back to the input space, and
the outputs of all neurons in a layer are summed:

```
yₜ = Wₒ xₜ                            (output_gate=False)
yₜ = Wₒ (xₜ ⊙ σ(Wᵍ uₜ))               (output_gate=True)
```

### Parallel Computation via Prefix Scan

Given x₀ and the sequence (r₁, s₁), …, (rₜ, sₜ), all states x₁, …, xₜ
can be computed in parallel:

1. **Prefix compose** — compute, for each position t, the MinMax operator
   Fₜ = fₜ ∘ fₜ₋₁ ∘ … ∘ f₁ that maps x₀ to xₜ.
   The current implementation uses the Hillis–Steele parallel prefix scan, with depth
   O(log T) and work O(T log T) for T the sequence length.

2. **Batch apply** — evaluate every prefix operator on x₀ simultaneously:
   xₜ = Fₜ(x₀).

For scalar-state neurons the scan is element-wise (`minmax_scan.all_states`);
for d > 1 it composes d × d matrices (`minmax_scan.all_states_matrix`), with
the neurons of a group folded into the batch axis.

At inference time (token-by-token generation) the recurrence runs
sequentially as a standard RNN with constant cost per step and a state of
fixed size. Any mix is possible: `unroll_steps = k` runs the scan in parallel
over chunks of k steps and the recurrence across chunks.

---

## Architecture

```
MinMaxRNC_LM
└── token_emb   : Embedding(vocab_size, d_model)
└── MinMaxRNC   : backbone
    ├── MinMaxLayer × n_layers
    │   ├── norm_conv   → BasicConv / GatedConv
    │   ├── norm_ffn    → FeedForward / GatedFeedForward
    │   └── norm_neuron → Σ MinMaxNeuronGroup   (neuron: d_state shorthand,
    │                                            neuron_groups: `neurons`)
    └── postlayers_ffn  (optional)
    └── postlayers_norm
└── head_drop   : Dropout
└── lm_head     : Linear(d_model, vocab_size)
```

### MinMax Neuron Groups

`MinMaxNeuronGroup` implements `num_neurons` neurons with the same state
dimension d, fed the same input. It projects the input to sₜ and Rₜ, runs one
batched parallel scan for all its neurons, and projects the resulting states
back to the residual-stream dimension; the neurons' outputs are summed
through a single `Linear(num_neurons·d, d_model)`. This is exactly equivalent
to `num_neurons` separate neurons whose outputs are summed; only the compute
is batched.

The group's state is the flat vector of its `num_neurons · d` coordinates
(neuron-major). For d = 1 the group is evaluated with the element-wise scan
and every parameter, tensor and operation is exactly that of 0.3.0's MinMax
Neuron. `MinMaxNeuron` (the 0.3.0 class) is the group of `d_state`
scalar-state neurons.

Learnable parameters: Wₛ, Wᵣ (or Wₖ, Wᵦ when `n_h > 0`), Wₒ (and optionally
Wᵍ, x₀).

### MinMax Layer

One residual block. The data flow uses pre-norm; only the neuron output is
added back to the residual stream:

```
conv   = Conv( norm(u) )
ffn    = FFN( norm(u + conv) )
neuron = Neuron( norm(u + conv + ffn) )
output = u + neuron
```

where `Neuron` is the sum of the outputs of all the layer's neuron groups,
all reading the same normalised input: the `d_state` shorthand group (if
`d_state > 0`, registered as `neuron` for compatibility with 0.3.0) and each
group of `neurons` (registered in `neuron_groups`). Each group's output
projection is initialised with `wang_init_` using the layer's total state
size Σ num_neurons · d as `dim`, so the layer's overall contribution to the
residual stream stays O(1) however its state is split across groups.

Conv and FFN outputs are used to compute a richer input for the neuron but
are not independently added to the residual stream.  The convolution provides
short-range context (one previous token).  The FFN mixes features.  The
neuron integrates information over arbitrarily long ranges via the recurrence.

### Convolution Variants

**BasicConv** (default) — a learned linear projection that concatenates [uₜ₋₁, uₜ]
and maps back to Rᴰ, giving more expressive local mixing at the cost of 2× the
parameters.

**GatedConv** — a learned per-feature scalar gate g ∈ Rᴰ interpolates
between the previous and current token:

```
outₜ = σ(g) ⊙ uₜ₋₁ + (1 − σ(g)) ⊙ uₜ
```

### MinMax RNC

Stacks `n_layers` MinMaxLayers.  An optional post-layers FFN and normalisation
are applied after the final layer.

### Language Model Wrapper

`MinMaxRNC_LM` adds:
- Token embedding with `small_init_` initialisation.
- A dropout before the LM head.
- An optional weight tie between the embedding and the LM head.

---

## Configuration Reference

All architecture hyperparameters are specified through a single flat
`MinMaxRNCConfig` dataclass.

### MinMaxRNCConfig

| Field | Type | Default | Description |
|---|---|---|---|
| `d_model` | int | — | Residual-stream / embedding width |
| `n_layers` | int | — | Number of MinMaxLayers |
| `d_state` | int | `0` | Number of scalar-state neurons per layer (shorthand for the group `NeuronGroup(1, d_state)`); `0` = none (then `neurons` is required) |
| `norm` | str | `'layernorm'` | In-layer pre-norm: `'layernorm'`, `'rmsnorm'`, `'none'` |
| `postlayers_norm` | str | `'layernorm'` | Norm applied after the final layer |
| `ffn_type` | str | `'gated'` | `'gated'` (ReGLU/SwiGLU) or `'basic'` (MLP) |
| `ffn_proj_factor` | float | `1.3` | FFN hidden-dim expansion factor |
| `ffn_act_fn` | str | `'relu'` | FFN activation: `'relu'`, `'relu^2'`, `'gelu'`, `'swish'`, `'sigmoid'`, `'selu'` |
| `ffn_dropout` | float | `0.1` | Dropout inside the FFN |
| `ffn_init` | str | `'scaled'` | FFN init: `'basic'` (PyTorch default) or `'scaled'` (small\_init + wang\_init) |
| `output_gate` | bool | `True` | Gate the neuron output by a learned projection of the input |
| `train_init` | bool | `False` | Make the initial hidden state x₀ a learned parameter |
| `neuron_dropout` | float | `0.0` | Dropout on the neuron input |
| `s_r_init` | str | `'small_init'` | Init scheme for s and r projections: `'small_init'`, `'kaiming'`, or `'asymmetric'` |
| `conv_type` | str | `'basic'` | `'gated'` (scalar gate) or `'basic'` (linear mix) |
| `conv_init_val` | float | `0.0` | Initial gate logit for GatedConv |
| `prelayers_dropout` | float | `0.0` | FFN dropout override for the first layer only |
| `use_postlayers_ffn` | bool | `False` | Add an FFN after all layers |
| `neurons` | tuple | `()` | *0.4.0.* Further neuron groups per layer, of any state dimension: `NeuronGroup(d_state, num_neurons=1, n_h=0, normalize_keys=False)` or equivalent tuples/lists |
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
| `output_gate` | bool | `True` | Gate each neuron output by σ(W_g u); overrides `backbone.output_gate` |
| `conv_type` | `'basic'` \| `'gated'` | `'basic'` | Short-range conv variant; overrides `backbone.conv_type` |
| `ffn_dropout` | float | `0.1` | Dropout inside the FFN of every layer (except the first, which uses `backbone.prelayers_dropout`); overrides `backbone.ffn_dropout` |

### Preset factories

```python
MinMaxRNCConfig.small()    # d_model=90,  n_layers=2,  d_state=40
MinMaxRNCConfig.medium()   # d_model=512, n_layers=8,  d_state=512
MinMaxRNCConfig.large()    # d_model=728, n_layers=12, d_state=1456
```


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
| `'fp32'` / `'bf16'` / `'fp16'` | cast to that dtype (Rₜ is also *built* in it) | disabled | that dtype |

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

Both functions initialise by sampling p ~ N(0, σ²); the table gives σ.

| Function | σ | Paper |
|---|---|---|
| `small_init_init_(p, dim)` | √(2 / (5·dim)) | Nguyen & Salazar, IWSLT 2019 |
| `wang_init_(p, dim, N)` | 2 / (N·√dim) | Radford et al. 2019 (GPT-2) + Nguyen & Salazar 2019; used in Beck et al. 2024 (xLSTM) |

`small_init_` is applied to all **input-side** projections: token embedding,
LM head (when untied), BasicConv, FFN up-projection, and the neuron Wₛ, Wᵣ,
and output-gate Wᵍ projections.  It keeps activations small at initialisation
in deep networks.

`wang_init_` is applied to the two **residual output** projections: the
neuron output Wₒ and the FFN down-projection.  It additionally divides by N
(number of residual blocks) so the total variance contributed by all residual
branches to the stream stays O(1/N) regardless of depth — following the GPT-2
scaled-init strategy.

### s and r Projection Initialisation

The `s_r_init` config field selects the initialisation scheme for the Wₛ and Wᵣ projections.

| Scheme | Wₛ weights | Wᵣ weights | bₛ | bᵣ |
|---|---|---|---|---|
| `'small_init'` | `small_init_` | `small_init_` | 0 | 0 |
| `'kaiming'` | Kaiming uniform (a=√5) | Kaiming uniform (a=√5) | 0 | 0 |
| `'asymmetric'` | Kaiming uniform (a=√5) | `small_init_` | +1.0 | 0 |

The **asymmetric** scheme addresses an initialisation dead zone: with x₀ = 0 and both projections near zero, `max(min(r, x), s)` keeps every hidden unit at 0 whenever s < 0 < r, preventing state writes early in training.  Setting bₛ = +1.0 ensures s > 0 at initialisation so writes happen from the first token; keeping Wᵣ small prevents r from immediately over-writing the written values.

---

## Feed-forward Variants

| Class | Formula | Paper |
|---|---|---|
| `FeedForward` | σ(W₁x) → W₂ | Vaswani et al., NeurIPS 2017 |
| `GatedFeedForward` | W₂(σ(gate) ⊙ value), where [gate \| value] = W₁x | Dauphin et al., ICML 2017; Shazeer 2020 |

The gated variant is the default (`ffn_type='gated'`).  With `act_fn='relu'`
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

- Hillis & Steele (1986). *Data parallel algorithms.* CACM 29(12).
