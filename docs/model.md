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

### The MinMax Recurrence (diagonal)

The MinMax Neuron (`MinMaxNeuron`, the default neuron of every layer)
maintains a hidden state xₜ ∈ Rᴰ updated as:

```
x_{t+1} = max(min(rₜ, xₜ), sₜ)       (elementwise over D dimensions)
```

where rₜ and sₜ are linear projections of the current input uₜ ∈ Rᴵ:

```
rₜ = Wᵣ uₜ      (upper bound — gate / forget signal)
sₜ = Wₛ uₜ      (lower bound — write / inject signal)
```

The output is projected back to the input space:

```
yₜ = Wₒ xₜ                            (output_gate=False)
yₜ = Wₒ (xₜ ⊙ σ(Wᵍ uₜ))               (output_gate=True)
```

### The MinMax Recurrence (full matrix)

*New in 0.4.0; opt-in via `MinMaxRNCConfig.matrix_neurons`.*

The MinMax matrix algebra generalises the scalar operator above from
per-coordinate clamping to a full linear mixing of the state, using the
(max,min)-semiring in place of ordinary (+,·):

```
(R ⊗ x)ᵢ = maxⱼ min(Rᵢⱼ, xⱼ)     (matrix-vector product in the (max,min) semiring)
```

MatrixMinMaxNeuron maintains a hidden state xₜ ∈ Rᴰ updated as:

```
x_{t+1} = max(Rₜ ⊗ xₜ, sₜ)
```

where Rₜ ∈ Rᴰˣᴰ and sₜ ∈ Rᴰ are computed from uₜ. `sₜ = Wₛ uₜ` always; `Rₜ`
is built one of two ways, selected per neuron group by `n_h`:

```
n_h = 0 (default):  Rₜ = reshape(Wᵣ uₜ, (D, D))    (D² outputs, reshaped to a matrix)

n_h > 0:            Rₜ = ∏_{i=1}^{n_h} (I - βᵢ(uₜ) kᵢ(uₜ) kᵢ(uₜ)ᵀ)
                     kᵢ = reshape(Wₖ uₜ, (n_h, D)),  βᵢ = Wᵦ uₜ
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
real D×D matrix once built, so it feeds into the recurrence/scan below
identically; params scale as O(d_state²) for `n_h = 0` vs. O(n_h · d_state)
for `n_h > 0` (typically far fewer once `n_h < d_state`).

`kᵢ`'s norm is left unconstrained by default: it drifts with `uₜ` and is
entangled with `βᵢ`, so both jointly determine each factor's magnitude,
which can make it harder for the network to keep `Rₜ`'s entries in a useful
range for the (max,min) comparisons. Setting `normalize_keys=True`
L2-normalises each `kᵢ` along its `D` axis before forming `kᵢ kᵢᵀ`, as
DeltaProduct does, so `βᵢ` alone controls the reflection's magnitude — at
the cost of `k`'s norm no longer carrying information and of coupling
gradients across all `D` entries of `kᵢ` (they jointly determine one shared
norm). Only affects neuron groups with `n_h > 0`.

The (max,min)-semiring matrix product ⊗ is associative and distributes over
element-wise max, so composition closure holds exactly as in the scalar case:

```
f₂(f₁(x)) = max(R ⊗ x, s)
  R = R₂ ⊗ R₁,  (R₂ ⊗ R₁)ᵢₖ = maxⱼ min(R₂ᵢⱼ, R₁ⱼₖ)
  s = max(R₂ ⊗ s₁, s₂)
```

Setting the off-diagonal entries of Rₜ to -∞ recovers the diagonal recurrence
exactly, so the diagonal MinMaxNeuron is a (much cheaper) special case: the matrix
neuron's parameter count and per-step scan cost scale as O(D²) and O(D³)
respectively (vs. O(D) and O(D) for the diagonal neuron), so d_state should
typically be kept small for matrix neurons — `MinMaxRNCConfig.matrix_neurons`
combines several groups of independent small-matrix neurons per layer rather
than growing one large one. With D = 1 the two neurons coincide (a 1×1 group
triggers a warning suggesting the diagonal neuron instead).

Output projection is identical in form to the diagonal neuron:

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

At inference time (token-by-token generation) the recurrence runs
sequentially as a standard RNN with O(1) cost per step and O(D) state.

---

## Architecture

```
MinMaxRNC_LM
└── token_emb   : Embedding(vocab_size, d_model)
└── MinMaxRNC   : backbone
    ├── MinMaxLayer × n_layers
    │   ├── norm_conv   → BasicConv / GatedConv
    │   ├── norm_ffn    → FeedForward / GatedFeedForward
    │   └── norm_neuron → MinMaxNeuron              (if d_state > 0)
    │                   + MatrixMinMaxNeuron × groups (if matrix_neurons)
    └── postlayers_ffn  (optional)
    └── postlayers_norm
└── head_drop   : Dropout
└── lm_head     : Linear(d_model, vocab_size)
```

### MinMax Neuron

The recurrent cell. Projects the input to upper-bound (r) and lower-bound (s)
signals, runs the parallel MinMax scan, and projects the resulting hidden
states back to the residual-stream dimension.

Learnable parameters: Wᵣ, Wₛ, Wₒ (and optionally Wᵍ, x₀).

### Matrix MinMax Neurons

`MatrixMinMaxNeuron` (new in 0.4.0) implements one *group* of `num_neurons`
independent matrix neurons of the same `d_state`, fed the same input. The
group's scans run as one batched parallel scan (the neuron axis is folded into
the batch axis) and its outputs are summed through a single
`Linear(num_neurons·D, d_model)`. This is exactly equivalent to `num_neurons`
separate neurons whose outputs are summed; only the compute is batched.

Learnable parameters: Wₛ, Wᵣ (or Wₖ, Wᵦ when `n_h > 0`), Wₒ (and optionally
Wᵍ, x₀) — per neuron of the group.

### MinMax Layer

One residual block. The data flow uses pre-norm; only the neuron output is
added back to the residual stream:

```
conv   = Conv( norm(u) )
ffn    = FFN( norm(u + conv) )
neuron = Neuron( norm(u + conv + ffn) )
output = u + neuron
```

where `Neuron` is the sum of the layer's diagonal MinMaxNeuron (if
`d_state > 0`) and of every matrix-neuron group (if any), all reading the
same normalised input. Each neuron's output projection is initialised with
`wang_init_` scaled by the total number of neurons in the layer, so the
layer's overall contribution to the residual stream stays O(1) regardless of
how its neurons are split.

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
| `d_state` | int | `0` | Hidden-state dimension of each layer's diagonal neuron; `0` = no diagonal neuron (then `matrix_neurons` is required) |
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
| `matrix_neurons` | tuple | `()` | *0.4.0.* Matrix-neuron groups per layer: `MatrixNeuronGroup(d_state, num_neurons=1, n_h=0, normalize_keys=False)` or equivalent tuples/lists |
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
| `'fp32'` / `'bf16'` / `'fp16'` | cast to that dtype (for matrix neurons, Rₜ is also *built* in it) | disabled | that dtype |

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
