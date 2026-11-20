<!--
SPDX-FileCopyrightText: 2026 Alessandro Ronca
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
-->

# MinMax Recurrent Neural Cascades (MinMax RNCs)


See [arxiv.org/abs/2605.06384](https://arxiv.org/abs/2605.06384) for the formal description and analyses.

## Mathematical Foundation

**Notation.** $`\oplus`$ and $`\otimes`$ are the MinMax operations defined below;
$`\cdot`$ is ordinary (matrix) multiplication; $`*`$ is the element-wise
(Hadamard) product of vectors; $`\sigma`$ is the logistic sigmoid.

### The MinMax Algebra

The MinMax algebra replaces addition and multiplication of real numbers with
max and min:

| Operation | Definition |
|---|---|
| MinMax "addition" | $`a \oplus b = \mathrm{max}(a, b)`$ |
| MinMax "multiplication" | $`a \otimes b = \mathrm{min}(a, b)`$ |

(the (max,min) semiring). Both operations extend to vectors and matrices in
the usual way, with $`\oplus`$ in place of $`+`$ and $`\otimes`$ in place of $`\times`$:

| Operation | Definition |
|---|---|
| vector "sum" | $`(x \oplus y)_i = \mathrm{max}(x_i, y_i)`$ |
| matrix-vector "product" | $`(R \otimes x)_i = \mathrm{max}_j\, \mathrm{min}(R_{ij}, x_j)`$ |
| matrix-matrix "product" | $`(R \otimes Q)_{ik} = \mathrm{max}_j\, \mathrm{min}(R_{ij}, Q_{jk})`$ |

### The MinMax Map

A MinMax map on states of dimension $`d`$ is the affine map of this algebra,
parameterised by a matrix $`R \in ℝ^{d \times d}`$ and a vector $`s \in ℝ^d`$:

```math
f(x) = (R \otimes x) \oplus s,
\qquad\text{i.e.}\qquad
f(x)_i = \mathrm{max}\bigl(\mathrm{max}_j\, \mathrm{min}(R_{ij}, x_j),\ s_i\bigr).
```

Every coordinate of $`f(x)`$ is *selected* among the entries of $`x`$, $`R`$ and $`s`$ —
never computed by arithmetic — so $`f`$ keeps values bounded and passes them
through unchanged.

**Scalar case ($`d = 1`$).** With a scalar $`r = R`$, the map is a clamp between
the lower bound $`s`$ and the upper bound $`r`$:

```math
f(x) = \mathrm{max}(\mathrm{min}(r, x), s)
```

| Input | Output | Interpretation |
|---|---|---|
| $`x < s`$ | $`f(x) = s`$ | lower clamp — set state to input value / store value into memory |
| $`s \le x \le r`$ | $`f(x) = x`$ | identity — propagate state / memory preserved unchanged |
| $`x > r`$ | $`f(x) = r`$ | upper clamp — reset state / clear memory  |

(when $`r \ge s`$; when $`r < s`$, $`f(x) = s`$ for every $`x`$). For $`d > 1`$ the map also
*routes* values between coordinates: coordinate $`i`$ can take the value of coordinate $`j`$,
limited by $`R_{ij}`$ — which is what lets a neuron permute its state.

### Composition Closure

The composition of two MinMax maps is again a MinMax map, because $`\otimes`$ is associative and
distributes over $`\oplus`$:

```math
\begin{aligned}
f_2(f_1(x)) = (R \otimes x) \oplus s, 
\qquad \text{where} 
R = R_2 \otimes R_1,
\quad
s = (R_2 \otimes s_1) \oplus s_2.
\end{aligned}
```

In the scalar case this reads $`r = \mathrm{min}(r_2, r_1)`$ and
$`s = \mathrm{max}(\mathrm{min}(r_2, s_1), s_2)`$.

This closure property is what makes the parallel scan possible: instead of
applying $`f_1, f_2, \ldots, f_T`$ one by one, all prefix compositions can be
computed simultaneously in $`O(\log T)`$ depth.

### The MinMax Recurrence

A MinMax neuron with state dimension $`d`$ maintains a hidden state $`x_t \in ℝ^d`$,
updated at every step by a MinMax map whose parameters are computed from the
current input $`u_{t+1} \in ℝ^m`$:

```math
x_{t+1} = (R_{t+1} \otimes x_t) \oplus s_{t+1}
```

where both parameters are affine projections of the input:

| Parameter | Definition | Role |
|---|---|---|
| $`R_t`$ | $`\mathtt{reshape}(W_r \cdot u_t, (d, d))`$ | $`d^2`$ outputs, reshaped to a matrix |
| $`s_t`$ | $`W_s \cdot u_t`$ | lower bound — set value / inject signal |

$`R_t`$ is unconstrained: every entry is an independent linear function of $`u_t`$,
and no bounding is needed for stability, since the recurrence only ever
*selects* values — every entry of a composed $`R_t`$, and of $`x_t`$ itself, is one
of the per-step values, never an accumulation of them.

**Scalar-state neurons ($`d = 1`$).** With $`d = 1`$, $`R_t = r_t`$ is a scalar and the
recurrence is the clamp above:

```math
x_{t+1} = \mathrm{max}(\mathrm{min}(r_{t+1}, x_t), s_{t+1})
```

| Parameter | Definition | Role |
|---|---|---|
| $`r_t`$ | $`W_r \cdot u_t`$ | upper bound — gate / reset signal |
| $`s_t`$ | $`W_s \cdot u_t`$ | lower bound — write / inject signal |

This is the neuron of 0.3.0: `MinMaxRNCConfig.scalar_neurons = N` (or its legacy
name `d_state`) gives every layer $`N`$ scalar-state neurons, i.e. a state
$`x_t \in ℝ^N`$ updated element-wise. Scalar-state neurons are evaluated with a
dedicated element-wise scan, so their cost per step is $`O(1)`$ per neuron.
Conversely, a $`d \times d`$ matrix whose off-diagonal entries are $`-\infty`$
decouples the coordinates into $`d`$ scalar-state neurons.

**Cost.** Per neuron, parameters scale as $`O(d^2)`$ and each composition step
of the scan as $`O(d^3)`$. A neuron with state dimension $`d`$ can realise
permutation semiautomata of degree $`d`$, so $`d`$ should be as large as the task
needs but typically small; layers scale by having more neurons
(`num_neurons`) and more neuron groups rather than larger $`d`$.

**Output.** Each neuron's state is projected back to the input space, and
the outputs of all neurons in a layer are summed:

| `output_gate` | Output |
|---|---|
| `False` | $`y_t = W_o \cdot x_t`$ |
| `True` (default) | $`y_t = W_o \cdot (x_t * \sigma(W_g \cdot u_t))`$ |

### Parallel Computation via Prefix Scan

Given $`x_0`$ and the per-step maps $`f_1, \ldots, f_T`$ ($`f_t`$ with parameters
$`R_t, s_t`$), all states $`x_1, \ldots, x_T`$ can be computed in parallel:

1. **Prefix compose** — compute, for each position $`t`$, the MinMax map
   $`F_t = f_t \circ f_{t-1} \circ \cdots \circ f_1`$ that maps $`x_0`$ to $`x_t`$,
   using composition closure. The current implementation uses the
   Hillis–Steele parallel prefix scan, with depth $`O(\log T)`$ and work
   $`O(T \log T)`$ for $`T`$ the sequence length (a work-efficient scan would
   need only $`O(T)`$ work, at about twice the depth).

2. **Batch apply** — evaluate every prefix map on $`x_0`$ simultaneously:
   $`x_t = F_t(x_0)`$.

For scalar-state neurons the scan is element-wise (`minmax_scan.all_states`);
for $`d > 1`$ it composes $`d \times d`$ matrices (`minmax_scan.all_states_matrix`),
with the neurons of a group folded into the batch axis, so every composition
costs $`O(d^3)`$.

At inference time (token-by-token generation) the recurrence runs
sequentially as a standard RNN with constant cost per step and a state of
fixed size. Any mix is possible: `unroll_steps = k` runs the scan in parallel
over chunks of $`k`$ steps and the recurrence across chunks — depth
$`O((T/k) \cdot \log k)`$, work $`O(T \log k)`$.

---

## Architecture

```
MinMaxRNC_LM
└── token_emb   : Embedding(vocab_size, d_model)
└── MinMaxRNC   : backbone
    ├── MinMaxLayer × n_layers
    │   ├── norm_conv   → BasicConv / GatedConv
    │   ├── norm_ffn    → FeedForward / GatedFeedForward
    │   └── norm_neuron → Σ MinMaxNeuronGroup   (neuron: scalar_neurons shorthand,
    │                                            neuron_groups: `neurons`)
    └── postlayers_ffn  (optional)
    └── postlayers_norm
└── head_drop   : Dropout
└── lm_head     : Linear(d_model, vocab_size)
```

### MinMax Neuron Groups

`MinMaxNeuronGroup` implements `num_neurons` neurons with the same state
dimension $`d`$, fed the same input. It projects the input to $`s_t`$ and $`R_t`$, runs one
batched parallel scan for all its neurons, and projects the resulting states
back to the residual-stream dimension; the neurons' outputs are summed
through a single `Linear(num_neurons·d, d_model)`. This is exactly equivalent
to `num_neurons` separate neurons whose outputs are summed; only the compute
is batched.

The group's state is the flat vector of its `num_neurons · d` coordinates
(neuron-major). For d = 1 the group is evaluated with the element-wise scan
and every parameter, tensor and operation is exactly that of 0.3.0's MinMax
Neuron. `MinMaxNeuron` (the 0.3.0 class) is the group of
`MinMaxNeuronConfig.d_state` scalar-state neurons, which is how the layer
holds the `scalar_neurons` shorthand.

Learnable parameters: $`W_s`$, $`W_r`$, $`W_o`$ (and optionally $`W_g`$, $`x_0`$).

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
all reading the same normalised input: the `scalar_neurons` shorthand group
(if present, registered as `neuron` for compatibility with 0.3.0) and each
group of `neurons` (registered in `neuron_groups`). Each group's output
projection is initialised with `wang_init_` using the layer's total state
size Σ num_neurons · d as `dim`, so the layer's overall contribution to the
residual stream stays O(1) however its state is split across groups.

Conv and FFN outputs are used to compute a richer input for the neuron but
are not independently added to the residual stream.  The convolution provides
short-range context (one previous token).  The FFN mixes features.  The
neuron integrates information over arbitrarily long ranges via the recurrence.

### Convolution Variants

**BasicConv** (default) — a learned linear projection that concatenates $`[u_{t-1}, u_t]`$
and maps back to $`ℝ^D`$, giving more expressive local mixing at the cost of 2× the
parameters.

**GatedConv** — a learned per-feature scalar gate $`g \in ℝ^D`$ interpolates
between the previous and current token:

```math
\mathrm{out}_t = \sigma(g) * u_{t-1} + (1 - \sigma(g)) * u_t
```

where $`*`$ is the element-wise product: each feature mixes the previous and
the current token with its own weight $`\sigma(g_i)`$.

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
| `output_gate` | bool | `True` | Gate each neuron output by $`\sigma(W_g \cdot u)`$; overrides `backbone.output_gate` |
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
| `FeedForward` | $`W_2 \cdot \phi(W_1 \cdot x)`$ | Vaswani et al., NeurIPS 2017 |
| `GatedFeedForward` | $`W_2 \cdot (\phi(\mathrm{gate}) * \mathrm{value})`$, where $`[\mathrm{gate}; \mathrm{value}] = W_1 \cdot x`$ | Dauphin et al., ICML 2017; Shazeer 2020 |

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

- Hillis & Steele (1986). *Data parallel algorithms.* CACM 29(12).
