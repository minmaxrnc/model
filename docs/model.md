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

The MinMax Neuron maintains a hidden state xₜ ∈ Rᴰ updated as:

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
    │   └── norm_neuron → MinMaxNeuron
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

### MinMax Layer

One residual block. The data flow uses pre-norm; only the neuron output is
added back to the residual stream:

```
conv   = Conv( norm(u) )
ffn    = FFN( norm(u + conv) )
neuron = Neuron( norm(u + conv + ffn) )
output = u + neuron
```

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
| `d_state` | int | — | Hidden-state dimension of each neuron |
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

The gated variant is the default (`ffn_type='ffn_gated'`).  With `act_fn='relu'`
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
