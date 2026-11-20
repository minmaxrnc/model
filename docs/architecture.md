<!--
SPDX-FileCopyrightText: 2026 Alessandro Ronca
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
-->

# MinMax RNCs — Architecture

[Index](model.md) · [Mathematical foundation](foundations.md) · [Architecture](architecture.md) · [Configuration and implementation](configuration.md)

Notation as in [Mathematical foundation](foundations.md): $`\cdot`$ is matrix multiplication,
$`*`$ the element-wise product, $`\sigma`$ the logistic sigmoid.

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

## MinMax Neuron Groups

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

Learnable parameters: weights $`W_s`$, $`W_r`$, $`W_o`$ and biases $`b_s`$, $`b_r`$, $`b_o`$; with the output
gate also $`W_g`$, $`b_g`$; with `train_init`, the initial state $`x_0`$.

## MinMax Layer

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

## Convolution Variants

**BasicConv** (default) — a learned affine projection of the concatenated previous and current
token, $`\mathrm{out}_t = W_c \cdot [u_{t-1}; u_t] + b_c`$, mapping back to $`ℝ^D`$; more expressive
local mixing at the cost of 2× the parameters.

**GatedConv** — a learned per-feature scalar gate $`g \in ℝ^D`$ interpolates
between the previous and current token:

```math
\mathrm{out}_t = \sigma(g) * u_{t-1} + (1 - \sigma(g)) * u_t
```

where $`*`$ is the element-wise product: each feature mixes the previous and
the current token with its own weight $`\sigma(g_i)`$.

## MinMax RNC

Stacks `n_layers` MinMaxLayers.  An optional post-layers FFN and normalisation
are applied after the final layer.

## Language Model Wrapper

`MinMaxRNC_LM` adds:
- Token embedding with `small_init_` initialisation.
- A dropout before the LM head.
- An optional weight tie between the embedding and the LM head.
