<!--
SPDX-FileCopyrightText: 2026 Alessandro Ronca
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
-->

# MinMax RNCs — Architecture

[Index](model.md) · [Mathematical foundation](foundations.md) · [Architecture](architecture.md) · [Configuration and implementation](configuration.md)

Notation as in [Mathematical foundation](foundations.md): 
the symbol '$`\cdot`$' denotes ordinary multiplication over scalars, vectors, or matrices; 
the symbol '$`*`$' denotes element-wise (Hadamard) product of vectors; 
the symbol '$`\sigma`$' denotes the logistic sigmoid.

The module tree of [`MinMaxRNC_LM`](../src/minmaxrnc/minmax_rnc_lm.py), as `attribute: Type` (the convention of PyTorch's
`print(model)`), listed in data-flow order. `MinMaxRNC_LM` is a subclass of [`MinMaxRNC`](../src/minmaxrnc/minmax_rnc.py), the
backbone: the backbone has the same tree without `token_emb`, `head_drop` and `lm_head`.

```
MinMaxRNC_LM
├── token_emb:            Embedding(vocab_size, d_model)
├── layers:               ModuleList of n_layers × MinMaxLayer
│   ├── norm_conv:        LayerNorm | RMSNorm | Identity
│   ├── conv:             BasicConv | GatedConv
│   ├── norm_ffn:         LayerNorm | RMSNorm | Identity
│   ├── ffn:              FeedForward | GatedFeedForward
│   ├── norm_neuron:      LayerNorm | RMSNorm | Identity
│   ├── neuron:           MinMaxNeuron, the scalar_neurons scalar-state neurons (None if 0)
│   └── neuron_groups:    ModuleList of MinMaxNeuronGroup, one per entry of `neurons`
├── postlayers_ffn_norm:  LayerNorm | RMSNorm | Identity (None unless use_postlayers_ffn)
├── postlayers_ffn:       FeedForward | GatedFeedForward (None unless use_postlayers_ffn)
├── postlayers_norm:      LayerNorm | RMSNorm (None if postlayers_norm='none')
├── head_drop:            Dropout
└── lm_head:              Linear(d_model, vocab_size, bias=False), weights tied to token_emb by default
```

The package's classes in the tree are defined in:

| Class | Source file |
|---|---|
| `MinMaxRNC_LM` | [`minmax_rnc_lm.py`](../src/minmaxrnc/minmax_rnc_lm.py) |
| `MinMaxRNC` | [`minmax_rnc.py`](../src/minmaxrnc/minmax_rnc.py) |
| `MinMaxLayer` | [`minmax_layer.py`](../src/minmaxrnc/minmax_layer.py) |
| `MinMaxNeuronGroup`, `MinMaxNeuron` | [`minmax_neuron.py`](../src/minmaxrnc/minmax_neuron.py) |
| `BasicConv` | [`modules/basic_conv.py`](../src/minmaxrnc/modules/basic_conv.py) |
| `GatedConv` | [`modules/gated_conv.py`](../src/minmaxrnc/modules/gated_conv.py) |
| `FeedForward`, `GatedFeedForward` | [`modules/feedforward.py`](../src/minmaxrnc/modules/feedforward.py) |

The other modules (`Embedding`, `ModuleList`, `LayerNorm`, `RMSNorm`, `Identity`, `Dropout`,
`Linear`) are PyTorch's.

## MinMax Neuron Groups

[`MinMaxNeuronGroup`](../src/minmaxrnc/minmax_neuron.py) provides $`K`$ = `num_neurons` neurons with the same state
dimension $`d`$, fed the same input $`u_t`$. Each neuron $`k`$ has its own parameters
$`R_t^{(k)} \in ℝ^{d \times d}`$ and $`s_t^{(k)} \in ℝ^d`$, and its own state
$`x_t^{(k)} \in ℝ^d`$. The group computes them for all its neurons at once:

1. two linear layers project $`u_t`$ to the parameters of every neuron: 
   one linear layer `Linear(d_model, K·d²)` produces a vector `r` of dimension
   $`K \cdot d^2`$, that is split into the $`K`$ matrices $`R_t^{(k)}`$;
   and the other linear layer `Linear(d_model, K·d)` produces a vector `s` of dimension
   $`K \cdot d`$, that is split into the $`K`$ vectors $`s_t^{(k)}`$;
2. the states of all $`K`$ neurons are updated together, as one batched computation;
3. the states are projected back to the residual-stream dimension and summed over the neurons, by
   a single `Linear(K·d, d_model)` applied to the concatenated states.

This is exactly equivalent to $`K`$ separate neurons whose outputs are summed ─ 
only the compute is batched.

The states are updated with the parallel prefix-scan over chunks of `unroll_steps` steps, carrying
the state from one chunk to the next: 
one scan over the whole sequence when `unroll_steps` equals its length, and a plain step-by-step
recurrence when `unroll_steps = 1`, e.g. for token-by-token generation
(see [Parallel computation](foundations.md#parallel-computation-via-prefix-scan)).

The group's state is the flat vector of its `num_neurons · d` coordinates (neuron-major). 
For d = 1 the group is evaluated with the element-wise scan and every parameter, tensor and
operation is exactly that of 0.3.0's MinMax Neuron. 
[`MinMaxNeuron`](../src/minmaxrnc/minmax_neuron.py) (the 0.3.0 class) is the group of [`MinMaxNeuronConfig.d_state`](../src/minmaxrnc/minmax_neuron.py) scalar-state neurons,
which is how the layer holds the `scalar_neurons` shorthand.

**Learnable parameters:** weights $`W_\mathrm{s}`$, $`W_\mathrm{r}`$, $`W_\mathrm{o}`$ and biases
$`b_\mathrm{s}`$, $`b_\mathrm{r}`$, $`b_\mathrm{o}`$; with the output gate also $`W_\mathrm{g}`$,
$`b_\mathrm{g}`$; with `train_init`, the initial state $`x_0`$.


## MinMax Layer

One residual block. The data flow uses pre-norm; 
only the neurons' output is added back to the residual stream:

```
conv_in   = ConvNorm(u)
conv      = Conv(conv_in)
ffn_in    = FFNNorm(u + conv)
ffn       = FFN(ffn_in)
neuron_in = NeuronNorm(u + conv + ffn)
neuron    = Sum( NeuronGroup(neuron_in) for each NeuronGroup )
output    = u + neuron
```

where operations are capitalised and variables are lowercase:

| Operation | What it is | Described in |
|---|---|---|
| `ConvNorm`, `FFNNorm`, `NeuronNorm` | pre-norms (`norm_conv`, `norm_ffn`, `norm_neuron` in the tree above): `LayerNorm` (default), `RMSNorm`, or none, set by `norm` | [`MinMaxRNCConfig`](configuration.md#minmaxrncconfig) |
| `Conv` | mixing of the current input with the previous one: `BasicConv` (default) or `GatedConv`, set by `conv_type` | [Convolution Variants](#convolution-variants) |
| `FFN` | feature-mixing feed-forward network: `GatedFeedForward` (default) or `FeedForward`, set by `ffn_type` | [Feed-forward Variants](#feed-forward-variants) |
| `NeuronGroup` | a group of MinMax neurons, evaluated as described above | [MinMax Neuron Groups](#minmax-neuron-groups) |

The sum ranges over all the layer's neuron groups — the group of the `scalar_neurons` (it is
`neuron` in the tree above) and each group of `neurons` (`neuron_groups`) — which all read the same
normalised input `neuron_in`. Recurrent states are left out: the convolution and every neuron
group also take the state from the previous chunk and return the updated one. Each group's output
projection is initialised with `wang_init_` using the layer's total state
dimension, so the layer's overall contribution to the residual stream stays O(1) however its state
is split across groups.

`Conv` and `FFN` outputs are used to compute a richer input for the neuron but are not independently
added to the residual stream. The convolution provides short-range context (one previous input).  
The `FFN` mixes features. The neurons integrate information over arbitrarily long ranges as they
maintain a state. 


## Convolution Variants

**[BasicConv](../src/minmaxrnc/modules/basic_conv.py)** (default) — a learned affine projection of the concatenated previous and current
token, $`\mathrm{out}_t = W_c \cdot [u_{t-1}; u_t] + b_c`$, mapping back to $`ℝ^{d_\mathrm{model}}`$; more expressive
local mixing at the cost of 2× the parameters.

**[GatedConv](../src/minmaxrnc/modules/gated_conv.py)** — a learned per-feature scalar gate $`g \in ℝ^{d_\mathrm{model}}`$ interpolates
between the previous and current token:

```math
\mathrm{out}_t = \sigma(g) * u_{t-1} + (1 - \sigma(g)) * u_t
```

where $`*`$ is the element-wise product: each feature mixes the previous and
the current token with its own weight $`\sigma(g_i)`$.

## Feed-forward Variants

| Class | Formula | Paper |
|---|---|---|
| `FeedForward` | $`W_2 \cdot \phi(W_1 \cdot x + b_1) + b_2`$ | Vaswani et al., NeurIPS 2017 |
| `GatedFeedForward` | $`W_2 \cdot (\phi(\mathrm{gate}) * \mathrm{value}) + b_2`$, where $`[\mathrm{gate}; \mathrm{value}] = W_1 \cdot x + b_1`$ | Dauphin et al., ICML 2017; Shazeer 2020 |

Here $`\phi`$ is the activation selected by `ffn_act_fn`, $`\cdot`$ matrix
multiplication and $`*`$ the element-wise product. The gated variant is the default (`ffn_type='gated'`). With `ffn_act_fn='relu'`
it is **ReGLU**; with `ffn_act_fn='swish'` it is **SwiGLU**.

## MinMax RNC

Stacks `n_layers` MinMaxLayers.  An optional post-layers FFN and normalisation
are applied after the final layer.

## Language Model Wrapper

`MinMaxRNC_LM` adds:
- Token embedding with `small_init_` initialisation.
- A dropout before the LM head.
- An optional weight tie between the embedding and the LM head.

## References

- Vaswani et al. (2017). *Attention Is All You Need.* NeurIPS.
  https://arxiv.org/abs/1706.03762

- Dauphin et al. (2017). *Language Modeling with Gated Convolutional Networks.* ICML.
  https://arxiv.org/abs/1612.08083

- Shazeer (2020). *GLU Variants Improve Transformer.*
  https://arxiv.org/abs/2002.05202
