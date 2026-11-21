<!--
SPDX-FileCopyrightText: 2026 Alessandro Ronca
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
-->

# MinMax RNCs — Architecture

[Index](README.md) · [Mathematical foundations](foundations.md) · [Architecture](architecture.md) · [Configuration](configuration.md) · [Numerical precision](precision.md) · [Upgrading from 0.3.0](upgrading.md)

---

This page describes the modules of a MinMax RNC, from neuron groups and layers to the backbone
and the language-model wrapper, and how their parameters are initialised.

**Notation.** As in [Mathematical foundations](foundations.md):
the symbol '$`\cdot`$' denotes ordinary multiplication over scalars, vectors, or matrices; 
the symbol '$`*`$' denotes element-wise (Hadamard) product of vectors; 
the symbol '$`\sigma`$' denotes the logistic sigmoid.

The module tree of [`MinMaxRNC_LM`](../src/minmaxrnc/minmax_rnc_lm.py) is below, 
as `attribute: Type`, listed in data-flow order. 
`MinMaxRNC_LM` is a subclass of [`MinMaxRNC`](../src/minmaxrnc/minmax_rnc.py), 
the backbone: the backbone has the same tree without `token_emb`, `head_drop` and `lm_head`.

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
dimension $`d`$, fed the same input $`u_t`$. Each neuron $`i`$ has its own parameters
$`R_t^{(i)} \in ℝ^{d \times d}`$ and $`s_t^{(i)} \in ℝ^d`$, and its own state
$`x_t^{(i)} \in ℝ^d`$. The group computes them for all its neurons at once:

1. two linear layers project the input to the parameters of every neuron, at all positions at
   once: for an input of shape `(B, T, d_model)` (batch size $`B`$, sequence length $`T`$),
   `Linear(d_model, K·d²)` produces `r` of shape `(B, T, K·d²)`, and `Linear(d_model, K·d)`
   produces `s` of shape `(B, T, K·d)`. At each position $`t`$, the vector $`r_t`$ is split
   into the $`K`$ matrices $`R_t^{(i)}`$, and the vector $`s_t`$ into the $`K`$ vectors
   $`s_t^{(i)}`$;
2. the states of all $`K`$ neurons are updated together, as one batched computation over the
   batch, the neurons and the positions: the parallel scan runs on matrices of shape
   `(B·K, T, d, d)`, with the neurons folded into the batch axis; in recurrent evaluation, all
   neurons advance one step at a time;
3. the states are projected back to the residual-stream dimension and summed over the neurons, by
   a single `Linear(K·d, d_model)` applied to the concatenated states.

This is exactly equivalent to $`K`$ separate neurons whose outputs are summed ─ 
only the compute is batched.

**Cost.** Scalar-state neurons ($`d = 1`$) use an element-wise scan: $`O(T \log T)`$ work per
neuron. For $`d > 1`$ each scan step composes $`d \times d`$ matrices: the work is
$`O(K \cdot T \log T \cdot d^3)`$, and the scan materialises a `(B·K, T, d, d, d)` intermediate.
Keep $`d`$ small (≈ 2–8), and scale with `num_neurons` and more groups instead.

**Known issue: `torch.compile`.** With PyTorch 2.14.1 on CUDA, the default Inductor backend
fails to compile groups with $`d > 1`$ in fp32 for inference (`torch.no_grad()` or
`torch.inference_mode()`), with `InductorError: TypeError: Expected a number but got Mul`.
Compiled training, bf16 AMP and models with scalar-state neurons only are unaffected.
Workaround: before compiling, set

```python
torch._inductor.config.triton.coalesce_tiling_analysis = False
```

which disables the Inductor analysis that fails; on an NVIDIA A100 this made every tested case
compile, with no change in speed. This is a global PyTorch setting, so the library does not set it
itself. Alternatives: evaluate with the uncompiled model, compile with `backend="aot_eager"`
(whose outputs are identical to eager), or run under bf16 AMP.

**Evidence so far.** On group state-tracking benchmarks, neurons with state dimension 3 learn the
symmetric group $`S_3`$ (tracking the permutation of 3 elements) and generalise to much longer
sequences than seen in training, but results vary across seeds; $`S_4`$ is solved only on some
seeds, and the larger groups (the alternating group $`A_5`$ and $`S_5`$) have not been solved yet.

The states are updated with the parallel prefix-scan over chunks of `unroll_steps` steps, carrying
the state from one chunk to the next: 
one scan over the whole sequence when `unroll_steps` equals its length, and a plain step-by-step
recurrence when `unroll_steps = 1`, e.g. for token-by-token generation
(see [Parallel computation](foundations.md#parallel-computation-via-prefix-scan)).

The group's state is the flat vector of its `num_neurons · d` coordinates (neuron-major). 
For d = 1 the group is evaluated with a dedicated element-wise scan.
The layer holds its `scalar_neurons` as a [`MinMaxNeuron`](../src/minmaxrnc/minmax_neuron.py), the group of
[`MinMaxNeuronConfig.d_state`](../src/minmaxrnc/minmax_neuron.py) scalar-state neurons.

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
| [`NeuronGroup`](../src/minmaxrnc/minmax_rnc.py) | a group of MinMax neurons, evaluated as described above | [MinMax Neuron Groups](#minmax-neuron-groups) |

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
token, $`\mathrm{out}_t = W_\mathrm{c} \cdot [u_{t-1}; u_t] + b_\mathrm{c}`$, mapping back to $`ℝ^{d_\mathrm{model}}`$; more expressive
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
- Token embedding with `small_init_init_` initialisation.
- A dropout before the LM head.
- An optional weight tie between the embedding and the LM head.

## Initialisation

Both functions initialise a parameter $`p`$ by sampling $`p \sim \mathcal{N}(0, \mathrm{std}^2)`$; the table gives $`\mathrm{std}`$.
Here $`\mathrm{dim}`$ is the fan-in of the projection being initialised (the size of its input):
$`d_\mathrm{model}`$ for most projections, $`2 \cdot d_\mathrm{model}`$ for BasicConv, and the FFN hidden
size for the FFN down-projection; for the neuron output projection it is the layer's total number of
state coordinates (the sum of $`K \cdot d`$ over the layer's neuron groups, i.e. the group's own
$`K \cdot d`$ when the layer has a single group). $`L`$ = `num_blocks` is the number of residual blocks.

| Function | $`\mathrm{std}`$ | Paper |
|---|---|---|
| `small_init_init_(p, dim)` | $`\sqrt{\dfrac{2}{5 \cdot \mathrm{dim}}}`$ | Nguyen & Salazar, IWSLT 2019 |
| `wang_init_(p, dim, num_blocks)` | $`\dfrac{2}{L \cdot \sqrt{\mathrm{dim}}}`$ | Radford et al. 2019 (GPT-2) + Nguyen & Salazar 2019; used in Beck et al. 2024 (xLSTM) |

`small_init_init_` is applied to all **input-side** projections: token embedding,
LM head (when untied), BasicConv, FFN up-projection, and the neuron $`W_\mathrm{s}`$, $`W_\mathrm{r}`$,
and output-gate $`W_\mathrm{g}`$ projections.  It keeps activations small at initialisation
in deep networks.

`wang_init_` is applied to the two **residual output** projections: the
neuron output $`W_\mathrm{o}`$ and the FFN down-projection.  It additionally divides by $`L`$
(the number of residual blocks) so the total variance contributed by all residual
branches to the stream stays $`O(1/L)`$ regardless of depth — following the GPT-2
scaled-init strategy.

### s and r Projection Initialisation

The `s_r_init` config field selects the initialisation scheme for the $`W_\mathrm{s}`$ and $`W_\mathrm{r}`$ projections
($`W_\mathrm{r}`$ produces $`R_t`$: one entry per neuron for scalar-state neurons, $`d^2`$ per neuron otherwise).

| Scheme | $`W_\mathrm{s}`$ weights | $`W_\mathrm{r}`$ weights | $`b_\mathrm{s}`$ | $`b_\mathrm{r}`$ |
|---|---|---|---|---|
| `'small_init'` | `small_init_init_` | `small_init_init_` | 0 | 0 |
| `'kaiming'` | Kaiming uniform ($`a = \sqrt{5}`$) | Kaiming uniform ($`a = \sqrt{5}`$) | 0 | 0 |
| `'asymmetric'` | Kaiming uniform ($`a = \sqrt{5}`$) | `small_init_init_` | +1.0 | 0 |

The **asymmetric** scheme addresses an initialisation dead zone, seen most easily in the scalar case: with $`x_0 = 0`$ and both projections near zero, `max(min(r, x), s)` keeps every hidden unit at 0 whenever s < 0 < r, preventing state writes early in training.  Setting $`b_\mathrm{s} = +1.0`$ ensures $`s > 0`$ at initialisation so writes happen from the first token; keeping $`W_\mathrm{r}`$ small prevents r from immediately over-writing the written values.

## References

- Vaswani et al. (2017). *Attention Is All You Need.* NeurIPS.
  https://arxiv.org/abs/1706.03762

- Dauphin et al. (2017). *Language Modeling with Gated Convolutional Networks.* ICML.
  https://arxiv.org/abs/1612.08083

- Shazeer (2020). *GLU Variants Improve Transformer.*
  https://arxiv.org/abs/2002.05202

- Radford et al. (2019). *Language Models are Unsupervised Multitask Learners.*
  OpenAI Blog.  https://openai.com/research/language-unsupervised

- Nguyen & Salazar (2019). *Transformers without Tears: Improving the
  Normalization of Self-Attention.* IWSLT.
  https://arxiv.org/abs/1910.05895

- Beck et al. (2024). *xLSTM: Extended Long Short-Term Memory.*
  https://arxiv.org/abs/2405.04517
