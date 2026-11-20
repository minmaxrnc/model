# MinMax Recurrent Neural Cascades

A parallelisable recurrent sequence model built on the **MinMax operator** — 
expressively powerful, efficiently implementable, and provably not affected by
vanishing or exploding gradient.

Resources:
- [Project website](https://minmaxrnc.org/) for an overview, results, and news.
- [Paper](https://arxiv.org/abs/2605.06384) for the formal description and analyses.
- <!-- github-only -->Architecture reference in [`docs/model.md`](docs/model.md).<!-- /github-only --><!-- pypi-only [Repository](https://github.com/minmaxrnc/model) including architecture reference in `docs/model.md`. -->

## Key properties

- **Perfect memory over long contexts.** A MinMax neuron can hold a value for arbitrarily many steps
  without corrupting it, and hence it can track states such as "has X happened yet?" or 
  "which phase are we in?" over arbitrarily long sequences. 
  Models trained on short sequences have been evaluated without loss of accuracy on sequences of
  over a million tokens. (Formally: all star-free languages with scalar-state neurons, up to all
  regular languages with neurons of larger state dimension)
- **Parallel, recurrent, or anything in between.** The same model can be evaluated in parallel over
  a whole sequence with a prefix scan ($`O(\log T)`$ depth), step by step as a recurrence (constant
  compute and memory per token, no KV cache), or by interleaving the two: parallel over chunks of $`k`$
  steps, recurrent across chunks ($`O((T/k) \cdot \log k)`$ depth). The value of $`k`$ can be set 
  with `unroll_steps`. It is fully parallel when $`k = T`$, and fully recurrent when $`k = 1`$.
  All choices give identical results. Typically one would train and prefill prompts in parallel,
  then generate token by token, and use an intermediate $`k`$ to process sequences too long to scan in
  one go.
- **Stable by construction.** States, activations, and gradients stay bounded, independently of how
  long the input sequence is, because the recurrence produces states by simply *selecting* scalars
  from the previous state or current inputs (via min and max operations). 
- **Long-range credit assignment.** Information from early tokens can reach the end of the sequence
  unchanged, and so can the gradient back to them, so models can learn dependencies spanning the
  whole context (i.e., they do not suffer from vanishing state gradients).
- **Competitive at scale.** On OpenWebText, a MinMax RNC of 112M parameters reaches the same
  validation loss 3.12 as GPT-2 of 124M parameters.
- **Plain PyTorch.** No custom CUDA kernels. Works with `torch.compile` and 
  mixed precision (`amp_dtype`, `scan_dtype`), and is a drop-in sequence backbone or language model.


## The model

Each layer contains three sub-modules applied sequentially with pre-norm and
residual connections:

1. **Convolution** — one-step causal mixing.
2. **Feed-forward network** — feature mixing (gated or standard MLP).
3. **MinMax neurons** — the recurrent cells. A MinMax neuron with state dimension $`d`$ updates
   its state $`x_t \in \mathbb{R}^d`$ by the MinMax recurrence
   $`x_{t+1} = (R_{t+1} \otimes x_t) \oplus s_{t+1}`$
   where the matrix $`R_{t+1} \in \mathbb{R}^{d \times d}`$ and the vector
   $`s_{t+1} \in \mathbb{R}^d`$ are computed from the input to the neuron.
   MinMax recurrence is obtained from ordinary affine recurrence by replacing sum and product
   operations $`(+, \times)`$ on vectors and matrices with their MinMax counterparts $`(\oplus, \otimes)`$, 
   that replace scalar addition and multiplication with max and min, respectively,
   and extend the resulting operations to vectors and matrices in the usual way.
   With $`d = 1`$ the state is a scalar and the recurrence can be written as
   $`x_{t+1} = \max(\min(r_{t+1}, x_t), s_{t+1})`$;
   with larger $`d`$ the state coordinates interact. 
   Each layer holds groups of neurons, of any state dimensions, evaluated with a parallel prefix
   scan, and sums their outputs 
   (see [Neurons of any state dimension](#neurons-of-any-state-dimension)).



## Installation

```bash
pip install minmaxrnc
```

PyTorch (≥ 2.0) is required. For GPU support, follow the
[PyTorch installation guide](https://pytorch.org/get-started/locally/) before
installing this package.

## Quick start

### Sequence backbone

```python
import torch
from minmaxrnc import MinMaxRNC, MinMaxRNCConfig

model = MinMaxRNC(MinMaxRNCConfig.medium())   # d_model=512

u = torch.randn(batch_size, seq_len, 512)

# Parallel over the full sequence (training)
y = model(u, unroll_steps=seq_len)            # (B, T, 512)

# Carry state across calls (streaming inference)
y, state = model(u, seq_len, return_state=True)
y_next   = model(u_next, seq_len, state=state)
```

### Language model

```python
import torch
from minmaxrnc import MinMaxRNC_LM, MinMaxRNCLMConfig, MinMaxRNCConfig

model = MinMaxRNC_LM(
    vocab_size = 50257,
    cfg = MinMaxRNCLMConfig(backbone=MinMaxRNCConfig.medium()),
)

tokens = torch.randint(0, 50257, (batch_size, seq_len))
logits = model(tokens, unroll_steps=seq_len)      # (B, T, vocab_size)

# Autoregressive generation: prefill on prompt, then generate token by token
logits, state = model(tokens, unroll_steps=seq_len, return_state=True)
for _ in range(max_new_tokens):
    next_tok = logits[:, -1].argmax(-1, keepdim=True)
    logits, state = model(next_tok, unroll_steps=1, state=state, return_state=True)
```

### Custom configuration

```python
from minmaxrnc import MinMaxRNC, MinMaxRNCConfig

cfg = MinMaxRNCConfig(
    d_model          = 768,
    n_layers         = 12,
    d_state          = 192,       # 192 scalar-state neurons per layer
    norm             = 'rmsnorm',
    ffn_type         = 'gated',
    ffn_act_fn       = 'swish',   # → SwiGLU
    output_gate      = True,
    use_postlayers_ffn = True,
)
model = MinMaxRNC(cfg)
```

#### Preset sizes

| Preset   | `d_model` | `n_layers` | `d_state` | Parameters (backbone) | Parameters (LM, GPT-2 vocab) |
|----------|-----------|------------|-----------|-----------------------|------------------------------|
| `small`  | 90        | 2          | 40        | ~0.1 M                | ~4.6 M                       |
| `medium` | 512       | 8          | 512       | ~16.6 M               | ~42.4 M                      |
| `large`  | 728       | 12         | 1456      | ~75.9 M               | ~112.5 M                     |

### Neurons of any state dimension

A MinMax neuron can have any state dimension $`d`$. Its state transition is a
$`d \times d`$ matrix applied in the (max, min) semiring,
$`(R \otimes x)_i = \max_j \min(R_{ij}, x_j)`$;
a neuron with state dimension $`d`$ can realise permutation semiautomata of
degree $`d`$. Each layer holds one or
more **neuron groups**, each made of `num_neurons` neurons with the same
state dimension, configured through `neurons` (and `d_state`, see below):

```python
from minmaxrnc import MinMaxRNC, MinMaxRNCConfig, NeuronGroup

cfg = MinMaxRNCConfig(
    d_model = 90,
    n_layers = 2,
    d_state = 64,                                   # 64 scalar-state neurons
    neurons = [
        NeuronGroup(d_state=3, num_neurons=16),     # 16 neurons with 3-dimensional state
        (4, 2),                                     # tuple shorthand: d_state=4, num_neurons=2
    ],
)
model = MinMaxRNC(cfg)
```

Each entry of `neurons` is a `NeuronGroup(d_state, num_neurons=1, n_h=0,
normalize_keys=False)` or the equivalent tuple/list (handy for YAML configs).
`MinMaxRNCConfig.d_state = N` is shorthand for the group
`NeuronGroup(d_state=1, num_neurons=N)` — `N` scalar-state neurons — and is
how 0.3.0 configured a layer. Each group runs as a single batched parallel
scan, and all neuron outputs of a layer are summed.

Things to know:

- **Cost.** Scalar-state neurons (`d_state=1`, in either form) use an
  element-wise scan: $`O(T \log T)`$ work per neuron, exactly the 0.3.0
  computation. For $`d > 1`$ each scan step composes $`d \times d`$ matrices: work is
  $`O(K \cdot T \log T \cdot d^3)`$ for $`K`$ = `num_neurons`, and the scan materialises a
  `(B·num_neurons, T, d, d, d)` intermediate. Keep $`d`$ small (≈ 2–8) and
  scale with `num_neurons` and more groups instead.
- **`n_h > 0` is experimental.** It builds $`R_t`$ as a product of `n_h`
  rank-1 corrections to the identity (DeltaProduct-style,
  $`\prod_i (I - \beta_i k_i k_i^\top)`$,
  optionally with `normalize_keys`), with fewer
  parameters than the default unconstrained projection. It has produced no
  positive results in our experiments so far.
- **Evidence so far.** On group state-tracking benchmarks, neurons with
  state dimension 3 and `n_h=0` learn $`S_3`$ and generalise to much longer sequences than seen in
  training, but results vary across seeds; $`S_4`$ is solved only on some seeds,
  and larger groups ($`A_5`$, $`S_5`$) have not been solved yet.

See [`docs/model.md`](docs/model.md#the-minmax-recurrence) for the maths.

### Mixed precision

New in 0.4.0. Two independent settings, both off by default:

| Setting | Values | Effect |
|---|---|---|
| `amp_dtype` | `None` (default), `'bf16'`, `'fp16'` | When set, `forward()` runs under `torch.autocast(device, dtype=…)` (the LM wrapper also covers the embedding and head). `None` leaves autocast to the caller, as in 0.3.0. |
| `scan_dtype` | `'ambient'` (default), `'fp32'`, `'bf16'`, `'fp16'` | Precision of the MinMax scan. `'ambient'` does no casting (0.3.0 behaviour). The others *pin* the scan to that dtype with autocast disabled, whatever the surrounding precision; the recurrent state is carried in that dtype. |

Common recipes:

| Goal | `amp_dtype` | `scan_dtype` |
|---|---|---|
| Full precision / 0.3.0 behaviour (default) | `None` | `'ambient'` |
| bf16 AMP, full-precision scan | `'bf16'` | `'fp32'` |
| Fully bf16, scan included | `'bf16'` | `'bf16'` |
| bf16 AMP, fp16 scan (untested) | `'bf16'` | `'fp16'` |

```python
import dataclasses
from minmaxrnc import MinMaxRNC, MinMaxRNCConfig

cfg   = dataclasses.replace(MinMaxRNCConfig.medium(), amp_dtype='bf16', scan_dtype='fp32')
model = MinMaxRNC(cfg)
```

Notes:

- The scan is a chain of element-wise min/max selections, not a matmul: it
  gets no tensor-core speed-up and is memory-bandwidth bound, so its dtype's
  byte width largely sets its cost. `'fp32'` is the safest choice but gives
  back part of the bf16 speed-up; `'bf16'` is fastest, but its 8-bit
  mantissa blurs the fine value distinctions the min/max routing relies on.
  `'fp16'` has the same width as bf16 with a 10-bit mantissa (but a smaller
  range). Validate accuracy against `'fp32'` before relying on `'bf16'` or
  `'fp16'`.
- With `scan_dtype='ambient'` under autocast (the 0.3.0 behaviour) the scan
  runs on bf16 projections with an fp32 initial state, so its precision is
  mixed and implicit; pick a pinned `scan_dtype` to make it explicit.
- An explicit `amp_dtype` overrides the dtype of an outer autocast inside the
  model. fp16 training also needs a `torch.amp.GradScaler`, which remains
  the caller's job.

## Upgrading from 0.3.0

0.4.0 is backward compatible with 0.3.0; no code changes are needed.

- **Same API.** Every 0.3.0 class, config field and argument keeps its name,
  position and meaning, including positional `MinMaxRNCConfig(d_model,
  n_layers, d_state)`. New config fields are appended after the existing
  ones and default to the 0.3.0 behaviour (`neurons=()`,
  `scan_dtype='ambient'`, `amp_dtype=None`). `MinMaxNeuron(cfg)` still
  builds the layer's `cfg.d_state` scalar-state neurons; it is now a special
  case of the new `MinMaxNeuronGroup`, which implements neurons of any state
  dimension.
- **Same weights.** With the defaults, a model has the same parameter names,
  shapes and seeded initial values as in 0.3.0, so 0.3.0 checkpoints load
  with `load_state_dict(strict=True)` and no key remapping.
- **Same outputs and computation.** Outputs and recurrent states are
  identical to 0.3.0, in fp32 and under a caller-side `torch.autocast`, and
  the forward pass runs exactly the same sequence of PyTorch operations, so
  speed and memory use are unchanged. The test suite checks all of this
  against reference values recorded with 0.3.0
  (`tests/test_backward_compat.py`).
- **Recurrent state format.** Per layer it is still
  `{'neuron': …, 'conv': …}`; models with `neurons` groups add a
  `'neuron_groups': [...]` entry (one `(B, num_neurons · d_state)` state per
  group), and models with `d_state=0` have no `'neuron'` entry.
- **Small behaviour changes**, only for configs 0.3.0 rejected anyway:
  `d_state` now defaults to 0, so leaving it out raises a `ValueError`
  (instead of a `TypeError`) unless `neurons` is given; invalid
  `scan_dtype`/`amp_dtype` values raise `ValueError`.
- **From the matrix-neuron experiment branch**, rename config keys
  `diagonal_d_state → d_state` and `d_states → neurons` (the
  `(d_state, num_neurons, n_h)` triples are accepted as-is), plus the global
  `normalize_keys` becomes a per-group field. Its checkpoints use different
  parameter names (`neurons.N.*`) and do not load directly.

See [CHANGELOG.md](CHANGELOG.md) for the full list of changes.

## Running the tests

```bash
pytest
```

## How to cite

```bibtex
@misc{ronca2026minmaxpaper,
      title={{MinMax} Recurrent Neural Cascades},
      author={Alessandro Ronca},
      year={2026},
      eprint={2605.06384},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2605.06384},
}
@software{ronca2026minmaxcode,
  author  = {Alessandro Ronca},
  title   = {{MinMax} Recurrent Neural Cascades},
  year    = {2026},
  url     = {https://github.com/minmaxrnc/model},
  version = {0.3.0},
}
```


## License

This project is source-available under the PolyForm Noncommercial License 1.0.0.

You may use, copy, modify, and distribute this software only for non-commercial purposes under the terms of that license.

Commercial use is not permitted without a separate commercial license from the copyright holder.

For commercial licensing, contact:

**Alessandro Ronca**
alessandro.ronca@iris-ai.org

## Third-party dependencies

This project depends on third-party software, including Python and PyTorch. 
These dependencies are licensed separately by their respective copyright holders.

See `THIRD_PARTY_NOTICES.md` for details.
