# MinMax Recurrent Neural Cascades

A parallelisable recurrent architecture truly for long-sequence processing — 
expressively powerful, efficiently implementable, provably stable, 
and not affected by vanishing gradients.

Resources:
- [Project website](https://minmaxrnc.org/) for an overview, results, and news.
- [NeurIPS paper](https://openreview.net/forum?id=8sPcuoX2cj) and 
  [arXiv paper](https://arxiv.org/abs/2605.06384) for the formal description and analyses.
  <!-- github-only -->
- Documentation in [`docs/`](docs/model.md): [mathematical foundation](docs/foundations.md),
  [architecture](docs/architecture.md), and [configuration and implementation](docs/configuration.md).
  <!-- /github-only -->
  <!-- pypi-only
  - [Repository](https://github.com/minmaxrnc/model), including the [documentation](docs/model.md) on the
    mathematical foundation, architecture, and configuration.
  -->

## Key properties

- **Perfect memory over long contexts.** A MinMax neuron can hold a value for arbitrarily many steps
  without corrupting it, and hence it can track states such as "has X happened yet?" or 
  "which phase are we in?" over arbitrarily long sequences. 
  Models trained on short sequences have been evaluated without loss of accuracy on sequences of
  over a million tokens. 
  *Formal expressivity:* all star-free languages with scalar-state neurons, up to all regular
  languages with neurons of larger state dimension.
- **Parallel, recurrent, or anything in between.** The same model can be evaluated in any of the
  following ways. 
  - In parallel over a whole sequence via parallel scan. 
    *Computational cost:* depth $`O(\log(T))`$ and work $`O(T)`$.
  - Step by step via a recurrence. 
    *Computational cost:* constant compute and memory per token. 
  - By interleaving parallel and recurrence evaluation: parallel scan over chunks of $`k`$ steps, 
    with recurrence across chunks. 
    The value of $`k`$ can be set with `unroll_steps`. 
    It is fully parallel when $`k = T`$, and fully recurrent when $`k = 1`$.
    All choices give identical results. 
    Typically one would train and prefill prompts in parallel, then generate token by token, and use
    an intermediate $`k`$ to process sequences too long to scan in one go.
    *Computational cost:* depth $`O((T/k) \cdot \log(k))`$ and work $`O(T \cdot \log(k))`$.
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
3. **MinMax neurons** — the recurrent cells. A MinMax neuron updates its state $`x_t \in ℝ^d`$ by
   the MinMax recurrence $`x_{t+1} = (R_{t+1} \otimes x_t) \oplus s_{t+1}`$
   where the matrix $`R_{t+1} \in ℝ^{d \times d}`$ and the vector
   $`s_{t+1} \in ℝ^d`$ are computed from the input to the neuron.
   MinMax recurrence is obtained from ordinary affine recurrence by replacing sum and product
   operations $`(+, \times)`$ on vectors and matrices with their MinMax counterparts $`(\oplus, \otimes)`$, 
   that replace scalar addition and multiplication with max and min, respectively,
   and extend the resulting operations to vectors and matrices in the usual way.
   With $`d = 1`$ the state is a scalar and the recurrence can be written as
   $`x_{t+1} = \mathrm{max}(\mathrm{min}(r_{t+1}, x_t), s_{t+1})`$;
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
    scalar_neurons   = 192,       # 192 scalar-state neurons per layer
    norm             = 'rmsnorm',
    ffn_type         = 'gated',
    ffn_act_fn       = 'swish',   # → SwiGLU
    output_gate      = True,
    use_postlayers_ffn = True,
)
model = MinMaxRNC(cfg)
```

#### Preset sizes

| Preset   | `d_model` | `n_layers` | `scalar_neurons` | Parameters (backbone) | Parameters (LM, GPT-2 vocab) |
|----------|-----------|------------|------------------|-----------------------|------------------------------|
| `small`  | 90        | 2          | 40               | ~0.1 M                | ~4.6 M                       |
| `medium` | 512       | 8          | 512              | ~16.6 M               | ~42.4 M                      |
| `large`  | 728       | 12         | 1456             | ~75.9 M               | ~112.5 M                     |

### Neurons of any state dimension

A MinMax neuron can have any state dimension $`d`$. Its state transition is a
$`d \times d`$ matrix applied in the (max, min) semiring,
$`(R \otimes x)_i = \mathrm{max}_j\, \mathrm{min}(R_{ij}, x_j)`$;
a neuron with state dimension $`d`$ can realise permutation semiautomata of
degree $`d`$. Each layer holds one or
more **neuron groups**, each made of `num_neurons` neurons with the same
state dimension, configured through `neurons` and `scalar_neurons`:

```python
from minmaxrnc import MinMaxRNC, MinMaxRNCConfig, NeuronGroup

cfg = MinMaxRNCConfig(
    d_model = 90,
    n_layers = 2,
    scalar_neurons = 64,                            # 64 scalar-state neurons
    neurons = [
        NeuronGroup(d_state=3, num_neurons=16),     # 16 neurons with 3-dimensional state
        (4, 2),                                     # tuple shorthand: d_state=4, num_neurons=2
    ],
)
model = MinMaxRNC(cfg)
```

Each entry of `neurons` is a `NeuronGroup(d_state, num_neurons=1)` or the
equivalent tuple/list (handy for YAML configs).
`scalar_neurons = N` is shorthand for the group
`NeuronGroup(d_state=1, num_neurons=N)`: `N` scalar-state neurons. The neurons
of a group are evaluated together, as one batched computation — in parallel or
step by step, as set by `unroll_steps` — and all neuron outputs of a layer are
summed.

Things to know:

- **Cost.** Scalar-state neurons (`scalar_neurons`, or groups with `d_state=1`) use an
  element-wise scan: $`O(T \log T)`$ work per neuron. For $`d > 1`$ each scan step composes $`d \times d`$ matrices: work is
  $`O(K \cdot T \log T \cdot d^3)`$ for $`K`$ = `num_neurons`, and the scan materialises a
  `(B·num_neurons, T, d, d, d)` intermediate. Keep $`d`$ small (≈ 2–8) and
  scale with `num_neurons` and more groups instead.
- **Evidence so far.** On group state-tracking benchmarks, neurons with
  state dimension 3 learn $`S_3`$ and generalise to much longer sequences than seen in
  training, but results vary across seeds; $`S_4`$ is solved only on some seeds,
  and larger groups ($`A_5`$, $`S_5`$) have not been solved yet.

See the [mathematical foundation](docs/foundations.md#minmax-recurrence) for the maths.

### Mixed precision

Two independent settings, both off by default:

| Setting | Values | Effect |
|---|---|---|
| `amp_dtype` | `None` (default), `'bf16'`, `'fp16'` | When set, `forward()` runs under `torch.autocast(device, dtype=…)` (the LM wrapper also covers the embedding and head). `None` leaves autocast to the caller. |
| `scan_dtype` | `'ambient'` (default), `'fp32'`, `'bf16'`, `'fp16'` | Precision of the MinMax scan, in both the forward and the backward pass. `'ambient'` does no casting. The others *pin* the scan to that dtype with autocast disabled, whatever the surrounding precision; the recurrent state is carried in that dtype. |

What the scan's precision affects:

- **Forward pass: only if the scan is *less* precise than its inputs.** The scan
  only *selects* among its inputs (min/max), never computes new values. Under
  bf16 AMP its inputs are already bf16, so running it in fp32 gives bitwise the
  same outputs as running it in bf16 (except for a learned initial state, which
  a bf16 scan rounds). In an fp32 model, a bf16 scan does round its inputs.
- **Backward pass: gradient accumulation.** A value that is selected and held
  for many steps receives a gradient contribution from each of them, summed in
  the scan's dtype. An fp32 scan sums them accurately even under bf16 AMP; a
  bf16 scan sums them with bf16's ~3 significant digits, which matters most for
  values held over long ranges.
- **Cost.** The scan is memory-bandwidth bound (min/max, no tensor cores) and
  its intermediates are a large share of activation memory for long
  sequences, so a 4-byte dtype costs about twice a 2-byte one.

Common recipes:

| Goal | `amp_dtype` | `scan_dtype` | Effect |
|---|---|---|---|
| Full precision (default) | `None` | `'ambient'` | everything in fp32 |
| bf16 AMP, fp32 scan | `'bf16'` | `'fp32'` | same forward as a bf16 scan; scan gradients summed in fp32; more scan memory |
| Fully bf16, scan included | `'bf16'` | `'bf16'` | least memory; scan gradients summed in bf16 |
| fp32 model, bf16 scan | `None` | `'bf16'` | only the scan in bf16: it rounds its inputs, saving scan memory and bandwidth |
| bf16 AMP, fp16 scan (untested) | `'bf16'` | `'fp16'` | bf16 values fit fp16 exactly within its range; finer gradient sums than bf16, but overflow above 65504 |

How much the more accurate gradients of an fp32 scan help training has not
been measured yet; validate `'bf16'` against `'fp32'` on your task.

```python
import dataclasses
from minmaxrnc import MinMaxRNC, MinMaxRNCConfig

cfg   = dataclasses.replace(MinMaxRNCConfig.medium(), amp_dtype='bf16', scan_dtype='fp32')
model = MinMaxRNC(cfg)
```

Notes:

- With `scan_dtype='ambient'` under autocast the scan
  runs on bf16 projections with an fp32 initial state, so its precision is
  mixed and implicit; pick a pinned `scan_dtype` to make it explicit.
- An explicit `amp_dtype` overrides the dtype of an outer autocast inside the
  model. fp16 training also needs a `torch.amp.GradScaler`, which remains
  the caller's job.

## Upgrading from 0.3.0

0.4.0 is backward compatible with 0.3.0: no code changes are needed, and 0.3.0 checkpoints load
as they are. See [docs/upgrading.md](docs/upgrading.md) for the details, and
[CHANGELOG.md](CHANGELOG.md) for the full list of changes.

## Running the tests

```bash
pytest
```

## How to cite

```bibtex
@inproceedings{ronca2026minmax,
  title={MinMax Recurrent Neural Cascades},
  author={Alessandro Ronca},
  booktitle={NeurIPS 2026},
  year={2026},
  url={https://openreview.net/forum?id=8sPcuoX2cj}
}
@misc{ronca2026minmax-arxiv,
  title={{MinMax} Recurrent Neural Cascades},
  author={Alessandro Ronca},
  year={2026},
  eprint={2605.06384},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2605.06384},
}
@software{ronca2026minmax-code,
  author  = {Alessandro Ronca},
  title   = {{MinMax} Recurrent Neural Cascades},
  year    = {2026},
  url     = {https://github.com/minmaxrnc/model},
  version = {0.4.0},
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
