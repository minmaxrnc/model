# MinMax Recurrent Neural Cascades

A parallelisable recurrent sequence model built on the **MinMax operator** — 
expressively powerful, efficiently implementable, and provably not affected by
vanishing or exploding gradient.

Resources:
- [Project website](https://minmaxrnc.org/) for an overview, results, and news.
- [Paper](https://arxiv.org/abs/2605.06384) for the formal description and analyses.
- [Repository](https://github.com/minmaxrnc/model) including architecture reference in `docs/model.md`.

## Key properties

- **Perfect memory over long contexts.** A MinMax neuron can hold a value for arbitrarily many steps
  without corrupting it, and hence it can track states such as "has X happened yet?" or 
  "which phase are we in?" over arbitrarily long sequences. 
  Models trained on short sequences have been evaluated without loss of accuracy on sequences of
  over a million tokens. (Formally: all star-free languages with scalar-state neurons, up to all
  regular languages with matrix neurons)
- **Parallel, recurrent, or anything in between.** The same model can be evaluated in parallel over
  a whole sequence with a prefix scan (O(log T) depth), step by step as a recurrence (constant
  compute and memory per token, no KV cache), or by interleaving the two: parallel over chunks of k
  steps, recurrent across chunks (O((T/k) · log k) depth). The value of k can be set 
  with `unroll_steps`. It is fully parallel when k = T, and fully recurrent when k = 1.
  All choices give identical results. Typically one would train and prefill prompts in parallel,
  then generate token by token, and use an intermediate k to process sequences too long to scan in
  one go.
- **Stable by construction.** States, activations, and gradients stay bounded, independently of how
  long the input sequence is, because the recurrence produces states by simply *selecting* scalars
  from the previous state or current inputs (via min and max operations). 
- **Long-range credit assignment.** Information from early tokens can reach the end of the sequence
  unchanged, and so can the gradient back to them, so models can learn dependencies spanning the
  whole context (i.e., they do not suffer from vanishing state gradients).
- **Competitive at scale.** On OpenWebText, a 112M-parameter MinMax RNC reaches the same validation
  loss 3.12 as 124M-params GPT-2 124M.
- **Plain PyTorch.** No custom CUDA kernels. Works with `torch.compile` and 
  mixed precision (`amp_dtype`, `scan_dtype`), and is a drop-in sequence backbone or language model.


## The model

Each layer contains three sub-modules applied sequentially with pre-norm and
residual connections:

1. **Convolution** — one-step causal mixing.
2. **Feed-forward network** — feature mixing (gated or standard MLP).
3. **MinMax Neuron** — the recurrent cell, updating a hidden state
   `x_{t+1} = max(min(r_t, x_t), s_t)` element-wise in parallel via a prefix scan.
   Since 0.4.0 a layer can additionally (or instead) hold groups of
   **matrix MinMax neurons**, `x_{t+1} = max(R_t ⊗ x_t, s_t)` in the
   (max,min) semiring — see [Matrix neurons](#matrix-neurons).


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
    d_state          = 192,       # hidden-state dimension per neuron
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

### Matrix neurons

New in 0.4.0. Besides the diagonal neuron (`d_state`), every layer can hold
groups of **matrix MinMax neurons**, whose state transition is a full
`d × d` matrix applied in the (max,min) semiring,
`(R_t ⊗ x)_i = max_j min(R_t[i,j], x_j)`, so state dimensions can mix:

```python
from minmaxrnc import MinMaxRNC, MinMaxRNCConfig, MatrixNeuronGroup

cfg = MinMaxRNCConfig(
    d_model        = 90,
    n_layers       = 2,
    d_state        = 0,                       # no diagonal neuron (or > 0 to keep one)
    matrix_neurons = [
        MatrixNeuronGroup(d_state=3, num_neurons=16),   # 16 independent 3×3 neurons
        (4, 2),                                         # tuple shorthand: d_state=4, num_neurons=2
    ],
)
model = MinMaxRNC(cfg)
```

Each entry is a `MatrixNeuronGroup(d_state, num_neurons=1, n_h=0,
normalize_keys=False)` or the equivalent tuple/list (handy for YAML configs).
The neurons of a group run as a single batched parallel scan; all neuron
outputs of a layer (diagonal and matrix) are summed.

Things to know:

- **Cost.** Each scan step composes `d × d` matrices: work is
  O(num_neurons · T log T · d³) and the scan materialises a
  `(B·num_neurons, T, d, d, d)` intermediate. Keep each group's `d_state`
  small (≈ 2–8) and scale with `num_neurons` and more groups instead.
- **`d_state=1` groups warn.** A 1×1 matrix neuron is a scalar-state
  neuron, equivalent to a diagonal one; use `d_state=num_neurons` on the
  diagonal neuron instead.
- **`n_h > 0` is experimental.** It builds `R_t` as a product of `n_h`
  rank-1 corrections to the identity (DeltaProduct-style,
  `∏ (I − β_i k_i k_iᵀ)`, optionally with `normalize_keys`), with fewer
  parameters than the default unconstrained projection. It has produced no
  positive results in our experiments so far.
- **Evidence so far.** On group state-tracking benchmarks, matrix neurons
  with `n_h=0` learn S3 and generalise to much longer sequences than seen in
  training, but results vary across seeds; S4 is solved only on some seeds,
  and larger groups (A5, S5) have not been solved yet.

See [`docs/model.md`](docs/model.md#the-minmax-recurrence-full-matrix) for
the maths.

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
  ones and default to the 0.3.0 behaviour (`matrix_neurons=()`,
  `scan_dtype='ambient'`, `amp_dtype=None`). `MinMaxNeuron` is still the
  diagonal neuron; the matrix neuron is the new `MatrixMinMaxNeuron`.
- **Same weights.** With the defaults, a model has the same parameter names,
  shapes and seeded initial values as in 0.3.0, so 0.3.0 checkpoints load
  with `load_state_dict(strict=True)` and no key remapping.
- **Same outputs.** Outputs and recurrent states are identical to 0.3.0, in
  fp32 and under a caller-side `torch.autocast`. The test suite checks this
  against reference values recorded with 0.3.0
  (`tests/test_backward_compat.py`).
- **Recurrent state format.** Per layer it is still
  `{'neuron': …, 'conv': …}`; models with matrix neurons add a
  `'matrix_neurons': [...]` entry, and models with `d_state=0` have no
  `'neuron'` entry.
- **Small behaviour changes**, only for configs 0.3.0 rejected anyway:
  `d_state` now defaults to 0, so leaving it out raises a `ValueError`
  (instead of a `TypeError`) unless `matrix_neurons` is given; invalid
  `scan_dtype`/`amp_dtype` values raise `ValueError`.
- **From the matrix-neuron experiment branch**, rename config keys
  `diagonal_d_state → d_state` and `d_states → matrix_neurons` (the
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
