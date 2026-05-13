# MinMax Recurrent Neural Cascades

A parallelisable recurrent sequence model built on the **MinMax operator** — 
expressively powerful, efficiently implementable, and provably not affected by
vanishing or exploding gradient.

Resources:
- [Paper](https://arxiv.org/abs/2605.06384) for the formal description and analyses.
- [Repository](https://github.com/minmaxrnc/model) including architecture reference in `docs/model.md`.


## Key properties

- **Perfect memory.** MinMax neurons can store and retain information arbitrarily long (formal
  expressivity: all group-free functions).

- **Parallel training.** All hidden states across a sequence of length T are
  computed simultaneously in O(log T) depth, with no sequential bottleneck.
- **Efficient inference.** Runs as a true RNN: O(1) compute and O(D) memory
  per token, making it practical for long-context streaming generation.
- **Stable recurrence.** The MinMax operator is bounded and its
  gradients cannot vanish or explode through the state path.

## The model

Each layer contains three sub-modules applied with pre-norm and residual
connections:

1. **MinMax Neuron** — the recurrent cell, updating a hidden state
   `x_{t+1} = max(min(r_t, x_t), s_t)` element-wise in parallel via a prefix scan.
2. **Convolution** — one-step causal mixing.
3. **Feed-forward network** — feature mixing (gated or standard MLP).


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
from minmax import MinMaxRNC, MinMaxRNCConfig

model = MinMaxRNC(MinMaxRNCConfig.medium())   # d_model=512

u = torch.randn(batch_size, seq_len, 512)

# Parallel over the full sequence (training)
y = model(u, unroll_steps=seq_len)            # (B, T, 512)

# Carry state across calls (streaming inference)
y, state = model(u, unroll_steps, return_state=True)
y_next   = model(u_next, unroll_steps, state=state)
```

### Language model

```python
import torch
from minmax import MinMaxRNC_LM, MinMaxRNCLMConfig, MinMaxRNCConfig

model = MinMaxRNC_LM(
    vocab_size = 50257,
    cfg = MinMaxRNCLMConfig(backbone=MinMaxRNCConfig.medium()),
)

tokens = torch.randint(0, 50257, (batch_size, seq_len))
logits = model(tokens, unroll_steps=seq_len)      # (B, T, vocab_size)

# Autoregressive generation
logits, state = model(tokens[:, :1], unroll_steps=seq_len-1, return_state=True)
for _ in range(max_new_tokens):
    next_tok = logits[:, -1].argmax(-1, keepdim=True)
    logits, state = model(next_tok, unroll_steps=1, state=state, return_state=True)
```

### Custom configuration

```python
from minmax import MinMaxRNC, MinMaxRNCConfig

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
  version = {0.1.2},
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
