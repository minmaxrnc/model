# MinMax Recurrent Neural Cascades

[Project website](https://minmaxrnc.org/) &nbsp;·&nbsp; [NeurIPS paper](https://openreview.net/forum?id=8sPcuoX2cj) &nbsp;·&nbsp;
[arXiv paper](https://arxiv.org/abs/2605.06384) &nbsp;·&nbsp; [Documentation](docs/README.md)

---

A parallelisable sequence model, based a novel form of recurrence — 
*truly for long-sequence processing.* 


## Key Properties


- **High expressivity with *perfect memory*.** A MinMax neuron can store and keep information
  for arbitrarily many steps without corrupting it, while being able to manipulate it in a rich way.

- **Guaranteed stability.** States, activations, and gradients stay bounded no matter how long the
  input sequence is, because MinMax recurrence operates by *selecting* values (in contrast with 
  other forms of recurrence such as the ones employed in Linear RNNs, where changes in state values
  can add up over time in an unbounded manner).

- **Efficient parallel and recurrent evaluation.**
  In addition to recurrence, MinMax RNCs admit *parallel scan* as an evaluation strategy, 
  with the possibility of combining the two strategies.
  
  *Parallel training.* All states across a sequence of length $`T`$ can be computed simultaneously
    in depth $`O(\log T)`$ and work $`O(T \cdot \log T)`$ with no sequential bottleneck — or, to control memory
    usage over long sequences, they can be computed in chunks of $`k`$ steps using recurrence across
    chunks (`unroll_steps = k`).
    All strategies give identical results; see
    [Parallel computation](docs/foundations.md#parallel-computation-via-prefix-scan) for their
    computational cost.

  *Efficient inference.* Recurrence enables inference with per-token compute and memory
    usage that is independent of the sequence/context length (fixed-size state, no growing cache), 
    making MinMax RNCs practical for long-context streaming generation.

- **Long-range credit assignment.** Gradients can backpropagate to earlier tokens without vanishing,
  no matter how many steps back they are, and 
  so MinMax RNCs can learn *arbitrarily-long range dependencies*. 


## Measured Performance

- **Ahead of SOTA recurrent models on long sequences.** On synthetic benchmarks testing
  long-term memory, sequential pattern matching, and recall capabilities, 
  MinMax RNCs achieve perfect validation accuracy 
  when trained on sequences of length up to 512 tokens, and retain the same accuracy on sequences of
  over 1M tokens; in contrast, the SOTA recurrent models Mamba, mLSTM, xLSTM, DeltaNet, and sLSTM
  lose accuracy as the sequences grow and eventually drop to random-guess accuracy. 
  On the word problems of the symmetric groups $`S_3`$ and $`S_4`$ 
  (that formally require a higher expressivity compared to the above benchmarks)
  MinMax RNCs again achieve perfect accuracy with seamless generalisation on sequences of over 
  1M tokens, 
  in contrast with DeltaProduct — one of the few models having the required expressivity —
  that loses accuracy as the sequences grow eventually drops to random-guess accuracy. 
- **Competitive at scale.** On OpenWebText, a MinMax RNC of 112M parameters reaches the same
  validation loss (3.12) as GPT-2 of 124M parameters.

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

## Configuration

### Presets

Three ready-made sizes. Each is an ordinary [`MinMaxRNCConfig`](src/minmaxrnc/minmax_rnc.py), so it can be used as it is or as a
starting point:

```python
import dataclasses
from minmaxrnc import MinMaxRNC_LM, MinMaxRNCConfig, MinMaxRNCLMConfig

cfg = MinMaxRNCConfig.medium()                    # as in the table below
cfg = MinMaxRNCConfig.medium(n_layers=4)          # same width, fewer layers
cfg = dataclasses.replace(MinMaxRNCConfig.medium(), norm='rmsnorm')   # any other field

# The 112M language model mentioned above: the large preset with the gated convolution
model = MinMaxRNC_LM(50257, MinMaxRNCLMConfig(backbone=MinMaxRNCConfig.large(), conv_type='gated'))
```

| Preset   | `d_model` | `n_layers` | `scalar_neurons` | Parameters (backbone) | Parameters (LM, GPT-2 vocab) |
|----------|-----------|------------|------------------|-----------------------|------------------------------|
| `small`  | 90        | 2          | 40               | ~0.1 M                | ~4.7 M                       |
| `medium` | 512       | 8          | 512              | ~20.8 M               | ~46.6 M                      |
| `large`  | 728       | 12         | 1456             | ~88.6 M               | ~125.2 M                     |

See [Presets](docs/configuration.md#presets) for the details.

### Custom configuration

Each layer holds arrays of MinMax neurons. A neuron can have any state dimension $`d`$: scalar-state
neurons ($`d = 1`$) are the cheapest, and larger $`d`$ lets a neuron track richer state, e.g.
permutations of $`d`$ elements.

```python
from minmaxrnc import MinMaxRNC, MinMaxRNCConfig, NeuronArray

cfg = MinMaxRNCConfig(
    d_model          = 768,
    n_layers         = 12,
    scalar_neurons   = 192,                           # 192 scalar-state neurons per layer
    neurons          = [
        NeuronArray(d_state=3, num_neurons=16),       # 16 neurons with 3-dimensional state
        (4, 2),                                       # tuple shorthand: d_state=4, num_neurons=2
    ],
    norm             = 'rmsnorm',
    ffn_type         = 'gated',
    ffn_act_fn       = 'swish',                       # → SwiGLU
    output_gate      = True,
    use_postlayers_ffn = True,
)
model = MinMaxRNC(cfg)
```

The neurons of an array are evaluated together, as one batched computation. The cost of a neuron
grows as $`d^3`$, so keep $`d`$ small (≈ 2–8) and scale with more neurons instead. See
[Configuration](docs/configuration.md) for every field, and
[MinMax Neuron Arrays](docs/architecture.md#minmax-neuron-arrays) for the cost.

### Mixed precision

Two settings, both off by default: `amp_dtype` (`'bf16'` or `'fp16'`) runs the model under
automatic mixed precision, and `recurrence_dtype` sets the dtype of the MinMax recurrence itself:

```python
import dataclasses
from minmaxrnc import MinMaxRNCConfig

cfg = dataclasses.replace(MinMaxRNCConfig.medium(), amp_dtype='bf16')
```

See [Numerical precision](docs/precision.md) for what each setting affects, and recommended
combinations.

## How it works

**MinMax recurrence.** A MinMax neuron with state dimension $`d`$ updates its state
$`x_t \in ℝ^d`$ by

```math
x_t = (R_t \otimes x_{t-1}) \oplus s_t
```

where the matrix $`R_t \in ℝ^{d \times d}`$ and the vector $`s_t \in ℝ^d`$ are affine projections
of the current input. It is an ordinary affine recurrence in which addition and multiplication are
replaced by max and min, i.e. $`\oplus`$ and $`\otimes`$ are the matrix operations of the
MinMax algebra. For scalar states ($`d = 1`$) it reads
$`x_t = \mathrm{max}(\mathrm{min}(r_t, x_{t-1}), s_t)`$: the reset signal $`r_t`$ caps the previous
state and the set signal $`s_t`$ raises it. Every state value is selected from the previous state or
the current inputs, so values can be held exactly for any number of steps, and states stay bounded.
Compositions of MinMax recurrence maps are again MinMax recurrence maps, which is what allows the
parallel scan.

**MinMax RNCs.** A MinMax RNC is a stack of layers on a residual stream. Each layer has three
sub-modules, each applied to a normalised input (pre-norm): a short **convolution** over the
current and the previous token, a **feed-forward network** for feature mixing, and **arrays of
MinMax neurons**, which receive the output of the other two. Only the neurons' output, summed over
all neurons, is added back to the residual stream.

See [Mathematical foundations](docs/foundations.md) and [Architecture](docs/architecture.md).

## Documentation

- [Mathematical foundations](docs/foundations.md) — the MinMax algebra and recurrence, and how it
  is computed in parallel.
- [Architecture](docs/architecture.md) — the modules, from neuron arrays to the language model,
  and their initialisation.
- [Configuration](docs/configuration.md) — every configuration field, and the presets.
- [Numerical precision](docs/precision.md) — mixed precision and the dtype of the recurrence.
- [Upgrading from 0.3.0](docs/upgrading.md) — 0.4.0 is backward compatible with 0.3.0: no code
  changes are needed, and 0.3.0 checkpoints load as they are.

## How to cite

```bibtex
@inproceedings{ronca2026minmax,
  title={{MinMax} Recurrent Neural Cascades},
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
