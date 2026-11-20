<!--
SPDX-FileCopyrightText: 2026 Alessandro Ronca
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
-->

# MinMax Recurrent Neural Cascades (MinMax RNCs) — Documentation

See the [NeurIPS paper](https://openreview.net/forum?id=8sPcuoX2cj) and the
[arXiv paper](https://arxiv.org/abs/2605.06384) for the formal description and analyses.

- [Mathematical foundations](foundations.md) — the MinMax algebra, the MinMax recurrence for any
  state dimension, composition closure, and the parallel prefix scan.
- [Architecture](architecture.md) — neuron groups, layers, convolutions, feed-forward networks,
  initialisation, the backbone and the language-model wrapper.
- [Configuration](configuration.md) — every configuration field, and the presets.
- [Numerical precision](precision.md) — mixed precision (`amp_dtype`) and the dtype of the MinMax
  recurrence (`recurrence_dtype`): what they affect, and recommended settings.
- [Upgrading from 0.3.0](upgrading.md) — backward compatibility with 0.3.0: API, weights,
  outputs, legacy names and precision defaults.
