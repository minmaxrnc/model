<!--
SPDX-FileCopyrightText: 2026 Alessandro Ronca
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
-->

# MinMax Recurrent Neural Cascades (MinMax RNCs) — Documentation

See [NeurIPS paper](https://openreview.net/forum?id=8sPcuoX2cj) 
and 
[arXiv paper](https://arxiv.org/abs/2605.06384) for the formal description and analyses.

## Index

- [Mathematical foundation](foundations.md) — the MinMax algebra, the MinMax recurrence for any
  state dimension, composition closure, and the parallel prefix scan.
- [Architecture](architecture.md) — neuron groups, layers, convolutions, the backbone and the
  language-model wrapper.
- [Configuration and implementation](configuration.md) — every configuration field, presets,
  numerical precision, initialisation, and feed-forward variants.
