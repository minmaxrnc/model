<!--
SPDX-FileCopyrightText: 2026 Alessandro Ronca
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
-->

# MinMax RNCs — Numerical Precision

[Index](README.md) · [Mathematical foundations](foundations.md) · [Architecture](architecture.md) · [Configuration](configuration.md) · [Numerical precision](precision.md) · [Upgrading from 0.3.0](upgrading.md)

---

This page explains the two precision settings: model-level mixed precision (`amp_dtype`) and the
dtype of the MinMax recurrence (`recurrence_dtype`). They are independent, and both are off by
default.

**Notation.** As in [Mathematical foundations](foundations.md).

**`amp_dtype`** — If set, [`MinMaxRNC.forward`](../src/minmaxrnc/minmax_rnc.py) runs inside 
`torch.autocast(device_type, dtype=amp_dtype)`; for [`MinMaxRNC_LM`](../src/minmaxrnc/minmax_rnc_lm.py), the same applies to
embedding and LM head. 
If not set, the model does not touch autocast, so a context opened by the caller applies as usual.
An explicit `amp_dtype` takes precedence over an outer autocast inside the model. 
Loss scaling for fp16 (`torch.amp.GradScaler`) stays with the caller.

**`recurrence_dtype`** — the dtype in which every neuron computes its MinMax recurrence, i.e. the
dtype of:

- the scan's inputs: the inputs $`R_t`$ and $`s_t`$ (as produced by the input projections)
  and the incoming state ($`x_0`$, or the state carried from the previous chunk);
- the intermediate prefix compositions;
- the states $`x_1, \ldots, x_T`$ it computes, including the final state carried to the next
  chunk;
- in the backward pass, the gradients of all of the above.

It does not affect the input projections that compute $`R_t`$ and $`s_t`$, the output gate and
projection, or the dtype of the parameters:

| Value | Scan inputs | Autocast during scan | States, incl. carried state |
|---|---|---|---|
| `None` | as they arrive (no casting) | unchanged | whatever the scan produces |
| `'fp32'` / `'bf16'` / `'fp16'` | cast to that dtype | disabled | that dtype |

With a pinned value, the computed states are cast back to the surrounding compute dtype before the
output gate and projection.

**What the recurrence's precision changes.** The scan is a chain of element-wise
min/max *selections*: every state value it produces is one of its input values
($`R_t`$, $`s_t`$ or $`x_0`$), never a newly computed one. Three consequences:

1. **Forward pass.** Running the scan in a *higher* precision than its inputs
   changes nothing: under bf16 AMP, $`R_t`$ and $`s_t`$ are already bf16, and
   min/max of bf16 values is exact in any wider type, so an `'fp32'` scan gives
   bitwise the same outputs as a `'bf16'` one. The exception is a learned
   initial state (`train_init=True`): $`x_0`$ is an fp32 parameter, which a
   bf16 recurrence rounds. Running the scan in a *lower* precision than its inputs
   (an fp32 model with `recurrence_dtype='bf16'`) rounds them, and so changes the
   forward pass.
2. **Backward pass.** The gradient of a selected value is the sum of the
   gradients of every later state it was selected into — for a value held over
   $`k`$ steps, a sum of $`k`$ terms. The scan's backward pass runs in the
   scan's dtype, so with `'fp32'` these sums are accumulated in fp32 even under
   bf16 AMP (and rounded to bf16 once, at the end), whereas with `'bf16'` every
   partial sum is rounded to bf16's 8-bit significand. The difference grows
   with how long values are held.
3. **Cost.** The scan is memory-bandwidth bound (no tensor-core acceleration),
   and its prefix-composition intermediates are a large share of activation
   memory for long sequences, so its cost and memory scale with the dtype's
   byte width: about 2× for fp32 vs bf16/fp16.

Under an outer bf16 autocast, with `None` the scan composes the prefixes in
bf16 while the fp32 initial state promotes the final application to fp32 —
mixed and implicit. Pinning makes the choice explicit.

**Recommended combinations.**

| Recipe | `amp_dtype` | `recurrence_dtype` | Forward | Recurrence gradient sums | Recurrence memory |
|---|---|---|---|---|---|
| Full precision (default) | `None` | `None` | fp32 | fp32 | 4 bytes |
| bf16 AMP | `'bf16'` | `'bf16'` (or `None`) | bf16 | bf16 | 2 bytes |
| fp16 AMP | `'fp16'` | `'fp16'` (or `None`) | fp16 | fp16 | 2 bytes |
| fp32 model, 2-byte recurrence | `None` | `'fp16'` | the recurrence rounds its inputs | fp16 (overflow above 65504) | 2 bytes |

**Combinations to avoid.**

| Combination | Why |
|---|---|
| `amp_dtype='bf16'`, `recurrence_dtype='fp32'` | No measurable gain: the gradient error is that of bf16 AMP, and peak memory was ~1.7× higher in our measurements |
| `amp_dtype='bf16'`, `recurrence_dtype='fp16'` | No better than a bf16 recurrence |
| `amp_dtype=None`, `recurrence_dtype='bf16'` | 3–4× less accurate than `'fp16'`, at the same memory |
| `amp_dtype='fp16'`, `recurrence_dtype='bf16'` | ~10× less accurate than `'fp16'`, at the same memory |

**Measured gradient accuracy.** Relative error of the gradients of the $`s`$/$`r`$
projections against full fp32, on sequences of length 8192 (NVIDIA A100 and L4,
mean over architectures with scalar-state neurons, neuron arrays and both; the
first figure is for inputs that make neurons hold values for long, the second for
random inputs). The share due to the recurrence itself is the error against the
run with the same `amp_dtype` and an fp32 recurrence, whose forward pass is
identical.

| `amp_dtype` | `recurrence_dtype` | Total gradient error | Share due to the recurrence |
|---|---|---|---|
| `None` | `'bf16'` | 5.8% / 4.7% | all of it |
| `None` | `'fp16'` | 1.5% / 1.7% | all of it |
| `'bf16'` | `'bf16'`, `None`, `'fp16'` or `'fp32'` | 10.3% / 8.9% | ≤ 0.6% |
| `'fp16'` | `'fp16'` or `None` | 4.3% / 3.2% | 0.4% / 0.3% |
| `'fp16'` | `'bf16'` | 5.9% / 4.9% | 5.8% / 4.6% |

Under bf16 AMP, almost all of the gradient error comes from the bf16 forward pass
outside the recurrence, which is why pinning the recurrence to fp32 does not help.
