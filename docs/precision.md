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

- the scan's inputs: the parameters $`R_t`$ and $`s_t`$ (as produced by the input projections)
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
mixed and implicit. Pinning makes the choice explicit:

| Recipe | `amp_dtype` | `recurrence_dtype` | Forward | Recurrence gradient sums | Recurrence memory |
|---|---|---|---|---|---|
| Full precision (default) | `None` | `None` | fp32 | fp32 | 4 bytes |
| bf16 AMP, fp32 recurrence | `'bf16'` | `'fp32'` | = bf16 recurrence | fp32 | 4 bytes |
| Fully bf16, recurrence included | `'bf16'` | `'bf16'` | bf16 | bf16 | 2 bytes |
| fp32 model, bf16 recurrence | `None` | `'bf16'` | the recurrence rounds its inputs | bf16 | 2 bytes |
| bf16 AMP, fp16 recurrence (untested) | `'bf16'` | `'fp16'` | = bf16 recurrence, within fp16's range | fp16 (finer than bf16; overflow above 65504) | 2 bytes |

The effect of the more accurate gradient sums on training has not been measured
yet; compare `'bf16'` against `'fp32'` on your task before relying on it.
