<!--
SPDX-FileCopyrightText: 2026 Alessandro Ronca
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
-->

# MinMax RNCs — Mathematical Foundation

[Index](model.md) · [Mathematical foundation](foundations.md) · [Architecture](architecture.md) · [Configuration and implementation](configuration.md)

See [arxiv.org/abs/2605.06384](https://arxiv.org/abs/2605.06384) for the formal description and analyses.

**Notation.** $`\oplus`$ and $`\otimes`$ are the MinMax operations defined below;
$`\cdot`$ is ordinary (matrix) multiplication; $`*`$ is the element-wise
(Hadamard) product of vectors; $`\sigma`$ is the logistic sigmoid.


## The MinMax Algebra

The MinMax algebra replaces addition and multiplication of real numbers with max and min:

| Operation | Definition |
|---|---|
| MinMax 'addition' | $`a \oplus b = \mathrm{max}(a, b)`$ |
| MinMax 'multiplication' | $`a \otimes b = \mathrm{min}(a, b)`$ |

Both operations extend to vectors and matrices in the usual way, 
with $`\oplus`$ in place of $`+`$ and $`\otimes`$ in place of $`\times`$:

| Operation | Definition |
|---|---|
| vector 'sum' | $`(x \oplus y)_i = \mathrm{max}(x_i, y_i)`$ |
| matrix-vector 'product' | $`(R \otimes x)_i=\mathrm{max}_j\, \mathrm{min}(R_{ij}, x_j)`$ |
| matrix-matrix 'product' | $`(R \otimes R')_{ik}=\mathrm{max}_j\, \mathrm{min}(R_{ij}, R'_{jk})`$ |


## MinMax Recurrence

A MinMax neuron with state dimension $`d`$ maintains a hidden state $`x_t \in ℝ^d`$,
obtained by updating the previous state by a MinMax affine map whose parameters are computed from the
current input $`u_t \in ℝ^m`$:

```math
x_t = (R_t \otimes x_{t-1}) \oplus s_t
```

where both parameters are affine projections of the input:

| Parameter | Dimension | Name | Definition |
|---|---|---|---|
| $`R_t`$ |$`d \times d`$ | Reset matrix | $`\mathtt{reshape}(W_r \cdot u_t + b_r, (d, d))`$ |
| $`s_t`$ |$`d`$          | Set vector   | $`W_s \cdot u_t + b_s`$                           |

with weights $`W_r \in ℝ^{d^2 \times m}`$, $`W_s \in ℝ^{d \times m}`$ and biases
$`b_r \in ℝ^{d^2}`$, $`b_s \in ℝ^d`$, all learned.

*Note:* Each entry of $`R_t`$ and $`s_t`$ is given by an affine function of $`u_t`$ independently of
any other entry. There is no immediate reason to adopt parametrisations that impose constraints on
the entries — in contrast to Linear RNNs where it is crucial to adopt parametrisations ensuring
properties such as non-expansiveness.

**Special case: Scalar-state neurons ($`d = 1`$).** For a single neuron with $`d = 1`$, states
$`x_t`$ are scalar, and so are resets $`R_t = r_t`$ and sets $`s_t`$. The recurrence admits the form

```math
x_t = \mathrm{max}\bigl(\mathrm{min}(r_t, x_{t-1}), s_t\bigr)
```

and the reset and set parameters can be more easily interpreted:

| Parameter | Definition | Role |
|---|---|---|
| $`r_t`$ | $`w_r \cdot u_t + b_r`$ | upper bound — reset signal / forget gate |
| $`s_t`$ | $`w_s \cdot u_t + b_s`$ | lower bound — set signal / write value |

where $`w_r, w_s \in ℝ^m`$ and $`b_r, b_s \in ℝ`$ are the neuron's input weight vectors 
and biases, respectively.

`MinMaxRNCConfig.scalar_neurons = N` gives every layer $`N`$ such neurons. Stacking their states,
weights and biases, the layer keeps $`x_t \in ℝ^N`$ with $`r_t = W_r \cdot u_t + b_r`$ and
$`s_t = W_s \cdot u_t + b_s`$ ($`W_r, W_s \in ℝ^{N \times m}`$, $`b_r, b_s \in ℝ^N`$), and the
recurrence above holds element-wise, one coordinate per neuron. 
They are evaluated with a dedicated element-wise scan, so the cost per step is $`O(1)`$ per neuron.

**Cost.** Per neuron, parameters scale as $`O(d^2)`$ and each composition step
of the scan as $`O(d^3)`$. A neuron with state dimension $`d`$ can realise
permutation semiautomata of degree $`d`$, so $`d`$ should be as large as the task
needs but typically small; layers scale by having more neurons
(`num_neurons`) and more neuron groups rather than larger $`d`$. 

**Output.** Each neuron's state is projected back to the input space, and
the outputs of all neurons in a layer are summed:

| `output_gate` | Output |
|---|---|
| `False`          | $`y_t = W_o \cdot x_t + b_o`$ |
| `True` (default) | $`y_t = W_o \cdot (x_t * \sigma(W_g \cdot u_t + b_g)) + b_o`$ |


## MinMax Recurrence Map

A MinMax Recurrence map on states of dimension $`d`$ is an affine map in the MinMax algebra,
parameterised by a matrix $`R \in ℝ^{d \times d}`$ and a vector $`s \in ℝ^d`$:

```math
f(x) = (R \otimes x) \oplus s,
\qquad\text{i.e.}\qquad
f(x)_i = \mathrm{max}\bigl(\mathrm{max}_j\, \mathrm{min}(R_{ij}, x_j),\ s_i\bigr).
```

Every coordinate of $`f(x)`$ is *selected* among the entries of $`x`$, $`R`$ and $`s`$,
so $`f`$ keeps values bounded and passes them through unchanged.

The map can *route* values between coordinates: coordinate $`i`$ can take the value of coordinate
$`j`$, limited by $`R_{ij}`$ — which is what lets a neuron permute its state.


**Scalar case ($`d = 1`$).** With a scalar $`r = R`$, the map is a clamp between
the lower bound $`s`$ and the upper bound $`r`$:

```math
f(x) = \mathrm{max}(\mathrm{min}(r, x), s)
```

We can see it has two regimes:

1. When $`r \ge s`$, the map is piecewise affine — it is constant in two regions of the state space,
   and identity in the third region:

| Input             | Output       | Interpretation                       |
|---|---|---|
| $`x < s`$         | $`f(x) = s`$ | constant $`s`$                       |
| $`s \le x \le r`$ | $`f(x) = x`$ | identity — propagate state unchanged |
| $`x > r`$         | $`f(x) = r`$ | constant $`r`$                       |

2. When $`r < s`$, the map is the constant map $`f(x) = s`$ over the entire state space.


## Composition Closure

The composition of two MinMax Recurrence maps is again a MinMax Recurrence map, 
because $`\otimes`$ is associative and distributes over $`\oplus`$:

```math
\begin{aligned}
f_2\bigl(f_1(x)\bigr) = (R \otimes x) \oplus s
\qquad \text{where} \qquad
R = R_2 \otimes R_1,
\quad
s = (R_2 \otimes s_1) \oplus s_2.
\end{aligned}
```

In the scalar case this reads $`r = \mathrm{min}(r_2, r_1)`$ and
$`s = \mathrm{max}(\mathrm{min}(r_2, s_1), s_2)`$.

This closure property is what makes the parallel scan possible: instead of
applying $`f_1, f_2, \ldots, f_T`$ one by one, all prefix compositions can be
computed simultaneously in $`O(\log T)`$ depth.


## Parallel Computation via Prefix Scan

Given $`x_0`$ and the per-step maps $`f_1, \ldots, f_T`$ ($`f_t`$ with parameters
$`R_t, s_t`$), all states $`x_1, \ldots, x_T`$ can be computed in parallel:

1. **Prefix compose** — compute, for each position $`t`$, the MinMax map
   $`F_t = f_t \circ f_{t-1} \circ \cdots \circ f_1`$ that maps $`x_0`$ to $`x_t`$,
   using composition closure. The current implementation uses the
   Hillis–Steele parallel prefix scan, with depth $`O(\log T)`$ and work
   $`O(T \log T)`$ for $`T`$ the sequence length (a work-efficient scan would
   need only $`O(T)`$ work, at about twice the depth).

2. **Batch apply** — evaluate every prefix map on $`x_0`$ simultaneously:
   $`x_t = F_t(x_0)`$.

For scalar-state neurons the scan is element-wise (`minmax_scan.all_states`);
for $`d > 1`$ it composes $`d \times d`$ matrices (`minmax_scan.all_states_matrix`),
with the neurons of a group folded into the batch axis, so every composition
costs $`O(d^3)`$.

At inference time (token-by-token generation) the recurrence runs
sequentially as a standard RNN with constant cost per step and a state of
fixed size. Any mix is possible: `unroll_steps = k` runs the scan in parallel
over chunks of $`k`$ steps and the recurrence across chunks — depth
$`O((T/k) \cdot \log k)`$, work $`O(T \log k)`$.

## References

- Hillis & Steele (1986). *Data parallel algorithms.* CACM 29(12).
