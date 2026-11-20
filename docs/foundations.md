<!--
SPDX-FileCopyrightText: 2026 Alessandro Ronca
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
-->

# MinMax RNCs — Mathematical Foundation

[Index](model.md) · [Mathematical foundation](foundations.md) · [Architecture](architecture.md) · [Configuration and implementation](configuration.md)

See [NeurIPS paper](https://openreview.net/forum?id=8sPcuoX2cj) 
and 
[arXiv paper](https://arxiv.org/abs/2605.06384) for the formal description and analyses.

**Notation.** 
The symbol $`\cdot`$ denotes ordinary multiplication over scalars, vectors, or matrices; 
the symbol $`*`$ denotes element-wise (Hadamard) product of vectors; 
the symbol $`\sigma`$ denotes the logistic sigmoid.


## The MinMax Algebra

The MinMax algebra replaces addition and multiplication of real numbers with max and min:

| MinMax Operation      | Definition                           |
|-----------------------|--------------------------------------|
| Scalar addition       | $`a \oplus b = \mathrm{max}(a, b)`$  |
| Scalar multiplication | $`a \otimes b = \mathrm{min}(a, b)`$ |

Both operations extend to vectors and matrices in the usual way, 
with $`\oplus`$ in place of $`+`$ and $`\otimes`$ in place of $`\cdot`$:

| MinMax Operation      | Definition                                                             |
|-----------------------|------------------------------------------------------------------------|
| Vector addition       | $`(x \oplus y)_i = \mathrm{max}(x_i, y_i)`$                            |
| Matrix-vector product | $`(R \otimes x)_i=\mathrm{max}_j\, \mathrm{min}(R_{ij}, x_j)`$         |
| Matrix-matrix product | $`(R \otimes R')_{ik}=\mathrm{max}_j\, \mathrm{min}(R_{ij}, R'_{jk})`$ |


## MinMax Recurrence

A MinMax neuron of state dimension $`d`$ maintains a state $`x_t \in ℝ^d`$
by computing it at each step from the previous state $`x_t \in ℝ^d`$ via a MinMax affine map 

```math
x_t = (R_t \otimes x_{t-1}) \oplus s_t
```

where $`R_t`$ and $`s_t`$ are ordinary affine projections of the current input $`u_t \in ℝ^m`$:

| Parameter | Dimension    | Name  | Definition         |
|----|-----|-----|----------------|
|$`R_t`$|$`d\times d`$|Reset matrix|$`\mathtt{reshape}\bigl(W_\mathrm{r}\cdot u_t +b_\mathrm{r},(d, d)\bigr)`$|
|$`s_t`$|$`d`$        | Set vector |$`W_\mathrm{s} \cdot u_t + b_\mathrm{s}`$                                 |

with weights $`W_\mathrm{r} \in ℝ^{d^2 \times m}`$, $`W_\mathrm{s} \in ℝ^{d \times m}`$ and biases
$`b_\mathrm{r} \in ℝ^{d^2}`$, $`b_\mathrm{s} \in ℝ^d`$, all learned.


**Cost.** 
Per neuron, parameters scale as $`O(d^2)`$ and each composition step
of the scan as $`O(d^3)`$. A neuron with state dimension $`d`$ can realise
permutation semiautomata of degree $`d`$, so $`d`$ should be as large as the task
needs but typically small; layers scale by having more neurons
(`num_neurons`) and more neuron groups rather than larger $`d`$. 



### Special case: Scalar-state neurons ($`d = 1`$)
For a single neuron with $`d = 1`$, states
$`x_t`$ are scalar, and so are resets $`R_t = r_t`$ and sets $`s_t`$. 
The recurrence can be written as

```math
x_t = \mathrm{max}\bigl(\mathrm{min}(r_t, x_{t-1}), s_t\bigr)
```

with reset and set given by 

```math
\begin{align}
r_t & = w_\mathrm{r} \cdot u_t + b_\mathrm{r},
\qquad&&
w_\mathrm{r} \in ℝ^m,
\quad
b_\mathrm{r} \in ℝ,
\\
s_t & = w_\mathrm{s} \cdot u_t + b_\mathrm{s},
\qquad&&
w_\mathrm{s} \in ℝ^m,
\quad
b_\mathrm{s} \in ℝ.
\end{align}
```

`MinMaxRNCConfig.scalar_neurons = N` gives every layer $`N`$ of such neurons. 
Stacking their states, weights and biases, the layer has states $`x^{(\ell)}_t \in ℝ^N`$ with 

```math
\begin{align*}
r^{(\ell)}_t & = W^{(\ell)}_\mathrm{r} \cdot u_t + b^{(\ell)}_\mathrm{r},
\qquad&&
W^{(\ell)}_\mathrm{r} \in ℝ^{N \times m},
\quad
b^{(\ell)}_\mathrm{r} \in ℝ^N,
\\
s^{(\ell)}_t & = W^{(\ell)}_\mathrm{s} \cdot u_t + b^{(\ell)}_\mathrm{s},
\qquad&&
W^{(\ell)}_\mathrm{s} \in ℝ^{N \times m},
\quad
b^{(\ell)}_\mathrm{s} \in ℝ^N,
\end{align*}
```

and the recurrence above holds element-wise, one coordinate per neuron. 
They are evaluated with a dedicated element-wise scan, so the cost per step is $`O(1)`$ per neuron.


## MinMax Layer's Output

Each neuron's state is projected back to the input space, and 
the outputs of all neurons in a layer are summed:

| `output_gate`  | Output                                                                               |
|----------------|--------------------------------------------------------------------------------------|
|`True` (default)|$`y_t=W_\mathrm{o}\cdot(x_t*\sigma(W_\mathrm{g}\cdot u_t+b_\mathrm{g}))+b_\mathrm{o}`$|
|`False`         |$`y_t=W_\mathrm{o}\cdot x_t + b_\mathrm{o}`$                                          |


## MinMax Recurrence Maps

A MinMax Recurrence map on states of dimension $`d`$ is an affine map in the MinMax algebra,
parameterised by a matrix $`R \in ℝ^{d \times d}`$ and a vector $`s \in ℝ^d`$:

```math
f(x) = (R \otimes x) \oplus s,
\qquad\qquad\text{i.e.,}\quad
f(x)_i = \mathrm{max}\bigl(\mathrm{max}_j\, \mathrm{min}(R_{ij}, x_j),\ s_i\bigr).
```

Each coordinate $`f(x)_i`$ is *selected* among the coordinates of $`x`$, $`R`$, and $`s`$,
so $`f`$ keeps values bounded since it passes them through unchanged.

The map $`f`$ can *route* state coordinates: each coordinate $`f(x)_i`$ can take the value
of $`x_j`$ for any chosen $`j`$ under an appropriate choice of $`R_{ij}`$ — which is what lets a
neuron permute its state.


### Special case: Scalar MinMax Recurrence Maps ($`d = 1`$)

Every MinMax Recurrence map over scalar states

```math
f(x) = \mathrm{max}(\mathrm{min}(r, x), s) 
\qquad\text{for}\qquad r,s \in ℝ
```

has two regimes:

1. When  $`r \ge s`$, 
   the map $`f`$ is a clamp between the lower bound $`s`$ and upper bound $`r`$.

| Case              | State update | Interpretation                     |
|-------------------|--------------|------------------------------------|
| $`x < s`$         | $`f(x) = s`$ | constant — maps states to $`s`$    |
| $`s \le x \le r`$ | $`f(x) = x`$ | identity — leaves states unchanged |
| $`x > r`$         | $`f(x) = r`$ | constant — maps states to $`r`$    |

2. When $`r < s`$, 
   the map $`f`$ is the constant map $`f(x) = s`$ over the entire state space.




## Parallel Computation via Prefix Scan

### Composition Closure

The composition of two MinMax Recurrence maps is again a MinMax Recurrence map, 
because $`\otimes`$ is associative and distributes over $`\oplus`$:

```math
\begin{aligned}
f_2\bigl(f_1(x)\bigr) = (R \otimes x) \oplus s
\qquad \text{for} \qquad
R = R_2 \otimes R_1,
\quad
s = (R_2 \otimes s_1) \oplus s_2.
\end{aligned}
```

In the scalar case, the above reads 
```math
\begin{aligned}
f_2\bigl(f_1(x)\bigr) = (r \otimes x) \oplus s
\qquad \text{for} \qquad
r = \mathrm{min}(r_2, r_1),
\quad
s = \mathrm{max}(\mathrm{min}(r_2, s_1), s_2).
\end{aligned}
```

This closure property is what makes the parallel scan possible: 
instead of applying $`f_1, f_2, \ldots, f_T`$ one by one in sequence, 
all prefix compositions can be computed simultaneously in parallel with depth $`O(\log T)`$.


### Parallel Evaluation

Given an initial state $`x_0`$ and per-step MinMax Recurrence maps $`f_1, \ldots, f_T`$ 
with each $`f_t`$ having parameters $`R_t, s_t`$, 
all states $`x_1, \ldots, x_T`$ can be computed in parallel:

1. **Prefix compose** — compute, for each position $`t`$, the MinMax map
   $`F_t = f_t \circ f_{t-1} \circ \cdots \circ f_1`$ that maps $`x_0`$ to $`x_t`$,
   using composition closure. The current implementation uses the
   Hillis–Steele parallel prefix scan, with depth $`O(\log T)`$ and work
   $`O(T \log T)`$ for $`T`$ the sequence length (a work-efficient scan would
   need only $`O(T)`$ work, at about twice the depth).

2. **Batch apply** — evaluate every prefix map on $`x_0`$ simultaneously: $`x_t = F_t(x_0)`$.

For scalar-state neurons the scan is element-wise (`minmax_scan.all_states`);
for $`d > 1`$ it composes $`d \times d`$ matrices (`minmax_scan.all_states_matrix`),
with the neurons of a group folded into the batch axis, so every composition
costs $`O(d^3)`$.

At inference time (token-by-token generation) the recurrence runs
sequentially as a standard RNN with constant cost per step and a state of
fixed size. Any mix is possible: `unroll_steps = k` runs the scan in parallel
over chunks of $`k`$ steps and the recurrence across chunks — depth
$`O\bigl((T/k) \cdot \log(k)\bigr)`$, work $`O\bigl(T \cdot \log(k)\bigr)`$.

## References

- Hillis & Steele (1986). *Data parallel algorithms.* CACM 29(12).

