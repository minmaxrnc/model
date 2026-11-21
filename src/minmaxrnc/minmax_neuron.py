# SPDX-FileCopyrightText: 2026 Alessandro Ronca
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""
MinMax neurons.

A MinMax neuron with state dimension d maintains a hidden state x_t ∈ R^d
updated by the MinMax recurrence — the affine recurrence of the (max,min)
semiring:

    x_t = max(R_t ⊗ x_{t-1}, s_t),     (R ⊗ x)_i = max_j min(R_ij, x_j)

with R_t ∈ R^{d×d} and s_t ∈ R^d computed from the input u_t.

MinMaxNeuronArray implements an array of K such neurons with the same d, fed
the same input and summed into one output. It is the only implementation;
the state dimension d is just a parameter. For d = 1 (scalar-state neurons),
R_t is a scalar and the recurrence is the element-wise
x_t = max(min(r_t, x_{t-1}), s_t), which the array evaluates with the
element-wise scan — the computation is then exactly that of 0.3.0's MinMax
Neuron. For d > 1 it uses the matrix scan. The choice depends only on d,
which is fixed at construction.

MinMaxNeuron / MinMaxNeuronConfig (the 0.3.0 API) remain: a MinMaxNeuron with
d_state = N is the array of N scalar-state neurons.
"""

import contextlib
import math
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Literal, Optional
from .modules.initialisers import wang_init_, small_init_init_

from . import minmax_scan


SRInitType = Literal['small_init', 'kaiming', 'asymmetric']

RecurrenceDtype = Optional[Literal['fp32', 'bf16', 'fp16']]
"""
Dtype in which the MinMax recurrence is computed: its parameters R_t and s_t,
the incoming state, the prefix compositions, the states (including the state
carried to the next chunk), and their gradients. It does not affect the input
projections that produce R_t and s_t, nor the output gate and projection.

None (default) — no casting: the recurrence runs in whatever dtypes its inputs
    arrive in. Under an outer torch.autocast this means R_t and s_t arrive in the
    autocast dtype while x_0 (an nn.Parameter) stays fp32, so the prefix
    composition runs in the autocast dtype and the final application is
    promoted to fp32.
'fp32' | 'bf16' | 'fp16' — *pinned*: R_t, s_t and the incoming state are cast to
    this dtype and the recurrence runs with autocast disabled, so its precision
    is exactly the requested one whatever the surrounding precision. The
    computed states are cast back to the surrounding compute dtype for the
    output gate and projection; the carried state stays in the pinned dtype.
"""

RECURRENCE_DTYPES = (None, 'fp32', 'bf16', 'fp16')

_RECURRENCE_TORCH_DTYPES = {
    'fp32': torch.float32,
    'bf16': torch.bfloat16,
    'fp16': torch.float16,
}


def _check_recurrence_dtype(recurrence_dtype):
    if recurrence_dtype not in RECURRENCE_DTYPES:
        raise ValueError(
            f"Unknown recurrence_dtype {recurrence_dtype!r}. Valid values: "
            f"{', '.join(map(repr, RECURRENCE_DTYPES))}."
        )


def _recurrence_region(recurrence_dtype, device: torch.device):
    """
    Return (context manager, target dtype or None) for computing the recurrence.

    None   → (nullcontext, None): no casting, surrounding autocast untouched.
    pinned → (autocast disabled, torch dtype): the caller casts its inputs
             to the returned dtype inside the context.
    """
    if recurrence_dtype is None:
        return contextlib.nullcontext(), None
    return (torch.autocast(device_type=device.type, enabled=False),
            _RECURRENCE_TORCH_DTYPES[recurrence_dtype])


# ======================================================================
# Array of MinMax neurons (any state dimension)
# ======================================================================

@dataclass(frozen=True)
class MinMaxNeuronArrayConfig:
    """
    Configuration for an array of `num_neurons` MinMax neurons, each with
    state dimension `d_state`, all fed the same input and summed into a
    single output.

    Normally built by MinMaxRNCConfig.layer_cfg from a NeuronArray in
    MinMaxRNCConfig.neurons.

    Fields
    ------
    _num_blocks : int
        Total number of residual blocks in the enclosing model.  Used to scale
        the output projection at initialisation (wang_init_) so that the
        combined contribution of all blocks to the residual stream stays O(1).
    d_model : int
        Dimension of the input and output (the residual-stream width).
    d_state : int
        State dimension d of *each* neuron. 1 gives scalar-state neurons
        (element-wise recurrence); d > 1 gives neurons whose state
        coordinates interact through a d × d matrix R_t, an unconstrained
        affine projection of the input (every entry of R_t is an independent
        linear function of u_t). Parameters scale as d_model · num_neurons · d²
        and the parallel scan's work as O(num_neurons · T log T · d³), so for
        d > 1 keep d small and scale with num_neurons instead.
    num_neurons : int
        Number K of independent neurons in the array. They are evaluated as
        one batched scan and their outputs summed — identical to K separate
        arrays of one neuron each, summed; only the compute is batched.
    dropout : float
        Dropout probability applied to the input u before projection.
    train_init : bool
        If True, the initial hidden state x_0 is a learned parameter.
        If False (default), x_0 is fixed at zero.
    output_gate : bool
        If True, the output is element-wise multiplied by a sigmoid-gated
        projection of the input before the final linear: y = W_o · (x * σ(W_g · u + b_g)) + b_o,
        with · matrix multiplication and * the element-wise product.
    s_r_init : 'small_init' | 'kaiming' | 'asymmetric'
        Initialisation scheme for the s and r (R_t) input projections.
        'small_init' (default) — normal with std = sqrt(2/(5·d_model)),
        keeping early activations small in deep networks.
        'kaiming' — PyTorch default Kaiming uniform (fan_in=d_model), which
        gives a larger initial scale and more aggressive early state transitions.
        'asymmetric' — treats s and r differently: s gets Kaiming weights
        plus a positive bias (+1.0) so it fires immediately from x_0=0 and
        encodes token identity with sufficient variance; r gets
        small_init weights and zero bias so they stay near zero early on and
        do not clip x back toward the dead zone.
    recurrence_dtype : None | 'fp32' | 'bf16' | 'fp16'
        Dtype of the MinMax recurrence; see RecurrenceDtype. None (default)
        does no casting.
    _state_size_in_layer : int | None
        Total number of state coordinates (Σ num_neurons · d_state over every
        neuron array) in the enclosing MinMaxLayer, whose outputs are summed.
        Used as wang_init_'s `dim` for the output projection, so the layer's
        *total* residual-stream contribution stays O(1) however its state is
        split across arrays — as if the layer had a single output projection
        over all its concatenated states. None (default): this array's own
        num_neurons · d_state, i.e. a layer with this array only.
    """

    _num_blocks:          int
    d_model:              int
    d_state:              int
    num_neurons:          int           = 1
    dropout:              float         = 0.0
    train_init:           bool          = False
    output_gate:          bool          = True
    s_r_init:             SRInitType    = 'small_init'
    recurrence_dtype:     RecurrenceDtype = None
    _state_size_in_layer: Optional[int] = None

    def __post_init__(self):
        _check_recurrence_dtype(self.recurrence_dtype)
        if self.d_state <= 0:
            raise ValueError(f"d_state must be > 0, got {self.d_state}")
        if self.num_neurons <= 0:
            raise ValueError(f"num_neurons must be > 0, got {self.num_neurons}")


class MinMaxNeuronArray(nn.Module):
    """
    An array of K = num_neurons MinMax neurons with state dimension d = d_state
    each, all fed the same input and summed into a single output.

    Each neuron i maintains x_t^(i) ∈ R^d, updated by

        x_t^(i) = max(R_t^(i) ⊗ x_{t-1}^(i), s_t^(i))

    where ⊗ is the (max,min)-semiring matrix-vector product and R_t^(i), s_t^(i)
    are unconstrained affine projections of the input u_t (weights and biases). For d = 1 this is
    the element-wise x_t = max(min(r_t, x_{t-1}), s_t).

    State layout: the array's state is the flat vector of its K·d state
    coordinates (neuron-major: neuron i occupies coordinates i·d … i·d+d-1).
    The initial state x_0 has shape (K·d,), the recurrent state (B, K·d), and
    the s / output projections act on K·d coordinates. For d = 1 every tensor
    and parameter has exactly the shape of 0.3.0's MinMax Neuron with
    d_state = K.

    Evaluation: all states for a sequence of length T, for all K neurons, are
    computed by one parallel prefix scan of depth O(log T):
    - d = 1: the element-wise scan (minmax_scan.all_states) runs directly on
      the (B, T, K) projections — no reshaping.
    - d > 1: the matrix scan (minmax_scan.all_states_matrix), with the neuron
      axis folded into the batch axis; each combine step costs O(d³) and
      materialises a (B·K, T, d, d, d) intermediate.

    Output projection (summed over the K neurons):

        y_t = W_o · x_t + b_o                          (output_gate=False)
        y_t = W_o · (x_t * σ(W_g · u_t + b_g)) + b_o   (output_gate=True; · matrix product, * element-wise)

    with x_t ∈ R^{K·d} the concatenated states — identical to summing per-neuron
    projections.

    Attributes: I = d_model (the input dimension, m in the docs), d = d_state, K = num_neurons, D = K·d (number of
    state coordinates).
    """

    def __init__(self, cfg: MinMaxNeuronArrayConfig):

        super().__init__()

        self.cfg = self.array_cfg = cfg
        self._build()


    def _build(self):
        cfg = self.array_cfg

        self.I = I = cfg.d_model
        self.d = d = cfg.d_state
        self.K = K = cfg.num_neurons
        self.D = D = K * d

        # Construction order (and hence RNG consumption and state_dict key
        # order): _initial_state, s, r, o, o_g — for d = 1 exactly as 0.3.0's
        # MinMax Neuron (r then has K·1·1 = D outputs).
        self._initial_state = nn.Parameter(torch.zeros(D), requires_grad=cfg.train_init)

        self.drop = nn.Dropout(cfg.dropout)

        self.s = nn.Linear(I, D)
        self.r = nn.Linear(I, K * d * d)
        self.o = nn.Linear(D, I)
        if cfg.output_gate:
            self.o_g = nn.Linear(I, D)

        self.reset()


    def reset(self):
        cfg = self.array_cfg
        # Init 's' and 'r' — fan-in is always self.I; these init functions
        # sample each element i.i.d. from a distribution that depends only on
        # the passed `dim`, so applying them to the stacked K-neuron tensors is
        # statistically identical to applying them per neuron. Same calls in
        # the same order as 0.3.0's neuron.
        if cfg.s_r_init == 'kaiming':
            nn.init.kaiming_uniform_(self.s.weight, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.r.weight, a=math.sqrt(5))
            if self.s.bias is not None:
                nn.init.zeros_(self.s.bias)
            if self.r.bias is not None:
                nn.init.zeros_(self.r.bias)
        elif cfg.s_r_init == 'asymmetric':
            # s: Kaiming weights + positive bias so s fires from x_0=0 in most
            # dimensions, eliminating the r>0>s dead zone.
            # r: small_init weights + zero bias so r stays near zero and does
            # not clip x back toward 0 while s is learning to write.
            nn.init.kaiming_uniform_(self.s.weight, a=math.sqrt(5))
            small_init_init_(self.r.weight, dim=self.I)
            if self.s.bias is not None:
                nn.init.constant_(self.s.bias, 1.0)
            if self.r.bias is not None:
                nn.init.zeros_(self.r.bias)
        else:
            small_init_init_(self.s.weight, dim=self.I)
            small_init_init_(self.r.weight, dim=self.I)
            if self.s.bias is not None:
                nn.init.zeros_(self.s.bias)
            if self.r.bias is not None:
                nn.init.zeros_(self.r.bias)
        # Init 'o' — dim = number of state coordinates summed at the layer
        # (default: this array's own D), so wang_init_'s residual-stream
        # scaling holds for any state size and any split across arrays.
        state_size = cfg._state_size_in_layer if cfg._state_size_in_layer is not None else self.D
        wang_init_(self.o.weight, dim=state_size, num_blocks=cfg._num_blocks)
        if self.o.bias is not None:
            nn.init.zeros_(self.o.bias)
        # Init 'o_g' — small weights keep the gate near 0.5 (sigmoid(~0)) at
        # init; zero bias removes the per-dimension offset from PyTorch's default.
        if cfg.output_gate:
            small_init_init_(self.o_g.weight, dim=self.I)
            if self.o_g.bias is not None:
                nn.init.zeros_(self.o_g.bias)


    @property
    def initial_state(self):
        return self._initial_state


    # ------------------------------------------------------------------
    # Scans
    # ------------------------------------------------------------------

    def _scan_scalar(self, s, r, x0, B: int, T: int) -> torch.Tensor:
        """d = 1: element-wise scan over the (B, T, K) projections."""
        return minmax_scan.all_states(r, s, x0)[:, 1:, :]            # (B,T,K)


    def _scan_matrix(self, s, r, x0, B: int, T: int) -> torch.Tensor:
        """d > 1: matrix scan, with the neuron axis folded into the batch."""
        d, K = self.d, self.K
        s_flat  = s.view(B, T, K, d).permute(0, 2, 1, 3).reshape(B * K, T, d)
        R_flat  = r.view(B, T, K, d, d).permute(0, 2, 1, 3, 4).reshape(B * K, T, d, d)
        x0_flat = x0.reshape(B * K, d)

        x_post = minmax_scan.all_states_matrix(R_flat, s_flat, x0_flat)[:, 1:, :]   # (B*K,T,d)
        return x_post.view(B, K, T, d).permute(0, 2, 1, 3).reshape(B, T, K * d)


    def forward(self, u: torch.Tensor, state: torch.Tensor):
        """
        Compute updated state for a sequence using closed form with initial state.

        u:     (B, T, I)
        state: (K·d,) or (B, K·d)   (state before the first step in the input sequence)

        Returns: 1) sequence of outputs (summed over the K neurons): (B, T, I)
                 2) last state:                                     (B, K·d)
        """

        B, T, I = u.shape
        D = self.D

        if state.dim() == 1: # state is the initial initial state
            x0 = state.unsqueeze(0).expand(B, D)
        else:
            x0 = state

        u = self.drop(u)                      # (B,T,I)
        s = self.s(u)                         # (B,T,D)
        r = self.r(u)                         # (B,T,K·d·d)

        compute_dtype = s.dtype
        region, rec_dtype = _recurrence_region(self.array_cfg.recurrence_dtype, u.device)
        with region:
            if rec_dtype is not None:
                r, s, x0 = r.to(rec_dtype), s.to(rec_dtype), x0.to(rec_dtype)
            if self.d == 1:
                x_post = self._scan_scalar(s, r, x0, B, T)
            else:
                x_post = self._scan_matrix(s, r, x0, B, T)

        # ----- Compute outputs -----
        x_latest = x_post[:, -1, :]           # (B,D) — carried state (recurrence dtype if pinned)
        if rec_dtype is not None:
            x_post = x_post.to(compute_dtype)
        if self.array_cfg.output_gate:
            x_post = x_post * torch.sigmoid(self.o_g(u))
        output = self.o(x_post)               # (B,T,I)

        return output, x_latest


# ======================================================================
# MinMax Neuron — the 0.3.0 API: an array of d_state scalar-state neurons
# ======================================================================

@dataclass(frozen=True)
class MinMaxNeuronConfig:
    """
    Configuration for a MinMax Neuron as in 0.3.0: `d_state` scalar-state
    neurons (equivalently, MinMaxNeuronArrayConfig(d_state=1,
    num_neurons=d_state)).

    Fields are those of MinMaxNeuronArrayConfig, with `d_state` here meaning
    the total state dimension, i.e. the number of scalar-state neurons. See
    there for their description.
    """

    _num_blocks: int
    d_model:     int
    d_state:     int
    dropout:     float      = 0.0
    train_init:  bool       = False
    output_gate: bool       = True
    s_r_init:    SRInitType = 'small_init'
    # --- new in 0.4.0 (appended, so positional construction is unchanged) ---
    recurrence_dtype:     RecurrenceDtype = None
    _state_size_in_layer: Optional[int] = None

    def __post_init__(self):
        _check_recurrence_dtype(self.recurrence_dtype)

    def to_array_config(self) -> MinMaxNeuronArrayConfig:
        return MinMaxNeuronArrayConfig(
            _num_blocks          = self._num_blocks,
            d_model              = self.d_model,
            d_state              = 1,
            num_neurons          = self.d_state,
            dropout              = self.dropout,
            train_init           = self.train_init,
            output_gate          = self.output_gate,
            s_r_init             = self.s_r_init,
            recurrence_dtype     = self.recurrence_dtype,
            _state_size_in_layer = self._state_size_in_layer,
        )


class MinMaxNeuron(MinMaxNeuronArray):
    """
    MinMax Neuron as in 0.3.0: a MinMaxNeuronArray of `cfg.d_state`
    scalar-state neurons, updating x_t ∈ R^N (N = cfg.d_state) element-wise:

        x_t = max(min(r_t, x_{t-1}), s_t)

    Same parameters (s, r, o, o_g, _initial_state), shapes, initialisation and
    computation as in 0.3.0. `cfg` is the MinMaxNeuronConfig it was built
    from; `array_cfg` the equivalent MinMaxNeuronArrayConfig.
    """

    def __init__(self, cfg: MinMaxNeuronConfig):
        nn.Module.__init__(self)
        self.cfg       = cfg
        self.array_cfg = cfg.to_array_config()
        self._build()
