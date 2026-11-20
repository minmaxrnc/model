# SPDX-FileCopyrightText: 2026 Alessandro Ronca
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""
MinMax neurons.

A MinMax neuron with state dimension d maintains a hidden state x_t ∈ R^d
updated by the MinMax recurrence — the affine recurrence of the (max,min)
semiring:

    x_{t+1} = max(R_t ⊗ x_t, s_t),     (R ⊗ x)_i = max_j min(R_ij, x_j)

with R_t ∈ R^{d×d} and s_t ∈ R^d computed from the input u_t.

MinMaxNeuronGroup implements a group of K such neurons with the same d, fed
the same input and summed into one output. It is the only implementation;
the state dimension d is just a parameter. For d = 1 (scalar-state neurons),
R_t is a scalar and the recurrence is the element-wise
x_{t+1} = max(min(r_t, x_t), s_t), which the group evaluates with the
element-wise scan — the computation is then exactly that of 0.3.0's MinMax
Neuron. For d > 1 it uses the matrix scan. The choice depends only on d,
which is fixed at construction.

MinMaxNeuron / MinMaxNeuronConfig (the 0.3.0 API) remain: a MinMaxNeuron with
d_state = D is the group of D scalar-state neurons.
"""

import contextlib
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Literal, Optional
from .modules.initialisers import wang_init_, small_init_init_

from . import minmax_scan


SRInitType = Literal['small_init', 'kaiming', 'asymmetric']

ScanDtype = Literal['ambient', 'fp32', 'bf16', 'fp16']
"""
Precision the MinMax scan runs in.

'ambient' (default) — no casting at all: the scan runs on whatever dtypes its
    inputs arrive in. This is exactly the 0.3.0 behaviour. Under an outer
    torch.autocast this means the s/r projections arrive in the autocast dtype
    while x_0 (an nn.Parameter) stays fp32, so the prefix composition runs in
    the autocast dtype and the final application is promoted to fp32.
'fp32' | 'bf16' | 'fp16' — *pinned*: every scan input (and the construction of
    R_t) is cast to this dtype and the scan runs with autocast disabled, so its
    precision is exactly the requested one regardless of any ambient autocast.
    The scan's outputs are cast back to the ambient compute dtype for the
    output projections; the carried recurrent state stays in the pinned dtype.
"""

SCAN_DTYPES = ('ambient', 'fp32', 'bf16', 'fp16')

_SCAN_TORCH_DTYPES = {
    'fp32': torch.float32,
    'bf16': torch.bfloat16,
    'fp16': torch.float16,
}


def _check_scan_dtype(scan_dtype: str):
    if scan_dtype not in SCAN_DTYPES:
        raise ValueError(
            f"Unknown scan_dtype {scan_dtype!r}. Valid values: {', '.join(map(repr, SCAN_DTYPES))}."
        )


def _scan_region(scan_dtype: str, device: torch.device):
    """
    Return (context manager, target dtype or None) for running the scan.

    'ambient' → (nullcontext, None): no casting, ambient autocast untouched.
    pinned    → (autocast disabled, torch dtype): the caller casts its inputs
                to the returned dtype inside the context.
    """
    if scan_dtype == 'ambient':
        return contextlib.nullcontext(), None
    return torch.autocast(device_type=device.type, enabled=False), _SCAN_TORCH_DTYPES[scan_dtype]


# ======================================================================
# Group of MinMax neurons (any state dimension)
# ======================================================================

@dataclass(frozen=True)
class MinMaxNeuronGroupConfig:
    """
    Configuration for a group of `num_neurons` MinMax neurons, each with
    state dimension `d_state`, all fed the same input and summed into a
    single output.

    Normally built by MinMaxRNCConfig.layer_cfg from a NeuronGroup in
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
        coordinates interact through a d × d matrix R_t. Parameters scale as
        d_model · num_neurons · d² (for n_h = 0) and the parallel scan's work
        as O(num_neurons · T log T · d³), so for d > 1 keep d small and scale
        with num_neurons instead.
    num_neurons : int
        Number K of independent neurons in the group. They are evaluated as
        one batched scan and their outputs summed — identical to K separate
        groups of one neuron each, summed; only the compute is batched.
    n_h : int
        Parametrisation of R_t.
        0 (default) — an unconstrained affine projection of u_t: every entry
        of R_t is an independent linear function of u_t.
        > 0 — EXPERIMENTAL. A product of n_h rank-1 corrections to the
        identity, in the style of DeltaProduct (Siems et al., NeurIPS 2025):

            R_t = ∏_{i=1}^{n_h} (I - β_i(u_t) · k_i(u_t) k_i(u_t)^T)

        with β_i an unconstrained affine function of u_t and k_i not
        normalised (unless normalize_keys): unlike DeltaProduct's linear
        recurrence, the (max,min) recurrence only ever *selects* values, so
        it needs no eigenvalue bounding for stability. Parameters scale as
        d_model · num_neurons · n_h · d. No positive experimental results yet.
    normalize_keys : bool
        Only meaningful when n_h > 0: L2-normalise each k_i before forming
        k_i k_i^T, as in DeltaProduct, so β_i alone sets each factor's
        magnitude.
    dropout : float
        Dropout probability applied to the input u before projection.
    train_init : bool
        If True, the initial hidden state x_0 is a learned parameter.
        If False (default), x_0 is fixed at zero.
    output_gate : bool
        If True, the output is element-wise multiplied by a sigmoid-gated
        projection of the input before the final linear: y = W_o(x ⊙ σ(W_g u)).
    s_r_init : 'small_init' | 'kaiming' | 'asymmetric'
        Initialisation scheme for the s and R_t (r, or k/β) input projections.
        'small_init' (default) — normal with std = sqrt(2/(5·d_model)),
        keeping early activations small in deep networks.
        'kaiming' — PyTorch default Kaiming uniform (fan_in=d_model), which
        gives a larger initial scale and more aggressive early state transitions.
        'asymmetric' — treats s and R_t differently: s gets Kaiming weights
        plus a positive bias (+1.0) so it fires immediately from x_0=0 and
        encodes token identity with sufficient variance; R_t's projections get
        small_init weights and zero bias so they stay near zero early on and
        do not clip x back toward the dead zone.
    scan_dtype : 'ambient' | 'fp32' | 'bf16' | 'fp16'
        Precision of the MinMax scan; see ScanDtype. 'ambient' (default) does
        no casting.
    _state_size_in_layer : int | None
        Total number of state coordinates (Σ num_neurons · d_state over every
        neuron group) in the enclosing MinMaxLayer, whose outputs are summed.
        Used as wang_init_'s `dim` for the output projection, so the layer's
        *total* residual-stream contribution stays O(1) however its state is
        split across groups — as if the layer had a single output projection
        over all its concatenated states. None (default): this group's own
        num_neurons · d_state, i.e. a layer with this group only.
    """

    _num_blocks:          int
    d_model:              int
    d_state:              int
    num_neurons:          int           = 1
    n_h:                  int           = 0
    normalize_keys:       bool          = False
    dropout:              float         = 0.0
    train_init:           bool          = False
    output_gate:          bool          = True
    s_r_init:             SRInitType    = 'small_init'
    scan_dtype:           ScanDtype     = 'ambient'
    _state_size_in_layer: Optional[int] = None

    def __post_init__(self):
        _check_scan_dtype(self.scan_dtype)
        if self.d_state <= 0:
            raise ValueError(f"d_state must be > 0, got {self.d_state}")
        if self.num_neurons <= 0:
            raise ValueError(f"num_neurons must be > 0, got {self.num_neurons}")
        if self.n_h < 0:
            raise ValueError(f"n_h must be >= 0, got {self.n_h}")


class MinMaxNeuronGroup(nn.Module):
    """
    A group of K = num_neurons MinMax neurons with state dimension d = d_state
    each, all fed the same input and summed into a single output.

    Each neuron k maintains x_t^(k) ∈ R^d, updated by

        x_{t+1}^(k) = max(R_t^(k) ⊗ x_t^(k), s_t^(k))

    where ⊗ is the (max,min)-semiring matrix-vector product. For d = 1 this is
    the element-wise x_{t+1} = max(min(r_t, x_t), s_t).

    State layout: the group's state is the flat vector of its K·d state
    coordinates (neuron-major: neuron k occupies coordinates k·d … k·d+d-1).
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

        y_t = W_o x_t                      (output_gate=False)
        y_t = W_o (x_t ⊙ σ(W_g u_t))       (output_gate=True)

    with x_t ∈ R^{K·d} the concatenated states — identical to summing per-neuron
    projections.

    Attributes: I = d_model, d = d_state, K = num_neurons, D = K·d (number of
    state coordinates), n_h.
    """

    def __init__(self, cfg: MinMaxNeuronGroupConfig):

        super().__init__()

        self.cfg = self.group_cfg = cfg
        self._build()


    def _build(self):
        cfg = self.group_cfg

        self.I   = I   = cfg.d_model
        self.d   = d   = cfg.d_state
        self.K   = K   = cfg.num_neurons
        self.n_h = n_h = cfg.n_h
        self.D   = D   = K * d

        # Construction order (and hence RNG consumption and state_dict key
        # order): _initial_state, s, r | k+beta, o, o_g — for d = 1 exactly
        # as 0.3.0's MinMax Neuron.
        self._initial_state = nn.Parameter(torch.zeros(D), requires_grad=cfg.train_init)

        self.drop = nn.Dropout(cfg.dropout)

        self.s = nn.Linear(I, D)
        if n_h > 0:
            self.k    = nn.Linear(I, K * n_h * d)
            self.beta = nn.Linear(I, K * n_h)
        else:
            self.r = nn.Linear(I, K * d * d)
        self.o = nn.Linear(D, I)
        if cfg.output_gate:
            self.o_g = nn.Linear(I, D)

        self.reset()


    def reset(self):
        cfg = self.group_cfg
        # Init 's' and R_t's projections ('r', or 'k'/'beta' when n_h > 0) —
        # fan-in is always self.I; these init functions sample each element
        # i.i.d. from a distribution that depends only on the passed `dim`,
        # so applying them to the stacked K-neuron tensors is statistically
        # identical to applying them per neuron. The order of RNG-consuming
        # calls (s.weight, then R_t's weights) is that of 0.3.0's neuron;
        # zero/constant fills consume no RNG.
        if self.n_h > 0:
            r_params = [self.k.weight, self.beta.weight]
            r_biases = [self.k.bias, self.beta.bias]
        else:
            r_params = [self.r.weight]
            r_biases = [self.r.bias]

        if cfg.s_r_init == 'kaiming':
            nn.init.kaiming_uniform_(self.s.weight, a=math.sqrt(5))
            for w in r_params:
                nn.init.kaiming_uniform_(w, a=math.sqrt(5))
            if self.s.bias is not None:
                nn.init.zeros_(self.s.bias)
        elif cfg.s_r_init == 'asymmetric':
            # s: Kaiming weights + positive bias so s fires from x_0=0 in most
            # dimensions, eliminating the r>0>s dead zone.
            # R_t: small_init weights + zero bias so it stays near zero and
            # does not clip x back toward 0 while s is learning to write.
            nn.init.kaiming_uniform_(self.s.weight, a=math.sqrt(5))
            for w in r_params:
                small_init_init_(w, dim=self.I)
            if self.s.bias is not None:
                nn.init.constant_(self.s.bias, 1.0)
        else:
            small_init_init_(self.s.weight, dim=self.I)
            for w in r_params:
                small_init_init_(w, dim=self.I)
            if self.s.bias is not None:
                nn.init.zeros_(self.s.bias)
        for b in r_biases:
            if b is not None:
                nn.init.zeros_(b)
        if self.n_h > 0:
            # R_t = prod_i (I - beta_i k_i k_i^T) = I + sum_i E_i + O(E^2) to
            # first order, with E_i = -beta_i k_i k_i^T independent across i.
            # Scaling beta's std by 1/sqrt(n_h) keeps Var(sum_i E_i), i.e.
            # R_t's initial deviation from the identity, independent of n_h.
            self.beta.weight.data.div_(math.sqrt(self.n_h))
        # Init 'o' — dim = number of state coordinates summed at the layer
        # (default: this group's own D), so wang_init_'s residual-stream
        # scaling holds for any state size and any split across groups.
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
    # R_t
    # ------------------------------------------------------------------

    def _project_transition(self, u: torch.Tensor):
        """The input projections R_t is built from (Linear ops, ambient dtype)."""
        if self.n_h > 0:
            return (self.k(u), self.beta(u))
        return (self.r(u),)


    def _build_R(self, projections, B: int, T: int) -> torch.Tensor:
        """
        R_t for every (batch, time, neuron): (B, T, K, d, d), in the dtype of
        the projections.

        n_h = 0: the reshaped r-projection.
        n_h > 0: prod_{i=1}^{n_h} (I - beta_i k_i k_i^T) — see
                 MinMaxNeuronGroupConfig.n_h / .normalize_keys.
        """
        d, K, n_h = self.d, self.K, self.n_h
        if n_h == 0:
            return projections[0].view(B, T, K, d, d)

        k    = projections[0].view(B, T, K, n_h, d)                # (B,T,K,n_h,d)
        beta = projections[1].view(B, T, K, n_h)                   # (B,T,K,n_h)
        eye  = torch.eye(d, device=k.device, dtype=k.dtype)

        R = None
        for i in range(n_h):
            k_i = k[..., i, :]                                       # (B,T,K,d)
            if self.group_cfg.normalize_keys:
                k_i = F.normalize(k_i, dim=-1)
            beta_i = beta[..., i]                                    # (B,T,K)
            kkt    = k_i.unsqueeze(-1) * k_i.unsqueeze(-2)           # (B,T,K,d,d)
            factor = eye - beta_i.unsqueeze(-1).unsqueeze(-1) * kkt  # (B,T,K,d,d)
            R = factor if R is None else R @ factor
        return R


    def _compute_R_low_rank(self, u: torch.Tensor, B: int, T: int) -> torch.Tensor:
        """R_t for n_h > 0, from an already dropout-applied u: (B, T, K, d, d)."""
        return self._build_R(self._project_transition(u), B, T)


    # ------------------------------------------------------------------
    # Scans
    # ------------------------------------------------------------------

    def _scan_scalar(self, s, projections, x0, B: int, T: int) -> torch.Tensor:
        """d = 1: element-wise scan over the (B, T, K) projections."""
        if self.n_h == 0:
            r = projections[0]                                       # (B,T,K)
        else:
            r = self._build_R(projections, B, T).reshape(B, T, self.K)
        return minmax_scan.all_states(r, s, x0)[:, 1:, :]            # (B,T,K)


    def _scan_matrix(self, s, projections, x0, B: int, T: int) -> torch.Tensor:
        """d > 1: matrix scan, with the neuron axis folded into the batch."""
        d, K = self.d, self.K
        R = self._build_R(projections, B, T)                         # (B,T,K,d,d)
        s_flat  = s.view(B, T, K, d).permute(0, 2, 1, 3).reshape(B * K, T, d)
        R_flat  = R.permute(0, 2, 1, 3, 4).reshape(B * K, T, d, d)
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
        projections = self._project_transition(u)

        compute_dtype = s.dtype
        region, scan_dtype = _scan_region(self.group_cfg.scan_dtype, u.device)
        with region:
            if scan_dtype is not None:
                s  = s.to(scan_dtype)
                x0 = x0.to(scan_dtype)
                projections = tuple(p.to(scan_dtype) for p in projections)
            # R_t is built inside the region so that, when pinned, its
            # construction also runs at the pinned precision.
            if self.d == 1:
                x_post = self._scan_scalar(s, projections, x0, B, T)
            else:
                x_post = self._scan_matrix(s, projections, x0, B, T)

        # ----- Compute outputs -----
        x_latest = x_post[:, -1, :]           # (B,D) — carried state (scan dtype if pinned)
        if scan_dtype is not None:
            x_post = x_post.to(compute_dtype)
        if self.group_cfg.output_gate:
            x_post = x_post * torch.sigmoid(self.o_g(u))
        output = self.o(x_post)               # (B,T,I)

        return output, x_latest


# ======================================================================
# MinMax Neuron — the 0.3.0 API: a group of d_state scalar-state neurons
# ======================================================================

@dataclass(frozen=True)
class MinMaxNeuronConfig:
    """
    Configuration for a MinMax Neuron as in 0.3.0: `d_state` scalar-state
    neurons (equivalently, MinMaxNeuronGroupConfig(d_state=1,
    num_neurons=d_state)).

    Fields are those of MinMaxNeuronGroupConfig, with `d_state` here meaning
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
    scan_dtype:           ScanDtype     = 'ambient'
    _state_size_in_layer: Optional[int] = None

    def __post_init__(self):
        _check_scan_dtype(self.scan_dtype)

    def to_group_config(self) -> MinMaxNeuronGroupConfig:
        return MinMaxNeuronGroupConfig(
            _num_blocks          = self._num_blocks,
            d_model              = self.d_model,
            d_state              = 1,
            num_neurons          = self.d_state,
            dropout              = self.dropout,
            train_init           = self.train_init,
            output_gate          = self.output_gate,
            s_r_init             = self.s_r_init,
            scan_dtype           = self.scan_dtype,
            _state_size_in_layer = self._state_size_in_layer,
        )


class MinMaxNeuron(MinMaxNeuronGroup):
    """
    MinMax Neuron as in 0.3.0: a MinMaxNeuronGroup of `cfg.d_state`
    scalar-state neurons, updating x_t ∈ R^D element-wise:

        x_{t+1} = max(min(r_t, x_t), s_t)

    Same parameters (s, r, o, o_g, _initial_state), shapes, initialisation and
    computation as in 0.3.0. `cfg` is the MinMaxNeuronConfig it was built
    from; `group_cfg` the equivalent MinMaxNeuronGroupConfig.
    """

    def __init__(self, cfg: MinMaxNeuronConfig):
        nn.Module.__init__(self)
        self.cfg       = cfg
        self.group_cfg = cfg.to_group_config()
        self._build()
