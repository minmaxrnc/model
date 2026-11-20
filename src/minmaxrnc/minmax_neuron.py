# SPDX-FileCopyrightText: 2026 Alessandro Ronca
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

import contextlib
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Literal
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
'fp32' | 'bf16' | 'fp16' — *pinned*: every scan input (and, for matrix
    neurons, the construction of R_t) is cast to this dtype and the scan runs
    with autocast disabled, so its precision is exactly the requested one
    regardless of any ambient autocast. The scan's outputs are cast back to the
    ambient compute dtype for the output projections; the carried recurrent
    state stays in the pinned dtype.
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
# Diagonal neuron (the 0.3.0 MinMax Neuron)
# ======================================================================

@dataclass(frozen=True)
class MinMaxNeuronConfig:
    """
    Configuration for a single (diagonal) MinMax Neuron.

    Fields
    ------
    _num_blocks : int
        Total number of residual blocks in the enclosing model.  Used to scale
        the output projection at initialisation (wang_init_) so that the
        combined contribution of all blocks to the residual stream stays O(1).
    d_model : int
        Dimension of the input and output (the residual-stream width).
    d_state : int
        Dimension of the hidden state x_t.  Larger values give the neuron
        more memory capacity but increase parameter count linearly.
    dropout : float
        Dropout probability applied to the input u before projection.
    train_init : bool
        If True, the initial hidden state x_0 is a learned parameter.
        If False (default), x_0 is fixed at zero.
    output_gate : bool
        If True, the output is element-wise multiplied by a sigmoid-gated
        projection of the input before the final linear: y = W_o(x ⊙ σ(W_g u)).
    s_r_init : 'small_init' | 'kaiming' | 'asymmetric'
        Initialisation scheme for the s and r input projections.
        'small_init' (default) — normal with std = sqrt(2/(5·d_model)),
        keeping early activations small in deep networks.
        'kaiming' — PyTorch default Kaiming uniform (fan_in=d_model), which
        gives a larger initial scale and more aggressive early state transitions.
        'asymmetric' — treats s and r differently: s gets Kaiming weights plus
        a positive bias (+1.0) so it fires immediately from x_0=0 and encodes
        token identity with sufficient variance; r gets small_init weights and
        zero bias so it stays near zero early on and does not clip x back toward
        the dead zone.
    scan_dtype : 'ambient' | 'fp32' | 'bf16' | 'fp16'
        Precision of the MinMax scan (new in 0.4.0); see ScanDtype.
        'ambient' (default) reproduces 0.3.0 exactly.
    _num_neurons_in_layer : int
        Total number of neurons in the enclosing MinMaxLayer whose outputs are
        summed (new in 0.4.0). Folded into wang_init_'s scaling of the output
        projection, so the layer's *total* residual-stream contribution stays
        O(1) however many neurons it has. 1 (default) for a layer whose only
        neuron is this one — identical to 0.3.0.
    """

    _num_blocks: int
    d_model:     int
    d_state:     int
    dropout:     float      = 0.0
    train_init:  bool       = False
    output_gate: bool       = True
    s_r_init:    SRInitType = 'small_init'
    # --- new in 0.4.0 (appended, so positional construction is unchanged) ---
    scan_dtype:            ScanDtype = 'ambient'
    _num_neurons_in_layer: int       = 1

    def __post_init__(self):
        _check_scan_dtype(self.scan_dtype)


class MinMaxNeuron(nn.Module):
    """
    The diagonal (element-wise) recurrent cell of the MinMax RNC.

    Maintains a hidden state x_t ∈ R^D updated by the MinMax recurrence:

        x_{t+1} = max(min(r_t, x_t), s_t)

    All states for a sequence of length T are computed simultaneously via a parallel prefix scan in
    O(log T) depth instead of O(T).

    Output projection:

        y_t = W_o x_t                             (output_gate=False)
        y_t = W_o (x_t ⊙ σ(W_g u_t))              (output_gate=True)

    See MatrixMinMaxNeuron for the full-matrix generalisation.
    """

    def __init__(self, cfg: MinMaxNeuronConfig):

        super().__init__()

        self.cfg = cfg

        self.I = I = cfg.d_model
        self.D = D = cfg.d_state

        self._initial_state = nn.Parameter(torch.zeros(D), requires_grad=cfg.train_init)

        self.drop = nn.Dropout(cfg.dropout)

        self.s = nn.Linear(I, D)
        self.r = nn.Linear(I, D)
        self.o = nn.Linear(D, I)
        if cfg.output_gate:
            self.o_g = nn.Linear(I, D)

        self.reset()


    def reset(self):
        # Init 's' and 'r'
        if self.cfg.s_r_init == 'kaiming':
            nn.init.kaiming_uniform_(self.s.weight, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.r.weight, a=math.sqrt(5))
            if self.s.bias is not None:
                nn.init.zeros_(self.s.bias)
            if self.r.bias is not None:
                nn.init.zeros_(self.r.bias)
        elif self.cfg.s_r_init == 'asymmetric':
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
        # Init 'o' — use dim=self.D (d_state) so wang_init_'s residual-stream
        # scaling holds for any d_state, not just when d_state == d_model;
        # multiplied by _num_neurons_in_layer so the scaling also accounts for
        # every sibling neuron this one's output is summed with at the layer
        # level (default 1 → plain dim=self.D, as in 0.3.0).
        wang_init_(
            self.o.weight,
            dim=self.D * self.cfg._num_neurons_in_layer,
            num_blocks=self.cfg._num_blocks,
        )
        if self.o.bias is not None:
            nn.init.zeros_(self.o.bias)
        # Init 'o_g' — small weights keep the gate near 0.5 (sigmoid(~0)) at
        # init; zero bias removes the per-dimension offset from PyTorch's default.
        if self.cfg.output_gate:
            small_init_init_(self.o_g.weight, dim=self.I)
            if self.o_g.bias is not None:
                nn.init.zeros_(self.o_g.bias)


    @property
    def initial_state(self):
        return self._initial_state


    def forward(self, u: torch.Tensor, state: torch.Tensor):
        """
        Compute updated state for a sequence using closed form with initial state.

        u:     (B, T, I)
        state: (1/B,D,)   (state before the first step in the input sequence)

        Returns: 1) sequence of outputs: (B, T, I)
                 2) last state:          (B, D)
        """

        B, T, I = u.shape
        D = self.D

        if state.dim() == 1: # state is the initial initial state
            x0 = state.unsqueeze(0).expand(B,D)
        else:
            x0 = state

        # Shape of u:  (B,T,I)
        # Shape of x0: (B,D)

        u = self.drop(u)  # (B,T,I)
        s = self.s(u)     # (B,T,D)
        r = self.r(u)     # (B,T,D)

        compute_dtype = s.dtype
        region, scan_dtype = _scan_region(self.cfg.scan_dtype, u.device)
        with region:
            if scan_dtype is not None:
                r, s, x0 = r.to(scan_dtype), s.to(scan_dtype), x0.to(scan_dtype)
            x_post = minmax_scan.all_states(r, s, x0)
        x_post = x_post[:,1:,:]

        # ----- Compute outputs -----
        x_latest = x_post[:,-1,:]         # (B,D) — carried state (scan dtype if pinned)
        if scan_dtype is not None:
            x_post = x_post.to(compute_dtype)
        if self.cfg.output_gate:
            x_post = x_post * torch.sigmoid(self.o_g(u))
        output = self.o(x_post)           # (B,T,I)

        return output, x_latest


# ======================================================================
# Matrix neurons (new in 0.4.0)
# ======================================================================

@dataclass(frozen=True)
class MatrixMinMaxNeuronConfig:
    """
    Configuration for a group of one or more matrix MinMax Neurons of the
    same d_state, all fed the same input and summed into a single output.

    Same fields as MinMaxNeuronConfig, plus `num_neurons`, `n_h` and
    `normalize_keys`; see there for a description of the shared fields. The
    structural difference from the diagonal neuron is that this neuron's
    r-projection produces a full d_state × d_state matrix R_t at every step,
    instead of a single d_state-vector.

    Normally built by MinMaxRNCConfig.layer_cfg from a MatrixNeuronGroup in
    MinMaxRNCConfig.matrix_neurons.

    num_neurons : int
        Batches `num_neurons` independent matrix neurons of this d_state into
        one module: their scans run as a single wider parallel scan (extra
        batch axis) instead of one Python call per neuron, and their outputs
        are summed internally. Mathematically and statistically this is
        identical to constructing `num_neurons` separate neurons with
        num_neurons=1 and summing their outputs — only the compute is batched,
        which matters because with a small d_state each neuron's scan does
        very little work per step.

        Params scale as d_model · d_state² · num_neurons (the r-projection,
        n_h=0) and the scan's work as O(num_neurons · T log T · d_state³), so
        d_state should be kept small; use several groups / more neurons
        rather than growing a single neuron's d_state.

        `_num_neurons_in_layer` is a *different* count: the total neuron count
        across the whole enclosing MinMaxLayer (this group's num_neurons, plus
        every other group's, plus 1 if there's a diagonal neuron) — that
        total, not num_neurons, is what wang_init_'s output-projection
        scaling needs.

    n_h : int
        0 (default) — R_t is produced by a single unconstrained affine
        projection of the input: every entry of the d_state × d_state matrix
        is an independent linear function of u_t.

        > 0 — EXPERIMENTAL. R_t is instead built as a product of n_h rank-1
        corrections to the identity, in the style of DeltaProduct's
        Householder-product state-transition matrix (Siems et al., NeurIPS
        2025):

            R_t = ∏_{i=1}^{n_h} (I - β_i(u_t) · k_i(u_t) k_i(u_t)^T)

        Unlike DeltaProduct, β_i is an unconstrained affine function of u_t
        (no sigmoid bounding into [0,2]) and k_i is not unit-normalised by
        default (so each factor is a general rank-1 perturbation of the
        identity): DeltaProduct needs bounded eigenvalues to keep its *linear*
        recurrence stable, but the (max,min)-semiring recurrence is idempotent
        — every entry of a composed R (and of x) is *selected* from, never
        accumulated from, per-step values — so there is no analogous
        blow-up/vanishing mode. Params scale as d_model · n_h · d_state.
        No positive experimental results with n_h > 0 yet.
    normalize_keys : bool
        Only meaningful when n_h > 0. If True, each k_i(u_t) is L2-normalised
        along its d_state axis before forming k_i k_i^T, as in DeltaProduct,
        so β_i alone controls each factor's magnitude.
    """

    _num_blocks:           int
    d_model:               int
    d_state:               int
    num_neurons:           int        = 1
    n_h:                   int        = 0
    normalize_keys:        bool       = False
    dropout:               float      = 0.0
    train_init:            bool       = False
    output_gate:           bool       = True
    s_r_init:              SRInitType = 'small_init'
    scan_dtype:            ScanDtype  = 'ambient'
    _num_neurons_in_layer: int        = 1

    def __post_init__(self):
        _check_scan_dtype(self.scan_dtype)
        if self.d_state <= 0:
            raise ValueError(f"d_state must be > 0, got {self.d_state}")
        if self.num_neurons <= 0:
            raise ValueError(f"num_neurons must be > 0, got {self.num_neurons}")
        if self.n_h < 0:
            raise ValueError(f"n_h must be >= 0, got {self.n_h}")


class MatrixMinMaxNeuron(nn.Module):
    """
    A group of `num_neurons` independent full-matrix recurrent cells of the
    MinMax RNC, all fed the same input and summed into a single output.

    Each of the `num_neurons` neurons maintains its own hidden state
    x_t ∈ R^D updated by the MinMax matrix recurrence:

        x_{t+1} = max(R_t ⊗ x_t, s_t)

    where R_t ∈ R^{D×D} and (R ⊗ x)_i = max_j min(R_ij, x_j) is the
    (max,min)-semiring matrix-vector product (see minmax_operator.py). Setting
    off-diagonal entries of R_t to -inf recovers the diagonal recurrence
    implemented by MinMaxNeuron as a special case; with D=1 the two coincide.

    R_t itself can be parametrised in two ways (see MatrixMinMaxNeuronConfig.n_h):
    a single unconstrained affine projection of u_t (n_h=0, default), or a
    product of n_h rank-1 corrections to the identity (experimental).

    All states for a sequence of length T, for all `num_neurons` neurons at
    once, are computed via a single parallel prefix scan in O(log T) depth
    (the neuron axis is folded into the scan's batch dimension); each combine
    step costs O(D³), so total work is O(num_neurons · T log T · D³) and the
    scan materialises a (B·num_neurons, T, D, D, D) intermediate per step —
    see minmax_scan.all_states_matrix.

    Output projection (summed over the `num_neurons` neurons):

        y_t = Σ_k W_o^(k) x_t^(k)                         (output_gate=False)
        y_t = Σ_k W_o^(k) (x_t^(k) ⊙ σ(W_g^(k) u_t))      (output_gate=True)

    Implemented as a single Linear(num_neurons·D, I) whose weight columns are
    the per-neuron W_o^(k) blocks concatenated — algebraically identical to
    summing num_neurons separate Linear(D, I) projections.
    """

    def __init__(self, cfg: MatrixMinMaxNeuronConfig):

        super().__init__()

        self.cfg = cfg

        self.I = I = cfg.d_model
        self.D = D = cfg.d_state
        self.K = K = cfg.num_neurons
        self.n_h = n_h = cfg.n_h

        self._initial_state = nn.Parameter(torch.zeros(K, D), requires_grad=cfg.train_init)

        self.drop = nn.Dropout(cfg.dropout)

        self.s = nn.Linear(I, K * D)
        if n_h > 0:
            self.k    = nn.Linear(I, K * n_h * D)
            self.beta = nn.Linear(I, K * n_h)
        else:
            self.r = nn.Linear(I, K * D * D)
        self.o = nn.Linear(K * D, I)
        if cfg.output_gate:
            self.o_g = nn.Linear(I, K * D)

        self.reset()


    def reset(self):
        # Init 's' and 'r' (or 's' and 'k'/'beta' when n_h > 0) — fan-in is
        # always self.I regardless of the output size, so the same schemes as
        # the diagonal neuron apply as-is; these init functions sample each
        # element i.i.d. from a distribution that depends only on the passed
        # `dim`, so applying them to the whole stacked K-neuron (and, for
        # k/beta, n_h-factor) tensor is statistically identical to applying
        # them separately per neuron/factor. r's (and k's/beta's) bias is
        # always zeroed regardless of scheme — only s's bias varies.
        if self.n_h > 0:
            r_params = [self.k.weight, self.beta.weight]
            r_biases = [self.k.bias, self.beta.bias]
        else:
            r_params = [self.r.weight]
            r_biases = [self.r.bias]

        if self.cfg.s_r_init == 'kaiming':
            nn.init.kaiming_uniform_(self.s.weight, a=math.sqrt(5))
            for w in r_params:
                nn.init.kaiming_uniform_(w, a=math.sqrt(5))
            if self.s.bias is not None:
                nn.init.zeros_(self.s.bias)
        elif self.cfg.s_r_init == 'asymmetric':
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
        # Init 'o' — dim=self.D (d_state, not K*D) so the per-neuron scaling
        # matches a single standalone neuron regardless of K, times
        # _num_neurons_in_layer so it also accounts for every other neuron in
        # the layer this group's output is summed with.
        wang_init_(
            self.o.weight,
            dim=self.D * self.cfg._num_neurons_in_layer,
            num_blocks=self.cfg._num_blocks,
        )
        if self.o.bias is not None:
            nn.init.zeros_(self.o.bias)
        if self.cfg.output_gate:
            small_init_init_(self.o_g.weight, dim=self.I)
            if self.o_g.bias is not None:
                nn.init.zeros_(self.o_g.bias)


    @property
    def initial_state(self):
        return self._initial_state


    def _project_R(self, u: torch.Tensor, B: int, T: int):
        """The input projections R_t is built from (Linear ops, ambient dtype)."""
        D, K, n_h = self.D, self.K, self.n_h
        if n_h > 0:
            return (
                self.k(u).view(B, T, K, n_h, D),        # (B,T,K,n_h,D)
                self.beta(u).view(B, T, K, n_h),        # (B,T,K,n_h)
            )
        return (self.r(u).view(B, T, K, D, D),)         # (B,T,K,D,D)


    def _build_R(self, projections) -> torch.Tensor:
        """
        Build R_t (B,T,K,D,D) from _project_R's outputs, in their dtype.

        n_h = 0: R_t is the reshaped r-projection.
        n_h > 0: R_t = prod_{i=1}^{n_h} (I - beta_i k_i k_i^T) — see
                 MatrixMinMaxNeuronConfig.n_h / .normalize_keys.
        """
        if self.n_h == 0:
            return projections[0]

        k, beta = projections
        eye = torch.eye(self.D, device=k.device, dtype=k.dtype)

        R = None
        for i in range(self.n_h):
            k_i = k[..., i, :]                                       # (B,T,K,D)
            if self.cfg.normalize_keys:
                k_i = F.normalize(k_i, dim=-1)
            beta_i = beta[..., i]                                    # (B,T,K)
            kkt    = k_i.unsqueeze(-1) * k_i.unsqueeze(-2)           # (B,T,K,D,D)
            factor = eye - beta_i.unsqueeze(-1).unsqueeze(-1) * kkt  # (B,T,K,D,D)
            R = factor if R is None else R @ factor
        return R


    def _compute_R_low_rank(self, u: torch.Tensor, B: int, T: int) -> torch.Tensor:
        """R_t for n_h > 0, from an already dropout-applied u: (B, T, K, D, D)."""
        return self._build_R(self._project_R(u, B, T))


    def forward(self, u: torch.Tensor, state: torch.Tensor):
        """
        Compute updated state for a sequence using closed form with initial state.

        u:     (B, T, I)
        state: (K,D) or (B,K,D)   (state before the first step in the input sequence)

        Returns: 1) sequence of outputs (already summed over the K neurons): (B, T, I)
                 2) last state, per neuron:                                  (B, K, D)
        """

        B, T, I = u.shape
        D = self.D
        K = self.K

        if state.dim() == 2: # (K,D): the initial initial state
            x0 = state.unsqueeze(0).expand(B, K, D)
        else:
            x0 = state

        u = self.drop(u)                              # (B,T,I)
        s = self.s(u).view(B, T, K, D)                # (B,T,K,D)
        projections = self._project_R(u, B, T)

        compute_dtype = s.dtype
        region, scan_dtype = _scan_region(self.cfg.scan_dtype, u.device)
        with region:
            if scan_dtype is not None:
                s  = s.to(scan_dtype)
                x0 = x0.to(scan_dtype)
                projections = tuple(p.to(scan_dtype) for p in projections)
            # Built inside the region so that, when pinned, the rank-1
            # products also run at the pinned precision.
            R = self._build_R(projections)            # (B,T,K,D,D)

            # Fold the neuron axis K into the scan's batch axis, alongside the
            # real batch B, so all K*B independent scans run as one call.
            s_flat  = s.permute(0, 2, 1, 3).reshape(B * K, T, D)          # (B*K,T,D)
            R_flat  = R.permute(0, 2, 1, 3, 4).reshape(B * K, T, D, D)    # (B*K,T,D,D)
            x0_flat = x0.reshape(B * K, D)                                # (B*K,D)

            x_post_flat = minmax_scan.all_states_matrix(R_flat, s_flat, x0_flat)
        x_post_flat = x_post_flat[:, 1:, :]                               # (B*K,T,D)

        x_post = x_post_flat.view(B, K, T, D).permute(0, 2, 1, 3)         # (B,T,K,D)

        # ----- Compute outputs -----
        x_latest = x_post[:, -1, :, :]                                    # (B,K,D)
        x_post = x_post.reshape(B, T, K * D)                              # (B,T,K*D)
        if scan_dtype is not None:
            x_post = x_post.to(compute_dtype)
        if self.cfg.output_gate:
            x_post = x_post * torch.sigmoid(self.o_g(u))
        output = self.o(x_post)           # (B,T,I) — already summed over K by construction of self.o

        return output, x_latest
