# SPDX-FileCopyrightText: 2026 Alessandro Ronca
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

import math
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Literal
from .modules.initialisers import wang_init_, small_init_init_

from . import minmax_scan


SRInitType = Literal['small_init', 'kaiming', 'asymmetric']


@dataclass(frozen=True)
class MinMaxNeuronConfig:
    """
    Configuration for a single MinMax Neuron.

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
    s_r_init : 'small_init' | 'kaiming'
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
    """

    _num_blocks: int
    d_model:     int
    d_state:     int
    dropout:     float      = 0.0
    train_init:  bool       = False
    output_gate: bool       = True
    s_r_init:    SRInitType = 'small_init'


class MinMaxNeuron(nn.Module):
    """
    The core recurrent cell of the MinMax RNC.

    Maintains a hidden state x_t ∈ R^D updated by the MinMax recurrence:

        x_{t+1} = max(min(r_t, x_t), s_t)

    All states for a sequence of length T are computed simultaneously via a parallel prefix scan in
    O(log T) depth instead of O(T).

    Output projection:

        y_t = W_o x_t                             (output_gate=False)
        y_t = W_o (x_t ⊙ σ(W_g u_t))              (output_gate=True)
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
        # scaling holds for any d_state, not just when d_state == d_model.
        wang_init_(self.o.weight, dim=self.D, num_blocks=self.cfg._num_blocks)
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
        device = u.device

        if state.dim() == 1: # state is the initial initial state
            x0 = state.unsqueeze(0).expand(B,D)
        else:
            x0 = state

        # Shape of u:  (B,T,I)
        # Shape of x0: (B,D)


        u = self.drop(u)  # (B,T,I)
        s = self.s(u)     # (B,T,D)
        r = self.r(u)     # (B,T,D)

        x_post = minmax_scan.all_states(r, s, x0)
        x_post = x_post[:,1:,:]


        # ----- Compute outputs -----
        x_latest = x_post[:,-1,:]         # (B,T,D)
        if self.cfg.output_gate:
            x_post = x_post * torch.sigmoid(self.o_g(u))
        output = self.o(x_post)           # (B,T,I)

        return output, x_latest

