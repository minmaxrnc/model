# SPDX-FileCopyrightText: 2026 Alessandro Ronca
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

import torch
import torch.nn as nn
from dataclasses import dataclass
from .modules.initialisers import wang_init_, small_init_init_

from . import minmax_scan


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
    """

    _num_blocks: int
    d_model:     int
    d_state:     int
    dropout:     float    = 0.0
    train_init:  bool     = False
    output_gate: bool     = True


class MinMaxNeuron(nn.Module):
    """
    The core recurrent cell of the MinMax RNC.

    Maintains a hidden state x_t ∈ R^D updated by the MinMax recurrence:

        x_{t+1} = max(min(r_t, x_t), s_t)

    All states for a sequence of length T are computed simultaneously via a parallel prefix scan in
    O(log T) depth instead of O(T).

    Output projection:

        y_t = W_o x_t                             (output_gate=False)
        y_t = W_o (x_t ⊙ W_g u_t)                 (output_gate=True)
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
        # Init 's'
        small_init_init_(self.s.weight, dim=self.I)
        if self.s.bias is not None:
            nn.init.zeros_(self.s.bias)
        small_init_init_(self.r.weight, dim=self.I)
        # Init 'r'
        if self.r.bias is not None:
            nn.init.zeros_(self.r.bias)
        # Init 'o'
        wang_init_(
            self.o.weight,
            dim=self.I,
            num_blocks= self.cfg._num_blocks,
        )
        if self.o.bias is not None:
            nn.init.zeros_(self.o.bias)


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
            x_post = x_post * self.o_g(u) # (B,T,D)
        output = self.o(x_post)           # (B,T,I)

        return output, x_latest

