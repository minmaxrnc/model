# SPDX-FileCopyrightText: 2026 Alessandro Ronca
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

import torch
import torch.nn as nn
import math
from typing import Literal
from dataclasses import dataclass


@dataclass
class GatedConvConfig:
    """
    Config for GatedConv.

    Fields
    ------
    embedding_dim : int
        Token / feature dimension.
    init_val : float
        Initial value of the gate logit g.  σ(g) is the weight given to
        the *previous* token; 1 - σ(g) to the *current* token.
        init_val=0.0  → equal mix (gate ≈ 0.5).
        init_val=-1.0 → biased toward the current token (gate ≈ 0.27).
    """

    embedding_dim: int
    init_val:      float = 0.0


class GatedConv(nn.Module):
    """
    Learned scalar interpolation between the previous and current token.

    For each feature dimension d independently:

        out_t = σ(g_d) · u_{t-1,d} + (1 − σ(g_d)) · u_{t,d}

    where g ∈ R^D is a learned parameter vector.  State is u_{t-1}.
    """

    def __init__(self, cfg):
        super().__init__()

        self.__cfg = cfg

        self._initial_state = nn.Parameter(
            torch.zeros(cfg.embedding_dim),
            requires_grad=False,
        )

        self.g = nn.Parameter(torch.empty(cfg.embedding_dim))

        self.reset()

    def reset(self):
        nn.init.constant_(self.g, self.__cfg.init_val)


    @property
    def initial_state(self):
        return self._initial_state

    def forward(self, u: torch.Tensor, state: torch.Tensor):
        B, T, D = u.shape

        if state.dim() == 1:  # initial state
            state = state.reshape(1, 1, -1).expand(B, 1, -1)
        else:
            state = state.reshape(B, 1, -1)

        prev_u = torch.cat([state, u[:, :-1, :]], dim=1)  # (B,T,D)

        gate = torch.sigmoid(self.g).view(1, 1, -1)       # (1,1,D), broadcasts over (B,T,D)
        out  = gate * prev_u + (1.0 - gate) * u

        return out, u[:, -1, :]

