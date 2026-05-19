# SPDX-FileCopyrightText: 2026 Alessandro Ronca
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

import torch
import torch.nn as nn
import math
from typing import Literal
from dataclasses import dataclass

from .initialisers import small_init_init_


InitType = Literal["default", "scaled"]

@dataclass
class BasicConvConfig:
    """
    Config for BasicConv.

    Fields
    ------
    embedding_dim : int
        Token / feature dimension.
    init : 'default' | 'scaled'
        Weight initialisation; 'scaled' uses small_init_.
    """

    embedding_dim:      int
    init:               InitType = 'scaled'
    _in_embedding_dim:  int      = -1

    def __post_init__(self):
        self._in_embedding_dim = 2*self.embedding_dim

class BasicConv(nn.Module):
    """
    Learned linear mixing of the previous and current token representations.

    Concatenates [u_{t-1}, u_t] and projects back to d_model, giving the
    network a one-step causal receptive field.  State is u_{t-1}.
    """

    def __init__(self, cfg):
        super().__init__()

        self.__cfg = cfg
        self._initial_state = nn.Parameter(torch.zeros(cfg.embedding_dim), requires_grad=False)
        self.conv_fn        = nn.Linear(cfg._in_embedding_dim, cfg.embedding_dim)

        self.reset()

    def reset(self):
        if self.__cfg.init == 'scaled':
            small_init_init_(self.conv_fn.weight, dim=self.__cfg._in_embedding_dim)
            if self.conv_fn.bias is not None:
                nn.init.zeros_(self.conv_fn.bias)

    @property
    def initial_state(self):
        return self._initial_state


    def forward(self, u: torch.Tensor, state: torch.Tensor):
        assert u.dim() == 3, f"Expected input 'u' of shape (B,T,D)"

        B, T, _ = u.shape

        if state.dim() == 1: # initial state
            state = state.reshape(1, 1, -1).expand(B, 1, -1)
        else:
            state = state.reshape(B, 1, -1)

        prev_u = torch.cat([state, u[:, :-1, :]], dim=1)
        conv   = self.conv_fn(torch.cat([prev_u, u], dim=-1))
        return conv, u[:, -1, :]


