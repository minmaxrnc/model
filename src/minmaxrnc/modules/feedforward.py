# SPDX-FileCopyrightText: 2026 Alessandro Ronca
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""
Feed-forward sub-layers.

FeedForward
    Two-layer MLP (proj_up → activation → proj_down).
    Vaswani et al. (2017). "Attention Is All You Need." NeurIPS.
    https://arxiv.org/abs/1706.03762

GatedFeedForward
    Gated variant: proj_up outputs a fused [gate | value] tensor; the gate
    branch passes through an activation, then gate ⊙ value is projected down.

        output = W_down · (σ(gate) ⊙ value)

    With σ = ReLU  → ReGLU
    With σ = GELU  → GeGLU
    With σ = SiLU  → SwiGLU

    Dauphin et al. (2017). "Language Modeling with Gated Convolutional
    Networks." ICML.  https://arxiv.org/abs/1612.08083

    Shazeer (2020). "GLU Variants Improve Transformer."
    https://arxiv.org/abs/2002.05202
"""

import math
from dataclasses import dataclass
from typing import Callable, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from .initialisers import small_init_init_, wang_init_


# ---------------------------------------------------------------------------
# Activation registry
# ---------------------------------------------------------------------------

_ACT_FNS: dict = {
    'relu':    F.relu,
    'relu^2':  lambda x: F.relu(x).square(),
    'gelu':    F.gelu,
    'swish':   F.silu,
    'sigmoid': torch.sigmoid,
    'selu':    F.selu,
}


def _get_act_fn(name: str) -> Callable:
    if name not in _ACT_FNS:
        raise ValueError(f"Unknown activation '{name}'. Available: {sorted(_ACT_FNS)}")
    return _ACT_FNS[name]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

InitType = Literal['basic', 'scaled']
FFType   = Literal['gated', 'basic']


@dataclass
class FeedForwardConfig:
    _num_blocks: int

    embedding_dim:                int      = -1       # set by the parent module via dataclasses.replace()
    embedding_dim_out:            int      = -1       # defaults to embedding_dim when <= 0
    proj_factor:                  float    = 1.3
    act_fn:                       str      = 'relu'
    bias:                         bool     = True
    dropout:                      float    = 0.0
    ffn_type:                     FFType   = 'gated'
    init:                         InitType = 'scaled'
    round_proj_up_to_multiple_of: int      = 2
    round_proj_up_dim_up:         bool     = True

    _proj_up_dim: int = None   # derived in __post_init__

    def __post_init__(self):
        _get_act_fn(self.act_fn)   # validate early
        if self.embedding_dim > 0:
            raw = self.proj_factor * self.embedding_dim
            k   = raw / self.round_proj_up_to_multiple_of
            k   = math.ceil(k) if self.round_proj_up_dim_up else math.floor(k)
            self._proj_up_dim = int(k * self.round_proj_up_to_multiple_of)
        else:
            self._proj_up_dim = 0


# ---------------------------------------------------------------------------
# Basic feed-forward
# ---------------------------------------------------------------------------

class FeedForward(nn.Module):

    def __init__(self, cfg: FeedForwardConfig):
        super().__init__()
        self.cfg = cfg
        d_out = cfg.embedding_dim_out if cfg.embedding_dim_out > 0 else cfg.embedding_dim
        self.proj_up   = nn.Linear(cfg.embedding_dim, cfg._proj_up_dim, bias=cfg.bias)
        self.proj_down = nn.Linear(cfg._proj_up_dim, d_out, bias=cfg.bias)
        self.act       = _get_act_fn(cfg.act_fn)
        self.drop      = nn.Dropout(cfg.dropout)
        self._init_weights()

    def _init_weights(self):
        if self.cfg.init == 'scaled':
            small_init_init_(self.proj_up.weight, dim=self.cfg.embedding_dim)
            if self.proj_up.bias is not None:
                nn.init.zeros_(self.proj_up.bias)
            wang_init_(self.proj_down.weight, dim=self.cfg._proj_up_dim,
                       num_blocks=self.cfg._num_blocks)
            if self.proj_down.bias is not None:
                nn.init.zeros_(self.proj_down.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.proj_down(self.act(self.proj_up(x))))


# ---------------------------------------------------------------------------
# Gated feed-forward  —  Dauphin et al. (2017); Shazeer (2020)
# ---------------------------------------------------------------------------

class GatedFeedForward(nn.Module):

    def __init__(self, cfg: FeedForwardConfig):
        super().__init__()
        self.cfg = cfg
        d_out = cfg.embedding_dim_out if cfg.embedding_dim_out > 0 else cfg.embedding_dim
        # Fused projection: first half is gate pre-activation, second half is value.
        self.proj_up   = nn.Linear(cfg.embedding_dim, 2 * cfg._proj_up_dim, bias=cfg.bias)
        self.proj_down = nn.Linear(cfg._proj_up_dim, d_out, bias=cfg.bias)
        self.act       = _get_act_fn(cfg.act_fn)
        self.drop      = nn.Dropout(cfg.dropout)
        self._init_weights()

    def _init_weights(self):
        if self.cfg.init == 'scaled':
            small_init_init_(self.proj_up.weight, dim=self.cfg.embedding_dim)
            if self.proj_up.bias is not None:
                nn.init.zeros_(self.proj_up.bias)
            wang_init_(self.proj_down.weight, dim=self.cfg._proj_up_dim,
                       num_blocks=self.cfg._num_blocks)
            if self.proj_down.bias is not None:
                nn.init.zeros_(self.proj_down.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_pre, value = self.proj_up(x).chunk(2, dim=-1)
        return self.drop(self.proj_down(self.act(gate_pre) * value))


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_feedforward(config: FeedForwardConfig) -> nn.Module:
    if config.ffn_type == 'gated':
        return GatedFeedForward(config)
    if config.ffn_type == 'basic':
        return FeedForward(config)
    raise ValueError(f"Unknown ffn_type '{config.ffn_type}'")
