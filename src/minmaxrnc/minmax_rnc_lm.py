# SPDX-FileCopyrightText: 2026 Alessandro Ronca
j
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

import torch
import torch.nn as nn

from dataclasses import dataclass

from .minmax_rnc             import MinMaxRNC, MinMaxRNCConfig
from .modules.initialisers   import small_init_init_


@dataclass(frozen=True)
class MinMaxRNCLMConfig:
    """
    Configuration for the MinMax RNC language model.

    Fields
    ------
    backbone : MinMaxRNCConfig
        Config for the MinMaxRNC backbone. The embedding dimension is taken
        from backbone.d_model.
    head_dropout : float
        Dropout applied to the backbone output before the LM head projection.
    tie_weights : bool
        If True (default), the LM head weight matrix is shared with the token
        embedding matrix, halving those parameters and acting as a regulariser.
    """

    backbone:     MinMaxRNCConfig
    head_dropout: float = 0.0
    tie_weights:  bool  = True


class MinMaxRNC_LM(MinMaxRNC):
    """
    MinMax RNC with a token embedding layer and a language-model head.

    Wraps MinMaxRNC with:
    - A token embedding  (vocab_size × d_model)
    - A dropout before the output projection
    - A linear LM head   (d_model × vocab_size), optionally tied to the embedding

    Inputs
    ------
    tokens : LongTensor  (B, T)
        Token indices in [0, vocab_size).
    state : list[dict] | None
        Recurrent state from a previous call.
    return_state : bool
        If True, also return the updated state.

    Outputs
    -------
    logits : Tensor  (B, T, vocab_size)
    state  : list[dict]  — only when return_state=True
    """

    def __init__(self, vocab_size: int, cfg: MinMaxRNCLMConfig):
        self.__lm_cfg   = cfg
        self.__vocab_size = vocab_size
        super().__init__(cfg.backbone)   # calls reset() → MinMaxRNC.reset() then our additions
        self.__lm_reset()

    def reset(self):
        super().reset()
        self.__lm_reset()

    def __lm_reset(self):
        d_model = self.__lm_cfg.backbone.d_model
        self.token_emb = nn.Embedding(self.__vocab_size, d_model)
        self.lm_head   = nn.Linear(d_model, self.__vocab_size, bias=False)
        self.head_drop = nn.Dropout(self.__lm_cfg.head_dropout)

        small_init_init_(self.token_emb.weight, dim=d_model)
        if self.__lm_cfg.tie_weights:
            self.lm_head.weight = self.token_emb.weight
        else:
            small_init_init_(self.lm_head.weight, dim=d_model)

    def forward(self, tokens: torch.Tensor, unroll_steps: int, state=None, return_state: bool = False):
        y, state = super().forward(
            self.token_emb(tokens), unroll_steps, state=state, return_state=True
        )
        logits = self.lm_head(self.head_drop(y))
        if return_state:
            return logits, state
        return logits
