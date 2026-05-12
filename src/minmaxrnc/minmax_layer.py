# SPDX-FileCopyrightText: 2026 Alessandro Ronca
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

from typing import Sequence, Optional, List, Tuple, Union, Literal
from dataclasses import dataclass, replace

import torch
import torch.nn as nn

from .minmax_neuron import MinMaxNeuron, MinMaxNeuronConfig

from .modules.feedforward import FeedForwardConfig, create_feedforward
from .modules.basic_conv import BasicConv, BasicConvConfig
from .modules.gated_conv import GatedConv, GatedConvConfig


NormType = Literal['none', 'layernorm', 'rmsnorm']


@dataclass(frozen=True)
class MinMaxLayerConfig:
    """
    Configuration for one MinMax Layer.

    This is normally constructed automatically by MinMaxRNCConfig.layer_cfg;
    direct construction is only needed for non-standard layer shapes.

    Fields
    ------
    neuron : MinMaxNeuronConfig
        Config for the MinMax Neuron sub-module.
    conv : BasicConvConfig | GatedConvConfig
        Config for the short-range convolution applied before the FFN.
    d_model : int
        Residual-stream width (must match neuron.d_model).
    first_in_dropout : float
        Dropout probability for the FFN in the *first* layer only.  Allows a
        higher input-level dropout without affecting deeper layers.
    feedforward : FeedForwardConfig | None
        Config for the feed-forward sub-layer.  Currently required (None
        is rejected at construction time).
    norm : 'none' | 'layernorm' | 'rmsnorm'
        Pre-norm applied before each of the three sub-layers (conv, FFN,
        neuron).
    """

    neuron:           MinMaxNeuronConfig
    conv:             Union[BasicConvConfig, GatedConvConfig]
    d_model:          int
    first_in_dropout: float                       = 0.0
    feedforward:      Optional[FeedForwardConfig] = None
    norm:             NormType                    = 'layernorm'


class MinMaxLayer(nn.Module):
    """
    One residual layer of the MinMax RNC backbone.

    Internal data flow (all operations use pre-norm and residual connections):

        conv_out  = Conv( norm(u) )          # short-range context
        ffn_out   = FFN( norm(u + conv_out) )
        neur_out  = Neuron( norm(u + ffn_out) )
        output    = u + neur_out
    """

    def __init__(self, cfg: MinMaxLayerConfig, first: bool):
        super().__init__()

        self.cfg = cfg

        if type(cfg.conv) == BasicConvConfig:
            self.conv = BasicConv(cfg.conv)
        else:
            self.conv = GatedConv(cfg.conv)

        self.neuron = MinMaxNeuron(cfg.neuron)

        self.use_ffn = (cfg.feedforward is not None)
        assert self.use_ffn

        ffn_dropout = cfg.feedforward.dropout
        if first:
            ffn_dropout = cfg.first_in_dropout

        self.ffn = create_feedforward(
            config=replace(
                cfg.feedforward,
                embedding_dim=cfg.d_model,
                embedding_dim_out=cfg.d_model,
                dropout=ffn_dropout
            )
        )

        if self.cfg.norm == 'layernorm':
            self.norm_ffn  = nn.LayerNorm(cfg.d_model)
            self.norm_neuron = nn.LayerNorm(cfg.d_model)
            self.norm_conv = nn.LayerNorm(cfg.d_model)
        elif self.cfg.norm == 'rmsnorm':
            self.norm_ffn  = nn.RMSNorm(cfg.d_model)
            self.norm_neuron = nn.RMSNorm(cfg.d_model)
            self.norm_conv = nn.RMSNorm(cfg.d_model)
        else:
            self.norm_conv = nn.Identity()
            self.norm_ffn  = nn.Identity()
            self.norm_neuron = nn.Identity()


    @property
    def initial_state(self):
        return {
            'neuron': self.neuron.initial_state,
            'conv':   self.conv.initial_state
        }


    def forward(self, u: torch.Tensor, state: dict):

        conv_in = self.norm_conv(u)
        conv, conv_state = self.conv(conv_in, state['conv'])

        ffn_in = self.norm_ffn(u + conv)
        ffn = self.ffn(ffn_in)

        neuron_in = self.norm_neuron(u + ffn)
        neuron, neuron_state = self.neuron(neuron_in, state['neuron'])

        output = u + neuron

        state = {'conv': conv_state, 'neuron': neuron_state}

        return output, state


