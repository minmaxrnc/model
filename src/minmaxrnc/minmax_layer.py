# SPDX-FileCopyrightText: 2026 Alessandro Ronca
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

from typing import Sequence, Optional, List, Tuple, Union, Literal
from dataclasses import dataclass, replace

import torch
import torch.nn as nn

from .minmax_neuron import (
    MinMaxNeuron, MinMaxNeuronConfig,
    MatrixMinMaxNeuron, MatrixMinMaxNeuronConfig,
)

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
    neuron : MinMaxNeuronConfig | None
        Config for the layer's diagonal MinMax Neuron, or None for a layer
        without one (new in 0.4.0; such a layer must have matrix neurons).
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
    matrix_neurons : tuple[MatrixMinMaxNeuronConfig, ...]
        Configs of the layer's matrix-neuron groups (new in 0.4.0; default
        none). Each becomes one MatrixMinMaxNeuron module; all neuron outputs
        are summed before the residual add.
    """

    neuron:           MinMaxNeuronConfig
    conv:             Union[BasicConvConfig, GatedConvConfig]
    d_model:          int
    first_in_dropout: float                       = 0.0
    feedforward:      Optional[FeedForwardConfig] = None
    norm:             NormType                    = 'layernorm'
    # --- new in 0.4.0 ---
    matrix_neurons:   Tuple[MatrixMinMaxNeuronConfig, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, 'matrix_neurons', tuple(self.matrix_neurons))
        if self.neuron is None and len(self.matrix_neurons) == 0:
            raise ValueError(
                "MinMaxLayerConfig requires at least one neuron: a diagonal "
                "`neuron` and/or a non-empty `matrix_neurons`."
            )


class MinMaxLayer(nn.Module):
    """
    One residual layer of the MinMax RNC backbone.

    Internal data flow (all operations use pre-norm and residual connections):

        conv_out  = Conv( norm(u) )                      # short-range context
        ffn_out   = FFN( norm(u + conv_out) )
        neur_out  = Neuron( norm(u + conv_out + ffn_out) )
        output    = u + neur_out

    Neuron is the sum of the layer's diagonal MinMaxNeuron (if any) and of
    every MatrixMinMaxNeuron group (if any), all fed the same input.
    """

    def __init__(self, cfg: MinMaxLayerConfig, first: bool):
        super().__init__()

        self.cfg = cfg

        if type(cfg.conv) == BasicConvConfig:
            self.conv = BasicConv(cfg.conv)
        else:
            self.conv = GatedConv(cfg.conv)

        # Construction order (conv → diagonal neuron → matrix neurons → ffn →
        # norms) and attribute names are chosen so that a diagonal-only layer
        # consumes the same RNG sequence and has the same state_dict keys as
        # in 0.3.0 (an empty ModuleList contributes no keys).
        self.neuron = MinMaxNeuron(cfg.neuron) if cfg.neuron is not None else None
        self.matrix_neurons = nn.ModuleList(
            MatrixMinMaxNeuron(m_cfg) for m_cfg in cfg.matrix_neurons
        )

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
        # Same structure as 0.3.0 for diagonal-only layers; 'neuron' is
        # omitted without a diagonal neuron, 'matrix_neurons' is present only
        # with matrix neurons.
        state = {}
        if self.neuron is not None:
            state['neuron'] = self.neuron.initial_state
        state['conv'] = self.conv.initial_state
        if len(self.matrix_neurons) > 0:
            state['matrix_neurons'] = [m.initial_state for m in self.matrix_neurons]
        return state


    def forward(self, u: torch.Tensor, state: dict):

        conv_in = self.norm_conv(u)
        conv, conv_state = self.conv(conv_in, state['conv'])

        ffn_in = self.norm_ffn(u + conv)
        ffn = self.ffn(ffn_in)

        neuron_in = self.norm_neuron(u + conv + ffn)
        new_state = {}
        neuron = None
        if self.neuron is not None:
            neuron, new_state['neuron'] = self.neuron(neuron_in, state['neuron'])
        if len(self.matrix_neurons) > 0:
            matrix_states = []
            for m, m_state in zip(self.matrix_neurons, state['matrix_neurons']):
                out_m, m_state = m(neuron_in, m_state)
                matrix_states.append(m_state)
                neuron = out_m if neuron is None else neuron + out_m
            new_state['matrix_neurons'] = matrix_states

        output = u + neuron

        state = {'conv': conv_state, **new_state}

        return output, state


