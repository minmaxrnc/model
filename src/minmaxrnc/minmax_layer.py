# SPDX-FileCopyrightText: 2026 Alessandro Ronca
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

from typing import Sequence, Optional, List, Tuple, Union, Literal
from dataclasses import dataclass, replace

import torch
import torch.nn as nn

from .minmax_neuron import (
    MinMaxNeuron, MinMaxNeuronConfig,
    MinMaxNeuronGroup, MinMaxNeuronGroupConfig,
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
        Config for the layer's scalar-state neurons (the 0.3.0 MinMax Neuron:
        MinMaxRNCConfig.d_state of them), or None for none (new in 0.4.0;
        the layer must then have neuron_groups).
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
    neuron_groups : tuple[MinMaxNeuronGroupConfig, ...]
        New in 0.4.0, default none. Further groups of MinMax neurons, of any
        state dimension. Each becomes one MinMaxNeuronGroup; all neuron
        outputs of the layer are summed before the residual add.
    """

    neuron:           MinMaxNeuronConfig
    conv:             Union[BasicConvConfig, GatedConvConfig]
    d_model:          int
    first_in_dropout: float                       = 0.0
    feedforward:      Optional[FeedForwardConfig] = None
    norm:             NormType                    = 'layernorm'
    # --- new in 0.4.0 ---
    neuron_groups:    Tuple[MinMaxNeuronGroupConfig, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, 'neuron_groups', tuple(self.neuron_groups))
        if self.neuron is None and len(self.neuron_groups) == 0:
            raise ValueError(
                "MinMaxLayerConfig requires at least one neuron: `neuron` "
                "and/or a non-empty `neuron_groups`."
            )


class MinMaxLayer(nn.Module):
    """
    One residual layer of the MinMax RNC backbone.

    Internal data flow (all operations use pre-norm and residual connections):

        conv_out  = Conv( norm(u) )                      # short-range context
        ffn_out   = FFN( norm(u + conv_out) )
        neur_out  = Neuron( norm(u + conv_out + ffn_out) )
        output    = u + neur_out

    Neuron is the sum of the outputs of all the layer's MinMax neuron groups,
    all fed the same input: `neuron` (the MinMaxRNCConfig.d_state shorthand
    for a group of scalar-state neurons, if present) and each of
    `neuron_groups`. All groups are MinMaxNeuronGroup instances and are
    handled identically; `neuron` is a separate attribute only so that
    parameter names and the recurrent-state format match 0.3.0.
    """

    def __init__(self, cfg: MinMaxLayerConfig, first: bool):
        super().__init__()

        self.cfg = cfg

        if type(cfg.conv) == BasicConvConfig:
            self.conv = BasicConv(cfg.conv)
        else:
            self.conv = GatedConv(cfg.conv)

        # Construction order (conv → neuron → neuron_groups → ffn → norms) and
        # attribute names are chosen so that a layer with only `neuron`
        # consumes the same RNG sequence and has the same state_dict keys as
        # in 0.3.0 (an empty ModuleList contributes no keys).
        self.neuron = MinMaxNeuron(cfg.neuron) if cfg.neuron is not None else None
        self.neuron_groups = nn.ModuleList(
            MinMaxNeuronGroup(g_cfg) for g_cfg in cfg.neuron_groups
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


    # ------------------------------------------------------------------
    # Uniform view of the layer's neuron groups
    # ------------------------------------------------------------------

    def neuron_group_list(self):
        """All the layer's neuron groups: `neuron` (if any), then `neuron_groups`."""
        groups = [] if self.neuron is None else [self.neuron]
        groups.extend(self.neuron_groups)
        return groups

    def _states_to_list(self, state: dict):
        """Recurrent-state dict → per-group states, aligned with neuron_group_list()."""
        states = [] if self.neuron is None else [state['neuron']]
        if len(self.neuron_groups) > 0:
            states.extend(state['neuron_groups'])
        return states

    def _states_to_dict(self, states, conv_state, conv_first: bool) -> dict:
        """Per-group states → recurrent-state dict, in the 0.3.0 format: 'neuron'
        only with `neuron`, 'neuron_groups' only with neuron_groups. Key order
        as in 0.3.0: 'neuron', 'conv' for initial_state; 'conv', 'neuron' for
        the state returned by forward ('neuron_groups' last)."""
        state = {'conv': conv_state} if conv_first else {}
        i = 0
        if self.neuron is not None:
            state['neuron'] = states[0]
            i = 1
        if not conv_first:
            state['conv'] = conv_state
        if len(self.neuron_groups) > 0:
            state['neuron_groups'] = list(states[i:])
        return state

    @property
    def initial_state(self):
        return self._states_to_dict(
            [g.initial_state for g in self.neuron_group_list()],
            self.conv.initial_state,
            conv_first=False,
        )


    def forward(self, u: torch.Tensor, state: dict):

        conv_in = self.norm_conv(u)
        conv, conv_state = self.conv(conv_in, state['conv'])

        ffn_in = self.norm_ffn(u + conv)
        ffn = self.ffn(ffn_in)

        neuron_in = self.norm_neuron(u + conv + ffn)

        neuron = None
        new_states = []
        for group, group_state in zip(self.neuron_group_list(), self._states_to_list(state)):
            out, group_state = group(neuron_in, group_state)
            new_states.append(group_state)
            neuron = out if neuron is None else neuron + out

        output = u + neuron

        state = self._states_to_dict(new_states, conv_state, conv_first=True)

        return output, state


