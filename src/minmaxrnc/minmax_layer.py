# SPDX-FileCopyrightText: 2026 Alessandro Ronca
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

from typing import Sequence, Optional, List, Tuple, Union, Literal
from dataclasses import dataclass, replace

import torch
import torch.nn as nn

from .minmax_neuron import (
    MinMaxNeuron, MinMaxNeuronConfig,
    MinMaxNeuronArray, MinMaxNeuronArrayConfig,
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
        MinMaxRNCConfig.scalar_neurons of them), or None for none (new in 0.4.0;
        the layer must then have neuron_arrays).
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
    neuron_arrays : tuple[MinMaxNeuronArrayConfig, ...]
        New in 0.4.0, default none. Further arrays of MinMax neurons, of any
        state dimension. Each becomes one MinMaxNeuronArray; all neuron
        outputs of the layer are summed before the residual add.
    """

    neuron:           MinMaxNeuronConfig
    conv:             Union[BasicConvConfig, GatedConvConfig]
    d_model:          int
    first_in_dropout: float                       = 0.0
    feedforward:      Optional[FeedForwardConfig] = None
    norm:             NormType                    = 'layernorm'
    # --- new in 0.4.0 ---
    neuron_arrays:    Tuple[MinMaxNeuronArrayConfig, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, 'neuron_arrays', tuple(self.neuron_arrays))
        if self.neuron is None and len(self.neuron_arrays) == 0:
            raise ValueError(
                "MinMaxLayerConfig requires at least one neuron: `neuron` "
                "and/or a non-empty `neuron_arrays`."
            )


class MinMaxLayer(nn.Module):
    """
    One residual layer of the MinMax RNC backbone.

    Internal data flow (all operations use pre-norm and residual connections):

        conv_out  = Conv( norm(u) )                      # short-range context
        ffn_out   = FFN( norm(u + conv_out) )
        neur_out  = Neuron( norm(u + conv_out + ffn_out) )
        output    = u + neur_out

    Neuron is the sum of the outputs of all the layer's MinMax neuron arrays,
    all fed the same input: `neuron` (the MinMaxRNCConfig.scalar_neurons shorthand
    for an array of scalar-state neurons, if present) and each of
    `neuron_arrays`. All arrays are MinMaxNeuronArray instances and are
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

        # Construction order (conv → neuron → neuron_arrays → ffn → norms) and
        # attribute names are chosen so that a layer with only `neuron`
        # consumes the same RNG sequence and has the same state_dict keys as
        # in 0.3.0 (an empty ModuleList contributes no keys).
        self.neuron = MinMaxNeuron(cfg.neuron) if cfg.neuron is not None else None
        self.neuron_arrays = nn.ModuleList(
            MinMaxNeuronArray(array_cfg) for array_cfg in cfg.neuron_arrays
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
    # Uniform view of the layer's neuron arrays
    # ------------------------------------------------------------------

    def neuron_array_list(self):
        """All the layer's neuron arrays: `neuron` (if any), then `neuron_arrays`."""
        arrays = [] if self.neuron is None else [self.neuron]
        arrays.extend(self.neuron_arrays)
        return arrays

    def _states_to_list(self, state: dict):
        """Recurrent-state dict → per-array states, aligned with neuron_array_list()."""
        states = [] if self.neuron is None else [state['neuron']]
        if len(self.neuron_arrays) > 0:
            states.extend(state['neuron_arrays'])
        return states

    def _states_to_dict(self, states, conv_state, conv_first: bool) -> dict:
        """Per-array states → recurrent-state dict, in the 0.3.0 format: 'neuron'
        only with `neuron`, 'neuron_arrays' only with neuron_arrays. Key order
        as in 0.3.0: 'neuron', 'conv' for initial_state; 'conv', 'neuron' for
        the state returned by forward ('neuron_arrays' last)."""
        state = {'conv': conv_state} if conv_first else {}
        i = 0
        if self.neuron is not None:
            state['neuron'] = states[0]
            i = 1
        if not conv_first:
            state['conv'] = conv_state
        if len(self.neuron_arrays) > 0:
            state['neuron_arrays'] = list(states[i:])
        return state

    @property
    def initial_state(self):
        return self._states_to_dict(
            [arr.initial_state for arr in self.neuron_array_list()],
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
        for array, array_state in zip(self.neuron_array_list(), self._states_to_list(state)):
            out, array_state = array(neuron_in, array_state)
            new_states.append(array_state)
            neuron = out if neuron is None else neuron + out

        output = u + neuron

        state = self._states_to_dict(new_states, conv_state, conv_first=True)

        return output, state


