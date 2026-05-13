# SPDX-FileCopyrightText: 2026 Alessandro Ronca
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

import torch
import torch.nn as nn

from typing      import Optional, Literal, Union
from dataclasses import dataclass, replace

from .minmax_layer              import MinMaxLayer, MinMaxLayerConfig, NormType
from .minmax_neuron             import MinMaxNeuronConfig
from .modules.feedforward       import FeedForwardConfig, FFType, InitType, create_feedforward
from .modules.basic_conv        import BasicConvConfig
from .modules.gated_conv        import GatedConvConfig


ConvType = Literal['basic', 'gated']


@dataclass(frozen=True)
class MinMaxRNCConfig:
    """
    Configuration for the MinMax RNC backbone.

    Configs for neuron, conv, FFN, and layer are derived.

    Core architecture
    -----------------
    d_model : int
        Residual-stream width.  Every sub-module input and output has this
        dimension.
    n_layers : int
        Number of stacked MinMaxLayers.
    d_state : int
        Hidden-state dimension of each MinMax Neuron.  Independent of d_model;
        larger values increase memory capacity at linear parameter cost.

    Normalisation
    -------------
    norm : 'layernorm' | 'rmsnorm' | 'none'
        Pre-norm type applied before each sub-layer inside each layer.
        'layernorm' (default) is stable; 'rmsnorm' is slightly faster;
        'none' disables normalisation entirely.
    postlayers_norm : 'layernorm' | 'rmsnorm' | 'none'
        Norm applied to the output of the final layer (before the optional
        post-layers FFN).

    Feed-forward network (within each layer)
    -----------------------------------------
    ffn_type : 'gated' | 'basic'
        'gated' (default) — gated FFN (ReGLU / SwiGLU depending on
        act_fn), from Shazeer (2020).  'basic' — standard two-layer MLP.
    ffn_proj_factor : float
        Hidden-layer expansion factor relative to d_model.  The hidden
        dimension is rounded to the nearest multiple of 2.
    ffn_act_fn : str
        Activation function name.  Choices: 'relu', 'relu^2', 'gelu',
        'swish', 'sigmoid', 'selu'.
    ffn_dropout : float
        Dropout applied inside the FFN of every layer except possibly the first
        (see prelayers_dropout).
    ffn_init : 'default' | 'scaled'
        Weight initialisation scheme.  'scaled' uses small_init for the
        up-projection and wang_init for the down-projection.

    Neuron
    ------
    output_gate : bool
        If True, the neuron output is element-wise gated by a learned
        projection of the input.
    train_init : bool
        If True, the neuron's initial hidden state x_0 is a learned parameter.
    neuron_dropout : float
        Dropout probability applied to the neuron input.

    Convolution
    -----------
    conv_type : 'gated' | 'basic'
        'gated' (default) — learned scalar gate interpolating between
        the previous and current token.  'basic' — learned linear mixing of
        the previous and current token representations.
    conv_init_val : float
        Initial value of the gate logit in GatedConv.  0.0 → gate ≈ 0.5
        (equal mix); negative values bias toward the current token.

    Pre/Post-layers
    -----------
    prelayers_dropout : float
        FFN dropout for the first layer only; overrides ffn_dropout.  Useful
        as an input-level regulariser without penalising deeper layers.
    use_postlayers_ffn : bool
        If True, an extra FFN (with the same type and factor as the in-layer
        FFN) is applied after all layers, before postlayers_norm.

    Forward
    -------
    unroll_steps : int
        Sequence chunk size for the forward pass.  The sequence is split into
        chunks of this length and processed sequentially (carrying the state
        across chunks).  unroll_steps=1 processes one token at a time;
        unroll_steps=T processes the whole sequence at once.  Both give
        identical outputs; larger values use more peak memory.

    """

    # Core architecture
    d_model:  int
    n_layers: int
    d_state:  int

    # Normalisation (within layers and post-layers)
    norm:            NormType = 'layernorm'
    postlayers_norm: NormType = 'layernorm'

    # FFN within each layer
    ffn_type:        FFType   = 'gated'
    ffn_proj_factor: float    = 1.3
    ffn_act_fn:      str      = 'relu'
    ffn_dropout:     float    = 0.1
    ffn_init:        InitType = 'scaled'

    # Neuron
    output_gate:     bool  = True
    train_init:      bool  = False
    neuron_dropout:  float = 0.0

    # Conv
    conv_type:       ConvType = 'basic'
    conv_init_val:   float    = 0.0

    # Per-layer options
    prelayers_dropout:  float = 0.0

    # Post-layers
    use_postlayers_ffn: bool = False

    @property
    def layer_cfg(self) -> MinMaxLayerConfig:
        neuron_cfg = MinMaxNeuronConfig(
            _num_blocks = self.n_layers,
            d_model     = self.d_model,
            d_state     = self.d_state,
            dropout     = self.neuron_dropout,
            train_init  = self.train_init,
            output_gate = self.output_gate,
        )
        if self.conv_type == 'basic':
            conv_cfg = BasicConvConfig(embedding_dim=self.d_model)
        else:
            conv_cfg = GatedConvConfig(
                embedding_dim = self.d_model,
                init_val      = self.conv_init_val,
            )
        ffn_cfg = FeedForwardConfig(
            _num_blocks  = self.n_layers,
            ffn_type     = self.ffn_type,
            proj_factor  = self.ffn_proj_factor,
            act_fn       = self.ffn_act_fn,
            dropout      = self.ffn_dropout,
            init         = self.ffn_init,
        )
        return MinMaxLayerConfig(
            d_model          = self.d_model,
            neuron           = neuron_cfg,
            conv             = conv_cfg,
            feedforward      = ffn_cfg,
            norm             = self.norm,
            first_in_dropout = self.prelayers_dropout,
        )

    # ------------------------------------------------------------------
    # Preset factories
    # ------------------------------------------------------------------

    @classmethod
    def small(cls, n_layers: int = 2) -> 'MinMaxRNCConfig':
        return cls(d_model=90, n_layers=n_layers, d_state=40)

    @classmethod
    def medium(cls, n_layers: int = 8) -> 'MinMaxRNCConfig':
        return cls(d_model=512, n_layers=n_layers, d_state=512)

    @classmethod
    def large(cls, n_layers: int = 12) -> 'MinMaxRNCConfig':
        return cls(d_model=728, n_layers=n_layers, d_state=1456)


class MinMaxRNC(nn.Module):
    """
    MinMax Recurrent Neural Cascade — the backbone sequence model.

    Stacks ``cfg.n_layers`` MinMaxLayers, each containing a short-range
    convolution, a feed-forward network, and a MinMax Neuron.  All three
    sub-layers use pre-norm and residual connections.

    Inputs
    ------
    u : Tensor  (B, T, d_model)
        Continuous input sequence (e.g. token embeddings).
    state : list[dict] | None
        Per-layer recurrent state from a previous call.  Pass None (or omit)
        to start from the default initial state.
    return_state : bool
        If True, also return the updated state after the last token.

    Outputs
    -------
    y : Tensor  (B, T, d_model)
    state : list[dict]  — only when return_state=True
    """

    def __init__(self, cfg: MinMaxRNCConfig):
        super().__init__()
        self.__cfg = cfg
        self.reset()

    def reset(self):
        layer_cfg = self.__cfg.layer_cfg

        self.layers = nn.ModuleList()
        firstlayer = True
        for _ in range(self.__cfg.n_layers):
            self.layers.append(MinMaxLayer(layer_cfg, first=firstlayer))
            firstlayer = False

        self.postlayers_norm = None
        if self.__cfg.postlayers_norm == 'layernorm':
            self.postlayers_norm = nn.LayerNorm(self.__cfg.d_model)
        elif self.__cfg.postlayers_norm == 'rmsnorm':
            self.postlayers_norm = nn.RMSNorm(self.__cfg.d_model)

        self.postlayers_ffn      = None
        self.postlayers_ffn_norm = None
        if self.__cfg.use_postlayers_ffn:
            self.postlayers_ffn = create_feedforward(
                config=replace(
                    layer_cfg.feedforward,
                    embedding_dim     = self.__cfg.d_model,
                    embedding_dim_out = self.__cfg.d_model,
                )
            )
            if self.__cfg.norm == 'layernorm':
                self.postlayers_ffn_norm = nn.LayerNorm(self.__cfg.d_model)
            elif self.__cfg.norm == 'rmsnorm':
                self.postlayers_ffn_norm = nn.RMSNorm(self.__cfg.d_model)
            else:
                self.postlayers_ffn_norm = nn.Identity()

    @property
    def initial_state(self):
        return [layer.initial_state for layer in self.layers]

    def _parallel_forward(self, u: torch.Tensor, state):
        """u: [B, T, D] — returns output [B, T, D] and updated state."""
        updated_state = []
        y = u
        for layer, layer_state in zip(self.layers, state):
            y, updated_layer_state = layer(y, layer_state)
            updated_state.append(updated_layer_state)

        if self.postlayers_ffn is not None:
            y = y + self.postlayers_ffn(self.postlayers_ffn_norm(y))

        if self.postlayers_norm is not None:
            y = self.postlayers_norm(y)

        return y, updated_state

    def forward(self, u: torch.Tensor,  unroll_steps: int, state=None, return_state: bool = False):
        if state is None:
            state = self.initial_state

        y_chunks = []
        for u_chunk in u.split(unroll_steps, dim=1):
            y_chunk, state = self._parallel_forward(u_chunk, state)
            y_chunks.append(y_chunk)
        y = torch.cat(y_chunks, dim=1)

        if return_state:
            return y, state
        return y
