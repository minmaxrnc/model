# SPDX-FileCopyrightText: 2026 Alessandro Ronca
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

import contextlib

import torch
import torch.nn as nn

from typing      import Optional, Literal, Union, Tuple
from dataclasses import dataclass, replace

from .minmax_layer              import MinMaxLayer, MinMaxLayerConfig, NormType
from .minmax_neuron             import (
    MinMaxNeuronConfig, MinMaxNeuronGroupConfig, SRInitType, ScanDtype, SCAN_DTYPES,
)
from .modules.feedforward       import FeedForwardConfig, FFType, InitType, create_feedforward
from .modules.basic_conv        import BasicConvConfig
from .modules.gated_conv        import GatedConvConfig


ConvType = Literal['basic', 'gated']

AmpDtype = Literal['bf16', 'fp16']
AMP_DTYPES = ('bf16', 'fp16')
_AMP_TORCH_DTYPES = {'bf16': torch.bfloat16, 'fp16': torch.float16}


@dataclass(frozen=True)
class NeuronGroup:
    """
    One group of MinMax neurons in every layer (new in 0.4.0): `num_neurons`
    neurons, each with state dimension `d_state`.

    Entries of MinMaxRNCConfig.neurons. Each layer gets one MinMaxNeuronGroup
    per entry. d_state = 1 gives scalar-state neurons (element-wise
    recurrence); d_state > 1 gives neurons whose state coordinates interact
    through a d_state × d_state matrix R_t. See MinMaxNeuronGroupConfig for
    `n_h` (experimental rank-1-product parametrisation of R_t) and
    `normalize_keys`.

    MinMaxRNCConfig also accepts a plain tuple/list
    (d_state[, num_neurons[, n_h[, normalize_keys]]]) in place of a
    NeuronGroup, e.g. ``neurons=[(3, 16)]``.
    """

    d_state:        int
    num_neurons:    int  = 1
    n_h:            int  = 0
    normalize_keys: bool = False

    def __post_init__(self):
        if self.d_state <= 0:
            raise ValueError(f"NeuronGroup.d_state must be > 0, got {self.d_state}")
        if self.num_neurons <= 0:
            raise ValueError(f"NeuronGroup.num_neurons must be > 0, got {self.num_neurons}")
        if self.n_h < 0:
            raise ValueError(f"NeuronGroup.n_h must be >= 0, got {self.n_h}")

    @classmethod
    def coerce(cls, value) -> 'NeuronGroup':
        if isinstance(value, cls):
            return value
        if isinstance(value, (tuple, list)) and 1 <= len(value) <= 4:
            return cls(*value)
        raise TypeError(
            "Each neurons entry must be a NeuronGroup or a tuple/list "
            f"(d_state[, num_neurons[, n_h[, normalize_keys]]]), got {value!r}"
        )

    @property
    def state_size(self) -> int:
        """Number of state coordinates: num_neurons · d_state."""
        return self.num_neurons * self.d_state


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
        Number of scalar-state MinMax neurons in each layer (equivalently,
        their total state dimension) — shorthand for a neuron group
        NeuronGroup(d_state=1, num_neurons=d_state), and the 0.3.0 way of
        configuring a layer's neurons. Independent of d_model; larger values
        increase memory capacity at linear parameter cost. 0 means none (new
        in 0.4.0; `neurons` must then be non-empty).
    neurons : tuple[NeuronGroup | tuple, ...]
        New in 0.4.0, default () — none. Further groups of MinMax neurons in
        every layer, each entry a NeuronGroup or a tuple
        (d_state[, num_neurons[, n_h[, normalize_keys]]]): num_neurons
        neurons of state dimension d_state each. All neuron outputs of a
        layer are summed. Scalar-state groups (d_state=1) cost as much as
        the `d_state` shorthand; for d_state > 1 the scan's work scales as
        d_state³ per step, so keep it small and scale with num_neurons.

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
    ffn_init : 'basic' | 'scaled'
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
    scan_dtype : 'ambient' | 'fp32' | 'bf16' | 'fp16'
        New in 0.4.0. Precision of every neuron's MinMax scan. 'ambient'
        (default) — no casting, exactly as in 0.3.0. 'fp32' / 'bf16' / 'fp16'
        — pinned: the scan always runs in that dtype with autocast disabled,
        whatever the ambient precision. See minmax_neuron.ScanDtype.

    Mixed precision
    ---------------
    amp_dtype : None | 'bf16' | 'fp16'
        New in 0.4.0. If set, forward() runs under
        torch.autocast(device, dtype=amp_dtype) — model-level automatic
        mixed precision. None (default) — the model does not touch autocast,
        exactly as in 0.3.0, so an autocast context opened by the caller
        still applies. When set, it overrides the dtype of any outer autocast
        inside the model. fp16 training additionally needs the caller's
        torch.amp.GradScaler. Combine with scan_dtype to choose the scan's
        precision independently, e.g. amp_dtype='bf16', scan_dtype='fp32'.

    Convolution
    -----------
    conv_type : 'basic' | 'gated'
        'basic' (default) — learned linear mixing of the previous and current
        token representations.  'gated' — learned scalar gate interpolating
        between them.
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
    """

    # Core architecture
    d_model:  int
    n_layers: int
    d_state:  int = 0

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
    output_gate:     bool       = True
    train_init:      bool       = False
    neuron_dropout:  float      = 0.0
    s_r_init:        SRInitType = 'small_init'

    # Conv
    conv_type:       ConvType = 'basic'
    conv_init_val:   float    = 0.0

    # Per-layer options
    prelayers_dropout:  float = 0.0

    # Post-layers
    use_postlayers_ffn: bool = False

    # --- new in 0.4.0 (appended, so positional construction is unchanged) ---
    neurons:         Tuple[NeuronGroup, ...] = ()
    scan_dtype:      ScanDtype          = 'ambient'
    amp_dtype:       Optional[AmpDtype] = None

    def __post_init__(self):
        groups = tuple(NeuronGroup.coerce(g) for g in self.neurons)
        object.__setattr__(self, 'neurons', groups)
        if self.d_state < 0:
            raise ValueError(f"d_state must be >= 0, got {self.d_state}")
        if self.d_state == 0 and len(groups) == 0:
            raise ValueError(
                "MinMaxRNCConfig requires at least one neuron per layer: set "
                "d_state > 0 and/or a non-empty `neurons`."
            )
        if self.scan_dtype not in SCAN_DTYPES:
            raise ValueError(
                f"Unknown scan_dtype {self.scan_dtype!r}. Valid values: "
                f"{', '.join(map(repr, SCAN_DTYPES))}."
            )
        if self.amp_dtype is not None and self.amp_dtype not in AMP_DTYPES:
            raise ValueError(
                f"Unknown amp_dtype {self.amp_dtype!r}. Valid values: None, "
                f"{', '.join(map(repr, AMP_DTYPES))}."
            )

    @property
    def neuron_groups(self) -> Tuple[NeuronGroup, ...]:
        """All neuron groups of each layer: the `d_state` shorthand (if > 0) first, then `neurons`."""
        shorthand = (NeuronGroup(d_state=1, num_neurons=self.d_state),) if self.d_state > 0 else ()
        return shorthand + self.neurons

    @property
    def num_neurons_per_layer(self) -> int:
        """Number of MinMax neurons in each layer."""
        return sum(g.num_neurons for g in self.neuron_groups)

    @property
    def state_size_per_layer(self) -> int:
        """Number of state coordinates in each layer: Σ num_neurons · d_state."""
        return sum(g.state_size for g in self.neuron_groups)

    @property
    def layer_cfg(self) -> MinMaxLayerConfig:
        # Every group's output projection is scaled by the layer's total state
        # size, so the layer's summed contribution stays O(1) however the
        # state is split (for a d_state-only layer: exactly as in 0.3.0).
        state_size = self.state_size_per_layer
        neuron_cfg = None
        if self.d_state > 0:
            neuron_cfg = MinMaxNeuronConfig(
                _num_blocks          = self.n_layers,
                d_model              = self.d_model,
                d_state              = self.d_state,
                dropout              = self.neuron_dropout,
                train_init           = self.train_init,
                output_gate          = self.output_gate,
                s_r_init             = self.s_r_init,
                scan_dtype           = self.scan_dtype,
                _state_size_in_layer = state_size,
            )
        group_cfgs = tuple(
            MinMaxNeuronGroupConfig(
                _num_blocks          = self.n_layers,
                d_model              = self.d_model,
                d_state              = g.d_state,
                num_neurons          = g.num_neurons,
                n_h                  = g.n_h,
                normalize_keys       = g.normalize_keys,
                dropout              = self.neuron_dropout,
                train_init           = self.train_init,
                output_gate          = self.output_gate,
                s_r_init             = self.s_r_init,
                scan_dtype           = self.scan_dtype,
                _state_size_in_layer = state_size,
            )
            for g in self.neurons
        )
        if self.conv_type == 'basic':
            conv_cfg = BasicConvConfig(embedding_dim=self.d_model)
        elif self.conv_type == 'gated':
            conv_cfg = GatedConvConfig(
                embedding_dim = self.d_model,
                init_val      = self.conv_init_val,
            )
        else:
            raise ValueError(f"Unknown conv_type '{self.conv_type}'. Valid values: 'basic', 'gated'.")
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
            neuron_groups    = group_cfgs,
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
    convolution, a feed-forward network, and groups of MinMax neurons.  All three sub-layers use pre-norm and
    residual connections.

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

    def _amp_ctx(self, device_type: str):
        """Model-level autocast per cfg.amp_dtype; a no-op when it is None."""
        amp_dtype = self.__cfg.amp_dtype
        if amp_dtype is None:
            return contextlib.nullcontext()
        return torch.autocast(device_type=device_type, dtype=_AMP_TORCH_DTYPES[amp_dtype])

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
        with self._amp_ctx(u.device.type):
            for u_chunk in u.split(unroll_steps, dim=1):
                y_chunk, state = self._parallel_forward(u_chunk, state)
                y_chunks.append(y_chunk)
            y = torch.cat(y_chunks, dim=1)

        if return_state:
            return y, state
        return y
