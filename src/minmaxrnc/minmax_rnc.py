# SPDX-FileCopyrightText: 2026 Alessandro Ronca
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

import contextlib
import warnings

import torch
import torch.nn as nn

from typing      import Optional, Literal, Union, Tuple
from dataclasses import dataclass, replace

from .minmax_layer              import MinMaxLayer, MinMaxLayerConfig, NormType
from .minmax_neuron             import (
    MinMaxNeuronConfig, MatrixMinMaxNeuronConfig, SRInitType, ScanDtype, SCAN_DTYPES,
)
from .modules.feedforward       import FeedForwardConfig, FFType, InitType, create_feedforward
from .modules.basic_conv        import BasicConvConfig
from .modules.gated_conv        import GatedConvConfig


ConvType = Literal['basic', 'gated']

AmpDtype = Literal['bf16', 'fp16']
AMP_DTYPES = ('bf16', 'fp16')
_AMP_TORCH_DTYPES = {'bf16': torch.bfloat16, 'fp16': torch.float16}


@dataclass(frozen=True)
class MatrixNeuronGroup:
    """
    One group of matrix MinMax Neurons in every layer (new in 0.4.0).

    Entries of MinMaxRNCConfig.matrix_neurons. Each layer gets `num_neurons`
    independent MatrixMinMaxNeurons of state dimension `d_state`, batched into
    a single module and a single parallel scan. See MatrixMinMaxNeuronConfig
    for `n_h` (experimental rank-1-product parametrisation of R_t) and
    `normalize_keys`.

    MinMaxRNCConfig also accepts a plain tuple/list
    (d_state[, num_neurons[, n_h[, normalize_keys]]]) in place of a
    MatrixNeuronGroup, e.g. ``matrix_neurons=[(3, 16)]``.
    """

    d_state:        int
    num_neurons:    int  = 1
    n_h:            int  = 0
    normalize_keys: bool = False

    def __post_init__(self):
        if self.d_state <= 0:
            raise ValueError(f"MatrixNeuronGroup.d_state must be > 0, got {self.d_state}")
        if self.num_neurons <= 0:
            raise ValueError(f"MatrixNeuronGroup.num_neurons must be > 0, got {self.num_neurons}")
        if self.n_h < 0:
            raise ValueError(f"MatrixNeuronGroup.n_h must be >= 0, got {self.n_h}")

    @classmethod
    def coerce(cls, value) -> 'MatrixNeuronGroup':
        if isinstance(value, cls):
            return value
        if isinstance(value, (tuple, list)) and 1 <= len(value) <= 4:
            return cls(*value)
        raise TypeError(
            "Each matrix_neurons entry must be a MatrixNeuronGroup or a tuple/list "
            f"(d_state[, num_neurons[, n_h[, normalize_keys]]]), got {value!r}"
        )


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
        Hidden-state dimension of each layer's (diagonal) MinMax Neuron.
        Independent of d_model; larger values increase memory capacity at
        linear parameter cost. 0 means no diagonal neuron (new in 0.4.0; the
        layers must then have matrix neurons).
    matrix_neurons : tuple[MatrixNeuronGroup | tuple, ...]
        New in 0.4.0, default () — none. Groups of full-matrix MinMax Neurons
        added to every layer alongside (or instead of) the diagonal neuron;
        each entry is a MatrixNeuronGroup or a tuple
        (d_state[, num_neurons[, n_h[, normalize_keys]]]). Their outputs are
        summed with the diagonal neuron's. Work scales as d_state³ per scan
        step, so keep each group's d_state small. A group with d_state=1 is a
        scalar-state neuron, equivalent to a diagonal one, and triggers a
        UserWarning.

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
    matrix_neurons:  Tuple[MatrixNeuronGroup, ...] = ()
    scan_dtype:      ScanDtype          = 'ambient'
    amp_dtype:       Optional[AmpDtype] = None

    def __post_init__(self):
        groups = tuple(MatrixNeuronGroup.coerce(g) for g in self.matrix_neurons)
        object.__setattr__(self, 'matrix_neurons', groups)
        if self.d_state < 0:
            raise ValueError(f"d_state must be >= 0, got {self.d_state}")
        if self.d_state == 0 and len(groups) == 0:
            raise ValueError(
                "MinMaxRNCConfig requires at least one neuron per layer: set "
                "d_state > 0 (diagonal neuron) and/or a non-empty matrix_neurons."
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
        for g in groups:
            if g.d_state == 1:
                warnings.warn(
                    f"matrix_neurons group {g} has d_state=1: a 1×1 matrix MinMax "
                    f"neuron is a scalar-state neuron, equivalent to a diagonal "
                    f"one. Consider adding d_state={g.num_neurons} to the diagonal "
                    f"neuron instead (fewer parameters, cheaper scan).",
                    UserWarning,
                    stacklevel=3,
                )

    @property
    def num_neurons_per_layer(self) -> int:
        """Neurons summed at each layer: 1 per diagonal neuron + every group's num_neurons."""
        return (1 if self.d_state > 0 else 0) + sum(g.num_neurons for g in self.matrix_neurons)

    @property
    def layer_cfg(self) -> MinMaxLayerConfig:
        n_in_layer = self.num_neurons_per_layer
        neuron_cfg = None
        if self.d_state > 0:
            neuron_cfg = MinMaxNeuronConfig(
                _num_blocks           = self.n_layers,
                d_model               = self.d_model,
                d_state               = self.d_state,
                dropout               = self.neuron_dropout,
                train_init            = self.train_init,
                output_gate           = self.output_gate,
                s_r_init              = self.s_r_init,
                scan_dtype            = self.scan_dtype,
                _num_neurons_in_layer = n_in_layer,
            )
        matrix_cfgs = tuple(
            MatrixMinMaxNeuronConfig(
                _num_blocks           = self.n_layers,
                d_model               = self.d_model,
                d_state               = g.d_state,
                num_neurons           = g.num_neurons,
                n_h                   = g.n_h,
                normalize_keys        = g.normalize_keys,
                dropout               = self.neuron_dropout,
                train_init            = self.train_init,
                output_gate           = self.output_gate,
                s_r_init              = self.s_r_init,
                scan_dtype            = self.scan_dtype,
                _num_neurons_in_layer = n_in_layer,
            )
            for g in self.matrix_neurons
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
            matrix_neurons   = matrix_cfgs,
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
    convolution, a feed-forward network, and MinMax Neurons (a diagonal one
    and/or matrix-neuron groups).  All three sub-layers use pre-norm and
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
