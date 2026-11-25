# SPDX-FileCopyrightText: 2026 Alessandro Ronca
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

__version__ = "0.4.1"

from .minmax_rnc     import MinMaxRNC, MinMaxRNCConfig, NeuronArray, AmpDtype
from .minmax_rnc_lm  import MinMaxRNC_LM, MinMaxRNCLMConfig
from .minmax_neuron  import (
    SRInitType,
    RecurrenceDtype,
    MinMaxNeuron,
    MinMaxNeuronConfig,
    MinMaxNeuronArray,
    MinMaxNeuronArrayConfig,
)

__all__ = [
    "MinMaxRNC",
    "MinMaxRNCConfig",
    "MinMaxRNC_LM",
    "MinMaxRNCLMConfig",
    "SRInitType",
    # new in 0.4.0
    "NeuronArray",
    "RecurrenceDtype",
    "AmpDtype",
    "MinMaxNeuron",
    "MinMaxNeuronConfig",
    "MinMaxNeuronArray",
    "MinMaxNeuronArrayConfig",
]
