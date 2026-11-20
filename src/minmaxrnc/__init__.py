# SPDX-FileCopyrightText: 2026 Alessandro Ronca
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

__version__ = "0.4.0"

from .minmax_rnc     import MinMaxRNC, MinMaxRNCConfig, NeuronGroup, AmpDtype
from .minmax_rnc_lm  import MinMaxRNC_LM, MinMaxRNCLMConfig
from .minmax_neuron  import (
    SRInitType,
    ScanDtype,
    MinMaxNeuron,
    MinMaxNeuronConfig,
    MinMaxNeuronGroup,
    MinMaxNeuronGroupConfig,
)

__all__ = [
    "MinMaxRNC",
    "MinMaxRNCConfig",
    "MinMaxRNC_LM",
    "MinMaxRNCLMConfig",
    "SRInitType",
    # new in 0.4.0
    "NeuronGroup",
    "ScanDtype",
    "AmpDtype",
    "MinMaxNeuron",
    "MinMaxNeuronConfig",
    "MinMaxNeuronGroup",
    "MinMaxNeuronGroupConfig",
]
