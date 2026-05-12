# SPDX-FileCopyrightText: 2026 Alessandro Ronca
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""
Parameter initialisers.

small_init_init_
    Nguyen & Salazar (2019). "Transformers without Tears: Improving the
    Normalization of Self-Attention." IWSLT 2019.
    https://arxiv.org/abs/1910.05895

wang_init_
    The 1/N rescaling of residual output projections was introduced in GPT-2:
        Radford et al. (2019). "Language Models are Unsupervised Multitask
        Learners." OpenAI Blog.
        https://openai.com/research/language-unsupervised
    The combined formula (1/N · 1/√d) is used in the xLSTM family:
        Beck et al. (2024). "xLSTM: Extended Long Short-Term Memory."
        https://arxiv.org/abs/2405.04517
"""

import math
import torch
import torch.nn as nn



def small_init_init_(param: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Normal initialisation with std = sqrt(2 / (5 · dim)).

    Keeps the output variance of a linear layer near 1 when inputs are drawn
    from a standard normal, while being slightly smaller than the Kaiming /
    He initialisation to improve early training stability in deep networks.

    Nguyen & Salazar (2019), IWSLT.
    """
    std = math.sqrt(2.0 / (5.0 * dim))
    nn.init.normal_(param, mean=0.0, std=std)
    return param


def wang_init_(param: torch.Tensor, dim: int, num_blocks: int) -> torch.Tensor:
    """
    Normal initialisation with std = 2 / (num_blocks · sqrt(dim)).

    Designed for the *output* projection of a residual branch.  If every
    residual block is initialised this way, the total variance added to the
    residual stream at initialisation is

        num_blocks · std² · dim  =  num_blocks · (2 / (num_blocks · √dim))² · dim
                                  =  4 / num_blocks  →  0  as depth grows,

    so the network starts close to the identity map regardless of depth.
    The 1/num_blocks factor is due to Radford et al. (2019, GPT-2); the
    1/√dim factor matches the small-init scaling of Nguyen & Salazar (2019).
    """
    std = 2.0 / (num_blocks * math.sqrt(dim))
    nn.init.normal_(param, mean=0.0, std=std)
    return param
