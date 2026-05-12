# SPDX-FileCopyrightText: 2026 Alessandro Ronca
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

import torch


def apply(a: torch.Tensor, b: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Apply the MinMax scalar operator f(x) = max(min(a, x), b)
    where
    - min and max are applied element-wise,
    - shapes (broadcastable):
      a: (..., D)
      b: (..., D)
      x: (..., D)
    """
    return torch.maximum(torch.minimum(a, x), b)


def compose(a2: torch.Tensor, b2: torch.Tensor, a1: torch.Tensor, b1: torch.Tensor):
    """
    Compose MinMax scalar operators.

    Given
        a1,b1, a2,b2
    having shape (..., D) and representing the MinMax scalar operators
        f1(x) = max(min(a1, x), b1),
        f2(x) = max(min(a2, x), b2),
    return
        a = min(a2, a1)
        b = max(min(a2, b1), b2)
    corresponding to the MinMax scalar operator
        f(x) = f2(f1(x)) = max(min(a, x), b)
    """
    a = torch.minimum(a2, a1)
    b = torch.maximum(torch.minimum(a2, b1), b2)
    return a, b


