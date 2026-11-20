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


def apply_matrix(R: torch.Tensor, s: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Apply the MinMax matrix operator f(x) = max(R ⊗ x, s)
    where
    - (R ⊗ x)_i = max_j min(R_ij, x_j)   (the (max,min)-semiring matrix-vector product)
    - max is applied element-wise for the outer max with s
    - shapes (broadcastable):
      R: (..., D, D)
      s: (..., D)
      x: (..., D)
    """
    Rx = torch.minimum(R, x.unsqueeze(-2))   # (..., D, D): entry [i,j] = min(R_ij, x_j)
    Rx = Rx.amax(dim=-1)                     # (..., D):    (R ⊗ x)_i
    return torch.maximum(Rx, s)


def compose_matrix(
    R2: torch.Tensor, s2: torch.Tensor,
    R1: torch.Tensor, s1: torch.Tensor,
):
    """
    Compose MinMax matrix operators.

    Given
        R1,s1, R2,s2
    having shapes (..., D, D) and (..., D) respectively, representing the
    MinMax matrix operators
        f1(x) = max(R1 ⊗ x, s1),
        f2(x) = max(R2 ⊗ x, s2),
    return
        R = R2 ⊗ R1,  where (R2 ⊗ R1)_ik = max_j min(R2_ij, R1_jk)
        s = max(R2 ⊗ s1, s2)
    corresponding to the MinMax matrix operator
        f(x) = f2(f1(x)) = max(R ⊗ x, s)

    The (max,min)-semiring matrix product ⊗ is associative and distributes
    over element-wise max, which is what makes this composition closed and
    what allows the parallel prefix scan (see minmax_scan.py) to work exactly
    as it does for the scalar/diagonal operator.
    """
    # R[..., i, k] = max_j min(R2[..., i, j], R1[..., j, k])
    R = torch.minimum(R2.unsqueeze(-1), R1.unsqueeze(-3)).amax(dim=-2)
    s = apply_matrix(R2, s2, s1)
    return R, s


