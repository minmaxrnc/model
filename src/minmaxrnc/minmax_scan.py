# SPDX-FileCopyrightText: 2026 Alessandro Ronca
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

import torch

from . import minmax_operator


def all_composition_prefixes(a: torch.Tensor, b: torch.Tensor):
    """
    Inclusive parallel prefix scan of MinMax recurrence (Hillis-Steele).

    Depth O(log T), work O(T log T).

    Inputs
    ------
    a, b : (B, T, D)
        Parameters of the per-step maps f_t(x) = max(min(a_t, x), b_t).

    Outputs
    -------
    a_pref, b_pref : (B, T, D)
        Parameters of the prefix-composed maps, so that
        ``minmax_operator.apply(a_pref[:, t], b_pref[:, t], x0)``
        equals the state after applying steps 0 through t from x0.
    """
    T = a.shape[1]
    stride = 1
    while stride < T:
        a2, b2 = a[:, stride:], b[:, stride:]
        a1, b1 = a[:, :-stride], b[:, :-stride]
        a_new, b_new = minmax_operator.compose(a2, b2, a1, b1)
        a = torch.cat([a[:, :stride], a_new], dim=1)
        b = torch.cat([b[:, :stride], b_new], dim=1)
        stride *= 2
    return a, b


def all_states(
    a: torch.Tensor,
    b: torch.Tensor,
    x0: torch.Tensor,
) -> torch.Tensor:
    """
    Compute all hidden states x_0, …, x_T in parallel.

    Recurrence: x_{t+1} = max(min(a_t, x_t), b_t)  (elementwise)

    Inputs
    ------
    a, b : (B, T, D) — per-step MinMax parameters
    x0   : (B, D)    — initial hidden state

    Output
    ------
    x : (B, T+1, D) — x[:, 0] == x0; x[:, t+1] is the state after step t
    """
    a_pref, b_pref = all_composition_prefixes(a, b)
    x1T = minmax_operator.apply(a_pref, b_pref, x0[:, None, :])
    return torch.cat([x0[:, None, :], x1T], dim=1)
