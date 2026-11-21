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
    a, b : (B, T, N)  — N = number of scalar-state neurons
        Parameters of the per-step MinMax recurrence maps
        f_t(x) = max(min(a_t, x), b_t), with a = reset r and b = set s;
        array position t holds step t+1 (steps are numbered 1..T).

    Outputs
    -------
    a_pref, b_pref : (B, T, N)
        Parameters of the prefix-composed maps, so that
        ``minmax_operator.apply(a_pref[:, t], b_pref[:, t], x0)``
        equals x_{t+1}, the state after applying steps 1 through t+1 from x0.
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

    Recurrence: x_t = max(min(a_t, x_{t-1}), b_t) for t = 1..T
    (element-wise, over N scalar-state neurons), with a = reset r, b = set s.

    Inputs
    ------
    a, b : (B, T, N) — per-step MinMax parameters; a[:, t-1] is a_t, b[:, t-1] is b_t
    x0   : (B, N)    — initial hidden state

    Output
    ------
    x : (B, T+1, N) — x[:, t] is x_t, with x[:, 0] == x0
    """
    a_pref, b_pref = all_composition_prefixes(a, b)
    x1T = minmax_operator.apply(a_pref, b_pref, x0[:, None, :])
    return torch.cat([x0[:, None, :], x1T], dim=1)


def all_composition_prefixes_matrix(R: torch.Tensor, s: torch.Tensor):
    """
    Inclusive parallel prefix scan of the MinMax matrix recurrence (Hillis-Steele).

    Depth O(log T), work O(T log T · d³) (each combine is a (max,min)-semiring
    matrix product, see minmax_operator.compose_matrix).

    Inputs
    ------
    R : (B, T, d, d)  — d = state dimension
    s : (B, T, d)
        Parameters of the per-step MinMax recurrence maps
        f_t(x) = max(R_t ⊗ x, s_t), with R = reset and s = set;
        array position t holds step t+1 (steps are numbered 1..T).

    Outputs
    -------
    R_pref, s_pref : (B, T, d, d), (B, T, d)
        Parameters of the prefix-composed maps, so that
        ``minmax_operator.apply_matrix(R_pref[:, t], s_pref[:, t], x0)``
        equals x_{t+1}, the state after applying steps 1 through t+1 from x0.
    """
    T = R.shape[1]
    stride = 1
    while stride < T:
        R2, s2 = R[:, stride:], s[:, stride:]
        R1, s1 = R[:, :-stride], s[:, :-stride]
        R_new, s_new = minmax_operator.compose_matrix(R2, s2, R1, s1)
        R = torch.cat([R[:, :stride], R_new], dim=1)
        s = torch.cat([s[:, :stride], s_new], dim=1)
        stride *= 2
    return R, s


def all_states_matrix(
    R: torch.Tensor,
    s: torch.Tensor,
    x0: torch.Tensor,
) -> torch.Tensor:
    """
    Compute all hidden states x_0, …, x_T in parallel for the matrix recurrence.

    Recurrence: x_t = max(R_t ⊗ x_{t-1}, s_t) for t = 1..T, with state dimension d

    Inputs
    ------
    R : (B, T, d, d), s : (B, T, d) — per-step MinMax matrix parameters;
                                      R[:, t-1] is R_t, s[:, t-1] is s_t
    x0 : (B, d)                     — initial hidden state

    Output
    ------
    x : (B, T+1, d) — x[:, t] is x_t, with x[:, 0] == x0
    """
    R_pref, s_pref = all_composition_prefixes_matrix(R, s)
    x1T = minmax_operator.apply_matrix(R_pref, s_pref, x0[:, None, :])
    return torch.cat([x0[:, None, :], x1T], dim=1)
