# SPDX-FileCopyrightText: 2026 Alessandro Ronca
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""
Generate tests/fixtures/v0_3_0_reference.pt from a *pristine 0.3.0* checkout.

This records, for a handful of small configurations, everything a later
version needs to prove it still behaves exactly like 0.3.0: the seeded
initial parameters, outputs and final recurrent state for a fixed input (in
fp32 and under an outer CPU bf16 autocast), the sequence of aten operations
of the forward pass (traced with make_fx; compared only under the same torch
minor version), and the parameter counts of the preset sizes.

Usage (run once, against minmaxrnc 0.3.0 — NOT against the current tree):

    git worktree add /tmp/minmaxrnc-v030 v0.3.0      # or commit 6ea620f
    python tests/fixtures/make_v0_3_0_reference.py \
        --src /tmp/minmaxrnc-v030/src \
        --out tests/fixtures/v0_3_0_reference.pt

The script only uses APIs that exist in 0.3.0 and are kept unchanged in later
versions, so the configuration kwargs it stores can be replayed verbatim by
tests/test_backward_compat.py.
"""

import argparse
import platform
import sys

import torch

# (name, kind, kwargs) — kind is 'backbone' (MinMaxRNC) or 'lm' (MinMaxRNC_LM).
CONFIGS = [
    ('default', 'backbone', dict(d_model=16, n_layers=2, d_state=8)),
    ('nogate_gatedconv', 'backbone', dict(
        d_model=16, n_layers=2, d_state=12, output_gate=False, conv_type='gated',
    )),
    ('trainit_asym_rms_postffn', 'backbone', dict(
        d_model=16, n_layers=3, d_state=6, train_init=True, s_r_init='asymmetric',
        norm='rmsnorm', postlayers_norm='rmsnorm', use_postlayers_ffn=True,
    )),
    ('lm_tied', 'lm', dict(d_model=16, n_layers=2, d_state=8)),
]

SEED       = 1234
INPUT_SEED = 99
B, T       = 2, 13          # T not a multiple of UNROLL, so chunking is exercised
UNROLL     = 5
VOCAB      = 37


def _detach_state(state):
    """Recursively detach/clone a recurrent state (list[dict[str, Tensor]])."""
    if torch.is_tensor(state):
        return state.detach().clone()
    if isinstance(state, dict):
        return {k: _detach_state(v) for k, v in state.items()}
    if isinstance(state, (list, tuple)):
        return [_detach_state(v) for v in state]
    return state


def trace_aten_ops(model, inp, unroll):
    """
    The forward pass's sequence of aten operations, traced with make_fx.

    tests/test_backward_compat.py has an identical copy of this function; keep
    them in sync.
    """
    from torch.fx.experimental.proxy_tensor import make_fx
    with torch.no_grad():
        graph = make_fx(lambda x: model(x, unroll, return_state=True))(inp)
    return [str(n.target) for n in graph.graph.nodes if n.op == 'call_function']


def build(M, kind, kwargs):
    torch.manual_seed(SEED)
    backbone = M.MinMaxRNCConfig(**kwargs)
    if kind == 'lm':
        return M.MinMaxRNC_LM(VOCAB, M.MinMaxRNCLMConfig(backbone=backbone))
    return M.MinMaxRNC(backbone)


def make_input(kind, d_model):
    g = torch.Generator().manual_seed(INPUT_SEED)
    if kind == 'lm':
        return torch.randint(0, VOCAB, (B, T), generator=g)
    return torch.randn(B, T, d_model, generator=g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True, help='path to the 0.3.0 src/ directory')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    sys.path.insert(0, args.src)
    import minmaxrnc as M

    records = {}
    for name, kind, kwargs in CONFIGS:
        model = build(M, kind, kwargs)
        model.eval()                                   # disable dropout
        u = make_input(kind, kwargs['d_model'])
        with torch.no_grad():
            y, state = model(u, UNROLL, return_state=True)
            with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
                y_bf16, state_bf16 = model(u, UNROLL, return_state=True)
        records[name] = dict(
            kind         = kind,
            kwargs       = kwargs,
            state_dict   = {k: v.detach().clone() for k, v in model.state_dict().items()},
            num_params   = sum(p.numel() for p in model.parameters()),
            input        = u,
            output       = y.detach().clone(),
            state        = _detach_state(state),
            output_bf16  = y_bf16.detach().clone(),
            state_bf16   = _detach_state(state_bf16),
            aten_ops     = trace_aten_ops(model, u, UNROLL),
        )

    presets = {}
    for preset in ('small', 'medium', 'large'):
        cfg = getattr(M.MinMaxRNCConfig, preset)()
        model = M.MinMaxRNC(cfg)
        presets[preset] = sum(p.numel() for p in model.parameters())

    torch.save(dict(
        meta = dict(
            minmaxrnc_version = '0.3.0',
            torch_version     = torch.__version__,
            # Bitwise reproducibility (seeded init, fp32/bf16 kernels) is only
            # expected in the same environment; tests compare these.
            platform_system   = platform.system(),
            platform_machine  = platform.machine(),
            seed = SEED, input_seed = INPUT_SEED, B = B, T = T, unroll = UNROLL, vocab = VOCAB,
        ),
        configs = records,
        presets = presets,
    ), args.out)
    print(f"wrote {args.out}: {len(records)} configs, presets={presets}")


if __name__ == '__main__':
    main()
