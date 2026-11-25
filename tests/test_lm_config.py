# SPDX-FileCopyrightText: 2026 Alessandro Ronca
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""MinMaxRNCLMConfig: warning when a backbone value would be silently replaced."""

import warnings
from dataclasses import replace

import pytest

from minmaxrnc import MinMaxRNC_LM, MinMaxRNCConfig, MinMaxRNCLMConfig

BACKBONE_SET = [
    ('output_gate', False),
    ('conv_type', 'gated'),
    ('ffn_dropout', 0.3),
]


def _no_warnings(fn):
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        return fn()


def test_defaults_do_not_warn():
    _no_warnings(lambda: MinMaxRNCLMConfig(backbone=MinMaxRNCConfig.small()))


@pytest.mark.parametrize('name,value', BACKBONE_SET)
def test_backbone_value_that_would_be_ignored_warns(name, value):
    backbone = replace(MinMaxRNCConfig.small(), **{name: value})
    with pytest.warns(UserWarning, match=rf'backbone\.{name}=.* is ignored'):
        MinMaxRNCLMConfig(backbone=backbone)


@pytest.mark.parametrize('name,value', BACKBONE_SET)
def test_same_value_on_both_does_not_warn(name, value):
    backbone = replace(MinMaxRNCConfig.small(), **{name: value})
    _no_warnings(lambda: MinMaxRNCLMConfig(backbone=backbone, **{name: value}))


@pytest.mark.parametrize('name,value', BACKBONE_SET)
def test_value_set_on_lm_config_only_does_not_warn(name, value):
    _no_warnings(lambda: MinMaxRNCLMConfig(backbone=MinMaxRNCConfig.small(), **{name: value}))


def test_built_model_is_unchanged_lm_value_wins():
    # The warning is the only change: the LM config's value is still the one used (as in 0.4.0).
    backbone = replace(MinMaxRNCConfig.small(), output_gate=False, conv_type='gated')
    with pytest.warns(UserWarning):
        cfg = MinMaxRNCLMConfig(backbone=backbone)
    model = MinMaxRNC_LM(32, cfg)
    reference = MinMaxRNC_LM(32, MinMaxRNCLMConfig(backbone=MinMaxRNCConfig.small()))
    assert [(k, v.shape) for k, v in model.state_dict().items()] == \
           [(k, v.shape) for k, v in reference.state_dict().items()]
