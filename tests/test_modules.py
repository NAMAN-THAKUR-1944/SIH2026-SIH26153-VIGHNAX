"""Run every module's built-in self-test under pytest:  python -m pytest -q"""

import importlib

import pytest

MODULES = [
    "src.data.stages",
    "src.data.packet_features",
    "src.data.state_builder",
    "src.data.flow_parser",
    "src.data.pcap_parser",
    "src.models.world_model",
    "src.models.baseline_lr",
    "src.evaluation.explainer",
]


@pytest.mark.parametrize("name", MODULES)
def test_selftest(name):
    importlib.import_module(name)._selftest()
