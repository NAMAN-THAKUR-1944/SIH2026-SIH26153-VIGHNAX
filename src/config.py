"""Load the YAML training / inference configuration."""

from __future__ import annotations

import copy
from typing import Any, Dict

import yaml


def load_config(path: str = "configs/default.yaml") -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return copy.deepcopy(yaml.safe_load(fh))
