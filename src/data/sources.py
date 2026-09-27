"""Where the prepared per-window states live, per dataset and split."""

from __future__ import annotations

import glob
import os
from typing import Dict

import pandas as pd

# Datasets whose temporal transitions are real recordings. CICIoT2023 episodes are
# composed (benign capture followed by an attack capture), so they are used for
# detection / stage metrics but not for forecasting or lead-time claims.
NON_TEMPORAL = {"ciciot2023"}

# Fixed order = site-embedding index of the world model (index len(SITES) = unknown network).
SITES = ["ctu13", "cicids2017", "cicids2018", "unswnb15", "lanl", "darpa1999", "ciciot2023"]


def site_index(dataset) -> int:
    return SITES.index(dataset) if dataset in SITES else -1


DATASET_NAMES = {
    "ctu13": "CTU-13", "cicids2017": "CIC-IDS2017", "cicids2018": "CSE-CIC-IDS2018", "unswnb15": "UNSW-NB15",
    "lanl": "LANL cyber1", "darpa1999": "DARPA 1999", "ciciot2023": "CICIoT2023",
}


def load_sources(cfg: Dict, split: str) -> Dict[str, pd.DataFrame]:
    """All prepared sources of one split ('train' or 'test'), keyed '<dataset>__<source>__<k>'."""
    d = cfg["data"]
    out: Dict[str, pd.DataFrame] = {}
    for sid in d["train_scenarios"] if split == "train" else d["test_scenarios"]:
        path = os.path.join(d["processed_dir"], f"s{sid}.pkl")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{path} missing - run scripts/prepare_data.py first")
        df = pd.read_pickle(path)
        df["dataset"] = "ctu13"
        out[f"ctu13__s{sid}__0"] = df
    for dataset in cfg.get("datasets", {}):
        for path in sorted(glob.glob(os.path.join(d["processed_dir"], f"{dataset}__*.pkl"))):
            df = pd.read_pickle(path)
            if str(df["split"].iloc[0]) == split:
                out[os.path.basename(path)[:-4]] = df
    return out
